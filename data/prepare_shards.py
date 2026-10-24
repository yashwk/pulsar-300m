"""Pulsar Multi-Domain Streaming Shard Builder (NVIDIA Nemotron 5-Pillar Stack).

Curriculum Shard Architecture across 12 Strictly Disjoint Shards (6.0 Billion Tokens):
  Strict 8-Substream Token Quota Mix (80% Direct / 20% <|think|>):
    - Pillar 1 (30.0%):
        * Sub-stream 1A (22.0%): Direct Clean Code, NO <|think|> (codeparrot/github-code-clean)
        * Sub-stream 1B ( 8.0%): Competitive Code with <|think|> (nvidia/OpenCodeReasoning)
    - Pillar 2 (25.0%):
        * Sub-stream 2A (17.0%): Direct Math Textbooks & Concise Proofs, NO <|think|> (HuggingFaceTB/finemath)
        * Sub-stream 2B ( 8.0%): CoT Math Reasoning with <|think|> (nvidia/OpenMathReasoning)
    - Pillar 3 (15.0%):
        * Sub-stream 3A (10.0%): Multi-Turn MCP Tool Calls (Agent-Ark/Toucan-1.5M)
        * Sub-stream 3B ( 5.0%): Structured Schemas & Linux Tasks (open-thoughts/TaskTrove)
    - Pillar 4 (22.0%): ZERO InfiniByte-Reasoning!
        * Sub-stream 4A (10.0%): Clean Markdown Wikipedia (HuggingFaceFW/finewiki)
        * Sub-stream 4B (12.0%): Clean STEM Educational Web (HuggingFaceFW/fineweb-edu)
    - Pillar 5 ( 8.0%):
        * Sub-stream 5A ( 8.0%): Multi-Turn Instructions & Assistant Chat (HuggingFaceTB/smoltalk + Nemotron-Post-Training)

Universal Schema Formatter:
  - Normalizes <think> -> <|think|> and </think> -> <|/think|>
  - Normalizes OpenAI function_call & tool_calls -> <|tool_call|>...<|/tool_call|>
  - Normalizes tool returns -> <|im_start|>user\n<|tool_response|>\n{content}\n<|/tool_response|><|im_end|>
  - Caps R1 reasoning traces to prevent 10,000-token runaway monologues
"""

import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

import argparse
import datetime
import gc
import hashlib
import io
import json
import random
import shutil
import sys
import tarfile
import time
from typing import Generator

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files
from transformers import AutoTokenizer

RAW_DIR = "data/raw"
MANIFEST_PATH = "data/shards_manifest.json"


# ==============================================================================
# Sub-stream Specification & Target Quotas
# ==============================================================================

SUBSTREAM_CONFIG = {
    "1A": {"name": "1A. Clean Code (Direct)", "weight": 22.0, "pillar": "Pillar 1: Code"},
    "1B": {"name": "1B. Competitive Code (CoT)", "weight": 8.0, "pillar": "Pillar 1: Code"},
    "2A": {"name": "2A. Math Textbooks (Direct)", "weight": 17.0, "pillar": "Pillar 2: Mathematics"},
    "2B": {"name": "2B. Math Reasoning (CoT)", "weight": 8.0, "pillar": "Pillar 2: Mathematics"},
    "3A": {"name": "3A. Agentic MCP Tools", "weight": 10.0, "pillar": "Pillar 3: Agentic MCP"},
    "3B": {"name": "3B. Structured Schemas & Tasks", "weight": 5.0, "pillar": "Pillar 3: Agentic MCP"},
    "4A": {"name": "4A. Encyclopedic FineWiki", "weight": 10.0, "pillar": "Pillar 4: STEM & Knowledge"},
    "4B": {"name": "4B. Educational STEM Web", "weight": 12.0, "pillar": "Pillar 4: STEM & Knowledge"},
    "5A": {"name": "5A. Multi-Turn Assistant Chat", "weight": 8.0, "pillar": "Pillar 5: Multi-Turn Chat"},
}

# Mix for the long-context stages (intermediate 8-16k and Stage 3). Reasoning streams
# are weighted far higher than in Stage 1/2 because 53-80% of these traces fit whole at
# 8k+, so the document-aligned loader can present complete question -> reasoning ->
# answer arcs. Stage 1/2 keep the 20% CoT mix: their T=1024/2048 windows only ever see
# the opening fragment of an 8k trace, so raising the share there buys fragments, not
# arcs. Weights are a PROPOSAL for the owner to confirm; see docs/CURRICULUM.md.
SUBSTREAM_CONFIG_LONG = {
    "1B": {"name": "1B. Competitive Code (CoT)", "weight": 15.0, "pillar": "Pillar 1: Code"},
    "2B": {"name": "2B. Math Reasoning (CoT)", "weight": 15.0, "pillar": "Pillar 2: Mathematics"},
    "3A": {"name": "3A. Agentic MCP Tools", "weight": 12.0, "pillar": "Pillar 3: Agentic MCP"},
    "1A": {"name": "1A. Clean Code (Direct)", "weight": 14.0, "pillar": "Pillar 1: Code"},
    "2A": {"name": "2A. Math Textbooks (Direct)", "weight": 10.0, "pillar": "Pillar 2: Mathematics"},
    "3B": {"name": "3B. Structured Schemas & Tasks", "weight": 8.0, "pillar": "Pillar 3: Agentic MCP"},
    "4A": {"name": "4A. Encyclopedic FineWiki", "weight": 10.0, "pillar": "Pillar 4: STEM & Knowledge"},
    "4B": {"name": "4B. Educational STEM Web", "weight": 10.0, "pillar": "Pillar 4: STEM & Knowledge"},
    "5A": {"name": "5A. Multi-Turn Assistant Chat", "weight": 6.0, "pillar": "Pillar 5: Multi-Turn Chat"},
}

# Reasoning share (1B + 2B + 3A) of each mix, used by the dry-run report.
STAGE1_REASONING_WEIGHT = 8.0 + 8.0 + 10.0   # 26.0% (18% CoT + agentic 3A)
LONG_REASONING_WEIGHT = 15.0 + 15.0 + 12.0   # 42.0%


def substream_config_for_stage(stage: int) -> dict:
    """Return the sub-stream mix for a curriculum stage.

    Stages 1 and 2 use the documented 20% CoT Nemotron mix. Stage 3 (and the
    intermediate long-context stage, which is built with ``--stage 3``) uses the
    reasoning-heavy mix because only there do whole traces fit in one window.
    """
    return SUBSTREAM_CONFIG_LONG if stage >= 3 else SUBSTREAM_CONFIG


# ==============================================================================
# Resource Safety Guard
# ==============================================================================

def check_system_resources(min_disk_gb: float = 3.0, min_ram_mb: float = 400.0) -> tuple[float, float]:
    """Verify available disk space and RAM, raising RuntimeError if resources are critically low."""
    stat = shutil.disk_usage(".")
    free_gb = stat.free / (1024**3)
    if free_gb < min_disk_gb:
        raise RuntimeError(
            f"CRITICAL: Insufficient disk space! Free: {free_gb:.2f} GB (minimum required: {min_disk_gb:.1f} GB)."
        )

    avail_mb = None
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    avail_mb = int(line.split()[1]) / 1024.0
                    break
    except Exception:
        pass

    if avail_mb is not None and avail_mb < min_ram_mb:
        gc.collect()
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        avail_mb = int(line.split()[1]) / 1024.0
                        break
        except Exception:
            pass
        if avail_mb < min_ram_mb:
            raise RuntimeError(
                f"CRITICAL: Insufficient available RAM! Available: {avail_mb:.1f} MB (minimum required: {min_ram_mb:.1f} MB)."
            )
    return free_gb, (avail_mb if avail_mb is not None else -1.0)


# ==============================================================================
# Special Token Normalization & Formatters
# ==============================================================================

def normalize_special_tags(text: str) -> str:
    """Normalize raw think and tool tags into Pulsar native special tokens."""
    return (
        text.replace("<think>", "<|think|>")
        .replace("</think>", "<|/think|>")
        .replace("<tool_call>", "<|tool_call|>")
        .replace("</tool_call>", "<|/tool_call|>")
        .replace("<function_call>", "<|tool_call|>")
        .replace("</function_call>", "<|/tool_call|>")
        .replace("<tool_response>", "<|tool_response|>")
        .replace("</tool_response>", "<|/tool_response|>")
    )


# A single document longer than this is capped. 131,072 characters is roughly 32k
# tokens, i.e. the longest context the model is ever trained on. In practice nothing
# in the sampled sources comes close (observed max ~24k tokens), so this is a safety
# valve, not an active filter.
MAX_TRACE_CHARS = 131072

# Legacy elision budget, retained ONLY so the dry-run can measure what the old build
# produced. Never used to build shards.
LEGACY_ELISION_CHARS = 1800


def derive_stream_seed(stream_id: str, shard_idx: int, base_seed: int = 20261023) -> int:
    """Deterministic seed for a (stream, shard) pair.

    Stable across processes and machines (sha256, not Python's salted ``hash``) so a
    rebuild with the same base seed reproduces the same shard byte-for-byte.
    """
    material = f"{base_seed}:{stream_id}:{shard_idx}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def clean_think_trace(think_raw: str, max_chars: int = MAX_TRACE_CHARS) -> str:
    """Return the reasoning trace verbatim or cleanly bounded without mid-sentence slicing.

    Never slices mid-sentence: cuts strictly at a paragraph/sentence boundary and
    preserves the final conclusion sentence before <|/think|>.
    """
    think = think_raw.strip()
    if len(think) <= max_chars:
        return think

    # Extract final conclusion sentence from the tail so the reasoning completes cleanly
    tail_candidates = [
        think.rfind(".\n"),
        think.rfind(". "),
        think.rfind("?\n"),
        think.rfind("! "),
    ]
    last_end = max(tail_candidates)
    tail_sent = ""
    if last_end != -1 and last_end > len(think) - 500:
        prev_punct = max(
            think.rfind(".\n", 0, last_end - 1),
            think.rfind(". ", 0, last_end - 1),
            think.rfind("\n\n", 0, last_end - 1),
        )
        if prev_punct != -1:
            tail_sent = think[prev_punct + 1 : last_end + 1].strip()

    reserve_tail = len(tail_sent) + 2 if tail_sent else 0
    head_budget = max(200, max_chars - reserve_tail)
    head = think[:head_budget]
    last_punct = max(
        head.rfind(".\n"), head.rfind(". "), head.rfind("?\n"), head.rfind("! "), head.rfind("\n\n"), head.rfind("\n")
    )
    if last_punct > 100:
        head = head[: last_punct + 1]

    if tail_sent and not head.endswith(tail_sent):
        return f"{head.strip()}\n{tail_sent}".strip()
    return head.strip()


def legacy_clean_think_trace(think_raw: str, max_chars: int = LEGACY_ELISION_CHARS) -> str:
    """Reproduce the pre-WI-13 mid-elision output.

    Used exclusively by the dry-run report to quantify how many tokens the old build
    spent per document, so the diversity trade-off can be measured rather than guessed.
    """
    think = think_raw.strip()
    if len(think) <= max_chars:
        return think

    head_chunk = think[:1000]
    last_period_head = max(
        head_chunk.rfind(".\n"), head_chunk.rfind(". "),
        head_chunk.rfind("?\n"), head_chunk.rfind("! ")
    )
    if last_period_head > 300:
        head = head_chunk[:last_period_head + 1].strip()
    else:
        head = head_chunk.strip()

    tail_chunk = think[-600:]
    candidates = [pos for pos in (tail_chunk.find(".\n"), tail_chunk.find(". "), tail_chunk.find("?\n")) if pos != -1]
    first_period_tail = min(candidates) if candidates else 0
    if 0 < first_period_tail < len(tail_chunk) - 50:
        tail = tail_chunk[first_period_tail + 2:].strip()
    else:
        tail = tail_chunk.strip()

    if not (tail.endswith(".") or tail.endswith("!") or tail.endswith("?")):
        last_punct = max(tail.rfind("."), tail.rfind("!"), tail.rfind("?"))
        if last_punct > 50:
            tail = tail[:last_punct + 1].strip()

    return f"{head}\n\n[...]\n\n{tail}"


def format_cot_block(problem: str, solution: str, max_think_chars: int = MAX_TRACE_CHARS) -> str:
    """Format a reasoning trace with the think block preserved in full."""
    sol = normalize_special_tags(solution.strip())
    if "<|think|>" in sol and "<|/think|>" in sol:
        pre_think, rest = sol.split("<|think|>", 1)
        think_content, post_think = rest.split("<|/think|>", 1)
        cleaned_think = clean_think_trace(think_content, max_chars=max_think_chars)
        sol = f"{pre_think}<|think|>\n{cleaned_think}\n<|/think|>{post_think}".strip()
    elif "<|think|>" not in sol:
        cleaned_think = clean_think_trace(sol, max_chars=max_think_chars)
        sol = f"<|think|>\n{cleaned_think}\n<|/think|>"

    p = problem.strip()
    return f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n{sol}<|im_end|>"


def legacy_format_cot_block(problem: str, solution: str, max_think_chars: int = LEGACY_ELISION_CHARS) -> str:
    """Pre-WI-13 COT formatting (mid-elided). Dry-run comparison only."""
    sol = normalize_special_tags(solution.strip())
    if "<|think|>" in sol and "<|/think|>" in sol:
        pre_think, rest = sol.split("<|think|>", 1)
        think_content, post_think = rest.split("<|/think|>", 1)
        cleaned_think = legacy_clean_think_trace(think_content, max_chars=max_think_chars)
        sol = f"{pre_think}<|think|>\n{cleaned_think}\n<|/think|>{post_think}".strip()
    elif "<|think|>" not in sol:
        cleaned_think = legacy_clean_think_trace(sol, max_chars=max_think_chars)
        sol = f"<|think|>\n{cleaned_think}\n<|/think|>"

    p = problem.strip()
    return f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n{sol}<|im_end|>"


def format_single_tool_call(call: dict, style: str) -> str:
    """Format a tool call according to the specified harness style (native, xml, md_json)."""
    name = call.get("name") or "tool"
    args = call.get("arguments") or {}
    if isinstance(args, str):
        try:
            args_obj = json.loads(args)
        except Exception:
            args_obj = args
    else:
        args_obj = args

    if style == "xml":
        # Cline / Claude Code XML style
        if isinstance(args_obj, dict):
            args_xml = "".join(f"\n<{k}>{v}</{k}>" for k, v in args_obj.items()) + "\n"
        else:
            args_xml = f"\n{args_obj}\n"
        return f'<|tool_call|>\n<invoke name="{name}">\n<arguments>{args_xml}</arguments>\n</invoke>\n<|/tool_call|>'
    elif style == "md_json":
        # Markdown JSON code block style
        payload = {"tool": name, "arguments": args_obj}
        return f"```json\n{json.dumps(payload, indent=2, ensure_ascii=False)}\n```"
    else:
        # Native Fast JSON style
        c_str = json.dumps(call, ensure_ascii=False)
        return f"<|tool_call|>\n{c_str}\n<|/tool_call|>"


def format_toucan_messages(messages, style_seed: float | None = None) -> str:
    """Format multi-turn agentic conversations with 3-way harness format randomization (70% Native, 15% XML, 15% MD JSON)."""
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except Exception:
            return ""
    if not isinstance(messages, (list, np.ndarray)):
        return ""

    # Probabilistic harness selection per conversation
    r = style_seed if style_seed is not None else random.random()
    if r < 0.70:
        harness_style = "native"
        harness_prompt_hint = "You may call tools using JSON: <|tool_call|>{\"name\": ..., \"arguments\": ...}<|/tool_call|>"
    elif r < 0.85:
        harness_style = "xml"
        harness_prompt_hint = "You may call tools using XML: <|tool_call|><invoke name=\"...\"><arguments>...</arguments></invoke><|/tool_call|>"
    else:
        harness_style = "md_json"
        harness_prompt_hint = "To call a tool, output a markdown JSON code block with 'tool' and 'arguments'."

    turns = []
    has_system = False
    for turn in messages:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("role") or turn.get("from") or "user").lower()
        content = str(turn.get("content") or turn.get("value") or "").strip()
        content = normalize_special_tags(content)

        if role in ("system",):
            has_system = True
            content_with_hint = f"{content}\n\n[Tool Calling Convention]\n{harness_prompt_hint}" if content else f"[Tool Calling Convention]\n{harness_prompt_hint}"
            turns.append(f"<|im_start|>system\n{content_with_hint}<|im_end|>")
        elif role in ("human", "user"):
            turns.append(f"<|im_start|>user\n{content}<|im_end|>")
        elif role in ("function", "tool", "observation", "environment"):
            if harness_style == "xml":
                turns.append(f"<|im_start|>user\n<|tool_response|>\n<result>\n{content}\n</result>\n<|/tool_response|><|im_end|>")
            elif harness_style == "md_json":
                turns.append(f"<|im_start|>user\nTool Response:\n```\n{content}\n```<|im_end|>")
            else:
                turns.append(f"<|im_start|>user\n<|tool_response|>\n{content}\n<|/tool_response|><|im_end|>")
        elif role in ("gpt", "assistant", "model"):
            fc = turn.get("function_call")
            tcs = turn.get("tool_calls")
            extra = []
            if fc and isinstance(fc, dict):
                extra.append(format_single_tool_call(fc, harness_style))
            elif tcs and isinstance(tcs, list):
                for call in tcs:
                    if isinstance(call, dict):
                        extra.append(format_single_tool_call(call, harness_style))
                    else:
                        extra.append(format_single_tool_call({"name": "tool", "arguments": str(call)}, harness_style))

            full_assist = "\n".join(extra)
            if content:
                full_assist = f"{full_assist}\n{content}" if full_assist else content
            turns.append(f"<|im_start|>assistant\n{full_assist.strip()}<|im_end|>")

    if not has_system and turns:
        # Prepend system instruction if missing
        turns.insert(0, f"<|im_start|>system\n[Tool Calling Convention]\n{harness_prompt_hint}<|im_end|>")

    return "\n".join(turns) if turns else ""


def format_tasktrove_row(row: dict) -> str:
    """Extract and format structured tasks, TOML schemas, and bash commands from TaskTrove."""
    raw_bin = row.get("task_binary")
    if not raw_bin or not isinstance(raw_bin, bytes):
        return ""

    inst = ""
    toml_str = ""
    solve_sh = ""
    manifest_data = None

    try:
        with tarfile.open(fileobj=io.BytesIO(raw_bin), mode="r:gz") as tar:
            for m in tar.getmembers():
                if m.name == "instruction.md" or m.name.endswith("/instruction.md"):
                    inst = tar.extractfile(m).read().decode("utf-8", errors="ignore").strip()
                elif m.name.endswith(".toml"):
                    toml_str = tar.extractfile(m).read().decode("utf-8", errors="ignore").strip()
                elif m.name.endswith("solve.sh"):
                    solve_sh = tar.extractfile(m).read().decode("utf-8", errors="ignore").strip()
                elif "manifest" in m.name and m.name.endswith(".json"):
                    try:
                        manifest_data = json.loads(tar.extractfile(m).read().decode("utf-8", errors="ignore"))
                    except Exception:
                        pass
    except Exception:
        return ""

    parts = []
    if inst:
        parts.append(f"<|im_start|>user\n{inst}<|im_end|>")
    elif manifest_data and isinstance(manifest_data, dict) and "mutated_nl" in manifest_data:
        parts.append(f"<|im_start|>user\n{manifest_data['mutated_nl']}<|im_end|>")
    else:
        return ""

    resp = []
    if toml_str:
        resp.append(f"```toml\n{toml_str}\n```")
    if solve_sh:
        resp.append(f"```bash\n{solve_sh}\n```")
    elif manifest_data and isinstance(manifest_data, dict) and "mutated_bash" in manifest_data:
        resp.append(f"```bash\n{manifest_data['mutated_bash']}\n```")

    if resp:
        parts.append(f"<|im_start|>assistant\n" + "\n\n".join(resp) + "<|im_end|>")
        return "\n".join(parts)
    return ""


# ==============================================================================
# Parquet Streaming Generators per Sub-stream
# ==============================================================================

def make_substream_stream(
    substream_id: str,
    num_shards: int = 12,
    shard_idx: int = 0,
    batch_size: int = 512,
    base_seed: int = 20261023,
) -> Generator[str, None, None]:
    """Create a persistent generator streaming documents for a given sub-stream.

    All per-(stream, shard) randomness comes from a seeded ``random.Random`` so the
    rebuild is reproducible; the seed is recorded in the manifest.
    """
    stream_rng = random.Random(derive_stream_seed(substream_id, shard_idx, base_seed))
    if substream_id == "1A":
        # codeparrot/github-code-clean
        repo_id = "codeparrot/github-code-clean"
        prefix = "data/train-"
    elif substream_id == "1B":
        # nvidia/OpenCodeReasoning
        repo_id = "nvidia/OpenCodeReasoning"
        prefix = "split_0/train-"
    elif substream_id == "2A":
        # HuggingFaceTB/finemath
        repo_id = "HuggingFaceTB/finemath"
        prefix = "finemath-4plus/train-"
    elif substream_id == "2B":
        # nvidia/OpenMathReasoning
        repo_id = "nvidia/OpenMathReasoning"
        prefix = "data/cot-"
    elif substream_id == "3A":
        # Agent-Ark/Toucan-1.5M
        repo_id = "Agent-Ark/Toucan-1.5M"
        prefix = "Kimi-K2/train-"
    elif substream_id == "3B":
        # open-thoughts/TaskTrove
        repo_id = "open-thoughts/TaskTrove"
        prefix = ""
    elif substream_id == "4A":
        # HuggingFaceFW/finewiki
        repo_id = "HuggingFaceFW/finewiki"
        prefix = "data/enwiki/"
    elif substream_id == "4B":
        # HuggingFaceFW/fineweb-edu
        repo_id = "HuggingFaceFW/fineweb-edu"
        prefix = "sample/100BT/"
    elif substream_id == "5A":
        # HuggingFaceTB/smoltalk + nvidia/Nemotron-Post-Training
        repo_id = "HuggingFaceTB/smoltalk"
        prefix = "data/all/train-"
    else:
        raise ValueError(f"Unknown sub-stream ID: {substream_id}")

    all_files = [
        f for f in list_repo_files(repo_id, repo_type="dataset")
        if f.startswith(prefix) and f.endswith(".parquet")
    ]
    all_files.sort()
    if not all_files:
        raise ValueError(f"No parquet files found in {repo_id} with prefix '{prefix}'")

    total_files = len(all_files)
    if substream_id in ("4A", "4B"):
        # For multi-gigabyte datasets (finewiki 2.5GB, fineweb-edu 2.15GB),
        # reuse cached files and enforce row-level disjoint partition
        # to prevent 60GB disk exhaustion across shards.
        assigned_indices = list(range(total_files))
        use_row_partition = True
    elif total_files >= num_shards:
        assigned_indices = [i for i in range(total_files) if i % num_shards == shard_idx]
        use_row_partition = False
    else:
        assigned_indices = list(range(total_files))
        use_row_partition = True

    while True:
        for f_idx in assigned_indices:
            fname = all_files[f_idx]
            # Download or reuse cached parquet file directly
            fpath = hf_hub_download(repo_id=repo_id, filename=fname, repo_type="dataset")
            pf = pq.ParquetFile(fpath)
            doc_counter = 0

            for batch in pf.iter_batches(batch_size=batch_size):
                pydict = batch.to_pydict()
                del batch
                keys = list(pydict.keys())
                num_rows = len(pydict[keys[0]])

                for r_i in range(num_rows):
                    if use_row_partition and (doc_counter % num_shards != shard_idx):
                        doc_counter += 1
                        continue
                    doc_counter += 1

                    row = {k: pydict[k][r_i] for k in keys}
                    formatted = ""

                    if substream_id == "1A":
                        path = str(row.get("path") or "")
                        if any(x in path for x in ["node_modules", "vendor", ".min.", "package-lock", "dist/"]):
                            continue
                        code = str(row.get("code") or "").strip()
                        if len(code) > 40 and len(code) < 40000:
                            formatted = code

                    elif substream_id == "1B":
                        inp = str(row.get("input") or row.get("problem") or "").strip()
                        out = str(row.get("output") or row.get("generated_solution") or "").strip()
                        if inp and out:
                            formatted = format_cot_block(inp, out)

                    elif substream_id == "2A":
                        txt = str(row.get("text") or "").strip()
                        if len(txt) > 50:
                            formatted = txt

                    elif substream_id == "2B":
                        p = str(row.get("problem") or "").strip()
                        s = str(row.get("generated_solution") or row.get("cot") or "").strip()
                        if p and s:
                            formatted = format_cot_block(p, s)

                    elif substream_id == "3A":
                        msgs = row.get("messages") or row.get("conversations")
                        formatted = format_toucan_messages(msgs, style_seed=stream_rng.random())

                    elif substream_id == "3B":
                        formatted = format_tasktrove_row(row)

                    elif substream_id == "4A":
                        txt = str(row.get("text") or "").strip()
                        if len(txt) > 50:
                            formatted = txt

                    elif substream_id == "4B":
                        # Educational STEM web: require high educational score
                        score = row.get("score") or row.get("int_score") or 0
                        if float(score) >= 2.5:
                            txt = str(row.get("text") or "").strip()
                            if len(txt) > 50:
                                formatted = txt

                    elif substream_id == "5A":
                        msgs = row.get("messages") or row.get("conversations")
                        if msgs:
                            formatted = format_toucan_messages(msgs, style_seed=stream_rng.random())

                    if formatted and len(formatted.strip()) > 15:
                        yield formatted.strip()

                del pydict
                gc.collect()


# ==============================================================================
# Manifest Management
# ==============================================================================

def update_manifest(
    shard_idx: int,
    stage: int,
    train_path: str,
    val_path: str,
    train_tokens: int,
    val_tokens: int,
    substream_tokens: dict | None = None,
    substream_docs: dict | None = None,
    stream_seeds: dict | None = None,
    base_seed: int | None = None,
):
    """Atomically record completed shard in shards_manifest.json."""
    manifest = {"curriculum_version": "2.0-NVIDIA-Nemotron-Refined", "shards": []}
    if os.path.exists(MANIFEST_PATH):
        try:
            with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception:
            pass

    stage_names = {
        1: "Stage 1 Foundation (NVIDIA Nemotron 5-Pillar Stack / 8 Sub-Streams)",
        2: "Stage 2 Reasoning & Niche Upsample (T=2048)",
        3: "Stage 3 Long-Context & Annealing (T in [2k..32k])",
    }

    total = train_tokens + val_tokens
    shard_entry = {
        "shard_idx": shard_idx,
        "stage": stage,
        "stage_name": stage_names.get(stage, f"Stage {stage}"),
        "train_path": train_path,
        "val_path": val_path,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "total_tokens": total,
        "status": "completed",
        "completed_at": datetime.datetime.now().isoformat(),
        # Provenance: traces are stored un-elided, so the reasoning share below is the
        # share of real reasoning tokens (not elided fragments).
        "trace_policy": "un-elided",
        "max_trace_chars": MAX_TRACE_CHARS,
        "corpus_version": "2.1-unelided-traces",
        "base_seed": base_seed,
    }
    if substream_tokens:
        shard_entry["substream_tokens"] = dict(substream_tokens)
        shard_entry["substream_shares_pct"] = {
            k: round(100.0 * v / total, 3) for k, v in substream_tokens.items()
        } if total else {}
        reasoning = sum(substream_tokens.get(k, 0) for k in ("1B", "2B", "3A"))
        shard_entry["reasoning_share_pct"] = round(100.0 * reasoning / total, 3) if total else 0.0
    if substream_docs:
        shard_entry["substream_docs"] = dict(substream_docs)
    if stream_seeds:
        shard_entry["stream_seeds"] = {k: str(v) for k, v in stream_seeds.items()}

    updated = False
    for i, s in enumerate(manifest.get("shards", [])):
        if s.get("shard_idx") == shard_idx:
            manifest["shards"][i] = shard_entry
            updated = True
            break
    if not updated:
        manifest.setdefault("shards", []).append(shard_entry)

    tmp_path = MANIFEST_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp_path, MANIFEST_PATH)
    print(f"[Manifest] Updated '{MANIFEST_PATH}' for Shard {shard_idx}")


# ==============================================================================
# Shard Builder Engine
# ==============================================================================

def build_shard(
    shard_idx: int,
    stage: int,
    num_shards: int,
    total_tokens: int,
    train_path: str,
    val_path: str,
    tokenizer_path: str = "data/pulsar_tokenizer",
    batch_size: int = 256,
    base_seed: int = 20261023,
):
    """Generate a single curriculum shard with exact sub-stream token budgeting."""
    os.makedirs(os.path.dirname(train_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(val_path) or ".", exist_ok=True)

    print("==================================================================")
    print(f"   PULSAR SHARD BUILDER: SHARD {shard_idx}/{num_shards - 1}   ")
    print(f"   Stage {stage} | Target: {total_tokens:,} tokens | Mix: "
          f"{'long-context (reasoning-heavy)' if stage >= 3 else 'Nemotron 80/20'}")
    print(f"   Traces: un-elided (cap {MAX_TRACE_CHARS:,} chars) | Seed: {base_seed} + per-stream sha256")
    print("==================================================================")
    free_gb, avail_ram = check_system_resources(min_disk_gb=3.0, min_ram_mb=400.0)
    print(f"[Resource Guard] Available Disk: {free_gb:.2f} GB | Available RAM: {avail_ram:.1f} MB (PASS)")

    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    eos_id = tok.eos_token_id if tok.eos_token_id is not None else 0
    print(f"[Tokenizer] Vocab size: {len(tok):,}, EOS id: {eos_id}")

    target_val = int(total_tokens * 0.05) if total_tokens >= 50_000_000 else 0
    target_train = total_tokens - target_val

    # Initialize sub-stream quotas for this stage's mix
    substream_cfg = substream_config_for_stage(stage)
    substream_keys = list(substream_cfg.keys())
    target_quota = {
        s: int(round((substream_cfg[s]["weight"] / 100.0) * total_tokens))
        for s in substream_keys
    }
    # Adjust rounding differences to match exact total_tokens
    diff = total_tokens - sum(target_quota.values())
    target_quota[substream_keys[0]] += diff

    print("\n--- Sub-Stream Quotas ---")
    for s in substream_keys:
        info = substream_cfg[s]
        print(f"  [{s}] {info['name']:32s} | Weight: {info['weight']:5.1f}% | Target: {target_quota[s]:,} tokens")

    # Initialize stream generators (seeded per (stream, shard) for reproducibility)
    stream_seeds = {s: derive_stream_seed(s, shard_idx, base_seed) for s in substream_keys}
    generators = {
        s: make_substream_stream(
            s,
            num_shards=num_shards,
            shard_idx=shard_idx,
            batch_size=batch_size,
            base_seed=base_seed,
        )
        for s in substream_keys
    }

    train_tmp = train_path + ".tmp"
    val_tmp = val_path + ".tmp" if target_val > 0 else None
    f_train = open(train_tmp, "wb")
    f_val = open(val_tmp, "wb") if val_tmp else None

    substream_tokens = {s: 0 for s in substream_keys}
    substream_docs = {s: 0 for s in substream_keys}
    train_tokens = 0
    val_tokens = 0
    doc_idx = 0

    t_start = time.time()
    last_print = t_start
    last_health_check = 0

    train_buf = []
    val_buf = []
    BUF_FLUSH_SIZE = 65536  # ~128KB in uint16

    try:
        while any(substream_tokens[s] < target_quota[s] for s in substream_keys):
            progress_in_cycle = False

            for s in substream_keys:
                if substream_tokens[s] >= target_quota[s]:
                    continue

                try:
                    txt = next(generators[s])
                except StopIteration:
                    continue

                if not txt or len(txt.strip()) < 15:
                    continue

                enc = tok.encode(txt, add_special_tokens=False)
                if not enc:
                    continue

                seq_len = len(enc) + 1  # includes eos_id
                progress_in_cycle = True
                doc_idx += 1

                is_val = (target_val > 0) and (doc_idx % 20 == 0) and (val_tokens < target_val)

                if is_val:
                    val_buf.extend(enc)
                    val_buf.append(eos_id)
                    val_tokens += seq_len
                else:
                    train_buf.extend(enc)
                    train_buf.append(eos_id)
                    train_tokens += seq_len

                substream_tokens[s] += seq_len
                substream_docs[s] += 1

                if len(train_buf) >= BUF_FLUSH_SIZE:
                    f_train.write(np.array(train_buf, dtype=np.uint16).tobytes())
                    train_buf.clear()

                if len(val_buf) >= BUF_FLUSH_SIZE and f_val:
                    f_val.write(np.array(val_buf, dtype=np.uint16).tobytes())
                    val_buf.clear()

                total_now = train_tokens + val_tokens
                now = time.time()
                if now - last_print >= 5.0 or total_now >= total_tokens:
                    dt = now - t_start
                    speed = total_now / max(dt, 0.01)
                    pct = 100.0 * total_now / total_tokens
                    eta_sec = (total_tokens - total_now) / max(1.0, speed)

                    if total_now - last_health_check >= 10_000_000 or total_now >= total_tokens:
                        curr_free_gb, curr_avail_ram = check_system_resources()
                        gc.collect()
                        last_health_check = total_now

                    print(
                        f"  [{pct:5.1f}%] {total_now / 1e6:6.2f}M / {total_tokens / 1e6:.1f}M tokens "
                        f"({speed / 1e6:4.2f}M tok/s, ETA: {eta_sec / 60:4.1f}m) | Docs: {doc_idx:,}"
                    )
                    last_print = now

            if not progress_in_cycle:
                break

        # Flush any remaining items in buffer
        if train_buf:
            f_train.write(np.array(train_buf, dtype=np.uint16).tobytes())
            train_buf.clear()
        if val_buf and f_val:
            f_val.write(np.array(val_buf, dtype=np.uint16).tobytes())
            val_buf.clear()

    finally:
        f_train.flush()
        f_train.close()
        if f_val:
            f_val.flush()
            f_val.close()

    os.replace(train_tmp, train_path)
    if val_tmp:
        os.replace(val_tmp, val_path)

    total_done = train_tokens + val_tokens
    elapsed_total = time.time() - t_start

    print("\n================== SHARD BUILD COMPLETE ==================")
    print(f"Total tokens generated: {total_done:,} in {elapsed_total / 60:.2f} mins ({total_done / elapsed_total / 1e6:.2f}M tok/s)")
    print(f"  -> Train: {train_tokens:,} tokens ({train_tokens / total_done * 100:.2f}%) -> {train_path} ({os.path.getsize(train_path) / 1e6:.1f} MB)")
    if val_tmp:
        print(f"  -> Val:   {val_tokens:,} tokens ({val_tokens / total_done * 100:.2f}%) -> {val_path} ({os.path.getsize(val_path) / 1e6:.1f} MB)")

    print(f"\nFinal {len(substream_keys)}-Substream Distribution (Shard {shard_idx}):")
    print(f"{'Sub-Stream':<32s} | {'Target %':<10s} | {'Actual %':<10s} | {'Tokens':<12s} | {'Docs':<9s} | {'Delta %':<10s}")
    print("-" * 92)
    for s in substream_keys:
        info = substream_cfg[s]
        s_toks = substream_tokens[s]
        actual_pct = (s_toks / total_done) * 100.0 if total_done > 0 else 0.0
        delta = actual_pct - info["weight"]
        print(f"{info['name']:<32s} | {info['weight']:9.1f}% | {actual_pct:9.2f}% | {s_toks:10,d} | {substream_docs[s]:9,d} | {delta:+6.2f}%")
    print("==========================================================")

    if total_tokens >= 50_000_000:
        update_manifest(
            shard_idx, stage, train_path, val_path, train_tokens, val_tokens,
            substream_tokens=substream_tokens,
            substream_docs=substream_docs,
            stream_seeds=stream_seeds,
            base_seed=base_seed,
        )


# ==============================================================================
# CLI Entrypoint
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Pulsar Multi-Domain Streaming Shard Builder")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--shard-idx", type=int, default=1)
    parser.add_argument("--num-shards", type=int, default=12)
    parser.add_argument("--total-tokens", type=int, default=500_000_000)
    parser.add_argument("--train-path", type=str, default=None)
    parser.add_argument("--val-path", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, default="data/pulsar_tokenizer")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--seed", type=int, default=20261023,
        help="Base seed; per-(stream, shard) seeds are derived via sha256 and recorded in the manifest",
    )
    args = parser.parse_args()

    train_path = args.train_path or f"data/train_shard{args.shard_idx}.bin"
    val_path = args.val_path or f"data/val_shard{args.shard_idx}.bin"

    build_shard(
        shard_idx=args.shard_idx,
        stage=args.stage,
        num_shards=args.num_shards,
        total_tokens=args.total_tokens,
        train_path=train_path,
        val_path=val_path,
        tokenizer_path=args.tokenizer,
        batch_size=args.batch_size,
        base_seed=args.seed,
    )


if __name__ == "__main__":
    main()

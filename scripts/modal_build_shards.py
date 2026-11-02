"""Modal App for Cloud-Native Shard Generation.

Generates pretraining shards directly in the cloud on high-performance multi-core
CPU instances using 10 Gbps datacenter pipes to Hugging Face, writing directly to
a persistent modal.Volume('pulsar-data').
"""

import json
import os
import subprocess
import sys
import time
import modal

# 1. Persistent volume for storing binary training shards
data_vol = modal.Volume.from_name("pulsar-data", create_if_missing=True)

# 2. Shared status dictionary for live static monitor telemetry
status_dict = modal.Dict.from_name("pulsar-shard-status", create_if_missing=True)

# 3. Clean container image with fast Rust tokenizers and high-performance Xet transfer
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "huggingface_hub>=0.24.0",
        "pyarrow>=14.0.0",
        "tokenizers>=0.19.0",
        "numpy",
        "tqdm",
        "requests",
    )
    .env({
        "HF_XET_HIGH_PERFORMANCE": "1",
        "PYTHONUNBUFFERED": "1",
    })
    .add_local_dir("data/pulsar_tokenizer", remote_path="/root/pulsar_tokenizer")
    .add_local_file("data/prepare_shards.py", remote_path="/root/prepare_shards.py")
)

app = modal.App("pulsar-shard-builder")


@app.function(
    image=image,
    cpu=8,                          # 8 vCPUs (4 physical cores) for balanced, cost-effective tokenization
    memory=8192,                    # 8 GB RAM
    timeout=60 * 60,                # 60 min safety timeout
    volumes={"/root/data": data_vol},
)
def generate_single_shard(
    shard_idx: int,
    stage: int = 1,
    total_tokens: int = 500_000_000,
    batch_size: int = 2048,
    hf_token: str = None,
    min_doc_tokens: int = 16,
    max_doc_tokens: int = 0,
    use_jev: bool = False,
    jev_max_usd: float = 0.60,
    typesafe_api_key: str = None,
) -> dict:
    """Generates a single pretraining shard with isolated file paths and commits to volume."""
    import json
    import os
    import subprocess
    import time
    from tokenizers import Tokenizer

    # 1. Verify tokenizer vocabulary size using pure Rust engine
    tok = Tokenizer.from_file("/root/pulsar_tokenizer/tokenizer.json")
    vocab_size = tok.get_vocab_size()
    assert vocab_size == 32768, f"Modal worker {shard_idx} tokenizer vocab mismatch: {vocab_size} != 32768"

    status_file = f"/tmp/shard_status_{shard_idx}.json"
    status_dict[shard_idx] = {
        "shard_idx": shard_idx,
        "stage": stage,
        "tokens": 0,
        "target": total_tokens,
        "pct": 0.0,
        "rate": 0.0,
        "eta_sec": 0,
        "docs": 0,
        "status": "STARTING",
        "updated_at": time.time(),
    }

    env = os.environ.copy()
    env["HF_XET_HIGH_PERFORMANCE"] = "1"
    env.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
    if hf_token:
        env["HF_TOKEN"] = hf_token
        env["HUGGING_FACE_HUB_TOKEN"] = hf_token
    if typesafe_api_key:
        env["TYPESAFE_API_KEY"] = typesafe_api_key
        env["JEV_API_KEY"] = typesafe_api_key
    env["PYTHONUNBUFFERED"] = "1"

    train_path = f"/root/data/train_shard{shard_idx}.bin"
    val_path = f"/root/data/val_shard{shard_idx}.bin"

    cmd = [
        "python", "-u", "/root/prepare_shards.py",
        "--shard-idx", str(shard_idx),
        "--stage", str(stage),
        "--total-tokens", str(total_tokens),
        "--train-path", train_path,
        "--val-path", val_path,
        "--tokenizer", "/root/pulsar_tokenizer",
        "--batch-size", str(batch_size),
        "--min-doc-tokens", str(min_doc_tokens),
        "--max-doc-tokens", str(max_doc_tokens),
        "--status-file", status_file,
        "--quiet",
    ]
    if hf_token:
        cmd.extend(["--hf-token", hf_token])
    if use_jev:
        cmd.extend(["--use-jev", "--jev-max-usd", f"{jev_max_usd:.4f}"])
        if typesafe_api_key:
            cmd.extend(["--jev-api-key", typesafe_api_key])

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Monitor progress file while worker process is running
    while proc.poll() is None:
        if os.path.exists(status_file):
            try:
                with open(status_file, "r", encoding="utf-8") as f:
                    st = json.load(f)
                status_dict[shard_idx] = {
                    "shard_idx": shard_idx,
                    "stage": stage,
                    "tokens": st.get("total_tokens", 0),
                    "target": total_tokens,
                    "pct": st.get("pct", 0.0),
                    "rate": st.get("speed", 0.0),
                    "eta_sec": st.get("eta_sec", 0),
                    "docs": st.get("docs", 0),
                    "status": "BUILDING",
                    "updated_at": time.time(),
                }
            except Exception:
                pass
        time.sleep(1.0)

    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        err_tail = "\n".join(stderr.strip().splitlines()[-30:]) if stderr else "Unknown error"
        status_dict[shard_idx] = {
            "shard_idx": shard_idx,
            "stage": stage,
            "tokens": 0,
            "target": total_tokens,
            "pct": 0.0,
            "rate": 0.0,
            "eta_sec": 0,
            "docs": 0,
            "status": "FAILED",
            "error": err_tail,
            "updated_at": time.time(),
        }
        raise RuntimeError(f"Shard {shard_idx} failed (code {proc.returncode}):\n{err_tail}")

    # Commit volume for this worker so data is persisted immediately
    data_vol.commit()
    final_tokens = total_tokens
    if os.path.exists(train_path):
        final_tokens = os.path.getsize(train_path) // 2
        if os.path.exists(val_path):
            final_tokens += os.path.getsize(val_path) // 2

    status_dict[shard_idx] = {
        "shard_idx": shard_idx,
        "stage": stage,
        "tokens": final_tokens,
        "target": total_tokens,
        "pct": 100.0,
        "rate": 0.0,
        "eta_sec": 0,
        "docs": 0,
        "status": "COMPLETED",
        "updated_at": time.time(),
    }
    return {"shard_idx": shard_idx, "status": "completed", "train_path": train_path, "val_path": val_path}


@app.function(
    image=image,
    cpu=2,
    memory=2048,
    timeout=60 * 10,
    volumes={"/root/data": data_vol},
)
def aggregate_manifests(start_shard: int = 0, end_shard: int = 7) -> dict:
    """Aggregates all completed shard records into a master shards_manifest.json and commits."""
    import json
    import os

    data_vol.reload()
    master_manifest = {
        "curriculum_version": "2.0-NVIDIA-Nemotron-Refined",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "shards": [],
    }

    total_tokens_all = 0
    for s in range(start_shard, end_shard + 1):
        train_path = f"/root/data/train_shard{s}.bin"
        val_path = f"/root/data/val_shard{s}.bin"
        if os.path.exists(train_path):
            train_bytes = os.path.getsize(train_path)
            val_bytes = os.path.getsize(val_path) if os.path.exists(val_path) else 0
            train_toks = train_bytes // 2  # uint16 = 2 bytes/token
            val_toks = val_bytes // 2
            total_toks = train_toks + val_toks
            total_tokens_all += total_toks
            master_manifest["shards"].append({
                "shard_idx": s,
                "stage": 1 if s <= 7 else (2 if s <= 10 else 3),
                "train_path": train_path,
                "val_path": val_path,
                "train_tokens": train_toks,
                "val_tokens": val_toks,
                "total_tokens": total_toks,
                "status": "completed",
            })

    master_manifest["total_tokens_all_shards"] = total_tokens_all
    manifest_path = "/root/data/shards_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(master_manifest, f, indent=2)

    data_vol.commit()
    print(f"\n[Modal Pipeline] Aggregated master manifest with {len(master_manifest['shards'])} shards ({total_tokens_all:,} total tokens). Committed to 'pulsar-data'.")
    return master_manifest


def format_monitor_table(shard_list: list[int], status_map: dict, start_time: float, total_target_tokens: int) -> str:
    """Format a clean, static, non-scrolling dashboard table for all active shards."""
    now = time.time()
    elapsed_sec = int(now - start_time)
    mins, secs = divmod(elapsed_sec, 60)

    total_tokens_now = 0
    total_docs_now = 0
    total_rate = 0.0

    rows = []
    for s in shard_list:
        st = status_map.get(s, {})
        toks = st.get("tokens", 0)
        tgt = st.get("target", 500_000_000)
        pct = st.get("pct", 0.0)
        rate = st.get("rate", 0.0)
        eta = st.get("eta_sec", 0.0)
        docs = st.get("docs", 0)
        status = st.get("status", "QUEUED")
        stage = st.get("stage", 1)

        total_tokens_now += toks
        total_docs_now += docs
        if status in ("BUILDING", "STARTING"):
            total_rate += rate

        eta_str = f"{eta / 60:4.1f}m" if eta > 0 else ("DONE" if status == "COMPLETED" else "--")
        rows.append(
            f"   {s:^5d} |   {stage:^5d} | {pct:5.1f}%  | {toks / 1e6:6.1f}M / {tgt / 1e6:5.1f}M | {rate / 1e6:5.2f}M t/s |  {eta_str:^6s} | {docs:10,d} | {status:^10s} "
        )

    overall_pct = (100.0 * total_tokens_now / max(1, total_target_tokens))
    overall_eta = (total_target_tokens - total_tokens_now) / max(1.0, total_rate)
    overall_eta_str = f"{overall_eta / 60:4.1f}m" if overall_eta > 0 and total_tokens_now < total_target_tokens else ("DONE" if overall_pct >= 99.9 else "--")

    sep = "=" * 98
    thin_sep = "-" * 98
    header = (
        f"{sep}\n"
        f"                  PULSAR PRETRAINING SHARD BUILDER - STATIC MONITOR\n"
        f"{sep}\n"
        f"   Shard | Stage | Progress|    Tokens / Target    |    Rate    |   ETA   |  Documents |   Status   \n"
        f"{thin_sep}\n"
    )
    body = "\n".join(rows)
    all_completed = all(status_map.get(s, {}).get("status") == "COMPLETED" for s in shard_list)
    footer_status = "COMPLETED" if all_completed else "RUNNING"
    footer = (
        f"\n{thin_sep}\n"
        f"   TOTAL |   -   | {overall_pct:5.1f}%  | {total_tokens_now / 1e6:6.1f}M / {total_target_tokens / 1e6:5.1f}M | {total_rate / 1e6:5.2f}M t/s |  {overall_eta_str:^6s} | {total_docs_now:10,d} | {footer_status:^10s} \n"
        f"{sep}\n"
        f" Elapsed: {mins:02d}:{secs:02d} | Shards: {shard_list} | Active Workers: {len(shard_list)} | Volume: 'pulsar-data'\n"
        f"{sep}\n"
    )
    return header + body + footer


@app.local_entrypoint()
def main(
    start_shard: int = 0,
    end_shard: int = 7,
    shards: str = None,
    stage: int = None,
    total_tokens: int = 500_000_000,
    batch_size: int = 2048,
    parallel: bool = True,
    skip_existing: bool = True,
    hf_token: str = None,
    min_doc_tokens: int = 16,
    max_doc_tokens: int = 0,
    use_jev: bool = False,
    jev_max_usd: float = 4.80,
    typesafe_api_key: str = None,
):
    import os
    if not hf_token:
        token_path = os.path.expanduser("~/.cache/huggingface/token")
        if os.path.exists(token_path):
            with open(token_path, "r", encoding="utf-8") as f:
                hf_token = f.read().strip()
                print("[Modal Shard Builder] Successfully loaded HF_TOKEN from ~/.cache/huggingface/token")

    ACTIVE_TYPESAFE_KEY = "apikey_255bfcee293b54f4fb0b99de3c2e0d24a6f_dc50b9ee756bd9c6f5d352f6d00afcc13ffb57556b90c22db441b4f7861367a2"
    if not typesafe_api_key or typesafe_api_key.startswith("apikey_255f0941"):
        env_k = os.environ.get("JEV_API_KEY") or os.environ.get("TYPESAFE_API_KEY")
        if env_k and not env_k.startswith("apikey_255f0941"):
            typesafe_api_key = env_k
        else:
            typesafe_api_key = ACTIVE_TYPESAFE_KEY

    if shards:
        shard_list = [int(s.strip()) for s in shards.split(",") if s.strip()]
    else:
        shard_list = list(range(start_shard, end_shard + 1))

    if skip_existing:
        try:
            entries = data_vol.listdir("/")
            existing_files = {e.path.lstrip("/"): e for e in entries if hasattr(e, "path")}
            filtered_shards = []
            for s in shard_list:
                t_file = f"train_shard{s}.bin"
                v_file = f"val_shard{s}.bin"
                if t_file in existing_files and v_file in existing_files:
                    print(f"[Modal Shard Builder] Shard {s} already exists on 'pulsar-data' -> Skipping.")
                else:
                    filtered_shards.append(s)
            shard_list = filtered_shards
        except Exception as e:
            print(f"[Modal Shard Builder] Warning: Could not inspect volume entries ({e}), building requested shards.")

    if not shard_list:
        print("[Modal Shard Builder] All requested shards already exist! Aggregating master manifest...")
        aggregate_manifests.remote(start_shard, end_shard)
        return

    # Clean status dictionary for active shards
    for s in shard_list:
        try:
            del status_dict[s]
        except KeyError:
            pass

    num_workers = max(1, len(shard_list))
    per_worker_jev_usd = jev_max_usd / num_workers
    if use_jev:
        print(f"[Modal Orchestrator] TypeSafe Jev enabled. Total Budget: ${jev_max_usd:.2f} | Workers: {num_workers} | Budget per Worker: ${per_worker_jev_usd:.4f}")

    tasks = [
        (
            s,
            stage if stage is not None else (1 if s <= 7 else (2 if s <= 10 else 3)),
            total_tokens,
            batch_size,
            hf_token,
            min_doc_tokens,
            max_doc_tokens,
            use_jev,
            per_worker_jev_usd,
            typesafe_api_key,
        )
        for s in shard_list
    ]

    start_time = time.time()
    total_target_all = len(shard_list) * total_tokens
    is_tty = sys.stdout.isatty()
    last_print_time = 0

    if parallel:
        print(f"\n[Modal Orchestrator] Spawning {len(shard_list)} parallel containers for shards {shard_list}...")
        handles = {s: generate_single_shard.spawn(*task) for s, task in zip(shard_list, tasks)}
        pending_shards = set(shard_list)

        while pending_shards:
            finished = []
            for s in list(pending_shards):
                h = handles[s]
                try:
                    res = h.get(timeout=0.05)
                    finished.append(s)
                except TimeoutError:
                    pass
                except Exception as e:
                    print(f"\n[Modal Orchestrator ERROR] Shard {s} error: {e}")
                    finished.append(s)

            for s in finished:
                pending_shards.remove(s)

            status_map = {}
            for s in shard_list:
                if s in status_dict:
                    status_map[s] = status_dict[s]

            now = time.time()
            if is_tty:
                if now - last_print_time >= 1.5 or not pending_shards:
                    sys.stdout.write("\033[H\033[2J")
                    sys.stdout.write(format_monitor_table(shard_list, status_map, start_time, total_target_all))
                    sys.stdout.flush()
                    last_print_time = now
            else:
                if now - last_print_time >= 20.0 or not pending_shards:
                    print(format_monitor_table(shard_list, status_map, start_time, total_target_all), flush=True)
                    last_print_time = now

            time.sleep(1.0)

        # Final static monitor render
        status_map = {s: status_dict.get(s, {}) for s in shard_list}
        print(format_monitor_table(shard_list, status_map, start_time, total_target_all), flush=True)
        aggregate_manifests.remote(start_shard, end_shard)

    else:
        print(f"\n[Modal Orchestrator] Launching sequential execution for shards {shard_list}...")
        for t in tasks:
            res = generate_single_shard.remote(*t)
            print(f"[Modal Orchestrator] Shard {res['shard_idx']} completed successfully: {res['status']}")
        aggregate_manifests.remote(start_shard, end_shard)

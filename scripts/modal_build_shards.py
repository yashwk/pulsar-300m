"""Modal App for Cloud-Native Shard Generation.

Generates pretraining shards directly in the cloud on high-performance multi-core
CPU instances using 10 Gbps datacenter pipes to Hugging Face, writing directly to
a persistent modal.Volume('pulsar-data').
"""

import subprocess
import time
import modal

# 1. Persistent volume for storing binary training shards
data_vol = modal.Volume.from_name("pulsar-data", create_if_missing=True)

# 2. Container image with HuggingFace, fast tokenizers, and PyArrow
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "huggingface_hub[hf_transfer]>=0.22.0",
        "pyarrow>=14.0.0",
        "transformers>=4.40.0",
        "tiktoken",
        "numpy",
        "tqdm",
    )
    .env({
        "HF_XET_HIGH_PERFORMANCE": "0",
        "PYTHONUNBUFFERED": "1",
    })
    .add_local_dir("data/pulsar_tokenizer", remote_path="/root/pulsar_tokenizer")
    .add_local_file("data/prepare_shards.py", remote_path="/root/prepare_shards.py")
)

app = modal.App("pulsar-shard-builder")


@app.function(
    image=image,
    cpu=8,                          # 8 vCPUs (4 physical cores) for balanced, cost-effective tokenization
    memory=8192,                    # 8 GB RAM (cost-efficient, avoids paying for idle memory)
    timeout=60 * 60,                # 60 min safety timeout
    volumes={"/root/data": data_vol},
)
def generate_single_shard(
    shard_idx: int,
    stage: int = 1,
    total_tokens: int = 500_000_000,
    batch_size: int = 2048,
    hf_token: str = None,
) -> dict:
    """Generates a single pretraining shard with isolated file paths and commits to volume."""
    import os
    import subprocess
    from transformers import AutoTokenizer

    # 1. Assert 32,768 vocabulary inside Modal container
    tok = AutoTokenizer.from_pretrained("/root/pulsar_tokenizer")
    assert len(tok) == 32768, f"Modal container tokenizer vocab size mismatch: expected 32768, got {len(tok)}"
    print(f"[Modal Worker {shard_idx}] Verified tokenizer vocabulary size: {len(tok):,} tokens")

    env = os.environ.copy()
    if hf_token:
        env["HF_TOKEN"] = hf_token
        env["HUGGING_FACE_HUB_TOKEN"] = hf_token
    env["HF_XET_HIGH_PERFORMANCE"] = "0"
    env["PYTHONUNBUFFERED"] = "1"

    train_path = f"/root/data/train_shard{shard_idx}.bin"
    val_path = f"/root/data/val_shard{shard_idx}.bin"

    print(f"\n{'='*70}")
    print(f"[Modal Worker] Generating Shard {shard_idx} (Stage {stage})")
    print(f"   Target Tokens: {total_tokens:,} | Batch Size: {batch_size}")
    print(f"   Output: {train_path} & {val_path}")
    print(f"{'='*70}\n")

    cmd = [
        "python", "-u", "/root/prepare_shards.py",
        "--shard-idx", str(shard_idx),
        "--stage", str(stage),
        "--total-tokens", str(total_tokens),
        "--train-path", train_path,
        "--val-path", val_path,
        "--tokenizer", "/root/pulsar_tokenizer",
        "--batch-size", str(batch_size),
    ]

    subprocess.run(cmd, env=env, check=True)

    # Commit volume for this worker so data is persisted immediately
    data_vol.commit()
    print(f"\n[Modal Volume] Shard {shard_idx} successfully built and committed to volume 'pulsar-data'.")
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
):
    import os
    if not hf_token:
        token_path = os.path.expanduser("~/.cache/huggingface/token")
        if os.path.exists(token_path):
            with open(token_path, "r", encoding="utf-8") as f:
                hf_token = f.read().strip()
                print("[Modal Shard Builder] Successfully loaded HF_TOKEN from ~/.cache/huggingface/token")

    if shards:
        shard_list = [int(s.strip()) for s in shards.split(",") if s.strip()]
    else:
        shard_list = list(range(start_shard, end_shard + 1))

    if skip_existing:
        # Check existing shards on volume to avoid burning credits on already completed shards
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

    tasks = [
        (s, stage if stage is not None else (1 if s <= 7 else (2 if s <= 10 else 3)), total_tokens, batch_size, hf_token)
        for s in shard_list
    ]

    if parallel:
        print(f"\n[Modal Orchestrator] Launching {len(shard_list)} parallel containers for shards {shard_list}...")
        for res in generate_single_shard.starmap(tasks):
            print(f"[Modal Orchestrator] Shard {res['shard_idx']} completed successfully: {res['status']}")
        aggregate_manifests.remote(start_shard, end_shard)
    else:
        print(f"\n[Modal Orchestrator] Launching sequential execution for shards {shard_list}...")
        for t in tasks:
            res = generate_single_shard.remote(*t)
            print(f"[Modal Orchestrator] Shard {res['shard_idx']} completed successfully: {res['status']}")
        aggregate_manifests.remote(start_shard, end_shard)

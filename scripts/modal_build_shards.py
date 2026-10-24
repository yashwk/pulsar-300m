"""Modal App for Cloud-Native Shard Generation.

Generates pretraining shards directly in the cloud on high-performance multi-core
CPU instances using 10 Gbps datacenter pipes to Hugging Face, writing directly to
a persistent modal.Volume('pulsar-data').
"""

import subprocess
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
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_XET_HIGH_PERFORMANCE": "1",
        "PYTHONUNBUFFERED": "1",
    })
    .add_local_dir("data/pulsar_tokenizer", remote_path="/root/pulsar_tokenizer")
    .add_local_file("data/prepare_shards.py", remote_path="/root/prepare_shards.py")
)

app = modal.App("pulsar-shard-builder")


@app.function(
    image=image,
    cpu=16,                         # 16 CPU cores for 2.5M tok/s fast tokenization (~3.5 min/shard)
    memory=8192,                    # 8 GB RAM (cost-efficient, avoids paying for idle memory)
    timeout=60 * 45,                # 45 min safety timeout
    volumes={"/root/data": data_vol},
)
def generate_shards_job(start_shard: int = 1, end_shard: int = 1, stage: int = None, total_tokens: int = 500_000_000, batch_size: int = 2048, hf_token: str = None):
    """Generates one or more shards sequentially in the same container, reusing cached parquet files."""
    import os
    env = os.environ.copy()
    if hf_token:
        env["HF_TOKEN"] = hf_token
        env["HUGGING_FACE_HUB_TOKEN"] = hf_token
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    env["HF_XET_HIGH_PERFORMANCE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    for shard_idx in range(start_shard, end_shard + 1):
        # Auto-detect stage from curriculum definition if not explicitly overridden:
        # Shards 1..7: Stage 1 (Foundation, T=1024)
        # Shards 8..10: Stage 2 (Reasoning Upsample, T=2048)
        # Shard 11: Stage 3 (Long-Context 32k Cooldown)
        current_stage = stage if stage is not None else (1 if shard_idx <= 7 else (2 if shard_idx <= 10 else 3))

        print(f"\n{'='*70}")
        print(f"🚀 [Modal Worker] Generating Shard {shard_idx} (Stage {current_stage})")
        print(f"   Target Tokens: {total_tokens:,} | Batch Size: {batch_size}")
        print(f"   Output: /root/data/train_shard{shard_idx}.bin & /root/data/val_shard{shard_idx}.bin")
        print(f"{'='*70}\n")

        train_path = f"/root/data/train_shard{shard_idx}.bin"
        val_path = f"/root/data/val_shard{shard_idx}.bin"

        cmd = [
            "python", "-u", "/root/prepare_shards.py",
            "--shard-idx", str(shard_idx),
            "--stage", str(current_stage),
            "--total-tokens", str(total_tokens),
            "--train-path", train_path,
            "--val-path", val_path,
            "--tokenizer", "/root/pulsar_tokenizer",
            "--batch-size", str(batch_size),
        ]

        subprocess.run(cmd, env=env, check=True)

        # Commit volume after each shard completes so data is immediately safe
        data_vol.commit()
        print(f"\n✅ Shard {shard_idx} successfully built and committed to volume 'pulsar-data'!")

    print(f"\n🎉 All requested shards ({start_shard}..{end_shard}) completed and saved permanently!")


@app.local_entrypoint()
def main(shard_idx: int = 1, end_shard: int = None, stage: int = None, total_tokens: int = 500_000_000, batch_size: int = 2048, hf_token: str = None):
    if end_shard is None:
        end_shard = shard_idx
    generate_shards_job.remote(
        start_shard=shard_idx,
        end_shard=end_shard,
        stage=stage,
        total_tokens=total_tokens,
        batch_size=batch_size,
        hf_token=hf_token,
    )

"""Modal App for Pulsar-300M Pretraining.

Runs high-performance training on cloud GPUs (L4, A100, H100) mounted with
persistent volumes for training shards ('pulsar-data') and checkpoints ('pulsar-checkpoints').
"""

import subprocess
import modal

# 1. Persistent Volumes
data_vol = modal.Volume.from_name("pulsar-data", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("pulsar-checkpoints", create_if_missing=True)

# 2. PyTorch CUDA Image
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0",
        "transformers",
        "numpy",
        "tiktoken",
        "tqdm",
    )
    .add_local_dir("src", remote_path="/root/src")
    .add_local_dir("data/pulsar_tokenizer", remote_path="/root/pulsar_tokenizer")
)

app = modal.App("pulsar-training")


@app.function(
    image=image,
    gpu="L4",                       # Default to L4 ($0.80/hr). Can be changed via CLI or config
    timeout=60 * 60 * 3,            # 3 hours max runtime
    volumes={
        "/root/data": data_vol,
        "/checkpoints": ckpt_vol,
    },
)
def train_remote(
    train_shards: str = "/root/data/train_shard1.bin,/root/data/train_shard2.bin",
    val_shards: str = "/root/data/val_shard1.bin,/root/data/val_shard2.bin",
    target_step: int = 500,
    micro_batch_size: int = 4,
    global_batch_tokens: int = 131072,
    seq_len: int = 1024,
    muon_lr: float = 0.02,
    adam_lr: float = 6e-4,
    eval_every: int = 50,
    save_every: int = 100,
    resume: str = None,
):
    """Executes Pulsar pretraining on remote GPU."""
    # Ensure checkpoint and data volumes are in sync
    ckpt_vol.reload()
    data_vol.reload()

    cmd = [
        "python", "/root/src/train.py",
        "--train-bin", train_shards,
        "--val-bin", val_shards,
        "--tokenizer", "/root/pulsar_tokenizer",
        "--seq-len", str(seq_len),
        "--micro-batch-size", str(micro_batch_size),
        "--global-batch-tokens", str(global_batch_tokens),
        "--max-steps", "30517",
        "--target-step", str(target_step),
        "--warmup-steps", "300",
        "--cooldown-steps", "0",
        "--lr-schedule", "wsd",
        "--muon-lr", str(muon_lr),
        "--adam-lr", str(adam_lr),
        "--bias-lr", "0.01",
        "--muon-momentum-fp32",
        "--moe-capacity-factor", "1.50",
        "--sampler", "flat",
        "--eval-every", str(eval_every),
        "--eval-iters", "20",
        "--save-every", str(save_every),
        "--checkpoint-dir", "/checkpoints",
    ]
    if resume:
        cmd.extend(["--resume", resume])

    print(f"\n{'='*70}")
    print(f"🚀 [Modal GPU Trainer] Launching Training")
    print(f"   Command: {' '.join(cmd)}")
    print(f"{'='*70}\n")

    subprocess.run(cmd, check=True)

    # Commit checkpoints to persistent storage
    ckpt_vol.commit()
    print("\n✅ Training session finished! Checkpoints successfully saved to 'pulsar-checkpoints' volume.")


@app.local_entrypoint()
def main(
    train_shards: str = "/root/data/train_shard1.bin,/root/data/train_shard2.bin",
    val_shards: str = "/root/data/val_shard1.bin,/root/data/val_shard2.bin",
    target_step: int = 500,
    mbs: int = 4,
    resume: str = None,
):
    train_remote.remote(
        train_shards=train_shards,
        val_shards=val_shards,
        target_step=target_step,
        micro_batch_size=mbs,
        resume=resume,
    )

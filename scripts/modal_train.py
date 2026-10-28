"""Modal App for Pulsar-300M Pretraining.

Runs high-performance training on cloud GPUs (B200, H200, H100) mounted with
persistent volumes for training shards ('pulsar-data') and checkpoints ('pulsar-checkpoints').
"""

import os
import subprocess
import threading
import time
import modal

# 1. Persistent Volumes
data_vol = modal.Volume.from_name("pulsar-data", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("pulsar-checkpoints", create_if_missing=True)

# 2. PyTorch CUDA Image supporting sm_100 (Blackwell) & sm_90 (Hopper)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4.0",
        "transformers",
        "numpy",
        "tiktoken",
        "tqdm",
    )
    .add_local_dir("src", remote_path="/root/src")
    .add_local_dir("data/pulsar_tokenizer", remote_path="/root/pulsar_tokenizer")
)

app = modal.App("pulsar-training")

ALL_TRAIN_SHARDS = ",".join([f"/root/data/train_shard{i}.bin" for i in range(8)])
ALL_VAL_SHARDS = ",".join([f"/root/data/val_shard{i}.bin" for i in range(8)])


def execute_training(
    train_shards: str,
    val_shards: str,
    target_step: int,
    micro_batch_size: int,
    global_batch_tokens: int,
    seq_len: int,
    muon_lr: float,
    adam_lr: float,
    eval_every: int,
    save_every: int,
    resume: str = None,
    compile: bool = True,
    output_ckpt: str = None,
):
    """Core training harness with background volume sync."""
    ckpt_vol.reload()
    data_vol.reload()

    # Launch periodic volume committer so telemetry streams live
    stop_event = threading.Event()

    def periodic_commit():
        while not stop_event.wait(timeout=10.0):
            try:
                ckpt_vol.commit()
            except Exception:
                pass

    committer_thread = threading.Thread(target=periodic_commit, daemon=True)
    committer_thread.start()

    cmd = [
        "python", "-u", "/root/src/train.py",
        "--train-bin", train_shards,
        "--val-bin", val_shards,
        "--tokenizer", "/root/pulsar_tokenizer",
        "--seq-len", str(seq_len),
        "--micro-batch-size", str(micro_batch_size),
        "--global-batch-tokens", str(global_batch_tokens),
        "--max-steps", "30517",
        "--target-step", str(target_step),
        "--warmup-steps", "150",
        "--cooldown-steps", "0",
        "--lr-schedule", "wsd",
        "--muon-lr", str(muon_lr),
        "--adam-lr", str(adam_lr),
        "--bias-lr", "0.01",
        "--muon-momentum-fp32",
        "--moe-capacity-factor", "1.50",
        "--sampler", "flat",
        "--eval-every", str(eval_every),
        "--eval-iters", "10",
        "--save-every", str(save_every),
        "--checkpoint-dir", "/checkpoints",
    ]
    if compile:
        cmd.append("--compile")
    if output_ckpt:
        cmd.extend(["--output-ckpt", output_ckpt])
    if resume:
        cmd.extend(["--resume", resume])

    print(f"\n{'='*70}")
    print(f"[Modal GPU Trainer] Launching Training")
    print(f"   Command: {' '.join(cmd)}")
    print(f"{'='*70}\n")

    try:
        subprocess.run(cmd, check=True)
    finally:
        stop_event.set()
        ckpt_vol.commit()
        print("\n[Modal Checkpoints] Training session finished. Checkpoints successfully committed to 'pulsar-checkpoints' volume.")


@app.function(
    image=image,
    gpu="B200",
    timeout=60 * 60 * 4,
    volumes={"/root/data": data_vol, "/checkpoints": ckpt_vol},
)
def train_remote_b200(**kwargs):
    execute_training(**kwargs)


@app.function(
    image=image,
    gpu="H200",
    timeout=60 * 60 * 4,
    volumes={"/root/data": data_vol, "/checkpoints": ckpt_vol},
)
def train_remote_h200(**kwargs):
    execute_training(**kwargs)


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60 * 4,
    volumes={"/root/data": data_vol, "/checkpoints": ckpt_vol},
)
def train_remote_h100(**kwargs):
    execute_training(**kwargs)


@app.local_entrypoint()
def main(
    gpu: str = "B200",
    train_shards: str = ALL_TRAIN_SHARDS,
    val_shards: str = ALL_VAL_SHARDS,
    target_step: int = 30,
    mbs: int = 8,
    global_batch_tokens: int = 131072,
    seq_len: int = 1024,
    muon_lr: float = 0.02,
    adam_lr: float = 6e-4,
    eval_every: int = 10,
    save_every: int = 30,
    resume: str = None,
    compile: bool = True,
    output_ckpt: str = None,
):
    kwargs = {
        "train_shards": train_shards,
        "val_shards": val_shards,
        "target_step": target_step,
        "micro_batch_size": mbs,
        "global_batch_tokens": global_batch_tokens,
        "seq_len": seq_len,
        "muon_lr": muon_lr,
        "adam_lr": adam_lr,
        "eval_every": eval_every,
        "save_every": save_every,
        "resume": resume,
        "compile": compile,
        "output_ckpt": output_ckpt,
    }

    selected_gpu = gpu.upper()
    print(f"\n[Modal Orchestrator] Target GPU: {selected_gpu} | Target Step: {target_step} | MBS: {mbs}")

    if selected_gpu == "B200":
        try:
            print("[Modal Orchestrator] Attempting to provision Priority 1: NVIDIA B200 (Blackwell)...")
            train_remote_b200.remote(**kwargs)
            return
        except Exception as e:
            print(f"[Modal Orchestrator] B200 allocation unavailable ({e}). Falling back to Priority 2: H200...")
            selected_gpu = "H200"

    if selected_gpu == "H200":
        try:
            print("[Modal Orchestrator] Attempting to provision Priority 2: NVIDIA H200...")
            train_remote_h200.remote(**kwargs)
            return
        except Exception as e:
            print(f"[Modal Orchestrator] H200 allocation unavailable ({e}). Falling back to Priority 3: H100...")
            selected_gpu = "H100"

    if selected_gpu == "H100":
        print("[Modal Orchestrator] Provisioning Priority 3: NVIDIA H100...")
        train_remote_h100.remote(**kwargs)

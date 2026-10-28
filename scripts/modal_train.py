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

# 2. PyTorch CUDA Image supporting sm_103 (Blackwell Ultra), sm_100 (Blackwell) & sm_90 (Hopper)
image = (
    modal.Image.from_registry("nvidia/cuda:13.2.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("build-essential", "ninja-build")
    .pip_install(
        "torch>=2.4.0",
        "transformers",
        "numpy",
        "tiktoken",
        "tqdm",
    )
    .add_local_dir("csrc", remote_path="/root/pulsar_src/csrc", copy=True)
    .add_local_file("setup.py", remote_path="/root/pulsar_src/setup.py", copy=True)
    .add_local_file("pyproject.toml", remote_path="/root/pulsar_src/pyproject.toml", copy=True)
    .add_local_file("README.md", remote_path="/root/pulsar_src/README.md", copy=True)
    .add_local_dir("src", remote_path="/root/pulsar_src/src", copy=True)
    .run_commands(
        "cd /root/pulsar_src && "
        "pip install --no-build-isolation --no-deps . && "
        "python -c \"import torch, pulsar_cuda; print('Verified pulsar_cuda:', pulsar_cuda.__file__)\"",
        env={
            "CC": "gcc",
            "CXX": "g++",
            "TORCH_DONT_CHECK_COMPILER_ABI": "1",
            "TORCH_CUDA_ARCH_LIST": "8.9;9.0;10.0+PTX",
            "MAX_JOBS": "4",
        },
    )
    .env({
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCHINDUCTOR_CACHE_DIR": "/checkpoints/.inductor_cache",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "PYTHONUNBUFFERED": "1",
    })
    .add_local_dir("src", remote_path="/root/src")
    .add_local_dir("data/pulsar_tokenizer", remote_path="/root/pulsar_tokenizer")
)

app = modal.App("pulsar-training")

ALL_TRAIN_SHARDS = ",".join([f"/root/data/train_shard{i}.bin" for i in range(8)])
ALL_VAL_SHARDS = ",".join([f"/root/data/val_shard{i}.bin" for i in range(8)])


def parse_tokens(val: str | int | float | None) -> int | None:
    """Parse human token representations like '3.8B', '4M', '131k', '131072'."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return int(val)
    s = str(val).strip().upper()
    if s.endswith("B"):
        return int(float(s[:-1]) * 1_000_000_000)
    elif s.endswith("M"):
        return int(float(s[:-1]) * 1_000_000)
    elif s.endswith("K"):
        return int(float(s[:-1]) * 1_000)
    return int(float(s))


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
    compile: bool = False,
    output_ckpt: str = None,
    max_steps: int = None,
    warmup_steps: int = 300,
    sampler: str = "bestfit",
    num_pulses: int = 3,
    moe_capacity_factor: float = 1.25,
    bias_lr: float = 0.01,
    per_pulse_bias: bool = True,
):
    """Core training harness with background volume sync."""
    ckpt_vol.reload()
    data_vol.reload()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/root/pulsar_tokenizer")
    assert len(tok) == 32768, f"Modal container tokenizer vocab size mismatch: expected 32768, got {len(tok)}"
    print(f"[Modal Trainer] Verified container tokenizer vocabulary size: {len(tok):,} tokens")

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

    effective_max_steps = max_steps if max_steps is not None else max(1, 3_800_000_000 // global_batch_tokens)

    cmd = [
        "python", "-u", "/root/src/train.py",
        "--train-bin", train_shards,
        "--val-bin", val_shards,
        "--tokenizer", "/root/pulsar_tokenizer",
        "--seq-len", str(seq_len),
        "--micro-batch-size", str(micro_batch_size),
        "--global-batch-tokens", str(global_batch_tokens),
        "--max-steps", str(effective_max_steps),
        "--target-step", str(target_step),
        "--warmup-steps", str(warmup_steps),
        "--cooldown-steps", "0",
        "--lr-schedule", "wsd",
        "--muon-lr", str(muon_lr),
        "--adam-lr", str(adam_lr),
        "--bias-lr", str(bias_lr),
        "--muon-momentum-fp32",
        "--moe-capacity-factor", str(moe_capacity_factor),
        "--sampler", sampler,
        "--num-pulses", str(num_pulses),
        "--eval-every", str(eval_every),
        "--eval-iters", "10",
        "--save-every", str(save_every),
        "--checkpoint-dir", "/checkpoints",
    ]
    if per_pulse_bias:
        cmd.append("--per-pulse-bias")
    else:
        cmd.append("--no-per-pulse-bias")
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
    gpu="B300",
    timeout=60 * 60 * 8,
    volumes={"/root/data": data_vol, "/checkpoints": ckpt_vol},
)
def train_remote_b300(**kwargs):
    execute_training(**kwargs)


@app.function(
    image=image,
    gpu="B200",
    timeout=60 * 60 * 8,
    volumes={"/root/data": data_vol, "/checkpoints": ckpt_vol},
)
def train_remote_b200(**kwargs):
    execute_training(**kwargs)


@app.function(
    image=image,
    gpu="H200",
    timeout=60 * 60 * 8,
    volumes={"/root/data": data_vol, "/checkpoints": ckpt_vol},
)
def train_remote_h200(**kwargs):
    execute_training(**kwargs)


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60 * 8,
    volumes={"/root/data": data_vol, "/checkpoints": ckpt_vol},
)
def train_remote_h100(**kwargs):
    execute_training(**kwargs)


@app.local_entrypoint()
def main(
    gpu: str = "B300",
    train_shards: str = ALL_TRAIN_SHARDS,
    val_shards: str = ALL_VAL_SHARDS,
    target_step: int = None,
    target_tokens: str = None,
    max_steps: int = None,
    max_tokens: str = None,
    mbs: int = 128,
    global_batch_tokens: int = 131072,
    seq_len: int = 1024,
    warmup_steps: int = None,
    warmup_tokens: str = None,
    sampler: str = "bestfit",
    num_pulses: int = 3,
    moe_capacity_factor: float = 1.25,
    bias_lr: float = 0.01,
    per_pulse_bias: bool = True,
    muon_lr: float = 0.02,
    adam_lr: float = 6e-4,
    eval_every: int = 15,
    save_every: int = 30,
    resume: str = None,
    compile: bool = False,
    output_ckpt: str = None,
):
    # Calculate invariant hardware-agnostic batch dynamics
    tokens_per_micro = mbs * seq_len
    grad_accum = max(1, (global_batch_tokens + tokens_per_micro - 1) // tokens_per_micro)
    effective_gbt = grad_accum * tokens_per_micro

    # 1. Resolve total curriculum steps from max_tokens or default 3.8B curriculum
    if max_steps is None:
        total_tok = parse_tokens(max_tokens) if max_tokens else 3_800_000_000
        effective_max_steps = max(1, total_tok // effective_gbt)
    else:
        effective_max_steps = max_steps

    # 2. Resolve target_step from target_tokens if provided, else default to step 30
    if target_tokens is not None:
        target_tok_val = parse_tokens(target_tokens)
        effective_target_step = max(1, (target_tok_val + effective_gbt - 1) // effective_gbt)
    elif target_step is not None:
        effective_target_step = target_step
    else:
        effective_target_step = 30

    # 3. Resolve warmup_steps from warmup_tokens or 1% curriculum default
    if warmup_tokens is not None:
        effective_warmup_steps = max(1, parse_tokens(warmup_tokens) // effective_gbt)
    elif warmup_steps is not None:
        effective_warmup_steps = warmup_steps
    else:
        effective_warmup_steps = max(1, int(round(effective_max_steps * 0.01)))

    tokens_targeted = effective_target_step * effective_gbt
    total_curriculum_tokens = effective_max_steps * effective_gbt

    print("\n" + "=" * 70)
    print(" [Modal Orchestrator] Pulsar-300M Curriculum & Batch Dynamics")
    print("=" * 70)
    print(f" Target GPU:                 NVIDIA {gpu.upper()}")
    print(f" Micro-Batch Size (MBS):     {mbs} (tokens/micro: {tokens_per_micro:,})")
    print(f" Gradient Accumulation:      GAS = {grad_accum}")
    print(f" Global Batch Tokens:        {effective_gbt:,} tokens/step")
    print(f" Total Curriculum Horizon:   {effective_max_steps:,} steps ({total_curriculum_tokens:,} tokens)")
    print(f" Warmup Schedule:            {effective_warmup_steps:,} steps ({effective_warmup_steps * effective_gbt:,} tokens)")
    print(f" Active Session Target:      Step {effective_target_step:,} ({tokens_targeted:,} tokens)")
    print("=" * 70 + "\n")

    kwargs = {
        "train_shards": train_shards,
        "val_shards": val_shards,
        "target_step": effective_target_step,
        "max_steps": effective_max_steps,
        "micro_batch_size": mbs,
        "global_batch_tokens": effective_gbt,
        "seq_len": seq_len,
        "warmup_steps": effective_warmup_steps,
        "sampler": sampler,
        "num_pulses": num_pulses,
        "moe_capacity_factor": moe_capacity_factor,
        "bias_lr": bias_lr,
        "per_pulse_bias": per_pulse_bias,
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

    if selected_gpu == "B300":
        print("[Modal Orchestrator] Provisioning NVIDIA B300 SXM6 (Blackwell Ultra)...")
        train_remote_b300.remote(**kwargs)
    elif selected_gpu == "B200":
        print("[Modal Orchestrator] Provisioning NVIDIA B200 (Blackwell)...")
        train_remote_b200.remote(**kwargs)
    elif selected_gpu == "H200":
        print("[Modal Orchestrator] Provisioning NVIDIA H200...")
        train_remote_h200.remote(**kwargs)
    elif selected_gpu == "H100":
        print("[Modal Orchestrator] Provisioning NVIDIA H100...")
        train_remote_h100.remote(**kwargs)
    else:
        raise ValueError(f"Unsupported GPU selection: {selected_gpu}")

#!/usr/bin/env python3
"""Pulsar-300M Checkpoint Inspector.

Quickly inspects metadata, hyperparameters, step numbers, validation losses,
and tensor shapes from a Pulsar checkpoint without allocating GPU memory.
"""

import argparse
import os
import sys
import torch


def inspect_checkpoint(checkpoint_path: str, show_tensors: bool = False):
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found: {checkpoint_path}")
        sys.exit(1)

    file_size_gb = os.path.getsize(checkpoint_path) / 1e9
    print("=" * 70)
    print(f" Pulsar Checkpoint: {checkpoint_path}")
    print(f" File Size: {file_size_gb:.2f} GB")
    print("=" * 70)

    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        sys.exit(1)

    global_step = ckpt.get("global_step", "Unknown")
    tokens_seen = ckpt.get("tokens_seen", 0)
    best_val_loss = ckpt.get("best_val_loss", float("inf"))
    num_pulses = ckpt.get("num_pulses", 3)
    config = ckpt.get("config", {})
    summary = ckpt.get("model_summary", {})

    print(f" Global Step:           {global_step}")
    print(f" Tokens Seen:           {tokens_seen:,} tokens" if isinstance(tokens_seen, int) else f" Tokens Seen: {tokens_seen}")
    print(f" Best Val CE Loss:      {best_val_loss:.4f}" if isinstance(best_val_loss, (int, float)) and best_val_loss != float("inf") else f" Best Val Loss: {best_val_loss}")
    print(f" Recurrent Pulses (K):  {num_pulses}")

    if summary:
        print("\n--- Model Architecture Summary ---")
        for k, v in summary.items():
            print(f"  • {k}: {v}")
    elif config:
        print("\n--- Model Configuration ---")
        for k in ["d_model", "n_core", "n_heads", "n_kv_heads", "head_dim", "n_routed_experts", "top_k", "d_expert", "d_shared", "vocab_size", "max_seq_len"]:
            if k in config:
                print(f"  • {k}: {config[k]}")

    state_dict = ckpt.get("model", {})
    total_params = sum(p.numel() for p in state_dict.values()) if state_dict else 0
    print(f"\n--- Model Parameters ---")
    print(f"  Total Parameters:     {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"  Total Tensors:        {len(state_dict)}")

    # Check optimizer states
    has_muon = "opt_muon" in ckpt and ckpt["opt_muon"] is not None
    has_adam = "opt_adam" in ckpt and ckpt["opt_adam"] is not None
    has_adam_sr = "opt_adam_sr" in ckpt and ckpt["opt_adam_sr"] is not None
    print(f"\n--- Optimizer States ---")
    print(f"  Muon Matrix State:    {'Present' if has_muon else 'None'}")
    print(f"  AdamW Norm/Gate State:{'Present' if has_adam else 'None'}")
    print(f"  AdamW BF16 SR State:  {'Present' if has_adam_sr else 'None'}")

    if show_tensors and state_dict:
        print("\n--- Tensor Directory ---")
        for name, tensor in sorted(state_dict.items()):
            print(f"  {name:45s} : {str(list(tensor.shape)):20s} ({str(tensor.dtype):14s})")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect a Pulsar checkpoint file")
    parser.add_argument("checkpoint", nargs="?", default="checkpoints/latest.pt", help="Path to checkpoint .pt file")
    parser.add_argument("--show-tensors", action="store_true", help="List all tensor names, shapes, and dtypes")
    args = parser.parse_args()

    inspect_checkpoint(args.checkpoint, show_tensors=args.show_tensors)

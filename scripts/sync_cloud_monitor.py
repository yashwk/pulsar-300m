#!/usr/bin/env python3
"""Syncs live telemetry from Modal 'pulsar-checkpoints' volume into a local directory.

Runs a zero-overhead background loop so your local monitor dashboard (http://localhost:7860)
displays real-time loss curves, gauge cards, and canaries from the cloud run.
"""

import argparse
import os
import subprocess
import time


def sync_cloud_telemetry(checkpoint_dir: str = "checkpoints", interval: float = 5.0):
    os.makedirs(checkpoint_dir, exist_ok=True)
    modal_bin = os.path.abspath(".venv/bin/modal")
    if not os.path.exists(modal_bin):
        modal_bin = "modal"

    files_to_sync = ["live_status.json", "metrics.jsonl", "run_config.json", "config.json"]

    print(f"[Sync Monitor] Starting telemetry sync from Modal 'pulsar-checkpoints' -> {checkpoint_dir}")
    while True:
        for fname in files_to_sync:
            dest = os.path.join(checkpoint_dir, fname)
            cmd = [modal_bin, "volume", "get", "--force", "pulsar-checkpoints", f"/{fname}", dest]
            try:
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            except Exception:
                pass
        time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync telemetry from Modal volume to local directory")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Local directory to write synced telemetry (default: checkpoints)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Sync interval in seconds (default: 5.0)",
    )
    args = parser.parse_args()
    sync_cloud_telemetry(checkpoint_dir=args.checkpoint_dir, interval=args.interval)

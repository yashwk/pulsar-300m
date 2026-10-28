#!/usr/bin/env python3
"""Syncs live telemetry from Modal 'pulsar-checkpoints' volume into local checkpoints/ folder.

Runs a zero-overhead background loop so your existing local monitor dashboard (http://localhost:7860)
displays real-time loss curves, gauge cards, and canaries from the cloud run without consuming mobile data.
"""

import os
import subprocess
import time

CHECKPOINT_DIR = "checkpoints"


def sync_cloud_telemetry():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    cmd = [
        ".venv/bin/modal", "volume", "get", "--force",
        "pulsar-checkpoints", "/live_status.json",
        os.path.join(CHECKPOINT_DIR, "live_status.json"),
    ]
    cmd_metrics = [
        ".venv/bin/modal", "volume", "get", "--force",
        "pulsar-checkpoints", "/metrics.jsonl",
        os.path.join(CHECKPOINT_DIR, "metrics.jsonl"),
    ]

    while True:
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            subprocess.run(cmd_metrics, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        except Exception:
            pass
        time.sleep(10.0)


if __name__ == "__main__":
    sync_cloud_telemetry()

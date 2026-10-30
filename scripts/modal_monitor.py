"""Modal App for Pulsar-300M Real-Time Cloud Training Monitor.

Deploys monitor/web_server.py as a serverless Modal web application mounted
directly with the persistent 'pulsar-checkpoints' volume.

Zero local sync daemon required. Telemetry reads occur directly from the cloud
volume with sub-millisecond local volume read latency.

Usage:
    # Ephemeral dev session with live URL:
    modal serve scripts/modal_monitor.py

    # Permanent production deployment:
    modal deploy scripts/modal_monitor.py
"""

from __future__ import annotations

import subprocess
import modal

ckpt_vol = modal.Volume.from_name("pulsar-checkpoints", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch>=2.4.0", "numpy")
    .add_local_dir("monitor", remote_path="/root/monitor")
    .add_local_dir("src", remote_path="/root/src")
)

app = modal.App("pulsar-monitor")


@app.function(
    image=image,
    volumes={"/checkpoints": ckpt_vol},
    timeout=60 * 60,
    scaledown_window=60,
)
@modal.web_server(port=7860, startup_timeout=60)
def serve():
    cmd = [
        "python", "-u", "/root/monitor/web_server.py",
        "--host", "0.0.0.0",
        "--port", "7860",
        "--checkpoint-dir", "/checkpoints",
    ]
    subprocess.Popen(cmd)

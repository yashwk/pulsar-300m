# Pulsar-300M Cloud Training Operations Runbook

This guide covers production pretraining workflows on Modal cloud GPUs (NVIDIA B300 SXM6, B200, H200, H100), persistent volumes, detached execution, and automated recovery.

---

## 1. Cloud Infrastructure & Accelerators

Pulsar pretraining runs inside custom CUDA containers with native C++ extensions compiled for NVIDIA Blackwell and Hopper architectures (`sm_103`, `sm_100`, `sm_90`).

### Supported Hardware Tiers

| GPU | Architecture | Memory | Typical Stage 1 Throughput | Primary Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **NVIDIA B300 SXM6** | Blackwell Ultra | 288 GB HBM3e | ~260,000+ tok/s (24% MFU) | Production Pretraining (Default) |
| **NVIDIA B200** | Blackwell | 192 GB HBM3e | ~220,000+ tok/s (22% MFU) | Production Pretraining |
| **NVIDIA H200** | Hopper | 141 GB HBM3e | ~150,000+ tok/s (28% MFU) | Production Fallback |
| **NVIDIA H100** | Hopper | 80 GB HBM3 | ~110,000+ tok/s (26% MFU) | Trial Runs & Profiling |

### Persistent Cloud Volumes

Two persistent Modal volumes support zero-copy data streaming and durable checkpointing:
1. **`pulsar-data`**: Holds the pretokenized uint16 binary dataset shards (`train_shard0.bin` through `train_shard7.bin` and validation shards).
2. **`pulsar-checkpoints`**: Stores model checkpoints (`latest.pt`, `best.pt`, `milestone_step_*.pt`) and real-time telemetry (`metrics.jsonl`, `metrics.csv`, `live_status.json`, `train.log`). Committed periodically to cloud storage during training.

---

## 2. Launching Training

### 2.1 Detached Mode (Recommended for Production)
The `--detach` flag launches the job in the cloud via Modal `.spawn()` and immediately terminates the local client process with code 0. Once spawned, the cloud container runs completely independently:
- **Resilience**: Immune to local Wi-Fi disconnects, VPN timeouts, or putting your laptop to sleep.
- **Monitoring**: Progress is tracked via the local dashboard (`http://localhost:7860`) or the Modal web console.

Launch command:
```bash
modal run scripts/modal_train.py \
    --gpu B300 \
    --detach \
    --target-step 28991 \
    --mbs 128 \
    --global-batch-tokens 131072 \
    --seq-len 1024 \
    --resume latest
```

### 2.2 Synchronous Mode (Interactive Debugging)
Omitting `--detach` runs the job synchronously, streaming container stdout directly to your local terminal:
```bash
modal run scripts/modal_train.py \
    --gpu B300 \
    --target-step 100 \
    --mbs 128 \
    --global-batch-tokens 131072 \
    --seq-len 1024
```
*Note: In synchronous mode, closing your terminal or losing network connection for more than 15 minutes will terminate the remote container.*

---

## 3. Resumption & Automated Telemetry Rollback

If a training session is interrupted (e.g. cloud spot preemption or manual cancellation), resuming with `--resume latest` guarantees strict state and telemetry consistency:

1. **Model & Optimizer Restoration**:
   `src/train.py` loads `/checkpoints/latest.pt` to CPU, restores model weights and optimizer moments (Muon momentum and AdamW states), and sets `start_step = ckpt["global_step"]`.

2. **Automated Telemetry Rollback**:
   Because training continues past checkpoint intervals until an interruption occurs, uncommitted log entries may exist on disk. `rollback_telemetry()` automatically:
   - Filters `metrics.jsonl` to records where `step <= start_step`.
   - Filters `metrics.csv` to rows where `step <= start_step`.
   - Truncates `train.log` right before the first line exceeding `start_step`.
   - Synchronizes `live_status.json` to the exact state of `start_step`.

This guarantees that resumption never creates duplicate steps or non-monotonic time-series curves on the dashboard.

---

## 4. Real-Time Cloud Monitoring

Telemetry streams from the remote container to your local machine or web browser through two streamlined options:

### Option A: Unified Local Monitor (`make monitor-cloud`)
Runs both the in-process Python SDK volume sync and the local web dashboard in a single terminal command:
```bash
make monitor-cloud
# Or:
python monitor/web_server.py --port 7860 --checkpoint-dir checkpoints --cloud
```
Access the dashboard at `http://localhost:7860`. The server automatically tails `pulsar-checkpoints` in an in-process background thread without spawning external CLI subprocesses.

### Option B: Serverless Cloud Monitor (`modal serve`)
Run the monitor directly in the cloud attached to the persistent volume with zero local daemon overhead and an authenticated public URL:
```bash
modal serve scripts/modal_monitor.py
# Or permanent deployment:
modal deploy scripts/modal_monitor.py
```

---

## 5. Inspecting & Fetching Checkpoints

### Inspect Remote Volume Contents
```bash
modal volume ls pulsar-checkpoints /
```

### Download a Checkpoint Locally
To pull a checkpoint (e.g. `best.pt` or `latest.pt`) from the cloud volume:
```bash
modal volume get --force pulsar-checkpoints /latest.pt checkpoints/latest.pt
```

### Standalone Telemetry Rollback
To manually roll back local or volume telemetry back to a specific step without running training:
```bash
python -m src.train --rollback-telemetry-to-step 1500 --checkpoint-dir checkpoints
```

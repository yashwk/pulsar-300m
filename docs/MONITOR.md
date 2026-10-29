# Pulsar-300M Real-Time Training Monitor Specification

## 1. Overview & Architecture

The Pulsar Training Monitor (`monitor/`, served live at `http://localhost:7860`) provides real-time, zero-overhead operational visibility into active pretraining runs.

It decouples training throughput from observation overhead:
1. **Comparable Learning Metrics**: Separates the auxiliary weighted training loss (`train_loss`, which includes z-loss and intermediate pulse cross-entropies) from the **final-pulse training cross-entropy** (`pulse_losses[-1]`), plotting it directly against validation final cross-entropy (`val_ce_final`).
2. **Clear Information Hierarchy**: Features an always-visible **Health Strip** (`OK` / `WATCH` / `ALERT`), followed by focused uPlot charts and drill-down tabs.
3. **Automated Server-Side Health Checks**: 9 automated health checks evaluated against configurable thresholds in `monitor/thresholds.json`.
4. **Scalable Incremental Tail-Reading**: An incremental tail-reader (`IncrementalMetricsReader`) tracks byte offsets and parses appended telemetry without re-reading entire multi-megabyte log files. Automatically detects file shrinkage or truncation (e.g. following checkpoint rollback) and resets safely.
5. **Zero CDN Dependencies**: All styling and charting libraries (`uPlot.iife.min.js`, `uPlot.min.css`) are strictly vendored locally for 100% offline reliability.
6. **First-Class Diagnostics**: Exposes recurrent pulse gain ($\Delta L = L_{p1} - L_{p3}$), MoE token drops, router bias drift, and hidden-state contraction ($\|\Delta h\|$).

---

## 2. Server Architecture & Telemetry Dataflow

```mermaid
flowchart TD
    subgraph Cloud [Modal GPU Container]
        TRAIN[src/train.py]
        VOL_DIR["/checkpoints (Modal Volume: pulsar-checkpoints)"]
        COMMIT["Periodic Volume Committer (Every 10s)"]
        TRAIN -->|Writes| VOL_DIR
        VOL_DIR --> COMMIT
    end

    subgraph SyncDaemon [Local Workstation Sync]
        SYNC["scripts/sync_cloud_monitor.py"]
        LOCAL_DIR["checkpoints/"]
        COMMIT -->|modal volume get --force| SYNC
        SYNC --> LOCAL_DIR
    end

    subgraph WebServer [monitor/web_server.py]
        IR["IncrementalMetricsReader (metrics.jsonl)"]
        LIVE["live_status.json Reader"]
        HEALTH["evaluate_health (thresholds.json)"]
        LOCAL_DIR --> IR
        LOCAL_DIR --> LIVE
        IR --> HEALTH
        LIVE --> HEALTH
    end

    subgraph BrowserUI [http://localhost:7860]
        INDEX["index.html (Dark Mode UI)"]
        CHARTS["Synchronized uPlot Charts"]
        CANARIES["Live Canary Generation Cards"]
        WebServer -->|REST API / SSE| INDEX
        INDEX --> CHARTS
        INDEX --> CANARIES
    end
```

---

## 3. Telemetry Schema & Files

All telemetry files reside in the single flat `checkpoints/` directory:

| Filename | Format | Purpose & Update Frequency |
| :--- | :--- | :--- |
| **`live_status.json`** | JSON | Atomic snapshot of instantaneous training state. Updated every step. Read by `/api/status`. |
| **`metrics.jsonl`** | JSON Lines | Authoritative step-by-step history records. Appended every step. Read by `/api/history`. |
| **`metrics.csv`** | CSV | Tabular metrics export for external visualization or spreadsheet import. |
| **`train.log`** | Plain Text | Mirrored stdout and stderr stream. |
| **`run_config.json`** | JSON | Authoritative session parameters and hyperparameters. |
| **`config.json`** | JSON | Model architecture configuration (`PulsarConfig`). |

---

## 4. Endpoints & REST API

The lightweight HTTP server (`monitor/web_server.py`) serves the following endpoints:

- `GET /`: Serves the primary dark-mode dashboard (`monitor/index.html`).
- `GET /api/status`: Returns the latest instantaneous status snapshot from `live_status.json`.
- `GET /api/history?last_n=2000`: Returns downsampled time-series records for loss curves and telemetry charts.
- `GET /api/health`: Evaluates the 9 health checks against `thresholds.json` and returns current status flags.
- `GET /api/canaries`: Returns qualitative model completions on standard reasoning, math, coding, and tool-calling probes.
- `GET /api/checkpoints`: Lists all saved `.pt` files, identifying `latest.pt`, `best.pt`, and milestone checkpoints.
- `GET /api/stream`: Server-Sent Events (SSE) delivering live updates as steps complete.

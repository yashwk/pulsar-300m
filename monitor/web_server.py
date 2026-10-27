"""Pulsar Real-Time Training Web Monitor Server.

Runs a lightweight HTTP server serving the real-time dark-mode training dashboard
and REST endpoints querying metrics from checkpoints/metrics.jsonl and checkpoints/live_status.json.

Endpoints:
  - GET /             -> Serves monitor/index.html
  - GET /api/status   -> Returns latest instantaneous training snapshot + canaries
  - GET /api/history  -> Returns historical series data for live charts (e.g. ?last_n=300)
"""

import argparse
import csv
import glob
import json
import mimetypes
import os
import re
import sys
import threading
import time
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from monitor.v2_api import (
    IncrementalMetricsReader,
    downsample_records,
    evaluate_health,
    load_thresholds,
    get_timeline_events,
    list_checkpoints,
    load_notes,
    add_note,
)

MONITOR_DIR = os.path.dirname(os.path.abspath(__file__))
V2_STATIC_DIR = os.path.join(MONITOR_DIR, "v2")
THRESHOLDS_FILE = os.path.join(MONITOR_DIR, "thresholds.json")
NOTES_FILE = os.path.join(MONITOR_DIR, "notes.jsonl")

CHECKPOINT_DIR = "checkpoints"
LIVE_STATUS_FILE = os.path.join(CHECKPOINT_DIR, "live_status.json")
METRICS_JSONL_FILE = os.path.join(CHECKPOINT_DIR, "metrics.jsonl")
STATIC_HTML_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
_v2_incremental_reader = IncrementalMetricsReader(METRICS_JSONL_FILE)

# Fallback used only when checkpoints/config.json is missing AND the trainer has not
# written model_config into live_status.json. Values are PulsarConfig() defaults
# (299,675,712 total / 146,800,640 active), not stale guesses.
DEFAULT_MODEL_SUMMARY = {
    "model_name": "Pulsar",
    "total_params": 299675712,
    "active_params": 146800640,
    "non_embed_active_params": 96468992,
    "n_core": 8,
    "num_pulses": 3,
    "effective_layers": 24,
    "d_model": 1024,
    "n_heads": 16,
    "n_kv_heads": 4,
    "head_dim": 64,
    "d_expert": 1024,
    "n_routed_experts": 8,
    "n_shared_experts": 1,
    "top_k": 2,
    "active_ffn_dim": 3072,
    "vocab_size": 49152,
    "max_seq_len": 32768,
}

_summary_cache = {"key": None, "value": None}
_summary_lock = threading.Lock()
_history_cache = {"key": None, "records": []}
_history_lock = threading.Lock()


def get_model_summary_cached():
    """Build the model summary from checkpoints/config.json at most once per file change."""
    config_json = os.path.join(CHECKPOINT_DIR, "config.json")
    try:
        st = os.stat(config_json)
        key = (st.st_mtime_ns, st.st_size)
    except OSError:
        return dict(DEFAULT_MODEL_SUMMARY)
    with _summary_lock:
        if _summary_cache["key"] == key and _summary_cache["value"] is not None:
            return _summary_cache["value"]
    summary = None
    try:
        with open(config_json, "r", encoding="utf-8") as f:
            cfg_dict = json.load(f)
        from src.model import PulsarConfig, PulsarForCausalLM
        cfg = PulsarConfig.from_dict(cfg_dict)
        summary = PulsarForCausalLM(cfg).get_model_summary(cfg.default_pulses)
    except Exception:
        summary = None
    if summary is None:
        summary = dict(DEFAULT_MODEL_SUMMARY)
    with _summary_lock:
        _summary_cache["key"], _summary_cache["value"] = key, summary
    return summary


def flop_scale_per_tok():
    """TFLOP/s per (tok/s): aligned with src/train.py baseline (2.05 GFLOPs/tok for K=3)."""
    mc = None
    try:
        with open(LIVE_STATUS_FILE, "r", encoding="utf-8") as f:
            mc = (json.load(f) or {}).get("model_config")
    except Exception:
        mc = None
    if not mc:
        mc = get_model_summary_cached()
    pulses = mc.get("num_pulses") or 3
    return 2.05e-3 * (pulses / 3.0)


RUN_CONFIG_FILE = os.path.join(CHECKPOINT_DIR, "run_config.json")
_run_config_cache = {"key": None, "value": None}
_run_config_lock = threading.Lock()


def get_run_config():
    """Read authoritative session configuration (SSOT written by train.py)."""
    if not os.path.exists(RUN_CONFIG_FILE):
        return {}
    try:
        st = os.stat(RUN_CONFIG_FILE)
        key = (st.st_mtime_ns, st.st_size)
    except OSError:
        return {}
    with _run_config_lock:
        if _run_config_cache["key"] == key and _run_config_cache["value"] is not None:
            return dict(_run_config_cache["value"])
    try:
        with open(RUN_CONFIG_FILE, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        cfg = {}
    with _run_config_lock:
        _run_config_cache["key"], _run_config_cache["value"] = key, cfg
    return dict(cfg)


def detect_target_and_curriculum_steps():
    # 1. Authoritative run_config.json (Single Source of Truth)
    rc = get_run_config()
    target_from_rc = rc.get("target_steps") or rc.get("target_step")
    curriculum_from_rc = rc.get("curriculum_steps") or rc.get("max_steps")
    if target_from_rc:
        return int(target_from_rc), int(curriculum_from_rc) if curriculum_from_rc else None

    # 2. Live status file
    if os.path.exists(LIVE_STATUS_FILE):
        try:
            with open(LIVE_STATUS_FILE, "r", encoding="utf-8") as f:
                d = json.load(f)
                tgt = d.get("target_steps") or d.get("target_step")
                cur = d.get("curriculum_steps") or d.get("max_steps")
                if tgt:
                    return int(tgt), int(cur) if cur else None
        except Exception:
            pass

    # 3. Log file scan without arbitrary byte limits
    log_file = os.path.join(CHECKPOINT_DIR, "train.log")
    target_step = None
    curriculum_steps = None
    if os.path.exists(log_file):
        try:
            with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                m = re.findall(r"Starting training: Steps \d+ -> (\d+)(?:\s+\(Curriculum Horizon:\s*(\d+)\))?", content)
                if m:
                    last_m = m[-1]
                    target_step = int(last_m[0])
                    if len(last_m) > 1 and last_m[1]:
                        curriculum_steps = int(last_m[1])
        except Exception:
            pass
    return target_step, curriculum_steps


def detect_max_steps():
    target_step, curriculum_steps = detect_target_and_curriculum_steps()
    return target_step or curriculum_steps or 100


def load_csv_grads():
    csv_file = os.path.join(CHECKPOINT_DIR, "metrics.csv")
    grads = {}
    if os.path.exists(csv_file):
        try:
            with open(csv_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        s = int(row["step"])
                        grads[s] = {
                            "grad_norm_total": float(row.get("grad_total", 0.0)),
                            "grad_norm_muon": float(row.get("grad_muon", 0.0)),
                            "grad_norm_adam": float(row.get("grad_adam", 0.0)),
                        }
                    except Exception:
                        pass
        except Exception:
            pass
    return grads


def load_history_records():
    """Parse metrics.jsonl into records, cached by (mtime, size); annotate estimated TFLOP/s."""
    if not os.path.exists(METRICS_JSONL_FILE):
        return []
    try:
        st = os.stat(METRICS_JSONL_FILE)
        key = (st.st_mtime_ns, st.st_size)
    except OSError:
        return []
    with _history_lock:
        if _history_cache["key"] == key:
            return _history_cache["records"]

    scale = flop_scale_per_tok()
    records = []
    with open(METRICS_JSONL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            # val_pulse_* are only meaningful on actual validation evaluation steps
            if item.get("val_loss") is None:
                item["val_pulse_losses"] = None
                item["val_pulse_ppls"] = None
            tps = item.get("tok_per_sec")
            if tps is not None and scale:
                item["tflops"] = round(tps * scale, 2)
            records.append(item)

    with _history_lock:
        _history_cache["key"], _history_cache["records"] = key, records
    return records


def infer_canary_meta(c):
    tag = c.get("tag")
    category = c.get("category")
    if tag and category:
        return tag, category
    p = c.get("prompt", "")
    p_lower = p.lower()
    if "select" in p_lower or "orders" in p_lower:
        return "SQL", "SQL"
    elif "def " in p_lower or "quicksort" in p_lower or "fibonacci" in p_lower:
        return "Python", "Python"
    elif "tool_call" in p_lower or "query_database" in p_lower or "{" in p:
        return "Tool Calling / JSON", "JSON Tool"
    elif "question:" in p_lower or "train travels" in p_lower or "speed" in p_lower:
        return "Math / Reasoning", "Math"
    elif "bash" in p_lower or "download" in p_lower:
        return "Bash / Linux CLI", "CLI"
    elif "mitochondria" in p_lower:
        return "STEM / World Knowledge", "STEM"
    elif "france" in p_lower or "capital" in p_lower:
        return "General Knowledge", "Knowledge"
    elif "recurrent" in p_lower or "neural network" in p_lower:
        return "Deep Learning / ML", "Knowledge"
    elif "law of nature" in p_lower:
        return "Physics / Science", "Knowledge"
    return "General Knowledge", "Knowledge"


def check_log_thermal_cooling():
    log_file = os.path.join(CHECKPOINT_DIR, "train.log")
    if not os.path.exists(log_file):
        return None
    try:
        with open(log_file, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            seek_pos = max(0, size - 8192)
            f.seek(seek_pos)
            content = f.read().decode("utf-8", errors="ignore")
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        last_pause_idx = -1
        last_resume_idx = -1
        pause_line = ""
        for i, line in enumerate(lines):
            if "[Hardware Protection] Pausing for" in line:
                last_pause_idx = i
                pause_line = line
            elif "[Hardware Protection] Cooldown complete" in line:
                last_resume_idx = i
        if last_pause_idx > last_resume_idx:
            mtime = os.path.getmtime(log_file)
            elapsed = time.time() - mtime
            total = 45
            match = re.search(r"Pausing for (\d+)s", pause_line)
            if match:
                total = int(match.group(1))
            remaining = max(1, total - int(elapsed))
            if elapsed < (total + 5):
                return {
                    "status": "cooling",
                    "cooling_remaining_s": remaining,
                    "cooling_total_s": total,
                }
    except Exception:
        pass
    return None


_RUN_CONFIG_CACHE = {"key": None, "value": {}}
# Set from the CLI for runs whose trainer predates the metrics.jsonl fields.
_OVERRIDES: dict[str, int] = {}


def _run_config_value(name, default):
    """Read a run-level parameter from the newest metrics.jsonl record.

    The live status file only contains per-step values, so run-level parameters are
    taken from the JSONL the trainer writes every step. Cached by file mtime+size to
    keep the 10 s poll cheap.
    """
    if name in _OVERRIDES:
        return _OVERRIDES[name]

    path = os.path.join(CHECKPOINT_DIR, "metrics.jsonl")
    try:
        st = os.stat(path)
        key = (st.st_mtime_ns, st.st_size)
    except OSError:
        return default

    if _RUN_CONFIG_CACHE["key"] != key:
        value = {}
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                if st.st_size > 65536:
                    f.seek(st.st_size - 65536)
                    f.readline()  # discard partial line
                lines = f.readlines()
            for line in reversed(lines[-50:]):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get(name) is not None:
                    value = {name: rec[name]}
                    break
        except (OSError, json.JSONDecodeError):
            value = {}
        _RUN_CONFIG_CACHE["key"] = key
        _RUN_CONFIG_CACHE["value"] = value
    return _RUN_CONFIG_CACHE["value"].get(name, default)


def detect_active_train_shards():
    # 1. Authoritative run_config.json
    rc = get_run_config()
    if rc.get("train_shards"):
        return [os.path.basename(s) for s in rc["train_shards"]]

    # 2. Inspect active running trainer process via /proc
    for cmd_path in glob.glob("/proc/[0-9]*/cmdline"):
        try:
            with open(cmd_path, "rb") as f:
                parts = f.read().split(b"\x00")
                cmd = [p.decode("utf-8", errors="ignore") for p in parts if p]
                if any("src/train.py" in arg for arg in cmd):
                    for i, arg in enumerate(cmd):
                        if arg == "--train-bin" and i + 1 < len(cmd):
                            return [os.path.basename(s.strip()) for s in cmd[i + 1].split(",") if s.strip()]
                        elif arg.startswith("--train-bin="):
                            return [os.path.basename(s.strip()) for s in arg.split("=", 1)[1].split(",") if s.strip()]
        except Exception:
            pass
    return None


def enrich_status_payload(data):
    cooling_info = check_log_thermal_cooling()
    if cooling_info:
        data["status"] = cooling_info["status"]
        data["cooling_remaining_s"] = cooling_info["cooling_remaining_s"]
        data["cooling_total_s"] = cooling_info["cooling_total_s"]
    elif "status" not in data or data["status"] == "cooling":
        data["status"] = "training"

    # Merge authoritative run configuration (SSOT)
    rc = get_run_config()
    for k in (
        "target_steps",
        "target_step",
        "max_steps",
        "curriculum_steps",
        "warmup_steps",
        "cooldown_steps",
        "global_batch_tokens",
        "seq_len",
        "num_pulses",
        "lr_schedule",
    ):
        if rc.get(k) is not None and data.get(k) is None:
            data[k] = rc[k]

    target_step, curriculum_steps = detect_target_and_curriculum_steps()
    if data.get("target_steps") is None:
        if data.get("target_step") is not None:
            data["target_steps"] = data["target_step"]
        elif target_step is not None:
            data["target_steps"] = target_step

    if data.get("curriculum_steps") is None:
        if curriculum_steps is not None:
            data["curriculum_steps"] = curriculum_steps
        elif data.get("max_steps") is not None:
            data["curriculum_steps"] = data["max_steps"]

    max_steps = data.get("curriculum_steps") or data.get("max_steps") or data.get("target_steps") or 0
    active_target = data.get("target_steps") or target_step or max_steps

    step = data.get("step", 0)
    tok_per_sec = float(data.get("tok_per_sec") or 0.0)

    grads = load_csv_grads()
    if step in grads:
        data["grad_norm_total"] = grads[step]["grad_norm_total"]
        data["grad_norm_muon"] = grads[step]["grad_norm_muon"]
        data["grad_norm_adam"] = grads[step]["grad_norm_adam"]
    elif grads:
        latest_s = max(grads.keys())
        data["grad_norm_total"] = grads[latest_s]["grad_norm_total"]
        data["grad_norm_muon"] = grads[latest_s]["grad_norm_muon"]
        data["grad_norm_adam"] = grads[latest_s]["grad_norm_adam"]
    else:
        data["grad_norm_total"] = data.get("grad_norm_total", 0.0)
        data["grad_norm_muon"] = data.get("grad_norm_muon", 0.0)
        data["grad_norm_adam"] = data.get("grad_norm_adam", 0.0)

    batch_tokens = float(data.get("global_batch_tokens") or 131072.0)
    rem = max(0, active_target - step)
    step_time = (batch_tokens / max(1.0, tok_per_sec)) if tok_per_sec > 0 else 0.0

    if step_time > 0 and rem > 0:
        eta_sec = rem * step_time
        m, s = divmod(int(eta_sec), 60)
        h, m = divmod(m, 60)
        data["eta_str"] = f"{h}h {m:02d}m" if h > 0 else f"{m:02d}m {s:02d}s"
    elif rem == 0:
        data["eta_str"] = "00m 00s"
    else:
        data["eta_str"] = "--:--"

    if step_time > 0:
        elapsed_sec = step * step_time
        em, es = divmod(int(elapsed_sec), 60)
        eh, em = divmod(em, 60)
        data["elapsed_str"] = f"{eh}h {em:02d}m" if eh > 0 else f"{em:02d}m {es:02d}s"
    else:
        data["elapsed_str"] = "--"

    warmup_steps = data.get("warmup_steps")
    if warmup_steps is None:
        warmup_steps = _run_config_value("warmup_steps", 0)
    cooldown_steps = data.get("cooldown_steps")
    if cooldown_steps is None:
        cooldown_steps = _run_config_value("cooldown_steps", 0)
    data["max_steps"] = max_steps
    data["active_target"] = active_target
    data["warmup_steps"] = warmup_steps
    data["cooldown_steps"] = cooldown_steps

    if data.get("status") == "completed" or (active_target > 0 and step >= active_target):
        data["phase"] = "Target Completed"
    elif step <= warmup_steps and warmup_steps > 0:
        data["phase"] = f"Warmup ({step}/{warmup_steps})"
    elif cooldown_steps > 0 and step >= (max_steps - cooldown_steps):
        data["phase"] = "Cooldown / Decay"
    else:
        data["phase"] = "Stable Phase"

    # Dynamic active train shards
    active_shards = detect_active_train_shards()
    if active_shards:
        data["train_shards"] = active_shards
        data["num_train_shards"] = len(active_shards)
        if not data.get("train_dataset_tokens"):
            data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
            data["train_dataset_tokens"] = sum(
                os.path.getsize(os.path.join(data_dir, s)) // 2
                for s in active_shards
                if os.path.exists(os.path.join(data_dir, s))
            )
    elif not data.get("num_train_shards") or not data.get("train_dataset_tokens"):
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
        disk_shards = sorted(glob.glob(os.path.join(data_dir, "train*.bin")))
        disk_shards = [s for s in disk_shards if not s.endswith(".tmp")]
        data["train_shards"] = [os.path.basename(s) for s in disk_shards]
        data["num_train_shards"] = len(disk_shards) if disk_shards else 1
        data["train_dataset_tokens"] = sum(os.path.getsize(p) // 2 for p in disk_shards) if disk_shards else 0

    # Dynamic model configuration summary (cached; never rebuilds the model per request)
    if not data.get("model_config"):
        data["model_config"] = get_model_summary_cached()

    if "device_name" not in data:
        data["device_name"] = "NVIDIA GeForce RTX 4060 Laptop GPU"

    # Enrich canaries with tags & categories
    if "canaries" in data and isinstance(data["canaries"], list):
        for c in data["canaries"]:
            t, cat = infer_canary_meta(c)
            c["tag"] = t
            c["category"] = cat

    return data


class PulsarMonitorHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        # Enable CORS and disable browser caching for live streaming data
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path in ("/", "/index.html", "/v2", "/v2/", "/v2/index.html"):
            self.serve_v2_dashboard()
        elif path in ("/v1", "/v1/", "/v1/index.html", "/legacy", "/legacy/"):
            self.serve_dashboard()
        elif path.startswith("/v2/"):
            self.serve_v2_static(path[4:])
        elif path in ("/style.css", "/app.js", "/charts.js", "/health.js") or path.startswith("/vendor/"):
            self.serve_v2_static(path)
        elif path == "/favicon.ico":
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
        elif path == "/api/status":
            self.serve_status()
        elif path == "/api/health":
            self.serve_health()
        elif path == "/api/events":
            self.serve_events()
        elif path == "/api/checkpoints":
            self.serve_checkpoints()
        elif path == "/api/notes":
            self.serve_notes()
        elif path == "/api/stream":
            self.serve_stream()
        elif path == "/api/canaries":
            self.serve_canaries()
        elif path == "/api/history":
            last_n = None
            max_points = 2000
            since_step = None
            fields = None
            if "last_n" in query:
                val = query["last_n"][0].lower()
                if val not in ("0", "all"):
                    try:
                        last_n = int(val)
                    except ValueError:
                        pass
            if "max_points" in query:
                try:
                    max_points = int(query["max_points"][0])
                except ValueError:
                    pass
            if "since_step" in query:
                try:
                    since_step = int(query["since_step"][0])
                except ValueError:
                    pass
            if "fields" in query:
                fields = [f.strip() for f in query["fields"][0].split(",") if f.strip()]
            self.serve_history(last_n, max_points=max_points, since_step=since_step, fields=fields)
        else:
            self.send_error(HTTPStatus.NOT_FOUND, f"File not found: {path}")

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/api/notes":
            self.handle_post_notes()
        else:
            self.send_error(HTTPStatus.NOT_FOUND, f"Endpoint not found: {path}")

    def serve_dashboard(self):
        if not os.path.exists(STATIC_HTML_FILE):
            self.send_error(HTTPStatus.NOT_FOUND, "index.html not found")
            return
        with open(STATIC_HTML_FILE, "rb") as f:
            content = f.read()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def serve_status(self):
        if not os.path.exists(LIVE_STATUS_FILE):
            payload = json.dumps({
                "status": "idle",
                "message": "Training has not started yet or checkpoints/live_status.json not found."
            }).encode("utf-8")
        else:
            try:
                with open(LIVE_STATUS_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if "status" not in data:
                    data["status"] = "running"
                data = enrich_status_payload(data)
                payload = json.dumps(data).encode("utf-8")
            except Exception as e:
                payload = json.dumps({"status": "error", "error": str(e)}).encode("utf-8")

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def serve_v2_dashboard(self):
        v2_index = os.path.join(V2_STATIC_DIR, "index.html")
        if not os.path.exists(v2_index):
            self.send_error(HTTPStatus.NOT_FOUND, "monitor/v2/index.html not found")
            return
        with open(v2_index, "rb") as f:
            content = f.read()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def serve_v2_static(self, rel_path: str):
        rel_path = rel_path.lstrip("/\\")
        full_path = os.path.abspath(os.path.join(V2_STATIC_DIR, rel_path))
        if not full_path.startswith(os.path.abspath(V2_STATIC_DIR)):
            self.send_error(HTTPStatus.FORBIDDEN, "Access denied")
            return
        if not os.path.exists(full_path) or os.path.isdir(full_path):
            self.send_error(HTTPStatus.NOT_FOUND, f"File not found: {rel_path}")
            return

        ctype, _ = mimetypes.guess_type(full_path)
        if not ctype:
            if full_path.endswith(".js"):
                ctype = "application/javascript"
            elif full_path.endswith(".css"):
                ctype = "text/css"
            else:
                ctype = "application/octet-stream"

        with open(full_path, "rb") as f:
            content = f.read()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def serve_health(self):
        th = load_thresholds(THRESHOLDS_FILE)
        status_data = {}
        if os.path.exists(LIVE_STATUS_FILE):
            try:
                with open(LIVE_STATUS_FILE, "r", encoding="utf-8") as f:
                    status_data = json.load(f)
                status_data = enrich_status_payload(status_data)
            except Exception:
                pass
        records = _v2_incremental_reader.read_records()
        health = evaluate_health(status_data, records, th)
        payload = json.dumps(health).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def serve_events(self):
        records = _v2_incremental_reader.read_records()
        events = get_timeline_events(records, NOTES_FILE)
        payload = json.dumps({"count": len(events), "events": events}).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def serve_checkpoints(self):
        ckpts = list_checkpoints(CHECKPOINT_DIR)
        payload = json.dumps({"count": len(ckpts), "checkpoints": ckpts}).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def serve_notes(self):
        notes = load_notes(NOTES_FILE)
        payload = json.dumps({"count": len(notes), "notes": notes}).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def handle_post_notes(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8")
            data = json.loads(body)
            step = int(data.get("step", 0))
            text = str(data.get("text", "")).strip()
            author = str(data.get("author", "User")).strip() or "User"
            tag = str(data.get("tag", "general")).strip() or "general"
            if not text:
                self.send_error(HTTPStatus.BAD_REQUEST, "Missing note text")
                return
            new_note = add_note(NOTES_FILE, step, text, author, tag)
            payload = json.dumps({"status": "success", "note": new_note}).encode("utf-8")
            self.send_response(HTTPStatus.CREATED)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except Exception as e:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(e))

    def serve_stream(self):
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        self.wfile.write(b": connected\n\n")
        self.wfile.flush()

        for _ in range(120):
            try:
                status_data = {}
                if os.path.exists(LIVE_STATUS_FILE):
                    try:
                        with open(LIVE_STATUS_FILE, "r", encoding="utf-8") as f:
                            status_data = json.load(f)
                        status_data = enrich_status_payload(status_data)
                    except Exception:
                        pass
                event_line = f"data: {json.dumps(status_data)}\n\n".encode("utf-8")
                self.wfile.write(event_line)
                self.wfile.flush()
                time.sleep(1.5)
            except (BrokenPipeError, ConnectionResetError):
                break
            except Exception:
                break

    def serve_history(
        self,
        last_n: int | None = None,
        max_points: int = 2000,
        since_step: int | None = None,
        fields: list[str] | None = None,
    ):
        records = _v2_incremental_reader.read_records()
        grads = load_csv_grads()

        for item in records:
            s = item.get("step")
            if s in grads:
                item["grad_norm_total"] = grads[s]["grad_norm_total"]
                item["grad_norm_muon"] = grads[s]["grad_norm_muon"]
                item["grad_norm_adam"] = grads[s]["grad_norm_adam"]

        if last_n is not None and last_n > 0 and len(records) > last_n:
            records = records[-last_n:]

        sampled = downsample_records(records, max_points=max_points, fields=fields, since_step=since_step)
        payload = json.dumps({"count": len(sampled), "history": sampled}).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def serve_canaries(self):
        milestones = []
        canary_hist_file = os.path.join(CHECKPOINT_DIR, "canary_history.json")
        if os.path.exists(canary_hist_file):
            try:
                with open(canary_hist_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    milestones = data.get("milestones", [])
            except Exception:
                pass

        # Also extract true evaluation milestones directly from metrics.jsonl
        if os.path.exists(METRICS_JSONL_FILE):
            try:
                with open(METRICS_JSONL_FILE, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                item = json.loads(line)
                                if item.get("val_loss") is not None and item.get("canaries"):
                                    s = item.get("step")
                                    if not any(m.get("step") == s for m in milestones):
                                        milestones.append({
                                            "step": s,
                                            "label": f"Step {s:,}",
                                            "val_loss": item.get("val_loss"),
                                            "val_ppl": item.get("val_ppl"),
                                            "canaries": item.get("canaries", []),
                                        })
                            except Exception:
                                continue
            except Exception:
                pass

        milestones.sort(key=lambda m: m.get("step", 0))

        # Enrich all canaries with category and tag metadata
        for m in milestones:
            canaries_list = m.get("canaries", [])
            if isinstance(canaries_list, list):
                for c in canaries_list:
                    tag, cat = infer_canary_meta(c)
                    c["tag"] = tag
                    c["category"] = cat

        payload = json.dumps({"count": len(milestones), "milestones": milestones}).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def run_server(port: int = 7860, host: str = "127.0.0.1"):
    server_address = (host, port)
    httpd = ThreadingHTTPServer(server_address, PulsarMonitorHandler)
    display_host = "localhost" if host == "0.0.0.0" else host
    print(f"=======================================================")
    print(f"Pulsar Real-Time Telemetry Monitor")
    print(f"Dashboard: http://{display_host}:{port}")
    print(f"REST Endpoints:")
    print(f"   • http://{display_host}:{port}/api/status")
    print(f"   • http://{display_host}:{port}/api/history?last_n=250")
    print(f"   • http://{display_host}:{port}/api/canaries")
    print(f"=======================================================")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[Pulsar Monitor] Shutting down web monitor.")
        httpd.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pulsar Web Dashboard Server")
    parser.add_argument("--port", type=int, default=7860, help="Server port (default: 7860)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Bind host (default: 127.0.0.1; use 0.0.0.0 to expose on the LAN)")
    parser.add_argument(
        "--warmup-steps", type=int, default=None,
        help="Override the warmup length. Normally read from the run's metrics.jsonl; "
             "needed only for a run started before the trainer began emitting it.",
    )
    parser.add_argument(
        "--cooldown-steps", type=int, default=None,
        help="Override the WSD cooldown length (see --warmup-steps).",
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default=None,
        help="Run directory to monitor (default: checkpoints). Point this at the trainer's "
             "--checkpoint-dir, e.g. checkpoints/baseline.",
    )
    args = parser.parse_args()

    if args.checkpoint_dir:
        CHECKPOINT_DIR = args.checkpoint_dir
        LIVE_STATUS_FILE = os.path.join(CHECKPOINT_DIR, "live_status.json")

    _OVERRIDES.clear()
    if args.warmup_steps is not None:
        _OVERRIDES["warmup_steps"] = args.warmup_steps
    if args.cooldown_steps is not None:
        _OVERRIDES["cooldown_steps"] = args.cooldown_steps

    METRICS_JSONL_FILE = os.path.join(CHECKPOINT_DIR, "metrics.jsonl")

    run_server(port=args.port, host=args.host)

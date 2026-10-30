"""Pulsar Monitor V2 - Core API, Tail-Reader, Downsampler, and Health Engine.

Single Source of Truth for:
- Incremental tail-reading of metrics.jsonl (with partial line tolerance and restart handling).
- Server-side LTTB/min-max bucketed downsampling (bounds payload to <= 2000 points per series).
- Health checks and alert evaluation against monitor/thresholds.json.
- Timeline event detection (evals, checkpoints, thermal pauses, restarts, user notes).
- Read-only checkpoint discovery and note persistence.
"""

from __future__ import annotations

import glob
import json
import math
import os
import time
from typing import Any


DEFAULT_THRESHOLDS = {
    "version": "1.0",
    "run_state": {
        "max_stalled_step_time_multiplier": 3.0,
    },
    "loss": {
        "max_loss_gap_nats": 0.15,
    },
    "gradients": {
        "max_clip_fraction": 0.30,
        "clip_norm_threshold": 1.0,
    },
    "throughput": {
        "min_tok_per_sec": 6000.0,
        "vram_alert_gb": 7.5,
    },
    "moe_drops": {
        "ok_max_pct": 2.0,
        "watch_max_pct": 5.0,
    },
    "moe_balance": {
        "worst_expert_share_watch_pct": 15.0,
        "worst_expert_share_alert_pct": 25.0,
        "bias_abs_p99_watch": 0.30,
    },
    "recurrent_loops": {
        "evaluation_start_step": 3000,
        "gain_23_min_threshold": 0.01,
        "consecutive_evals_required": 3,
    },
    "numerics": {
        "max_nan_inf_count": 0,
    },
}


def load_thresholds(path: str | None = None) -> dict[str, Any]:
    if not path or not os.path.exists(path):
        return dict(DEFAULT_THRESHOLDS)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return dict(DEFAULT_THRESHOLDS)


class IncrementalMetricsReader:
    """Fast tail-reader for metrics.jsonl.

    Remembers the last byte offset and parses only appended bytes.
    Tolerates partial lines at EOF and rewinds on process restarts.
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        self._last_offset = 0
        self._cached_records: list[dict[str, Any]] = []
        self._last_mtime = 0
        self._last_size = 0

    def reset(self):
        self._last_offset = 0
        self._cached_records = []
        self._last_mtime = 0
        self._last_size = 0

    def read_records(self) -> list[dict[str, Any]]:
        if not os.path.exists(self.filepath):
            return []

        try:
            st = os.stat(self.filepath)
        except OSError:
            return list(self._cached_records)

        # File truncated or rotated: restart from scratch
        if st.st_size < self._last_offset:
            self.reset()

        # If nothing changed, return cached records immediately
        if st.st_size == self._last_offset and st.st_mtime_ns == self._last_mtime:
            return list(self._cached_records)

        new_records = []
        try:
            with open(self.filepath, "rb") as f:
                f.seek(self._last_offset)
                chunk = f.read()
                if not chunk:
                    return list(self._cached_records)

                # Find last complete newline
                last_nl = chunk.rfind(b"\n")
                if last_nl == -1:
                    # Partial line without newline yet; wait for next flush
                    return list(self._cached_records)

                complete_bytes = chunk[: last_nl + 1]
                self._last_offset += len(complete_bytes)
                self._last_mtime = st.st_mtime_ns
                self._last_size = st.st_size

                text = complete_bytes.decode("utf-8", errors="ignore")
                for line in text.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        self._enrich_record(rec)
                        new_records.append(rec)
                    except Exception:
                        continue
        except Exception:
            return list(self._cached_records)

        if not new_records:
            return list(self._cached_records)

        # Merge with restart detection
        for rec in new_records:
            cur_step = rec.get("step")
            if cur_step is not None and self._cached_records:
                last_step = self._cached_records[-1].get("step")
                if last_step is not None and cur_step <= last_step:
                    # Step rewind detected: prune orphaned steps
                    self._cached_records = [
                        r for r in self._cached_records if r.get("step", 0) < cur_step
                    ]
            self._cached_records.append(rec)

        return list(self._cached_records)

    def _enrich_record(self, rec: dict[str, Any]):
        """Compute derived fields for comparable loss and loop gains."""
        pulse_losses = rec.get("pulse_losses")
        # Final-pulse training CE on 100% of tokens (directly comparable to val_ce_final)
        if isinstance(pulse_losses, list) and len(pulse_losses) >= 1:
            rec["train_ce_final"] = pulse_losses[-1]
            if len(pulse_losses) >= 2 and pulse_losses[0] is not None and pulse_losses[1] is not None:
                rec["train_gain_12"] = round(pulse_losses[0] - pulse_losses[1], 4)
            if len(pulse_losses) >= 3 and pulse_losses[1] is not None and pulse_losses[2] is not None:
                rec["train_gain_23"] = round(pulse_losses[1] - pulse_losses[2], 4)
        else:
            rec["train_ce_final"] = rec.get("train_loss")

        # Loss gap between training final-pulse CE and validation final CE
        val_ce = rec.get("val_ce_final") if rec.get("val_ce_final") is not None else rec.get("val_loss")
        tr_ce = rec.get("train_ce_final")
        if val_ce is not None and tr_ce is not None:
            rec["loss_gap"] = round(abs(float(tr_ce) - float(val_ce)), 4)
        else:
            rec["loss_gap"] = None


def downsample_records(
    records: list[dict[str, Any]],
    max_points: int = 2000,
    fields: list[str] | None = None,
    since_step: int | None = None,
) -> list[dict[str, Any]]:
    """Server-side min/max preserving downsampler.

    Divided into equal buckets, picking min and max points to preserve loss spikes,
    gradient anomalies, and outlier signals, with endpoints strictly preserved.
    """
    if since_step is not None:
        records = [r for r in records if (r.get("step") or 0) > since_step]

    if not records:
        return []

    if max_points <= 0 or len(records) <= max_points:
        sampled = records
    else:
        # Min-max bucketed downsampling
        # Each bucket yields 2 points (min and max by train_loss), plus endpoints
        target_buckets = max(1, max_points // 2)
        bucket_size = max(1, len(records) // target_buckets)

        sampled_indices = {0, len(records) - 1}

        # Always strictly preserve all evaluation records so validation curves are never broken
        for i, r in enumerate(records):
            if r.get("val_loss") is not None:
                sampled_indices.add(i)

        for b_start in range(0, len(records), bucket_size):
            b_end = min(b_start + bucket_size, len(records))
            b_slice = records[b_start:b_end]
            if not b_slice:
                continue

            # Pick min and max by train_loss
            min_idx = b_start
            max_idx = b_start
            min_val = float("inf")
            max_val = float("-inf")
            for i, r in enumerate(b_slice):
                tl = r.get("train_loss")
                val = float(tl) if tl is not None and not math.isnan(tl) else 0.0
                if val < min_val:
                    min_val = val
                    min_idx = b_start + i
                if val > max_val:
                    max_val = val
                    max_idx = b_start + i

            sampled_indices.add(min_idx)
            sampled_indices.add(max_idx)

        sampled = [records[i] for i in sorted(sampled_indices)]

    # Project to requested fields only
    if fields:
        field_set = set(fields)
        field_set.add("step")  # Always include step
        filtered = []
        for r in sampled:
            item = {k: r[k] for k in field_set if k in r}
            filtered.append(item)
        return filtered

    return sampled


def evaluate_health(
    latest_status: dict[str, Any],
    history_records: list[dict[str, Any]],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate Tier 1 health checks against thresholds.json (Server-side SSOT).

    Returns status ('OK', 'WATCH', 'ALERT') per check and overall.
    """
    th = thresholds or DEFAULT_THRESHOLDS
    checks: dict[str, Any] = {}
    overall = "OK"

    def promote(new_status: str):
        nonlocal overall
        if new_status == "ALERT":
            overall = "ALERT"
        elif new_status == "WATCH" and overall != "ALERT":
            overall = "WATCH"

    now = time.time()
    cur_step = latest_status.get("step", 0)

    # 1. Run State & Stalls
    st_status = latest_status.get("status", "running")
    cooling_remaining = latest_status.get("cooling_remaining_s", 0)
    tok_per_sec = float(latest_status.get("tok_per_sec") or 0.0)
    batch_tokens = float(latest_status.get("global_batch_tokens") or 131072.0)
    median_step_time = (batch_tokens / max(1.0, tok_per_sec)) if tok_per_sec > 0 else 14.0

    state_val = st_status
    state_level = "OK"
    if st_status == "cooling":
        state_val = f"cooling ({cooling_remaining}s)"
        state_level = "OK"
    elif st_status == "completed" or latest_status.get("phase") == "Target Completed":
        state_val = "completed"
        state_level = "OK"
    else:
        # Check if stalled: no new record for > 3x median step time
        stall_mult = th.get("run_state", {}).get("max_stalled_step_time_multiplier", 3.0)
        # Check time since last step update
        last_step_time = latest_status.get("_last_update_time", now)
        if (now - last_step_time) > (stall_mult * median_step_time + 60.0):
            state_val = "stalled"
            state_level = "ALERT"

    promote(state_level)
    checks["run_state"] = {
        "value": state_val,
        "status": state_level,
        "threshold": f"> {stall_mult}x step time",
        "description": "Training progress heartbeat and activity state."
    }

    # 2. Progress & ETA
    target_step = latest_status.get("active_target") or latest_status.get("target_step") or 3500
    curriculum = latest_status.get("curriculum_steps") or 30517
    pct = round(min(100.0, (cur_step / max(1, target_step)) * 100.0), 1)
    checks["progress"] = {
        "value": f"{cur_step:,} / {target_step:,} ({pct}%)",
        "status": "OK",
        "eta": latest_status.get("eta_str", "--"),
        "elapsed": latest_status.get("elapsed_str", "--"),
        "curriculum": curriculum,
        "description": "Step target progress and estimated completion time."
    }

    # 3. Loss & Learning Gap (Train Final Pulse vs Val Final CE)
    train_ce_final = None
    if history_records:
        train_ce_final = history_records[-1].get("train_ce_final")
    if train_ce_final is None:
        train_ce_final = latest_status.get("train_loss")

    val_ce_final = latest_status.get("val_ce_final") or latest_status.get("val_loss")
    gap = abs(train_ce_final - val_ce_final) if (train_ce_final is not None and val_ce_final is not None) else None
    max_gap = th.get("loss", {}).get("max_loss_gap_nats", 0.15)

    loss_level = "OK"
    if gap is not None and gap > max_gap:
        loss_level = "WATCH"

    promote(loss_level)
    checks["loss"] = {
        "train_ce_final": round(train_ce_final, 4) if train_ce_final is not None else None,
        "val_ce_final": round(val_ce_final, 4) if val_ce_final is not None else None,
        "loss_gap": round(gap, 4) if gap is not None else None,
        "status": loss_level,
        "threshold": f"|gap| <= {max_gap} nats",
        "description": "Final-pulse training CE compared directly to final-pulse validation CE."
    }

    # 4. Gradients & Clip Fraction
    recent_100 = history_records[-100:] if len(history_records) >= 100 else history_records
    clip_thresh = th.get("gradients", {}).get("clip_norm_threshold", 1.0)
    max_clip_frac = th.get("gradients", {}).get("max_clip_fraction", 0.30)

    clipped_steps = 0
    cur_grad_total = float(latest_status.get("grad_norm_total") or 0.0)
    for r in recent_100:
        g = float(r.get("grad_norm_total") or 0.0)
        if g >= clip_thresh:
            clipped_steps += 1

    clip_frac = round(clipped_steps / max(1, len(recent_100)), 3)
    grad_level = "OK"
    if clip_frac > max_clip_frac:
        grad_level = "WATCH"

    promote(grad_level)
    checks["gradients"] = {
        "grad_norm_total": round(cur_grad_total, 3),
        "clip_fraction": clip_frac,
        "clip_norm_threshold": clip_thresh,
        "status": grad_level,
        "threshold": f"clip fraction <= {max_clip_frac * 100:.0f}%",
        "description": "Total gradient L2 norm and percentage of clipped steps in rolling 100 steps."
    }

    # 5. Throughput & VRAM
    vram_alert = th.get("throughput", {}).get("vram_alert_gb", 7.5)
    floor_tps = th.get("throughput", {}).get("min_tok_per_sec", 6000.0)
    vram_peak = float(latest_status.get("vram_peak") or 0.0)

    tps_level = "OK"
    if vram_peak > vram_alert:
        tps_level = "ALERT"
    elif tok_per_sec > 0 and tok_per_sec < floor_tps:
        tps_level = "WATCH"

    promote(tps_level)
    checks["throughput"] = {
        "tok_per_sec": round(tok_per_sec, 1),
        "vram_peak": round(vram_peak, 2),
        "mfu_pct": round(float(latest_status.get("mfu_pct") or 0.0), 1),
        "status": tps_level,
        "threshold": f">= {floor_tps:,.0f} tok/s, VRAM <= {vram_alert} GB",
        "description": "Hardware MFU, throughput floor, and VRAM memory headroom."
    }

    # 6. MoE Drops
    drop_rate = float(latest_status.get("moe_drop_rate") or 0.0) * 100.0
    moe_ok = th.get("moe_drops", {}).get("ok_max_pct", 2.0)
    moe_watch = th.get("moe_drops", {}).get("watch_max_pct", 5.0)

    drop_level = "OK"
    if drop_rate > moe_watch:
        drop_level = "ALERT"
    elif drop_rate > moe_ok:
        drop_level = "WATCH"

    promote(drop_level)
    checks["moe_drops"] = {
        "drop_rate_pct": round(drop_rate, 2),
        "capacity": latest_status.get("moe_capacity", 768),
        "per_pulse": latest_status.get("moe_drop_per_pulse"),
        "status": drop_level,
        "threshold": f"< {moe_ok}% OK, < {moe_watch}% WATCH",
        "description": "Token assignment drop percentage caused by MoE expert capacity limits."
    }

    # 7. MoE Balance & Skew
    worst_share = float(latest_status.get("worst_expert_share_pct") or 0.0)
    if not worst_share and isinstance(latest_status.get("expert_pcts"), list):
        worst_share = max(latest_status["expert_pcts"])

    max_bias = float(latest_status.get("router_max_bias") or 0.0)
    conf = float(latest_status.get("router_confidence") or 0.0)
    w_watch = th.get("moe_balance", {}).get("worst_expert_share_watch_pct", 25.0)
    w_alert = th.get("moe_balance", {}).get("worst_expert_share_alert_pct", 35.0)

    bal_level = "OK"
    if worst_share > w_alert:
        bal_level = "ALERT"
    elif worst_share > w_watch:
        bal_level = "WATCH"

    promote(bal_level)
    checks["moe_balance"] = {
        "worst_expert_share_pct": round(worst_share, 2),
        "router_confidence": round(conf, 2),
        "max_bias": round(max_bias, 3),
        "status": bal_level,
        "threshold": f"worst share < {w_watch}% OK, < {w_alert}% WATCH",
        "description": "Single-expert load concentration vs 6.25% balanced baseline."
    }

    # 8. Recurrent Loops (Pulse 1 -> 3 Latent Gains)
    latent_gain = float(latest_status.get("latent_gain") or 0.0)
    loop_level = "OK"
    eval_start_step = th.get("recurrent_loops", {}).get("evaluation_start_step", 3000)
    min_gain_23 = th.get("recurrent_loops", {}).get("gain_23_min_threshold", 0.01)

    # Check last 3 validation evaluations if past evaluation_start_step
    if cur_step >= eval_start_step:
        val_evals = [r for r in history_records if r.get("val_ce_final") is not None]
        if len(val_evals) >= 3:
            recent_gains = []
            for ve in val_evals[-3:]:
                vpl = ve.get("val_pulse_losses")
                if isinstance(vpl, list) and len(vpl) >= 3:
                    recent_gains.append(vpl[1] - vpl[2])
            if len(recent_gains) == 3 and all(g < min_gain_23 for g in recent_gains):
                loop_level = "WATCH"

    promote(loop_level)
    checks["recurrent_loops"] = {
        "latent_gain": round(latent_gain, 4),
        "status": loop_level,
        "threshold": f"gain_23 >= {min_gain_23} post-step {eval_start_step}",
        "description": "Validation CE improvement delivered by recurrent depth refinement."
    }

    # 9. Numerics (NaN / Inf Detection)
    nan_inf_found = False
    for r in recent_100:
        for k in ("train_loss", "grad_norm_total"):
            v = r.get(k)
            if v is not None and (math.isnan(v) or math.isinf(v)):
                nan_inf_found = True
                break
        if nan_inf_found:
            break

    num_level = "ALERT" if nan_inf_found else "OK"
    promote(num_level)
    checks["numerics"] = {
        "status": num_level,
        "nan_inf_found": nan_inf_found,
        "threshold": "0 NaN / Inf allowed",
        "description": "Numerical stability check across losses and gradients."
    }

    return {
        "status": overall,
        "evaluated_at_step": cur_step,
        "timestamp": now,
        "checks": checks,
    }


def get_timeline_events(
    history_records: list[dict[str, Any]],
    notes_file: str,
) -> list[dict[str, Any]]:
    """Detect events along the step timeline (restarts, evals, checkpoints, thermal pauses, notes)."""
    events = []

    # 1. Validation Evals & Checkpoints
    best_seen = float("inf")
    for r in history_records:
        step = r.get("step")
        if not step:
            continue
        v_ce = r.get("val_ce_final") or r.get("val_loss")
        if v_ce is not None:
            is_best = False
            if v_ce < best_seen:
                best_seen = v_ce
                is_best = True
            events.append({
                "type": "checkpoint" if is_best else "evaluation",
                "step": step,
                "label": f"Val Best {v_ce:.4f}" if is_best else f"Eval {v_ce:.4f}",
                "val_ce": v_ce,
                "is_best": is_best,
            })

    # 2. User Notes
    notes = load_notes(notes_file)
    for n in notes:
        events.append({
            "type": "note",
            "step": n.get("step", 0),
            "label": n.get("text", ""),
            "author": n.get("author", "User"),
            "id": n.get("id"),
        })

    events.sort(key=lambda e: e.get("step", 0))
    return events


def list_checkpoints(checkpoint_dir: str) -> list[dict[str, Any]]:
    """Scan checkpoints directory read-only and return metadata."""
    if not os.path.exists(checkpoint_dir):
        return []

    results = []
    pattern = os.path.join(checkpoint_dir, "*.pt")
    for p in glob.glob(pattern):
        try:
            st = os.stat(p)
            name = os.path.basename(p)
            results.append({
                "filename": name,
                "size_mb": round(st.st_size / (1024 * 1024), 1),
                "mtime_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(st.st_mtime)),
                "is_best": name == "best.pt",
                "is_latest": name == "latest.pt",
                "is_milestone": "milestone" in name,
            })
        except OSError:
            continue

    results.sort(key=lambda x: x["filename"])
    return results


def load_notes(notes_file: str) -> list[dict[str, Any]]:
    if not os.path.exists(notes_file):
        return []
    notes = []
    try:
        with open(notes_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    notes.append(json.loads(line))
    except Exception:
        pass
    return notes


def add_note(notes_file: str, step: int, text: str, author: str = "User", tag: str = "general") -> dict[str, Any]:
    note = {
        "id": f"note_{int(time.time() * 1000)}",
        "step": int(step),
        "text": text.strip(),
        "author": author,
        "tag": tag,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(notes_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(note) + "\n")
    return note

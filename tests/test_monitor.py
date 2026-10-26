"""Contract and regression tests for the Pulsar telemetry monitor and web dashboard.

Verifies:
1. Strict Single Source of Truth (run_config.json -> live_status.json -> train.log)
2. Contract schema of /api/status payload consumed by index.html
3. Multi-run resumption target resolution
4. Phase transitions (Warmup -> Stable Phase -> Target Completed)
5. Zero reliance on brittle hardcoded step limits
"""

import json
import os
import shutil
import tempfile
import urllib.request
import pytest
from unittest.mock import patch

import monitor.web_server as ws


@pytest.fixture
def temp_checkpoint_dir(monkeypatch):
    """Provide an isolated temporary checkpoint directory for monitor tests."""
    temp_dir = tempfile.mkdtemp()
    monkeypatch.setattr(ws, "CHECKPOINT_DIR", temp_dir)
    monkeypatch.setattr(ws, "RUN_CONFIG_FILE", os.path.join(temp_dir, "run_config.json"))
    monkeypatch.setattr(ws, "LIVE_STATUS_FILE", os.path.join(temp_dir, "live_status.json"))
    monkeypatch.setattr(ws, "METRICS_JSONL_FILE", os.path.join(temp_dir, "metrics.jsonl"))
    ws._run_config_cache = {"key": None, "value": None}
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_run_config_is_primary_ssot(temp_checkpoint_dir):
    """Authoritative run_config.json takes precedence over all other logs."""
    run_cfg = {
        "start_step": 500,
        "target_step": 2000,
        "target_steps": 2000,
        "max_steps": 30517,
        "curriculum_steps": 30517,
        "warmup_steps": 300,
        "cooldown_steps": 0,
        "global_batch_tokens": 131072,
    }
    with open(ws.RUN_CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(run_cfg, f)

    target, curriculum = ws.detect_target_and_curriculum_steps()
    assert target == 2000
    assert curriculum == 30517


def test_live_status_fallback_when_no_run_config(temp_checkpoint_dir):
    """When run_config.json is absent, live_status.json provides target and horizon."""
    status_payload = {
        "step": 750,
        "target_steps": 1500,
        "max_steps": 25000,
        "tokens_seen": 98304000,
    }
    with open(ws.LIVE_STATUS_FILE, "w", encoding="utf-8") as f:
        json.dump(status_payload, f)

    target, curriculum = ws.detect_target_and_curriculum_steps()
    assert target == 1500
    assert curriculum == 25000


def test_multirun_log_scan_fallback(temp_checkpoint_dir):
    """When no JSON files exist, multi-session train.log correctly extracts the LAST session."""
    log_content = (
        "[Pulsar] Starting training: Steps 1 -> 500 (Curriculum Horizon: 30517)...\n"
        "[Step  500/500 (Horizon: 30517)] Loss: 2.6761\n"
        "[Pulsar] Saved checkpoint -> checkpoints/milestone_step_500.pt\n"
        "[Pulsar] Starting training: Steps 501 -> 2000 (Curriculum Horizon: 30517)...\n"
        "[Step  550/2000 (Horizon: 30517)] Loss: 2.5830\n"
    )
    with open(os.path.join(temp_checkpoint_dir, "train.log"), "w", encoding="utf-8") as f:
        f.write(log_content)

    target, curriculum = ws.detect_target_and_curriculum_steps()
    assert target == 2000, "Should pick the active session target (2000), not the stale initial session (500)"
    assert curriculum == 30517


def test_enrich_status_payload_contract_schema(temp_checkpoint_dir):
    """The enriched payload must guarantee all fields required by index.html."""
    raw_status = {
        "step": 600,
        "max_steps": 30517,
        "target_steps": 2000,
        "tokens_seen": 78643200,
        "train_dataset_tokens": 952436152,
        "num_train_shards": 2,
        "global_batch_tokens": 131072,
        "warmup_steps": 300,
        "tok_per_sec": 9500.0,
        "train_loss": 2.65,
    }

    enriched = ws.enrich_status_payload(raw_status)

    required_keys = [
        "step",
        "active_target",
        "target_steps",
        "max_steps",
        "curriculum_steps",
        "phase",
        "eta_str",
        "elapsed_str",
        "status",
        "tokens_seen",
        "train_dataset_tokens",
        "num_train_shards",
        "global_batch_tokens",
        "model_config",
        "device_name",
    ]
    for key in required_keys:
        assert key in enriched, f"Missing required contract key in status payload: {key}"

    assert enriched["active_target"] == 2000
    assert enriched["phase"] == "Stable Phase"
    assert "h" in enriched["eta_str"] or "m" in enriched["eta_str"]


def test_phase_progression_logic():
    """Verify phase transitions adhere strictly to warmup and target thresholds."""
    # Warmup
    data_warmup = ws.enrich_status_payload({
        "step": 150,
        "warmup_steps": 300,
        "target_steps": 2000,
        "max_steps": 30517,
    })
    assert data_warmup["phase"] == "Warmup (150/300)"

    # Stable Phase
    data_stable = ws.enrich_status_payload({
        "step": 600,
        "warmup_steps": 300,
        "target_steps": 2000,
        "max_steps": 30517,
    })
    assert data_stable["phase"] == "Stable Phase"

    # Target Reached
    data_completed = ws.enrich_status_payload({
        "step": 2000,
        "warmup_steps": 300,
        "target_steps": 2000,
        "max_steps": 30517,
    })
    assert data_completed["phase"] == "Target Completed"
    assert data_completed["eta_str"] == "00m 00s"


def test_live_server_endpoints():
    """Verify the live server responds with 200 OK and valid JSON across main endpoints."""
    endpoints = ["/api/status", "/api/canaries", "/api/history?last_n=10"]
    for ep in endpoints:
        url = f"http://127.0.0.1:7860{ep}"
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=3) as resp:
                assert resp.status == 200
                data = json.loads(resp.read().decode("utf-8"))
                assert isinstance(data, (dict, list))
        except urllib.error.URLError:
            pytest.skip("Web server not running on port 7860 (skipped in headless test runner)")

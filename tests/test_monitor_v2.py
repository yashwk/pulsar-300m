import math
import os
import tempfile
import pytest

from monitor.v2_api import (
    IncrementalMetricsReader,
    downsample_records,
    evaluate_health,
    load_thresholds,
    add_note,
    load_notes,
    DEFAULT_THRESHOLDS,
)


def test_downsample_preserves_endpoints_and_spikes():
    # Create 1000 dummy records with a prominent loss spike at step 500
    records = []
    for s in range(1, 1001):
        tl = 2.0 + math.sin(s / 10.0) * 0.2
        if s == 500:
            tl = 10.5  # Extreme spike
        records.append({"step": s, "train_loss": tl})

    # Downsample to 100 points
    downsampled = downsample_records(records, max_points=100)
    assert len(downsampled) <= 100
    assert len(downsampled) > 20

    # Endpoints preserved
    assert downsampled[0]["step"] == 1
    assert downsampled[-1]["step"] == 1000

    # Spike preserved
    steps = [r["step"] for r in downsampled]
    assert 500 in steps
    spike_record = next(r for r in downsampled if r["step"] == 500)
    assert spike_record["train_loss"] == 10.5


def test_downsample_since_step_and_fields():
    records = [{"step": s, "train_loss": 2.0, "val_loss": 2.1, "extra": "foo"} for s in range(1, 51)]

    # Filter since_step=40 and project fields
    filtered = downsample_records(records, max_points=100, fields=["train_loss"], since_step=40)
    assert len(filtered) == 10
    assert filtered[0]["step"] == 41
    assert filtered[-1]["step"] == 50
    assert "train_loss" in filtered[0]
    assert "val_loss" not in filtered[0]
    assert "extra" not in filtered[0]


def test_incremental_metrics_reader_partial_lines_and_restarts():
    with tempfile.NamedTemporaryFile("w+", delete=False) as f:
        path = f.name
        f.write('{"step": 1, "train_loss": 2.5, "pulse_losses": [2.6, 2.5, 2.4]}\n')
        f.write('{"step": 2, "train_loss": 2.4, "pulse_losses": [2.5, 2.4, 2.3]}\n')
        # Incomplete line without newline
        f.write('{"step": 3, "train_loss": 2.3')
        f.flush()

    try:
        reader = IncrementalMetricsReader(path)
        records = reader.read_records()
        assert len(records) == 2
        assert records[0]["step"] == 1
        assert records[0]["train_ce_final"] == 2.4  # Last pulse loss
        assert records[1]["step"] == 2

        # Now complete line 3 and append line 4
        with open(path, "a") as f:
            f.write(', "pulse_losses": [2.4, 2.3, 2.2]}\n')
            f.write('{"step": 4, "train_loss": 2.2, "pulse_losses": [2.3, 2.2, 2.1]}\n')

        records2 = reader.read_records()
        assert len(records2) == 4
        assert records2[2]["step"] == 3
        assert records2[3]["step"] == 4

        # Test restart: trainer restarts at step 3, writing step 3 and 4 again
        with open(path, "a") as f:
            f.write('{"step": 3, "train_loss": 2.15, "pulse_losses": [2.3, 2.2, 2.05]}\n')

        records3 = reader.read_records()
        # Should have pruned previous step 3 & 4 and replaced step 3 with newest
        assert records3[-1]["step"] == 3
        assert records3[-1]["train_loss"] == 2.15
        assert len(records3) == 3
    finally:
        os.remove(path)


def test_evaluate_health_thresholds():
    th = dict(DEFAULT_THRESHOLDS)

    # 1. Healthy state
    status = {
        "status": "running",
        "step": 3200,
        "active_target": 3500,
        "train_loss": 2.05,
        "val_ce_final": 2.00,
        "grad_norm_total": 0.25,
        "tok_per_sec": 9200.0,
        "vram_peak": 6.8,
        "moe_drop_rate": 0.015,
        "worst_expert_share_pct": 13.5,
        "latent_gain": 0.11,
    }
    history = [
        {"step": 3200, "train_ce_final": 2.05, "val_ce_final": 2.00, "grad_norm_total": 0.25}
    ]

    h_ok = evaluate_health(status, history, th)
    assert h_ok["status"] == "OK"
    assert h_ok["checks"]["loss"]["status"] == "OK"
    assert h_ok["checks"]["moe_drops"]["status"] == "OK"

    # 2. Watch state: loss gap > 0.15 nats
    status_watch = dict(status, train_loss=2.30, val_ce_final=2.00)
    history_watch = [
        {"step": 3200, "train_ce_final": 2.30, "val_ce_final": 2.00, "grad_norm_total": 0.25}
    ]
    h_watch = evaluate_health(status_watch, history_watch, th)
    assert h_watch["status"] == "WATCH"
    assert h_watch["checks"]["loss"]["status"] == "WATCH"

    # 3. Alert state: MoE drop rate > 5.0%
    status_alert = dict(status, moe_drop_rate=0.065)
    h_alert = evaluate_health(status_alert, history, th)
    assert h_alert["status"] == "ALERT"
    assert h_alert["checks"]["moe_drops"]["status"] == "ALERT"

    # 4. Alert state: NaN in loss
    history_nan = [
        {"step": 3200, "train_loss": float("nan"), "train_ce_final": 2.0, "grad_norm_total": 0.2}
    ]
    h_nan = evaluate_health(status, history_nan, th)
    assert h_nan["status"] == "ALERT"
    assert h_nan["checks"]["numerics"]["status"] == "ALERT"


def test_notes_persistence():
    with tempfile.NamedTemporaryFile("w+", delete=False) as f:
        path = f.name

    try:
        note1 = add_note(path, step=1500, text="Resumed with CF 1.50", author="Engineer")
        assert note1["step"] == 1500
        assert note1["text"] == "Resumed with CF 1.50"

        note2 = add_note(path, step=2000, text="Passed 2.00 val CE milestone", author="Lead")
        notes = load_notes(path)
        assert len(notes) == 2
        assert notes[0]["step"] == 1500
        assert notes[1]["step"] == 2000
    finally:
        os.remove(path)

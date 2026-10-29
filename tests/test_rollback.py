import json
import os
import tempfile
import pytest

from src.train import rollback_telemetry, DualLogger


def test_rollback_metrics_jsonl():
    with tempfile.TemporaryDirectory() as tmpdir:
        jsonl_path = os.path.join(tmpdir, "metrics.jsonl")
        records = [
            {"step": 100, "train_loss": 3.0, "tokens_seen": 1000},
            {"step": 200, "train_loss": 2.5, "tokens_seen": 2000},
            {"step": 300, "train_loss": 2.2, "tokens_seen": 3000},
            {"step": 400, "train_loss": 2.0, "tokens_seen": 4000},
        ]
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        stats = rollback_telemetry(tmpdir, max_step=250)
        assert stats["pruned_jsonl"] == 2
        assert stats["kept_jsonl"] == 2

        with open(jsonl_path, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]

        assert len(lines) == 2
        assert lines[0]["step"] == 100
        assert lines[1]["step"] == 200


def test_rollback_metrics_csv():
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "metrics.csv")
        header = "step,tokens_seen,train_loss\n"
        rows = [
            "100,1000,3.0\n",
            "200,2000,2.5\n",
            "300,3000,2.2\n",
        ]
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(header)
            for r in rows:
                f.write(r)

        stats = rollback_telemetry(tmpdir, max_step=200)
        assert stats["pruned_csv"] == 1
        assert stats["kept_csv"] == 2

        with open(csv_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        assert len(lines) == 3
        assert lines[0] == header
        assert lines[1].startswith("100,")
        assert lines[2].startswith("200,")


def test_rollback_live_status():
    with tempfile.TemporaryDirectory() as tmpdir:
        jsonl_path = os.path.join(tmpdir, "metrics.jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"step": 100, "train_loss": 2.5, "tokens_seen": 1000}) + "\n")
            f.write(json.dumps({"step": 200, "train_loss": 2.1, "tokens_seen": 2000}) + "\n")

        live_path = os.path.join(tmpdir, "live_status.json")
        with open(live_path, "w", encoding="utf-8") as f:
            json.dump({"step": 250, "train_loss": 1.9, "status": "training"}, f)

        stats = rollback_telemetry(tmpdir, max_step=100)
        assert stats["rolled_back_live_status"] == 1

        with open(live_path, "r", encoding="utf-8") as f:
            live_data = json.load(f)

        assert live_data["step"] == 100
        assert live_data["train_loss"] == 2.5
        assert live_data["status"] == "resumed"


def test_rollback_train_log():
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = os.path.join(tmpdir, "train.log")
        content = (
            "Starting run...\n"
            "[Step  100/1000] Loss: 3.1000\n"
            "  ↳ MoE Experts: ...\n"
            "[Step  200/1000] Loss: 2.8000\n"
            "  ↳ Pulses: ...\n"
            "[Pulsar] Saved checkpoint -> latest.pt (step 200)\n"
            "\n"
            "[Step  300/1000] Loss: 2.5000\n"
            "  ↳ MoE Experts: ...\n"
            "[Step  400/1000] Loss: 2.3000\n"
        )
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(content)

        stats = rollback_telemetry(tmpdir, max_step=200)
        assert stats["pruned_log_lines"] > 0

        with open(log_path, "r", encoding="utf-8") as f:
            pruned_content = f.read()

        assert "[Step  100/1000]" in pruned_content
        assert "[Step  200/1000]" in pruned_content
        assert "[Pulsar] Saved checkpoint -> latest.pt (step 200)" in pruned_content
        assert "[Step  300/1000]" not in pruned_content
        assert "[Step  400/1000]" not in pruned_content


def test_rollback_noop_and_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Empty dir
        stats = rollback_telemetry(tmpdir, max_step=500)
        assert stats["pruned_jsonl"] == 0
        assert stats["pruned_csv"] == 0
        assert stats["pruned_log_lines"] == 0

        # No-op when max_step is beyond records
        jsonl_path = os.path.join(tmpdir, "metrics.jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"step": 100}) + "\n")
            f.write(json.dumps({"step": 200}) + "\n")

        stats2 = rollback_telemetry(tmpdir, max_step=300)
        assert stats2["pruned_jsonl"] == 0
        assert stats2["kept_jsonl"] == 2


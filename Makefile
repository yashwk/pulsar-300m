.PHONY: help build test test-fast smoke monitor sync inspect clean lint

PYTHON ?= .venv/bin/python
PYTEST ?= uv run pytest

help:
	@echo "Pulsar-300M Development Automation"
	@echo "=================================="
	@echo "make build       - Compile fused CUDA C++ extension (pulsar_cuda)"
	@echo "make test        - Run complete test suite"
	@echo "make test-fast   - Run fast kernel & rollback tests"
	@echo "make smoke       - Run a 2-step CPU/GPU training smoke test"
	@echo "make monitor     - Launch web dashboard server on port 7860"
	@echo "make sync        - Launch background Modal volume telemetry sync"
	@echo "make inspect     - Inspect local latest.pt checkpoint metadata"
	@echo "make clean       - Clean build artifacts and pycache"

build:
	pip install -e . --no-build-isolation

test:
	$(PYTEST) tests/

test-fast:
	$(PYTEST) tests/test_kernels.py tests/test_rollback.py

smoke:
	$(PYTHON) -m src.train --max-steps 2 --warmup-steps 1 --no-save \
		--micro-batch-size 2 --seq-len 1024 --train-bin data/train_shard0.bin \
		--val-bin data/val_shard0.bin --tokenizer data/pulsar_tokenizer

monitor:
	$(PYTHON) monitor/web_server.py --port 7860 --checkpoint-dir checkpoints

sync:
	$(PYTHON) scripts/sync_cloud_monitor.py --checkpoint-dir checkpoints

inspect:
	$(PYTHON) scripts/inspect_checkpoint.py checkpoints/latest.pt

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf build/ dist/ *.egg-info .pytest_cache/

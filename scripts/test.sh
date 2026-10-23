#!/usr/bin/env bash
# Run the Pulsar test suite.
#
#   ./scripts/test.sh              # CPU suite (GPU tests auto-skip)
#   ./scripts/test.sh --gpu        # everything, including GPU tests
#   ./scripts/test.sh -k moe       # any extra args are forwarded to pytest
set -euo pipefail

cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/bin/python}"

if [[ ! -x "$PY" ]]; then
  PY=python
fi

if [[ "${1:-}" == "--gpu" ]]; then
  shift
  exec "$PY" -m pytest -x -q "$@"
fi

exec "$PY" -m pytest -x -q -m "not gpu" "$@"

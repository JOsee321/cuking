#!/usr/bin/env bash
# End-to-end CPU smoke test runner for Cuking
set -e

echo "=== Running Cuking End-to-End Smoke Test on CPU ==="
python -m pytest tests/test_smoke.py -v --tb=short
echo "=== Smoke test completed successfully ==="

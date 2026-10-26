"""Unit tests for report-only test evaluation module."""

from pathlib import Path
import numpy as np
import pytest

from cuking.evaluate import REPORT_ONLY_WARNING, evaluate_run
from cuking.utils import save_json


def test_evaluate_synthetic_logits(tmp_path: Path, capsys):
    """Verify evaluation on synthetic logits produces expected metrics and report."""
    run_dir = tmp_path / "synthetic_run"
    run_dir.mkdir(parents=True)

    # Save metrics metadata
    save_json({"task": "smsa", "model": "mock-model", "lr": 1e-4, "seed": 42}, run_dir / "metrics.json")

    # SmSA test ground truth has 500 rows and 3 classes
    num_samples = 500
    num_classes = 3
    rng = np.random.RandomState(42)
    synthetic_logits = rng.randn(num_samples, num_classes)
    np.save(run_dir / "test_logits.npy", synthetic_logits)

    result = evaluate_run(run_dir=run_dir, split="test", data_dir="data", save_report=True)

    assert result["task"] == "smsa"
    assert result["split"] == "test"
    assert "test_macro_f1" in result
    assert "test_acc" in result
    assert result["num_samples"] == 500
    assert 0.0 <= result["test_macro_f1"] <= 1.0
    assert 0.0 <= result["test_acc"] <= 1.0

    # Verify test_report.json was written to disk
    report_file = run_dir / "test_report.json"
    assert report_file.exists()

    # Verify REPORT ONLY warning was printed
    captured = capsys.readouterr()
    assert "REPORT ONLY" in captured.out


def test_evaluate_row_count_mismatch(tmp_path: Path):
    """Verify that a row count mismatch between logits and ground truth raises ValueError."""
    run_dir = tmp_path / "mismatch_run"
    run_dir.mkdir(parents=True)

    save_json({"task": "smsa", "model": "mock-model"}, run_dir / "metrics.json")

    # Intentionally incorrect row count (300 instead of 500)
    synthetic_logits = np.zeros((300, 3))
    np.save(run_dir / "test_logits.npy", synthetic_logits)

    with pytest.raises(ValueError, match="Row count mismatch"):
        evaluate_run(run_dir=run_dir, split="test", data_dir="data", save_report=False)

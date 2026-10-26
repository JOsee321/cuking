"""Unit tests for experiment results aggregation."""

from pathlib import Path
import pandas as pd
from cuking.aggregate import aggregate_results
from cuking.utils import save_json


def test_aggregate_results_empty(tmp_path: Path):
    """Verify aggregation handles empty experiments folder cleanly."""
    empty_exp = tmp_path / "experiments"
    empty_exp.mkdir()
    out_csv = tmp_path / "summary.csv"

    df = aggregate_results(experiments_dir=empty_exp, output_csv=out_csv)
    assert df.empty
    assert out_csv.exists()


def test_aggregate_results_with_runs(tmp_path: Path):
    """Verify aggregation correctly collates, sorts, and saves multiple metrics.json files."""
    exp_dir = tmp_path / "experiments"

    run1 = exp_dir / "emot" / "indobert-base_lr2e-5_s42"
    save_json({
        "task": "emot",
        "model": "indobert-base",
        "lr": 2e-5,
        "seed": 42,
        "valid_macro_f1": 0.72,
        "valid_acc": 0.74,
        "best_epoch": 4.0,
        "train_time_s": 120.5,
    }, run1 / "metrics.json")

    run2 = exp_dir / "emot" / "indobert-base_lr3e-5_s42"
    save_json({
        "task": "emot",
        "model": "indobert-base",
        "lr": 3e-5,
        "seed": 42,
        "valid_macro_f1": 0.75,
        "valid_acc": 0.77,
        "best_epoch": 5.0,
        "train_time_s": 125.0,
    }, run2 / "metrics.json")

    out_csv = tmp_path / "summary.csv"
    df = aggregate_results(experiments_dir=exp_dir, output_csv=out_csv)

    assert len(df) == 2
    assert out_csv.exists()

    # Verify sorting: higher valid_macro_f1 should come first
    assert df.iloc[0]["valid_macro_f1"] == 0.75
    assert df.iloc[1]["valid_macro_f1"] == 0.72
    assert "run_dir" in df.columns

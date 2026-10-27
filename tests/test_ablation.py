"""Unit tests for ablation report generation and delta calculations."""

from pathlib import Path
import pandas as pd
import pytest

from cuking.ablation import generate_ablation_report
from cuking.utils import save_json


def test_ablation_generation(tmp_path):
    """Verify ablation report groups best_single, family baselines, and ensembles correctly."""
    exp_dir = tmp_path / "experiments"
    task_dir = exp_dir / "emot"
    task_dir.mkdir(parents=True)

    # 1. Single run: xlmr-large (best single, f1=0.796)
    r1 = task_dir / "xlmr-large_lr2e-05_s42"
    r1.mkdir()
    save_json({"model": "xlmr-large", "valid_macro_f1": 0.796, "is_ensemble": False}, r1 / "metrics.json")

    # 2. Single run: xlmr-large lower lr (f1=0.788)
    r2 = task_dir / "xlmr-large_lr1e-05_s42"
    r2.mkdir()
    save_json({"model": "xlmr-large", "valid_macro_f1": 0.788, "is_ensemble": False}, r2 / "metrics.json")

    # 3. Single run: indobert-large (f1=0.780)
    r3 = task_dir / "indobert-large_lr2e-05_s42"
    r3.mkdir()
    save_json({"model": "indobert-large", "valid_macro_f1": 0.780, "is_ensemble": False}, r3 / "metrics.json")

    # 4. Ensemble run (f1=0.805)
    r4 = task_dir / "ensemble_top3"
    r4.mkdir()
    save_json({
        "model": "ensemble_top3",
        "valid_macro_f1": 0.805,
        "is_ensemble": True,
        "members": ["r1", "r2", "r3"],
    }, r4 / "metrics.json")

    out_csv = tmp_path / "ablation.csv"
    out_readme = tmp_path / "ablation_README.md"

    df = generate_ablation_report(
        experiments_dir=exp_dir,
        output_csv=out_csv,
        output_readme=out_readme,
        tasks=["emot"],
    )

    assert out_csv.exists()
    assert out_readme.exists()

    expected_cols = ["task", "name", "valid_macro_f1", "delta_vs_best_single", "members"]
    assert list(df.columns) == expected_cols

    # Names present
    names = list(df["name"])
    assert any("best_single" in n for n in names)
    assert any("best_xlmr-large" in n for n in names)
    assert any("best_indobert-large" in n for n in names)
    assert any("ensemble_top3" in n for n in names)

    # Delta checks
    # best_single delta must be 0.0
    best_single_row = df[df["name"].str.startswith("best_single")].iloc[0]
    assert abs(best_single_row["delta_vs_best_single"]) < 1e-9

    # ensemble delta must be 0.805 - 0.796 = 0.009
    ens_row = df[df["name"] == "ensemble_top3"].iloc[0]
    assert abs(ens_row["delta_vs_best_single"] - 0.009) < 1e-6

    # README contains the required caution note
    readme_content = out_readme.read_text(encoding="utf-8")
    assert "optimistis" in readme_content.lower()
    assert "multi-seed" in readme_content.lower()

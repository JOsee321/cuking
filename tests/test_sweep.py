"""Unit tests for sweep runner, dry-run ordering, resumption logic, and isolation rules."""

import ast
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch
import pytest

from cuking.sweep import (
    DEFAULT_GRID_ORDER,
    DEFAULT_TASKS,
    build_sweep_plan,
    format_lr,
    make_results_zip,
    run_sweep,
)
from cuking.utils import save_json


def test_dry_run_order():
    """Verify default grid has exactly 12 runs in the required execution sequence."""
    plan = build_sweep_plan()
    assert len(plan) == 12

    # First 6 runs must be EmoT, remaining 6 runs must be SmSA
    emot_runs = plan[:6]
    smsa_runs = plan[6:]

    assert all(r["task"] == "emot" for r in emot_runs)
    assert all(r["task"] == "smsa" for r in smsa_runs)

    expected_sequence = [
        ("xlmr-large", 1e-5),
        ("indobert-large", 3e-5),
        ("xlmr-large", 2e-5),
        ("indobert-large", 4e-5),
        ("xlmr-large", 5e-6),
        ("indobert-large", 2e-5),
    ]

    for run, (exp_model, exp_lr) in zip(emot_runs, expected_sequence):
        assert run["model"] == exp_model
        assert abs(run["lr"] - exp_lr) < 1e-9
        assert run["seed"] == 42

    for run, (exp_model, exp_lr) in zip(smsa_runs, expected_sequence):
        assert run["model"] == exp_model
        assert abs(run["lr"] - exp_lr) < 1e-9
        assert run["seed"] == 42


def test_task_filtering_and_max_runs():
    """Test task selection and max_runs limits."""
    emot_plan = build_sweep_plan(tasks=["emot"])
    assert len(emot_plan) == 6
    assert all(r["task"] == "emot" for r in emot_plan)

    smsa_plan = build_sweep_plan(tasks=["smsa"], max_runs=3)
    assert len(smsa_plan) == 3
    assert all(r["task"] == "smsa" for r in smsa_plan)
    assert smsa_plan[0]["model"] == "xlmr-large"
    assert abs(smsa_plan[0]["lr"] - 1e-5) < 1e-9


def test_format_lr():
    """Verify learning rates are formatted cleanly."""
    assert format_lr(1e-5) == "1e-5"
    assert format_lr(2e-5) == "2e-5"
    assert format_lr(5e-6) == "5e-6"
    assert format_lr(3e-5) == "3e-5"
    assert format_lr(4e-5) == "4e-5"


def test_resume_logic_skipping(tmp_path):
    """Verify that completed runs and previously failed runs are skipped without --retry-failed."""
    exp_dir = tmp_path / "experiments"
    summary_file = tmp_path / "summary.csv"

    # Setup 3 runs:
    # 1. Completed run with metrics.json
    run1_dir = exp_dir / "emot" / "xlmr-large_lr1e-05_s42"
    run1_dir.mkdir(parents=True)
    save_json({"valid_macro_f1": 0.765, "train_time_s": 480.0}, run1_dir / "metrics.json")

    # 2. Failed run with failed.json
    run2_dir = exp_dir / "emot" / "indobert-large_lr3e-05_s42"
    run2_dir.mkdir(parents=True)
    save_json({"alasan": "loss_is_nan"}, run2_dir / "failed.json")

    # 3. Fresh run (not executed yet)
    # Target will be run3_dir = exp_dir / "emot" / "xlmr-large_lr2e-05_s42"

    plan = [
        {"task": "emot", "model": "xlmr-large", "lr": 1e-5, "seed": 42},
        {"task": "emot", "model": "indobert-large", "lr": 3e-5, "seed": 42},
        {"task": "emot", "model": "xlmr-large", "lr": 2e-5, "seed": 42},
    ]

    with patch("subprocess.run") as mock_subproc:
        mock_subproc.return_value = MagicMock(returncode=0)
        # Mock writing metrics for run 3 upon execution
        def fake_train(*args, **kwargs):
            run3_dir = exp_dir / "emot" / "xlmr-large_lr2e-05_s42"
            run3_dir.mkdir(parents=True, exist_ok=True)
            save_json({"valid_macro_f1": 0.772, "train_time_s": 490.0}, run3_dir / "metrics.json")
            return MagicMock(returncode=0)

        mock_subproc.side_effect = fake_train

        results = run_sweep(
            plan=plan,
            experiments_dir=exp_dir,
            summary_csv=summary_file,
            dry_run=False,
            retry_failed=False,
        )

        # Only run 3 should have invoked subprocess.run
        assert mock_subproc.call_count == 1
        executed_cmd = mock_subproc.call_args[0][0]
        assert "cuking.train" in executed_cmd
        assert "--lr" in executed_cmd and "2e-05" in executed_cmd or "2e-5" in executed_cmd

        assert results[0]["status"] == "skipped_completed"
        assert results[1]["status"] == "skipped_failed"
        assert results[2]["status"] == "success"


def test_resume_logic_retry_failed(tmp_path):
    """Verify that --retry-failed will re-execute a run that only has failed.json."""
    exp_dir = tmp_path / "experiments"
    summary_file = tmp_path / "summary.csv"

    run_dir = exp_dir / "emot" / "indobert-large_lr3e-05_s42"
    run_dir.mkdir(parents=True)
    save_json({"alasan": "loss_is_nan"}, run_dir / "failed.json")

    plan = [
        {"task": "emot", "model": "indobert-large", "lr": 3e-5, "seed": 42},
    ]

    with patch("subprocess.run") as mock_subproc:
        mock_subproc.return_value = MagicMock(returncode=0)

        results = run_sweep(
            plan=plan,
            experiments_dir=exp_dir,
            summary_csv=summary_file,
            dry_run=False,
            retry_failed=True,
        )

        assert mock_subproc.call_count == 1
        cmd = mock_subproc.call_args[0][0]
        assert "--force" in cmd


def test_strict_isolation_rules():
    """Verify sweep.py strictly NEVER imports cuking.evaluate or reads test labels."""
    sweep_path = Path("src/cuking/sweep.py")
    assert sweep_path.exists()
    content = sweep_path.read_text(encoding="utf-8")

    # AST check for imports
    tree = ast.parse(content, filename=str(sweep_path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "evaluate" not in alias.name, f"Forbidden import found: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            assert "evaluate" not in mod, f"Forbidden from-import module found: {mod}"
            for alias in node.names:
                assert "evaluate" not in alias.name, f"Forbidden from-import symbol found: {alias.name}"

    # String checks for test splits or test datasets
    assert "--split test" not in content
    assert "test_preprocess" not in content
    assert "test.tsv" not in content
    assert "test.csv" not in content

    # Runtime check: importing cuking.sweep must not import cuking.evaluate
    if "cuking.evaluate" in sys.modules:
        del sys.modules["cuking.evaluate"]

    import cuking.sweep
    assert "cuking.evaluate" not in sys.modules, "cuking.evaluate was unexpectedly imported by cuking.sweep!"


def test_make_results_zip(tmp_path):
    """Verify zip creation utility creates an archive of the experiments folder."""
    exp_dir = tmp_path / "results" / "experiments"
    exp_dir.mkdir(parents=True)
    (exp_dir / "test_file.txt").write_text("sample content", encoding="utf-8")

    zip_target = tmp_path / "working" / "cuking_phase2_results.zip"
    make_results_zip(experiments_dir=exp_dir, zip_path=zip_target)

    assert zip_target.exists()
    assert zip_target.stat().st_size > 0

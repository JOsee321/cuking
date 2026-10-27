"""Unit tests for probability-averaging ensemble logic, isolation rules, and member selection."""

import ast
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
import pytest

from cuking.ensemble import run_ensemble, select_ensemble_members
from cuking.utils import save_json


def test_complementary_synthetic_ensemble(tmp_path):
    """Verify that ensembling complementary models yields higher macro-F1 than individual models."""
    exp_dir = tmp_path / "experiments"
    task_dir = exp_dir / "emot"
    task_dir.mkdir(parents=True)

    # 4 validation samples with binary labels [0, 1, 0, 1]
    # Model 1: confident on samples 0 & 2, unconfident/wrong on sample 1
    m1_val_logits = np.array([
        [5.0, -5.0],   # pred 0 (true 0, high conf)
        [0.1, -0.1],   # pred 0 (true 1, weak wrong)
        [5.0, -5.0],   # pred 0 (true 0, high conf)
        [-0.1, 0.1],   # pred 1 (true 1, weak correct)
    ])
    # Model 2: confident on samples 1 & 3, unconfident/wrong on sample 0
    m2_val_logits = np.array([
        [-0.1, 0.1],   # pred 1 (true 0, weak wrong)
        [-5.0, 5.0],   # pred 1 (true 1, high conf)
        [0.1, -0.1],   # pred 0 (true 0, weak correct)
        [-5.0, 5.0],   # pred 1 (true 1, high conf)
    ])

    # Dummy test logits (same shapes)
    dummy_test_logits = np.zeros((4, 2))

    r1_dir = task_dir / "run1"
    r2_dir = task_dir / "run2"
    r1_dir.mkdir()
    r2_dir.mkdir()

    np.save(r1_dir / "valid_logits.npy", m1_val_logits)
    np.save(r1_dir / "test_logits.npy", dummy_test_logits)
    save_json({"model": "model_a", "valid_macro_f1": 0.73, "is_ensemble": False}, r1_dir / "metrics.json")

    np.save(r2_dir / "valid_logits.npy", m2_val_logits)
    np.save(r2_dir / "test_logits.npy", dummy_test_logits)
    save_json({"model": "model_b", "valid_macro_f1": 0.73, "is_ensemble": False}, r2_dir / "metrics.json")

    # Mock load_task to return the true labels
    with patch("cuking.ensemble.load_task") as mock_load:
        mock_load.return_value = {
            "validation": {"label": [0, 1, 0, 1]}
        }

        # Run individual evaluations for verification
        from cuking.metrics import compute_classification_metrics
        from cuking.ensemble import softmax

        m1_f1 = compute_classification_metrics(softmax(m1_val_logits), np.array([0, 1, 0, 1]))["macro_f1"]
        m2_f1 = compute_classification_metrics(softmax(m2_val_logits), np.array([0, 1, 0, 1]))["macro_f1"]

        assert m1_f1 < 1.0
        assert m2_f1 < 1.0

        # Execute ensemble
        result = run_ensemble(
            task="emot",
            name="test_pair",
            member_dirs=[r1_dir, r2_dir],
            strategy="custom",
            experiments_dir=exp_dir,
        )

        assert result["valid_macro_f1"] > m1_f1
        assert result["valid_macro_f1"] > m2_f1
        assert result["valid_macro_f1"] == 1.0
        assert result["valid_acc"] == 1.0

        # Verify output files
        ens_dir = task_dir / "ensemble_test_pair"
        assert (ens_dir / "metrics.json").exists()
        assert (ens_dir / "valid_probs.npy").exists()
        assert (ens_dir / "test_probs.npy").exists()
        assert (ens_dir / "valid_logits.npy").exists()
        assert (ens_dir / "test_logits.npy").exists()


def test_ensemble_excludes_existing_ensembles(tmp_path):
    """Verify that ensemble_* runs are strictly excluded from being selected as ensemble members."""
    exp_dir = tmp_path / "experiments"
    task_dir = exp_dir / "emot"
    task_dir.mkdir(parents=True)

    dummy_vl = np.zeros((4, 2))
    dummy_tl = np.zeros((4, 2))

    # Single model runs
    r1 = task_dir / "xlmr-large_lr1e-05_s42"
    r2 = task_dir / "indobert-large_lr2e-05_s42"
    r1.mkdir()
    r2.mkdir()
    np.save(r1 / "valid_logits.npy", dummy_vl)
    np.save(r1 / "test_logits.npy", dummy_tl)
    save_json({"model": "xlmr-large", "valid_macro_f1": 0.80, "is_ensemble": False}, r1 / "metrics.json")

    np.save(r2 / "valid_logits.npy", dummy_vl)
    np.save(r2 / "test_logits.npy", dummy_tl)
    save_json({"model": "indobert-large", "valid_macro_f1": 0.78, "is_ensemble": False}, r2 / "metrics.json")

    # An existing ensemble run
    ens_old = task_dir / "ensemble_previous"
    ens_old.mkdir()
    np.save(ens_old / "valid_logits.npy", dummy_vl)
    np.save(ens_old / "test_logits.npy", dummy_tl)
    save_json({"model": "ensemble_previous", "valid_macro_f1": 0.85, "is_ensemble": True}, ens_old / "metrics.json")

    # Strategy topk should select r1 and r2, NEVER ens_old
    selected_topk = select_ensemble_members(
        task="emot",
        strategy="topk",
        experiments_dir=exp_dir,
        summary_csv=None,
        k=2,
    )
    assert len(selected_topk) == 2
    assert ens_old not in selected_topk
    assert r1 in selected_topk
    assert r2 in selected_topk

    # Strategy best_per_model should select r1 and r2, NEVER ens_old
    selected_bpm = select_ensemble_members(
        task="emot",
        strategy="best_per_model",
        experiments_dir=exp_dir,
        summary_csv=None,
    )
    assert len(selected_bpm) == 2
    assert ens_old not in selected_bpm

    # Passing ensemble run directly to run_ensemble must raise ValueError
    with pytest.raises(ValueError, match="cannot be a member"):
        run_ensemble(
            task="emot",
            name="nested_fail",
            member_dirs=[r1, ens_old],
            experiments_dir=exp_dir,
        )


def test_ensemble_dimension_mismatch(tmp_path):
    """Verify that dimension mismatches between members are caught and rejected."""
    exp_dir = tmp_path / "experiments"
    task_dir = exp_dir / "emot"
    task_dir.mkdir(parents=True)

    r1 = task_dir / "run1"
    r2 = task_dir / "run2"
    r1.mkdir()
    r2.mkdir()

    np.save(r1 / "valid_logits.npy", np.zeros((4, 2)))
    np.save(r1 / "test_logits.npy", np.zeros((4, 2)))
    np.save(r2 / "valid_logits.npy", np.zeros((5, 2)))  # 5 rows instead of 4
    np.save(r2 / "test_logits.npy", np.zeros((4, 2)))

    with pytest.raises(ValueError, match="mismatch"):
        run_ensemble(
            task="emot",
            name="mismatch_fail",
            member_dirs=[r1, r2],
            experiments_dir=exp_dir,
        )


def test_strict_no_test_label_reading():
    """Verify ensemble.py never imports cuking.evaluate or reads test label files."""
    ens_path = Path("src/cuking/ensemble.py")
    assert ens_path.exists()
    content = ens_path.read_text(encoding="utf-8")

    tree = ast.parse(content, filename=str(ens_path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "evaluate" not in alias.name
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            assert "evaluate" not in mod

    assert "split=\"test\"" not in content
    assert "split='test'" not in content
    assert "test_preprocess" not in content

    # Runtime module check
    if "cuking.evaluate" in sys.modules:
        del sys.modules["cuking.evaluate"]
    import cuking.ensemble
    assert "cuking.evaluate" not in sys.modules

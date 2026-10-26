"""End-to-end smoke tests for classification pipeline on CPU."""

from pathlib import Path
import shutil
import numpy as np
import pytest
from cuking.train import train_single_run


@pytest.fixture(scope="module")
def smoke_output_dir(tmp_path_factory) -> Path:
    """Create temporary directory for smoke test outputs."""
    temp_dir = tmp_path_factory.mktemp("smoke_experiments")
    yield temp_dir
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_emot_smoke_run(smoke_output_dir: Path):
    """Verify complete end-to-end smoke run on EmoT task."""
    out_dir = smoke_output_dir / "emot_smoke"
    metrics = train_single_run(
        task="emot",
        model_alias="indobert-base",
        output_dir=str(out_dir),
        is_smoke=True,
        force=True,
    )

    # Check metrics output dictionary
    assert metrics["task"] == "emot"
    assert "valid_macro_f1" in metrics
    assert "valid_acc" in metrics
    assert "train_time_s" in metrics
    assert metrics["smoke"] is True
    assert metrics["train_time_s"] < 60.0  # Must finish well below 5 minutes

    # Verify generated artifact files on disk
    assert (out_dir / "metrics.json").exists()
    assert (out_dir / "config.yaml").exists()

    valid_logits_path = out_dir / "valid_logits.npy"
    test_logits_path = out_dir / "test_logits.npy"
    assert valid_logits_path.exists()
    assert test_logits_path.exists()

    valid_logits = np.load(valid_logits_path)
    test_logits = np.load(test_logits_path)

    # EmoT has 5 classes; smoke slice has 20 samples
    assert valid_logits.ndim == 2
    assert valid_logits.shape == (20, 5)
    assert test_logits.ndim == 2
    assert test_logits.shape == (20, 5)


def test_smsa_smoke_run(smoke_output_dir: Path):
    """Verify complete end-to-end smoke run on SmSA task."""
    out_dir = smoke_output_dir / "smsa_smoke"
    metrics = train_single_run(
        task="smsa",
        model_alias="indobert-base",
        output_dir=str(out_dir),
        is_smoke=True,
        force=True,
    )

    assert metrics["task"] == "smsa"
    assert "valid_macro_f1" in metrics
    assert "valid_acc" in metrics
    assert metrics["smoke"] is True
    assert metrics["train_time_s"] < 60.0

    assert (out_dir / "metrics.json").exists()
    assert (out_dir / "config.yaml").exists()

    valid_logits = np.load(out_dir / "valid_logits.npy")
    test_logits = np.load(out_dir / "test_logits.npy")

    # SmSA has 3 classes; smoke slice has 20 samples
    assert valid_logits.ndim == 2
    assert valid_logits.shape == (20, 3)
    assert test_logits.ndim == 2
    assert test_logits.shape == (20, 3)

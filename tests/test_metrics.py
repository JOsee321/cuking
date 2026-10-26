"""Unit tests for classification evaluation metrics."""

import numpy as np
import pytest
from cuking.metrics import build_compute_metrics_fn, compute_classification_metrics


def test_perfect_classification():
    """Verify metrics when predictions match labels exactly."""
    labels = [0, 1, 2, 0, 1, 2]
    preds = [0, 1, 2, 0, 1, 2]

    metrics = compute_classification_metrics(preds, labels)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0


def test_zero_accuracy_classification():
    """Verify metrics when all predictions are incorrect."""
    labels = [0, 0, 0]
    preds = [1, 1, 1]

    metrics = compute_classification_metrics(preds, labels)
    assert metrics["accuracy"] == 0.0
    assert metrics["macro_f1"] == 0.0


def test_logits_input():
    """Verify that 2D logits input is converted correctly via argmax."""
    # 3 samples, 3 classes
    logits = np.array([
        [10.0, 1.0, -1.0],  # argmax 0
        [-2.0, 5.0, 0.0],   # argmax 1
        [0.0, 2.0, 8.0],    # argmax 2
    ])
    labels = np.array([0, 1, 1])  # sample 2 predicted 2 but label 1

    metrics = compute_classification_metrics(logits, labels)
    assert metrics["accuracy"] == pytest.approx(2.0 / 3.0)
    assert 0.0 < metrics["macro_f1"] < 1.0


def test_trainer_compute_metrics_fn():
    """Verify callback format matching transformers.Trainer."""
    fn = build_compute_metrics_fn()
    logits = np.array([
        [5.0, 0.0],
        [0.0, 5.0],
    ])
    labels = np.array([0, 1])

    eval_pred = (logits, labels)
    res = fn(eval_pred)
    assert res["accuracy"] == 1.0
    assert res["macro_f1"] == 1.0

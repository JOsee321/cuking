"""Evaluation metrics for IndoNLU classification tasks."""

from typing import Any, Callable, Dict, Sequence, Union
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def compute_classification_metrics(
    predictions: Union[np.ndarray, Sequence[int]],
    labels: Union[np.ndarray, Sequence[int]],
) -> Dict[str, float]:
    """Compute official IndoNLU classification metrics: macro-F1 and accuracy.

    Args:
        predictions: Predicted class IDs or 2D probability/logit array.
        labels: Ground truth class integer IDs.

    Returns:
        Dictionary containing 'macro_f1' and 'accuracy' as floats.
    """
    preds_arr = np.asarray(predictions)
    labels_arr = np.asarray(labels)

    # If predictions are logits or probabilities (2D array), take argmax
    if preds_arr.ndim > 1:
        preds_arr = np.argmax(preds_arr, axis=-1)

    macro_f1 = f1_score(labels_arr, preds_arr, average="macro", zero_division=0)
    acc = accuracy_score(labels_arr, preds_arr)

    return {
        "macro_f1": float(macro_f1),
        "accuracy": float(acc),
    }


def build_compute_metrics_fn() -> Callable[[Any], Dict[str, float]]:
    """Build a compute_metrics callback compatible with transformers.Trainer."""
    def compute_metrics_fn(eval_pred) -> Dict[str, float]:
        predictions, labels = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        return compute_classification_metrics(predictions, labels)

    return compute_metrics_fn

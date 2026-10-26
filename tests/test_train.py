"""Unit tests for training utilities and best epoch extraction."""

from types import SimpleNamespace
from cuking.train import extract_best_epoch


def test_extract_best_epoch_from_checkpoint():
    """Verify that best_epoch is extracted from checkpoint step in log_history."""
    mock_state = SimpleNamespace(
        best_model_checkpoint="checkpoints/tmp_emot_1234/checkpoint-25",
        best_global_step=None,
        best_metric=0.82,
        epoch=8.0,
        log_history=[
            {"step": 10, "epoch": 1.0, "eval_macro_f1": 0.70},
            {"step": 25, "epoch": 2.5, "eval_macro_f1": 0.82},
            {"step": 50, "epoch": 5.0, "eval_macro_f1": 0.79},
            {"step": 80, "epoch": 8.0, "eval_macro_f1": 0.75},
        ],
    )
    mock_trainer = SimpleNamespace(state=mock_state)

    # Even though training finished at epoch 8.0, the best checkpoint was at epoch 2.5
    best_epoch = extract_best_epoch(mock_trainer)
    assert best_epoch == 2.5


def test_extract_best_epoch_from_best_metric():
    """Verify fallback to best_metric matching when checkpoint path is absent."""
    mock_state = SimpleNamespace(
        best_model_checkpoint=None,
        best_global_step=None,
        best_metric=0.885,
        epoch=6.0,
        log_history=[
            {"step": 10, "epoch": 1.0, "eval_macro_f1": 0.75},
            {"step": 20, "epoch": 2.0, "eval_macro_f1": 0.885},
            {"step": 30, "epoch": 3.0, "eval_macro_f1": 0.81},
        ],
    )
    mock_trainer = SimpleNamespace(state=mock_state)

    best_epoch = extract_best_epoch(mock_trainer)
    assert best_epoch == 2.0


def test_extract_best_epoch_fallback_state_epoch():
    """Verify fallback to state.epoch when no checkpoint or log entries match."""
    mock_state = SimpleNamespace(
        best_model_checkpoint=None,
        best_global_step=None,
        best_metric=None,
        epoch=4.0,
        log_history=[],
    )
    mock_trainer = SimpleNamespace(state=mock_state)

    best_epoch = extract_best_epoch(mock_trainer)
    assert best_epoch == 4.0


def test_extract_best_epoch_no_state():
    """Verify graceful handling when state is None."""
    mock_trainer = SimpleNamespace(state=None)
    assert extract_best_epoch(mock_trainer) == 0.0

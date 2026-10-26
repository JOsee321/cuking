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


def test_nan_guard_callback():
    """Verify NaNGuardCallback sets nan_detected and should_training_stop on NaN loss."""
    from transformers.trainer_callback import TrainerControl
    from cuking.train import NaNGuardCallback

    control = TrainerControl()
    cb = NaNGuardCallback()

    cb.on_log(None, None, control, logs={"loss": float("nan")})
    assert cb.nan_detected is True
    assert control.should_training_stop is True
    assert "loss" in cb.nan_reason


def test_nan_guard_writes_failed_json(tmp_path, monkeypatch):
    """Verify NaNGuard writes failed.json and halts without writing metrics.json."""
    from pathlib import Path
    import pytest
    from transformers import Trainer
    from cuking.train import NaNGuardCallback, train_single_run
    from cuking.utils import load_json

    def mock_train(self, *args, **kwargs):
        for cb in self.callback_handler.callbacks:
            if isinstance(cb, NaNGuardCallback):
                cb.nan_detected = True
                cb.nan_reason = "Mock NaN loss detected"

    monkeypatch.setattr(Trainer, "train", mock_train)

    run_dir = tmp_path / "nan_run"
    with pytest.raises(RuntimeError, match="NaN guard triggered"):
        train_single_run(
            task="emot",
            model_alias="tiny-smoke",
            output_dir=str(run_dir),
            is_smoke=True,
            force=True,
        )

    assert (run_dir / "failed.json").exists()
    assert not (run_dir / "metrics.json").exists()

    failed_info = load_json(run_dir / "failed.json")
    assert failed_info["task"] == "emot"
    assert failed_info["model"] == "tiny-smoke"
    assert "alasan" in failed_info


def test_oom_fallback_retries_and_records_flag(tmp_path, monkeypatch):
    """Verify OOM on first attempt triggers fallback to batch=4, accum=4 and sets oom_fallback=True."""
    from pathlib import Path
    import torch
    from transformers import Trainer
    from cuking.train import train_single_run
    from cuking.utils import load_json

    call_count = 0
    recorded_batches = []
    orig_train = Trainer.train

    def mock_train(self, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        recorded_batches.append(self.args.per_device_train_batch_size)
        if call_count == 1:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return orig_train(self, *args, **kwargs)

    monkeypatch.setattr(Trainer, "train", mock_train)

    run_dir = tmp_path / "oom_run"
    metrics = train_single_run(
        task="emot",
        model_alias="tiny-smoke",
        output_dir=str(run_dir),
        is_smoke=True,
        force=True,
    )

    assert call_count == 2
    assert recorded_batches[1] == 4
    assert metrics["oom_fallback"] is True
    assert (run_dir / "metrics.json").exists()

    saved_m = load_json(run_dir / "metrics.json")
    assert saved_m["oom_fallback"] is True

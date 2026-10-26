"""Single-run training entrypoint for IndoNLU classification benchmarks."""

import argparse
import inspect
import os
from pathlib import Path
import shutil
import time
from typing import Any, Dict, Optional
import numpy as np
import torch
from transformers import (
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from cuking.data import TASK_LABELS, load_task, tokenize_dataset
from cuking.metrics import build_compute_metrics_fn, compute_classification_metrics
from cuking.model import build_model_and_tokenizer
from cuking.utils import load_yaml, save_json, save_yaml, set_seed, setup_logger


logger = setup_logger("cuking.train")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for training."""
    parser = argparse.ArgumentParser(description="Train single run on IndoNLU task.")
    parser.add_argument("--task", type=str, required=True, help="Task name (e.g. emot, smsa)")
    parser.add_argument("--model", type=str, default="indobert-base", help="Model alias or Hugging Face ID")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML file")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate override")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    parser.add_argument("--epochs", type=int, default=None, help="Epochs override")
    parser.add_argument("--batch_size", type=int, default=None, help="Per device train batch size override")
    parser.add_argument("--output_dir", type=str, default=None, help="Target results directory override")
    parser.add_argument("--smoke", action="store_true", help="Run in ultra-fast smoke test mode (CPU, small subset)")
    parser.add_argument("--force", action="store_true", help="Re-run even if metrics.json already exists")
    return parser.parse_args()


def load_merged_config(task: str, config_path: Optional[str] = None) -> Dict[str, Any]:
    """Merge base.yaml, task-specific yaml, and custom yaml if provided."""
    base_cfg_path = Path("configs/base.yaml")
    merged = load_yaml(base_cfg_path) if base_cfg_path.exists() else {}

    task_cfg_path = Path(f"configs/{task.lower()}.yaml")
    if task_cfg_path.exists():
        merged.update(load_yaml(task_cfg_path))

    if config_path:
        custom_cfg = Path(config_path)
        if custom_cfg.exists():
            merged.update(load_yaml(custom_cfg))

    return merged


def train_single_run(
    task: str,
    model_alias: str = "indobert-base",
    config_path: Optional[str] = None,
    lr: Optional[float] = None,
    seed: Optional[int] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    output_dir: Optional[str] = None,
    is_smoke: bool = False,
    force: bool = False,
) -> Dict[str, Any]:
    """Execute a single reproducible training run and persist artifacts."""
    task_key = task.lower()
    config = load_merged_config(task_key, config_path)

    # Apply command line overrides
    if lr is not None:
        config["learning_rate"] = lr
    if seed is not None:
        config["seed"] = seed
    if epochs is not None:
        config["num_train_epochs"] = epochs
    if batch_size is not None:
        config["per_device_train_batch_size"] = batch_size

    run_seed = int(config.get("seed", 42))
    run_lr = float(config.get("learning_rate", 2e-5))
    set_seed(run_seed)

    # Determine experiment output directory: results/experiments/<task>/<model>_lr<lr>_s<seed>/
    if output_dir:
        run_dir = Path(output_dir)
    else:
        model_tag = "smoke_model" if is_smoke else model_alias.replace("/", "_")
        run_name = f"{model_tag}_lr{run_lr}_s{run_seed}"
        run_dir = Path("results/experiments") / task_key / run_name

    metrics_file = run_dir / "metrics.json"
    if metrics_file.exists() and not force:
        logger.info(f"Run output already exists at {metrics_file}. Skipping execution.")
        import json
        with open(metrics_file, "r", encoding="utf-8") as f:
            return json.load(f)

    run_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    raw_dataset = load_task(task_key, data_dir="data")
    labels = TASK_LABELS[task_key]
    num_labels = len(labels)
    id2label = {i: lbl for i, lbl in enumerate(labels)}
    label2id = {lbl: i for i, lbl in enumerate(labels)}

    # Build model and tokenizer
    model, tokenizer = build_model_and_tokenizer(
        model_name_or_alias=model_alias,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        is_smoke=is_smoke,
    )

    # Tokenize dataset
    max_length = int(config.get("max_length", 128))
    tokenized = tokenize_dataset(raw_dataset, tokenizer, max_length=max_length)

    train_data = tokenized["train"]
    val_data = tokenized["validation"]
    test_data = tokenized["test"]

    # In smoke test mode: use tiny slice (50 train, 20 val, 20 test) and 3 steps
    if is_smoke:
        train_data = train_data.select(range(min(50, len(train_data))))
        val_data = val_data.select(range(min(20, len(val_data))))
        test_data = test_data.select(range(min(20, len(test_data))))
        config["max_steps"] = 3
        config["num_train_epochs"] = 1
        config["logging_steps"] = 1
        config["eval_steps"] = 3
        config["per_device_train_batch_size"] = 4
        config["per_device_eval_batch_size"] = 4
        config["gradient_accumulation_steps"] = 1
        config["fp16"] = False

    use_cuda = torch.cuda.is_available()
    use_fp16 = bool(config.get("fp16", False)) and use_cuda

    # Temporary directory for training checkpoints (kept outside results, ignored by git)
    temp_ckpt_dir = Path("checkpoints") / f"tmp_{task_key}_{int(time.time() * 1000)}"
    temp_ckpt_dir.mkdir(parents=True, exist_ok=True)

    sig = inspect.signature(TrainingArguments.__init__).parameters
    args_dict: Dict[str, Any] = {
        "output_dir": str(temp_ckpt_dir),
        "learning_rate": float(config["learning_rate"]),
        "per_device_train_batch_size": int(config.get("per_device_train_batch_size", 8)),
        "per_device_eval_batch_size": int(config.get("per_device_eval_batch_size", 32)),
        "gradient_accumulation_steps": int(config.get("gradient_accumulation_steps", 2)),
        "num_train_epochs": float(config.get("num_train_epochs", 8)),
        "max_steps": int(config.get("max_steps", -1)),
        "weight_decay": float(config.get("weight_decay", 0.01)),
        "fp16": use_fp16,
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "save_total_limit": 1,
        "logging_steps": int(config.get("logging_steps", 50)),
        "seed": run_seed,
        "report_to": "none",
    }

    strat = "steps" if is_smoke else "epoch"
    if "eval_strategy" in sig:
        args_dict["eval_strategy"] = strat
    elif "evaluation_strategy" in sig:
        args_dict["evaluation_strategy"] = strat

    args_dict["save_strategy"] = strat
    if is_smoke:
        args_dict["eval_steps"] = int(config.get("eval_steps", 3))
        args_dict["save_steps"] = int(config.get("eval_steps", 3))

    warmup_ratio = float(config.get("warmup_ratio", 0.1))
    if "warmup_ratio" in sig:
        args_dict["warmup_ratio"] = warmup_ratio
    else:
        total_steps = config.get("max_steps", -1)
        if total_steps <= 0:
            eff_batch = int(config.get("per_device_train_batch_size", 8)) * int(config.get("gradient_accumulation_steps", 1))
            steps_per_epoch = max(1, len(train_data) // eff_batch)
            total_steps = steps_per_epoch * int(config.get("num_train_epochs", 8))
        args_dict["warmup_steps"] = max(1, int(total_steps * warmup_ratio))

    if "use_cpu" in sig:
        args_dict["use_cpu"] = not use_cuda

    training_args = TrainingArguments(**args_dict)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    compute_metrics_fn = build_compute_metrics_fn()

    callbacks = []
    if not is_smoke:
        patience = int(config.get("early_stopping_patience", 3))
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))

    trainer_kwargs: Dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_data,
        "eval_dataset": val_data,
        "data_collator": data_collator,
        "compute_metrics": compute_metrics_fn,
        "callbacks": callbacks,
    }
    trainer_sig = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_sig:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_sig:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    logger.info(f"Starting training run: task={task_key}, model={model_alias}, lr={run_lr}, seed={run_seed}")
    start_time = time.time()
    train_result = trainer.train()
    train_time_s = time.time() - start_time

    # Generate validation logits and compute final metrics
    logger.info("Computing validation predictions and logits...")
    val_output = trainer.predict(val_data)
    valid_logits = val_output.predictions
    if isinstance(valid_logits, tuple):
        valid_logits = valid_logits[0]
    val_metrics = compute_classification_metrics(valid_logits, val_output.label_ids)

    # Generate test logits
    logger.info("Computing test predictions and logits...")
    test_output = trainer.predict(test_data)
    test_logits = test_output.predictions
    if isinstance(test_logits, tuple):
        test_logits = test_logits[0]

    # Clean up temporary training checkpoints to save disk space
    if temp_ckpt_dir.exists():
        shutil.rmtree(temp_ckpt_dir, ignore_errors=True)

    # Persist outputs in accordance with PRD section 9.3
    np.save(run_dir / "valid_logits.npy", valid_logits)
    np.save(run_dir / "test_logits.npy", test_logits)

    final_metrics = {
        "task": task_key,
        "model": model_alias,
        "lr": run_lr,
        "seed": run_seed,
        "valid_macro_f1": float(val_metrics["macro_f1"]),
        "valid_acc": float(val_metrics["accuracy"]),
        "best_epoch": float(trainer.state.epoch or 0.0),
        "train_time_s": round(train_time_s, 2),
        "smoke": is_smoke,
    }

    save_json(final_metrics, run_dir / "metrics.json")
    save_yaml(config, run_dir / "config.yaml")

    logger.info(f"Run completed successfully: {final_metrics}")
    return final_metrics


if __name__ == "__main__":
    cli_args = parse_args()
    train_single_run(
        task=cli_args.task,
        model_alias=cli_args.model,
        config_path=cli_args.config,
        lr=cli_args.lr,
        seed=cli_args.seed,
        epochs=cli_args.epochs,
        batch_size=cli_args.batch_size,
        output_dir=cli_args.output_dir,
        is_smoke=cli_args.smoke,
        force=cli_args.force,
    )

"""Report-only evaluation entrypoint for test split logits.

STRICT RULE: Test split evaluation is strictly for final reporting.
IT IS FORBIDDEN to use test split metrics for model selection, hyperparameter tuning,
early stopping, or ensemble member selection. All decision making must be based
on validation set performance.
"""

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Union
import numpy as np

from cuking.data import load_task
from cuking.metrics import compute_classification_metrics
from cuking.utils import load_json, save_json, setup_logger


logger = setup_logger("cuking.evaluate")

REPORT_ONLY_WARNING = (
    "\n" + "=" * 72 + "\n"
    "[WARNING: REPORT ONLY]\n"
    "Skor split test HANYA untuk laporan akhir (final reporting).\n"
    "DILARANG KERAS dipakai untuk memilih model, hyperparameter, early stopping,\n"
    "atau memilih anggota ensemble! Seluruh seleksi WAJIB menggunakan validation.\n"
    + "=" * 72 + "\n"
)


def evaluate_run(
    run_dir: Union[str, Path],
    split: str = "test",
    data_dir: str = "data",
    save_report: bool = True,
    task_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate predictions against ground truth labels.

    Args:
        run_dir: Experiment output directory containing logits and metrics.json.
        split: Split to evaluate ('test' or 'validation').
        data_dir: Local data directory.
        save_report: Whether to persist report JSON to disk.
        task_override: Task name override if metrics.json is absent.

    Returns:
        Dictionary containing evaluated metrics and sample counts.
    """
    run_path = Path(run_dir)
    if not run_path.exists():
        raise FileNotFoundError(f"Run directory not found: {run_path}")

    # Emit prominent warning for test split
    if split == "test":
        print(REPORT_ONLY_WARNING)
        logger.warning("REPORT ONLY: Test split evaluation requested.")

    # Determine task from metrics.json or override
    metrics_file = run_path / "metrics.json"
    metrics_meta: Dict[str, Any] = {}
    if metrics_file.exists():
        metrics_meta = load_json(metrics_file)

    task_name = task_override or metrics_meta.get("task")
    if not task_name:
        raise ValueError(f"Task name could not be determined for {run_path}. Pass task_override.")

    task_key = str(task_name).lower()

    # Load ground truth labels from real dataset (never masked test)
    dataset = load_task(task_key, data_dir=data_dir, use_masked_test=False)
    split_dataset = dataset["validation" if split == "valid" else split]
    ground_truth = np.array(split_dataset["label"])

    # Load logits
    logits_file = run_path / f"{split}_logits.npy"
    if not logits_file.exists():
        raise FileNotFoundError(f"Logits file not found: {logits_file}")

    logits = np.load(logits_file)

    # Verify matching row counts
    if len(logits) != len(ground_truth):
        raise ValueError(
            f"Row count mismatch for {split} split in {run_path}: "
            f"logits has {len(logits)} rows, but ground truth has {len(ground_truth)} rows."
        )

    # Calculate metrics
    eval_metrics = compute_classification_metrics(logits, ground_truth)

    report: Dict[str, Any] = {
        "task": task_key,
        "model": metrics_meta.get("model", "unknown"),
        "lr": metrics_meta.get("lr"),
        "seed": metrics_meta.get("seed"),
        "split": split,
        "test_macro_f1" if split == "test" else f"{split}_macro_f1": float(eval_metrics["macro_f1"]),
        "test_acc" if split == "test" else f"{split}_acc": float(eval_metrics["accuracy"]),
        "macro_f1": float(eval_metrics["macro_f1"]),
        "accuracy": float(eval_metrics["accuracy"]),
        "num_samples": int(len(ground_truth)),
    }

    if save_report:
        report_path = run_path / f"{split}_report.json"
        save_json(report, report_path)
        logger.info(f"Saved {split} report to {report_path}")

    return report


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate run logits against ground truth.")
    parser.add_argument("--run", type=str, required=True, help="Path to experiment run directory")
    parser.add_argument("--split", type=str, default="test", choices=["test", "validation", "valid"], help="Split to evaluate")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to data directory")
    parser.add_argument("--no_save", action="store_true", help="Do not save test_report.json")
    parser.add_argument("--task", type=str, default=None, help="Task name override")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    norm_split = "validation" if args.split == "valid" else args.split
    result = evaluate_run(
        run_dir=args.run,
        split=norm_split,
        data_dir=args.data_dir,
        save_report=not args.no_save,
        task_override=args.task,
    )
    print("\n--- Evaluation Result ---")
    print(f"Task: {result['task']}")
    print(f"Split: {result['split']}")
    print(f"Macro-F1: {result['macro_f1']:.4f}")
    print(f"Accuracy: {result['accuracy']:.4f}")
    print(f"Samples: {result['num_samples']}")

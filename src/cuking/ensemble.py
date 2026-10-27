"""Probability-averaging ensemble for IndoNLU classification benchmarks."""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from cuking.data import load_task
from cuking.metrics import compute_classification_metrics
from cuking.utils import UnionPath, load_json, save_json, setup_logger


logger = setup_logger("cuking.ensemble")


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute numerically stable softmax along the specified axis."""
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals, axis=axis, keepdims=True)


def select_ensemble_members(
    task: str,
    strategy: str,
    experiments_dir: UnionPath = "results/experiments",
    summary_csv: Optional[UnionPath] = "results/summary.csv",
    k: int = 3,
) -> List[Path]:
    """Select member runs for ensemble according to topk or best_per_model strategy.

    Ensemble folders (ensemble_*) and runs marked is_ensemble are strictly excluded.
    """
    exp_path = Path(experiments_dir)
    task_dir = exp_path / task.lower()

    # Collect candidate runs from summary_csv if available, else scan directory
    records: List[Dict[str, Any]] = []
    if summary_csv and Path(summary_csv).exists():
        df = pd.read_csv(summary_csv)
        if not df.empty and "task" in df.columns:
            task_df = df[df["task"].str.lower() == task.lower()].copy()
            for _, row in task_df.iterrows():
                run_dir_rel = str(row.get("run_dir", ""))
                run_dir_full = exp_path / run_dir_rel if not Path(run_dir_rel).is_absolute() else Path(run_dir_rel)
                is_ens = bool(row.get("is_ensemble", False)) or Path(run_dir_rel).name.startswith("ensemble_")
                records.append({
                    "run_dir": run_dir_full,
                    "rel_dir": run_dir_rel,
                    "model": str(row.get("model", "")),
                    "valid_macro_f1": float(row.get("valid_macro_f1", 0.0)),
                    "is_ensemble": is_ens,
                })

    if not records:
        # Fallback to directory scan
        if task_dir.exists():
            for folder in task_dir.iterdir():
                if folder.is_dir() and not folder.name.startswith("ensemble_"):
                    mf = folder / "metrics.json"
                    if mf.exists():
                        try:
                            m = load_json(mf)
                            is_ens = bool(m.get("is_ensemble", False)) or folder.name.startswith("ensemble_")
                            records.append({
                                "run_dir": folder,
                                "rel_dir": f"{task.lower()}/{folder.name}",
                                "model": str(m.get("model", "")),
                                "valid_macro_f1": float(m.get("valid_macro_f1", 0.0)),
                                "is_ensemble": is_ens,
                            })
                        except Exception as e:
                            logger.warning(f"Could not load {mf}: {e}")

    # Exclude any ensemble runs
    valid_candidates = [
        r for r in records
        if not r["is_ensemble"]
        and not Path(r["run_dir"]).name.startswith("ensemble_")
        and (Path(r["run_dir"]) / "valid_logits.npy").exists()
        and (Path(r["run_dir"]) / "test_logits.npy").exists()
    ]

    if not valid_candidates:
        raise ValueError(f"No valid non-ensemble runs found for task '{task}' in {experiments_dir}")

    if strategy == "topk":
        # Sort descending by valid_macro_f1
        valid_candidates.sort(key=lambda x: x["valid_macro_f1"], reverse=True)
        selected = valid_candidates[:k]
        return [Path(r["run_dir"]) for r in selected]

    elif strategy == "best_per_model":
        # Group by model family/name and pick the best run per family
        families: Dict[str, Dict[str, Any]] = {}
        for r in valid_candidates:
            model_key = r["model"]
            if (model_key not in families) or (r["valid_macro_f1"] > families[model_key]["valid_macro_f1"]):
                families[model_key] = r

        # Sort selected by valid_macro_f1 descending
        selected = sorted(families.values(), key=lambda x: x["valid_macro_f1"], reverse=True)
        return [Path(r["run_dir"]) for r in selected]

    else:
        raise ValueError(f"Unknown ensemble strategy: '{strategy}'. Supported: 'topk', 'best_per_model'")


def run_ensemble(
    task: str,
    name: str,
    member_dirs: List[Path],
    strategy: str = "custom",
    experiments_dir: UnionPath = "results/experiments",
    data_dir: str = "data",
) -> Dict[str, Any]:
    """Combine member predictions via softmax probability averaging and evaluate on validation set."""
    if not member_dirs:
        raise ValueError("Cannot create ensemble with empty member list.")

    # Guard: members must NOT be ensemble runs
    for m in member_dirs:
        m_path = Path(m)
        if m_path.name.startswith("ensemble_"):
            raise ValueError(f"Ensemble run '{m_path.name}' cannot be a member of another ensemble!")
        mf = m_path / "metrics.json"
        if mf.exists():
            data = load_json(mf)
            if data.get("is_ensemble"):
                raise ValueError(f"Run '{m_path.name}' is marked as is_ensemble and cannot be a member!")

    valid_probs_list: List[np.ndarray] = []
    test_probs_list: List[np.ndarray] = []
    expected_val_shape: Optional[Tuple[int, ...]] = None
    expected_test_shape: Optional[Tuple[int, ...]] = None

    member_rel_names: List[str] = []
    exp_path = Path(experiments_dir)

    for m_dir in member_dirs:
        m_path = Path(m_dir)
        vl_path = m_path / "valid_logits.npy"
        tl_path = m_path / "test_logits.npy"

        if not vl_path.exists():
            raise FileNotFoundError(f"Missing valid_logits.npy in member: {m_path}")
        if not tl_path.exists():
            raise FileNotFoundError(f"Missing test_logits.npy in member: {m_path}")

        vl = np.load(vl_path)
        tl = np.load(tl_path)

        # Dimension and shape checks
        if expected_val_shape is None:
            expected_val_shape = vl.shape
        elif vl.shape != expected_val_shape:
            raise ValueError(
                f"Validation logits shape mismatch: {m_path} has shape {vl.shape}, expected {expected_val_shape}"
            )

        if expected_test_shape is None:
            expected_test_shape = tl.shape
        elif tl.shape != expected_test_shape:
            raise ValueError(
                f"Test logits shape mismatch: {m_path} has shape {tl.shape}, expected {expected_test_shape}"
            )

        valid_probs_list.append(softmax(vl, axis=-1))
        test_probs_list.append(softmax(tl, axis=-1))

        try:
            rel_name = str(m_path.relative_to(exp_path))
        except ValueError:
            rel_name = m_path.name
        member_rel_names.append(rel_name)

    # Average softmax probabilities
    ensemble_valid_probs = np.mean(valid_probs_list, axis=0)
    ensemble_test_probs = np.mean(test_probs_list, axis=0)

    # Convert to log-probabilities for compatibility with downstream evaluators
    ensemble_valid_logits = np.log(np.clip(ensemble_valid_probs, 1e-12, 1.0))
    ensemble_test_logits = np.log(np.clip(ensemble_test_probs, 1e-12, 1.0))

    # Evaluate validation metrics ONLY
    dataset = load_task(task_name=task, data_dir=data_dir)
    val_labels = np.array(dataset["validation"]["label"])

    if len(val_labels) != ensemble_valid_probs.shape[0]:
        raise ValueError(
            f"Validation sample count mismatch: dataset has {len(val_labels)} labels, "
            f"logits have {ensemble_valid_probs.shape[0]} rows."
        )

    val_metrics = compute_classification_metrics(
        predictions=ensemble_valid_probs,
        labels=val_labels,
    )
    val_macro_f1 = float(val_metrics["macro_f1"])
    val_acc = float(val_metrics["accuracy"])

    # Output directory
    out_dir = exp_path / task.lower() / f"ensemble_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "valid_probs.npy", ensemble_valid_probs)
    np.save(out_dir / "test_probs.npy", ensemble_test_probs)
    np.save(out_dir / "valid_logits.npy", ensemble_valid_logits)
    np.save(out_dir / "test_logits.npy", ensemble_test_logits)

    metrics_data = {
        "task": task.lower(),
        "model": f"ensemble_{name}",
        "name": name,
        "strategy": strategy,
        "k": len(member_dirs),
        "members": member_rel_names,
        "valid_macro_f1": val_macro_f1,
        "valid_acc": val_acc,
        "is_ensemble": True,
    }
    save_json(metrics_data, out_dir / "metrics.json")

    logger.info(
        f"Ensemble '{name}' created ({strategy}, k={len(member_dirs)}): "
        f"valid macro-F1: {val_macro_f1:.4f}, valid acc: {val_acc:.4f}"
    )
    return metrics_data


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for ensemble."""
    parser = argparse.ArgumentParser(description="Probability-averaging ensemble for IndoNLU benchmarks.")
    parser.add_argument("--task", type=str, required=True, help="Task name (e.g. emot, smsa)")
    parser.add_argument("--name", type=str, required=True, help="Ensemble run name identifier")
    parser.add_argument("--runs", nargs="*", default=None, help="Explicit list of run directories to ensemble")
    parser.add_argument("--strategy", choices=["topk", "best_per_model"], default=None, help="Selection strategy")
    parser.add_argument("--k", type=int, default=3, help="Number of models to select for topk strategy")
    parser.add_argument("--experiments-dir", type=str, default="results/experiments", help="Path to experiments root")
    parser.add_argument("--summary-csv", type=str, default="results/summary.csv", help="Path to summary CSV")
    parser.add_argument("--data-dir", type=str, default="data", help="Path to data directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.runs:
        member_dirs = [Path(r) for r in args.runs]
        strategy_used = args.strategy or "explicit"
    elif args.strategy:
        member_dirs = select_ensemble_members(
            task=args.task,
            strategy=args.strategy,
            experiments_dir=args.experiments_dir,
            summary_csv=args.summary_csv,
            k=args.k,
        )
        strategy_used = args.strategy
    else:
        raise ValueError("Either --runs or --strategy must be specified.")

    print(f"\nBuilding ensemble '{args.name}' for task '{args.task}' ({strategy_used}):")
    for idx, m in enumerate(member_dirs, 1):
        print(f"  [{idx}/{len(member_dirs)}] {m}")

    result = run_ensemble(
        task=args.task,
        name=args.name,
        member_dirs=member_dirs,
        strategy=strategy_used,
        experiments_dir=args.experiments_dir,
        data_dir=args.data_dir,
    )
    print(f"\nEnsemble {args.name} complete: valid macro-F1 = {result['valid_macro_f1']:.4f}, acc = {result['valid_acc']:.4f}")

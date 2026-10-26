"""Resumable sweep runner for IndoNLU classification benchmarks.

STRICT RULE: sweep.py is FORBIDDEN from reading test labels, importing cuking.evaluate,
or invoking evaluation on the test split. All hyperparameter decisions and model
comparisons must be based exclusively on validation metrics.
"""

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

from cuking.aggregate import aggregate_results
from cuking.utils import UnionPath, load_json, setup_logger


logger = setup_logger("cuking.sweep")

# Default sweep grid and execution order according to PRD section 9.2
DEFAULT_TASKS = ["emot", "smsa"]
DEFAULT_GRID_ORDER: List[Tuple[str, float]] = [
    ("xlmr-large", 1e-5),
    ("indobert-large", 3e-5),
    ("xlmr-large", 2e-5),
    ("indobert-large", 4e-5),
    ("xlmr-large", 5e-6),
    ("indobert-large", 2e-5),
]


def format_lr(lr: float) -> str:
    """Format learning rate as clean string without unnecessary trailing zeros."""
    s = f"{lr:g}"
    if "e-0" in s:
        s = s.replace("e-0", "e-")
    return s


def make_results_zip(experiments_dir: UnionPath, zip_path: UnionPath) -> None:
    """Create or overwrite zip archive containing experiments directory."""
    exp_path = Path(experiments_dir).resolve()
    target_zip = Path(zip_path).resolve()
    target_zip.parent.mkdir(parents=True, exist_ok=True)
    base_name = str(target_zip.with_suffix(""))

    if exp_path.parent.name == "results":
        root_dir = exp_path.parent.parent
        base_dir = f"results/{exp_path.name}"
    else:
        root_dir = exp_path.parent
        base_dir = exp_path.name

    shutil.make_archive(
        base_name=base_name,
        format="zip",
        root_dir=str(root_dir),
        base_dir=str(base_dir),
    )


def build_sweep_plan(
    tasks: Optional[List[str]] = None,
    models: Optional[List[str]] = None,
    lrs: Optional[List[float]] = None,
    seeds: Optional[List[int]] = None,
    max_runs: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Construct ordered list of run configurations for the sweep.

    Order: All EmoT runs first, then SmSA.
    Within each task:
      1. xlmr-large 1e-5
      2. indobert-large 3e-5
      3. xlmr-large 2e-5
      4. indobert-large 4e-5
      5. xlmr-large 5e-6
      6. indobert-large 2e-5
    """
    selected_tasks = [t.lower() for t in (tasks or DEFAULT_TASKS)]
    selected_seeds = seeds or [42]

    # Build model-lr pairs
    if models is None and lrs is None:
        model_lr_pairs = list(DEFAULT_GRID_ORDER)
    elif models is not None and lrs is None:
        filter_models = set(m.lower() for m in models)
        model_lr_pairs = [(m, lr) for m, lr in DEFAULT_GRID_ORDER if m.lower() in filter_models]
    else:
        # Custom grid
        custom_models = models or ["xlmr-large", "indobert-large"]
        custom_lrs = lrs or [1e-5, 2e-5, 3e-5]
        model_lr_pairs = [(m, lr) for m in custom_models for lr in custom_lrs]

    plan: List[Dict[str, Any]] = []
    for task in selected_tasks:
        for model, lr in model_lr_pairs:
            for seed in selected_seeds:
                plan.append({
                    "task": task,
                    "model": model,
                    "lr": lr,
                    "seed": seed,
                })

    if max_runs is not None and max_runs > 0:
        plan = plan[:max_runs]

    return plan


def run_sweep(
    plan: List[Dict[str, Any]],
    experiments_dir: UnionPath = "results/experiments",
    summary_csv: UnionPath = "results/summary.csv",
    dry_run: bool = False,
    retry_failed: bool = False,
    is_smoke: bool = False,
) -> List[Dict[str, Any]]:
    """Execute sweep runs with resumption, failure isolation, and auto-archiving.

    Args:
        plan: Ordered list of run configuration dictionaries.
        experiments_dir: Root directory where experiment outputs reside.
        summary_csv: Path to summary CSV.
        dry_run: If True, only display execution plan without running.
        retry_failed: If True, re-execute runs that previously created failed.json.
        is_smoke: If True, pass --smoke flag to train subprocesses.

    Returns:
        List of results metadata for all runs.
    """
    exp_dir_path = Path(experiments_dir)
    total_runs = len(plan)
    results: List[Dict[str, Any]] = []

    print(f"\n=== Cuking Sweep Runner: {total_runs} Run(s) Planned ===")
    if dry_run:
        print("[DRY-RUN MODE] Listing planned runs in execution order:\n")

    for idx, run_cfg in enumerate(plan, 1):
        task = run_cfg["task"]
        model = run_cfg["model"]
        lr = float(run_cfg["lr"])
        seed = int(run_cfg["seed"])

        model_tag = "smoke_model" if is_smoke else model.replace("/", "_")
        run_name = f"{model_tag}_lr{lr}_s{seed}"
        run_dir = exp_dir_path / task / run_name
        metrics_file = run_dir / "metrics.json"
        failed_file = run_dir / "failed.json"

        lr_str = format_lr(lr)

        if dry_run:
            print(f"[{idx}/{total_runs}] {task} {model} lr={lr_str}")
            results.append({"status": "planned", "task": task, "model": model, "lr": lr, "seed": seed, "run_dir": str(run_dir)})
            continue

        # Resume check: Skip completed runs
        if metrics_file.exists():
            try:
                m = load_json(metrics_file)
                f1 = m.get("valid_macro_f1", 0.0)
                dur = m.get("train_time_s", 0.0)
                msg = f"[{idx}/{total_runs}] {task} {model} lr={lr_str} -> valid macro-F1 {f1:.4f} ({dur:.0f}s) [SKIPPED: metrics.json exists]"
            except Exception:
                msg = f"[{idx}/{total_runs}] {task} {model} lr={lr_str} -> SKIPPED (metrics.json exists)"
            print(msg)
            results.append({"status": "skipped_completed", "run_dir": str(run_dir)})
            continue

        # Resume check: Skip previously failed runs unless retry_failed
        if failed_file.exists() and not retry_failed:
            try:
                f_info = load_json(failed_file)
                reason = f_info.get("alasan", f_info.get("reason", "failed"))
                msg = f"[{idx}/{total_runs}] {task} {model} lr={lr_str} -> SKIPPED (failed.json exists: {reason}; use --retry-failed)"
            except Exception:
                msg = f"[{idx}/{total_runs}] {task} {model} lr={lr_str} -> SKIPPED (failed.json exists; use --retry-failed)"
            print(msg)
            results.append({"status": "skipped_failed", "run_dir": str(run_dir)})
            continue

        # Execute training as an isolated subprocess
        cmd = [
            sys.executable,
            "-m",
            "cuking.train",
            "--task", task,
            "--model", model,
            "--lr", str(lr),
            "--seed", str(seed),
            "--output_dir", str(run_dir),
        ]
        if is_smoke:
            cmd.append("--smoke")
        if retry_failed:
            cmd.append("--force")

        logger.info(f"Executing: {' '.join(cmd)}")
        proc = subprocess.run(cmd, check=False)

        # Evaluate outcome from disk
        if metrics_file.exists():
            try:
                m = load_json(metrics_file)
                f1 = m.get("valid_macro_f1", 0.0)
                dur = m.get("train_time_s", 0.0)
                print(f"[{idx}/{total_runs}] {task} {model} lr={lr_str} -> valid macro-F1 {f1:.4f} ({dur:.0f}s)")
                results.append({"status": "success", "run_dir": str(run_dir), "valid_macro_f1": f1})
            except Exception as read_err:
                print(f"[{idx}/{total_runs}] {task} {model} lr={lr_str} -> COMPLETED (could not read metrics: {read_err})")
                results.append({"status": "success", "run_dir": str(run_dir)})
        elif failed_file.exists():
            try:
                f_info = load_json(failed_file)
                reason = f_info.get("alasan", f_info.get("reason", "failed"))
                print(f"[{idx}/{total_runs}] {task} {model} lr={lr_str} -> FAILED ({reason})")
            except Exception:
                print(f"[{idx}/{total_runs}] {task} {model} lr={lr_str} -> FAILED (failed.json written)")
            results.append({"status": "failed", "run_dir": str(run_dir)})
        else:
            print(f"[{idx}/{total_runs}] {task} {model} lr={lr_str} -> FAILED (exit code {proc.returncode})")
            results.append({"status": "error", "exit_code": proc.returncode, "run_dir": str(run_dir)})

        # Update summary.csv after each run
        try:
            aggregate_results(experiments_dir=experiments_dir, output_csv=summary_csv)
        except Exception as agg_err:
            logger.error(f"Failed to update {summary_csv}: {agg_err}")

        # Update /kaggle/working/cuking_phase2_results.zip if running in Kaggle environment
        kaggle_working = Path("/kaggle/working")
        if kaggle_working.exists():
            try:
                make_results_zip(
                    experiments_dir=experiments_dir,
                    zip_path=kaggle_working / "cuking_phase2_results.zip",
                )
                logger.info("Updated /kaggle/working/cuking_phase2_results.zip")
            except Exception as zip_err:
                logger.error(f"Failed to create Kaggle results zip: {zip_err}")

    print("\n=== Sweep Execution Finished ===\n")
    return results


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for sweep runner."""
    parser = argparse.ArgumentParser(description="Run hyperparameter sweep on IndoNLU tasks.")
    parser.add_argument("--tasks", nargs="+", default=None, help="Tasks to sweep (e.g. emot smsa)")
    parser.add_argument("--models", nargs="+", default=None, help="Model aliases to sweep (e.g. xlmr-large indobert-large)")
    parser.add_argument("--lrs", nargs="+", type=float, default=None, help="Learning rates to sweep")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Seeds to sweep (default: 42)")
    parser.add_argument("--max-runs", type=int, default=None, help="Maximum number of runs to execute")
    parser.add_argument("--dry-run", action="store_true", help="Print sweep plan without training")
    parser.add_argument("--retry-failed", action="store_true", help="Re-run configurations that previously failed")
    parser.add_argument("--smoke", action="store_true", help="Run in smoke test mode (CPU, small subsets)")
    parser.add_argument("--experiments-dir", type=str, default="results/experiments", help="Target experiments folder")
    parser.add_argument("--summary-csv", type=str, default="results/summary.csv", help="Target summary CSV file")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()

    # Normalization of list arguments if user passed comma-separated strings
    raw_tasks = None
    if cli_args.tasks:
        raw_tasks = []
        for t in cli_args.tasks:
            raw_tasks.extend([item.strip() for item in t.split(",") if item.strip()])

    raw_models = None
    if cli_args.models:
        raw_models = []
        for m in cli_args.models:
            raw_models.extend([item.strip() for item in m.split(",") if item.strip()])

    sweep_plan = build_sweep_plan(
        tasks=raw_tasks,
        models=raw_models,
        lrs=cli_args.lrs,
        seeds=cli_args.seeds,
        max_runs=cli_args.max_runs,
    )

    run_sweep(
        plan=sweep_plan,
        experiments_dir=cli_args.experiments_dir,
        summary_csv=cli_args.summary_csv,
        dry_run=cli_args.dry_run,
        retry_failed=cli_args.retry_failed,
        is_smoke=cli_args.smoke,
    )

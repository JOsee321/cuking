"""Aggregate metrics from all experiment runs into a unified summary table."""

import argparse
from pathlib import Path
from typing import List, Optional
import pandas as pd
from cuking.utils import load_json, setup_logger


logger = setup_logger("cuking.aggregate")


def aggregate_results(
    experiments_dir: str | Path = "results/experiments",
    output_csv: Optional[str | Path] = "results/summary.csv",
) -> pd.DataFrame:
    """Collect all metrics.json files under experiments_dir into a single DataFrame.

    Args:
        experiments_dir: Root directory containing task experiment folders.
        output_csv: Path to save summary CSV. If None, CSV is not written.

    Returns:
        DataFrame containing sorted aggregated metrics.
    """
    exp_path = Path(experiments_dir)
    if not exp_path.exists():
        logger.warning(f"Experiments directory '{experiments_dir}' does not exist.")
        df = pd.DataFrame()
        if output_csv:
            Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_csv, index=False)
        return df

    metric_files: List[Path] = list(exp_path.glob("**/metrics.json"))
    records = []

    for mf in metric_files:
        try:
            data = load_json(mf)
            # Add relative folder path for traceability
            rel_dir = str(mf.parent.relative_to(exp_path))
            data["run_dir"] = rel_dir
            # Mark ensemble runs
            is_ens = (
                mf.parent.name.startswith("ensemble_")
                or "ensemble" in str(data.get("strategy", "")).lower()
                or bool(data.get("members"))
            )
            data["is_ensemble"] = bool(is_ens)
            records.append(data)
        except Exception as e:
            logger.error(f"Error loading {mf}: {e}")

    if not records:
        logger.info(f"No metrics.json files found in '{experiments_dir}'.")
        df = pd.DataFrame(columns=[
            "task", "model", "lr", "seed", "valid_macro_f1",
            "valid_acc", "best_epoch", "train_time_s", "is_ensemble", "smoke", "run_dir",
        ])
    else:
        df = pd.DataFrame(records)
        # Ensure column ordering
        primary_cols = [
            "task", "model", "lr", "seed", "valid_macro_f1",
            "valid_acc", "best_epoch", "train_time_s", "is_ensemble", "run_dir",
        ]
        other_cols = [c for c in df.columns if c not in primary_cols]
        ordered_cols = [c for c in primary_cols if c in df.columns] + other_cols
        df = df[ordered_cols]

        # Sort by task, then descending valid_macro_f1
        sort_cols = [c for c in ["task", "valid_macro_f1"] if c in df.columns]
        ascending = [True, False][:len(sort_cols)]
        df = df.sort_values(by=sort_cols, ascending=ascending).reset_index(drop=True)

    if output_csv:
        out_file = Path(output_csv)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_file, index=False)
        logger.info(f"Aggregated {len(df)} run(s) into '{output_csv}'.")

    return df


def print_top_rankings(df: pd.DataFrame) -> None:
    """Print leaderboard rankings per task based on validation macro-F1."""
    if df.empty or "task" not in df.columns:
        print("No results to rank.")
        return

    print("\n" + "=" * 76)
    print("TOP RANKINGS PER TASK (Ranked by Validation Macro-F1)")
    print("=" * 76)

    for task_name, group in df.groupby("task", sort=False):
        sorted_group = group.sort_values(by="valid_macro_f1", ascending=False).reset_index(drop=True)
        sorted_group.index = sorted_group.index + 1
        print(f"\n[Task: {task_name.upper()}]")
        cols = [c for c in ["model", "lr", "seed", "valid_macro_f1", "valid_acc", "best_epoch", "train_time_s"] if c in sorted_group.columns]
        print(sorted_group[cols].to_string())
    print("\n" + "=" * 76)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Aggregate experiment metrics.json files into summary.csv")
    parser.add_argument("--experiments-dir", type=str, default="results/experiments", help="Path to experiments folder")
    parser.add_argument("--output-csv", type=str, default="results/summary.csv", help="Path to output summary CSV")
    parser.add_argument("--top", action="store_true", help="Print top-ranked runs per task based on validation macro-F1")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summary_df = aggregate_results(
        experiments_dir=args.experiments_dir,
        output_csv=args.output_csv,
    )
    if not summary_df.empty:
        if args.top:
            print_top_rankings(summary_df)
        else:
            print("\n=== Experiment Results Summary ===")
            print(summary_df.to_string(index=False))
    else:
        print("No results found.")

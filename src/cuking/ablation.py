"""Generate ablation and performance comparison report across models and ensembles."""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from cuking.utils import UnionPath, load_json, setup_logger


logger = setup_logger("cuking.ablation")

ABLATION_WARNING_NOTE = (
    "CATATAN PENTING: Seleksi top-k dan evaluasi pada validation yang sama bersifat "
    "optimistis, sehingga kenaikan metrik yang kecil perlu diverifikasi lebih lanjut "
    "dengan multi-seed atau pengujian statistik independen."
)


def format_markdown_table(df: pd.DataFrame) -> str:
    """Format DataFrame as standard Markdown table without external dependencies."""
    if df.empty:
        return ""
    headers = list(df.columns)
    rows = df.values.tolist()
    header_line = "| " + " | ".join(str(h) for h in headers) + " |"
    sep_line = "| " + " | ".join("---" for _ in headers) + " |"
    row_lines = ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join([header_line, sep_line] + row_lines)


def generate_ablation_report(
    experiments_dir: UnionPath = "results/experiments",
    output_csv: Optional[UnionPath] = "results/ablation.csv",
    output_readme: Optional[UnionPath] = "results/ablation_README.md",
    tasks: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Generate ablation comparison table containing best single, family baselines, and ensembles.

    Columns: task, name, valid_macro_f1, delta_vs_best_single, members.

    Args:
        experiments_dir: Root directory containing task experiment folders.
        output_csv: Target path to save the ablation CSV.
        output_readme: Target path to save the markdown README with method notes.
        tasks: Optional list of tasks to analyze. If None, all detected tasks are analyzed.

    Returns:
        DataFrame containing formatted ablation results.
    """
    exp_path = Path(experiments_dir)
    if not exp_path.exists():
        logger.warning(f"Experiments path '{experiments_dir}' does not exist.")
        df = pd.DataFrame(columns=["task", "name", "valid_macro_f1", "delta_vs_best_single", "members"])
        if output_csv:
            Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_csv, index=False)
        return df

    # Detect task folders
    if tasks is None:
        task_folders = [p.name for p in exp_path.iterdir() if p.is_dir()]
    else:
        task_folders = [t.lower() for t in tasks]

    all_rows: List[Dict[str, Any]] = []

    for task_name in sorted(task_folders):
        t_dir = exp_path / task_name
        if not t_dir.exists():
            continue

        single_runs: List[Dict[str, Any]] = []
        ensemble_runs: List[Dict[str, Any]] = []

        # Scan runs under task directory
        for run_folder in sorted(t_dir.iterdir()):
            if not run_folder.is_dir():
                continue

            mf = run_folder / "metrics.json"
            if not mf.exists():
                continue

            try:
                data = load_json(mf)
            except Exception as e:
                logger.warning(f"Could not load {mf}: {e}")
                continue

            rel_name = run_folder.name
            is_ens = (
                rel_name.startswith("ensemble_")
                or bool(data.get("is_ensemble", False))
                or "ensemble" in str(data.get("strategy", "")).lower()
                or bool(data.get("members"))
            )

            entry = {
                "folder_name": rel_name,
                "model": str(data.get("model", rel_name)),
                "valid_macro_f1": float(data.get("valid_macro_f1", 0.0)),
                "valid_acc": float(data.get("valid_acc", 0.0)),
                "members": data.get("members", [rel_name]),
                "raw": data,
            }

            if is_ens:
                ensemble_runs.append(entry)
            else:
                single_runs.append(entry)

        if not single_runs and not ensemble_runs:
            continue

        # 1. Best single model run
        best_single = None
        best_single_f1 = 0.0
        if single_runs:
            single_runs.sort(key=lambda x: x["valid_macro_f1"], reverse=True)
            best_single = single_runs[0]
            best_single_f1 = best_single["valid_macro_f1"]

            all_rows.append({
                "task": task_name,
                "name": f"best_single ({best_single['model']})",
                "valid_macro_f1": best_single_f1,
                "delta_vs_best_single": 0.0,
                "members": best_single["folder_name"],
            })

            # 2. Best per model family (xlmr-large, indobert-large, etc.)
            families: Dict[str, Dict[str, Any]] = {}
            for r in single_runs:
                fam = r["model"]
                if (fam not in families) or (r["valid_macro_f1"] > families[fam]["valid_macro_f1"]):
                    families[fam] = r

            # Sort families descending by valid_macro_f1
            for fam_name in sorted(families.keys()):
                fam_run = families[fam_name]
                delta = fam_run["valid_macro_f1"] - best_single_f1
                all_rows.append({
                    "task": task_name,
                    "name": f"best_{fam_name}",
                    "valid_macro_f1": fam_run["valid_macro_f1"],
                    "delta_vs_best_single": delta,
                    "members": fam_run["folder_name"],
                })

        # 3. All existing ensembles
        # Sort ensemble runs descending by valid_macro_f1
        ensemble_runs.sort(key=lambda x: x["valid_macro_f1"], reverse=True)
        for ens in ensemble_runs:
            ens_f1 = ens["valid_macro_f1"]
            delta = ens_f1 - best_single_f1 if best_single else 0.0
            if isinstance(ens["members"], list):
                members_str = "; ".join(str(m).replace("\\", "/") for m in ens["members"])
            else:
                members_str = str(ens["members"]).replace("\\", "/")
            all_rows.append({
                "task": task_name,
                "name": ens["folder_name"],
                "valid_macro_f1": ens_f1,
                "delta_vs_best_single": delta,
                "members": members_str,
            })

    columns = ["task", "name", "valid_macro_f1", "delta_vs_best_single", "members"]
    df = pd.DataFrame(all_rows, columns=columns)

    if output_csv:
        csv_path = Path(output_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved ablation report to '{output_csv}'.")

    if output_readme:
        readme_path = Path(output_readme)
        readme_path.parent.mkdir(parents=True, exist_ok=True)
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write("# Ablation and Ensemble Performance Report\n\n")
            f.write(f"> **Catatan Metodologi**: {ABLATION_WARNING_NOTE}\n\n")
            f.write("## Ringkasan Performa Validasi\n\n")
            f.write(format_markdown_table(df))
            f.write("\n")
        logger.info(f"Saved ablation README to '{output_readme}'.")

    return df


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for ablation report generator."""
    parser = argparse.ArgumentParser(description="Generate ablation report comparing single models and ensembles.")
    parser.add_argument("--experiments-dir", type=str, default="results/experiments", help="Path to experiments directory")
    parser.add_argument("--output-csv", type=str, default="results/ablation.csv", help="Target ablation CSV file")
    parser.add_argument("--output-readme", type=str, default="results/ablation_README.md", help="Target markdown summary")
    parser.add_argument("--tasks", nargs="+", default=None, help="List of tasks to analyze")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("\n" + "=" * 70)
    print(ABLATION_WARNING_NOTE)
    print("=" * 70 + "\n")

    report_df = generate_ablation_report(
        experiments_dir=args.experiments_dir,
        output_csv=args.output_csv,
        output_readme=args.output_readme,
        tasks=args.tasks,
    )
    if not report_df.empty:
        print("=== ABLATION REPORT SUMMARY ===")
        print(report_df.to_string(index=False))
    else:
        print("No experiment results found to generate ablation report.")

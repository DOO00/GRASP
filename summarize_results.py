from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METRICS = ["ACC", "NMI", "F1", "ARI", "CS"]


def load_summary_rows(root: Path) -> list[dict]:
    rows: list[dict] = []
    for summary_file in sorted(root.rglob("summary.json")):
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
        metrics = summary.get("best_metrics", {})
        row = {
            "dataset": summary.get("dataset", ""),
            "method": summary.get("method", ""),
            "seed": summary.get("seed", ""),
            "best_epoch": summary.get("best_epoch", ""),
            "selection_metric": summary.get("selection_metric", ""),
            "selection_score": summary.get("selection_score", ""),
            "eval_feature": summary.get("eval_feature", ""),
            "total_time_sec": summary.get("total_time_sec", ""),
            "run_dir": str(summary_file.parent),
        }
        for metric in METRICS:
            row[metric] = metrics.get(metric, 0.0)
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "dataset",
        "method",
        "seed",
        "best_epoch",
        *METRICS,
        "selection_metric",
        "selection_score",
        "eval_feature",
        "total_time_sec",
        "run_dir",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# GRASP Results\n\n")
        f.write("| dataset | method | seed | best epoch | ACC | NMI | F1 | ARI | CS |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            metric_text = " | ".join(f"{float(row.get(metric, 0.0)):.2f}" for metric in METRICS)
            f.write(
                f"| {row.get('dataset', '')} | {row.get('method', '')} | "
                f"{row.get('seed', '')} | {row.get('best_epoch', '')} | {metric_text} |\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize completed clustering runs")
    parser.add_argument("--results-root", required=True)
    args = parser.parse_args()

    root = Path(args.results_root)
    root.mkdir(parents=True, exist_ok=True)
    rows = load_summary_rows(root)
    if not rows:
        raise SystemExit(f"no summary.json files found under {root}")

    csv_path = root / "final_metrics.csv"
    report_path = root / "final_report.md"
    write_csv(csv_path, rows)
    write_markdown(report_path, rows)
    print(f"wrote {csv_path} and {report_path}")


if __name__ == "__main__":
    main()

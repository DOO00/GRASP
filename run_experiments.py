from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from data import DATASETS


def run(cmd: list[str], dry_run: bool = False) -> None:
    print("+ " + " ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Train one or more datasets and generate a final metric report",
        allow_abbrev=False,
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", default=str(root / "results"))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--method", default="GRASP")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-summary", action="store_true")
    args, train_extra = parser.parse_known_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for dataset in args.datasets:
        for seed in args.seeds:
            cmd = [
                args.python,
                str(root / "train.py"),
                "--data-root",
                args.data_root,
                "--output-root",
                str(output_root),
                "--dataset",
                dataset,
                "--seed",
                str(seed),
                "--device",
                args.device,
                "--epochs",
                str(args.epochs),
                "--method",
                args.method,
                *train_extra,
            ]
            run(cmd, dry_run=args.dry_run)

    if not args.skip_summary:
        run(
            [args.python, str(root / "summarize_results.py"), "--results-root", str(output_root)],
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()

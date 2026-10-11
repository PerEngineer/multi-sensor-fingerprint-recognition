from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fingerprint_exp.plots import plot_training_history, plot_validation_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize validation-only model selection.")
    parser.add_argument("--outputs-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for summary_path in sorted(args.outputs_root.glob("*/seed_*/training_summary.json")):
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        if payload.get("test_evaluated") is not False:
            raise RuntimeError(f"Unexpected test state in {summary_path}")
        history_path = summary_path.parent / "history.csv"
        history = pd.read_csv(history_path)
        best = history.loc[history["val_top1"].idxmax()]
        rows.append(
            {
                "model": payload["model"],
                "seed": payload["seed"],
                "best_epoch": int(payload["best_epoch"]),
                "epochs_run": int(len(history)),
                "best_val_top1": float(payload["best_val_top1"]),
                "best_val_top5": float(best["val_top5"]),
                "best_val_loss": float(best["val_loss"]),
                "training_seconds": float(payload["training_seconds"]),
                "summary_path": str(summary_path.resolve()),
            }
        )
        plot_training_history(history_path, args.output_dir / payload["model"])
    if not rows:
        raise SystemExit("No completed validation-only training summaries were found")
    frame = pd.DataFrame(rows).sort_values(
        ["best_val_top1", "best_val_top5"], ascending=False
    )
    frame.to_csv(args.output_dir / "validation_model_comparison.csv", index=False)
    plot_validation_comparison(frame, args.output_dir)
    print(
        frame[
            [
                "model",
                "best_epoch",
                "epochs_run",
                "best_val_top1",
                "best_val_top5",
                "training_seconds",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()

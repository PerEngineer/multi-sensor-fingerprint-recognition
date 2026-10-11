from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fingerprint_exp.plots import plot_model_comparison, plot_sensor_model_comparison


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize fixed-test model results.")
    parser.add_argument("--outputs-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for metrics_path in sorted(args.outputs_root.glob("**/final_test/test_metrics.json")):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        payload["metrics_path"] = str(metrics_path)
        rows.append(payload)
    if not rows:
        raise SystemExit("No final test metrics were found")
    frame = pd.DataFrame(rows).sort_values(["top1", "macro_f1"], ascending=False)
    frame.to_csv(args.output_dir / "model_comparison.csv", index=False)
    plot_model_comparison(frame, args.output_dir)

    sensor_rows = []
    for metrics_path in sorted(args.outputs_root.glob("**/final_test/sensor_metrics.csv")):
        sensor_frame = pd.read_csv(metrics_path)
        sensor_frame["model"] = metrics_path.parents[2].name
        sensor_rows.append(sensor_frame)
    if sensor_rows:
        sensor_comparison = pd.concat(sensor_rows, ignore_index=True)
        sensor_comparison.to_csv(
            args.output_dir / "sensor_model_comparison.csv", index=False
        )
        plot_sensor_model_comparison(sensor_comparison, args.output_dir)
    print(frame[["model", "seed", "top1", "top5", "macro_f1", "verification_auc", "verification_eer"]].to_string(index=False))


if __name__ == "__main__":
    main()

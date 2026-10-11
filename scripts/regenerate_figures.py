from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fingerprint_exp.plots import (
    plot_confusion,
    plot_roc,
    plot_score_distributions,
    plot_sensor_accuracy,
    plot_training_history,
)
from scripts.evaluate import verification_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate report figures only from saved final-test artifacts."
    )
    parser.add_argument("--outputs-root", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_dirs = sorted(args.outputs_root.glob("*/seed_*/final_test"))
    if not result_dirs:
        raise SystemExit("No saved final-test artifacts were found")
    for result_dir in result_dirs:
        metrics = json.loads((result_dir / "test_metrics.json").read_text())
        predictions = pd.read_csv(result_dir / "predictions.csv")
        embeddings = np.load(result_dir / "embeddings.npy")
        matrix = np.load(result_dir / "confusion_matrix.npy")
        sensor_metrics = pd.read_csv(result_dir / "sensor_metrics.csv")
        genuine, impostor = verification_scores(
            embeddings,
            predictions["target"].to_numpy(),
            metrics["impostor_pair_count"],
            metrics["seed"],
        )
        pair_labels = np.concatenate(
            [np.ones(len(genuine)), np.zeros(len(impostor))]
        )
        pair_scores = np.concatenate([genuine, impostor])
        fpr, tpr, _ = roc_curve(pair_labels, pair_scores)
        auc_value = roc_auc_score(pair_labels, pair_scores)
        assert abs(auc_value - metrics["verification_auc"]) < 1e-12
        plot_confusion(matrix, result_dir)
        plot_roc(
            fpr,
            tpr,
            metrics["verification_auc"],
            metrics["verification_eer"],
            result_dir,
        )
        plot_score_distributions(genuine, impostor, result_dir)
        plot_sensor_accuracy(sensor_metrics, result_dir)
        history_path = result_dir.parent / "history.csv"
        plot_training_history(history_path, result_dir)
        print(f"PASS regenerated figures: {metrics['model']}")


if __name__ == "__main__":
    main()

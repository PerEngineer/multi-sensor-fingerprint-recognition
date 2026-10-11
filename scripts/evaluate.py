from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fingerprint_exp.data import create_test_loader
from fingerprint_exp.models import create_model, forward_embeddings
from fingerprint_exp.plots import (
    plot_confusion,
    plot_roc,
    plot_score_distributions,
    plot_sensor_accuracy,
    plot_training_history,
)
from fingerprint_exp.utils import atomic_json_dump, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-shot final test evaluation.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--negative-pairs", type=int, default=9540)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--allow-final-test", action="store_true")
    return parser.parse_args()


def verification_scores(embeddings, targets, negative_pairs, seed):
    embeddings = F.normalize(torch.from_numpy(embeddings), dim=1).numpy()
    rng = np.random.default_rng(seed)
    by_class = {}
    for index, target in enumerate(targets):
        by_class.setdefault(int(target), []).append(index)

    genuine = []
    for indices in by_class.values():
        for left in range(len(indices)):
            for right in range(left + 1, len(indices)):
                genuine.append(float(np.dot(embeddings[indices[left]], embeddings[indices[right]])))

    impostor = []
    n_samples = len(targets)
    seen = set()
    while len(impostor) < negative_pairs:
        left = int(rng.integers(0, n_samples))
        right = int(rng.integers(0, n_samples))
        if left == right or targets[left] == targets[right]:
            continue
        pair = (min(left, right), max(left, right))
        if pair in seen:
            continue
        seen.add(pair)
        impostor.append(float(np.dot(embeddings[left], embeddings[right])))
    return np.asarray(genuine), np.asarray(impostor)


def main() -> None:
    args = parse_args()
    if not args.allow_final_test:
        raise SystemExit(
            "Refusing to read the final test set. Re-run with --allow-final-test only "
            "after all model and hyperparameter choices are frozen."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = create_model(
        checkpoint["model_name"], checkpoint["num_classes"], pretrained=False
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    loader = create_test_loader(
        args.manifest,
        args.data_root,
        checkpoint["image_size"],
        args.batch_size,
        args.workers,
    )

    targets = []
    predictions = []
    confidences = []
    top5_predictions = []
    embeddings = []
    metadata = []
    start = time.time()
    with torch.inference_mode():
        for batch in tqdm(loader, desc="final test"):
            images = batch["image"].to(device, non_blocking=True)
            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)
            confidence, prediction = probabilities.max(dim=1)
            top5 = probabilities.topk(5, dim=1).indices
            embedding = forward_embeddings(model, images)
            targets.extend(batch["target"].numpy().tolist())
            predictions.extend(prediction.cpu().numpy().tolist())
            confidences.extend(confidence.cpu().numpy().tolist())
            top5_predictions.extend(top5.cpu().numpy().tolist())
            embeddings.append(embedding.cpu().float().numpy())
            for index in range(len(batch["relative_path"])):
                metadata.append(
                    {
                        "relative_path": batch["relative_path"][index],
                        "sensor": batch["sensor"][index],
                        "subject": batch["subject"][index],
                        "finger": int(batch["finger"][index]),
                        "capture": int(batch["capture"][index]),
                    }
                )

    targets_array = np.asarray(targets)
    predictions_array = np.asarray(predictions)
    top5_array = np.asarray(top5_predictions)
    embeddings_array = np.concatenate(embeddings, axis=0)
    top1 = accuracy_score(targets_array, predictions_array)
    top5 = float(np.mean(np.any(top5_array == targets_array[:, None], axis=1)))
    macro_f1 = f1_score(targets_array, predictions_array, average="macro")
    matrix = confusion_matrix(
        targets_array, predictions_array, labels=np.arange(checkpoint["num_classes"])
    )

    prediction_frame = pd.DataFrame(metadata)
    prediction_frame["target"] = targets_array
    prediction_frame["prediction"] = predictions_array
    prediction_frame["confidence"] = confidences
    prediction_frame["correct"] = targets_array == predictions_array
    prediction_frame.to_csv(args.output_dir / "predictions.csv", index=False)
    np.save(args.output_dir / "embeddings.npy", embeddings_array)
    np.save(args.output_dir / "confusion_matrix.npy", matrix)

    sensor_frame = (
        prediction_frame.groupby("sensor")
        .agg(count=("correct", "size"), correct=("correct", "sum"), top1=("correct", "mean"))
        .reset_index()
    )
    sensor_frame.to_csv(args.output_dir / "sensor_metrics.csv", index=False)

    genuine, impostor = verification_scores(
        embeddings_array, targets_array, args.negative_pairs, args.seed
    )
    pair_labels = np.concatenate([np.ones(len(genuine)), np.zeros(len(impostor))])
    pair_scores = np.concatenate([genuine, impostor])
    fpr, tpr, thresholds = roc_curve(pair_labels, pair_scores)
    auc_value = roc_auc_score(pair_labels, pair_scores)
    fnr = 1.0 - tpr
    eer_index = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float((fpr[eer_index] + fnr[eer_index]) / 2.0)
    eer_threshold = float(thresholds[eer_index])

    metrics = {
        "status": "COMPLETED",
        "model": checkpoint["model_name"],
        "seed": checkpoint["seed"],
        "checkpoint_epoch": checkpoint["epoch"],
        "test_count": len(targets),
        "top1": float(top1),
        "top5": top5,
        "macro_f1": float(macro_f1),
        "verification_auc": float(auc_value),
        "verification_eer": eer,
        "verification_eer_threshold": eer_threshold,
        "genuine_pair_count": int(len(genuine)),
        "impostor_pair_count": int(len(impostor)),
        "evaluation_seconds": time.time() - start,
        "test_manifest": str(args.manifest.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
    }
    atomic_json_dump(metrics, args.output_dir / "test_metrics.json")
    plot_confusion(matrix, args.output_dir)
    plot_roc(fpr, tpr, auc_value, eer, args.output_dir)
    plot_score_distributions(genuine, impostor, args.output_dir)
    plot_sensor_accuracy(sensor_frame, args.output_dir)
    history_path = args.checkpoint.parent / "history.csv"
    if history_path.exists():
        plot_training_history(history_path, args.output_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()


from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fingerprint_exp.data import create_loaders
from fingerprint_exp.models import MODEL_ALIASES, create_model
from fingerprint_exp.utils import atomic_json_dump, seed_everything, topk_correct


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a fingerprint classifier.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--model", required=True, choices=sorted(MODEL_ALIASES))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--pretrained-weights", type=Path)
    return parser.parse_args()


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_count = 0
    correct_1 = 0
    correct_5 = 0
    iterator = tqdm(loader, leave=False, desc="train" if training else "val")
    for batch in iterator:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, targets)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        batch_size = targets.size(0)
        top1, top5 = topk_correct(logits.detach(), targets, topk=(1, 5))
        total_loss += float(loss.detach().item()) * batch_size
        total_count += batch_size
        correct_1 += top1
        correct_5 += top5
        iterator.set_postfix(loss=f"{total_loss / total_count:.4f}")
    return {
        "loss": total_loss / total_count,
        "top1": correct_1 / total_count,
        "top5": correct_5 / total_count,
        "count": total_count,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = create_loaders(
        args.manifest,
        args.data_root,
        args.image_size,
        args.batch_size,
        args.workers,
    )
    model = create_model(
        args.model,
        loaders.num_classes,
        pretrained=not args.no_pretrained,
        weights_path=str(args.pretrained_weights) if args.pretrained_weights else None,
    ).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup_epochs = min(2, max(1, args.epochs // 5))
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs),
            CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - warmup_epochs)),
        ],
        milestones=[warmup_epochs],
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    config = vars(args).copy()
    config.update(
        {
            "manifest": str(args.manifest.resolve()),
            "data_root": str(args.data_root.resolve()),
            "output_dir": str(args.output_dir.resolve()),
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "num_classes": loaders.num_classes,
            "pretrained_weights": (
                str(args.pretrained_weights.resolve())
                if args.pretrained_weights
                else None
            ),
        }
    )
    atomic_json_dump(config, args.output_dir / "config.json")

    history_path = args.output_dir / "history.csv"
    fieldnames = [
        "epoch",
        "lr",
        "train_loss",
        "train_top1",
        "train_top5",
        "val_loss",
        "val_top1",
        "val_top5",
        "elapsed_seconds",
    ]
    best_val = -1.0
    best_epoch = -1
    epochs_without_improvement = 0
    start_time = time.time()
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch(
                model, loaders.train, criterion, device, optimizer, scaler
            )
            with torch.inference_mode():
                val_metrics = run_epoch(model, loaders.val, criterion, device)
            row = {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train_loss": train_metrics["loss"],
                "train_top1": train_metrics["top1"],
                "train_top5": train_metrics["top5"],
                "val_loss": val_metrics["loss"],
                "val_top1": val_metrics["top1"],
                "val_top5": val_metrics["top5"],
                "elapsed_seconds": time.time() - start_time,
            }
            writer.writerow(row)
            handle.flush()
            print(json.dumps(row))

            checkpoint = {
                "model_name": args.model,
                "num_classes": loaders.num_classes,
                "image_size": args.image_size,
                "seed": args.seed,
                "epoch": epoch,
                "val_top1": val_metrics["top1"],
                "state_dict": model.state_dict(),
            }
            torch.save(checkpoint, args.output_dir / "last.pt")
            if val_metrics["top1"] > best_val:
                best_val = val_metrics["top1"]
                best_epoch = epoch
                epochs_without_improvement = 0
                torch.save(checkpoint, args.output_dir / "best.pt")
            else:
                epochs_without_improvement += 1
            scheduler.step()
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    summary = {
        "status": "COMPLETED",
        "model": args.model,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_val_top1": best_val,
        "training_seconds": time.time() - start_time,
        "test_evaluated": False,
    }
    atomic_json_dump(summary, args.output_dir / "training_summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

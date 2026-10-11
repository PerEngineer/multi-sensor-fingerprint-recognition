from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fingerprint_exp.data import create_loaders
from fingerprint_exp.models import MODEL_ALIASES, create_model, forward_embeddings


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke-test data and model interfaces.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--weights-dir", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = create_loaders(
        args.manifest,
        args.data_root,
        image_size=224,
        batch_size=2,
        workers=0,
    )
    batch = next(iter(loaders.train))
    images = batch["image"].to(device)
    assert images.shape == (2, 3, 224, 224)
    assert loaders.num_classes == 636
    print(f"PASS data batch: shape={tuple(images.shape)}, classes={loaders.num_classes}")

    for model_name in MODEL_ALIASES:
        weights_path = None
        if args.weights_dir:
            weights_path = args.weights_dir / f"{MODEL_ALIASES[model_name]}.safetensors"
            if not weights_path.exists():
                raise FileNotFoundError(weights_path)
        model = create_model(
            model_name,
            num_classes=loaders.num_classes,
            pretrained=False,
            weights_path=str(weights_path) if weights_path else None,
        ).to(device)
        model.eval()
        with torch.inference_mode():
            logits = model(images)
            embeddings = forward_embeddings(model, images)
        assert logits.shape == (2, loaders.num_classes)
        assert embeddings.ndim == 2 and embeddings.shape[0] == 2
        print(
            f"PASS model={model_name}, local_pretrained={bool(weights_path)}, "
            f"logits={tuple(logits.shape)}, embeddings={tuple(embeddings.shape)}"
        )
        del model, logits, embeddings
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

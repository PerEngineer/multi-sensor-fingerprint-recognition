from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def atomic_json_dump(payload: dict, destination: str | Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(destination)


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def topk_correct(logits: torch.Tensor, targets: torch.Tensor, topk=(1, 5)):
    max_k = min(max(topk), logits.shape[1])
    _, prediction = logits.topk(max_k, dim=1, largest=True, sorted=True)
    prediction = prediction.t()
    correct = prediction.eq(targets.view(1, -1).expand_as(prediction))
    return [int(correct[: min(k, max_k)].reshape(-1).sum().item()) for k in topk]


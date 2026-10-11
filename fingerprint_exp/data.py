from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


class SquarePad:
    """Pad a PIL image to a square without distorting ridge geometry."""

    def __init__(self, fill: int = 255) -> None:
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        side = max(width, height)
        pad_left = (side - width) // 2
        pad_top = (side - height) // 2
        pad_right = side - width - pad_left
        pad_bottom = side - height - pad_top
        return ImageOps.expand(
            image,
            border=(pad_left, pad_top, pad_right, pad_bottom),
            fill=self.fill,
        )


def build_transform(image_size: int, train: bool):
    common = [
        transforms.Grayscale(num_output_channels=3),
        SquarePad(fill=255),
    ]
    if train:
        spatial = [
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.82, 1.0),
                ratio=(0.9, 1.1),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.RandomRotation(
                degrees=12,
                interpolation=InterpolationMode.BILINEAR,
                fill=(255, 255, 255),
            ),
            transforms.ColorJitter(brightness=0.12, contrast=0.18),
        ]
    else:
        spatial = [
            transforms.Resize(
                (image_size, image_size), interpolation=InterpolationMode.BICUBIC
            )
        ]
    return transforms.Compose(
        common
        + spatial
        + [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


class FingerprintDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        data_root: str | Path,
        split: str,
        image_size: int = 224,
        train_transform: bool = False,
    ) -> None:
        manifest = pd.read_csv(manifest_path)
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {split}")
        self.frame = manifest.loc[manifest["split"] == split].reset_index(drop=True)
        if self.frame.empty:
            raise ValueError(f"No rows found for split={split}")
        self.data_root = Path(data_root).resolve()
        if not self.data_root.is_dir():
            raise NotADirectoryError(f"Dataset root does not exist: {self.data_root}")
        self.transform = build_transform(image_size=image_size, train=train_transform)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        relative_path = Path(str(row["relative_path"]))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(
                f"Manifest path must be relative to the dataset root: {relative_path}"
            )
        path = self.data_root / relative_path
        if not path.is_file():
            raise FileNotFoundError(
                f"Image not found under dataset root: {relative_path} "
                f"(root={self.data_root})"
            )
        with Image.open(path) as image:
            image = image.convert("L")
            tensor = self.transform(image)
        return {
            "image": tensor,
            "target": torch.tensor(int(row["class_id"]), dtype=torch.long),
            "sensor": str(row["sensor"]),
            "subject": str(row["subject"]),
            "finger": int(row["finger"]),
            "capture": int(row["capture"]),
            "relative_path": str(row["relative_path"]),
        }


@dataclass(frozen=True)
class LoaderBundle:
    train: DataLoader
    val: DataLoader
    num_classes: int


def create_loaders(
    manifest_path: str | Path,
    data_root: str | Path,
    image_size: int,
    batch_size: int,
    workers: int,
) -> LoaderBundle:
    train_dataset = FingerprintDataset(
        manifest_path, data_root, "train", image_size, train_transform=True
    )
    val_dataset = FingerprintDataset(
        manifest_path, data_root, "val", image_size, train_transform=False
    )
    manifest = pd.read_csv(manifest_path, usecols=["class_id"])
    num_classes = int(manifest["class_id"].nunique())
    common = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=False, **common)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **common)
    return LoaderBundle(train=train_loader, val=val_loader, num_classes=num_classes)


def create_test_loader(
    manifest_path: str | Path,
    data_root: str | Path,
    image_size: int,
    batch_size: int,
    workers: int,
) -> DataLoader:
    dataset = FingerprintDataset(
        manifest_path, data_root, "test", image_size, train_transform=False
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


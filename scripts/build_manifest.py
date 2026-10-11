from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fingerprint_exp.utils import sha256_file


FILE_PATTERN = re.compile(r"^(?P<finger>[1-6])_(?P<capture>[1-8])\.bmp$", re.I)
EXPECTED_SENSORS = ["AES2501", "FPR620", "FT-2BU", "URU4000B", "ZY202-B"]
TEST_CAPTURE_ORDER = [8, 7, 5, 4, 3, 2, 1, 6]
VAL_CAPTURE_ORDER = [6, 7, 4, 3, 2, 1, 5, 8]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a deterministic, de-duplicated split manifest."
    )
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--hash-files",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Hash every image and remove exact duplicates (enabled by default).",
    )
    parser.add_argument("--seed", type=int, default=20261010)
    return parser.parse_args()


def stable_key(path: str, seed: int, phase: str) -> str:
    value = f"{seed}:{phase}:{path}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def allocate_quotas(
    class_sizes: pd.Series,
    target_total: int,
    phase_offset: int,
) -> dict[int, int]:
    quotas = {
        int(class_id): max(1, int(round(size * 0.15)))
        for class_id, size in class_sizes.items()
    }
    class_order = sorted(
        quotas,
        key=lambda class_id: (
            int(class_sizes.loc[class_id]),
            (class_id + phase_offset) % len(quotas),
        ),
    )
    cursor = 0
    while sum(quotas.values()) > target_total:
        class_id = class_order[cursor % len(class_order)]
        minimum_quota = min(5, int(class_sizes.loc[class_id]) - 2)
        if quotas[class_id] > minimum_quota:
            quotas[class_id] -= 1
        cursor += 1
    while sum(quotas.values()) < target_total:
        class_id = class_order[cursor % len(class_order)]
        if quotas[class_id] < int(class_sizes.loc[class_id]) - 2:
            quotas[class_id] += 1
        cursor += 1
    return quotas


def select_balanced_indices(
    group: pd.DataFrame,
    quota: int,
    sensors: list[str],
    seed: int,
    phase: str,
    excluded: set[int] | None = None,
    stable_path_prefix: str = "",
) -> set[int]:
    excluded = excluded or set()
    capture_order = TEST_CAPTURE_ORDER if phase == "test" else VAL_CAPTURE_ORDER
    capture_rank = {capture: rank for rank, capture in enumerate(capture_order)}
    selected: set[int] = set()

    def candidate_key(index: int):
        row = group.loc[index]
        relative_path = str(row["relative_path"])
        split_key_path = (
            f"{stable_path_prefix}/{relative_path}"
            if stable_path_prefix
            else relative_path
        )
        return (
            capture_rank[int(row["capture"])],
            stable_key(split_key_path, seed, phase),
        )

    # Quotas are at least five for this dataset, so every sensor is represented.
    for sensor in sensors:
        candidates = [
            int(index)
            for index in group.index
            if index not in excluded
            and index not in selected
            and group.at[index, "sensor"] == sensor
        ]
        if candidates and len(selected) < quota:
            selected.add(min(candidates, key=candidate_key))

    rotation = (
        int(group["class_id"].iloc[0]) + (0 if phase == "test" else 2)
    ) % len(sensors)
    sensor_cycle = sensors[rotation:] + sensors[:rotation]
    while len(selected) < quota:
        added = False
        for sensor in sensor_cycle:
            candidates = [
                int(index)
                for index in group.index
                if index not in excluded
                and index not in selected
                and group.at[index, "sensor"] == sensor
            ]
            if candidates:
                selected.add(min(candidates, key=candidate_key))
                added = True
                if len(selected) == quota:
                    break
        if not added:
            raise RuntimeError(
                f"Unable to allocate {phase} quota for class {group['class_id'].iloc[0]}"
            )
    return selected


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sensors = sorted(path.name for path in data_root.iterdir() if path.is_dir())
    if sensors != sorted(EXPECTED_SENSORS):
        raise RuntimeError(f"Unexpected sensor set: {sensors}")

    rows = []
    rejected = []
    for sensor in sensors:
        sensor_dir = data_root / sensor
        for subject_dir in sorted(path for path in sensor_dir.iterdir() if path.is_dir()):
            subject = subject_dir.name
            if not re.fullmatch(r"\d{3}", subject):
                rejected.append(str(subject_dir))
                continue
            for image_path in sorted(subject_dir.glob("*.bmp")):
                match = FILE_PATTERN.fullmatch(image_path.name)
                if not match:
                    rejected.append(str(image_path))
                    continue
                finger = int(match.group("finger"))
                capture = int(match.group("capture"))
                class_id = (int(subject) - 1) * 6 + (finger - 1)
                # Manifest paths are always relative to --data-root so the same
                # directory can be passed unchanged to training and evaluation.
                relative_path = image_path.relative_to(data_root).as_posix()
                rows.append(
                    {
                        "relative_path": relative_path,
                        "sensor": sensor,
                        "subject": subject,
                        "finger": finger,
                        "capture": capture,
                        "class_id": class_id,
                        "class_name": f"{subject}_f{finger}",
                        "bytes": image_path.stat().st_size,
                    }
                )

    original = pd.DataFrame(rows).sort_values(
        ["class_id", "sensor", "capture"], ignore_index=True
    )
    if not args.hash_files:
        raise RuntimeError("Hashing is required for leakage-safe splitting")
    original["sha256"] = [
        sha256_file(data_root / relative_path)
        for relative_path in original["relative_path"]
    ]

    excluded_records = []
    keep_indices = set()
    duplicate_groups = []
    for digest, group in original.groupby("sha256", sort=True):
        if len(group) == 1:
            keep_indices.add(int(group.index[0]))
            continue
        conflicting = group["class_id"].nunique() > 1
        duplicate_groups.append(
            {
                "sha256": digest,
                "count": int(len(group)),
                "conflicting_labels": bool(conflicting),
                "paths": group["relative_path"].tolist(),
                "class_ids": sorted(
                    int(value) for value in group["class_id"].unique()
                ),
            }
        )
        if conflicting:
            for index in group.index:
                record = original.loc[index].to_dict()
                record["exclusion_reason"] = "conflicting_duplicate_labels"
                excluded_records.append(record)
        else:
            canonical_index = int(
                group.sort_values("relative_path", kind="stable").index[0]
            )
            keep_indices.add(canonical_index)
            for index in group.index:
                if int(index) == canonical_index:
                    continue
                record = original.loc[index].to_dict()
                record["exclusion_reason"] = "exact_duplicate"
                excluded_records.append(record)

    cleaned = original.loc[sorted(keep_indices)].copy().reset_index(drop=True)
    if cleaned["sha256"].duplicated().any():
        raise RuntimeError("De-duplication failed: duplicate hashes remain")

    class_sizes = cleaned.groupby("class_id").size()
    test_total = int(round(len(cleaned) * 0.15))
    val_total = int(round(len(cleaned) * 0.15))
    test_quotas = allocate_quotas(class_sizes, test_total, phase_offset=0)
    val_quotas = allocate_quotas(
        class_sizes, val_total, phase_offset=len(class_sizes) // 2
    )

    cleaned["split"] = "train"
    for class_id, group in cleaned.groupby("class_id", sort=True):
        test_indices = select_balanced_indices(
            group,
            test_quotas[int(class_id)],
            sensors,
            args.seed,
            "test",
            stable_path_prefix=data_root.name,
        )
        val_indices = select_balanced_indices(
            group,
            val_quotas[int(class_id)],
            sensors,
            args.seed,
            "val",
            excluded=test_indices,
            stable_path_prefix=data_root.name,
        )
        cleaned.loc[list(test_indices), "split"] = "test"
        cleaned.loc[list(val_indices), "split"] = "val"

    errors = []
    if len(original) != 25_440:
        errors.append(f"Expected 25440 raw images, found {len(original)}")
    if cleaned["class_id"].nunique() != 636:
        errors.append(f"Expected 636 classes, found {cleaned['class_id'].nunique()}")
    if cleaned["relative_path"].duplicated().any():
        errors.append("Duplicate paths exist in the cleaned manifest")
    if cleaned["sha256"].duplicated().any():
        errors.append("Duplicate image hashes remain in the cleaned manifest")
    class_split_counts = cleaned.groupby(["class_id", "split"]).size().unstack(
        fill_value=0
    )
    if (class_split_counts == 0).any().any():
        errors.append("At least one class is absent from a split")

    split_counts = cleaned["split"].value_counts().to_dict()
    sensor_split_counts = (
        cleaned.groupby(["sensor", "split"])
        .size()
        .unstack(fill_value=0)
        .to_dict("index")
    )
    exclusion_frame = pd.DataFrame(excluded_records)
    exclusion_counts = (
        exclusion_frame["exclusion_reason"].value_counts().to_dict()
        if not exclusion_frame.empty
        else {}
    )
    audit = {
        "data_root": str(data_root),
        "raw_image_count": int(len(original)),
        "clean_image_count": int(len(cleaned)),
        "excluded_image_count": int(len(exclusion_frame)),
        "exclusion_counts": {
            key: int(value) for key, value in exclusion_counts.items()
        },
        "num_classes": int(cleaned["class_id"].nunique()),
        "num_subjects": int(cleaned["subject"].nunique()),
        "num_fingers_per_subject": 6,
        "sensors": sensors,
        "split_counts": {key: int(value) for key, value in split_counts.items()},
        "split_percentages": {
            key: round(100.0 * value / len(cleaned), 6)
            for key, value in split_counts.items()
        },
        "sensor_split_counts": sensor_split_counts,
        "rejected_paths": rejected,
        "file_hashing_enabled": True,
        "raw_duplicate_hash_group_count": len(duplicate_groups),
        "conflicting_label_duplicate_group_count": sum(
            group["conflicting_labels"] for group in duplicate_groups
        ),
        "clean_duplicate_hash_group_count": 0,
        "cross_split_duplicate_group_count": 0,
        "seed": args.seed,
        "errors": errors,
        "status": "PASS" if not errors else "FAIL",
    }

    cleaned.to_csv(output_dir / "manifest.csv", index=False)
    exclusion_frame.to_csv(output_dir / "excluded_images.csv", index=False)
    (output_dir / "integrity_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    (output_dir / "duplicate_hash_groups.json").write_text(
        json.dumps(duplicate_groups, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

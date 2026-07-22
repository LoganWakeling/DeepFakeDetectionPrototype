from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image


IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
FIELDNAMES = [
    "sample_id",
    "image_path",
    "label",
    "class_name",
    "split",
    "source_dataset",
    "quality_label",
    "generator",
    "generator_type",
    "manipulation_type",
    "image_format",
    "width",
    "height",
    "channels",
    "file_size_bytes",
    "bytes_per_pixel",
]
STRATIFY_COLUMNS = ("class_name", "generator_type", "quality_label", "source_dataset")

DEFAULT_CONFIG: dict[str, Any] = {
    "real_names": ["real", "authentic", "original", "pristine", "bonafide", "bona_fide"],
    "fake_names": ["fake", "deepfake", "synthetic", "manipulated", "generated"],
    "quality_names": {"high": "high", "hq": "high", "low": "low", "lq": "low"},
    "split_names": {"train": "train", "training": "train", "test": "test", "testing": "test"},
    "generators": {
        "simswap": {"generator": "SimSwap", "generator_type": "gan", "manipulation_type": "face_swap"},
        "deepfacelab": {"generator": "DeepFaceLab", "generator_type": "autoencoder", "manipulation_type": "face_swap"},
        "stable_diffusion": {"generator": "Stable Diffusion", "generator_type": "diffusion", "manipulation_type": "full_synthesis"},
        "stablediffusion": {"generator": "Stable Diffusion", "generator_type": "diffusion", "manipulation_type": "full_synthesis"},
        "ldm": {"generator": "LDM", "generator_type": "diffusion", "manipulation_type": "full_synthesis"},
    },
    "source_datasets": {},
}


def _key(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def load_config(path: Path | None) -> dict[str, Any]:
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    if path is None:
        return config
    with path.open(encoding="utf-8") as handle:
        custom = json.load(handle)
    for name in ("real_names", "fake_names"):
        if name in custom:
            config[name] = custom[name]
    for name in ("quality_names", "split_names", "generators", "source_datasets"):
        config[name].update(custom.get(name, {}))
    return config


def first_match(parts: list[str], mapping: dict[str, Any]) -> Any:
    normalized = {_key(name): value for name, value in mapping.items()}
    for part in reversed(parts):
        if _key(part) in normalized:
            return normalized[_key(part)]
    return ""


def infer_metadata(path: Path, root: Path, config: dict[str, Any]) -> dict[str, Any]:
    parts = list(path.relative_to(root).parts[:-1])
    keys = {_key(part) for part in parts}
    real = keys.intersection(map(_key, config["real_names"]))
    fake = keys.intersection(map(_key, config["fake_names"]))
    if real and fake:
        raise ValueError("path contains both real and fake directory labels")
    label = 0 if real else 1 if fake else ""
    generator = first_match(parts, config["generators"])
    if label == 0:
        generator = {"generator": "real", "generator_type": "real", "manipulation_type": "none"}
    elif not generator:
        generator = {}
    return {
        "label": label,
        "class_name": "real" if label == 0 else "fake" if label == 1 else "",
        "split": first_match(parts, config["split_names"]),
        "source_dataset": first_match(parts, config["source_datasets"]) or root.name,
        "quality_label": first_match(parts, config["quality_names"]),
        "generator": generator.get("generator", ""),
        "generator_type": generator.get("generator_type", ""),
        "manipulation_type": generator.get("manipulation_type", ""),
    }


def assign_stratified_splits(rows: list[dict[str, Any]], test_fraction: float) -> None:
    """Assign stable train/test splits while preserving important metadata groups."""
    strata: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[column] or "<missing>" for column in STRATIFY_COLUMNS)
        strata.setdefault(key, []).append(row)

    for members in strata.values():
        unassigned = [row for row in members if not row["split"]]
        existing_test = sum(row["split"] == "test" for row in members)
        if len(members) < 2:
            desired_test = 0
        else:
            desired_test = min(len(members) - 1, max(1, round(len(members) * test_fraction)))
        needed_test = min(len(unassigned), max(0, desired_test - existing_test))
        ordered = sorted(
            unassigned,
            key=lambda row: hashlib.sha256(row["sample_id"].encode("ascii")).hexdigest(),
        )
        for index, row in enumerate(ordered):
            row["split"] = "test" if index < needed_test else "train"


def inspect_image(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        width, height = image.size
        channels = len(image.getbands())
        image_format = image.format or path.suffix.lstrip(".").upper()
    size = path.stat().st_size
    return {
        "image_format": image_format,
        "width": width,
        "height": height,
        "channels": channels,
        "file_size_bytes": size,
        "bytes_per_pixel": round(size / (width * height), 6),
    }


def create_rows(root: Path, config: dict[str, Any], test_fraction: float) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    paths = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    for path in paths:
        relative = path.relative_to(root).as_posix()
        sample_id = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:16]
        try:
            metadata = infer_metadata(path, root, config)
            properties = inspect_image(path)
        except (OSError, ValueError) as error:
            warnings.append(f"Skipped {relative}: {error}")
            continue
        if metadata["label"] == "":
            warnings.append(f"Missing real/fake label for {relative}")
        if metadata["label"] == 1 and not metadata["generator_type"]:
            warnings.append(f"Missing generator metadata for {relative}")
        if not metadata["quality_label"]:
            warnings.append(f"Missing quality label for {relative}")
        rows.append(
            {
                "sample_id": sample_id,
                "image_path": str(path.resolve()),
                **metadata,
                **properties,
            }
        )
    assign_stratified_splits(rows, test_fraction)
    return rows, warnings


def _counts(rows: list[dict[str, Any]], column: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for row in rows:
        value = str(row[column]) if row[column] != "" else "<missing>"
        result[value] = result.get(value, 0) + 1
    return dict(sorted(result.items()))


def build_summary(root: Path, rows: list[dict[str, Any]], warnings: list[str]) -> dict[str, Any]:
    distributions = {
        column: _counts(rows, column)
        for column in (
            "split",
            "class_name",
            "source_dataset",
            "quality_label",
            "generator",
            "generator_type",
            "manipulation_type",
            "image_format",
        )
    }
    split_distributions = {
        split: {
            column: _counts([row for row in rows if row["split"] == split], column)
            for column in STRATIFY_COLUMNS
        }
        for split in ("train", "test")
    }
    strata: dict[str, int] = {}
    for row in rows:
        key = " | ".join(str(row[column] or "<missing>") for column in STRATIFY_COLUMNS)
        strata[key] = strata.get(key, 0) + 1
    return {
        "dataset_root": str(root),
        "total_images": len(rows),
        "total_file_size_bytes": sum(row["file_size_bytes"] for row in rows),
        "average_width": round(sum(row["width"] for row in rows) / len(rows), 2),
        "average_height": round(sum(row["height"] for row in rows) / len(rows), 2),
        "distributions": distributions,
        "distributions_by_split": split_distributions,
        "stratification": {
            "columns": list(STRATIFY_COLUMNS),
            "number_of_strata": len(strata),
            "singleton_strata": sorted(key for key, count in strata.items() if count == 1),
        },
        "warning_count": len(warnings),
        "warnings": warnings,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a deepfake dataset manifest from an image directory.")
    parser.add_argument("dataset_root", type=Path, help="Folder to scan recursively.")
    parser.add_argument("--output", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument("--summary", type=Path, help="Summary JSON path (default: manifest name with .summary.json).")
    parser.add_argument("--config", type=Path, help="Optional JSON file containing dataset-specific folder mappings.")
    parser.add_argument("--test-fraction", type=float, default=0.2, help="Deterministic test fraction when split folders are absent (default: 0.2).")
    parser.add_argument("--strict", action="store_true", help="Fail if a label or fake generator type cannot be inferred.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.dataset_root.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Dataset root is not a directory: {root}")
    if not 0 < args.test_fraction < 1:
        raise SystemExit("--test-fraction must be between 0 and 1")

    rows, warnings = create_rows(root, load_config(args.config), args.test_fraction)
    if not rows:
        raise SystemExit(f"No readable images found under {root}")
    if args.strict and warnings:
        for warning in warnings:
            print(f"WARNING: {warning}", file=sys.stderr)
        raise SystemExit("Manifest validation failed in strict mode; add folder mappings to the config.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    summary_path = args.summary or args.output.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(build_summary(root, rows, warnings), handle, indent=2, sort_keys=True)
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    print(f"Wrote {len(rows)} rows to {args.output} and dataset summary to {summary_path} ({len(warnings)} warnings).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

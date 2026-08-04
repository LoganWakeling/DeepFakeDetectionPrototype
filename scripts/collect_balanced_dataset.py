from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from feature_extractors.arcface_embedding import create_face_analyzer
from feature_extractors.feature_extraction import detect_and_crop_largest_face


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}
FAKE_FAMILIES = ("autoencoder", "gan", "diffusion")
FIELDNAMES = [
    "sample_id",
    "parent_id",
    "group_id",
    "image_path",
    "label",
    "class_name",
    "source_dataset",
    "source_url",
    "license",
    "source_path",
    "original_path",
    "standardized_face_path",
    "source_type",
    "generator",
    "generator_type",
    "manipulation_type",
    "container_video_id",
    "segment_id",
    "frame_timestamp_seconds",
    "quality_label",
    "quality_variant",
    "jpeg_quality",
    "face_crop_applied",
    "image_format",
    "width",
    "height",
    "channels",
    "file_size_bytes",
    "sha256",
    "split",
]


@dataclass(frozen=True)
class Candidate:
    source_index: int
    source_path: Path
    source_dataset: str
    source_url: str
    license: str
    label: int
    generator: str
    generator_type: str
    manipulation_type: str
    source_type: str
    timestamp: float | None = None
    segment_id: str = ""
    container_video_id: str = ""

    @property
    def family(self) -> str:
        return "real" if self.label == 0 else self.generator_type

    @property
    def identity(self) -> str:
        timestamp = "" if self.timestamp is None else f"@{self.timestamp:.6f}"
        return f"{self.source_dataset}|{self.source_path.resolve()}{timestamp}"


def load_config(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        config = json.load(file)
    sources = config.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Configuration must contain a non-empty 'sources' list.")

    for index, source in enumerate(sources):
        required = {"path", "source_dataset", "label", "media_type"}
        missing = required - set(source)
        if missing:
            raise ValueError(f"Source {index} is missing fields: {sorted(missing)}")
        if source["label"] not in {0, 1}:
            raise ValueError(f"Source {index} label must be 0 or 1.")
        if source["media_type"] not in {"images", "videos"}:
            raise ValueError(f"Source {index} media_type must be 'images' or 'videos'.")
        if source["label"] == 1 and source.get("generator_type") not in FAKE_FAMILIES:
            raise ValueError(
                f"Source {index} fake generator_type must be one of {FAKE_FAMILIES}."
            )
        if source["label"] == 1:
            missing_fake = {"generator", "manipulation_type"} - set(source)
            if missing_fake:
                raise ValueError(f"Source {index} is missing fake metadata: {sorted(missing_fake)}")
        if source.get("quality_label") != "high":
            raise ValueError(
                f"Source {index} must be explicitly reviewed and configured with quality_label='high'."
            )
    return sources


def inspect_high_quality_image(path: Path, min_face_source_size: int) -> bool:
    try:
        with Image.open(path) as image:
            width, height = image.size
            image.verify()
        return min(width, height) >= min_face_source_size
    except (OSError, ValueError):
        return False


def separated_timestamps(
    duration: float,
    count: int,
    minimum_spacing: float,
    rng: random.Random,
    boundary_margin: float = 0.5,
) -> list[float]:
    """Choose random timestamps with a guaranteed minimum distance."""
    if duration <= 2 * boundary_margin or count <= 0:
        return []
    available = max(0.0, duration - 2 * boundary_margin)
    if count > 1 and available < (count - 1) * minimum_spacing:
        count = int(available // minimum_spacing) + 1
    if count == 1:
        return [rng.uniform(boundary_margin, duration - boundary_margin)]

    # Divide the usable interval into ordered slots, leaving minimum_spacing
    # between their starts, then add bounded random jitter inside each slot.
    slack = available - (count - 1) * minimum_spacing
    jitter_width = slack / count
    return [
        boundary_margin + index * minimum_spacing + index * jitter_width + rng.random() * jitter_width
        for index in range(count)
    ]


def video_timestamps(
    video_path: Path,
    source: dict[str, Any],
    rng: random.Random,
) -> list[tuple[float, str]]:
    capture = cv2.VideoCapture(str(video_path))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if fps <= 0 or frame_count <= 0:
        return []
    duration = frame_count / fps
    video_key = hashlib.sha256(str(video_path.resolve()).encode("utf-8")).hexdigest()[:10]
    segment_duration = source.get("segment_duration_seconds")
    margin = float(source.get("segment_boundary_margin_seconds", 1.0))

    if segment_duration:
        segment_duration = float(segment_duration)
        frames_per_segment = int(source.get("frames_per_segment", 1))
        timestamps: list[tuple[float, str]] = []
        segment_count = int(duration // segment_duration)
        for segment_number in range(segment_count):
            start = segment_number * segment_duration
            usable_start = start + margin
            usable_end = min(start + segment_duration - margin, duration - margin)
            if usable_end <= usable_start:
                continue
            local = separated_timestamps(
                usable_end - usable_start,
                frames_per_segment,
                float(source.get("minimum_frame_spacing_seconds", 2.0)),
                rng,
                boundary_margin=0.0,
            )
            segment_id = f"{video_path.stem}-{video_key}_segment_{segment_number:05d}"
            timestamps.extend((usable_start + value, segment_id) for value in local)
        return timestamps

    count = int(source.get("frames_per_video", 10))
    spacing = float(source.get("minimum_frame_spacing_seconds", 5.0))
    return [(value, f"{video_path.stem}-{video_key}_frame_{index:05d}") for index, value in enumerate(
        separated_timestamps(duration, count, spacing, rng, boundary_margin=margin)
    )]


def source_metadata(source: dict[str, Any]) -> dict[str, Any]:
    label = int(source["label"])
    return {
        "label": label,
        "generator": "real" if label == 0 else source["generator"],
        "generator_type": "real" if label == 0 else source["generator_type"],
        "manipulation_type": "none" if label == 0 else source["manipulation_type"],
    }


def discover_candidates(
    sources: list[dict[str, Any]],
    min_source_size: int,
    rng: random.Random,
) -> tuple[list[Candidate], list[str]]:
    candidates: list[Candidate] = []
    warnings: list[str] = []
    for source_index, source in enumerate(sources):
        root = Path(source["path"]).expanduser().resolve()
        if not root.exists():
            warnings.append(f"Missing source path: {root}")
            continue
        metadata = source_metadata(source)
        if source["media_type"] == "images":
            paths = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
            for path in paths:
                if inspect_high_quality_image(path, int(source.get("minimum_source_size", min_source_size))):
                    candidates.append(
                        Candidate(
                            source_index=source_index,
                            source_path=path,
                            source_dataset=source["source_dataset"],
                            source_url=source.get("source_url", ""),
                            license=source.get("license", ""),
                            source_type="still_image",
                            **metadata,
                        )
                    )
        else:
            paths = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)
            for path in paths:
                for timestamp, segment_id in video_timestamps(path, source, rng):
                    candidates.append(
                        Candidate(
                            source_index=source_index,
                            source_path=path,
                            source_dataset=source["source_dataset"],
                            source_url=source.get("source_url", ""),
                            license=source.get("license", ""),
                            source_type="video_frame",
                            timestamp=timestamp,
                            segment_id=segment_id,
                            container_video_id=path.stem,
                            **metadata,
                        )
                    )
    return candidates, warnings


def balanced_across_sources(
    candidates: list[Candidate],
    count: int,
    rng: random.Random,
) -> list[Candidate]:
    """Round-robin sampling prevents one configured dataset dominating a family."""
    buckets: dict[int, list[Candidate]] = {}
    for candidate in candidates:
        buckets.setdefault(candidate.source_index, []).append(candidate)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    selected: list[Candidate] = []
    source_order = sorted(buckets)
    rng.shuffle(source_order)
    while len(selected) < count:
        progressed = False
        for source_index in source_order:
            if buckets[source_index] and len(selected) < count:
                selected.append(buckets[source_index].pop())
                progressed = True
        if not progressed:
            break
    return selected


def choose_balanced_dataset(
    candidates: list[Candidate],
    per_fake_family: int,
    real_count: int,
    rng: random.Random,
) -> list[Candidate]:
    selected: list[Candidate] = []
    requirements = {"real": real_count, **{family: per_fake_family for family in FAKE_FAMILIES}}
    for family, required in requirements.items():
        available = [candidate for candidate in candidates if candidate.family == family]
        chosen = balanced_across_sources(available, required, rng)
        if len(chosen) < required:
            raise ValueError(
                f"Requested {required} {family} samples, but only {len(chosen)} eligible samples are available."
            )
        selected.extend(chosen)
    rng.shuffle(selected)
    return selected


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def category_parts(candidate: Candidate) -> tuple[str, ...]:
    return ("real",) if candidate.label == 0 else ("fake", candidate.generator_type)


def save_original(candidate: Candidate, output_root: Path, parent_id: str) -> Path:
    destination_dir = output_root / "originals" / Path(*category_parts(candidate))
    destination_dir.mkdir(parents=True, exist_ok=True)
    if candidate.source_type == "still_image":
        destination = destination_dir / f"{parent_id}{candidate.source_path.suffix.lower()}"
        shutil.copy2(candidate.source_path, destination)
        return destination

    destination = destination_dir / f"{parent_id}.png"
    capture = cv2.VideoCapture(str(candidate.source_path))
    capture.set(cv2.CAP_PROP_POS_MSEC, float(candidate.timestamp) * 1000.0)
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None or not cv2.imwrite(str(destination), frame):
        raise OSError(
            f"Could not extract frame from {candidate.source_path} at {candidate.timestamp:.3f}s"
        )
    return destination


def standardize_face(
    original_path: Path,
    candidate: Candidate,
    output_root: Path,
    parent_id: str,
    analyzer: Any,
    size: int,
) -> tuple[Path, bool]:
    image = cv2.imread(str(original_path), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"Could not read collected original: {original_path}")
    crop, detected = detect_and_crop_largest_face(image, analyzer)
    interpolation = cv2.INTER_AREA if max(crop.shape[:2]) >= size else cv2.INTER_CUBIC
    standardized = cv2.resize(crop, (size, size), interpolation=interpolation)
    destination = (
        output_root / "derived" / "standardized_faces" / Path(*category_parts(candidate)) / f"{parent_id}.png"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), standardized):
        raise OSError(f"Could not save standardized face: {destination}")
    return destination, detected


def image_properties(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        width, height = image.size
        channels = len(image.getbands())
        image_format = image.format or path.suffix.lstrip(".").upper()
    return {
        "image_format": image_format,
        "width": width,
        "height": height,
        "channels": channels,
        "file_size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def save_quality_variants(
    candidate: Candidate,
    original_path: Path,
    standardized_path: Path,
    output_root: Path,
    parent_id: str,
    crop_applied: bool,
) -> list[dict[str, Any]]:
    class_name = "real" if candidate.label == 0 else "fake"
    variants = (("original", None), ("jpeg_q75", 75), ("jpeg_q50", 50), ("jpeg_q25", 25))
    rows: list[dict[str, Any]] = []
    with Image.open(standardized_path) as standardized:
        rgb_image = standardized.convert("RGB")
        for variant, jpeg_quality in variants:
            extension = ".png" if jpeg_quality is None else ".jpg"
            destination = (
                output_root
                / "derived"
                / "controlled_quality"
                / variant
                / Path(*category_parts(candidate))
                / f"{parent_id}{extension}"
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            if jpeg_quality is None:
                shutil.copy2(standardized_path, destination)
            else:
                rgb_image.save(destination, format="JPEG", quality=jpeg_quality, subsampling=0)
            sample_id = f"{parent_id}-{variant}"
            rows.append(
                {
                    "sample_id": sample_id,
                    "parent_id": parent_id,
                    "group_id": candidate.segment_id or parent_id,
                    "image_path": str(destination.resolve()),
                    "label": candidate.label,
                    "class_name": class_name,
                    "source_dataset": candidate.source_dataset,
                    "source_url": candidate.source_url,
                    "license": candidate.license,
                    "source_path": str(candidate.source_path.resolve()),
                    "original_path": str(original_path.resolve()),
                    "standardized_face_path": str(standardized_path.resolve()),
                    "source_type": candidate.source_type,
                    "generator": candidate.generator,
                    "generator_type": candidate.generator_type,
                    "manipulation_type": candidate.manipulation_type,
                    "container_video_id": candidate.container_video_id,
                    "segment_id": candidate.segment_id,
                    "frame_timestamp_seconds": (
                        "" if candidate.timestamp is None else round(candidate.timestamp, 6)
                    ),
                    "quality_label": variant,
                    "quality_variant": variant,
                    "jpeg_quality": "" if jpeg_quality is None else jpeg_quality,
                    "face_crop_applied": crop_applied,
                    **image_properties(destination),
                    "split": "",
                }
            )
    return rows


def assign_grouped_splits(
    rows: list[dict[str, Any]],
    validation_fraction: float,
    test_fraction: float,
    rng: random.Random,
) -> None:
    groups: dict[str, dict[str, Any]] = {}
    for row in rows:
        groups.setdefault(row["group_id"], row)
    strata: dict[tuple[str, str, str], list[str]] = {}
    for group_id, row in groups.items():
        key = (row["class_name"], row["generator_type"], row["source_dataset"])
        strata.setdefault(key, []).append(group_id)

    assignments: dict[str, str] = {}
    for group_ids in strata.values():
        rng.shuffle(group_ids)
        test_count = round(len(group_ids) * test_fraction)
        validation_count = round(len(group_ids) * validation_fraction)
        for index, group_id in enumerate(group_ids):
            if index < test_count:
                assignments[group_id] = "test"
            elif index < test_count + validation_count:
                assignments[group_id] = "validation"
            else:
                assignments[group_id] = "train"
    for row in rows:
        row["split"] = assignments[row["group_id"]]


def write_exclusions(path: Path, exclusions: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["source_path", "reason"])
        writer.writeheader()
        writer.writerows(exclusions)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect a balanced, provenance-labelled high-quality face dataset."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--originals-per-class", type=int, default=1500)
    parser.add_argument("--minimum-source-size", type=int, default=256)
    parser.add_argument("--standardized-size", type=int, default=512)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    existing_manifest = args.output_dir / "metadata" / "manifest.csv"
    if existing_manifest.exists():
        raise SystemExit(
            f"Refusing to overwrite existing collection manifest: {existing_manifest}. "
            "Use a new output directory."
        )
    if args.originals_per_class <= 0 or args.originals_per_class % len(FAKE_FAMILIES):
        raise SystemExit("--originals-per-class must be positive and divisible by 3.")
    if args.standardized_size <= 0:
        raise SystemExit("--standardized-size must be positive.")
    if not 0 <= args.validation_fraction < 1 or not 0 <= args.test_fraction < 1:
        raise SystemExit("Split fractions must be between 0 and 1.")
    if args.validation_fraction + args.test_fraction >= 1:
        raise SystemExit("Validation and test fractions must sum to less than 1.")
    per_fake_family = args.originals_per_class // len(FAKE_FAMILIES)

    rng = random.Random(args.random_state)
    sources = load_config(args.config)
    candidates, warnings = discover_candidates(sources, args.minimum_source_size, rng)
    selected = choose_balanced_dataset(
        candidates, per_fake_family, args.originals_per_class, rng
    )

    metadata_dir = args.output_dir / "metadata"
    licenses_dir = metadata_dir / "licenses"
    licenses_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "originals" / "auxiliary" / "face2face").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "originals" / "auxiliary" / "faceswap").mkdir(parents=True, exist_ok=True)
    manifest_path = metadata_dir / "manifest.csv"
    analyzer = create_face_analyzer()
    rows: list[dict[str, Any]] = []
    original_hashes: set[str] = set()
    exclusions: list[dict[str, str]] = []
    for candidate in selected:
        parent_id = hashlib.sha256(candidate.identity.encode("utf-8")).hexdigest()[:16]
        try:
            original_path = save_original(candidate, args.output_dir, parent_id)
            original_hash = sha256(original_path)
            if original_hash in original_hashes:
                original_path.unlink()
                exclusions.append(
                    {"source_path": str(candidate.source_path), "reason": "exact duplicate"}
                )
                continue
            original_hashes.add(original_hash)
            standardized_path, crop_applied = standardize_face(
                original_path,
                candidate,
                args.output_dir,
                parent_id,
                analyzer,
                args.standardized_size,
            )
            rows.extend(
                save_quality_variants(
                    candidate,
                    original_path,
                    standardized_path,
                    args.output_dir,
                    parent_id,
                    crop_applied,
                )
            )
        except OSError as error:
            exclusions.append({"source_path": str(candidate.source_path), "reason": str(error)})

    expected_rows = args.originals_per_class * 2 * 4
    if len(rows) != expected_rows:
        write_exclusions(metadata_dir / "exclusions.csv", exclusions)
        raise RuntimeError(
            f"Expected {expected_rows} variant rows, produced {len(rows)}. "
            "Review metadata/exclusions.csv, add eligible sources, and rerun into an empty output directory."
        )
    assign_grouped_splits(rows, args.validation_fraction, args.test_fraction, rng)

    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    with (metadata_dir / "sources.csv").open("w", newline="", encoding="utf-8") as file:
        source_fields = sorted({key for source in sources for key in source})
        writer = csv.DictWriter(file, fieldnames=source_fields)
        writer.writeheader()
        writer.writerows(sources)
    write_exclusions(metadata_dir / "exclusions.csv", exclusions)
    with (metadata_dir / "collection_summary.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "random_state": args.random_state,
                "originals_per_class": args.originals_per_class,
                "originals_per_fake_family": per_fake_family,
                "quality_variants": ["original", "jpeg_q75", "jpeg_q50", "jpeg_q25"],
                "saved_manifest_samples": len(rows),
                "counts_by_generator_type": {
                    family: sum(row["generator_type"] == family for row in rows)
                    for family in ("real", *FAKE_FAMILIES)
                },
                "excluded_originals": len(exclusions),
                "warnings": warnings,
            },
            file,
            indent=2,
            sort_keys=True,
        )
    print(f"Saved {len(rows)} quality-variant samples and manifest to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

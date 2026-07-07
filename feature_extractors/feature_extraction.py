from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path
from typing import Any, Iterable

from feature_extractors.arcface_embedding import create_face_analyzer, extract_arcface_features
from feature_extractors.frequency import extract_frequency_features


FEATURE_GROUPS = ("identity", "frequency", "geometry")


def _load_geometry_extractor():
    """
    Load the MediaPipe geometry extractor when it exists.

    Prefer naming the file mediapipe_features.py or geometry.py instead of
    mediapipe.py, because mediapipe.py can shadow the installed mediapipe package.
    """
    if find_spec("feature_extractors.mediapipe_features") is None:
        raise ImportError(
            "Geometry features were requested, but feature_extractors/mediapipe_features.py "
            "was not found. Add that file with a function named extract_geometry_features(image) "
            "before running geometry experiments."
        )

    from feature_extractors.mediapipe_features import extract_geometry_features

    return extract_geometry_features


def validate_feature_groups(feature_groups: Iterable[str]) -> tuple[str, ...]:
    groups = tuple(feature_groups)
    unknown = sorted(set(groups) - set(FEATURE_GROUPS))
    if unknown:
        raise ValueError(f"Unknown feature group(s): {unknown}. Expected one or more of {FEATURE_GROUPS}.")
    if not groups:
        raise ValueError("At least one feature group is required.")
    return groups


def create_feature_context(feature_groups: Iterable[str]) -> dict[str, Any]:
    """
    Create reusable model objects needed by the requested feature groups.

    ArcFace model loading is relatively expensive, so experiments and inference
    should create it once and pass the returned context into extract_image_features.
    """
    groups = validate_feature_groups(feature_groups)
    context: dict[str, Any] = {}

    if "identity" in groups:
        context["arcface_analyzer"] = create_face_analyzer()

    if "geometry" in groups:
        context["geometry_extractor"] = _load_geometry_extractor()

    return context


def extract_image_features(
    image: str | Path,
    feature_groups: Iterable[str],
    context: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Extract and combine the requested feature groups for one image."""
    groups = validate_feature_groups(feature_groups)
    ctx = context or {}
    image_path = Path(image)
    features: dict[str, float] = {}

    if "identity" in groups:
        features.update(
            extract_arcface_features(
                image_path,
                analyzer=ctx.get("arcface_analyzer"),
                allow_no_face=True,
            )
        )

    if "frequency" in groups:
        features.update(extract_frequency_features(image_path))

    if "geometry" in groups:
        geometry_extractor = ctx.get("geometry_extractor") or _load_geometry_extractor()
        features.update(geometry_extractor(image_path))

    return features

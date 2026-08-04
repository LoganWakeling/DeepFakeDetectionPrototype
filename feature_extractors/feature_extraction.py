from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from feature_extractors.arcface_embedding import create_face_analyzer, extract_arcface_features
from feature_extractors.frequency import extract_frequency_features


FEATURE_GROUPS = ("identity", "frequency", "geometry")
FEATURE_PIPELINE_VERSION = "face_crop_v1"
FACE_CROP_MARGIN = 0.25


def crop_largest_face(
    image: np.ndarray,
    analyzer: Any,
    margin: float = FACE_CROP_MARGIN,
) -> np.ndarray:
    """Return a square crop around the largest detected face, or the original image."""
    return detect_and_crop_largest_face(image, analyzer, margin)[0]


def detect_and_crop_largest_face(
    image: np.ndarray,
    analyzer: Any,
    margin: float = FACE_CROP_MARGIN,
) -> tuple[np.ndarray, bool]:
    """Return the largest-face crop and whether a face was detected."""
    if margin < 0:
        raise ValueError("Face crop margin must be non-negative.")
    faces = analyzer.get(image)
    if not faces:
        return image, False

    face = max(
        faces,
        key=lambda item: (item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1]),
    )
    x1, y1, x2, y2 = map(float, face.bbox)
    center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    side = max(x2 - x1, y2 - y1) * (1.0 + 2.0 * margin)
    height, width = image.shape[:2]
    left = max(0, int(np.floor(center_x - side / 2.0)))
    top = max(0, int(np.floor(center_y - side / 2.0)))
    right = min(width, int(np.ceil(center_x + side / 2.0)))
    bottom = min(height, int(np.ceil(center_y + side / 2.0)))
    if right <= left or bottom <= top:
        return image, False
    return image[top:bottom, left:right], True


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

    # Cropping is shared preprocessing, so every feature experiment uses the
    # same detector even when identity features are not requested.
    if groups:
        context["arcface_analyzer"] = create_face_analyzer()

    if "geometry" in groups:
        context["geometry_extractor"] = _load_geometry_extractor()

    return context


def extract_image_features(
    image: str | Path,
    feature_groups: Iterable[str],
    context: dict[str, Any] | None = None,
    apply_face_crop: bool = True,
) -> dict[str, float]:
    """Extract and combine the requested feature groups for one image."""
    groups = validate_feature_groups(feature_groups)
    ctx = context or {}
    image_path = Path(image)
    loaded_image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if loaded_image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    analyzer = ctx.get("arcface_analyzer")
    if analyzer is None:
        analyzer = create_face_analyzer()
    cropped_image = crop_largest_face(loaded_image, analyzer) if apply_face_crop else loaded_image
    features: dict[str, float] = {}

    if "identity" in groups:
        features.update(
            extract_arcface_features(
                cropped_image,
                analyzer=analyzer,
                allow_no_face=True,
            )
        )

    if "frequency" in groups:
        features.update(extract_frequency_features(cropped_image))

    if "geometry" in groups:
        geometry_extractor = ctx.get("geometry_extractor") or _load_geometry_extractor()
        features.update(geometry_extractor(cropped_image))

    return features

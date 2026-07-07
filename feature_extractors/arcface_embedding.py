from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np


DEFAULT_MODEL_PACK = "buffalo_l"
DEFAULT_PROVIDERS = ("CPUExecutionProvider",)
DEFAULT_DET_SIZE = (640, 640)
DEFAULT_MODEL_ROOT = Path(".insightface")
EMBEDDING_DIM = 512


@dataclass(frozen=True)
class ArcFaceResult:
    """ArcFace embedding plus the face detection metadata used to create it."""

    embedding: np.ndarray
    bbox: np.ndarray
    keypoints: np.ndarray
    det_score: float | None = None


def load_image(image_path: str | Path) -> np.ndarray:
    """Load an image from disk as BGR, matching OpenCV and insightface conventions."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return image


def create_face_analyzer(
    model_pack: str = DEFAULT_MODEL_PACK,
    providers: Sequence[str] = DEFAULT_PROVIDERS,
    det_size: tuple[int, int] = DEFAULT_DET_SIZE,
    ctx_id: int = 0,
    root: str | Path = DEFAULT_MODEL_ROOT,
):
    """
    Create the insightface FaceAnalysis app used for ArcFace embeddings.

    The first run may download the requested model pack into .insightface/models.
    The buffalo_l pack includes SCRFD face detection, 5-point face alignment, and
    the w600k_r50 ArcFace recognition model that outputs 512-D normed embeddings.
    """
    try:
        from insightface.app import FaceAnalysis
    except ImportError as exc:
        raise ImportError(
            "ArcFace extraction requires insightface and onnxruntime. "
            "Install them with: pip install insightface onnxruntime"
        ) from exc

    analyzer = FaceAnalysis(name=model_pack, root=str(root), providers=list(providers))
    analyzer.prepare(ctx_id=ctx_id, det_size=det_size)
    return analyzer


def largest_face(faces: Sequence[Any]) -> Any | None:
    """Return the detected face with the largest bounding-box area."""
    if not faces:
        return None

    return max(
        faces,
        key=lambda face: (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]),
    )


def extract_arcface_result(
    image: str | Path | np.ndarray,
    analyzer: Any | None = None,
    providers: Sequence[str] = DEFAULT_PROVIDERS,
    det_size: tuple[int, int] = DEFAULT_DET_SIZE,
) -> ArcFaceResult:
    """
    Extract a 512-D ArcFace identity embedding from the largest face in one image.

    Parameters
    ----------
    image:
        Image path or already-loaded BGR numpy array.
    analyzer:
        Optional reusable FaceAnalysis instance. Pass one when processing many
        images to avoid reloading the model for every image.
    providers:
        ONNX Runtime providers used only when analyzer is not supplied.
    det_size:
        Detector input size used only when analyzer is not supplied.
    """
    loaded_image = load_image(image) if isinstance(image, (str, Path)) else image
    if not isinstance(loaded_image, np.ndarray):
        raise TypeError("image must be a path or a numpy array.")

    app = analyzer or create_face_analyzer(providers=providers, det_size=det_size)
    face = largest_face(app.get(loaded_image))
    if face is None:
        raise ValueError("No face detected in image.")

    return ArcFaceResult(
        embedding=face.normed_embedding.astype(np.float32),
        bbox=face.bbox.astype(np.float32),
        keypoints=face.kps.astype(np.float32),
        det_score=float(face.det_score) if hasattr(face, "det_score") else None,
    )


def extract_arcface_embedding(
    image: str | Path | np.ndarray,
    analyzer: Any | None = None,
    providers: Sequence[str] = DEFAULT_PROVIDERS,
    det_size: tuple[int, int] = DEFAULT_DET_SIZE,
    allow_no_face: bool = False,
) -> np.ndarray:
    """
    Return only the 512-D normalized ArcFace embedding for one image.

    If allow_no_face is True, images without a detected face return a 512-D vector
    filled with NaN values instead of raising ValueError.
    """
    try:
        return extract_arcface_result(
            image,
            analyzer=analyzer,
            providers=providers,
            det_size=det_size,
        ).embedding
    except ValueError:
        if allow_no_face:
            return np.full(EMBEDDING_DIM, np.nan, dtype=np.float32)
        raise


def extract_arcface_features(
    image: str | Path | np.ndarray,
    analyzer: Any | None = None,
    providers: Sequence[str] = DEFAULT_PROVIDERS,
    det_size: tuple[int, int] = DEFAULT_DET_SIZE,
    prefix: str = "arcface",
    allow_no_face: bool = False,
) -> dict[str, float]:
    """
    Extract ArcFace features as a flat dictionary suitable for tabular models.

    Keys are named arcface_000 through arcface_511 by default.
    """
    embedding = extract_arcface_embedding(
        image,
        analyzer=analyzer,
        providers=providers,
        det_size=det_size,
        allow_no_face=allow_no_face,
    )

    return {f"{prefix}_{idx:03d}": float(value) for idx, value in enumerate(embedding)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract ArcFace identity features from one image.")
    parser.add_argument("image_path", type=Path, help="Path to an image sample.")
    parser.add_argument(
        "--allow-no-face",
        action="store_true",
        help="Return NaN features instead of failing when no face is detected.",
    )
    parser.add_argument(
        "--metadata",
        action="store_true",
        help="Include face box, 5 keypoints, and detection score in the JSON output.",
    )
    args = parser.parse_args()

    analyzer = create_face_analyzer()
    if args.metadata:
        result = extract_arcface_result(args.image_path, analyzer=analyzer)
        output: dict[str, Any] = {
            f"arcface_{idx:03d}": float(value)
            for idx, value in enumerate(result.embedding)
        }
        output.update(
            {
                "face_bbox": result.bbox.tolist(),
                "face_keypoints": result.keypoints.tolist(),
                "face_det_score": result.det_score,
            }
        )
    else:
        output = extract_arcface_features(
            args.image_path,
            analyzer=analyzer,
            allow_no_face=args.allow_no_face,
        )

    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

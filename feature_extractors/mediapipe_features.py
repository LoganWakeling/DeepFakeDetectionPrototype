from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from prototypes.mediapipe_landmarks import facial_regions


LEFT_EYE = 33
RIGHT_EYE = 263
NOSE = 1
MODEL_PATH = Path(__file__).resolve().parents[1] / "prototypes" / "mediapipe_landmarks" / "face_landmarker.task"
FACIAL_REGIONS = facial_regions.FACIAL_REGIONS


def normalize_landmarks(landmarks: list[Any]) -> np.ndarray:
    """
    Normalize MediaPipe landmarks by removing translation and scale.

    Returns a numpy array of shape (478, 3).
    """
    points = np.array(
        [[lm.x, lm.y, lm.z] for lm in landmarks],
        dtype=np.float32,
    )

    points -= points[NOSE]

    eye_distance = np.linalg.norm(
        points[LEFT_EYE, :2] - points[RIGHT_EYE, :2],
    )

    if eye_distance > 1e-6:
        points /= eye_distance

    return points


@lru_cache(maxsize=1)
def create_detector() -> vision.FaceLandmarker:
    base_options = python.BaseOptions(
        model_asset_path=str(MODEL_PATH),
        delegate=python.BaseOptions.Delegate.CPU,
    )
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1,
    )
    return vision.FaceLandmarker.create_from_options(options)


def process_image(image_path: str | Path) -> None:
    """Show an image with MediaPipe landmarks drawn on top for manual debugging."""
    image_path = str(image_path)
    mp_image = mp.Image.create_from_file(image_path)
    opencv_image = cv2.imread(image_path)
    result = create_detector().detect(mp_image)

    if result.face_landmarks:
        h, w, _ = opencv_image.shape

        for face_landmarks in result.face_landmarks:
            for landmark in face_landmarks:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(
                    opencv_image,
                    (x, y),
                    1,
                    (0, 255, 0),
                    -1,
                )
    else:
        print(f"No face detected in {image_path}")

    cv2.imshow(image_path, opencv_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def extract_landmarks(image_path: str | Path) -> np.ndarray | None:
    mp_image = mp.Image.create_from_file(str(image_path))
    result = create_detector().detect(mp_image)

    if not result.face_landmarks:
        return None

    face_landmarks = result.face_landmarks[0]
    landmarks = normalize_landmarks(face_landmarks)

    return landmarks


def split_landmarks_by_region(landmarks: np.ndarray) -> dict[str, np.ndarray]:
    return {
        name: landmarks[indices]
        for name, indices in FACIAL_REGIONS.items()
    }


def compare_regions(
    regions1: dict[str, np.ndarray],
    regions2: dict[str, np.ndarray],
) -> tuple[dict[str, float | None], float | None]:
    """
    Compare two faces by calculating the average landmark distance
    for each facial region.

    Returns:
        scores: Average distance for each facial region.
        overall_score: Mean distance across all regions.
    """
    scores = {}

    for region in FACIAL_REGIONS:
        pts1 = np.asarray(regions1[region])
        pts2 = np.asarray(regions2[region])

        if pts1.shape != pts2.shape:
            scores[region] = None
            continue

        distances = np.linalg.norm(pts1 - pts2, axis=1)
        scores[region] = float(np.mean(distances))

    valid_scores = [score for score in scores.values() if score is not None]
    overall_score = float(np.mean(valid_scores)) if valid_scores else None

    return scores, overall_score


def run_landmark_extraction(file_path: str | Path) -> dict[str, np.ndarray] | None:
    image_path = Path(file_path)
    image = extract_landmarks(image_path)

    if image is None:
        print("No face detected.")
        return None

    regions = split_landmarks_by_region(image)
    print("landmarks extracted")
    return regions


def _empty_geometry_features() -> dict[str, float]:
    features: dict[str, float] = {
        "geometry_landmark_count": np.nan,
        "geometry_overall_std_x": np.nan,
        "geometry_overall_std_y": np.nan,
        "geometry_overall_std_z": np.nan,
        "geometry_overall_span_x": np.nan,
        "geometry_overall_span_y": np.nan,
        "geometry_overall_span_z": np.nan,
    }

    for region in FACIAL_REGIONS:
        features.update(
            {
                f"geometry_{region}_mean_x": np.nan,
                f"geometry_{region}_mean_y": np.nan,
                f"geometry_{region}_mean_z": np.nan,
                f"geometry_{region}_std_x": np.nan,
                f"geometry_{region}_std_y": np.nan,
                f"geometry_{region}_std_z": np.nan,
                f"geometry_{region}_span_x": np.nan,
                f"geometry_{region}_span_y": np.nan,
                f"geometry_{region}_span_z": np.nan,
                f"geometry_{region}_centroid_distance": np.nan,
            }
        )

    return features


def extract_geometry_features(image_path: str | Path) -> dict[str, float]:
    """
    Extract MediaPipe landmark geometry features for one image.

    This keeps the existing landmark extraction pipeline intact, then summarizes
    its normalized regional landmarks into fixed numeric columns for training.
    """
    landmarks = extract_landmarks(image_path)
    if landmarks is None:
        return _empty_geometry_features()

    features: dict[str, float] = {
        "geometry_landmark_count": float(len(landmarks)),
        "geometry_overall_std_x": float(np.std(landmarks[:, 0])),
        "geometry_overall_std_y": float(np.std(landmarks[:, 1])),
        "geometry_overall_std_z": float(np.std(landmarks[:, 2])),
        "geometry_overall_span_x": float(np.ptp(landmarks[:, 0])),
        "geometry_overall_span_y": float(np.ptp(landmarks[:, 1])),
        "geometry_overall_span_z": float(np.ptp(landmarks[:, 2])),
    }

    for region, points in split_landmarks_by_region(landmarks).items():
        centroid = np.mean(points, axis=0)
        spans = np.ptp(points, axis=0)
        stds = np.std(points, axis=0)
        features.update(
            {
                f"geometry_{region}_mean_x": float(centroid[0]),
                f"geometry_{region}_mean_y": float(centroid[1]),
                f"geometry_{region}_mean_z": float(centroid[2]),
                f"geometry_{region}_std_x": float(stds[0]),
                f"geometry_{region}_std_y": float(stds[1]),
                f"geometry_{region}_std_z": float(stds[2]),
                f"geometry_{region}_span_x": float(spans[0]),
                f"geometry_{region}_span_y": float(spans[1]),
                f"geometry_{region}_span_z": float(spans[2]),
                f"geometry_{region}_centroid_distance": float(np.linalg.norm(centroid)),
            }
        )

    return features

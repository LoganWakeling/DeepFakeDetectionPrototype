from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from feature_extractors.feature_extraction import FEATURE_PIPELINE_VERSION
from training.experiment_runner import (
    IMAGE_COLUMN,
    LABEL_COLUMN,
    SPLIT_COLUMN,
    build_feature_table,
    load_manifest,
    train_classifier,
)


EXPERIMENT_NAME = "09_classifier_fusion"
FEATURE_GROUPS = ("identity", "frequency", "geometry")
FEATURE_EXPERIMENTS = {
    "identity": "01_identity",
    "frequency": "02_frequency",
    "geometry": "03_geometry",
}


def load_cached_features(
    output_dir: Path,
    group: str,
    manifest: pd.DataFrame,
) -> pd.DataFrame | None:
    """Reuse a specialist experiment's features only when every row matches."""
    path = output_dir / FEATURE_EXPERIMENTS[group] / "features.csv"
    metadata_path = path.parent / "feature_pipeline.json"
    if not path.exists() or not metadata_path.exists():
        return None
    try:
        with metadata_path.open(encoding="utf-8") as file:
            if json.load(file).get("version") != FEATURE_PIPELINE_VERSION:
                return None
    except (OSError, json.JSONDecodeError):
        return None
    features = pd.read_csv(path)
    required = {IMAGE_COLUMN, LABEL_COLUMN}
    if not required.issubset(features.columns) or len(features) != len(manifest):
        return None
    paths_match = features[IMAGE_COLUMN].astype(str).equals(manifest[IMAGE_COLUMN].astype(str))
    labels_match = features[LABEL_COLUMN].astype(int).equals(manifest[LABEL_COLUMN].astype(int))
    return features if paths_match and labels_match else None


def split_manifest(
    manifest: pd.DataFrame,
    validation_size: float,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create one shared train/validation/test split for all base classifiers."""
    if not 0.0 < validation_size < 1.0:
        raise ValueError("validation_size must be between 0 and 1.")

    split = (
        manifest[SPLIT_COLUMN].astype(str).str.lower()
        if SPLIT_COLUMN in manifest.columns
        else pd.Series("", index=manifest.index)
    )
    train = manifest.loc[split == "train"].copy()
    validation = manifest.loc[split.isin(["validation", "val"])].copy()
    test = manifest.loc[split == "test"].copy()

    if train.empty and test.empty:
        train, test = train_test_split(
            manifest,
            test_size=test_size,
            random_state=random_state,
            stratify=manifest[LABEL_COLUMN] if manifest[LABEL_COLUMN].nunique() == 2 else None,
        )
    elif train.empty or test.empty:
        raise ValueError("Manifest split must contain both train and test rows.")

    if validation.empty:
        train, validation = train_test_split(
            train,
            test_size=validation_size,
            random_state=random_state,
            stratify=train[LABEL_COLUMN] if train[LABEL_COLUMN].nunique() == 2 else None,
        )

    return train.copy(), validation.copy(), test.copy()


def _metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, object]:
    predictions = (probabilities >= threshold).astype(int)
    result: dict[str, object] = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=[0, 1]).tolist(),
        "threshold": float(threshold),
    }
    if len(np.unique(y_true)) == 2:
        result["roc_auc"] = float(roc_auc_score(y_true, probabilities))
        result["average_precision"] = float(average_precision_score(y_true, probabilities))
    else:
        result["roc_auc"] = None
        result["average_precision"] = None
    return result


def select_fusion(
    y_validation: np.ndarray,
    probability_matrix: np.ndarray,
    max_false_positive_rate: float,
) -> tuple[np.ndarray, float]:
    """Select weights and threshold on validation data, prioritizing fake recall."""
    if not 0.0 <= max_false_positive_rate <= 1.0:
        raise ValueError("max_false_positive_rate must be between 0 and 1.")

    weight_grid = np.arange(0.0, 1.01, 0.1)
    thresholds = np.arange(0.1, 0.91, 0.05)
    candidates: list[tuple[tuple[float, float, float, float], np.ndarray, float]] = []
    for first, second in itertools.product(weight_grid, repeat=2):
        third = 1.0 - first - second
        if third < -1e-9:
            continue
        weights = np.array([first, second, max(0.0, third)])
        scores = probability_matrix @ weights
        for threshold in thresholds:
            predictions = scores >= threshold
            negatives = y_validation == 0
            fpr = float(predictions[negatives].mean()) if negatives.any() else 0.0
            recall = recall_score(y_validation, predictions, zero_division=0)
            precision = precision_score(y_validation, predictions, zero_division=0)
            # Constraint satisfaction, recall, precision, then lower FPR.
            rank = (float(fpr <= max_false_positive_rate), float(recall), float(precision), -fpr)
            candidates.append((rank, weights, float(threshold)))

    _, weights, threshold = max(candidates, key=lambda item: item[0])
    return weights, threshold


def run_fusion_experiment(
    manifest_path: Path,
    output_dir: Path,
    validation_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = 42,
    max_false_positive_rate: float = 0.1,
) -> dict[str, object]:
    experiment_dir = output_dir / EXPERIMENT_NAME
    experiment_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(manifest_path).reset_index(drop=True)
    train, validation, test = split_manifest(manifest, validation_size, test_size, random_state)

    split_indices = {
        "train": train.index.to_numpy(),
        "validation": validation.index.to_numpy(),
        "test": test.index.to_numpy(),
    }
    probabilities: dict[str, dict[str, np.ndarray]] = {name: {} for name in split_indices}
    models: dict[str, object] = {}
    feature_columns: dict[str, list[str]] = {}

    for group in FEATURE_GROUPS:
        features = load_cached_features(output_dir, group, manifest)
        if features is None:
            print(f"Extracting {group} features...")
            features = build_feature_table(manifest, (group,))
            features.to_csv(experiment_dir / f"features_{group}.csv", index=False)
        else:
            print(f"Reusing validated features from {FEATURE_EXPERIMENTS[group]}/features.csv")
        columns = [column for column in features if column not in {IMAGE_COLUMN, LABEL_COLUMN}]
        model = train_classifier(random_state)
        model.fit(features.loc[split_indices["train"], columns], train[LABEL_COLUMN].astype(int))
        models[group] = model
        feature_columns[group] = columns
        for split_name, indices in split_indices.items():
            probabilities[split_name][group] = model.predict_proba(features.loc[indices, columns])[:, 1]

    validation_matrix = np.column_stack([probabilities["validation"][g] for g in FEATURE_GROUPS])
    weights, threshold = select_fusion(
        validation[LABEL_COLUMN].to_numpy(), validation_matrix, max_false_positive_rate
    )

    metrics: dict[str, object] = {
        "experiment": EXPERIMENT_NAME,
        "feature_groups": list(FEATURE_GROUPS),
        "weights": dict(zip(FEATURE_GROUPS, map(float, weights))),
        "threshold": threshold,
        "max_false_positive_rate": max_false_positive_rate,
    }
    for split_name, split_frame in (("validation", validation), ("test", test)):
        matrix = np.column_stack([probabilities[split_name][g] for g in FEATURE_GROUPS])
        fused = matrix @ weights
        labels = split_frame[LABEL_COLUMN].to_numpy()
        metrics[split_name] = _metrics(labels, fused, threshold)
        predictions = split_frame.copy()
        for column_index, group in enumerate(FEATURE_GROUPS):
            predictions[f"{group}_fake_probability"] = matrix[:, column_index]
        predictions["fake_probability"] = fused
        predictions["predicted_label"] = (fused >= threshold).astype(int)
        predictions.to_csv(experiment_dir / f"{split_name}_predictions.csv", index=False)

    joblib.dump(
        {
            "models": models,
            "feature_columns": feature_columns,
            "feature_groups": FEATURE_GROUPS,
            "weights": weights,
            "threshold": threshold,
        },
        experiment_dir / "model.joblib",
    )
    with (experiment_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, sort_keys=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train three specialist classifiers and fuse them.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/experiments"))
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-false-positive-rate", type=float, default=0.1)
    args = parser.parse_args()
    result = run_fusion_experiment(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        validation_size=args.validation_size,
        test_size=args.test_size,
        random_state=args.random_state,
        max_false_positive_rate=args.max_false_positive_rate,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

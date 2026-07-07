from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
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
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from feature_extractors.feature_extraction import (
    create_feature_context,
    extract_image_features,
    validate_feature_groups,
)


LABEL_COLUMN = "label"
IMAGE_COLUMN = "image_path"
SPLIT_COLUMN = "split"
GENERATOR_COLUMN = "generator"
GENERATOR_TYPE_COLUMN = "generator_type"


EXPERIMENTS: dict[str, tuple[str, ...]] = {
    "01_identity": ("identity",),
    "02_frequency": ("frequency",),
    "03_geometry": ("geometry",),
    "04_identity_frequency": ("identity", "frequency"),
    "05_identity_geometry": ("identity", "geometry"),
    "06_frequency_geometry": ("frequency", "geometry"),
    "07_all_features": ("identity", "frequency", "geometry"),
}


def load_manifest(manifest_path: Path) -> pd.DataFrame:
    """
    Load a dataset manifest CSV.

    Required columns:
    - image_path: path to the image, absolute or relative to the manifest folder
    - label: 0 for real, 1 for fake

    Optional column:
    - split: train or test
    """
    df = pd.read_csv(manifest_path)
    missing = {IMAGE_COLUMN, LABEL_COLUMN} - set(df.columns)
    if missing:
        raise ValueError(f"Manifest is missing required column(s): {sorted(missing)}")

    base_dir = manifest_path.parent
    df[IMAGE_COLUMN] = df[IMAGE_COLUMN].map(
        lambda value: str(Path(value) if Path(value).is_absolute() else base_dir / str(value))
    )
    return df


def build_feature_table(
    manifest: pd.DataFrame,
    feature_groups: Iterable[str],
) -> pd.DataFrame:
    groups = validate_feature_groups(feature_groups)
    context = create_feature_context(groups)
    rows: list[dict[str, float | int | str]] = []

    for idx, sample in manifest.iterrows():
        image_path = sample[IMAGE_COLUMN]
        features = extract_image_features(image_path, groups, context=context)
        rows.append(
            {
                IMAGE_COLUMN: image_path,
                LABEL_COLUMN: int(sample[LABEL_COLUMN]),
                **features,
            }
        )
        if (idx + 1) % 25 == 0:
            print(f"Extracted features for {idx + 1}/{len(manifest)} images")

    return pd.DataFrame(rows)


def split_features(
    features: pd.DataFrame,
    manifest: pd.DataFrame,
    test_size: float,
    random_state: int,
):
    feature_columns = [column for column in features.columns if column not in {IMAGE_COLUMN, LABEL_COLUMN}]
    x = features[feature_columns]
    y = features[LABEL_COLUMN].astype(int)

    if SPLIT_COLUMN in manifest.columns:
        split = manifest[SPLIT_COLUMN].astype(str).str.lower().to_numpy()
        train_mask = split == "train"
        test_mask = split == "test"
        if not train_mask.any() or not test_mask.any():
            raise ValueError("Manifest split column must include at least one train and one test row.")
        return x[train_mask], x[test_mask], y[train_mask], y[test_mask], feature_columns

    stratify = y if y.nunique() == 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    return x_train, x_test, y_train, y_test, feature_columns


def train_classifier(random_state: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def evaluate_model(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, object]:
    y_pred = model.predict(x_test)
    y_score = model.predict_proba(x_test)[:, 1]

    metrics: dict[str, object] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    if len(np.unique(y_test)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_score))
        metrics["average_precision"] = float(average_precision_score(y_test, y_score))
    else:
        metrics["roc_auc"] = None
        metrics["average_precision"] = None

    return metrics


def build_prediction_table(
    model: Pipeline,
    x: pd.DataFrame,
    y: pd.Series,
    manifest: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    """Build one row per image with expected label, prediction, and fake probability."""
    y_pred = model.predict(x)
    y_score = model.predict_proba(x)[:, 1]
    metadata_columns = [
        column
        for column in [IMAGE_COLUMN, LABEL_COLUMN, SPLIT_COLUMN, GENERATOR_COLUMN, GENERATOR_TYPE_COLUMN]
        if column in manifest.columns
    ]

    table = manifest.loc[x.index, metadata_columns].copy()
    table["expected_label"] = y.astype(int).to_numpy()
    table["fake_probability"] = y_score
    table["predicted_label"] = y_pred.astype(int)
    table["predicted_class"] = np.where(table["predicted_label"] == 1, "fake", "real")
    table["correct"] = table["predicted_label"] == table["expected_label"]
    table["evaluation_split"] = split_name
    return table


def summarize_prediction_table(predictions: pd.DataFrame) -> dict[str, object]:
    """Summarize a prediction table overall and by available dataset metadata."""
    summary: dict[str, object] = {
        "num_samples": int(len(predictions)),
        "accuracy": float(predictions["correct"].mean()) if len(predictions) else None,
    }

    for group_column in [GENERATOR_COLUMN, GENERATOR_TYPE_COLUMN]:
        if group_column not in predictions.columns:
            continue

        grouped = (
            predictions.groupby(group_column, dropna=False)
            .agg(
                num_samples=("correct", "size"),
                accuracy=("correct", "mean"),
                mean_fake_probability=("fake_probability", "mean"),
            )
            .reset_index()
        )
        summary[f"by_{group_column}"] = grouped.to_dict(orient="records")

    return summary


def run_experiment(
    experiment_name: str,
    feature_groups: Iterable[str],
    manifest_path: Path,
    output_dir: Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, object]:
    groups = validate_feature_groups(feature_groups)
    experiment_dir = output_dir / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(manifest_path)
    features = build_feature_table(manifest, groups)
    features_path = experiment_dir / "features.csv"
    features.to_csv(features_path, index=False)

    x_train, x_test, y_train, y_test, feature_columns = split_features(
        features,
        manifest,
        test_size=test_size,
        random_state=random_state,
    )

    model = train_classifier(random_state=random_state)
    model.fit(x_train, y_train)
    train_metrics = evaluate_model(model, x_train, y_train)
    test_metrics = evaluate_model(model, x_test, y_test)
    train_predictions = build_prediction_table(model, x_train, y_train, manifest, split_name="train")
    test_predictions = build_prediction_table(model, x_test, y_test, manifest, split_name="test")
    train_predictions.to_csv(experiment_dir / "train_predictions.csv", index=False)
    test_predictions.to_csv(experiment_dir / "test_predictions.csv", index=False)

    metrics: dict[str, object] = {
        "train": train_metrics,
        "test": test_metrics,
        "train_prediction_summary": summarize_prediction_table(train_predictions),
        "test_prediction_summary": summarize_prediction_table(test_predictions),
    }
    metrics.update(
        {
            "experiment": experiment_name,
            "feature_groups": list(groups),
            "num_samples": int(len(features)),
            "num_features": int(len(feature_columns)),
            "feature_columns": feature_columns,
        }
    )

    joblib.dump(
        {
            "model": model,
            "feature_groups": groups,
            "feature_columns": feature_columns,
        },
        experiment_dir / "model.joblib",
    )
    with (experiment_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, sort_keys=True)

    return metrics


def parse_args(default_experiment: str | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one deepfake feature ablation experiment.")
    if default_experiment is None:
        parser.add_argument("experiment", choices=sorted(EXPERIMENTS), help="Experiment id to run.")
    else:
        parser.set_defaults(experiment=default_experiment)
    parser.add_argument("--manifest", type=Path, required=True, help="CSV with image_path,label columns.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/experiments"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main(default_experiment: str | None = None) -> None:
    args = parse_args(default_experiment=default_experiment)
    metrics = run_experiment(
        experiment_name=args.experiment,
        feature_groups=EXPERIMENTS[args.experiment],
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

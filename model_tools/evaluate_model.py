from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from training.experiment_runner import (
    IMAGE_COLUMN,
    LABEL_COLUMN,
    build_prediction_table,
    evaluate_model,
    load_manifest,
    summarize_prediction_table,
)
from feature_extractors.feature_extraction import create_feature_context, extract_image_features


def build_features_for_model(manifest: pd.DataFrame, feature_groups: tuple[str, ...]) -> pd.DataFrame:
    context = create_feature_context(feature_groups)
    rows: list[dict[str, float]] = []

    for idx, sample in manifest.iterrows():
        rows.append(extract_image_features(sample[IMAGE_COLUMN], feature_groups, context=context))
        if (idx + 1) % 25 == 0:
            print(f"Extracted test features for {idx + 1}/{len(manifest)} images")

    return pd.DataFrame(rows, index=manifest.index)


def evaluate_saved_model(model_path: Path, manifest_path: Path, output_path: Path) -> dict[str, object]:
    artifact = joblib.load(model_path)
    model = artifact["model"]
    feature_groups = tuple(artifact["feature_groups"])
    feature_columns = list(artifact["feature_columns"])

    manifest = load_manifest(manifest_path)
    features = build_features_for_model(manifest, feature_groups)
    x = pd.DataFrame(
        [{column: row.get(column) for column in feature_columns} for row in features.to_dict(orient="records")],
        index=features.index,
    )

    if LABEL_COLUMN in manifest.columns:
        y = manifest[LABEL_COLUMN].astype(int)
        predictions = build_prediction_table(model, x, y, manifest, split_name="external_test")
        metrics = evaluate_model(model, x, y)
        metrics["prediction_summary"] = summarize_prediction_table(predictions)
    else:
        fake_probability = model.predict_proba(x)[:, 1]
        predicted_label = model.predict(x).astype(int)
        predictions = manifest.copy()
        predictions["fake_probability"] = fake_probability
        predictions["predicted_label"] = predicted_label
        predictions["predicted_class"] = predictions["predicted_label"].map({0: "real", 1: "fake"})
        metrics = {"prediction_summary": {"num_samples": int(len(predictions))}}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)

    metrics_path = output_path.with_suffix(".metrics.json")
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, sort_keys=True)

    return {
        "predictions_path": str(output_path),
        "metrics_path": str(metrics_path),
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained deepfake model on a manifest CSV.")
    parser.add_argument("--model", type=Path, required=True, help="Path to model.joblib.")
    parser.add_argument("--manifest", type=Path, required=True, help="CSV with image_path and optional label metadata.")
    parser.add_argument("--output", type=Path, required=True, help="Where to write prediction CSV.")
    args = parser.parse_args()

    result = evaluate_saved_model(args.model, args.manifest, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

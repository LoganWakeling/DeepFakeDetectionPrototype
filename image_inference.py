from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from feature_extraction import create_feature_context, extract_image_features


def predict_image(image_path: Path, model_path: Path) -> dict[str, object]:
    artifact = joblib.load(model_path)
    model = artifact["model"]
    feature_groups = tuple(artifact["feature_groups"])
    feature_columns = list(artifact["feature_columns"])

    context = create_feature_context(feature_groups)
    features = extract_image_features(image_path, feature_groups, context=context)
    row = pd.DataFrame([{column: features.get(column) for column in feature_columns}])

    fake_probability = float(model.predict_proba(row)[0, 1])
    prediction = int(fake_probability >= 0.5)

    return {
        "image_path": str(image_path),
        "model_path": str(model_path),
        "feature_groups": list(feature_groups),
        "fake_probability": fake_probability,
        "prediction": prediction,
        "prediction_label": "fake" if prediction == 1 else "real",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deepfake prediction on one image.")
    parser.add_argument("image_path", type=Path, help="Path to the image to classify.")
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to a trained model.joblib from outputs/experiments/<experiment>/.",
    )
    args = parser.parse_args()

    result = predict_image(args.image_path, args.model)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

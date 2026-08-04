from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

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

from training.experiment_runner import (
    IMAGE_COLUMN,
    LABEL_COLUMN,
    SPLIT_COLUMN,
    load_manifest,
    summarize_prediction_table,
)


EXPERIMENT_NAME = "08_xception"
IMAGE_SIZE = (299, 299)


def _tensorflow():
    """Import TensorFlow only when the Xception experiment is requested."""
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "The Xception experiment requires TensorFlow. "
            "Install the optional dependency with `pip install tensorflow`."
        ) from exc
    return tf


def split_manifest(
    manifest: pd.DataFrame,
    validation_size: float = 0.1,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return disjoint train, validation, and test manifest rows."""
    if SPLIT_COLUMN not in manifest:
        raise ValueError(
            "The Xception manifest must contain a split column with train and test rows."
        )

    split = manifest[SPLIT_COLUMN].astype(str).str.lower()
    train = manifest.loc[split == "train"].copy()
    validation = manifest.loc[split.isin(["validation", "val"])].copy()
    test = manifest.loc[split == "test"].copy()

    if train.empty or test.empty:
        raise ValueError("The manifest split column must include train and test rows.")
    if not 0.0 < validation_size < 1.0:
        raise ValueError("validation_size must be between 0 and 1.")

    if validation.empty:
        labels = train[LABEL_COLUMN].astype(int)
        stratify = labels if labels.nunique() == 2 and labels.value_counts().min() >= 2 else None
        train, validation = train_test_split(
            train,
            test_size=validation_size,
            random_state=random_state,
            stratify=stratify,
        )

    return train.copy(), validation.copy(), test.copy()


def create_xception_model(dropout: float = 0.3, learning_rate: float = 1e-4):
    tf = _tensorflow()
    base_model = tf.keras.applications.Xception(
        weights="imagenet",
        include_top=False,
        input_shape=(*IMAGE_SIZE, 3),
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name="image")
    x = tf.keras.applications.xception.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    output = tf.keras.layers.Dense(1, activation="sigmoid", name="fake_probability")(x)

    model = tf.keras.Model(inputs, output, name="xception_deepfake_detector")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="roc_auc"),
            tf.keras.metrics.AUC(name="average_precision", curve="PR"),
        ],
    )
    return model


def _decode_and_resize(image_path, label):
    tf = _tensorflow()
    image_bytes = tf.io.read_file(image_path)
    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False,
    )
    image.set_shape((None, None, 3))
    image = tf.image.resize(image, IMAGE_SIZE, antialias=True)
    return image, tf.cast(label, tf.float32)


def _cache_path(
    cache_dir: Path,
    split_name: str,
    manifest: pd.DataFrame,
) -> Path:
    fingerprint_values = (
        manifest[[IMAGE_COLUMN, LABEL_COLUMN]].astype(str).agg("|".join, axis=1).tolist()
    )
    digest = hashlib.sha256("\n".join(fingerprint_values).encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"{split_name}_{digest}.tf-cache"


def create_dataset(
    manifest: pd.DataFrame,
    batch_size: int,
    training: bool,
    split_name: str,
    cache_dir: Path | None,
    random_state: int,
    cache_images: bool = True,
):
    """
    Build an image dataset and cache decoded/resized images before batching.

    A disk cache avoids repeating image decoding and resizing across epochs and
    is reusable on later runs while the manifest rows remain unchanged. Passing
    cache_dir=None uses an in-memory cache for the current run. Set cache_images
    to False to avoid both disk and memory caching.
    """
    tf = _tensorflow()
    dataset = tf.data.Dataset.from_tensor_slices(
        (
            manifest[IMAGE_COLUMN].astype(str).to_numpy(),
            manifest[LABEL_COLUMN].astype(np.float32).to_numpy(),
        )
    )
    dataset = dataset.map(_decode_and_resize, num_parallel_calls=tf.data.AUTOTUNE)

    if cache_images:
        if cache_dir is None:
            dataset = dataset.cache()
        else:
            cache_dir.mkdir(parents=True, exist_ok=True)
            dataset = dataset.cache(str(_cache_path(cache_dir, split_name, manifest)))

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(manifest),
            seed=random_state,
            reshuffle_each_iteration=True,
        )

    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def calculate_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, object]:
    predictions = (probabilities >= threshold).astype(int)
    metrics: dict[str, object] = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
        "roc_auc": None,
        "average_precision": None,
    }
    if len(np.unique(labels)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(labels, probabilities))
        metrics["average_precision"] = float(
            average_precision_score(labels, probabilities)
        )
    return metrics


def build_prediction_table(
    model,
    dataset,
    manifest: pd.DataFrame,
    split_name: str,
    threshold: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    probabilities = model.predict(dataset, verbose=0).reshape(-1)
    labels = manifest[LABEL_COLUMN].astype(int).to_numpy()
    predictions = (probabilities >= threshold).astype(int)

    table = manifest.copy()
    table["expected_label"] = labels
    table["fake_probability"] = probabilities
    table["predicted_label"] = predictions
    table["predicted_class"] = np.where(predictions == 1, "fake", "real")
    table["correct"] = predictions == labels
    table["evaluation_split"] = split_name
    return table, calculate_metrics(labels, probabilities, threshold)


def run_xception_experiment(
    manifest_path: Path,
    output_dir: Path,
    epochs: int = 10,
    batch_size: int = 16,
    validation_size: float = 0.1,
    random_state: int = 42,
    cache_dir: Path | None = None,
    threshold: float = 0.5,
    cache_images: bool = True,
) -> dict[str, object]:
    tf = _tensorflow()
    tf.keras.utils.set_random_seed(random_state)

    experiment_dir = output_dir / EXPERIMENT_NAME
    experiment_dir.mkdir(parents=True, exist_ok=True)
    effective_cache_dir = cache_dir or experiment_dir / "image_cache"

    manifest = load_manifest(manifest_path)
    train, validation, test = split_manifest(
        manifest,
        validation_size=validation_size,
        random_state=random_state,
    )
    train_dataset = create_dataset(
        train, batch_size, True, "train", effective_cache_dir, random_state, cache_images
    )
    train_evaluation_dataset = create_dataset(
        train, batch_size, False, "train", effective_cache_dir, random_state, cache_images
    )
    validation_dataset = create_dataset(
        validation, batch_size, False, "validation", effective_cache_dir, random_state, cache_images
    )
    test_dataset = create_dataset(
        test, batch_size, False, "test", effective_cache_dir, random_state, cache_images
    )

    model = create_xception_model()
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=3,
                restore_best_weights=True,
            )
        ],
    )

    train_predictions, train_metrics = build_prediction_table(
        model, train_evaluation_dataset, train, "train", threshold
    )
    test_predictions, test_metrics = build_prediction_table(
        model, test_dataset, test, "test", threshold
    )
    train_predictions.to_csv(experiment_dir / "train_predictions.csv", index=False)
    test_predictions.to_csv(experiment_dir / "test_predictions.csv", index=False)
    model.save(experiment_dir / "model.keras")

    history_data = {
        key: [float(value) for value in values] for key, values in history.history.items()
    }
    with (experiment_dir / "history.json").open("w", encoding="utf-8") as file:
        json.dump(history_data, file, indent=2, sort_keys=True)

    metrics: dict[str, Any] = {
        "experiment": EXPERIMENT_NAME,
        "feature_groups": ["raw_image"],
        "num_samples": int(len(manifest)),
        "image_size": list(IMAGE_SIZE),
        "decision_threshold": threshold,
        "train": train_metrics,
        "test": test_metrics,
        "train_prediction_summary": summarize_prediction_table(train_predictions),
        "test_prediction_summary": summarize_prediction_table(test_predictions),
    }
    with (experiment_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, sort_keys=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate Xception using an image manifest."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/experiments"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--validation-size", type=float, default=0.1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Disk cache for decoded/resized images (default: output experiment folder).",
    )
    parser.add_argument(
        "--no-image-cache",
        action="store_true",
        help="Decode and resize images each pass instead of consuming cache storage.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    metrics = run_xception_experiment(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_size=args.validation_size,
        random_state=args.random_state,
        cache_dir=args.cache_dir,
        threshold=args.threshold,
        cache_images=not args.no_image_cache,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

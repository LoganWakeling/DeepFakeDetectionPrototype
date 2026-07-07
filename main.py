from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from experiment_runner import EXPERIMENTS, run_experiment


def _format_accuracy(value: object) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value) * 100:.2f}%"


def _print_table(title: str, table: pd.DataFrame) -> None:
    print(f"\n{title}")
    if table.empty:
        print("No rows to display.")
        return

    display_table = table.copy()
    for column in display_table.columns:
        if column.endswith("accuracy"):
            display_table[column] = display_table[column].map(_format_accuracy)

    print(display_table.to_string(index=False))


def _overall_row(experiment_name: str, metrics: dict[str, Any]) -> dict[str, object]:
    test_metrics = metrics["test"]
    return {
        "experiment": experiment_name,
        "feature_groups": "+".join(metrics["feature_groups"]),
        "num_test_samples": metrics["test_prediction_summary"]["num_samples"],
        "test_accuracy": test_metrics["accuracy"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
    }


def _group_rows(
    experiment_name: str,
    metrics: dict[str, Any],
    summary_key: str,
    group_column: str,
) -> list[dict[str, object]]:
    records = metrics["test_prediction_summary"].get(summary_key, [])
    rows: list[dict[str, object]] = []
    for record in records:
        rows.append(
            {
                "experiment": experiment_name,
                group_column: record[group_column],
                "num_test_samples": record["num_samples"],
                "test_accuracy": record["accuracy"],
                "mean_fake_probability": record["mean_fake_probability"],
            }
        )
    return rows


def run_all_experiments(
    manifest_path: Path,
    output_dir: Path,
    test_size: float,
    random_state: int,
) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)

    overall_rows: list[dict[str, object]] = []
    generator_rows: list[dict[str, object]] = []
    generator_type_rows: list[dict[str, object]] = []

    for experiment_name, feature_groups in EXPERIMENTS.items():
        print(f"\nRunning {experiment_name}: {', '.join(feature_groups)}")
        metrics = run_experiment(
            experiment_name=experiment_name,
            feature_groups=feature_groups,
            manifest_path=manifest_path,
            output_dir=output_dir,
            test_size=test_size,
            random_state=random_state,
        )

        overall_rows.append(_overall_row(experiment_name, metrics))
        generator_rows.extend(
            _group_rows(
                experiment_name,
                metrics,
                summary_key="by_generator",
                group_column="generator",
            )
        )
        generator_type_rows.extend(
            _group_rows(
                experiment_name,
                metrics,
                summary_key="by_generator_type",
                group_column="generator_type",
            )
        )

    tables = {
        "overall": pd.DataFrame(overall_rows),
        "by_generator": pd.DataFrame(generator_rows),
        "by_generator_type": pd.DataFrame(generator_type_rows),
    }

    for name, table in tables.items():
        table.to_csv(output_dir / f"summary_{name}.csv", index=False)

    return tables


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all deepfake detection ablation experiments.")
    parser.add_argument("--manifest", type=Path, required=True, help="CSV with image_path,label columns.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/experiments"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    tables = run_all_experiments(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    _print_table("Overall test accuracy by experiment", tables["overall"])
    _print_table("Test accuracy by generator", tables["by_generator"])
    _print_table("Test accuracy by generator type", tables["by_generator_type"])
    print(f"\nSaved summary CSV files in {args.output_dir}")


if __name__ == "__main__":
    main()

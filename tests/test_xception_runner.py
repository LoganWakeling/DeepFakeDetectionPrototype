from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from training.xception_runner import calculate_metrics, split_manifest


class XceptionRunnerTests(unittest.TestCase):
    def test_manifest_split_keeps_test_set_separate(self) -> None:
        manifest = pd.DataFrame(
            {
                "image_path": [f"image_{index}.jpg" for index in range(12)],
                "label": [0, 1] * 6,
                "split": ["train"] * 10 + ["test"] * 2,
            }
        )

        train, validation, test = split_manifest(
            manifest, validation_size=0.2, random_state=7
        )

        self.assertEqual(len(train), 8)
        self.assertEqual(len(validation), 2)
        self.assertEqual(len(test), 2)
        self.assertTrue(set(train.index).isdisjoint(validation.index))
        self.assertTrue(set(train.index).isdisjoint(test.index))
        self.assertTrue(set(validation.index).isdisjoint(test.index))

    def test_metrics_use_fake_as_positive_class(self) -> None:
        metrics = calculate_metrics(
            np.array([0, 0, 1, 1]),
            np.array([0.1, 0.7, 0.8, 0.9]),
        )

        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [0, 2]])
        self.assertEqual(metrics["recall"], 1.0)
        self.assertAlmostEqual(metrics["accuracy"], 0.75)


if __name__ == "__main__":
    unittest.main()

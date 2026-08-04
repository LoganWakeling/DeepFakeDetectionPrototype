import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from training.fusion_runner import load_cached_features, select_fusion, split_manifest
from feature_extractors.feature_extraction import FEATURE_PIPELINE_VERSION


class FusionRunnerTests(unittest.TestCase):
    def test_shared_split_is_disjoint_and_complete(self):
        manifest = pd.DataFrame({"image_path": [str(i) for i in range(20)], "label": [0, 1] * 10})
        train, validation, test = split_manifest(manifest, 0.25, 0.2, 7)
        self.assertTrue(set(train.index).isdisjoint(validation.index))
        self.assertTrue(set(train.index).isdisjoint(test.index))
        self.assertTrue(set(validation.index).isdisjoint(test.index))
        self.assertEqual(set(train.index) | set(validation.index) | set(test.index), set(manifest.index))

    def test_fusion_selection_returns_valid_parameters(self):
        labels = np.array([0, 0, 1, 1])
        probabilities = np.array([[0.1, 0.2, 0.3], [0.2, 0.1, 0.4], [0.8, 0.7, 0.6], [0.9, 0.8, 0.7]])
        weights, threshold = select_fusion(labels, probabilities, 0.1)
        self.assertAlmostEqual(float(weights.sum()), 1.0)
        self.assertTrue(np.all(weights >= 0.0))
        self.assertGreaterEqual(threshold, 0.0)
        self.assertLessEqual(threshold, 1.0)

    def test_cached_features_must_match_manifest_rows(self):
        manifest = pd.DataFrame({"image_path": ["a", "b"], "label": [0, 1]})
        with TemporaryDirectory() as directory:
            path = Path(directory) / "01_identity"
            path.mkdir()
            pd.DataFrame(
                {"image_path": ["a", "b"], "label": [0, 1], "identity_0": [0.1, 0.2]}
            ).to_csv(path / "features.csv", index=False)
            with (path / "feature_pipeline.json").open("w", encoding="utf-8") as file:
                json.dump({"version": FEATURE_PIPELINE_VERSION}, file)
            loaded = load_cached_features(Path(directory), "identity", manifest)
            self.assertIsNotNone(loaded)

            reversed_manifest = manifest.iloc[::-1].reset_index(drop=True)
            self.assertIsNone(load_cached_features(Path(directory), "identity", reversed_manifest))


if __name__ == "__main__":
    unittest.main()

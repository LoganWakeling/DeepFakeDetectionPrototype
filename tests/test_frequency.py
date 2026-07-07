from __future__ import annotations

from pathlib import Path
import unittest

import cv2
import numpy as np

from feature_extractors.frequency import extract_frequency_features


SAMPLE_IMAGES = sorted(Path("data/test_images").glob("sample*.jpg"))


def extract_frequency_features_cv2dct(image: np.ndarray, size: int = 128) -> dict[str, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    resized = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    normalized = resized / 255.0
    normalized = normalized - np.mean(normalized)

    dct_coefficients = cv2.dct(normalized)

    coefficients = np.abs(dct_coefficients)
    rows, cols = np.indices(coefficients.shape)
    normalized_frequency = (rows + cols) / (2 * (size - 1))

    low_mask = normalized_frequency < 0.15
    mid_mask = (normalized_frequency >= 0.15) & (normalized_frequency < 0.40)
    high_mask = normalized_frequency >= 0.40

    low_energy = float(np.mean(coefficients[low_mask] ** 2))
    mid_energy = float(np.mean(coefficients[mid_mask] ** 2))
    high_energy = float(np.mean(coefficients[high_mask] ** 2))

    return {
        "dct_low_energy": low_energy,
        "dct_mid_energy": mid_energy,
        "dct_high_energy": high_energy,
        "dct_global_std": float(np.std(dct_coefficients)),
        "dct_high_low_ratio": float(high_energy / (low_energy + 1e-8)),
    }


class FrequencyFeatureTests(unittest.TestCase):
    def test_sample_images_match_cv2_dct_reference(self) -> None:
        self.assertEqual(len(SAMPLE_IMAGES), 5)

        for image_path in SAMPLE_IMAGES:
            with self.subTest(image=str(image_path)):
                image = cv2.imread(str(image_path))
                self.assertIsNotNone(image)

                features = extract_frequency_features(image)
                cv2dct_features = extract_frequency_features_cv2dct(image)

                self.assertEqual(list(features), list(cv2dct_features))
                self.assertEqual(len(features), 5)
                self.assertTrue(np.all(np.isfinite(list(features.values()))))
                self.assertTrue(
                    np.allclose(
                        list(features.values()),
                        list(cv2dct_features.values()),
                        rtol=1e-5,
                        atol=1e-7,
                    )
                )


if __name__ == "__main__":
    unittest.main()

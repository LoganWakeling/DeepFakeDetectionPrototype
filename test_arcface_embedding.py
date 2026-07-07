from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from arcface_embedding import EMBEDDING_DIM, create_face_analyzer, extract_arcface_result


SAMPLE_IMAGES = sorted(Path("data/test_images").glob("sample*.jpg"))


class ArcFaceEmbeddingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.analyzer = create_face_analyzer()

    def test_sample_images_produce_valid_arcface_results(self) -> None:
        self.assertEqual(len(SAMPLE_IMAGES), 5)

        for image_path in SAMPLE_IMAGES:
            with self.subTest(image=str(image_path)):
                result = extract_arcface_result(image_path, analyzer=self.analyzer)

                self.assertEqual(result.embedding.shape, (EMBEDDING_DIM,))
                self.assertTrue(np.all(np.isfinite(result.embedding)))
                self.assertAlmostEqual(float(np.linalg.norm(result.embedding)), 1.0, places=4)

                self.assertEqual(result.bbox.shape, (4,))
                self.assertTrue(np.all(np.isfinite(result.bbox)))
                self.assertEqual(result.keypoints.shape, (5, 2))
                self.assertTrue(np.all(np.isfinite(result.keypoints)))

                if result.det_score is not None:
                    self.assertGreaterEqual(result.det_score, 0.0)
                    self.assertLessEqual(result.det_score, 1.0)


if __name__ == "__main__":
    unittest.main()

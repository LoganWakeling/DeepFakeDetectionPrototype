import unittest

import numpy as np

from feature_extractors.feature_extraction import crop_largest_face


class _Face:
    def __init__(self, bbox):
        self.bbox = np.asarray(bbox, dtype=np.float32)


class _Analyzer:
    def __init__(self, faces):
        self.faces = faces

    def get(self, image):
        return self.faces


class FaceCropTests(unittest.TestCase):
    def test_crops_largest_face_with_square_margin(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        analyzer = _Analyzer([_Face([10, 10, 20, 20]), _Face([80, 30, 120, 70])])
        crop = crop_largest_face(image, analyzer, margin=0.25)
        self.assertEqual(crop.shape, (60, 60, 3))

    def test_no_detection_returns_original_image(self):
        image = np.zeros((20, 30, 3), dtype=np.uint8)
        self.assertIs(crop_largest_face(image, _Analyzer([])), image)


if __name__ == "__main__":
    unittest.main()

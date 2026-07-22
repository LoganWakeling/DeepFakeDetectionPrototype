import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.create_manifest import assign_stratified_splits, build_summary, create_rows, load_config


class CreateManifestTests(unittest.TestCase):
    def test_infers_labels_metadata_and_image_properties(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real = root / "FFHQ" / "high" / "real" / "a.png"
            fake = root / "CelebDF" / "low" / "fake" / "SimSwap" / "b.jpg"
            real.parent.mkdir(parents=True)
            fake.parent.mkdir(parents=True)
            Image.new("RGB", (12, 10)).save(real)
            Image.new("RGB", (8, 6)).save(fake)
            config = load_config(None)
            config["source_datasets"] = {"FFHQ": "FFHQ", "CelebDF": "Celeb-DF"}

            rows, warnings = create_rows(root, config, 0.2)

            self.assertEqual([], warnings)
            by_class = {row["class_name"]: row for row in rows}
            self.assertEqual({"real", "fake"}, set(by_class))
            self.assertEqual("high", by_class["real"]["quality_label"])
            self.assertEqual("gan", by_class["fake"]["generator_type"])
            self.assertEqual("Celeb-DF", by_class["fake"]["source_dataset"])
            self.assertEqual(
                (8, 6, 3),
                (by_class["fake"]["width"], by_class["fake"]["height"], by_class["fake"]["channels"]),
            )
            self.assertIn(by_class["real"]["split"], {"train", "test"})

    def test_warns_when_fake_generator_is_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "fake" / "unknown_method" / "a.png"
            image.parent.mkdir(parents=True)
            Image.new("RGB", (2, 2)).save(image)

            _, warnings = create_rows(root, load_config(None), 0.2)

            self.assertTrue(any("Missing generator metadata" in warning for warning in warnings))
            self.assertTrue(any("Missing quality label" in warning for warning in warnings))

    def test_split_is_stratified_and_deterministic(self):
        rows = []
        for class_name, generator_type in (("real", "real"), ("fake", "gan")):
            for index in range(10):
                rows.append(
                    {
                        "sample_id": f"{class_name}-{index}",
                        "split": "",
                        "class_name": class_name,
                        "generator_type": generator_type,
                        "quality_label": "high",
                        "source_dataset": "example",
                    }
                )

        assign_stratified_splits(rows, 0.2)

        for class_name in ("real", "fake"):
            members = [row for row in rows if row["class_name"] == class_name]
            self.assertEqual(2, sum(row["split"] == "test" for row in members))
        original = [row["split"] for row in rows]
        for row in rows:
            row["split"] = ""
        assign_stratified_splits(rows, 0.2)
        self.assertEqual(original, [row["split"] for row in rows])

    def test_summary_reports_distributions_and_singletons(self):
        row = {
            "split": "train",
            "class_name": "real",
            "source_dataset": "FFHQ",
            "quality_label": "high",
            "generator": "real",
            "generator_type": "real",
            "manipulation_type": "none",
            "image_format": "PNG",
            "file_size_bytes": 120,
            "width": 12,
            "height": 10,
        }

        summary = build_summary(Path("/dataset"), [row], [])

        self.assertEqual(1, summary["total_images"])
        self.assertEqual({"real": 1}, summary["distributions"]["class_name"])
        self.assertEqual(1, len(summary["stratification"]["singleton_strata"]))


if __name__ == "__main__":
    unittest.main()

import random
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.collect_balanced_dataset import (
    Candidate,
    assign_grouped_splits,
    choose_balanced_dataset,
    save_quality_variants,
    separated_timestamps,
)


def candidate(source_index, family, number):
    label = 0 if family == "real" else 1
    return Candidate(
        source_index=source_index,
        source_path=Path(f"/{family}/{number}.png"),
        source_dataset=f"source-{source_index}",
        source_url="https://example.test",
        license="research",
        label=label,
        generator="real" if label == 0 else family,
        generator_type=family,
        manipulation_type="none" if label == 0 else "full_synthesis",
        source_type="still_image",
    )


class CollectBalancedDatasetTests(unittest.TestCase):
    def test_timestamps_respect_spacing(self):
        values = separated_timestamps(60, 8, 5, random.Random(7), boundary_margin=1)
        self.assertEqual(8, len(values))
        self.assertTrue(all(right - left >= 5 for left, right in zip(values, values[1:])))
        self.assertGreaterEqual(values[0], 1)
        self.assertLessEqual(values[-1], 59)

    def test_selects_all_three_fake_families_and_balanced_real_count(self):
        candidates = []
        for family in ("real", "autoencoder", "gan", "diffusion"):
            for source in (0, 1):
                candidates.extend(candidate(source, family, f"{source}-{index}") for index in range(6))
        selected = choose_balanced_dataset(candidates, 4, 12, random.Random(3))
        self.assertEqual(24, len(selected))
        self.assertEqual(12, sum(item.family == "real" for item in selected))
        for family in ("autoencoder", "gan", "diffusion"):
            family_items = [item for item in selected if item.family == family]
            self.assertEqual(4, len(family_items))
            self.assertEqual({0, 1}, {item.source_index for item in family_items})

    def test_creates_four_quality_variants_in_one_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            standardized = root / "standardized.png"
            original = root / "source.png"
            Image.new("RGB", (32, 32), "red").save(standardized)
            Image.new("RGB", (64, 64), "red").save(original)
            item = candidate(0, "gan", "one")
            rows = save_quality_variants(
                item, original, standardized, root, "parent-one", True
            )
            self.assertEqual(
                {"original", "jpeg_q75", "jpeg_q50", "jpeg_q25"},
                {row["quality_variant"] for row in rows},
            )
            self.assertTrue(all(Path(row["image_path"]).exists() for row in rows))
            assign_grouped_splits(rows, 0.15, 0.15, random.Random(1))
            self.assertEqual(1, len({row["split"] for row in rows}))


if __name__ == "__main__":
    unittest.main()

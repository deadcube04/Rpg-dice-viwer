"""Known numeric images pin feature definitions without model inference."""

from pathlib import Path
import math
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from PIL import Image

from dice_viewer.recognition.image_features import extract_image_features


class ImageFeaturesTests(unittest.TestCase):
    def test_constant_red_has_known_brightness_channels_and_zero_variance(self):
        image = Image.new("RGB", (8, 4), (255, 0, 0))
        values = extract_image_features(image)
        self.assertEqual((values["width"], values["height"], values["pixels"]), (8, 4, 32))
        self.assertEqual(values["aspect_ratio"], 2)
        self.assertEqual(values["rgb_mean"], {"red": 1, "green": 0, "blue": 0})
        self.assertEqual(values["rgb_std"], {"red": 0, "green": 0, "blue": 0})
        self.assertAlmostEqual(values["brightness_mean"], 0.299)
        self.assertAlmostEqual(values["contrast_std"], 0)
        self.assertAlmostEqual(values["laplacian_variance"], 0)
        self.assertEqual(values["laplacian_valid_pixels"], 12)

    def test_black_white_checkerboard_has_hand_calculated_population_moments(self):
        checker = (np.indices((4, 4)).sum(axis=0) % 2 * 255).astype("uint8")
        image = Image.fromarray(np.repeat(checker[:, :, None], 3, axis=2))
        values = extract_image_features(image)
        self.assertAlmostEqual(values["brightness_mean"], 0.5)
        self.assertAlmostEqual(values["contrast_std"], 0.5)
        # Four interior Laplacians alternate -4,+4: mean 0, variance 16.
        self.assertAlmostEqual(values["laplacian_variance"], 16)
        for channel in ("red", "green", "blue"):
            self.assertAlmostEqual(values["rgb_mean"][channel], 0.5)
            self.assertAlmostEqual(values["rgb_std"][channel], 0.5)

    def test_bounded_sample_preserves_original_pixels_metadata_and_dimensions(self):
        image = Image.new("RGB", (4096, 2048), (20, 50, 100))
        image.info["private"] = "PRIVATE_TOKEN"
        before = image.tobytes()
        values = extract_image_features(image)
        self.assertEqual((values["sample_width"], values["sample_height"]), (256, 128))
        self.assertEqual((values["width"], values["height"]), (4096, 2048))
        self.assertEqual(image.tobytes(), before)
        self.assertEqual(image.info, {"private": "PRIVATE_TOKEN"})
        self.assertNotIn("PRIVATE_TOKEN", str(values))
        self.assertLessEqual(values["sample_width"] * values["sample_height"], 65_536)

    def test_tiny_image_is_not_upscaled_and_has_no_sharpness_measurement(self):
        values = extract_image_features(Image.new("RGB", (1, 2)))
        self.assertEqual((values["sample_width"], values["sample_height"]), (1, 2))
        self.assertIsNone(values["laplacian_variance"])
        self.assertEqual(values["laplacian_valid_pixels"], 0)
        self.assertEqual(values["brightness_mean"], 0)

    def test_portrait_sampling_uses_height_as_largest_side(self):
        values = extract_image_features(Image.new("RGB", (100, 1000)))
        self.assertEqual((values["sample_width"], values["sample_height"]), (26, 256))

    def test_nonuniform_downsampling_pins_bilinear_population_contrast(self):
        image = Image.new("RGB", (512, 8), (0, 0, 0))
        image.paste((255, 255, 255), (256, 0, 512, 8))
        values = extract_image_features(image)
        # At a 2:1 BILINEAR reduction, the step yields 127 black columns,
        # values 32 and 223, then 127 white columns. Mean is exactly 0.5.
        expected_std = math.sqrt((254 * 0.25 + 2 * (0.5 - 32 / 255) ** 2) / 256)
        self.assertEqual((values["sample_width"], values["sample_height"]), (256, 4))
        self.assertAlmostEqual(values["brightness_mean"], 0.5)
        self.assertAlmostEqual(values["contrast_std"], expected_std)
        self.assertAlmostEqual(values["rgb_std"]["red"], expected_std)

    def test_non_rgb_is_rejected_instead_of_changing_the_model_input(self):
        with self.assertRaises(ValueError):
            extract_image_features(Image.new("L", (8, 8)))


if __name__ == "__main__":
    unittest.main()

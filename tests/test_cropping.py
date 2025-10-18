import unittest
from unittest.mock import patch
import numpy as np
import cv2
from preprocess.cropping import ROICropping


class TestROICropping(unittest.TestCase):

    def setUp(self):
        """Setup for tests"""
        self.cropper = ROICropping(target_size=(512, 512))

        # Create test image (breast on right side)
        self.test_image = np.ones((300, 400), dtype=np.uint8) * 50
        self.test_image[50:250, 200:350] = 200  # Simulated breast region

    def test_initialization(self):
        """Test class initialization"""
        cropper = ROICropping(target_size=(256, 256), margin_mm=10.0)
        self.assertEqual(cropper.target_size, (256, 256))
        self.assertEqual(cropper.margin_mm, 10.0)

    def test_breast_mask(self):
        """Test breast mask generation"""
        mask = self.cropper.breast_mask(self.test_image)

        self.assertEqual(mask.shape, self.test_image.shape)
        self.assertEqual(mask.dtype, np.uint8)
        self.assertTrue(np.all(np.unique(mask) == [0, 1]))

    def test_erode_mask_mm(self):
        """Test mask erosion"""
        mask = np.ones((100, 100), dtype=np.uint8)
        eroded = self.cropper.erode_mask_mm(mask)

        self.assertEqual(eroded.shape, mask.shape)
        self.assertLessEqual(eroded.sum(), mask.sum())

    def test_remove_pectoral_MLO(self):
        """Test pectoral removal doesn't modify input"""
        original_mask = np.ones((100, 100), dtype=np.uint8)
        image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)

        result = self.cropper.remove_pectoral_MLO(image, original_mask)

        # Should not modify original
        self.assertEqual(original_mask.sum(), 100 * 100)
        # Should return modified copy
        self.assertLess(result.sum(), original_mask.sum())

    def test_orient_left_breast_on_right(self):
        """Test auto-orientation when breast is on right"""
        # Create image with breast on right (brighter on right)
        image = np.ones((100, 200), dtype=np.uint8) * 50
        image[:, 100:] = 200

        oriented = self.cropper.orient_left(image)

        # After orientation, left side should be brighter
        left_mean = np.mean(oriented[:, :100])
        right_mean = np.mean(oriented[:, 100:])
        self.assertGreater(left_mean, right_mean)

    def test_orient_by_laterality_right(self):
        """Test orientation with known right laterality"""
        image = np.array([[1, 2], [3, 4]], dtype=np.uint8)

        oriented = self.cropper.orient_by_laterality(image, 'R')

        # Should be flipped
        expected = np.array([[2, 1], [4, 3]], dtype=np.uint8)
        np.testing.assert_array_equal(oriented, expected)

    def test_orient_by_laterality_left(self):
        """Test orientation with known left laterality"""
        image = np.array([[1, 2], [3, 4]], dtype=np.uint8)

        oriented = self.cropper.orient_by_laterality(image, 'L')

        # Should remain unchanged
        np.testing.assert_array_equal(oriented, image)

    def test_orient_by_laterality_invalid(self):
        """Test orientation with invalid laterality falls back to auto"""
        image = np.ones((100, 200), dtype=np.uint8) * 50
        image[:, 100:] = 200

        oriented = self.cropper.orient_by_laterality(image, 'INVALID')

        # Should use auto-orientation
        self.assertIsInstance(oriented, np.ndarray)
        self.assertEqual(oriented.shape, image.shape)

    def test_bbox_with_margin_mm_aniso(self):
        """Test bounding box calculation"""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[25:75, 25:75] = 1

        bbox = self.cropper.bbox_with_margin_mm_aniso(mask)
        x_min, y_min, x_max, y_max = bbox

        self.assertLess(x_min, 25)  # Should have margin
        self.assertLess(y_min, 25)
        self.assertGreater(x_max, 75)
        self.assertGreater(y_max, 75)

    def test_bbox_with_empty_mask(self):
        """Test bounding box with empty mask returns full image"""
        empty_mask = np.zeros((100, 100), dtype=np.uint8)

        bbox = self.cropper.bbox_with_margin_mm_aniso(empty_mask)
        x_min, y_min, x_max, y_max = bbox

        self.assertEqual(x_min, 0)
        self.assertEqual(y_min, 0)
        self.assertEqual(x_max, 100)
        self.assertEqual(y_max, 100)

    def test_crop_to_roi(self):
        """Test cropping to ROI"""
        image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[25:75, 25:75] = 1

        cropped = self.cropper.crop_to_roi(image, mask)

        self.assertEqual(cropped.shape, (512, 512))  # Target size

    def test_process_one(self):
        """Test generic processing pipeline"""
        result = self.cropper.process_one(self.test_image)

        self.assertEqual(result.shape, (512, 512))
        self.assertEqual(result.dtype, np.uint8)

    def test_process_with_metadata_mlo_left(self):
        """Test processing with MLO left metadata"""
        result = self.cropper.process_with_metadata(
            self.test_image,
            view='MLO',
            laterality='L'
        )

        self.assertEqual(result.shape, (512, 512))

    def test_process_with_metadata_mlo_right(self):
        """Test processing with MLO right metadata"""
        result = self.cropper.process_with_metadata(
            self.test_image,
            view='MLO',
            laterality='R'
        )

        self.assertEqual(result.shape, (512, 512))

    def test_process_with_metadata_cc(self):
        """Test processing with CC view (no pectoral removal)"""
        result = self.cropper.process_with_metadata(
            self.test_image,
            view='CC',
            laterality='L'
        )

        self.assertEqual(result.shape, (512, 512))

    def test_process_with_metadata_no_view(self):
        """Test processing without view info"""
        result = self.cropper.process_with_metadata(
            self.test_image,
            view=None,
            laterality='L'
        )

        self.assertEqual(result.shape, (512, 512))

    def test_process_with_metadata_no_laterality(self):
        """Test processing without laterality info"""
        result = self.cropper.process_with_metadata(
            self.test_image,
            view='MLO',
            laterality=None
        )

        self.assertEqual(result.shape, (512, 512))

    def test_process_invalid_image_dimensions(self):
        """Test error with invalid image dimensions"""
        invalid_image = np.ones((100, 100, 3), dtype=np.uint8)  # 3D image

        with self.assertRaises(ValueError):
            self.cropper.process_one(invalid_image)

        with self.assertRaises(ValueError):
            self.cropper.process_with_metadata(invalid_image, 'MLO', 'L')

    def test_srp_respect_no_data_handling(self):
        """Test that class doesn't handle data extraction (SRP)"""
        # The class should only accept already-extracted view/laterality
        # Not dictionaries or complex data structures
        result = self.cropper.process_with_metadata(
            self.test_image,
            view='MLO',  # Simple string
            laterality='L'  # Simple string
        )

        self.assertEqual(result.shape, (512, 512))


if __name__ == '__main__':
    unittest.main()
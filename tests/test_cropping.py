# test_cropping.py
import pytest
import numpy as np
import cv2 as cv
from unittest.mock import Mock, MagicMock
from preprocess.cropping import Cropping


class TestCropping:

    def setup_method(self):
        """Setup pour chaque test"""
        # Mocks des dépendances
        self.mock_config = Mock()
        self.mock_loader = Mock()
        self.mock_dataset_manager = Mock()

        # Configuration ROI simulée
        self.mock_config.roi_config = {
            "min_area_px": 1000,
            "morpho_disk": 3,
            "use_convex_hull": True,
            "inset_mm_y": 2.0,
            "inset_mm_x": 0.8,
            "margins_mm": {
                "CC": {"x": 7.0, "y": 6.5},
                "MLO": {"x": 9.0, "y": 6.5},
            },
            "extra_right_mm": 3.0,
            "max_margin_frac": 0.08,
            "enable_profile_fallback": True,
            "touch_crit_thresh": 1,
            "soft_tanh_k": 3.0
        }

        self.cropper = Cropping(self.mock_config, self.mock_loader, self.mock_dataset_manager)

    def test_breast_mask_basic(self):
        """Test de segmentation basique"""
        # Création d'une image synthétique avec un "sein"
        img = np.zeros((100, 100), dtype=np.float32)
        img[20:80, 20:80] = 0.8  # Zone claire simulant un sein

        mask, bbox = self.cropper.breast_mask(img)

        assert mask.shape == img.shape
        assert mask.dtype == bool
        assert len(bbox) == 4
        assert bbox[0] <= bbox[2]  # y0 <= y1
        assert bbox[1] <= bbox[3]  # x0 <= x1

    def test_breast_mask_empty(self):
        """Test avec image vide"""
        img = np.zeros((100, 100), dtype=np.float32)

        mask, bbox = self.cropper.breast_mask(img)

        assert not np.any(mask)  # Masque doit être vide
        assert bbox == (0, 0, 100, 100)  # Bbox couvre toute l'image

    def test_orient_left_right_breast(self):
        """Test d'orientation pour sein droit"""
        img = np.zeros((100, 100), dtype=np.float32)
        mask = np.zeros((100, 100), dtype=bool)

        # Simuler un sein à droite (doit être flip)
        mask[30:70, 60:90] = True

        img_orient, mask_orient, flipped = self.cropper.orient_left(img, mask)

        assert flipped == True
        assert img_orient.shape == img.shape
        assert mask_orient.shape == mask.shape

    def test_orient_left_left_breast(self):
        """Test d'orientation pour sein gauche"""
        img = np.zeros((100, 100), dtype=np.float32)
        mask = np.zeros((100, 100), dtype=bool)

        # Simuler un sein à gauche (ne doit pas être flip)
        mask[30:70, 10:40] = True

        img_orient, mask_orient, flipped = self.cropper.orient_left(img, mask)

        assert flipped == False
        assert np.array_equal(img_orient, img)
        assert np.array_equal(mask_orient, mask)

    def test_erode_mask_mm(self):
        """Test de l'érosion du masque"""
        mask = np.zeros((100, 100), dtype=bool)
        mask[10:90, 10:90] = True  # Grand rectangle

        spacing = (1.0, 1.0)  # 1mm par pixel

        eroded = self.cropper.erode_mask_mm(mask, spacing)

        assert eroded.shape == mask.shape
        assert eroded.dtype == bool
        # L'érosion doit réduire la surface
        assert eroded.sum() <= mask.sum()

    def test_soft_tanh_norm(self):
        """Test de la normalisation soft-tanh"""
        # Création de données de test
        data = np.random.normal(100, 50, (50, 50)).astype(np.float32)

        normalized = self.cropper.soft_tanh_norm(data)

        assert normalized.shape == data.shape
        assert normalized.dtype == np.float32
        # Doit être dans [0, 1]
        assert normalized.min() >= 0
        assert normalized.max() <= 1

    def test_bbox_with_margin_mm_aniso(self):
        """Test de l'expansion de bbox avec marges"""
        bbox = (10, 10, 50, 50)  # y0, x0, y1, x1
        spacing = (0.1, 0.1)  # 0.1mm par pixel
        h, w = 100, 100
        view = "CC"

        new_bbox = self.cropper.bbox_with_margin_mm_aniso(bbox, spacing, h, w, view)

        assert len(new_bbox) == 4
        # La nouvelle bbox doit être plus grande ou égale
        assert new_bbox[0] <= bbox[0]  # y0 réduit
        assert new_bbox[1] <= bbox[1]  # x0 réduit
        assert new_bbox[2] >= bbox[2]  # y1 augmenté
        assert new_bbox[3] >= bbox[3]  # x1 augmenté

    def test_process_one_integration(self):
        """Test d'intégration de process_one"""
        # Mock du chargement DICOM
        test_img = np.random.rand(200, 200).astype(np.float32)
        test_spacing = (0.1, 0.1)
        self.mock_loader.load_dicom_for_roi.return_value = (test_img, test_spacing)
        self.mock_loader.load_dicom_linear.return_value = (test_img, test_spacing)
        self.mock_dataset_manager.get_dicom_path.return_value = "/fake/path.dcm"

        result = self.cropper.process_one(
            patient_id=123,
            image_id=456,
            laterality="L",
            view="CC",
            dicom_path="/fake/path.dcm"
        )

        # Vérifications de base
        assert 'patient_id' in result
        assert 'image_id' in result
        assert 'bbox' in result
        assert 'crop_model' in result
        assert 'raw_crop' in result
        assert result['patient_id'] == 123
        assert result['image_id'] == 456
        assert result['laterality'] == "L"
        assert result['view'] == "CC"

        # Vérification des shapes
        assert len(result['bbox']) == 4
        assert result['crop_model'].ndim == 2
        assert result['raw_crop'].ndim == 2

    def test_remove_pectoral_mlo_non_mlo(self):
        """Test que remove_pectoral_MLO ne fait rien pour les vues non-MLO"""
        img = np.random.rand(100, 100).astype(np.float32)
        mask = np.ones((100, 100), dtype=bool)

        result = self.cropper.remove_pectoral_MLO(img, mask, "L", "CC")

        assert np.array_equal(result, mask)


if __name__ == "__main__":
    # Exécution des tests
    pytest.main([__file__, "-v"])
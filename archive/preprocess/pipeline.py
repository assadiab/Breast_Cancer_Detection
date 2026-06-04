"""
PreprocessPipeline — orchestrateur de preprocessing mammographique.

5 modes disponibles :
  raw          : DICOM → float32 normalisé [0,1], aucun traitement
  crop_only    : ROI crop uniquement (pas de windowing)
  window_only  : Windowing adaptatif uniquement (pas de crop)
  full         : Crop + Windowing adaptatif → image prête pour le DL
  full_iso     : full + rééchantillonnage isotropique (pour des analyses morphologiques)

Usage:
    from preprocess.pipeline import PreprocessPipeline
    pipeline = PreprocessPipeline(config, mode="full", target_hw=(1024, 512))
    img = pipeline.process_one(patient_id, image_id, laterality, view, density, dicom_path)
    # → np.ndarray float32 (H, W), valeurs dans [0, 1]
"""

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_VALID_MODES = ("raw", "crop_only", "window_only", "full", "full_iso")


class PreprocessPipeline:
    """
    Pipeline de preprocessing configurable pour mammographies RSNA.

    Args:
        config:      Instance de core.configuration.Config
        mode:        Mode de preprocessing (voir module docstring)
        target_hw:   (H, W) de sortie après resize final. None = pas de resize.
    """

    def __init__(self, config, mode: str = "full", target_hw: Optional[Tuple[int, int]] = (1024, 512)):
        if mode not in _VALID_MODES:
            raise ValueError(f"mode doit être parmi {_VALID_MODES}, reçu: {mode!r}")

        self.config = config
        self.mode = mode
        self.target_hw = target_hw

        # Imports locaux pour éviter les dépendances circulaires
        from core.loader import Loader
        self._loader = Loader(config)

        self._cropping = None
        self._windowing = None
        self._resampler = None

        if mode in ("crop_only", "full", "full_iso"):
            from preprocess.cropping import Cropping
            from core.dataset_manager import DatasetManager
            self._dm = DatasetManager(config, self._loader)
            self._cropping = Cropping(config, self._loader, self._dm)

        if mode in ("window_only", "full", "full_iso"):
            from preprocess.windowing import Windowing
            self._windowing = Windowing(
                preserve_range=config.config.get("windowing", {}).get("preserve_range", (0.0, 1.0))
                if hasattr(config, "config") else (0.0, 1.0)
            )

        if mode == "full_iso":
            from preprocess.resampler import Resampler
            self._resampler = Resampler(config)

        logger.info("PreprocessPipeline initialized — mode=%s, target_hw=%s", mode, target_hw)

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    def process_one(
        self,
        patient_id: int,
        image_id: int,
        laterality: str,
        view: str,
        density: str,
        dicom_path: str,
    ) -> np.ndarray:
        """
        Prétraite une seule image selon le mode configuré.

        Args:
            patient_id:  Identifiant patient
            image_id:    Identifiant image
            laterality:  'L' ou 'R'
            view:        'CC' ou 'MLO'
            density:     'A', 'B', 'C' ou 'D' (fallback 'B' si invalide)
            dicom_path:  Chemin absolu vers le fichier .dcm

        Returns:
            np.ndarray float32 (H, W) dans [0, 1], shape = target_hw si défini.
        """
        # Normaliser la densité
        density = density.upper() if density.upper() in ("A", "B", "C", "D") else "B"

        if self.mode == "raw":
            return self._mode_raw(dicom_path)

        elif self.mode == "crop_only":
            return self._mode_crop_only(patient_id, image_id, laterality, view, dicom_path)

        elif self.mode == "window_only":
            return self._mode_window_only(dicom_path, density)

        elif self.mode == "full":
            return self._mode_full(patient_id, image_id, laterality, view, density, dicom_path)

        elif self.mode == "full_iso":
            return self._mode_full_iso(patient_id, image_id, laterality, view, density, dicom_path)

        else:
            raise RuntimeError(f"Mode inconnu : {self.mode}")

    # ------------------------------------------------------------------
    # Modes internes
    # ------------------------------------------------------------------

    def _mode_raw(self, dicom_path: str) -> np.ndarray:
        """Chargement brut : DICOM → float32 normalisé [0,1], pas de crop ni windowing."""
        loaded = self._loader.load_dicom_full(dicom_path)
        img = loaded["img01"]  # float32 [0,1] robuste
        return self._resize(img)

    def _mode_crop_only(
        self, patient_id, image_id, laterality, view, dicom_path
    ) -> np.ndarray:
        """Crop ROI uniquement, pas de windowing adaptatif."""
        result = self._cropping.process_one(patient_id, image_id, laterality, view, dicom_path)
        raw_crop = result["raw_crop"]
        # Normalisation simple [0,1] car pas de windowing adaptatif
        img01 = self._loader.robust_normalize01(raw_crop, 0.5, 99.5)
        return self._resize(img01)

    def _mode_window_only(self, dicom_path: str, density: str) -> np.ndarray:
        """Windowing adaptatif uniquement (pas de crop)."""
        loaded = self._loader.load_dicom_full(dicom_path)
        img_raw = loaded["img_raw"]
        img_windowed = self._windowing.process_one(img_raw, density=density)
        return self._resize(img_windowed)

    def _mode_full(
        self, patient_id, image_id, laterality, view, density, dicom_path
    ) -> np.ndarray:
        """
        Pipeline complet : Crop ROI → Windowing adaptatif → Resize.
        C'est le mode recommandé pour l'entraînement du modèle.
        """
        # Crop retourne raw_crop (valeurs calibrées, non normalisées — fix C2)
        result = self._cropping.process_one(patient_id, image_id, laterality, view, dicom_path)
        raw_crop = result["raw_crop"]

        # Windowing adaptatif sur les valeurs brutes (fix W1: pas de re-stretch)
        img_windowed = self._windowing.process_one(raw_crop, density=density)

        return self._resize(img_windowed)

    def _mode_full_iso(
        self, patient_id, image_id, laterality, view, density, dicom_path
    ) -> np.ndarray:
        """
        Pipeline complet + rééchantillonnage isotropique.
        Utile pour des analyses morphologiques. Déconseillé pour le DL pur
        (ajoute ~300ms/image sans bénéfice démontré sur RSNA).
        """
        result = self._cropping.process_one(patient_id, image_id, laterality, view, dicom_path)
        raw_crop = result["raw_crop"]
        spacing = (result["spacing"]["row"], result["spacing"]["col"])

        img_windowed = self._windowing.process_one(raw_crop, density=density)

        # Rééchantillonnage isotropique
        img_iso = self._resampler.resample(img_windowed, spacing)

        return self._resize(img_iso)

    # ------------------------------------------------------------------
    # Utilitaire
    # ------------------------------------------------------------------

    def _resize(self, img: np.ndarray) -> np.ndarray:
        """Redimensionne vers target_hw si défini, sinon retourne img tel quel."""
        if self.target_hw is None:
            return img.astype(np.float32)
        H, W = self.target_hw
        if img.shape[0] == H and img.shape[1] == W:
            return img.astype(np.float32)
        interp = cv2.INTER_AREA if img.shape[0] > H else cv2.INTER_LINEAR
        return cv2.resize(img, (W, H), interpolation=interp).astype(np.float32)

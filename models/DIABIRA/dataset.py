import os
import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from monai.transforms import (
    Compose,
    NormalizeIntensity,
    RandGaussianNoise,
    Rand2DElastic,
    ToTensor,
)

logger = logging.getLogger(__name__)

# Modes de preprocessing acceptés
_PREPROCESS_MODES = ("raw", "crop_only", "window_only", "full", "full_iso")


def _build_train_transforms() -> Compose:
    """MONAI transforms pour le train — augmentations médicales."""
    return Compose([
        NormalizeIntensity(nonzero=True),
        RandGaussianNoise(prob=0.2, mean=0.0, std=0.01),
        Rand2DElastic(
            prob=0.2,
            spacing=(20, 20),
            magnitude_range=(1, 3),
            padding_mode="reflection",
        ),
        ToTensor(),
    ])


def _build_val_transforms() -> Compose:
    """MONAI transforms pour val/test — normalisation uniquement."""
    return Compose([
        NormalizeIntensity(nonzero=True),
        ToTensor(),
    ])


class MammographyDataset(Dataset):
    """
    Dataset PyTorch pour mammographies RSNA.

    Charge les images DICOM via PreprocessPipeline et applique des
    transforms MONAI. Supporte train / val / test.

    Args:
        df: DataFrame avec colonnes patient_id, image_id, cancer, density,
            laterality, view
        images_dir: dossier contenant les DICOM.
            Structure attendue : {images_dir}/{patient_id}/{image_id}.dcm
            (ou flat : {images_dir}/{patient_id}_{image_id}.dcm si flat_dicoms=True)
        target_size: (H, W) final après resize
        mode: "train" | "val" | "test"
        preprocess_mode: mode du pipeline preprocessing :
            "raw"          — chargement brut normalisé [0,1]
            "crop_only"    — crop ROI uniquement
            "window_only"  — windowing adaptatif uniquement
            "full"         — crop + windowing (recommandé pour l'entraînement)
            "full_iso"     — full + rééchantillonnage isotropique
            None / False   — désactive le preprocessing (raw fallback)
        pipeline: instance PreprocessPipeline déjà construite (optionnel).
            Si fourni, preprocess_mode est ignoré.
        flat_dicoms: si True, cherche les DICOM à plat
            ({images_dir}/{patient_id}_{image_id}.dcm)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        images_dir: str,
        target_size: Tuple[int, int] = (512, 512),
        mode: str = "train",
        preprocess_mode: Optional[str] = "full",
        pipeline=None,
        flat_dicoms: bool = False,
    ):
        assert mode in ("train", "val", "test"), f"mode doit être train/val/test, reçu: {mode}"
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.target_size = target_size
        self.mode = mode
        self.flat_dicoms = flat_dicoms

        self.transforms = _build_train_transforms() if mode == "train" else _build_val_transforms()

        # Pipeline de preprocessing
        if pipeline is not None:
            self._pipeline = pipeline
        elif preprocess_mode and preprocess_mode in _PREPROCESS_MODES:
            self._pipeline = self._build_pipeline(preprocess_mode)
        else:
            # Pas de pipeline → chargement brut via Loader seul
            self._pipeline = None
            self._init_raw_loader()

        logger.info(
            "MammographyDataset [%s] — %d images, cancer rate: %.3f, preprocess: %s",
            mode, len(self.df), self.df['cancer'].mean() if 'cancer' in self.df.columns else float('nan'),
            preprocess_mode if pipeline is None else "external pipeline",
        )

    # ------------------------------------------------------------------
    # Init helpers
    # ------------------------------------------------------------------

    def _build_pipeline(self, preprocess_mode: str):
        """Construit et retourne une PreprocessPipeline."""
        from preprocess.pipeline import PreprocessPipeline
        from core.configuration import Config

        csv_path = os.path.join(os.path.dirname(self.images_dir), "train.csv")
        if not os.path.isfile(csv_path):
            csv_path = "/tmp/_mammo_tmp.csv"
            self.df.to_csv(csv_path, index=False)

        config = Config(csv_path=csv_path, images_dir=self.images_dir)
        H, W = self.target_size
        return PreprocessPipeline(config, mode=preprocess_mode, target_hw=(H, W))

    def _init_raw_loader(self) -> None:
        """Initialise un Loader minimal pour le mode sans pipeline."""
        from core.configuration import Config
        from core.loader import Loader

        csv_path = os.path.join(os.path.dirname(self.images_dir), "train.csv")
        if not os.path.isfile(csv_path):
            csv_path = "/tmp/_mammo_tmp.csv"
            self.df.to_csv(csv_path, index=False)

        self._config = Config(csv_path=csv_path, images_dir=self.images_dir)
        self._loader = Loader(self._config)

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        row = self.df.iloc[idx]
        patient_id = int(row["patient_id"])
        image_id = int(row["image_id"])
        label = float(row.get("cancer", 0))
        density = str(row.get("density", "B"))
        laterality = str(row.get("laterality", "L"))
        view = str(row.get("view", "CC"))

        dicom_path = self._get_dicom_path(patient_id, image_id)

        try:
            image = self._load_image(dicom_path, patient_id, image_id, laterality, view, density)
        except Exception as e:
            logger.warning("Erreur chargement %s: %s — image nulle utilisée", dicom_path, e)
            image = np.zeros((1, *self.target_size), dtype=np.float32)

        # Transforms MONAI — NormalizeIntensity attend (C, H, W)
        image_tensor = self.transforms(image)  # → torch.Tensor (1, H, W)

        return (
            image_tensor.float(),
            torch.tensor(label, dtype=torch.float32),
            patient_id,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_dicom_path(self, patient_id: int, image_id: int) -> str:
        """Construit le chemin DICOM selon la structure (flat ou hiérarchique)."""
        if self.flat_dicoms:
            return os.path.join(self.images_dir, f"{patient_id}_{image_id}.dcm")
        return os.path.join(self.images_dir, str(patient_id), f"{image_id}.dcm")

    def _load_image(
        self,
        dicom_path: str,
        patient_id: int,
        image_id: int,
        laterality: str,
        view: str,
        density: str,
    ) -> np.ndarray:
        """
        Charge et prétraite une image DICOM → numpy (1, H, W) float32 [0, 1].

        Si un pipeline est disponible, délègue entièrement à PreprocessPipeline.
        Sinon, fait un chargement brut + resize.
        """
        import cv2

        if self._pipeline is not None:
            img = self._pipeline.process_one(
                patient_id, image_id, laterality, view, density, dicom_path
            )
            # pipeline retourne (H, W) → on ajoute la dim canal
            return img[np.newaxis, ...].astype(np.float32)

        # Fallback brut (mode raw via Loader)
        loaded = self._loader.load_dicom_full(dicom_path)
        img01 = loaded["img01"]
        H, W = self.target_size
        img_resized = cv2.resize(img01, (W, H), interpolation=cv2.INTER_AREA)
        return img_resized[np.newaxis, ...].astype(np.float32)

    @staticmethod
    def patient_level_aggregate(
        preds: np.ndarray,
        patient_ids: np.ndarray,
        method: str = "max",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Agrège les prédictions image→patient.

        Args:
            preds: (N,) probabilités par image
            patient_ids: (N,) identifiants patients
            method: "max" ou "mean"
        Returns:
            patient_preds: (P,) probabilités par patient
            unique_patients: (P,) identifiants
        """
        unique = np.unique(patient_ids)
        agg = np.zeros(len(unique))
        for i, pid in enumerate(unique):
            mask = patient_ids == pid
            agg[i] = preds[mask].max() if method == "max" else preds[mask].mean()
        return agg, unique

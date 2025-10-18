# cropping.py
import numpy as np
import pandas as pd
import os
import json
import math
import cv2 as cv
from pathlib import Path
from tqdm.auto import tqdm
from typing import Dict, Any, Tuple, List
import matplotlib.pyplot as plt


class Cropping:
    """
    Classe SRP pour le cropping ROI des mammographies
    Utilise les autres classes pour le chargement et la configuration
    """

    def __init__(self, config, loader, dataset_manager):
        self.config = config
        self.loader = loader
        self.dataset_manager = dataset_manager
        self.roi_cfg = config.roi_config

    # ========== SEGMENTATION ET TRAITEMENT ROI ==========

    def breast_mask(self, img01: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Segmentation du sein"""
        min_area = self.roi_cfg.get("min_area_px", 12000)
        morpho_disk = self.roi_cfg.get("morpho_disk", 5)
        use_hull = self.roi_cfg.get("use_convex_hull", True)

        h, w = img01.shape
        if not np.any(img01 > 0):
            return np.zeros((h, w), bool), (0, 0, h, w)

        # Seuillage d'Otsu
        u8 = (np.clip(img01, 0, 1) * 255).astype(np.uint8)
        thr_val, _ = cv.threshold(u8, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        thr = max(thr_val / 255.0, 0.05)
        mask = (img01 > thr).astype(np.uint8)

        # Fermeture morphologique
        k = 2 * morpho_disk + 1
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k, k))
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

        # Plus grande composante connexe
        num, labels, stats, _ = cv.connectedComponentsWithStats(mask, 8)
        if num <= 1:
            return np.zeros((h, w), bool), (0, 0, h, w)

        idx = int(np.argmax(stats[1:, cv.CC_STAT_AREA])) + 1
        if stats[idx, cv.CC_STAT_AREA] < min_area:
            return np.zeros((h, w), bool), (0, 0, h, w)

        comp = (labels == idx).astype(np.uint8)

        # Enveloppe convexe
        if use_hull:
            cnts, _ = cv.findContours(comp, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            if len(cnts):
                hull = cv.convexHull(max(cnts, key=cv.contourArea))
                comp = np.zeros_like(comp)
                cv.fillConvexPoly(comp, hull, 1)

        ys, xs = np.where(comp > 0)
        bbox = (int(ys.min()), int(xs.min()), int(ys.max() + 1), int(xs.max() + 1))

        return comp.astype(bool), bbox

    def remove_pectoral_MLO(self, img01: np.ndarray, mask: np.ndarray, laterality: str, view: str) -> np.ndarray:
        """Retrait du muscle pectoral pour les vues MLO"""
        if str(view).upper() != "MLO":
            return mask

        h, w = img01.shape

        # Définition de la région d'intérêt selon la latéralité
        if str(laterality).upper() == "R":
            roi = img01[0:int(0.45 * h), int(0.55 * w):w]
            xoff = int(0.55 * w)
            flip = False
        else:
            roi = img01[0:int(0.45 * h), 0:int(0.45 * w)]
            xoff = 0
            flip = True

        # Détection de lignes
        edges = cv.Canny((roi * 255).astype(np.uint8), 30, 90)
        if flip:
            edges = np.fliplr(edges)

        lines = cv.HoughLines(edges, 1, np.pi / 180, threshold=80)
        if lines is None:
            return mask

        # Sélection de la meilleure ligne
        best, score = None, -1
        for rho, theta in lines[:, 0, :]:
            sc = 90 - min(90, abs(theta * 180 / np.pi - 45))
            if sc > score:
                best, score = (rho, theta), sc

        if best is None:
            return mask

        rho, theta = best
        a, b = math.cos(theta), math.sin(theta)
        x0, y0 = a * rho, b * rho

        # Points de la ligne
        p1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
        p2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))

        # Création du triangle pectoral
        if str(laterality).upper() == "R":
            tri = np.array([
                [p1[0] + xoff, p1[1]],
                [p2[0] + xoff, p2[1]],
                [w - 1, 0]
            ], np.int32)
        else:
            p1x = (roi.shape[1] - 1 - p1[0]) + xoff
            p2x = (roi.shape[1] - 1 - p2[0]) + xoff
            tri = np.array([
                [p1x, p1[1]],
                [p2x, p2[1]],
                [0, 0]
            ], np.int32)

        pect = np.zeros_like(mask, np.uint8)
        cv.fillConvexPoly(pect, tri, 1)

        before = mask.sum()
        newm = mask & (pect == 0)

        return newm if newm.sum() >= 0.75 * max(before, 1) else mask

    def erode_mask_mm(self, mask_bool: np.ndarray, spacing_mm: Tuple[float, float]) -> np.ndarray:
        """Érosion du masque en mm"""
        mm_y = self.roi_cfg.get("inset_mm_y", 2.0)
        mm_x = self.roi_cfg.get("inset_mm_x", 0.8)

        if (mm_y <= 0) and (mm_x <= 0):
            return mask_bool

        row_mm, col_mm = spacing_mm
        ky = max(1, int(round(mm_y / max(row_mm, 1e-6))))
        kx = max(1, int(round(mm_x / max(col_mm, 1e-6))))

        K = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * kx + 1, 2 * ky + 1))
        eroded = cv.erode(mask_bool.astype(np.uint8), K, iterations=1).astype(bool)

        return eroded if eroded.sum() >= 0.70 * max(mask_bool.sum(), 1) else mask_bool

    def orient_left(self, img01: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Orientation standard (sein gauche)"""
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return img01, mask, False

        flipped = (xs.mean() > img01.shape[1] / 2)
        if flipped:
            return (np.ascontiguousarray(np.fliplr(img01)),
                    np.ascontiguousarray(np.fliplr(mask)), True)

        return img01, mask, False

    def bbox_with_margin_mm_aniso(self, bbox: Tuple[int, int, int, int], spacing_mm: Tuple[float, float],
                                  h: int, w: int, view: str) -> Tuple[int, int, int, int]:
        """Expansion de la bbox avec marges en mm"""
        y0, x0, y1, x1 = map(int, bbox)

        margins = self.roi_cfg["margins_mm"].get(view.upper(), {"x": 10.0, "y": 8.0})
        margin_mm_y = margins["y"]
        margin_mm_x = margins["x"]
        clamp_frac = self.roi_cfg.get("max_margin_frac", 0.1)

        dy = int(round(margin_mm_y / max(spacing_mm[0], 1e-6)))
        dx = int(round(margin_mm_x / max(spacing_mm[1], 1e-6)))

        dy = min(dy, int(round(clamp_frac * h)))
        dx = min(dx, int(round(clamp_frac * w)))

        y0 = max(0, y0 - dy)
        y1 = min(h, y1 + dy)
        x0 = max(0, x0 - dx)
        x1 = min(w, x1 + dx)

        # Bonus marge côté mamelon
        extra_right_mm = self.roi_cfg.get("extra_right_mm", 3.0)
        x1 = min(w, x1 + int(round(extra_right_mm / max(spacing_mm[1], 1e-6))))

        return y0, x0, y1, x1

    def _smooth_1d(self, v: np.ndarray, frac: float = 0.01) -> np.ndarray:
        """Lissage 1D"""
        k = max(3, int(len(v) * frac))
        k = k if k % 2 == 1 else (k + 1)
        kernel = np.ones(k) / k
        return np.convolve(v, kernel, mode="same")

    def profile_crop_box(self, img01: np.ndarray, pad_frac: float = 0.02) -> Tuple[int, int, int, int]:
        """Fallback basé sur les profils"""
        H, W = img01.shape

        # Profil horizontal
        col_sum = self._smooth_1d(img01.sum(axis=0))
        col_std = self._smooth_1d(img01.std(axis=0))
        s = (col_sum * col_std)
        xs = np.where(s > 0.05 * (s.max() + 1e-6))[0]
        if len(xs) == 0:
            return 0, 0, H, W
        x0, x1 = xs.min(), xs.max()

        # Profil vertical
        row_sum = self._smooth_1d(img01.sum(axis=1))
        row_std = self._smooth_1d(img01.std(axis=1))
        ss = (row_sum * row_std)
        ys = np.where(ss > 0.05 * (ss.max() + 1e-6))[0]
        if len(ys) == 0:
            return 0, 0, H, W
        y0, y1 = ys.min(), ys.max()

        pad_x = int(W * pad_frac)
        pad_y = int(H * pad_frac)

        y0 = max(0, y0 - pad_y)
        y1 = min(H, y1 + pad_y)
        x0 = max(0, x0 - pad_x)
        x1 = min(W, x1 + pad_x)

        return y0, x0, y1, x1

    def refine_vertical_bounds(self, mask_bool: np.ndarray, y0: int, y1: int,
                               spacing_mm: Tuple[float, float]) -> Tuple[int, int]:
        """Raffinement des limites verticales"""
        H, _ = mask_bool.shape
        rows = np.where(mask_bool.any(axis=1))[0]
        if rows.size == 0:
            return y0, y1

        occ = mask_bool.sum(axis=1).astype(np.float32)
        csum = np.cumsum(occ)
        total = float(csum[-1]) if csum.size else 0.0
        if total <= 0:
            return y0, y1

        min_keep_frac = 0.98
        pad_mm = 3.0

        lo_cut = int(np.searchsorted(csum, (1.0 - min_keep_frac) * total))
        hi_cut = int(np.searchsorted(csum, min_keep_frac * total))

        pad_y = int(round(pad_mm / max(spacing_mm[0], 1e-6)))
        y0_new = max(0, lo_cut - pad_y)
        y1_new = min(H, hi_cut + pad_y)

        y0_new = max(y0, y0_new)
        y1_new = min(y1, y1_new)

        return (y0, y1) if y1_new <= y0_new else (y0_new, y1_new)

    def soft_tanh_norm(self, x: np.ndarray) -> np.ndarray:
        """Normalisation soft-tanh"""
        k = self.roi_cfg.get("soft_tanh_k", 3.0)
        eps = 1e-6

        med = np.median(x)
        mad = np.median(np.abs(x - med)) + eps
        s = 1.4826 * mad
        y = np.tanh((x - med) / (k * s + eps))
        y01 = (y + 1.0) / 2.0
        return y01.astype(np.float32)

    # ========== PIPELINE PRINCIPALE ==========

    def process_one(self, patient_id: int, image_id: int, laterality: str, view: str,
                    dicom_path: str) -> Dict[str, Any]:
        """Traite une seule image - version SRP"""
        # 1) Chargement pour ROI via Loader
        img01, spacing = self.loader.load_dicom_for_roi(dicom_path)

        # 2) Segmentation du sein
        mask, _ = self.breast_mask(img01)

        # 3) Retrait pectoral + érosion
        mask = self.remove_pectoral_MLO(img01, mask, laterality, view)
        mask = self.erode_mask_mm(mask, spacing)

        # 4) Orientation standard
        imgO, maskO, flipped = self.orient_left(img01, mask)

        # 5) BBox + marges
        num, labels, stats, _ = cv.connectedComponentsWithStats(maskO.astype(np.uint8), 8)
        if num <= 1:
            # Fallback: image entière
            h, w = imgO.shape
            y0, x0, y1, x1 = 0, 0, h, w
        else:
            idx = int(np.argmax(stats[1:, cv.CC_STAT_AREA])) + 1
            x = stats[idx, cv.CC_STAT_LEFT]
            y = stats[idx, cv.CC_STAT_TOP]
            wbox = stats[idx, cv.CC_STAT_WIDTH]
            hbox = stats[idx, cv.CC_STAT_HEIGHT]
            base_bbox = (int(y), int(x), int(y + hbox), int(x + wbox))

            y0, x0, y1, x1 = self.bbox_with_margin_mm_aniso(
                base_bbox, spacing, imgO.shape[0], imgO.shape[1], view
            )

            # Fallback si bords touchés
            touches = int(y0 == 0) + int(y1 == imgO.shape[0]) + int(x1 == imgO.shape[1])
            if self.roi_cfg.get("enable_profile_fallback", True) and (
                    touches >= self.roi_cfg.get("touch_crit_thresh", 1)):
                y0, x0, y1, x1 = self.profile_crop_box(imgO, pad_frac=0.03)

            # Raffinement vertical
            y0, y1 = self.refine_vertical_bounds(maskO, y0, y1, spacing_mm=spacing)

        # 6) Crop modèle
        img_lin, _ = self.loader.load_dicom_linear(dicom_path)
        if flipped:
            img_lin = np.ascontiguousarray(np.fliplr(img_lin))

        raw_crop = img_lin[y0:y1, x0:x1].astype(np.float32)
        crop_model = self.soft_tanh_norm(raw_crop)

        # Redimensionnement optionnel
        target_hw = self.roi_cfg.get("target_hw")
        if target_hw is not None:
            H, W = target_hw
            crop_model = cv.resize(crop_model, (W, H), interpolation=cv.INTER_AREA)

        return {
            'patient_id': patient_id,
            'image_id': image_id,
            'laterality': laterality,
            'view': view,
            'bbox': [int(y0), int(x0), int(y1), int(x1)],
            'flipped': bool(flipped),
            'spacing': {'row': float(spacing[0]), 'col': float(spacing[1])},
            'crop_shape': crop_model.shape,
            'crop_model': crop_model,
            'raw_crop': raw_crop
        }

    def process_dataframe(self, df: pd.DataFrame, output_dir: Path, subset_n: int = None) -> List[Dict[str, Any]]:
        """Traite un dataframe complet d'images"""
        if subset_n is not None:
            df = df.sample(int(subset_n), random_state=self.roi_cfg.get("random_state", 42)).reset_index(drop=True)

        results = []
        skipped = failed = 0

        for row in tqdm(df.to_dict(orient="records")):
            try:
                pid = int(row["patient_id"])
                iid = int(row["image_id"])
                laterality = str(row["laterality"])
                view = str(row["view"])

                # Utilise DatasetManager pour le chemin
                dicom_path = self.dataset_manager.get_dicom_path(pid, iid)

                if not os.path.exists(dicom_path):
                    skipped += 1
                    continue

                result = self.process_one(pid, iid, laterality, view, dicom_path)
                results.append(result)

            except Exception as e:
                failed += 1
                print(f"ERROR @ {row.get('patient_id')} {row.get('image_id')} → {e}")

        print(f"Terminé : {len(results)} crops — SKIPPED={skipped}, FAILED={failed}")
        return results
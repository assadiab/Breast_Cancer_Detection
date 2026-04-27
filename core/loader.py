from core.configuration import Config
from typing import Optional
import pandas as pd
import os
import numpy as np
import pydicom
import shutil

class Loader:
    """
    Handles data loading operations using a Config instance.
    Supports:
    - Loading a single CSV
    - Loading DICOM images from file paths, including compressed formats
    """

    def __init__(self, config: Config):
        self.config = config
        self.df: Optional[pd.DataFrame] = None

    # ---------- CSV Loading ---------- #
    def load_df(self) -> None:
        """Loads the CSV file into memory."""
        csv_path = self.config.config['paths']['csv']
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        try:
            self.df = pd.read_csv(csv_path)
            print(f"Data loaded successfully - {len(self.df)} rows")
        except Exception as e:
            print(f"Error while loading dataframe: {e}")
            raise

    def get_df(self) -> pd.DataFrame:
        """Returns the loaded dataframe, loading it if necessary."""
        if self.df is None:
            self.load_df()
        return self.df

    # ---------- DICOM Loading ---------- #
    def load_dicom(self, dicom_path: str, verbose: bool = False, output_dtype=np.uint16) -> np.ndarray:
        """
        Load a single DICOM image.

        Args:
            dicom_path: Path to the DICOM file.
            verbose: If True, prints debug info.
            output_dtype: numpy dtype for returned array (uint16 for OpenCV, float32 for ML)

        Returns:
            Numpy array of image.
        """
        if not os.path.exists(dicom_path):
            raise FileNotFoundError(f"DICOM not found: {dicom_path}")

        # --- Read DICOM ---
        ds = pydicom.dcmread(dicom_path, force=True)

        # --- Handle compression ---
        ts = getattr(ds.file_meta, "TransferSyntaxUID", None)
        if ts and ts.is_compressed:
            try:
                ds.decompress()
                if verbose:
                    print(f"[pydicom] decompressed: {ts.name}")
            except Exception as e:
                # fallback: gdcmconv
                tmp_path = f"{dicom_path}.tmp.dcm"
                if shutil.which("gdcmconv") is None:
                    raise RuntimeError(f"Compressed DICOM {ts} cannot be handled (gdcmconv missing). Original error: {e}")
                os.system(f"gdcmconv -w {dicom_path} {tmp_path}")
                ds = pydicom.dcmread(tmp_path, force=True)
                os.remove(tmp_path)
                if verbose:
                    print(f"[gdcmconv] fallback decompression done: {ts.name}")

        # --- Extract pixel array ---
        img = ds.pixel_array.astype(np.float32)

        # Apply slope/intercept FIRST (DICOM standard: raw → modality LUT → VOI LUT) — fix L1
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        img = img * slope + intercept

        # Apply VOI LUT on calibrated values
        try:
            from pydicom.pixel_data_handlers.util import apply_voi_lut
            img = apply_voi_lut(img, ds).astype(np.float32)
        except Exception:
            pass

        # Handle MONOCHROME1
        if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
            img = img.max() - img

        # Convert to proper dtype for OpenCV / processing
        if np.issubdtype(output_dtype, np.integer):
            img = np.clip(img, 0, np.iinfo(output_dtype).max).astype(output_dtype)
        else:
            img = img.astype(output_dtype)

        return img

    def load_multiple_dicoms(self, dicom_paths: list[str], verbose: bool = False, output_dtype=np.uint16) -> list[np.ndarray]:
        """
        Load multiple DICOM files at once.
        Returns a list of numpy arrays.

        Args:
            dicom_paths: list of DICOM file paths
            verbose: print debug info
            output_dtype: output array type
        """
        images = []
        for path in dicom_paths:
            try:
                img = self.load_dicom(path, verbose=verbose, output_dtype=output_dtype)
                images.append(img)
            except Exception as e:
                print(f"Failed to load DICOM {path}: {e}")
        return images

    def load_dicom_full(self, dicom_path: str, verbose: bool = False) -> dict:
        """
        Charge un DICOM en une seule lecture et retourne img_raw, img01 et spacing.

        Remplace le pattern précédent qui faisait 2 lectures séparées
        (load_dicom_for_roi → load_dicom + _get_dicom_spacing) et
        (load_dicom_linear → load_dicom + _get_dicom_spacing). (fix L2+L3)

        Args:
            dicom_path: Chemin vers le fichier DICOM.
            verbose: Affiche les infos de décompression.

        Returns:
            dict avec:
                "img_raw"  — float32 calibré (après rescale + VOI LUT)
                "img01"    — float32 normalisé dans [0, 1] (robuste p0.5/p99.5)
                "spacing"  — tuple (row_mm, col_mm)
        """
        if not os.path.exists(dicom_path):
            raise FileNotFoundError(f"DICOM not found: {dicom_path}")

        # --- Lecture unique ---
        ds = pydicom.dcmread(dicom_path, force=True)

        # --- Décompression ---
        ts = getattr(ds.file_meta, "TransferSyntaxUID", None) if hasattr(ds, "file_meta") else None
        if ts and ts.is_compressed:
            try:
                ds.decompress()
                if verbose:
                    print(f"[pydicom] decompressed: {ts.name}")
            except Exception as e:
                tmp_path = f"{dicom_path}.tmp.dcm"
                if shutil.which("gdcmconv") is None:
                    raise RuntimeError(
                        f"Compressed DICOM {ts} cannot be handled (gdcmconv missing). Error: {e}"
                    )
                os.system(f"gdcmconv -w {dicom_path} {tmp_path}")
                ds = pydicom.dcmread(tmp_path, force=True)
                os.remove(tmp_path)

        # --- Pixel array ---
        img = ds.pixel_array.astype(np.float32)

        # Rescale slope/intercept AVANT VOI LUT (fix L1 dans ce chemin aussi)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        img = img * slope + intercept

        try:
            from pydicom.pixel_data_handlers.util import apply_voi_lut
            img = apply_voi_lut(img, ds).astype(np.float32)
        except Exception:
            pass

        if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
            img = img.max() - img

        # --- Spacing depuis le dataset déjà chargé (pas de 2ème lecture) ---
        spacing = self._extract_spacing_from_ds(ds)

        img_raw = img.astype(np.float32)
        img01 = self.robust_normalize01(img_raw, 0.5, 99.5)

        return {"img_raw": img_raw, "img01": img01, "spacing": spacing}

    def _extract_spacing_from_ds(self, ds) -> tuple:
        """
        Extrait le spacing depuis un dataset DICOM déjà chargé (pas de 2ème lecture). (fix L3)

        Args:
            ds: Dataset pydicom déjà lu.

        Returns:
            tuple (row_mm, col_mm)
        """
        spacing = getattr(ds, "PixelSpacing", [0.2, 0.2])
        return (float(spacing[0]), float(spacing[1]))

    def load_dicom_for_roi(self, dicom_path: str, verbose: bool = False) -> tuple[np.ndarray, tuple]:
        """Chargement DICOM pour ROI avec normalisation [0,1]. Délègue à load_dicom_full."""
        result = self.load_dicom_full(dicom_path, verbose=verbose)
        return result["img01"], result["spacing"]

    def load_dicom_linear(self, dicom_path: str, verbose: bool = False) -> tuple[np.ndarray, tuple]:
        """Chargement DICOM linéaire sans normalisation. Délègue à load_dicom_full."""
        result = self.load_dicom_full(dicom_path, verbose=verbose)
        return result["img_raw"], result["spacing"]

    def _get_dicom_spacing(self, dicom_path: str) -> tuple:
        """Extrait le spacing DICOM. Conservé pour compatibilité ascendante."""
        ds = pydicom.dcmread(dicom_path, stop_before_pixels=True)
        return self._extract_spacing_from_ds(ds)

    @staticmethod
    def robust_normalize01(arr, p_low=0.5, p_high=99.5):
        """Normalisation robuste vers [0,1]"""
        lo, hi = np.percentile(arr, (p_low, p_high))
        x = (arr - lo) / max(hi - lo, 1e-6)
        return np.clip(x, 0, 1).astype(np.float32)
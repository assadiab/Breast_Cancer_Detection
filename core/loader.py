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

        # Apply VOI LUT if available
        try:
            from pydicom.pixel_data_handlers.util import apply_voi_lut
            img = apply_voi_lut(img, ds).astype(np.float32)
        except Exception:
            pass

        # Apply slope/intercept
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        img = img * slope + intercept

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

    def load_dicom_for_roi(self, dicom_path: str, verbose: bool = False) -> tuple[np.ndarray, tuple]:
        """Chargement DICOM optimisé pour ROI avec normalisation [0,1]"""
        img = self.load_dicom(dicom_path, verbose=verbose, output_dtype=np.float32)
        img01 = self.robust_normalize01(img, 0.5, 99.5)
        spacing = self._get_dicom_spacing(dicom_path)
        return img01, spacing

    def load_dicom_linear(self, dicom_path: str, verbose: bool = False) -> tuple[np.ndarray, tuple]:
        """Chargement DICOM linéaire sans normalisation"""
        img = self.load_dicom(dicom_path, verbose=verbose, output_dtype=np.float32)
        spacing = self._get_dicom_spacing(dicom_path)
        return img, spacing

    def _get_dicom_spacing(self, dicom_path: str) -> tuple:
        """Extrait le spacing DICOM sans recharger l'image"""
        ds = pydicom.dcmread(dicom_path, stop_before_pixels=True)
        spacing = getattr(ds, "PixelSpacing", [0.2, 0.2])
        return (float(spacing[0]), float(spacing[1]))

    @staticmethod
    def robust_normalize01(arr, p_low=0.5, p_high=99.5):
        """Normalisation robuste vers [0,1]"""
        lo, hi = np.percentile(arr, (p_low, p_high))
        x = (arr - lo) / max(hi - lo, 1e-6)
        return np.clip(x, 0, 1).astype(np.float32)
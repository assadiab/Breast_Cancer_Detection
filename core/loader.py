from core.configuration import Config
from typing import Optional
import pandas as pd
import os
import numpy as np
import pydicom

class Loader:
    """
    Handles data loading operations using a Config instance.
    Supports:
    - Loading a single CSV
    - Loading DICOM images from file paths
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
    def load_dicom(self, dicom_path: str, verbose: bool = False) -> np.ndarray:
        if not os.path.exists(dicom_path):
            raise FileNotFoundError(f"DICOM not found: {dicom_path}")

        ds = pydicom.dcmread(dicom_path, force=True)

        # Handle compression
        ts = getattr(ds.file_meta, "TransferSyntaxUID", None)
        if ts and ts.is_compressed:
            try:
                ds.decompress()
                if verbose: print(f"[pydicom] decompressed: {ts.name}")
            except Exception:
                if shutil.which("gdcmconv") is None:
                    raise CompressedDICOMError(f"Compressed DICOM {ts} cannot be handled (gdcmconv missing)")
                tmp_path = f"{dicom_path}.tmp.dcm"
                os.system(f"gdcmconv -w {dicom_path} {tmp_path}")
                ds = pydicom.dcmread(tmp_path, force=True)
                os.remove(tmp_path)
                if verbose: print(f"[gdcmconv] fallback decompression done: {ts.name}")

        # Load pixel array
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

        return img

    def load_multiple_dicoms(self, dicom_paths: list[str]) -> list[np.ndarray]:
        """
        Load multiple DICOM files at once.
        Returns a list of normalized numpy arrays.
        """
        images = []
        for path in dicom_paths:
            try:
                img = self.load_dicom(path)
                images.append(img)
            except Exception as e:
                print(f"Failed to load DICOM {path}: {e}")
        return images
from pathlib import Path
from typing import Dict, Any
import pydicom
from core.configuration import Config
from core.loader import Loader
import os
import pandas as pd

class DatasetManager:
    """
    Minimal dataset manager: handles CSV info and DICOM paths.
    """

    def __init__(self, config: Config, loader: Loader):
        self.config = config
        self.loader = loader
        self.loaded_dicoms: dict[str, dict] = {}

    def get_dicom_path(self, patient_id: int, image_id: int) -> str:
        """Return the full path to the DICOM file."""
        return os.path.join(self.config.images_dir, str(patient_id), f"{image_id}.dcm")

    def get_dicom_info(self, patient_id: int, image_id: int) -> dict:
        """
        Return all CSV information for a given image as a dict.
        Missing values are replaced with None.
        """
        df = self.loader.get_df().copy()
        df['patient_id'] = df['patient_id'].astype(int)
        df['image_id'] = df['image_id'].astype(int)

        row = df.loc[(df['patient_id'] == patient_id) & (df['image_id'] == image_id)]

        info = {
            'dicom_path': self.get_dicom_path(patient_id, image_id),
            'patient_id': patient_id,
            'image_id': image_id
        }

        if not row.empty:
            for k, v in row.iloc[0].to_dict().items():
                info[k] = None if pd.isna(v) else v
        else:
            # Fill CSV columns with None without overwriting patient_id/image_id
            for col in df.columns:
                if col not in info:
                    info[col] = None

        return info

    def dicom_record(self, dicom_path: Path, verbose: bool = False) -> Dict[str, Any]:
        """
        Wrapper for Loader.load_dicom that also returns spacing.
        Only essential info (image + spacing) is returned.
        """
        # Load the image with slope/intercept and MONOCHROME1 applied
        img = self.loader.load_dicom(str(dicom_path), verbose=verbose)

        # Extract spacing without reloading the image
        ds = pydicom.dcmread(str(dicom_path), stop_before_pixels=True)
        spacing = getattr(ds, "PixelSpacing", [0.2, 0.2])
        spacing = (float(spacing[0]), float(spacing[1]))

        return {
            "image": img,
            "spacing": spacing,
            "path": str(dicom_path)
        }
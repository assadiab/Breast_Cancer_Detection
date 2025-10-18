import math
import numpy as np
import SimpleITK as sitk
import json
from pathlib import Path
import gc
from tqdm import tqdm
import pandas as pd

class IsotropicResampler:
    """
    Resample 2D medical images to isotropic spacing.
    - Works only on images already loaded as numpy arrays.
    - Chooses reasonable spacing based on target and up/downsample limits.
    - Saves .npz compressed images and JSON metadata.
    """

    def __init__(self, out_dir: Path,
                 target_nominal: float = 0.10,
                 max_pixels: int = 100_000_000,
                 upsample_max: float = 2.0,
                 downsample_max: float = 3.0,
                 chunk_size: int = 500):
        self.OUT_DIR = Path(out_dir)
        self.TARGET_NOMINAL = target_nominal
        self.MAX_PIXELS = max_pixels
        self.UPSAMPLE_MAX = upsample_max
        self.DOWNSAMPLE_MAX = downsample_max
        self.CHUNK_SIZE = chunk_size
        self.OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------
    def resample_isotropic(self, img_np: np.ndarray, spacing_mm: tuple[float, float], target_spacing_mm: float = None, interp: str = "linear"):
        """
        Resample a 2D numpy image to isotropic spacing using SimpleITK.

        Args:
            img_np (np.ndarray): 2D image (H, W)
            spacing_mm (tuple): Original pixel spacing (sy, sx)
            target_spacing_mm (float, optional): Override target spacing
            interp (str): 'linear' or 'nearest'

        Returns:
            np.ndarray: Resampled image
            tuple: New spacing (s_iso, s_iso)
        """
        sy, sx = spacing_mm
        s_iso = min(sy, sx) if target_spacing_mm is None else float(target_spacing_mm)

        # Early exit if already isotropic and close to target
        if abs(sy - sx) < 1e-6 and abs(sy - s_iso) < 1e-6:
            return img_np.astype(np.float32), (sy, sx)

        # Numpy → SimpleITK
        img = sitk.GetImageFromArray(img_np)
        img.SetSpacing((sx, sy))

        # Compute new output size
        size_in = np.array([img_np.shape[1], img_np.shape[0]], dtype=np.float64)
        spacing_in = np.array([sx, sy], dtype=np.float64)
        spacing_out = np.array([s_iso, s_iso], dtype=np.float64)
        size_out = np.floor(size_in * (spacing_in / spacing_out) + 0.5).astype(int).tolist()

        # Interpolation
        ip = sitk.sitkNearestNeighbor if interp == "nearest" else sitk.sitkLinear

        res = sitk.Resample(img, size_out, sitk.Transform(), ip,
                            img.GetOrigin(), tuple(spacing_out.tolist()),
                            img.GetDirection(), 0.0, img.GetPixelID())

        return sitk.GetArrayFromImage(res).astype(np.float32), (s_iso, s_iso)

    # ----------------------------------------------------------
    def pick_reasonable_target(self, h: int, w: int, sy: float, sx: float):
        """
        Automatically limit extreme up/downsampling.
        """
        f_h, f_w = sy / self.TARGET_NOMINAL, sx / self.TARGET_NOMINAL
        up_ok = (f_h <= self.UPSAMPLE_MAX) and (f_w <= self.UPSAMPLE_MAX)
        down_ok = (f_h >= 1 / self.DOWNSAMPLE_MAX) and (f_w >= 1 / self.DOWNSAMPLE_MAX)

        if up_ok and down_ok:
            t, reason = self.TARGET_NOMINAL, "target_ok"
        elif not up_ok:
            t, reason = max(sy / self.UPSAMPLE_MAX, sx / self.UPSAMPLE_MAX), "capped_upsample"
        elif not down_ok:
            t, reason = min(sy * self.DOWNSAMPLE_MAX, sx * self.DOWNSAMPLE_MAX), "capped_downsample"
        else:
            t, reason = max(sy, sx), "kept_original"

        # Limit total pixel count
        t_min = math.sqrt((h * w * sy * sx) / max(1, self.MAX_PIXELS))
        if t < t_min:
            t = t_min
            reason += "+capped_max_pixels"

        return float(t), reason

    # ----------------------------------------------------------
    def save_iso_artifacts(self, stem: str, img_iso: np.ndarray, meta: dict):
        """
        Save resampled image and JSON metadata.
        """
        npz_path = self.OUT_DIR / f"{stem}_iso.npz"
        np.savez_compressed(npz_path, img=img_iso.astype(np.float32))

        json_path = self.OUT_DIR / f"{stem}_iso.json"
        with open(json_path, "w") as f:
            json.dump(meta, f, indent=2)

        return str(npz_path), str(json_path)

    # ----------------------------------------------------------
    def process_one(self, stem: str, img_np: np.ndarray, spacing: tuple[float, float]):
        """
        Process a single image array: resample + save artifacts.
        """
        h, w = img_np.shape
        t_safe, reason = self.pick_reasonable_target(h, w, *spacing)
        img_iso, new_sp = self.resample_isotropic(img_np, spacing, target_spacing_mm=t_safe)

        meta = {
            "orig_spacing_mm": spacing,
            "new_spacing_mm": new_sp,
            "shape": list(img_iso.shape),
            "policy_reason": reason
        }

        npz_path, json_path = self.save_iso_artifacts(stem, img_iso, meta)
        return {"stem": stem, "npy": npz_path, "json": json_path, **meta}

    # ----------------------------------------------------------
    def process_all_chunks(self, df: pd.DataFrame, id_col: str = "image_id"):
        """
        Process multiple images in chunks.
        Assumes df has columns: id_col, 'patient_id', 'image' (numpy array), 'spacing'.
        """
        total = len(df)
        all_records = []

        for start in range(0, total, self.CHUNK_SIZE):
            chunk = df.iloc[start:start + self.CHUNK_SIZE]
            records = []

            for _, row in tqdm(chunk.iterrows(), total=len(chunk), desc=f"Chunk {start // self.CHUNK_SIZE + 1}"):
                try:
                    rec = self.process_one(stem=str(row[id_col]), img_np=row['image'], spacing=row['spacing'])
                    rec.update({"patient_id": row["patient_id"], id_col: row[id_col]})
                    records.append(rec)
                except Exception as e:
                    records.append({id_col: row[id_col], "error": str(e)})

            all_records.append(pd.DataFrame(records))
            gc.collect()

        return pd.concat(all_records, ignore_index=True)

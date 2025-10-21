# utils/radiomics_utils.py
"""
Utilities to extract radiomics features using PyRadiomics.
This is a placeholder: calling pyradiomics requires proper image/label arrays,
and configuration of feature classes. Use the following pattern in your preprocessing step.
"""
from typing import Tuple
import numpy as np
# import radiomics modules if available
try:
    from radiomics import featureextractor
    PYRADIOMICS_AVAILABLE = True
except Exception:
    PYRADIOMICS_AVAILABLE = False

def extract_radiomics(image_array: np.ndarray, mask_array: np.ndarray, params: dict = None) -> np.ndarray:
    """
    Extract a set of radiomics features for a single image and mask.
    Returns a fixed-size numpy vector (fill with zeros if pyradiomics not installed).
    """
    if not PYRADIOMICS_AVAILABLE:
        # return placeholder zeros (length 64)
        return np.zeros(64, dtype=np.float32)
    # else initialize extractor with reasonable settings
    extractor = featureextractor.RadiomicsFeatureExtractor() if params is None else featureextractor.RadiomicsFeatureExtractor(params)
    res = extractor.execute(image_array, mask_array)
    # convert selected keys to vector (choose and order consistently)
    keys = [k for k in res.keys() if k.startswith('original')]  # crude selection
    vec = np.array([float(res[k]) for k in sorted(keys)])  # might vary length
    # ensure fixed length by padding/truncation to 64
    if vec.size < 64:
        vec = np.pad(vec, (0, 64 - vec.size))
    else:
        vec = vec[:64]
    return vec.astype(np.float32)
def compute_radiomics_features(image_array: np.ndarray, mask_array: np.ndarray = None, params: dict = None) -> np.ndarray:
    """
    Compute radiomics features for a given image (and optional mask).
    If no mask is provided, assumes the whole image is foreground.
    Returns a fixed-length vector of size 64.
    """
    if mask_array is None:
        mask_array = np.ones_like(image_array, dtype=np.uint8)  # whole image as mask
    return extract_radiomics(image_array, mask_array, params)


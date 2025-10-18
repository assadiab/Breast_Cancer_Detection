import numpy as np
import cv2
from typing import Tuple, Optional


class Cropping:
    """
    Handles breast ROI extraction, orientation correction,
    pectoral muscle removal, and cropping operations.

    SRP: Only responsible for image processing operations.
    """

    def __init__(self, target_size: Tuple[int, int] = (512, 512), margin_mm: float = 5.0):
        """
        Initialize the cropping configuration.

        Args:
            target_size: Final image size after cropping (width, height).
            margin_mm: Margin (in mm) added around the bounding box (anisotropic).
        """
        self.target_size = target_size
        self.margin_mm = margin_mm

    # -----------------------------------------------------------
    # --- MASKING OPERATIONS ---
    # -----------------------------------------------------------

    def breast_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Compute a binary breast mask from the mammography image.

        Args:
            image: 2D grayscale image array.

        Returns:
            Binary mask isolating the breast region.
        """
        img = np.copy(image)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morphological cleaning
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        return (mask > 0).astype(np.uint8)

    def erode_mask_mm(self, mask: np.ndarray, iterations: int = 2) -> np.ndarray:
        """
        Erode a binary mask to remove small border artifacts.

        Args:
            mask: Binary mask.
            iterations: Number of erosion passes.

        Returns:
            Eroded mask.
        """
        kernel = np.ones((3, 3), np.uint8)
        return cv2.erode(mask, kernel, iterations=iterations)

    def remove_pectoral_MLO(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Remove pectoral region from MLO view images.

        Args:
            image: Original image (grayscale).
            mask: Breast mask.

        Returns:
            Mask without pectoral region.
        """
        # Create copy to avoid in-place modification
        mask = mask.copy()
        h, w = mask.shape
        triangle = np.array([[0, 0], [int(0.2 * w), 0], [0, int(0.3 * h)]], np.int32)
        cv2.fillConvexPoly(mask, triangle, 0)
        return mask

    # -----------------------------------------------------------
    # --- ORIENTATION ---
    # -----------------------------------------------------------

    def orient_left(self, image: np.ndarray) -> np.ndarray:
        """
        Ensure that the breast is oriented to the left side of the image.

        Args:
            image: Input mammogram.

        Returns:
            Oriented image.
        """
        left_mean = np.mean(image[:, :image.shape[1] // 2])
        right_mean = np.mean(image[:, image.shape[1] // 2:])
        if right_mean > left_mean:
            return np.fliplr(image)
        return image

    def orient_by_laterality(self, image: np.ndarray, laterality: str) -> np.ndarray:
        """
        Orient image based on known laterality.

        Args:
            image: Input mammogram.
            laterality: 'L' for left breast, 'R' for right breast.

        Returns:
            Oriented image (always with breast on left side).
        """
        if laterality.upper() == 'R':
            return np.fliplr(image)
        elif laterality.upper() == 'L':
            return image
        else:
            # Fallback to auto-detection for invalid laterality
            return self.orient_left(image)

    # -----------------------------------------------------------
    # --- BOUNDING BOX / CROPPING ---
    # -----------------------------------------------------------

    def bbox_with_margin_mm_aniso(self, mask: np.ndarray, margin_ratio: float = 0.05) -> Tuple[int, int, int, int]:
        """
        Compute a bounding box around the breast with anisotropic margin.

        Args:
            mask: Binary breast mask.
            margin_ratio: Relative margin applied to each side.

        Returns:
            Bounding box (x_min, y_min, x_max, y_max)
        """
        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return 0, 0, mask.shape[1], mask.shape[0]
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        margin_x = int((x_max - x_min) * margin_ratio)
        margin_y = int((y_max - y_min) * margin_ratio)

        x_min = max(0, x_min - margin_x)
        x_max = min(mask.shape[1], x_max + margin_x)
        y_min = max(0, y_min - margin_y)
        y_max = min(mask.shape[0], y_max + margin_y)
        return x_min, y_min, x_max, y_max

    def crop_to_roi(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Crop image to ROI defined by the breast mask.

        Args:
            image: Original image.
            mask: Binary mask of breast region.

        Returns:
            Cropped image (resized to target size).
        """
        x_min, y_min, x_max, y_max = self.bbox_with_margin_mm_aniso(mask)
        cropped = image[y_min:y_max, x_min:x_max]
        resized = cv2.resize(cropped, self.target_size, interpolation=cv2.INTER_AREA)
        return resized

    # -----------------------------------------------------------
    # --- MAIN PIPELINE ---
    # -----------------------------------------------------------

    def process_with_metadata(self, image: np.ndarray, view: str = None, laterality: str = None) -> np.ndarray:
        """
        Enhanced pipeline using view and laterality metadata.
        SRP: Only processes images, doesn't handle data extraction.

        Args:
            image: Grayscale image as NumPy array.
            view: 'MLO' or 'CC'
            laterality: 'L' or 'R'

        Returns:
            Cropped image ready for model input.
        """
        if image.ndim != 2:
            raise ValueError("Input image must be 2D grayscale.")

        # Step 1: Orientation
        if laterality and laterality.upper() in ['L', 'R']:
            image = self.orient_by_laterality(image, laterality)
        else:
            image = self.orient_left(image)

        # Step 2: Breast masking
        mask = self.breast_mask(image)

        # Step 3: Pectoral removal ONLY for MLO views
        if view and view.upper() == 'MLO':
            mask = self.remove_pectoral_MLO(image, mask)

        # Step 4: Post-processing
        mask = self.erode_mask_mm(mask)
        cropped = self.crop_to_roi(image, mask)

        return cropped
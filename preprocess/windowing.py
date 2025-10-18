import torch
import torch.nn as nn
import cv2
import numpy as np
from typing import Optional, Dict, Union

class WindowingNet(nn.Module):
    """
    Class dedicated to windowing mammography images.
    SRP-compliant: only responsible for windowing logic.
    Supports density-guided windowing, AI prediction, and fallback methods.
    """

    def __init__(self, input_size: tuple = (512, 512)):
        super().__init__()
        self.input_size = input_size

        # Lightweight encoder for feature extraction
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8))
        )

        # Head for predicting windowing parameters
        self.param_predictor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 4)  # center_offset, width_offset, gamma, density_logit
        )

        # Baseline lookup table for densities
        self.density_baselines = {
            'A': {'center': 0.35, 'width': 0.85, 'gamma': 1.0},
            'B': {'center': 0.45, 'width': 0.70, 'gamma': 1.1},
            'C': {'center': 0.55, 'width': 0.55, 'gamma': 1.3},
            'D': {'center': 0.65, 'width': 0.40, 'gamma': 1.5}
        }

        # Fallback methods (purely on image array)
        self.fallback_methods = {
            'A': lambda x: self._clahe(x, clip_limit=1.5),
            'B': lambda x: self._percentile(x, low=10, high=90),
            'C': lambda x: self._percentile(x, low=20, high=85),
            'D': lambda x: self._percentile(x, low=25, high=80)
        }

    def forward(self, x: torch.Tensor, density: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for windowing.
        If density is provided, applies baseline + offset; else predicts full parameters.
        """
        features = self.encoder(x)
        params = self.param_predictor(features)

        center_offset = torch.sigmoid(params[:, 0:1]) * 0.3 - 0.15
        width_offset = torch.sigmoid(params[:, 1:2]) * 0.4 - 0.2
        gamma = torch.sigmoid(params[:, 2:3]) * 1.5 + 0.5
        density_logits = params[:, 3:4]

        if density is not None:
            baseline = self._get_density_baseline(density)
            center_norm = baseline['center'] + center_offset.squeeze()
            width_norm = baseline['width'] + width_offset.squeeze()
            gamma = gamma.squeeze()
        else:
            center_norm = torch.sigmoid(params[:, 0])
            width_norm = torch.sigmoid(params[:, 1]) * 0.8 + 0.2
            gamma = torch.sigmoid(params[:, 2]) * 1.5 + 0.5

        windowed = self._apply_windowing(x, center_norm, width_norm, gamma)

        return {
            'windowed_image': windowed,
            'params': {
                'center': center_norm,
                'width': width_norm,
                'gamma': gamma,
                'density_logit': density_logits
            }
        }

    def apply_fallback(self, image: Union[np.ndarray, torch.Tensor], density: str) -> np.ndarray:
        """
        Apply fallback method (CLAHE or percentile) based on density.
        """
        if isinstance(image, torch.Tensor):
            image = image.squeeze().cpu().numpy()
        fallback_fn = self.fallback_methods.get(density, self.fallback_methods['B'])
        return fallback_fn(image)

    def _get_density_baseline(self, density: str) -> Dict[str, float]:
        return self.density_baselines.get(density, self.density_baselines['B'])

    @staticmethod
    def _apply_windowing(x: torch.Tensor, center_norm: torch.Tensor, width_norm: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        x_min, x_max = x.min(), x.max()
        x_range = x_max - x_min
        center = x_min + center_norm * x_range
        width = width_norm * x_range

        window_min = center - width / 2
        window_max = center + width / 2

        windowed = torch.clamp(x, window_min, window_max)
        windowed = (windowed - window_min) / (width + 1e-8)
        windowed = torch.pow(windowed + 1e-8, gamma)

        return windowed

    @staticmethod
    def _clahe(image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
        img_uint8 = (image * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        enhanced = clahe.apply(img_uint8)
        return enhanced.astype(np.float32) / 255.0

    @staticmethod
    def _percentile(image: np.ndarray, low: float = 5, high: float = 95) -> np.ndarray:
        p_low, p_high = np.percentile(image, [low, high])
        windowed = np.clip(image, p_low, p_high)
        return (windowed - p_low) / (p_high - p_low + 1e-8)

import os
from typing import Dict, Any, Optional


class Config:
    """
    Handles dataset configuration, including file paths, preprocessing, and model parameters.
    Manages and exposes dataset configuration values.
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        """
        Initialize the dataset configuration.

        Args:
            config_path: Default RSNA settings are used.
        """
        self.config: Dict[str, Any] = {}
        self._initialize_default_config()

    # ==========================================================
    # ================= DEFAULT CONFIGURATION ==================
    # ==========================================================

    def _initialize_default_config(self) -> None:
        """Initialize the default configuration for the RSNA Breast Cancer Detection dataset."""
        root_dir = '../data'

        self.config = {
            'paths': {
                'root_dir': root_dir,
                'train_images': os.path.join(root_dir, 'train_images'),
                'test_images': os.path.join(root_dir, 'test_images'),
                'train_csv': os.path.join(root_dir, 'train.csv'),
                'test_csv': os.path.join(root_dir, 'test.csv'),
            },
            'preprocessing': {
                'target_size': [512, 512],
                'default_density': 'B',  # Fallback if density value is missing
                'density_categories': {
                    'A': 'Almost entirely fatty',
                    'B': 'Scattered fibroglandular densities',
                    'C': 'Heterogeneously dense',
                    'D': 'Extremely dense'
                }
            },
            'model': {
                'input_shape': [512, 512, 1],
                'batch_size': 32
            }
        }

    # ==========================================================
    # ================= DENSITY DESCRIPTIONS ===================
    # ==========================================================

    @property
    def density_categories(self) -> Dict[str, str]:
        """
        Return the dictionary mapping breast density categories to their descriptions.

        Returns:
            dict: Mapping of density category codes ('A', 'B', 'C', 'D') to textual descriptions.
        """
        return self.config['preprocessing']['density_categories']

    def get_density_description(self, density: str) -> str:
        """
        Return the human-readable description for a given breast density category.

        Args:
            density: The density category code (e.g., 'A', 'B', 'C', 'D').

        Returns:
            str: The corresponding density description or 'Unknown density' if not found.
        """
        return self.density_categories.get(density, 'Unknown density')

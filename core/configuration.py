import os
from typing import Dict, Any

class Config:
    """
    Handles dataset configuration, including file paths, preprocessing, output directory,
    and model parameters.
    Works without differentiating between train and test datasets.
    """

    def __init__(self, csv_path: str, images_dir: str, out_dir: str = None) -> None:
        """
        Initialize the dataset configuration.

        Args:
            csv_path: Path to the dataset CSV file.
            images_dir: Path to the folder containing all images.
            out_dir: Path to the folder where preprocessed outputs will be saved.
                     If None, outputs are not saved by default.
        """
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        if not os.path.isdir(images_dir):
            raise FileNotFoundError(f"Images directory not found: {images_dir}")

        self.csv_path = csv_path
        self.images_dir = images_dir

        # Handle output directory
        if out_dir is not None:
            os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir

        self.config: Dict[str, Any] = {}
        self._initialize_default_config()

    # ==========================================================
    # ================= DEFAULT CONFIGURATION ==================
    # ==========================================================
    def _initialize_default_config(self) -> None:
        """Initialize the default configuration for the dataset."""
        self.config = {
            'paths': {
                'csv': self.csv_path,          # user-provided CSV path
                'images': self.images_dir,     # user-provided images directory
                'out': self.out_dir            # output directory (can be None)
            },
            'preprocessing': {
                'target_size': [512, 512],
                'default_density': 'B',        # fallback if density missing
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
        """Return the dictionary mapping breast density categories to their descriptions."""
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

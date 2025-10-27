import cv2
import numpy as np
from typing import Optional


class Windowing:
    """
    Windowing optimisé pour mammographies basé sur percentile adaptatif.
    Méthode déterministe, rapide et efficace.

    Pipeline: Percentile clipping → Gamma correction → CLAHE léger → Fusion

    Usage:
        windowing = Windowing()
        img_windowed = windowing.process_one(img_cropped, density="B")
    """

    def __init__(self, preserve_range: tuple[float, float] = (0.0, 1.0)):
        """
        Args:
            preserve_range: Plage de sortie [min, max], par défaut [0, 1]
        """
        self.preserve_range = preserve_range

        # Paramètres optimaux par densité mammaire
        # Basés sur les caractéristiques physiques du tissu mammaire
        self.density_params = {
            'A': {  # Almost entirely fatty (tissu principalement graisseux)
                'percentiles': (2.0, 98.0),  # Outliers modérés
                'gamma': 0.9,  # Légère augmentation luminosité
                'clahe_clip': 1.5,  # CLAHE doux
                'clahe_weight': 0.25  # Faible poids CLAHE
            },
            'B': {  # Scattered fibroglandular (densité dispersée)
                'percentiles': (1.0, 99.0),  # Outliers faibles
                'gamma': 1.0,  # Pas de correction gamma
                'clahe_clip': 2.0,  # CLAHE modéré
                'clahe_weight': 0.30  # Poids modéré CLAHE
            },
            'C': {  # Heterogeneously dense (densité hétérogène)
                'percentiles': (1.0, 99.0),  # Conservation détails
                'gamma': 1.1,  # Légère compression dynamique
                'clahe_clip': 2.5,  # CLAHE fort
                'clahe_weight': 0.35  # Poids élevé CLAHE
            },
            'D': {  # Extremely dense (tissu extrêmement dense)
                'percentiles': (0.5, 99.5),  # Conservation maximale
                'gamma': 1.2,  # Compression dynamique
                'clahe_clip': 3.0,  # CLAHE très fort
                'clahe_weight': 0.40  # Poids maximal CLAHE
            }
        }

        print(f"[Windowing] Initialized - Adaptive percentile method")
        print(f"[Windowing] Output range: {preserve_range}")

    def process_one(
            self,
            image: np.ndarray,
            density: Optional[str] = None
    ) -> np.ndarray:
        """
        Applique le windowing adaptatif sur une image.

        Args:
            image: Image 2D numpy array (H, W), n'importe quel dtype
            density: Densité mammaire ('A', 'B', 'C', 'D')
                     Si None, utilise 'B' (densité moyenne)

        Returns:
            Image windowée normalisée dans preserve_range

        Raises:
            ValueError: Si l'image n'est pas 2D ou contient des valeurs invalides
        """
        # Validation de l'entrée
        if image.ndim != 2:
            raise ValueError(f"Expected 2D image, got shape {image.shape}")

        if not np.isfinite(image).all():
            raise ValueError("Image contains non-finite values (NaN or Inf)")

        img_min, img_max = image.min(), image.max()
        if img_max - img_min < 1e-8:
            print("[Windowing] WARNING: Image has no contrast")
            return np.zeros_like(image, dtype=np.float32)

        # Convertir en float32 si nécessaire
        if image.dtype != np.float32:
            image = image.astype(np.float32)

        # Sélectionner les paramètres selon la densité
        if density is None or density.upper() not in self.density_params:
            density = 'B'  # Densité par défaut (la plus courante)
        else:
            density = density.upper()

        params = self.density_params[density]

        # Appliquer le windowing adaptatif
        windowed = self._adaptive_percentile_windowing(image, params)

        # Normalisation finale dans preserve_range
        return self._normalize_to_range(windowed)

    def _adaptive_percentile_windowing(
            self,
            image: np.ndarray,
            params: dict
    ) -> np.ndarray:
        """
        Pipeline de windowing adaptatif complet.

        Étapes:
            1. Clipping par percentiles (élimine outliers)
            2. Normalisation [0, 1]
            3. Gamma correction (ajuste contraste global)
            4. CLAHE (améliore contraste local)
            5. Fusion pondérée

        Args:
            image: Image en float32
            params: Paramètres de densité

        Returns:
            Image windowée [0, 1]
        """
        # ============================================================
        # ÉTAPE 1: Windowing par percentiles
        # ============================================================
        p_low, p_high = params['percentiles']
        low_val, high_val = np.percentile(image, [p_low, p_high])

        # Clipping vectorisé ultra-rapide
        img_clipped = np.clip(image, low_val, high_val)

        # Normalisation dans [0, 1]
        img_norm = (img_clipped - low_val) / (high_val - low_val + 1e-8)

        # ============================================================
        # ÉTAPE 2: Gamma correction
        # ============================================================
        # Ajuste la distribution des intensités
        # gamma < 1 : éclaircit les zones sombres
        # gamma > 1 : assombrit les zones claires
        gamma = params['gamma']
        img_gamma = np.power(img_norm, gamma)

        # ============================================================
        # ÉTAPE 3: CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # ============================================================
        # Améliore le contraste local sans amplifier le bruit

        # Convertir en uint8 pour OpenCV CLAHE
        img_uint8 = (img_gamma * 255).astype(np.uint8)

        # Créer et appliquer CLAHE
        clahe = cv2.createCLAHE(
            clipLimit=params['clahe_clip'],  # Limite d'amplification du contraste
            tileGridSize=(16, 16)  # Taille des tuiles (16x16 = bon compromis)
        )
        img_clahe = clahe.apply(img_uint8).astype(np.float32) / 255.0

        # ============================================================
        # ÉTAPE 4: Fusion pondérée
        # ============================================================
        # Combine gamma (contraste global) et CLAHE (contraste local)
        clahe_weight = params['clahe_weight']
        img_final = (1 - clahe_weight) * img_gamma + clahe_weight * img_clahe

        return img_final

    def _normalize_to_range(self, image: np.ndarray) -> np.ndarray:
        """
        Normalise l'image dans preserve_range.

        Args:
            image: Image en float32, supposée dans [0, 1]

        Returns:
            Image normalisée dans preserve_range
        """
        min_out, max_out = self.preserve_range

        # Sécurité: re-normaliser si hors [0, 1]
        img_min, img_max = image.min(), image.max()
        if img_max - img_min > 1e-8:
            image = (image - img_min) / (img_max - img_min)

        # Mapper vers preserve_range
        return image * (max_out - min_out) + min_out

    def process_batch(
            self,
            images: list[np.ndarray],
            densities: Optional[list[str]] = None
    ) -> list[np.ndarray]:
        """
        Traite un batch d'images de manière séquentielle.

        Pour du traitement parallèle, utilisez multiprocessing externalement.

        Args:
            images: Liste d'images 2D numpy
            densities: Liste de densités correspondantes (optionnel)

        Returns:
            Liste d'images windowées
        """
        if densities is None:
            densities = [None] * len(images)

        if len(images) != len(densities):
            raise ValueError(f"Mismatch: {len(images)} images but {len(densities)} densities")

        return [
            self.process_one(img, density)
            for img, density in zip(images, densities)
        ]


# data_augmentation_mammo.py
"""
Data augmentation pour mammographies DICOM 1 canal
"""
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Optional, List
import random
from scipy import ndimage
from scipy.ndimage import gaussian_filter, map_coordinates, zoom, rotate, shift


class MammoAugmentation:
    """Augmentations spécialisées MAMMOGRAPHIE DICOM - SANS OpenCV"""

    def __init__(self, mode='train', intensity=0.7):
        self.mode = mode
        self.intensity = intensity

        # Probabilités optimisées mammo
        self.probs = {
            'flip': 0.5,
            'rotation': 0.4 * intensity,
            'shift_scale': 0.3 * intensity,
            'brightness_contrast': 0.6 * intensity,
            'gamma': 0.4 * intensity,
            'noise': 0.2 * intensity,
            'elastic': 0.2 * intensity,
            'cutout': 0.2 * intensity,
            'histogram': 0.3 * intensity,
            'sharpening': 0.3
        }

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Apply augmentations optimisées mammo"""
        if self.mode != 'train':
            return image

        image = self.apply_mammo_augmentations(image)
        return image

    def apply_mammo_augmentations(self, image: np.ndarray) -> np.ndarray:
        """Chaîne d'augmentations SPÉCIALISÉES mammographie"""

        # 1. FLIPS - Horizontal seulement (symétrie sein G/D)
        if random.random() < self.probs['flip']:
            image = np.flip(image, axis=2).copy()

        # 2. ROTATION LÉGÈRE
        if random.random() < self.probs['rotation']:
            angle = random.uniform(-8, 8)
            image = self.rotate_preserve_bounds(image, angle)

        # 3. SHIFT/SCALE MINIME
        if random.random() < self.probs['shift_scale']:
            image = self.minimal_shift_scale(image)

        # 4. AUGMENTATIONS INTENSITÉ
        if random.random() < self.probs['brightness_contrast']:
            image = self.medical_brightness_contrast(image)

        if random.random() < self.probs['gamma']:
            gamma = random.uniform(0.85, 1.15)
            image = self.gamma_correction(image, gamma)  # ✅ CORRIGÉ

        if random.random() < self.probs['histogram']:
            image = self.adaptive_histogram_mammo(image)

        # 5. SHARPENING - SANS OpenCV
        if random.random() < self.probs['sharpening']:
            image = self.smart_sharpening_nocv2(image)

        # 6. BRUIT et DÉFORMATIONS
        if random.random() < self.probs['noise']:
            image = self.medical_gaussian_noise(image)

        if random.random() < self.probs['elastic']:
            image = self.mild_elastic_transform(image)

        if random.random() < self.probs['cutout']:
            image = self.focal_erasing(image)

        return image

    def gamma_correction(self, image: np.ndarray, gamma: float) -> np.ndarray:
        """Correction gamma pour mammographie - CORRIGÉ"""
        # Éviter les valeurs nulles
        epsilon = 1e-8
        image_corrected = np.power(np.maximum(image, epsilon), gamma)
        return np.clip(image_corrected, 0, 1)

    def rotate_preserve_bounds(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotation qui préserve les bords"""
        rotated = rotate(image, angle, axes=(1, 2), reshape=False,
                         order=1, mode='reflect')
        return rotated

    def minimal_shift_scale(self, image: np.ndarray) -> np.ndarray:
        """Shift et scale très légers"""
        C, H, W = image.shape

        scale = random.uniform(0.95, 1.05)
        shift_h = random.randint(-int(H * 0.02), int(H * 0.02))
        shift_w = random.randint(-int(W * 0.02), int(W * 0.02))

        # Scale d'abord
        image_scaled = zoom(image, (1, scale, scale), order=1)

        # Puis shift
        if shift_h != 0 or shift_w != 0:
            image_scaled = shift(image_scaled, (0, shift_h, shift_w), order=1)

        return self.smart_crop_pad(image_scaled, (H, W))

    def smart_crop_pad(self, image: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """Recadrage/padding intelligent"""
        C, H, W = image.shape
        target_H, target_W = target_shape

        if H > target_H:
            start_h = (H - target_H) // 2
            image = image[:, start_h:start_h + target_H, :]
        elif H < target_H:
            pad_h_before = (target_H - H) // 2
            pad_h_after = target_H - H - pad_h_before
            image = np.pad(image, ((0, 0), (pad_h_before, pad_h_after), (0, 0)),
                           mode='reflect')

        if W > target_W:
            start_w = (W - target_W) // 2
            image = image[:, :, start_w:start_w + target_W]
        elif W < target_W:
            pad_w_before = (target_W - W) // 2
            pad_w_after = target_W - W - pad_w_before
            image = np.pad(image, ((0, 0), (0, 0), (pad_w_before, pad_w_after)),
                           mode='reflect')

        return image

    def medical_brightness_contrast(self, image: np.ndarray) -> np.ndarray:
        """Brightness/contrast adapté DICOM"""
        brightness_factor = random.uniform(0.9, 1.1)
        contrast_factor = random.uniform(0.9, 1.1)

        mean = image.mean()
        image = (image - mean) * contrast_factor + mean
        image = image * brightness_factor

        return np.clip(image, 0, 1)

    def adaptive_histogram_mammo(self, image: np.ndarray) -> np.ndarray:
        """Histogram adaptation sans skimage"""
        try:
            # Implémentation manuelle de l'égalisation d'histogramme adaptative
            result = np.zeros_like(image)
            for c in range(image.shape[0]):
                # Égalisation d'histogramme simple
                img_flat = image[c].flatten()
                hist, bins = np.histogram(img_flat, bins=256, range=(0, 1))
                cdf = hist.cumsum()
                cdf = cdf / cdf[-1]  # Normaliser

                # Interpolation
                result[c] = np.interp(image[c], bins[:-1], cdf)

            return np.clip(result, 0, 1)
        except:
            return image

    def smart_sharpening_nocv2(self, image: np.ndarray) -> np.ndarray:
        """Sharpening SANS OpenCV - utilisant scipy"""
        # Créer un noyau de sharpening
        kernel = np.array([[-0.1, -0.1, -0.1],
                           [-0.1, 1.8, -0.1],
                           [-0.1, -0.1, -0.1]])

        result = np.zeros_like(image)
        for c in range(image.shape[0]):
            # Convolution avec scipy
            result[c] = ndimage.convolve(image[c], kernel, mode='reflect')

        return np.clip(result, 0, 1)

    def medical_gaussian_noise(self, image: np.ndarray) -> np.ndarray:
        """Bruit gaussien médical"""
        std = random.uniform(0.005, 0.015)
        noise = np.random.normal(0, std, image.shape)
        return np.clip(image + noise, 0, 1)

    def mild_elastic_transform(self, image: np.ndarray) -> np.ndarray:
        """Transformée élastique légère"""
        C, H, W = image.shape

        alpha = random.uniform(10, 30)
        sigma = random.uniform(4, 6)

        dx = gaussian_filter((np.random.rand(H, W) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((np.random.rand(H, W) * 2 - 1), sigma) * alpha

        x, y = np.meshgrid(np.arange(W), np.arange(H))
        indices = (np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)))

        result = np.zeros_like(image)
        for c in range(C):
            result[c] = map_coordinates(
                image[c],
                indices,
                order=1,
                mode='reflect'
            ).reshape(H, W)

        return result

    def focal_erasing(self, image: np.ndarray) -> np.ndarray:
        """Random erasing focalisé"""
        C, H, W = image.shape

        center_y, center_x = H // 2, W // 2
        max_offset = min(H, W) // 3

        top = random.randint(0, H - 20)
        left = random.randint(0, W - 20)

        if abs(top - center_y) < max_offset and abs(left - center_x) < max_offset:
            return image

        h = random.randint(5, 15)
        w = random.randint(5, 15)

        if top + h < H and left + w < W:
            local_mean = image[:, top:top + h, left:left + w].mean()
            image[:, top:top + h, left:left + w] = local_mean

        return image


class AdvancedMammoAugmentation:
    """Techniques d'augmentation avancées pour mammo - SANS OpenCV"""

    def __init__(self, alpha: float = 0.4):
        self.alpha = alpha

    def mammo_mixup(self, img1: torch.Tensor, img2: torch.Tensor,
                    label1: float, label2: float) -> Tuple[torch.Tensor, float]:
        """Mixup adapté mammographie"""
        lam = np.random.beta(self.alpha, self.alpha)
        mixed_img = lam * img1 + (1 - lam) * img2
        mixed_label = lam * label1 + (1 - lam) * label2
        return mixed_img, mixed_label

    def mammo_cutmix(self, img1: torch.Tensor, img2: torch.Tensor,
                     label1: float, label2: float) -> Tuple[torch.Tensor, float]:
        """CutMix adapté mammo"""
        lam = np.random.beta(self.alpha, self.alpha)

        _, H, W = img1.shape
        cut_rat = np.sqrt(1.0 - lam)
        cut_h = int(H * cut_rat)
        cut_w = int(W * cut_rat)

        center_y, center_x = H // 2, W // 2
        max_offset = min(H, W) // 4

        if random.random() < 0.7:
            cx = random.choice([random.randint(0, center_x - max_offset),
                                random.randint(center_x + max_offset, W)])
            cy = random.choice([random.randint(0, center_y - max_offset),
                                random.randint(center_y + max_offset, H)])
        else:
            cx = random.randint(0, W)
            cy = random.randint(0, H)

        x1 = np.clip(cx - cut_w // 2, 0, W)
        y1 = np.clip(cy - cut_h // 2, 0, H)
        x2 = np.clip(cx + cut_w // 2, 0, W)
        y2 = np.clip(cy + cut_h // 2, 0, H)

        mixed_img = img1.clone()
        mixed_img[:, y1:y2, x1:x2] = img2[:, y1:y2, x1:x2]

        lam = 1 - ((x2 - x1) * (y2 - y1) / (H * W))
        mixed_label = lam * label1 + (1 - lam) * label2

        return mixed_img, mixed_label


class TorchMammoAugmentation:
    """Version PyTorch pour intégration directe dans DataLoader"""

    def __init__(self, mode='train'):
        self.mode = mode
        self.augmentation = MammoAugmentation(mode=mode) if mode == 'train' else None

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if self.mode != 'train' or self.augmentation is None:
            return image

        # Convertir en numpy, appliquer, reconvertir
        if image.dim() == 3:
            image_np = image.numpy()
            augmented_np = self.augmentation(image_np)
            return torch.from_numpy(augmented_np)
        return image


# Factory functions
def get_mammo_augmentation(mode='train', intensity=0.7):
    """Retourne augmentation spécialisée mammo"""
    return MammoAugmentation(mode=mode, intensity=intensity)


def get_torch_mammo_augmentation(mode='train'):
    """Retourne version PyTorch"""
    return TorchMammoAugmentation(mode=mode)


def get_advanced_mammo_augmentation(alpha=0.4):
    """Retourne techniques avancées"""
    return AdvancedMammoAugmentation(alpha=alpha)


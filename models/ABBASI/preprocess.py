"""
Preprocessing module for hybrid ResNet-50 + metadata mammography classification.

This module provides specialized data preparation pipelines adapted for our hybrid model architecture.
While initial DICOM preprocessing was performed, this module handles model-specific adaptations:

- Image transformations optimized for ResNet-50 (224x224, ImageNet normalization)
- Metadata encoding tailored for clinical features (density, age, site, machine)
- Dataset formatting for combined image + metadata input
- Data augmentation strategies for medical imaging
- GPU acceleration setup for distributed computation

The preprocessing ensures compatibility between preprocessed DICOM files and
the specific requirements of our hybrid ResNet-50 + metadata model architecture.
GPU utilization is configured to accelerate data processing and model training.
"""
import os
import random
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import pydicom
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split


class DeviceManager:
    @staticmethod
    def setup_device(seed=42):
        """
        Configure and return the available computing device.

        Sets random seeds for reproducibility and detects the best available
        device in order: CUDA (GPU) > MPS (Apple Silicon) > CPU.

        Args:
            seed (int): Random seed for reproducibility (default: 42)

        Returns:
            str: Device name ('cuda', 'mps', or 'cpu')
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            device = "cuda"
            torch.cuda.manual_seed_all(seed)
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        print(f"Using device: {device}")
        return device


class DataPreprocessor:
    """
    Handles loading, merging, and preprocessing of mammography data.

    Manages DICOM images and CSV metadata, providing methods for data validation,
    filtering, splitting, and image preprocessing for ResNet compatibility.

    Args:
        csv_x_path (str): Path to features CSV file
        csv_y_path (str): Path to labels CSV file
        img_dir (str): Directory containing DICOM images
        seed (int): Random seed for reproducibility
    """
    def __init__(self, csv_x_path, csv_y_path, img_dir, seed=42):
        self.csv_x_path = Path(csv_x_path)
        self.csv_y_path = Path(csv_y_path)
        self.img_dir = Path(img_dir)
        self.seed = seed
        self.df = None

    def load_and_merge_data(self):
        X_data = pd.read_csv(self.csv_x_path)
        Y_data = pd.read_csv(self.csv_y_path)

        assert len(X_data) == len(Y_data), "X and Y data have different lengths"

        self.df = pd.concat([X_data, Y_data], axis=1)
        print(f"Merged dataset: {len(self.df)} rows, {len(self.df.columns)} columns")

        return self.df

    def filter_available_images(self):
        available_images = [img_path.stem for img_path in self.img_dir.glob("*.dcm")]
        self.df['filename'] = self.df['patient_id'].astype(str) + '_' + self.df['image_id'].astype(str)
        self.df = self.df[self.df['filename'].isin(available_images)].copy()
        print(f"Filtered dataset: {len(self.df)} rows (existing images)")
        return self.df

    def get_class_balance(self):
        if 'cancer' in self.df.columns:
            cancer_counts = self.df['cancer'].value_counts()
            return cancer_counts
        return None

    def get_pos_weight(self):
        if 'cancer' in self.df.columns:
            num_negative = (self.df['cancer'] == 0).sum()
            num_positive = (self.df['cancer'] == 1).sum()
            return num_negative / num_positive if num_positive > 0 else 1.0
        return 1.0

    def split_data(self, test_size=0.2):
        df_train, df_val = train_test_split(
            self.df,
            test_size=test_size,
            random_state=self.seed,
            stratify=self.df['cancer'] if 'cancer' in self.df.columns else None
        )
        return df_train, df_val

    def verify_images(self, num_samples=10):
        img_files = list(self.img_dir.glob("*.dcm"))[:num_samples]
        stats = {'shapes': [], 'dtypes': [], 'value_ranges': [], 'channels': []}

        for img_path in img_files:
            ds = pydicom.dcmread(str(img_path))
            arr = ds.pixel_array

            stats['shapes'].append(arr.shape)
            stats['dtypes'].append(str(arr.dtype))
            stats['value_ranges'].append((arr.min(), arr.max()))
            stats['channels'].append(len(arr.shape))

        needs_resize = len(set(stats['shapes'])) > 1
        needs_normalize = any(r[1] > 1.0 for r in stats['value_ranges'])
        needs_rgb_conversion = all(c == 2 for c in stats['channels'])

        return needs_resize, needs_normalize, needs_rgb_conversion

    def preprocess_dicom_for_resnet(self, dicom_path):
        ds = pydicom.dcmread(str(dicom_path))
        img_array = ds.pixel_array.astype(np.float32)

        img_min, img_max = img_array.min(), img_array.max()
        img_array = (img_array - img_min) / (img_max - img_min + 1e-8)
        img_array = (img_array * 255).astype(np.uint8)

        img_rgb = np.stack([img_array, img_array, img_array], axis=-1)
        img_pil = Image.fromarray(img_rgb, mode='RGB')

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        img_tensor = transform(img_pil)
        return img_tensor, img_pil


class MammographyDataset(Dataset):
    """
    PyTorch Dataset for mammography images and metadata.

    Loads DICOM images, applies preprocessing transforms, and encodes categorical
    features (density, site, machine) for hybrid ResNet + metadata model training.

    Args:
        df (DataFrame): DataFrame containing image metadata and labels
        img_dir (str): Directory containing DICOM image files
        transform: Torchvision transforms to apply to images
    """
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.transform = transform
        self._encode_categorical_features()

    def _encode_categorical_features(self):
        density_map = {'N': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4}
        self.df['density_encoded'] = self.df['density'].map(density_map).fillna(0)

        self.df['site_encoded'] = pd.factorize(self.df['site_id'])[0]
        self.df['machine_encoded'] = pd.factorize(self.df['machine_id'])[0]

        age_min = self.df['age'].min()
        age_max = self.df['age'].max()
        self.df['age_normalized'] = (self.df['age'] - age_min) / (age_max - age_min + 1e-8)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_filename = f"{row['patient_id']}_{row['image_id']}.dcm"
        img_path = self.img_dir / img_filename

        try:
            ds = pydicom.dcmread(str(img_path))
            img_array = ds.pixel_array.astype(np.float32)

            img_min, img_max = img_array.min(), img_array.max()
            img_array = (img_array - img_min) / (img_max - img_min + 1e-8)
            img_array = (img_array * 255).astype(np.uint8)

            img_array = np.stack([img_array, img_array, img_array], axis=-1)
            img_pil = Image.fromarray(img_array, mode='RGB')

            if self.transform:
                img = self.transform(img_pil)
            else:
                img = transforms.ToTensor()(img_pil)

        except Exception as e:
            print(f"Error loading {img_path.name}: {e}")
            img = torch.zeros(3, 224, 224)

        metadata = torch.tensor([
            row['age_normalized'],
            row['density_encoded'] / 4.0,
            row['site_encoded'] / max(self.df['site_encoded'].max(), 1),
            row['machine_encoded'] / max(self.df['machine_encoded'].max(), 1)
        ], dtype=torch.float32)

        label = torch.tensor(row['cancer'], dtype=torch.float32)

        return img, metadata, label


class TransformManager:
    """
    Provides image transformations for training and validation.

    Contains static methods to get standardized transform pipelines
    compatible with ResNet-50 pretrained weights (ImageNet normalization).
    """
    @staticmethod
    def get_train_transform():
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @staticmethod
    def get_val_transform():
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


class DataLoaderManager:
    """
    Creates PyTorch DataLoader instances for training and validation.

    Provides standardized DataLoader configuration with appropriate shuffling
    and batching for mammography dataset.
    """
    @staticmethod
    def create_data_loaders(train_dataset, val_dataset, batch_size=16):
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )

        return train_loader, val_loader
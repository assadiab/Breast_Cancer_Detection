def __getitem__(self, idx):
    row = self.df.iloc[idx]

    # Récupérer image_id depuis le DataFrame
    image_id = str(row['image_id'])
    patient_id = str(row['patient_id'])

    # Chercher fichier DICOM
    # Format attendu: patient_id_image_id.dcm
    image_path = None

    # Essai 1 : image_id seul
    if image_id in self.image_dict:
        image_path = self.image_dict[image_id]
    # Essai 2 : patient_id_image_id
    else:
        combined_name = f"{patient_id}_{image_id}"
        matching = [p for p in self.image_paths if combined_name in p.stem or image_id in p.stem]
        if matching:
            image_path = matching[0]

    # Si toujours pas trouvé, utiliser image noire (éviter crash)
    if image_path is None or not image_path.exists():
        logger.warning(f"⚠️  Image not found for patient={patient_id}, image={image_id}")
        image = torch.zeros(1, 224, 224)
    else:
        # Charger image DICOM
        image = self._load_dicom(image_path)

        # Transform
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image)

    # Label (0 ou 1)
    label = int(row['cancer'])

    sample = {
        'image': image,
        'label': label,
        'patient_id': patient_id,
        'image_id': image_id
    }

    if self.return_indices:
        sample['indices'] = idx

    return sample  # parallel_training.py


"""
Script d'entraînement parallélisé optimisé pour macOS (MPS) avec gestion dynamique des ressources
Métrique principale : F1 Score (adapté aux données déséquilibrées)
"""

# FIX pour macOS : résoudre conflit OpenMP
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'  # Éviter overhead OpenMP

import sys
import time
import json
import psutil
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import logging
from datetime import datetime
import pydicom
from PIL import Image

# Sklearn metrics
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, balanced_accuracy_score,
    roc_curve, precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# Imports de vos modules
from models.multi_expert import OptimizedMultiExpertModel
from models.imbalanced_strategies import ImbalanceStrategyManager, BalancedPatientSampler
from models.enhanced_losses import WeightedFocalLoss, AUCMLoss

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================================
# DATASET DICOM
# ============================================================================
class DICOMDataset(Dataset):
    """Dataset pour images DICOM avec labels CSV"""

    def __init__(
            self,
            image_dir: Path,
            csv_features: Path,
            csv_labels: Path,
            transform=None,
            return_indices: bool = True
    ):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.return_indices = return_indices

        # Charger features et labels
        self.features_df = pd.read_csv(csv_features)
        self.labels_df = pd.read_csv(csv_labels)

        # Y_train.csv n'a que la colonne 'cancer', pas de patient_id
        # On assume que l'ordre des lignes correspond entre X et Y
        if 'patient_id' not in self.labels_df.columns:
            logger.info("ℹ️  Labels CSV has no patient_id, assuming same order as features")
            self.df = self.features_df.copy()

            # Vérifier que les tailles correspondent
            if len(self.features_df) != len(self.labels_df):
                raise ValueError(
                    f"Mismatch: features has {len(self.features_df)} rows, "
                    f"labels has {len(self.labels_df)} rows"
                )

            # Ajouter colonne cancer
            self.df['cancer'] = self.labels_df['cancer'].values
        else:
            # Merge classique si patient_id existe
            self.df = self.features_df.merge(
                self.labels_df,
                on='patient_id',
                how='inner'
            )

        # Vérifier colonnes requises
        required_cols = ['image_id', 'patient_id', 'cancer']
        missing = [col for col in required_cols if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Créer mapping image_id -> path
        # Les fichiers DICOM sont nommés comme: patient_id_image_id.dcm
        self.image_paths = list(self.image_dir.glob("*.dcm"))

        # Créer dict : image_id -> path
        self.image_dict = {}
        for path in self.image_paths:
            # Format: patient_id_image_id.dcm
            stem = path.stem  # Enlève .dcm
            if '_' in stem:
                # Extraire image_id (partie après dernier _)
                image_id = stem.split('_')[-1]
                self.image_dict[image_id] = path

        # Vérifier qu'on a trouvé des images
        if not self.image_dict:
            logger.warning(f"⚠️  No images found in {self.image_dir}")
            logger.warning(f"    Expected format: patientID_imageID.dcm")
            logger.warning(f"    Found {len(self.image_paths)} .dcm files")
            if len(self.image_paths) > 0:
                logger.warning(f"    Example: {self.image_paths[0].name}")

        logger.info(f"📊 Dataset loaded: {len(self.df)} samples")
        logger.info(f"   Images found: {len(self.image_dict)}")
        logger.info(f"   Positive: {self.df['cancer'].sum()} ({self.df['cancer'].mean() * 100:.2f}%)")
        logger.info(f"   Negative: {(1 - self.df['cancer']).sum()}")

        # Statistiques patients
        n_patients = self.df['patient_id'].nunique()
        logger.info(f"   Unique patients: {n_patients}")
        logger.info(f"   Images per patient: {len(self.df) / n_patients:.1f} avg")

    def __len__(self):
        return len(self.df)

    def _load_dicom(self, filepath: Path) -> np.ndarray:
        """Charge et normalise image DICOM avec resize à taille fixe"""
        try:
            dcm = pydicom.dcmread(filepath)
            image = dcm.pixel_array.astype(np.float32)

            # Normalisation basique
            image = (image - image.min()) / (image.max() - image.min() + 1e-8)

            # Convertir en PIL pour resize
            image_pil = Image.fromarray((image * 255).astype(np.uint8))

            # Resize à taille fixe (224x224 pour compatibilité avec backbones)
            image_pil = image_pil.resize((224, 224), Image.BILINEAR)

            # Retour numpy et normaliser [0, 1]
            image = np.array(image_pil).astype(np.float32) / 255.0

            # Convertir en 3D (1, H, W) pour grayscale
            if image.ndim == 2:
                image = image[np.newaxis, ...]

            return image
        except Exception as e:
            logger.warning(f"⚠️  Error loading {filepath}: {e}")
            # Retourner image noire en cas d'erreur
            return np.zeros((1, 224, 224), dtype=np.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Récupérer image path (adapter selon votre naming)
        # Essayer différentes colonnes possibles
        image_name = None
        for col in ['image_id', 'image_name', 'file_name', 'patient_id']:
            if col in row.index:
                image_name = str(row[col])
                break

        if image_name is None:
            # Fallback : utiliser index
            image_name = self.image_paths[idx].stem

        # Chercher fichier DICOM
        if image_name in self.image_dict:
            image_path = self.image_dict[image_name]
        else:
            # Essayer avec extension
            image_path = self.image_dir / f"{image_name}.dcm"
            if not image_path.exists():
                # Dernier recours : premier fichier correspondant
                matching = [p for p in self.image_paths if image_name in p.stem]
                image_path = matching[0] if matching else self.image_paths[idx]

        # Charger image
        image = self._load_dicom(image_path)

        # Transform
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image)

        # Label
        label = int(row['cancer'])

        # Patient ID pour stratification
        patient_id = row.get('patient_id', f"patient_{idx}")

        sample = {
            'image': image,
            'label': label,
            'patient_id': patient_id
        }

        if self.return_indices:
            sample['indices'] = idx

        return sample


# ============================================================================
# DÉTECTION AUTOMATIQUE DES RESSOURCES
# ============================================================================
@dataclass
class SystemResources:
    """Détecte et stocke les ressources disponibles"""
    device: str
    num_gpus: int
    gpu_memory_gb: float
    cpu_cores: int
    ram_gb: float
    optimal_batch_size: int
    optimal_num_workers: int
    use_amp: bool

    def __str__(self):
        return f"""
🖥️  System Resources Detected:
   Device: {self.device}
   GPUs: {self.num_gpus} ({self.gpu_memory_gb:.1f}GB total)
   CPU Cores: {self.cpu_cores}
   RAM: {self.ram_gb:.1f}GB
   Optimal Batch Size: {self.optimal_batch_size}
   Num Workers: {self.optimal_num_workers}
   Mixed Precision: {self.use_amp}
"""


def detect_resources() -> SystemResources:
    """Détecte automatiquement les ressources disponibles sur macOS"""

    if torch.backends.mps.is_available():
        device = "mps"
        num_gpus = 1
        gpu_memory_gb = 16.0 if psutil.virtual_memory().total > 16e9 else 8.0
        use_amp = False
    elif torch.cuda.is_available():
        device = "cuda"
        num_gpus = torch.cuda.device_count()
        gpu_memory_gb = sum(
            torch.cuda.get_device_properties(i).total_memory / 1e9
            for i in range(num_gpus)
        )
        use_amp = True
    else:
        device = "cpu"
        num_gpus = 0
        gpu_memory_gb = 0
        use_amp = False

    cpu_cores = psutil.cpu_count(logical=False)
    ram_gb = psutil.virtual_memory().total / 1e9

    if device == "mps":
        optimal_batch_size = min(16, int(gpu_memory_gb / 2))
    elif device == "cuda":
        optimal_batch_size = min(32, int(gpu_memory_gb / 1.5) * num_gpus)
    else:
        optimal_batch_size = 4

    optimal_num_workers = min(cpu_cores, 8)

    return SystemResources(
        device=device,
        num_gpus=num_gpus,
        gpu_memory_gb=gpu_memory_gb,
        cpu_cores=cpu_cores,
        ram_gb=ram_gb,
        optimal_batch_size=optimal_batch_size,
        optimal_num_workers=optimal_num_workers,
        use_amp=use_amp
    )


# ============================================================================
# METRICS CALCULATOR - COMPLET
# ============================================================================
class MetricsCalculator:
    """Calcule toutes les métriques pour données déséquilibrées"""

    @staticmethod
    def calculate_all_metrics(
            y_true: np.ndarray,
            y_pred_proba: np.ndarray,
            threshold: float = 0.5
    ) -> Dict[str, float]:
        """Calcule ensemble complet de métriques"""

        y_pred = (y_pred_proba >= threshold).astype(int)

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # Métriques de base
        metrics = {
            # Prédictions binaires
            'accuracy': (tp + tn) / (tp + tn + fp + fn),
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0.0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),

            # Confusion matrix
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),

            # Rates
            'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0.0,  # False Positive Rate
            'fnr': fn / (fn + tp) if (fn + tp) > 0 else 0.0,  # False Negative Rate

            # Threshold
            'threshold': threshold
        }

        # Métriques basées sur probabilités
        if len(np.unique(y_true)) > 1:
            metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba)
            metrics['auc_pr'] = average_precision_score(y_true, y_pred_proba)
        else:
            metrics['auc_roc'] = 0.0
            metrics['auc_pr'] = 0.0

        # F2 score (favorise recall)
        beta = 2
        metrics['f2'] = (1 + beta ** 2) * metrics['precision'] * metrics['recall'] / \
                        (beta ** 2 * metrics['precision'] + metrics['recall'] + 1e-8)

        # Youden's J statistic
        metrics['youdens_j'] = metrics['recall'] + metrics['specificity'] - 1

        # Matthews Correlation Coefficient
        mcc_num = (tp * tn) - (fp * fn)
        mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        metrics['mcc'] = mcc_num / mcc_den if mcc_den > 0 else 0.0

        return metrics

    @staticmethod
    def find_optimal_threshold(
            y_true: np.ndarray,
            y_pred_proba: np.ndarray,
            metric: str = 'f1'
    ) -> Tuple[float, Dict[str, float]]:
        """Trouve threshold optimal selon métrique donnée"""

        thresholds = np.linspace(0.01, 0.99, 99)
        best_score = -1
        best_threshold = 0.5
        best_metrics = {}

        for thresh in thresholds:
            metrics = MetricsCalculator.calculate_all_metrics(
                y_true, y_pred_proba, thresh
            )

            score = metrics.get(metric, metrics['f1'])

            if score > best_score:
                best_score = score
                best_threshold = thresh
                best_metrics = metrics

        return best_threshold, best_metrics

    @staticmethod
    def plot_confusion_matrix(
            y_true: np.ndarray,
            y_pred: np.ndarray,
            save_path: Path,
            title: str = "Confusion Matrix"
    ):
        """Plot et sauvegarde matrice de confusion"""
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'],
            cbar_kws={'label': 'Count'}
        )
        plt.title(title, fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    @staticmethod
    def plot_roc_curve(
            y_true: np.ndarray,
            y_pred_proba: np.ndarray,
            save_path: Path,
            title: str = "ROC Curve"
    ):
        """Plot et sauvegarde courbe ROC"""
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        auc = roc_auc_score(y_true, y_pred_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    @staticmethod
    def plot_precision_recall_curve(
            y_true: np.ndarray,
            y_pred_proba: np.ndarray,
            save_path: Path,
            title: str = "Precision-Recall Curve"
    ):
        """Plot et sauvegarde courbe Precision-Recall"""
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        auc_pr = average_precision_score(y_true, y_pred_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, linewidth=2, label=f'PR (AUC = {auc_pr:.4f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(loc="lower left", fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


# ============================================================================
# METRICS TRACKER
# ============================================================================
class MetricsTracker:
    """Track et sauvegarde métriques d'entraînement"""

    def __init__(self, save_dir: Path):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.history = {
            'epoch': [],
            'train_loss': [], 'train_focal': [], 'train_auc_loss': [],
            'val_loss': [],
            'val_auc_roc': [], 'val_auc_pr': [],
            'val_f1': [], 'val_f2': [],
            'val_precision': [], 'val_recall': [], 'val_specificity': [],
            'val_accuracy': [], 'val_balanced_accuracy': [],
            'val_tp': [], 'val_tn': [], 'val_fp': [], 'val_fn': [],
            'val_mcc': [], 'val_youdens_j': [],
            'optimal_threshold': [],
            'learning_rate': [],
            'epoch_time': []
        }

    def update(self, epoch: int, metrics: Dict[str, float]):
        """Ajoute métriques pour une epoch"""
        self.history['epoch'].append(epoch)
        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(value)

    def save(self):
        """Sauvegarde historique en JSON et CSV"""
        # JSON
        json_path = self.save_dir / 'training_history.json'
        with open(json_path, 'w') as f:
            json.dump(self.history, f, indent=2)

        # CSV pour analyse facile
        csv_path = self.save_dir / 'training_history.csv'
        df = pd.DataFrame(self.history)
        df.to_csv(csv_path, index=False)

        logger.info(f"📊 Metrics saved to {json_path} and {csv_path}")

    def get_best_epoch(self, metric: str = 'val_f1') -> Tuple[int, float]:
        """Retourne meilleure epoch et valeur"""
        values = self.history.get(metric, [])
        if not values:
            return 0, 0.0
        best_idx = np.argmax(values)
        return self.history['epoch'][best_idx], values[best_idx]

    def plot_training_curves(self):
        """Plot courbes d'entraînement"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        epochs = self.history['epoch']

        # Loss
        axes[0, 0].plot(epochs, self.history['train_loss'], label='Train')
        axes[0, 0].plot(epochs, self.history['val_loss'], label='Val')
        axes[0, 0].set_title('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)

        # F1 & F2
        axes[0, 1].plot(epochs, self.history['val_f1'], label='F1')
        axes[0, 1].plot(epochs, self.history['val_f2'], label='F2')
        axes[0, 1].set_title('F1 & F2 Scores')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)

        # AUC ROC & PR
        axes[0, 2].plot(epochs, self.history['val_auc_roc'], label='AUC-ROC')
        axes[0, 2].plot(epochs, self.history['val_auc_pr'], label='AUC-PR')
        axes[0, 2].set_title('AUC Scores')
        axes[0, 2].legend()
        axes[0, 2].grid(alpha=0.3)

        # Precision & Recall
        axes[1, 0].plot(epochs, self.history['val_precision'], label='Precision')
        axes[1, 0].plot(epochs, self.history['val_recall'], label='Recall')
        axes[1, 0].plot(epochs, self.history['val_specificity'], label='Specificity')
        axes[1, 0].set_title('Precision, Recall, Specificity')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)

        # Accuracy
        axes[1, 1].plot(epochs, self.history['val_accuracy'], label='Accuracy')
        axes[1, 1].plot(epochs, self.history['val_balanced_accuracy'], label='Balanced Acc')
        axes[1, 1].set_title('Accuracy Metrics')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)

        # Learning Rate
        axes[1, 2].plot(epochs, self.history['learning_rate'])
        axes[1, 2].set_title('Learning Rate')
        axes[1, 2].set_yscale('log')
        axes[1, 2].grid(alpha=0.3)

        plt.tight_layout()
        save_path = self.save_dir / 'training_curves.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"📈 Training curves saved to {save_path}")


# ============================================================================
# PARALLEL TRAINER
# ============================================================================
class ParallelTrainer:
    """Trainer optimisé avec parallélisation automatique"""

    def __init__(
            self,
            model: nn.Module,
            train_dataset,
            val_dataset,
            resources: SystemResources,
            config: Dict,
            save_dir: str = "./checkpoints"
    ):
        self.config = config
        self.resources = resources
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Créer sous-dossiers pour visualisations
        (self.save_dir / 'plots').mkdir(exist_ok=True)
        (self.save_dir / 'confusion_matrices').mkdir(exist_ok=True)

        self.device = torch.device(resources.device)
        logger.info(f"🎯 Using device: {self.device}")

        if resources.num_gpus > 1 and resources.device == "cuda":
            logger.info(f"🔗 Wrapping model with DataParallel ({resources.num_gpus} GPUs)")
            model = nn.DataParallel(model)

        self.model = model.to(self.device)

        # Strategy Manager
        self.strategy_manager = ImbalanceStrategyManager(
            cancer_rate=config.get('cancer_rate', 0.0732),
            use_mixup=config.get('use_mixup', True),
            use_hard_mining=config.get('use_hard_mining', True),
            device=str(self.device)
        )

        # DataLoaders
        batch_size = config.get('batch_size', resources.optimal_batch_size)
        num_workers = config.get('num_workers', resources.optimal_num_workers)

        logger.info(f"📦 Batch size: {batch_size}, Workers: {num_workers}")

        train_labels = train_dataset.df['cancer'].values
        train_patient_ids = train_dataset.df['patient_id'].values

        train_sampler = BalancedPatientSampler(
            patient_ids=train_patient_ids,
            labels=train_labels,
            samples_per_epoch=None,
            pos_weight=self.strategy_manager.pos_weight
        )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=(resources.device == "cuda"),
            persistent_workers=True if num_workers > 0 else False,
            drop_last=True
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(resources.device == "cuda"),
            persistent_workers=True if num_workers > 0 else False
        )

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.get('lr', 1e-4),
            weight_decay=config.get('weight_decay', 1e-4)
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config.get('scheduler_t0', 10),
            T_mult=2,
            eta_min=config.get('lr_min', 1e-6)
        )

        # Mixed Precision
        self.scaler = GradScaler() if resources.use_amp else None
        self.grad_accum_steps = config.get('grad_accum_steps', 1)

        # Metrics
        self.metrics_tracker = MetricsTracker(self.save_dir)
        self.metrics_calculator = MetricsCalculator()

        # Best model tracking - BASÉ SUR F1
        self.best_val_f1 = 0.0
        self.best_val_metrics = {}
        self.patience_counter = 0
        self.patience = config.get('patience', 15)

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Entraîne une epoch"""
        self.model.train()

        running_loss = 0.0
        total_focal = 0.0
        total_auc = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device, non_blocking=True)
            labels = batch['label'].to(self.device, non_blocking=True).float().unsqueeze(1)

            # Mixup
            images, labels = self.strategy_manager.apply_mixup(images, labels)

            # Forward
            if self.scaler:
                with autocast():
                    preds, gates, embeddings = self.model(images)
                    loss_dict = self.strategy_manager.compute_loss(preds, labels)
                    loss = loss_dict['total'] / self.grad_accum_steps
            else:
                preds, gates, embeddings = self.model(images)
                loss_dict = self.strategy_manager.compute_loss(preds, labels)
                loss = loss_dict['total'] / self.grad_accum_steps

            # Backward
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                self.optimizer.zero_grad()

            # Update hard mining
            if 'indices' in batch:
                self.strategy_manager.update_hard_mining(
                    batch['indices'],
                    preds.detach(),
                    labels
                )

            # Tracking
            running_loss += loss.item() * self.grad_accum_steps
            total_focal += loss_dict['focal'].item()
            total_auc += loss_dict['auc'].item()
            num_batches += 1

            pbar.set_postfix({
                'loss': f"{running_loss / num_batches:.4f}",
                'focal': f"{total_focal / num_batches:.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })

        self.scheduler.step()

        return {
            'train_loss': running_loss / num_batches,
            'train_focal': total_focal / num_batches,
            'train_auc_loss': total_auc / num_batches,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validation avec métriques complètes"""
        self.model.eval()

        all_preds = []
        all_labels = []
        running_loss = 0.0

        for batch in tqdm(self.val_loader, desc="Validation"):
            images = batch['image'].to(self.device, non_blocking=True)
            labels = batch['label'].to(self.device, non_blocking=True).float().unsqueeze(1)

            if self.scaler:
                with autocast():
                    preds, _, _ = self.model(images)
                    loss_dict = self.strategy_manager.compute_loss(preds, labels)
            else:
                preds, _, _ = self.model(images)
                loss_dict = self.strategy_manager.compute_loss(preds, labels)

            running_loss += loss_dict['total'].item()

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

        # Calcul métriques
        all_preds = torch.cat(all_preds).numpy().flatten()
        all_labels = torch.cat(all_labels).numpy().flatten()

        # Trouver threshold optimal basé sur F1
        optimal_threshold, optimal_metrics = self.metrics_calculator.find_optimal_threshold(
            all_labels,
            all_preds,
            metric='f1'
        )

        # Calculer aussi métriques à threshold 0.5 pour comparaison
        metrics_05 = self.metrics_calculator.calculate_all_metrics(
            all_labels, all_preds, threshold=0.5
        )

        # Utiliser métriques optimales
        metrics = optimal_metrics.copy()
        metrics['val_loss'] = running_loss / len(self.val_loader)

        # Renommer pour cohérence avec tracker
        metric_mapping = {
            'auc_roc': 'val_auc_roc',
            'auc_pr': 'val_auc_pr',
            'f1': 'val_f1',
            'f2': 'val_f2',
            'precision': 'val_precision',
            'recall': 'val_recall',
            'specificity': 'val_specificity',
            'accuracy': 'val_accuracy',
            'balanced_accuracy': 'val_balanced_accuracy',
            'true_positives': 'val_tp',
            'true_negatives': 'val_tn',
            'false_positives': 'val_fp',
            'false_negatives': 'val_fn',
            'mcc': 'val_mcc',
            'youdens_j': 'val_youdens_j',
            'threshold': 'optimal_threshold'
        }

        val_metrics = {}
        for old_key, new_key in metric_mapping.items():
            if old_key in metrics:
                val_metrics[new_key] = metrics[old_key]
        val_metrics['val_loss'] = metrics['val_loss']

        # AFFICHAGE DÉTAILLÉ DES MÉTRIQUES
        logger.info("\n" + "=" * 80)
        logger.info(f"📊 VALIDATION METRICS - Epoch {epoch}")
        logger.info("=" * 80)
        logger.info(f"🎯 PRIMARY METRIC (for best model):")
        logger.info(f"   F1 Score: {val_metrics['val_f1']:.4f}")
        logger.info(f"\n📈 CLASSIFICATION METRICS:")
        logger.info(f"   Accuracy:          {val_metrics['val_accuracy']:.4f}")
        logger.info(f"   Balanced Accuracy: {val_metrics['val_balanced_accuracy']:.4f}")
        logger.info(f"   Precision:         {val_metrics['val_precision']:.4f}")
        logger.info(f"   Recall:            {val_metrics['val_recall']:.4f}")
        logger.info(f"   Specificity:       {val_metrics['val_specificity']:.4f}")
        logger.info(f"   F2 Score:          {val_metrics['val_f2']:.4f}")
        logger.info(f"\n📊 AUC METRICS:")
        logger.info(f"   AUC-ROC:           {val_metrics['val_auc_roc']:.4f}")
        logger.info(f"   AUC-PR:            {val_metrics['val_auc_pr']:.4f}")
        logger.info(f"\n🎲 CONFUSION MATRIX:")
        logger.info(f"   True Positives:    {val_metrics['val_tp']}")
        logger.info(f"   True Negatives:    {val_metrics['val_tn']}")
        logger.info(f"   False Positives:   {val_metrics['val_fp']}")
        logger.info(f"   False Negatives:   {val_metrics['val_fn']}")
        logger.info(f"\n📉 OTHER METRICS:")
        logger.info(f"   MCC:               {val_metrics['val_mcc']:.4f}")
        logger.info(f"   Youden's J:        {val_metrics['val_youdens_j']:.4f}")
        logger.info(f"   Optimal Threshold: {val_metrics['optimal_threshold']:.3f}")
        logger.info(f"   Validation Loss:   {val_metrics['val_loss']:.4f}")
        logger.info("=" * 80 + "\n")

        # Comparaison avec threshold 0.5
        logger.info(f"📌 Comparison with threshold=0.5:")
        logger.info(f"   F1 (t=0.5):   {metrics_05['f1']:.4f} vs F1 (optimal): {val_metrics['val_f1']:.4f}")
        logger.info(
            f"   Recall (t=0.5): {metrics_05['recall']:.4f} vs Recall (optimal): {val_metrics['val_recall']:.4f}")
        logger.info(
            f"   Precision (t=0.5): {metrics_05['precision']:.4f} vs Precision (optimal): {val_metrics['val_precision']:.4f}\n")

        # Classification report détaillé
        y_pred_binary = (all_preds >= optimal_threshold).astype(int)
        report = classification_report(
            all_labels,
            y_pred_binary,
            target_names=['Negative', 'Positive'],
            digits=4
        )
        logger.info("📋 CLASSIFICATION REPORT:")
        logger.info("\n" + report)

        # Plot matrice de confusion
        cm_path = self.save_dir / 'confusion_matrices' / f'confusion_matrix_epoch_{epoch}.png'
        self.metrics_calculator.plot_confusion_matrix(
            all_labels,
            y_pred_binary,
            cm_path,
            title=f'Confusion Matrix - Epoch {epoch} (F1={val_metrics["val_f1"]:.4f})'
        )

        # Plot courbes ROC et PR toutes les 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            roc_path = self.save_dir / 'plots' / f'roc_curve_epoch_{epoch}.png'
            self.metrics_calculator.plot_roc_curve(
                all_labels, all_preds, roc_path,
                title=f'ROC Curve - Epoch {epoch} (AUC={val_metrics["val_auc_roc"]:.4f})'
            )

            pr_path = self.save_dir / 'plots' / f'pr_curve_epoch_{epoch}.png'
            self.metrics_calculator.plot_precision_recall_curve(
                all_labels, all_preds, pr_path,
                title=f'Precision-Recall Curve - Epoch {epoch} (AUC={val_metrics["val_auc_pr"]:.4f})'
            )

        return val_metrics

    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """Sauvegarde checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config,
            'best_val_f1': self.best_val_f1,
            'best_val_metrics': self.best_val_metrics
        }

        # Checkpoint régulier
        checkpoint_path = self.save_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)

        # Best model (basé sur F1)
        if is_best:
            best_path = self.save_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"💾 ⭐ BEST MODEL SAVED! F1: {metrics['val_f1']:.4f} (Epoch {epoch})")

            # Sauvegarder aussi les métriques du best model
            best_metrics_path = self.save_dir / 'best_model_metrics.json'
            with open(best_metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)

        # Cleanup anciens checkpoints (garder 3 derniers)
        checkpoints = sorted(self.save_dir.glob("checkpoint_epoch_*.pt"))
        if len(checkpoints) > 3:
            for old_ckpt in checkpoints[:-3]:
                old_ckpt.unlink()

    def train(self, num_epochs: int):
        """Boucle d'entraînement principale"""
        logger.info(f"\n{'=' * 80}")
        logger.info(f"🚀 STARTING TRAINING FOR {num_epochs} EPOCHS")
        logger.info(f"{'=' * 80}\n")
        logger.info(str(self.resources))
        logger.info(f"📌 Primary metric for best model: F1 SCORE")
        logger.info(f"📌 Early stopping patience: {self.patience} epochs\n")

        start_time = time.time()

        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()

            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate(epoch)

            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics}
            epoch_metrics['epoch_time'] = time.time() - epoch_start

            # GPU memory
            if self.resources.device == "cuda":
                epoch_metrics['gpu_memory'] = torch.cuda.max_memory_allocated() / 1e9

            # Update tracker
            self.metrics_tracker.update(epoch, epoch_metrics)

            # Summary log
            logger.info(f"\n{'=' * 80}")
            logger.info(f"📊 EPOCH {epoch}/{num_epochs} SUMMARY")
            logger.info(f"{'=' * 80}")
            logger.info(f"⏱️  Time: {epoch_metrics['epoch_time']:.1f}s")
            logger.info(f"📉 Train Loss: {train_metrics['train_loss']:.4f} | Val Loss: {val_metrics['val_loss']:.4f}")
            logger.info(f"🎯 F1: {val_metrics['val_f1']:.4f} | AUC-ROC: {val_metrics['val_auc_roc']:.4f}")
            logger.info(f"📊 Precision: {val_metrics['val_precision']:.4f} | Recall: {val_metrics['val_recall']:.4f}")
            logger.info(f"{'=' * 80}\n")

            # Check if best model (basé sur F1)
            is_best = val_metrics['val_f1'] > self.best_val_f1
            if is_best:
                improvement = val_metrics['val_f1'] - self.best_val_f1
                logger.info(f"✨ NEW BEST F1! {self.best_val_f1:.4f} → {val_metrics['val_f1']:.4f} (+{improvement:.4f})")
                self.best_val_f1 = val_metrics['val_f1']
                self.best_val_metrics = val_metrics.copy()
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                logger.info(f"⏳ No improvement. Patience: {self.patience_counter}/{self.patience}")

            # Save checkpoint
            self.save_checkpoint(epoch, epoch_metrics, is_best)

            # Early stopping
            if self.patience_counter >= self.patience:
                logger.info(f"\n{'=' * 80}")
                logger.info(f"⚠️  EARLY STOPPING at epoch {epoch}")
                logger.info(f"   No improvement for {self.patience} epochs")
                logger.info(
                    f"   Best F1: {self.best_val_f1:.4f} at epoch {self.metrics_tracker.get_best_epoch('val_f1')[0]}")
                logger.info(f"{'=' * 80}\n")
                break

        # Training finished
        total_time = time.time() - start_time
        best_epoch, best_f1 = self.metrics_tracker.get_best_epoch('val_f1')

        logger.info(f"\n{'=' * 80}")
        logger.info(f"✅ TRAINING COMPLETE!")
        logger.info(f"{'=' * 80}")
        logger.info(f"⏱️  Total Time: {total_time / 3600:.2f}h ({total_time / 60:.1f}min)")
        logger.info(f"🏆 Best F1 Score: {best_f1:.4f} (Epoch {best_epoch})")
        logger.info(f"\n📊 Best Model Metrics:")
        for key, value in self.best_val_metrics.items():
            if isinstance(value, (int, float)):
                logger.info(f"   {key}: {value:.4f}" if isinstance(value, float) else f"   {key}: {value}")
        logger.info(f"{'=' * 80}\n")

        # Save final metrics et plots
        self.metrics_tracker.save()
        self.metrics_tracker.plot_training_curves()

        # Plot final confusion matrix du best model
        logger.info("📊 Generating final reports...")
        self._generate_final_report()

    def _generate_final_report(self):
        """Génère rapport final avec best model"""
        # Charger best model
        best_model_path = self.save_dir / 'best_model.pt'
        if not best_model_path.exists():
            logger.warning("⚠️  Best model not found, skipping final report")
            return

        checkpoint = torch.load(best_model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        best_epoch = checkpoint['epoch']

        # Re-valider pour obtenir prédictions
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Generating final report"):
                images = batch['image'].to(self.device, non_blocking=True)
                labels = batch['label'].cpu().numpy()

                preds, _, _ = self.model(images)
                preds = preds.cpu().numpy().flatten()

                all_preds.extend(preds)
                all_labels.extend(labels)

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # Metrics avec threshold optimal
        optimal_threshold = checkpoint['metrics']['optimal_threshold']
        final_metrics = self.metrics_calculator.calculate_all_metrics(
            all_labels, all_preds, optimal_threshold
        )

        # Plot confusion matrix finale
        y_pred_binary = (all_preds >= optimal_threshold).astype(int)
        final_cm_path = self.save_dir / 'confusion_matrix_best_model.png'
        self.metrics_calculator.plot_confusion_matrix(
            all_labels, y_pred_binary, final_cm_path,
            title=f'Best Model Confusion Matrix (Epoch {best_epoch}, F1={final_metrics["f1"]:.4f})'
        )

        # Plot courbes finales
        final_roc_path = self.save_dir / 'roc_curve_best_model.png'
        self.metrics_calculator.plot_roc_curve(
            all_labels, all_preds, final_roc_path,
            title=f'Best Model ROC Curve (AUC={final_metrics["auc_roc"]:.4f})'
        )

        final_pr_path = self.save_dir / 'pr_curve_best_model.png'
        self.metrics_calculator.plot_precision_recall_curve(
            all_labels, all_preds, final_pr_path,
            title=f'Best Model PR Curve (AUC={final_metrics["auc_pr"]:.4f})'
        )

        # Sauvegarder rapport texte
        report_path = self.save_dir / 'final_report.txt'
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"FINAL EVALUATION REPORT - BEST MODEL (Epoch {best_epoch})\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Optimal Threshold: {optimal_threshold:.4f}\n\n")
            f.write("METRICS:\n")
            f.write("-" * 40 + "\n")
            for key, value in final_metrics.items():
                if isinstance(value, (int, float)):
                    f.write(f"{key:25s}: {value:.4f}\n" if isinstance(value, float) else f"{key:25s}: {value}\n")
            f.write("\n" + "=" * 80 + "\n")

        logger.info(f"📄 Final report saved to {report_path}")
        logger.info(f"📊 All plots saved in {self.save_dir}")


# ============================================================================
# FONCTION PRINCIPALE
# ============================================================================
def main():
    """Point d'entrée principal"""

    # Configuration
    config = {
        # Paths - ADAPTER À VOTRE STRUCTURE
        'data_dir': './data',
        'train_dir': './data/train',
        'val_dir': './data/val',
        'train_features_csv': './data/csv/X_train.csv',
        'train_labels_csv': './data/csv/y_train.csv',
        'val_features_csv': './data/csv/X_val.csv',
        'val_labels_csv': './data/csv/y_val.csv',
        'save_dir': './checkpoints',

        # Model
        'embed_dim': 256,
        'use_checkpoint': True,

        # Training
        'num_epochs': 100,
        'batch_size': None,  # Auto-detect
        'num_workers': None,  # Auto-detect
        'lr': 3e-4,
        'lr_min': 1e-6,
        'weight_decay': 1e-4,
        'grad_accum_steps': 2,

        # Scheduler
        'scheduler_t0': 10,

        # Imbalance (adapter à votre taux réel)
        'cancer_rate': 0.0732,
        'use_mixup': True,
        'use_hard_mining': True,

        # Early stopping (basé sur F1)
        'patience': 15,
    }

    # Detect resources
    logger.info("🔍 Detecting system resources...")
    resources = detect_resources()
    logger.info(str(resources))

    # Update config
    if config['batch_size'] is None:
        config['batch_size'] = resources.optimal_batch_size
    if config['num_workers'] is None:
        config['num_workers'] = resources.optimal_num_workers

    # Load datasets
    logger.info("📂 Loading DICOM datasets...")

    train_dataset = DICOMDataset(
        image_dir=Path(config['train_dir']),
        csv_features=Path(config['train_features_csv']),
        csv_labels=Path(config['train_labels_csv']),
        return_indices=True
    )

    val_dataset = DICOMDataset(
        image_dir=Path(config['val_dir']),
        csv_features=Path(config['val_features_csv']),
        csv_labels=Path(config['val_labels_csv']),
        return_indices=False
    )

    logger.info(f"✅ Train dataset: {len(train_dataset)} samples")
    logger.info(f"✅ Val dataset: {len(val_dataset)} samples")

    # Initialize model
    logger.info("🏗️  Building multi-expert model...")
    model = OptimizedMultiExpertModel(
        embed_dim=config['embed_dim'],
        use_checkpoint=config['use_checkpoint']
    )

    # Optionnel : freeze backbones pour warmup
    # model.freeze_backbones()
    # logger.info("❄️  Backbones frozen for warmup")

    # Create trainer
    logger.info("🎯 Initializing parallel trainer...")
    trainer = ParallelTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        resources=resources,
        config=config,
        save_dir=config['save_dir']
    )

    # Train
    logger.info("\n🚀 Starting training...\n")
    trainer.train(num_epochs=config['num_epochs'])

    logger.info("\n🎉 Training pipeline complete!")
    logger.info(f"📁 All results saved to: {config['save_dir']}")


if __name__ == "__main__":
    # Setup multiprocessing pour macOS
    mp.set_start_method('spawn', force=True)

    # Run
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n⚠️  Training interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Error during training: {e}", exc_info=True)
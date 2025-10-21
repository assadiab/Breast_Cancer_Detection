# train_final_mps.py
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import numpy as np
from pathlib import Path
import sys
from datetime import datetime
import warnings
import time
import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional, Tuple
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, \
    balanced_accuracy_score

warnings.filterwarnings('ignore')


# ==========================================================
# CONFIGURATION MPS
# ==========================================================
@dataclass
class OptimizedConfig:
    # Training
    epochs: int = 20
    batch_size: int = 4  # Plus petit pour MPS
    grad_accum_steps: int = 4  # Accumulation pour simuler batch plus grand
    lr: float = 1.5e-4
    weight_decay: float = 1e-4

    # Image sizes
    high_res: Tuple[int, int] = (512, 512)
    low_res: Tuple[int, int] = (224, 224)

    # Device - MPS pour Apple Silicon
    device: str = "mps"
    num_workers: int = 4  # Moins de workers pour MPS
    pin_memory: bool = False  # Pas de pin_memory pour MPS

    # Model
    embed_dim: int = 256
    use_checkpoint: bool = True

    # Training strategy
    freeze_epochs: int = 4
    warmup_epochs: int = 2

    # Loss
    focal_alpha: float = 0.75
    focal_gamma: float = 2.5

    # Paths
    save_dir: str = "checkpoints_final_mps"


# ==========================================================
# DATASET MPS OPTIMISÉ
# ==========================================================
class SimpleMedicalAugmentation:
    """Transformations simples optimisées pour MPS"""

    def __init__(self, img_size: Tuple[int, int], is_train: bool = True):
        self.img_size = img_size
        self.is_train = is_train

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        from PIL import Image

        # Redimensionnement
        img_pil = Image.fromarray((image * 255).astype(np.uint8))
        img_resized = img_pil.resize(self.img_size, Image.BILINEAR)
        img_array = np.array(img_resized) / 255.0

        # Conversion tensor
        img_tensor = torch.from_numpy(img_array).float()
        img_tensor = img_tensor.unsqueeze(0)  # Add channel dimension

        # Normalisation
        img_tensor = (img_tensor - 0.5) / 0.5

        return img_tensor


class OptimizedDicomDataset(Dataset):
    def __init__(
            self,
            x_csv: str,
            dicom_root: str,
            y_csv: Optional[str] = None,
            config: OptimizedConfig = None,
            is_train: bool = True
    ):
        self.df = pd.read_csv(x_csv)
        self.dicom_root = Path(dicom_root)
        self.is_train = is_train
        self.config = config or OptimizedConfig()

        if is_train:
            if y_csv is None:
                raise ValueError("Training requires y_csv")
            self.labels = pd.read_csv(y_csv)
        else:
            self.labels = None

        # Transformations
        self.transform = SimpleMedicalAugmentation(
            img_size=self.config.high_res,
            is_train=is_train
        )
        self.transform_low = SimpleMedicalAugmentation(
            img_size=self.config.low_res,
            is_train=False
        )

        self._cache = {}

    def _load_dicom(self, idx: int) -> np.ndarray:
        if idx in self._cache:
            return self._cache[idx]

        row = self.df.iloc[idx]
        path = self.dicom_root / f"{row['patient_id']}_{row['image_id']}.dcm"

        try:
            import pydicom
            ds = pydicom.dcmread(path)
            img = ds.pixel_array.astype(np.float32)

            # Normalisation robuste
            p1, p99 = np.percentile(img, (1, 99))
            img = np.clip(img, p1, p99)
            img = (img - p1) / (p99 - p1 + 1e-8)

            self._cache[idx] = img
            return img
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return np.zeros((512, 512), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        img = self._load_dicom(idx)

        # Transformations
        img_high = self.transform(img)
        img_low = self.transform_low(img)

        if self.is_train:
            label = torch.tensor(
                self.labels.iloc[idx]['cancer'],
                dtype=torch.float32
            )
            return img_high, img_low, label, torch.tensor(idx)

        return img_high, img_low


# ==========================================================
# EMA POUR MPS
# ==========================================================
class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + \
                                    (1 - self.decay) * param.data

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


# ==========================================================
# STRATÉGIES DÉSÉQUILIBRE MPS
# ==========================================================
class SimpleImbalanceManager:
    def __init__(self, cancer_rate=0.0732):
        self.pos_weight = 1.0 / cancer_rate

    def create_balanced_loader(self, dataset, batch_size, num_workers=4):
        """Crée un DataLoader équilibré pour MPS"""
        # Calcul des poids
        labels = np.array([dataset.labels.iloc[i]['cancer'] for i in range(len(dataset))])
        class_counts = np.bincount(labels.astype(int))
        class_weights = 1.0 / class_counts
        sample_weights = [class_weights[label] for label in labels]

        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=False,  # Pas de pin_memory pour MPS
            persistent_workers=True if num_workers > 0 else False
        )


class AdaptiveFocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, targets):
        bce = F.binary_cross_entropy(preds, targets, reduction='none')
        pt = torch.exp(-bce)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce
        return focal_loss.mean()


# ==========================================================
# MÉTRIQUES
# ==========================================================
class MetricsLogger:
    def __init__(self):
        self.epoch_metrics = []
        self.best_metrics = {}

    def log_epoch(self, epoch, train_metrics, val_metrics, learning_rate, epoch_time):
        metrics = {
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'train_auc': train_metrics['auc'],
            'train_ap': train_metrics['average_precision'],
            'val_auc': val_metrics['auc'],
            'val_ap': val_metrics['average_precision'],
            'val_f1': val_metrics['f1'],
            'val_precision': val_metrics['precision'],
            'val_recall': val_metrics['recall'],
            'val_balanced_accuracy': val_metrics['balanced_accuracy'],
            'learning_rate': learning_rate,
            'epoch_time': epoch_time
        }

        self.epoch_metrics.append(metrics)
        self._update_best_metrics(metrics)
        self._print_epoch_metrics(metrics)

        return metrics

    def _update_best_metrics(self, metrics):
        if not self.best_metrics:
            self.best_metrics = metrics.copy()
            return

        if metrics['val_auc'] > self.best_metrics['val_auc']:
            self.best_metrics.update({k: metrics[k] for k in ['val_auc', 'epoch']})

    def _print_epoch_metrics(self, metrics):
        print(f"\n{'=' * 80}")
        print(f"📊 EPOCH {metrics['epoch']:02d} - RÉSULTATS MPS")
        print(f"{'=' * 80}")
        print(f"⏰ Temps: {metrics['epoch_time']:.1f}s | LR: {metrics['learning_rate']:.2e}")

        print(f"\n🏋️  TRAIN:")
        print(f"   📉 Loss: {metrics['train_loss']:.4f}")
        print(f"   📈 AUC: {metrics['train_auc']:.4f}")
        print(f"   🎯 AP: {metrics['train_ap']:.4f}")

        print(f"\n🧪 VALIDATION:")
        best_flag = "🎯 BEST" if metrics['val_auc'] == self.best_metrics.get('val_auc', 0) else ""
        print(f"   📊 AUC: {metrics['val_auc']:.4f} {best_flag}")
        print(f"   🎯 AP: {metrics['val_ap']:.4f}")
        print(f"   ⚖️  F1: {metrics['val_f1']:.4f}")
        print(f"   🎯 Precision: {metrics['val_precision']:.4f}")
        print(f"   🔄 Recall: {metrics['val_recall']:.4f}")
        print(f"   ⚖️  Balanced Acc: {metrics['val_balanced_accuracy']:.4f}")


# ==========================================================
# TRAINER MPS
# ==========================================================
class MpsTrainer:
    def __init__(self, config: OptimizedConfig):
        self.cfg = config
        self.device = self._setup_device()

        # Modèle
        from models.multi_expert import OptimizedMultiExpertModel
        self.model = OptimizedMultiExpertModel(
            embed_dim=config.embed_dim,
            use_checkpoint=config.use_checkpoint
        ).to(self.device)

        # Gestionnaires
        self.imbalance_manager = SimpleImbalanceManager(cancer_rate=0.0732)
        self.loss_fn = AdaptiveFocalLoss(alpha=0.75, gamma=2.5)
        self.ema = EMA(self.model, decay=0.999)
        self.metrics_logger = MetricsLogger()

        Path(config.save_dir).mkdir(exist_ok=True)
        print("✅ Trainer MPS initialisé")

    def _setup_device(self):
        if torch.backends.mps.is_available():
            device = torch.device('mps')
            # Optimisation mémoire MPS
            torch.mps.set_per_process_memory_fraction(0.7)
            print("🚀 MPS (Apple Silicon) disponible et configuré")
            return device
        else:
            print("⚠️  MPS non disponible, utilisation CPU")
            return torch.device("cpu")

    def setup_data(self, train_csv, val_csv, train_y_csv, val_y_csv, dicom_root):
        print("📊 Configuration des données MPS...")

        self.train_dataset = OptimizedDicomDataset(
            train_csv, dicom_root, train_y_csv, self.cfg, is_train=True
        )
        self.val_dataset = OptimizedDicomDataset(
            val_csv, dicom_root, val_y_csv, self.cfg, is_train=True
        )

        # DataLoaders optimisés MPS
        self.train_loader = self.imbalance_manager.create_balanced_loader(
            self.train_dataset, self.cfg.batch_size, self.cfg.num_workers
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.cfg.batch_size * 2,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=False  # Important pour MPS
        )

        print(f"✅ Données chargées: {len(self.train_dataset)} train, {len(self.val_dataset)} val")

    def compute_metrics(self, preds, targets):
        if len(np.unique(targets)) < 2:
            return {'auc': 0.5, 'average_precision': 0.5, 'f1': 0.0,
                    'precision': 0.0, 'recall': 0.0, 'balanced_accuracy': 0.5}

        auc = roc_auc_score(targets, preds)
        ap = average_precision_score(targets, preds)

        # Optimisation threshold
        thresholds = np.linspace(0.1, 0.9, 50)
        best_f1 = 0

        for thresh in thresholds:
            preds_binary = (preds >= thresh).astype(int)
            f1 = f1_score(targets, preds_binary, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1

        best_preds = (preds >= 0.5).astype(int)  # Utiliser 0.5 pour les autres métriques
        precision = precision_score(targets, best_preds, zero_division=0)
        recall = recall_score(targets, best_preds, zero_division=0)
        balanced_acc = balanced_accuracy_score(targets, best_preds)

        return {
            'auc': auc, 'average_precision': ap, 'f1': best_f1,
            'precision': precision, 'recall': recall, 'balanced_accuracy': balanced_acc
        }

    def train_epoch(self, optimizer, epoch):
        self.model.train()
        total_loss = 0
        all_preds, all_targets = [], []

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train MPS]")

        for batch_idx, (img_high, img_low, labels, indices) in enumerate(pbar):
            img_high = img_high.to(self.device)
            img_low = img_low.to(self.device)
            labels = labels.to(self.device)

            # Forward
            preds, gates, embeddings = self.model(img_high, img_low)
            loss = self.loss_fn(preds.squeeze(), labels) / self.cfg.grad_accum_steps

            # Backward
            loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.cfg.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                self.ema.update()

            total_loss += loss.item() * self.cfg.grad_accum_steps
            all_preds.extend(preds.squeeze().detach().cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

            # Monitoring mémoire MPS
            if batch_idx % 20 == 0:
                if self.device.type == 'mps':
                    mem = torch.mps.current_allocated_memory() / 1024 ** 3
                    pbar.set_postfix({
                        'loss': f"{loss.item() * self.cfg.grad_accum_steps:.4f}",
                        'mem_gb': f"{mem:.2f}"
                    })
                else:
                    pbar.set_postfix({
                        'loss': f"{loss.item() * self.cfg.grad_accum_steps:.4f}"
                    })

            # Nettoyage mémoire MPS
            if batch_idx % 50 == 0 and self.device.type == 'mps':
                torch.mps.empty_cache()

        train_metrics = self.compute_metrics(np.array(all_preds), np.array(all_targets))
        train_metrics['loss'] = total_loss / len(self.train_loader)

        return train_metrics

    @torch.no_grad()
    def validate(self):
        self.ema.apply_shadow()
        self.model.eval()

        all_preds, all_targets = [], []

        for img_high, img_low, labels in tqdm(self.val_loader, desc="Validation MPS"):
            img_high = img_high.to(self.device)
            img_low = img_low.to(self.device)
            labels = labels.to(self.device)

            preds, gates, _ = self.model(img_high, img_low)
            all_preds.extend(preds.squeeze().cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

        self.ema.restore()
        return self.compute_metrics(np.array(all_preds), np.array(all_targets))

    def fit(self, train_csv, val_csv, train_y_csv, val_y_csv, dicom_root):
        print("🚀 DÉMARRAGE ENTRAÎNEMENT MPS")
        print("=" * 80)

        self.setup_data(train_csv, val_csv, train_y_csv, val_y_csv, dicom_root)

        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
            betas=(0.9, 0.999)
        )

        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, T_max=self.cfg.epochs)

        for epoch in range(self.cfg.epochs):
            epoch_start = time.time()

            # Unfreeze après certaines epochs
            if epoch == self.cfg.freeze_epochs:
                print("🔓 Déblocage des backbones...")
                self.model.unfreeze_backbones()

            # Entraînement
            train_metrics = self.train_epoch(optimizer, epoch)
            val_metrics = self.validate()

            # Métriques
            epoch_time = time.time() - epoch_start
            current_lr = optimizer.param_groups[0]['lr']

            self.metrics_logger.log_epoch(
                epoch + 1, train_metrics, val_metrics, current_lr, epoch_time
            )

            # Sauvegarde best model
            if val_metrics['auc'] > self.metrics_logger.best_metrics.get('val_auc', 0):
                torch.save({
                    'epoch': epoch,
                    'model_state': self.model.state_dict(),
                    'ema_shadow': self.ema.shadow,
                    'metrics': val_metrics,
                    'config': self.cfg
                }, f"{self.cfg.save_dir}/best_auc_{val_metrics['auc']:.4f}.pth")
                print(f"💾 Best model saved (AUC: {val_metrics['auc']:.4f})")

            scheduler.step()

        # Final save
        self.ema.apply_shadow()
        torch.save({
            'epoch': self.cfg.epochs,
            'model_state': self.model.state_dict(),
            'ema_shadow': self.ema.shadow,
            'config': self.cfg
        }, f"{self.cfg.save_dir}/final_ema.pth")

        print(f"\n🎉 ENTRAÎNEMENT MPS TERMINÉ!")
        print(f"🏆 Best AUC: {self.metrics_logger.best_metrics['val_auc']:.4f}")


# ==========================================================
# LANCEMENT MPS
# ==========================================================
def main():
    cfg = OptimizedConfig()

    # Chemins
    DATA_ROOT = "../data"
    TRAIN_CSV = f"{DATA_ROOT}/csv/X_train.csv"
    TRAIN_Y_CSV = f"{DATA_ROOT}/csv/y_train.csv"
    VAL_CSV = f"{DATA_ROOT}/csv/X_val.csv"
    VAL_Y_CSV = f"{DATA_ROOT}/csv/y_val.csv"
    DICOM_ROOT = f"{DATA_ROOT}/train"

    # Vérification
    for path in [TRAIN_CSV, TRAIN_Y_CSV, VAL_CSV, VAL_Y_CSV]:
        if not Path(path).exists():
            print(f"❌ Fichier manquant: {path}")
            return

    # Entraînement
    trainer = MpsTrainer(cfg)
    trainer.fit(TRAIN_CSV, VAL_CSV, TRAIN_Y_CSV, VAL_Y_CSV, DICOM_ROOT)


if __name__ == "__main__":
    main()
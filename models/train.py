# train_original.py
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts

import pandas as pd
import numpy as np
import pydicom
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

warnings.filterwarnings('ignore')


@dataclass
class OptimizedConfig:
    # Training
    epochs: int = 15
    batch_size: int = 4  # Batch réel
    grad_accum_steps: int = 4  # Simule batch_size=16
    lr: float = 2e-4
    weight_decay: float = 1e-4

    # Image sizes
    high_res: Tuple[int, int] = (512, 512)
    low_res: Tuple[int, int] = (224, 224)

    # Device
    device: str = "mps"
    num_workers: int = 4
    pin_memory: bool = False

    # Model
    embed_dim: int = 256
    use_checkpoint: bool = True

    # Training strategy
    freeze_epochs: int = 3  # Epochs avec backbones gelés
    warmup_epochs: int = 2

    # Loss
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0

    # Paths
    save_dir: str = "checkpoints"


class MedicalAugmentation:
    """Augmentations spécifiques pour mammographies"""

    def __init__(self, img_size: Tuple[int, int], is_train: bool = True):
        if is_train:
            self.transform = A.Compose([
                A.Resize(*img_size, interpolation=1),
                # Transformations géométriques douces
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.1,
                    rotate_limit=10,
                    border_mode=0,
                    p=0.5
                ),
                # Augmentations d'intensité
                A.OneOf([
                    A.RandomBrightnessContrast(
                        brightness_limit=0.15,
                        contrast_limit=0.15,
                        p=1.0
                    ),
                    A.RandomGamma(gamma_limit=(80, 120), p=1.0),
                    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
                ], p=0.7),
                # Bruit léger
                A.GaussNoise(var_limit=(5.0, 15.0), p=0.3),
                # Normalisation
                A.Normalize(mean=[0.5], std=[0.5]),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(*img_size, interpolation=1),
                A.Normalize(mean=[0.5], std=[0.5]),
                ToTensorV2()
            ])

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        return self.transform(image=image)['image']


class OptimizedDicomDataset(Dataset):
    """Dataset optimisé avec cache et augmentations"""

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
        self.transform = MedicalAugmentation(
            img_size=self.config.high_res,
            is_train=is_train
        )
        self.transform_low = MedicalAugmentation(
            img_size=self.config.low_res,
            is_train=False  # Pas d'augmentation pour low-res
        )

        # Cache pour accélérer (optionnel si RAM suffisante)
        self._cache = {}

    def _load_dicom(self, idx: int) -> np.ndarray:
        """Charge et normalise image DICOM"""
        if idx in self._cache:
            return self._cache[idx]

        row = self.df.iloc[idx]
        path = self.dicom_root / f"{row['patient_id']}_{row['image_id']}.dcm"

        try:
            ds = pydicom.dcmread(path)
            img = ds.pixel_array.astype(np.float32)

            # Normalisation robuste
            p1, p99 = np.percentile(img, (1, 99))
            img = np.clip(img, p1, p99)
            img = (img - p1) / (p99 - p1 + 1e-8)

            return img
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return np.zeros((2048, 1664), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        img = self._load_dicom(idx)

        # Augmentation haute résolution
        img_high = self.transform(img)

        # Version basse résolution (pour contexte)
        img_low = self.transform_low(img)

        if self.is_train:
            label = torch.tensor(
                self.labels.iloc[idx]['cancer'],
                dtype=torch.float32
            )
            return img_high, img_low, label

        return img_high, img_low


class FocalLoss(nn.Module):
    """Focal Loss pour déséquilibre de classes"""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy(preds, targets, reduction='none')
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


class EMA:
    """Exponential Moving Average pour stabilité"""

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


class Trainer:
    def __init__(self, config: OptimizedConfig):
        self.cfg = config
        self.device = self._setup_device()

        # Model
        self.model = OptimizedMultiExpertModel(
            embed_dim=config.embed_dim,
            use_checkpoint=config.use_checkpoint
        ).to(self.device)

        # Freeze backbones initialement
        self.model.freeze_backbones()

        # EMA
        self.ema = EMA(self.model, decay=0.999)

        # Loss
        self.criterion = FocalLoss(
            alpha=config.focal_alpha,
            gamma=config.focal_gamma
        )

        # Métriques
        self.best_loss = float('inf')

        # Créer répertoire checkpoints
        Path(config.save_dir).mkdir(exist_ok=True)

    def _setup_device(self) -> torch.device:
        """Configure MPS avec optimisations"""
        if not torch.backends.mps.is_available():
            print("⚠️  MPS non disponible, utilisation CPU")
            return torch.device("cpu")

        device = torch.device("mps")
        torch.mps.set_per_process_memory_fraction(0.8)
        print("✅ MPS configuré avec 80% mémoire")
        return device

    def train_epoch(
            self,
            loader: DataLoader,
            optimizer: optim.Optimizer,
            epoch: int
    ) -> float:
        """Epoch d'entraînement avec gradient accumulation"""
        self.model.train()
        total_loss = 0
        optimizer.zero_grad()

        pbar = tqdm(loader, desc=f"Epoch {epoch}")

        for batch_idx, (img_high, img_low, labels) in enumerate(pbar):
            img_high = img_high.to(self.device)
            img_low = img_low.to(self.device)
            labels = labels.to(self.device)

            # Forward
            preds, gates, embeddings = self.model(img_high, img_low)

            # Loss
            loss = self.criterion(preds.squeeze(), labels)
            loss = loss / self.cfg.grad_accum_steps

            # Backward
            loss.backward()

            # Update chaque grad_accum_steps
            if (batch_idx + 1) % self.cfg.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

                # EMA update
                self.ema.update()

            total_loss += loss.item() * self.cfg.grad_accum_steps

            # Monitoring
            if batch_idx % 20 == 0:
                mem = torch.mps.current_allocated_memory() / 1024 ** 3 \
                    if self.device.type == 'mps' else 0
                pbar.set_postfix({
                    'loss': f"{loss.item() * self.cfg.grad_accum_steps:.4f}",
                    'mem_gb': f"{mem:.2f}"
                })

            # Nettoyage MPS
            if self.device.type == 'mps' and batch_idx % 50 == 0:
                torch.mps.empty_cache()

        return total_loss / len(loader)

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> float:
        """Validation avec EMA"""
        self.ema.apply_shadow()
        self.model.eval()

        total_loss = 0

        for img_high, img_low, labels in tqdm(loader, desc="Validation"):
            img_high = img_high.to(self.device)
            img_low = img_low.to(self.device)
            labels = labels.to(self.device)

            preds, _, _ = self.model(img_high, img_low)
            loss = self.criterion(preds.squeeze(), labels)
            total_loss += loss.item()

        self.ema.restore()
        return total_loss / len(loader)

    def fit(
            self,
            train_csv: str,
            val_csv: str,
            train_y_csv: str,
            val_y_csv: str,
            dicom_root: str
    ):
        """Pipeline d'entraînement complet"""
        # Datasets
        train_ds = OptimizedDicomDataset(
            train_csv, dicom_root, train_y_csv,
            self.cfg, is_train=True
        )
        val_ds = OptimizedDicomDataset(
            val_csv, dicom_root, val_y_csv,
            self.cfg, is_train=True
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            persistent_workers=True
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.cfg.batch_size * 2,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory
        )

        # Optimizer
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
            betas=(0.9, 0.999)
        )

        # Scheduler
        steps_per_epoch = len(train_loader) // self.cfg.grad_accum_steps
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.cfg.lr,
            epochs=self.cfg.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.3,
            anneal_strategy='cos'
        )

        # Training loop
        for epoch in range(self.cfg.epochs):
            # Dégeler backbones après freeze_epochs
            if epoch == self.cfg.freeze_epochs:
                print("\n🔓 Unfreezing backbones...")
                self.model.unfreeze_backbones()
                # Réduire LR
                for g in optimizer.param_groups:
                    g['lr'] *= 0.1

            train_loss = self.train_epoch(train_loader, optimizer, epoch)
            val_loss = self.validate(val_loader)

            scheduler.step()

            print(f"\nEpoch {epoch}: train_loss={train_loss:.4f}, "
                  f"val_loss={val_loss:.4f}")

            # Save best
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save_checkpoint(
                    f"{self.cfg.save_dir}/best_model.pth",
                    epoch, val_loss
                )
                print(f"✅ Best model saved (val_loss={val_loss:.4f})")

        # Save final avec EMA
        self.ema.apply_shadow()
        self.save_checkpoint(
            f"{self.cfg.save_dir}/final_ema.pth",
            self.cfg.epochs, self.best_loss
        )

    def save_checkpoint(self, path: str, epoch: int, val_loss: float):
        """Sauvegarde checkpoint"""
        torch.save({
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'ema_shadow': self.ema.shadow,
            'val_loss': val_loss,
            'config': self.cfg
        }, path)

    def load_checkpoint(self, path: str, use_ema: bool = True):
        """Charge checkpoint"""
        ckpt = torch.load(path, map_location=self.device)

        if use_ema and 'ema_shadow' in ckpt:
            # Charger EMA weights
            for name, param in self.model.named_parameters():
                if name in ckpt['ema_shadow']:
                    param.data = ckpt['ema_shadow'][name]
        else:
            self.model.load_state_dict(ckpt['model_state'])

        print(f"✅ Checkpoint loaded from epoch {ckpt['epoch']}")


@torch.no_grad()
def predict(
        model: OptimizedMultiExpertModel,
        test_csv: str,
        dicom_root: str,
        config: OptimizedConfig,
        device: torch.device,
        tta: bool = False
) -> np.ndarray:
    """Prédiction avec TTA optionnel"""
    model.eval()

    test_ds = OptimizedDicomDataset(
        test_csv, dicom_root,
        y_csv=None, config=config, is_train=False
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size * 2,
        shuffle=False,
        num_workers=config.num_workers
    )

    predictions = []

    for img_high, img_low in tqdm(test_loader, desc="Inference"):
        img_high = img_high.to(device)
        img_low = img_low.to(device)

        if tta:
            # Test-Time Augmentation
            preds_tta = []

            # Original
            pred, _, _ = model(img_high, img_low)
            preds_tta.append(pred)

            # Flip horizontal
            pred_flip, _, _ = model(
                torch.flip(img_high, [-1]),
                torch.flip(img_low, [-1])
            )
            preds_tta.append(pred_flip)

            # Average TTA
            pred = torch.stack(preds_tta).mean(dim=0)
        else:
            pred, _, _ = model(img_high, img_low)

        predictions.extend(pred.squeeze().cpu().numpy())

    return np.array(predictions)


def create_submission(
        predictions: np.ndarray,
        test_csv: str,
        output_path: str = "submission.csv"
):
    """Crée fichier submission Kaggle"""
    df = pd.read_csv(test_csv)

    submission = pd.DataFrame({
        'prediction_id': df['patient_id'].astype(str) + '_' + df['image_id'].astype(str),
        'cancer': predictions
    })

    submission.to_csv(output_path, index=False)
    print(f"✅ Submission saved to {output_path}")


# ==========================================================
# MAIN EXECUTION
# ==========================================================
if __name__ == "__main__":
    # Configuration
    cfg = OptimizedConfig(
        epochs=15,
        batch_size=4,
        grad_accum_steps=4,  # Effective batch size = 16
        lr=2e-4,
        high_res=(512, 512),
        low_res=(224, 224),
        embed_dim=256,
        use_checkpoint=True,
        freeze_epochs=3,
        save_dir="checkpoints"
    )

    # Paths (à adapter)
    DATA_ROOT = "/Users/assadiabira/Bureau/Kaggle/Projet_kaggle/data"
    TRAIN_CSV = f"{DATA_ROOT}/csv/X_train.csv"
    TRAIN_Y_CSV = f"{DATA_ROOT}/csv/y_train.csv"
    VAL_CSV = f"{DATA_ROOT}/csv/X_val.csv"  # À créer si besoin
    VAL_Y_CSV = f"{DATA_ROOT}/csv/y_val.csv"
    DICOM_ROOT = f"{DATA_ROOT}/train"

    # Entraînement
    print("=" * 60)
    print("🚀 TRAINING MULTI-EXPERT MODEL")
    print("=" * 60)

    trainer = Trainer(cfg)

    # Si pas de validation split, créer un split
    if not Path(VAL_CSV).exists():
        print("⚠️  Creating train/val split...")
        df_train = pd.read_csv(TRAIN_CSV)
        df_y = pd.read_csv(TRAIN_Y_CSV)

        from sklearn.model_selection import train_test_split

        train_idx, val_idx = train_test_split(
            range(len(df_train)),
            test_size=0.15,
            random_state=42,
            stratify=df_y['cancer']
        )

        df_train.iloc[train_idx].to_csv(TRAIN_CSV.replace('.csv', '_split.csv'), index=False)
        df_train.iloc[val_idx].to_csv(VAL_CSV, index=False)
        df_y.iloc[train_idx].to_csv(TRAIN_Y_CSV.replace('.csv', '_split.csv'), index=False)
        df_y.iloc[val_idx].to_csv(VAL_Y_CSV, index=False)

        TRAIN_CSV = TRAIN_CSV.replace('.csv', '_split.csv')
        TRAIN_Y_CSV = TRAIN_Y_CSV.replace('.csv', '_split.csv')

    # Launch training
    trainer.fit(
        train_csv=TRAIN_CSV,
        val_csv=VAL_CSV,
        train_y_csv=TRAIN_Y_CSV,
        val_y_csv=VAL_Y_CSV,
        dicom_root=DICOM_ROOT
    )

    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE")
    print("=" * 60)

    # Optionnel : Prédiction sur test set
    """
    TEST_CSV = f"{DATA_ROOT}/csv/X_test.csv"
    TEST_ROOT = f"{DATA_ROOT}/test"

    trainer.load_checkpoint(f"{cfg.save_dir}/final_ema.pth", use_ema=True)

    predictions = predict(
        model=trainer.model,
        test_csv=TEST_CSV,
        dicom_root=TEST_ROOT,
        config=cfg,
        device=trainer.device,
        tta=True  # Active TTA pour meilleure performance
    )

    create_submission(predictions, TEST_CSV, "submission_final.csv")
    """
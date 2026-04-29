import contextlib
import logging
import os
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

from models.losses import FocalAUCLoss

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Mixed-precision helpers (CUDA fp16 / MPS bf16 / CPU no-op)
# ─────────────────────────────────────────────────────────────

def get_autocast_context(device: torch.device, enabled: bool = True):
    """Context manager mixed-precision adapté au device."""
    if not enabled:
        return contextlib.nullcontext()
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    if device.type == "mps":
        # MPS supporte bfloat16 depuis PyTorch 2.0
        return torch.autocast(device_type="mps", dtype=torch.bfloat16)
    # CPU — pas de mixed precision
    return contextlib.nullcontext()


def get_grad_scaler(device: torch.device, enabled: bool = True) -> torch.cuda.amp.GradScaler:
    """GradScaler uniquement sur CUDA (MPS/CPU non supportés)."""
    cuda_amp = (device.type == "cuda") and enabled
    return torch.cuda.amp.GradScaler(enabled=cuda_amp)


# ─────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────

class Trainer:
    """
    Boucle d'entraînement OOP pour MultiHeadMammoModel.

    Features :
    - Mixed precision adaptée au device (CUDA fp16 / MPS bf16 / CPU no-op)
    - GradScaler conditionnel CUDA uniquement
    - Gradient clipping (max_norm=1.0)
    - Cosine Annealing LR
    - Early stopping sur AUROC val
    - Patient-level aggregation pour l'évaluation
    - Sauvegarde du meilleur checkpoint
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        n_epochs: int = 20,
        patience: int = 5,
        checkpoint_dir: str = "checkpoints",
        pos_weight: float = 10.0,
        use_amp: bool = True,
    ):
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device
        self.n_epochs     = n_epochs
        self.patience     = patience
        self.checkpoint_dir = checkpoint_dir
        self.use_amp      = use_amp

        os.makedirs(checkpoint_dir, exist_ok=True)

        self.criterion = FocalAUCLoss(
            focal_weight=0.7,
            auc_weight=0.3,
            alpha=0.75,
            gamma=2.5,
            pos_weight=min(pos_weight, 10.0),   # plafonné à 10
        )

        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )

        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=n_epochs,
            eta_min=1e-6,
        )

        self.scaler = get_grad_scaler(device, enabled=use_amp)

        self.best_auroc        = 0.0
        self.epochs_no_improve = 0
        self.history: Dict[str, list] = {
            "train_loss": [], "val_loss": [], "val_auroc": [], "val_f1": []
        }

    def train(self) -> Dict[str, list]:
        """Lance l'entraînement complet. Retourne l'historique des métriques."""
        logger.info(
            f"Début entraînement — {self.n_epochs} epochs  device={self.device}  "
            f"amp={self.use_amp}  scaler={'cuda' if self.device.type == 'cuda' else 'off'}"
        )

        for epoch in range(1, self.n_epochs + 1):
            train_loss  = self._train_epoch(epoch)
            val_metrics = self._eval_epoch()

            self.scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_auroc"].append(val_metrics["auroc"])
            self.history["val_f1"].append(val_metrics["f1"])

            logger.info(
                f"Epoch {epoch:02d}/{self.n_epochs} | "
                f"train={train_loss:.4f} | val={val_metrics['loss']:.4f} | "
                f"AUROC={val_metrics['auroc']:.4f} | F1={val_metrics['f1']:.4f} | "
                f"Recall={val_metrics['recall']:.4f} | "
                f"LR={self.scheduler.get_last_lr()[0]:.2e}"
            )

            if val_metrics["auroc"] > self.best_auroc:
                self.best_auroc        = val_metrics["auroc"]
                self.epochs_no_improve = 0
                self._save_checkpoint(epoch, val_metrics["auroc"])
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= self.patience:
                    logger.info(
                        f"Early stopping epoch {epoch} (best AUROC={self.best_auroc:.4f})"
                    )
                    break

        return self.history

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0

        for batch_idx, (images, labels, _pids) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            with get_autocast_context(self.device, self.use_amp):
                logits, _gates = self.model(images)
                loss_dict = self.criterion(logits.squeeze(1), labels)
                loss = loss_dict["total"]

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

            if (batch_idx + 1) % 50 == 0:
                logger.debug(
                    f"  Epoch {epoch} [{batch_idx+1}/{len(self.train_loader)}] "
                    f"loss={loss.item():.4f} "
                    f"(focal={loss_dict['focal'].item():.4f}, auc={loss_dict['auc'].item():.4f})"
                )

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def _eval_epoch(self) -> Dict[str, float]:
        self.model.eval()
        all_logits, all_labels, all_pids = [], [], []
        total_loss = 0.0

        for images, labels, patient_ids in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with get_autocast_context(self.device, self.use_amp):
                logits, _ = self.model(images)
                loss_dict = self.criterion(logits.squeeze(1), labels)

            total_loss += loss_dict["total"].item()
            all_logits.append(logits.squeeze(1).cpu())
            all_labels.append(labels.cpu())
            all_pids.append(patient_ids)

        all_logits = torch.cat(all_logits).numpy()
        all_labels = torch.cat(all_labels).numpy()
        all_pids   = (torch.cat(all_pids).numpy()
                      if isinstance(all_pids[0], torch.Tensor)
                      else np.concatenate(all_pids))

        probs = 1 / (1 + np.exp(-all_logits))   # sigmoid

        from models.dataset import MammographyDataset
        patient_probs, unique_pids = MammographyDataset.patient_level_aggregate(
            probs, all_pids, method="max"
        )
        patient_labels = np.array([
            all_labels[all_pids == pid].max() for pid in unique_pids
        ])

        best_thresh, _ = _find_best_threshold(patient_probs, patient_labels)
        patient_preds  = (patient_probs >= best_thresh).astype(int)

        auroc = roc_auc_score(patient_labels, patient_probs) if patient_labels.sum() > 0 else 0.0

        return {
            "loss":      total_loss / len(self.val_loader),
            "auroc":     auroc,
            "f1":        f1_score(patient_labels, patient_preds, zero_division=0),
            "recall":    recall_score(patient_labels, patient_preds, zero_division=0),
            "precision": precision_score(patient_labels, patient_preds, zero_division=0),
            "threshold": best_thresh,
        }

    def _save_checkpoint(self, epoch: int, auroc: float) -> None:
        path = os.path.join(
            self.checkpoint_dir, f"best_model_auroc{auroc:.4f}_ep{epoch}.pt"
        )
        torch.save({
            "epoch":              epoch,
            "model_state_dict":   self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "auroc":              auroc,
        }, path)
        logger.info(f"Checkpoint sauvegardé : {path}")

    def load_best_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Checkpoint chargé : {path} (AUROC={ckpt.get('auroc', '?'):.4f})")


def _find_best_threshold(probs: np.ndarray, labels: np.ndarray) -> tuple:
    """Retourne le seuil maximisant le F1 sur une grille [0.05, 0.95]."""
    best_f1, best_thresh = 0.0, 0.5
    for t in np.linspace(0.05, 0.95, 91):
        preds = (probs >= t).astype(int)
        f1    = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t
    return best_thresh, best_f1

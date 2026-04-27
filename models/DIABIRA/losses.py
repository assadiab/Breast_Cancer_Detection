from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss pour dataset fortement déséquilibré (7% positifs).
    alpha=0.75 favorise la classe positive (cancer rare).
    gamma=2.5 concentre l'entraînement sur les exemples difficiles.
    """

    def __init__(
        self,
        alpha: float = 0.75,
        gamma: float = 2.5,
        pos_weight: Optional[float] = None,
        label_smoothing: float = 0.05,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: raw logits (B,) ou (B, 1) — pas de sigmoid appliqué
            targets: labels binaires float (B,) ou (B, 1)
        """
        logits = logits.view(-1)
        targets = targets.view(-1).float()

        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce)
        focal_weight = (1 - pt) ** self.gamma
        alpha_weight = targets * self.alpha + (1 - targets) * (1 - self.alpha)

        if self.pos_weight is not None:
            class_weight = targets * self.pos_weight + (1 - targets)
            loss = alpha_weight * class_weight * focal_weight * bce
        else:
            loss = alpha_weight * focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class AUCLoss(nn.Module):
    """
    Approximation différentiable de l'AUC par pairwise ranking loss.
    Maximise la séparation entre scores positifs et négatifs.
    """

    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds = torch.sigmoid(logits.view(-1))
        targets = targets.view(-1).float()

        pos_mask = targets > 0.5
        neg_mask = ~pos_mask

        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        pos_preds = preds[pos_mask]
        neg_preds = preds[neg_mask]

        # Smooth hinge loss pairwise
        diff = pos_preds.unsqueeze(1) - neg_preds.unsqueeze(0)  # (n_pos, n_neg)
        return torch.clamp(self.margin - diff, min=0).pow(2).mean()


class FocalAUCLoss(nn.Module):
    """
    Loss combinée : 70% Focal + 30% AUC.
    Optimale pour RSNA : AUROC est la métrique principale, Focal gère le déséquilibre.
    """

    def __init__(
        self,
        focal_weight: float = 0.7,
        auc_weight: float = 0.3,
        alpha: float = 0.75,
        gamma: float = 2.5,
        margin: float = 0.5,
        pos_weight: Optional[float] = None,
    ):
        super().__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma, pos_weight=pos_weight)
        self.auc = AUCLoss(margin=margin)
        self.focal_weight = focal_weight
        self.auc_weight = auc_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        focal_val = self.focal(logits, targets)
        auc_val = self.auc(logits, targets)
        total = self.focal_weight * focal_val + self.auc_weight * auc_val
        return {"total": total, "focal": focal_val, "auc": auc_val}

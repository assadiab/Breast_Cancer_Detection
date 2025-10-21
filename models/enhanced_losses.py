# enhanced_losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedFocalLoss(nn.Module):
    """Focal Loss avec pondération pour déséquilibre sévère"""

    def __init__(self, alpha=0.25, gamma=2.0, class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights

    def forward(self, preds, targets):
        bce = F.binary_cross_entropy(preds, targets, reduction='none')
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce

        if self.class_weights is not None:
            weights = torch.where(targets == 1,
                                  self.class_weights[1],
                                  self.class_weights[0])
            focal = focal * weights

        return focal.mean()


class AUCMLoss(nn.Module):
    """Loss optimisée pour l'AUC - très efficace pour datasets déséquilibrés"""

    def __init__(self, margin=1.0, imratio=0.0732):  # 7.32% de cancer dans votre dataset
        super().__init__()
        self.margin = margin
        self.imratio = imratio
        self.a = torch.zeros(1, requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)

    def forward(self, preds, targets):
        positive = preds[targets == 1]
        negative = preds[targets == 0]

        diff = positive.unsqueeze(1) - negative.unsqueeze(0)
        loss = (1 - diff) ** 2 + self.margin * F.relu(1 - diff)

        return loss.mean()
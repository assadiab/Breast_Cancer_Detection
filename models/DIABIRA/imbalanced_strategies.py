# imbalanced_strategies.py
"""Stratégies avancées pour gérer le fort déséquilibre (7% positifs)"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, List
from torch.utils.data import Sampler, WeightedRandomSampler, DataLoader
from collections import Counter, defaultdict
import warnings

warnings.filterwarnings('ignore')


# ==========================================================
# WEIGHTED SAMPLER AVEC STRATIFICATION PAR PATIENT
# ==========================================================
class BalancedPatientSampler(Sampler):
    """
    Sampler qui balance les classes tout en respectant les patients.
    Important : évite data leakage en gardant toutes les images d'un patient ensemble.
    """

    def __init__(
            self,
            patient_ids: np.ndarray,
            labels: np.ndarray,
            samples_per_epoch: Optional[int] = None,
            pos_weight: float = 10.0
    ):
        self.patient_ids = patient_ids
        self.labels = labels
        self.pos_weight = pos_weight

        # Grouper indices par patient
        self.patient_to_indices = {}
        for idx, pid in enumerate(patient_ids):
            if pid not in self.patient_to_indices:
                self.patient_to_indices[pid] = []
            self.patient_to_indices[pid].append(idx)

        # Séparer patients positifs/négatifs
        self.pos_patients = []
        self.neg_patients = []

        for pid, indices in self.patient_to_indices.items():
            patient_labels = labels[indices]
            if patient_labels.max() > 0:  # Si au moins 1 image cancer
                self.pos_patients.append(pid)
            else:
                self.neg_patients.append(pid)

        self.n_pos = len(self.pos_patients)
        self.n_neg = len(self.neg_patients)

        # Nombre de samples par epoch
        if samples_per_epoch is None:
            # Par défaut : équilibrer en sursampling positifs
            self.samples_per_epoch = int(self.n_neg + self.n_pos * pos_weight)
        else:
            self.samples_per_epoch = samples_per_epoch

        print(f"📊 Sampler Stats:")
        print(f"  - Positive patients: {self.n_pos}")
        print(f"  - Negative patients: {self.n_neg}")
        print(f"  - Samples per epoch: {self.samples_per_epoch}")
        print(f"  - Pos weight: {pos_weight}")

    def __iter__(self):
        # Calculer nombre de patients à sampler de chaque classe
        n_pos_sample = int(self.samples_per_epoch * self.pos_weight / (1 + self.pos_weight))
        n_neg_sample = self.samples_per_epoch - n_pos_sample

        # Sample patients avec replacement pour positifs (oversample)
        sampled_pos = np.random.choice(
            self.pos_patients,
            size=min(n_pos_sample, len(self.pos_patients) * 3),
            replace=True
        )
        sampled_neg = np.random.choice(
            self.neg_patients,
            size=min(n_neg_sample, len(self.neg_patients)),
            replace=n_neg_sample > len(self.neg_patients)
        )

        # Récupérer tous les indices des patients samplés
        indices = []
        for pid in np.concatenate([sampled_pos, sampled_neg]):
            indices.extend(self.patient_to_indices[pid])

        # Shuffle final
        np.random.shuffle(indices)
        return iter(indices[:self.samples_per_epoch])  # Truncate au besoin

    def __len__(self):
        return self.samples_per_epoch


# ==========================================================
# FOCAL LOSS AMÉLIORÉ AVEC CLASS BALANCING
# ==========================================================
class AdaptiveFocalLoss(nn.Module):
    """
    Focal Loss avec poids de classe adaptatifs.
    alpha contrôle l'importance de la classe positive (rare).
    gamma contrôle le focus sur hard examples.
    """

    def __init__(
            self,
            alpha: float = 0.75,  # Plus élevé pour classe rare (cancer)
            gamma: float = 2.5,  # Plus élevé pour focus sur hard cases
            pos_weight: Optional[float] = None,
            label_smoothing: float = 0.05,
            reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Label smoothing si activé
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        # BCE de base
        bce = F.binary_cross_entropy(preds, targets, reduction='none')

        # Focal term
        pt = torch.exp(-bce)
        focal_weight = (1 - pt) ** self.gamma

        # Alpha weighting pour classe positive
        alpha_weight = targets * self.alpha + (1 - targets) * (1 - self.alpha)

        # Pos weight additionnel si fourni
        if self.pos_weight is not None:
            class_weight = targets * self.pos_weight + (1 - targets) * 1.0
            focal_loss = alpha_weight * class_weight * focal_weight * bce
        else:
            focal_loss = alpha_weight * focal_weight * bce

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# ==========================================================
# LOSS COMBINÉE : FOCAL + AUC OPTIMIZATION
# ==========================================================
class CombinedLoss(nn.Module):
    """
    Combine Focal Loss avec AUC maximization loss.
    Pour dataset déséquilibré, AUC est meilleure métrique que accuracy.
    """

    def __init__(
            self,
            focal_weight: float = 0.7,
            auc_weight: float = 0.3,
            alpha: float = 0.75,
            gamma: float = 2.5,
            margin: float = 0.5
    ):
        super().__init__()
        self.focal = AdaptiveFocalLoss(alpha=alpha, gamma=gamma)
        self.focal_weight = focal_weight
        self.auc_weight = auc_weight
        self.margin = margin

    def auc_loss(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Approximation différentiable de AUC loss.
        Favorise séparation entre positifs et négatifs.
        """
        pos_mask = targets == 1
        neg_mask = targets == 0

        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            return torch.tensor(0.0, device=preds.device)

        pos_preds = preds[pos_mask]
        neg_preds = preds[neg_mask]

        # Pairwise ranking loss
        # On veut que tous les positifs soient > tous les négatifs
        diff = pos_preds.unsqueeze(1) - neg_preds.unsqueeze(0)  # (n_pos, n_neg)

        # Smooth hinge loss
        loss = torch.clamp(self.margin - diff, min=0).pow(2).mean()

        return loss

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        focal_loss = self.focal(preds, targets)
        auc_loss_val = self.auc_loss(preds, targets)

        total_loss = self.focal_weight * focal_loss + self.auc_weight * auc_loss_val

        return {
            'total': total_loss,
            'focal': focal_loss,
            'auc': auc_loss_val
        }


# ==========================================================
# MIXUP AUGMENTATION POUR BALANCER
# ==========================================================
class MixupAugmentation:
    """
    Mixup spécialement conçu pour images médicales déséquilibrées.
    Mélange préférentiellement positifs avec négatifs.
    """

    def __init__(
            self,
            alpha: float = 0.4,
            prob: float = 0.5,
            pos_mixup_prob: float = 0.8  # Prob de mixer avec un positif
    ):
        self.alpha = alpha
        self.prob = prob
        self.pos_mixup_prob = pos_mixup_prob

    def __call__(
            self,
            images: torch.Tensor,
            labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if np.random.rand() > self.prob or len(images) < 2:
            return images, labels

        batch_size = images.size(0)
        lam = np.random.beta(self.alpha, self.alpha)

        # Identifier indices positifs et négatifs
        pos_indices = (labels == 1).nonzero(as_tuple=True)[0]
        neg_indices = (labels == 0).nonzero(as_tuple=True)[0]

        # Stratégie de mixup intelligente
        mixed_images = images.clone()
        mixed_labels = labels.clone()

        for i in range(batch_size):
            # Si image courante est négative, mixer avec positive si possible
            if i in neg_indices and len(pos_indices) > 0 and np.random.rand() < self.pos_mixup_prob:
                j = pos_indices[np.random.randint(len(pos_indices))]
            else:
                # Sinon mixer aléatoirement
                j = np.random.randint(batch_size)
                while j == i:  # Éviter de mixer avec soi-même
                    j = np.random.randint(batch_size)

            mixed_images[i] = lam * images[i] + (1 - lam) * images[j]
            mixed_labels[i] = lam * labels[i] + (1 - lam) * labels[j]

        return mixed_images, mixed_labels


# ==========================================================
# HARD NEGATIVE MINING
# ==========================================================
class HardNegativeMiner:
    """
    Sélectionne les négatifs les plus difficiles (faux positifs) pour entraînement.
    Réduit le déséquilibre en se concentrant sur exemples informatifs.
    """

    def __init__(
            self,
            ratio: float = 3.0,  # Ratio négatifs:positifs
            update_freq: int = 100,  # Update scores tous les N batches
            memory_size: int = 10000
    ):
        self.ratio = ratio
        self.update_freq = update_freq
        self.memory_size = memory_size
        self.step = 0
        self.hard_scores = {}  # idx -> difficulty score

    def update_scores(
            self,
            indices: torch.Tensor,
            preds: torch.Tensor,
            labels: torch.Tensor
    ):
        """Met à jour les scores de difficulté"""
        self.step += 1

        if self.step % self.update_freq != 0:
            return

        # Score = probabilité prédite pour négatifs (plus haut = plus dur)
        for idx, pred, label in zip(indices.cpu().numpy(),
                                    preds.detach().cpu().numpy(),
                                    labels.cpu().numpy()):
            if label == 0:  # Négatif
                self.hard_scores[idx] = float(pred)

        # Garder seulement les plus récents si mémoire pleine
        if len(self.hard_scores) > self.memory_size:
            # Garder les scores les plus élevés (les plus durs)
            sorted_items = sorted(self.hard_scores.items(), key=lambda x: x[1], reverse=True)
            self.hard_scores = dict(sorted_items[:self.memory_size])

    def get_hard_negatives(
            self,
            neg_indices: np.ndarray,
            n_positives: int
    ) -> np.ndarray:
        """Retourne les négatifs les plus durs"""
        n_select = min(int(n_positives * self.ratio), len(neg_indices))

        if len(self.hard_scores) < len(neg_indices) // 2:
            # Pas assez de scores : sélection random
            return np.random.choice(neg_indices, n_select, replace=False)

        # Trier négatifs par difficulté
        scored = [(idx, self.hard_scores.get(idx, 0.3))
                  for idx in neg_indices]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Prendre les plus durs + quelques random
        hard_ratio = 0.7
        n_hard = int(n_select * hard_ratio)
        n_random = n_select - n_hard

        hard_indices = [idx for idx, _ in scored[:n_hard]]

        if n_random > 0:
            remaining = [idx for idx, _ in scored[n_hard:]]
            if len(remaining) >= n_random:
                random_sel = np.random.choice(remaining, n_random, replace=False).tolist()
            else:
                random_sel = remaining
            hard_indices.extend(random_sel)

        return np.array(hard_indices)


# ==========================================================
# STRATEGY MANAGER - COORDONNE TOUTES LES STRATÉGIES
# ==========================================================
class ImbalanceStrategyManager:
    """
    Manager principal qui coordonne toutes les stratégies de déséquilibre.
    """

    def __init__(
            self,
            cancer_rate: float = 0.0732,
            use_mixup: bool = True,
            use_hard_mining: bool = True,
            device: str = "mps"
    ):
        self.cancer_rate = cancer_rate
        self.device = torch.device(device)

        # Calcul poids automatique basé sur déséquilibre
        self.pos_weight = 1.0 / cancer_rate  # ~13.6 pour 7.32%

        # Initialiser stratégies
        self.loss_fn = CombinedLoss(
            focal_weight=0.7,
            auc_weight=0.3,
            alpha=0.75,
            gamma=2.5
        )

        self.mixup = MixupAugmentation(
            alpha=0.4,
            prob=0.5,
            pos_mixup_prob=0.8
        ) if use_mixup else None

        self.hard_miner = HardNegativeMiner(
            ratio=3.0,
            update_freq=100
        ) if use_hard_mining else None

        self.threshold_optimizer = ThresholdOptimizer(metric='f1')

        print(f"🎯 Imbalance Strategy Manager Initialized:")
        print(f"   - Cancer rate: {cancer_rate:.3f}")
        print(f"   - Pos weight: {self.pos_weight:.1f}")
        print(f"   - Use Mixup: {use_mixup}")
        print(f"   - Use Hard Mining: {use_hard_mining}")

    def create_balanced_loader(self, dataset, batch_size: int, num_workers: int = 4):
        """Crée DataLoader avec sampling balancé"""
        # Extraire labels et patient_ids
        labels = np.array([dataset.labels_df.iloc[i]['cancer'] for i in range(len(dataset))])
        patient_ids = np.array([dataset.df.iloc[i]['patient_id'] for i in range(len(dataset))])

        # Créer sampler
        sampler = BalancedPatientSampler(
            patient_ids=patient_ids,
            labels=labels,
            pos_weight=self.pos_weight
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=False,
            persistent_workers=True if num_workers > 0 else False,
            drop_last=True
        )

        return loader

    def apply_mixup(self, images: torch.Tensor, labels: torch.Tensor):
        """Applique mixup si activé"""
        if self.mixup is not None:
            return self.mixup(images, labels)
        return images, labels

    def update_hard_mining(self, indices: torch.Tensor, preds: torch.Tensor, labels: torch.Tensor):
        """Met à jour le hard negative miner"""
        if self.hard_miner is not None:
            self.hard_miner.update_scores(indices, preds, labels)

    def compute_loss(self, preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Calcule la loss combinée"""
        return self.loss_fn(preds, targets)


# ==========================================================
# THRESHOLD OPTIMIZATION POST-TRAINING
# ==========================================================
class ThresholdOptimizer:
    """
    Optimise threshold de classification pour maximiser F1 ou autre métrique.
    Crucial pour dataset déséquilibré où 0.5 n'est pas optimal.
    """

    def __init__(self, metric: str = 'f1'):
        self.metric = metric
        self.best_threshold = 0.5

    def optimize(
            self,
            y_true: np.ndarray,
            y_pred_proba: np.ndarray,
            thresholds: Optional[np.ndarray] = None
    ) -> Tuple[float, Dict]:
        """
        Trouve le meilleur threshold en testant une grille.
        """
        if thresholds is None:
            thresholds = np.linspace(0.05, 0.95, 181)  # Plus de précision

        best_score = -1
        best_thresh = 0.5
        results = []

        for thresh in thresholds:
            y_pred = (y_pred_proba >= thresh).astype(int)

            from sklearn.metrics import f1_score, precision_score, recall_score, balanced_accuracy_score

            if len(np.unique(y_true)) < 2:
                continue

            f1 = f1_score(y_true, y_pred, zero_division=0)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            bal_acc = balanced_accuracy_score(y_true, y_pred)

            if self.metric == 'f1':
                score = f1
            elif self.metric == 'f2':  # Favorise recall
                score = (5 * prec * rec) / (4 * prec + rec + 1e-8)
            elif self.metric == 'balanced_f1':
                score = (f1 + bal_acc) / 2
            else:
                score = f1

            results.append({
                'threshold': thresh,
                'f1': f1,
                'precision': prec,
                'recall': rec,
                'balanced_accuracy': bal_acc,
                'score': score
            })

            if score > best_score:
                best_score = score
                best_thresh = thresh

        self.best_threshold = best_thresh

        print(f"\n🎯 Optimal Threshold: {best_thresh:.3f}")
        print(f"   Best {self.metric.upper()}: {best_score:.4f}")
        print(f"   F1: {results[thresholds.tolist().index(best_thresh)]['f1']:.4f}")
        print(f"   Precision: {results[thresholds.tolist().index(best_thresh)]['precision']:.4f}")
        print(f"   Recall: {results[thresholds.tolist().index(best_thresh)]['recall']:.4f}")

        return best_thresh, results


"""
# ==========================================================
# TEST DE LA STRATÉGIE
# ==========================================================
def test_imbalance_strategies():
    print("🧪 TEST DES STRATÉGIES DE DÉSÉQUILIBRE")
    print("=" * 60)

    # Simuler dataset déséquilibré (7% positifs)
    n_samples = 1000
    n_positives = int(n_samples * 0.0732)

    # Données simulées
    patient_ids = np.array([f"patient_{i}" for i in range(n_samples)])
    labels = np.zeros(n_samples)
    labels[:n_positives] = 1  # Premiers n_positives sont positifs

    print(f"Dataset simulé: {n_samples} samples, {n_positives} positifs ({n_positives / n_samples * 100:.1f}%)")

    # Test BalancedPatientSampler
    sampler = BalancedPatientSampler(
        patient_ids=patient_ids,
        labels=labels,
        pos_weight=10.0
    )

    sampled_indices = list(sampler)
    sampled_labels = labels[sampled_indices]

    print(f"\n📊 Après sampling:")
    print(f"   Positifs: {sampled_labels.sum()} ({sampled_labels.sum() / len(sampled_labels) * 100:.1f}%)")
    print(f"   Négatifs: {len(sampled_labels) - sampled_labels.sum()}")

    # Test Loss functions
    loss_fn = CombinedLoss()
    preds = torch.randn(32, 1).sigmoid()
    targets = torch.cat([torch.ones(8), torch.zeros(24)]).unsqueeze(1)  # 25% positifs

    loss_dict = loss_fn(preds, targets)
    print(f"\n📉 Losses combinées:")
    for k, v in loss_dict.items():
        print(f"   {k}: {v.item():.4f}")

    # Test Mixup
    mixup = MixupAugmentation()
    images = torch.randn(32, 1, 224, 224)
    mixed_images, mixed_labels = mixup(images, targets.squeeze())

    print(f"\n🎭 Mixup appliqué:")
    print(f"   Labels originaux - Pos: {targets.sum().item()}")
    print(f"   Labels mixup - Pos: {mixed_labels.sum().item():.1f}")

    print("\n✅ Toutes les stratégies fonctionnent correctement!")
"""

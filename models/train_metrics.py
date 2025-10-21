# train_metrics.py
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import sys
from datetime import datetime
import warnings
import time
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, \
    balanced_accuracy_score, confusion_matrix

warnings.filterwarnings('ignore')

# Optimisations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Ajouter les chemins
sys.path.append(str(Path(__file__).parent))

# Import depuis le fichier original train.py
from train import OptimizedDicomDataset, OptimizedConfig, EMA, FocalLoss
from models.multi_expert import OptimizedMultiExpertModel
from imbalanced_strategies import ImbalanceStrategyManager, ThresholdOptimizer


class MetricsLogger:
    """Logger complet pour toutes les métriques"""

    def __init__(self):
        self.epoch_metrics = []
        self.best_metrics = {}

    def log_epoch(self, epoch, train_metrics, val_metrics, learning_rate, epoch_time):
        """Log toutes les métriques d'une epoch"""
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
            'val_best_threshold': val_metrics['best_threshold'],
            'learning_rate': learning_rate,
            'epoch_time': epoch_time
        }

        self.epoch_metrics.append(metrics)
        self._update_best_metrics(metrics)
        self._print_epoch_metrics(metrics)

        return metrics

    def _update_best_metrics(self, metrics):
        """Met à jour les meilleures métriques"""
        if not self.best_metrics:
            self.best_metrics = metrics.copy()
            return

        if metrics['val_auc'] > self.best_metrics['val_auc']:
            self.best_metrics.update({k: metrics[k] for k in ['val_auc', 'epoch']})

        if metrics['val_f1'] > self.best_metrics['val_f1']:
            self.best_metrics.update({k: metrics[k] for k in ['val_f1', 'epoch']})

        if metrics['val_ap'] > self.best_metrics['val_ap']:
            self.best_metrics.update({k: metrics[k] for k in ['val_ap', 'epoch']})

    def _print_epoch_metrics(self, metrics):
        """Affiche les métriques de l'epoch de manière claire"""
        print(f"\n{'=' * 80}")
        print(f"📊 EPOCH {metrics['epoch']:02d} - RÉSULTATS DÉTAILLÉS")
        print(f"{'=' * 80}")

        # Timing et LR
        print(f"⏰ Temps epoch: {metrics['epoch_time']:.1f}s | LR: {metrics['learning_rate']:.2e}")

        # Métriques TRAIN
        print(f"\n🏋️  TRAIN:")
        print(f"   📉 Loss:      {metrics['train_loss']:>8.4f}")
        print(f"   📈 AUC:       {metrics['train_auc']:>8.4f}")
        print(f"   🎯 AP:        {metrics['train_ap']:>8.4f}")

        # Métriques VALIDATION
        print(f"\n🧪 VALIDATION:")
        print(
            f"   📊 AUC:       {metrics['val_auc']:>8.4f} {'🎯 BEST' if metrics['val_auc'] == self.best_metrics.get('val_auc', 0) else ''}")
        print(
            f"   🎯 AP:        {metrics['val_ap']:>8.4f} {'🎯 BEST' if metrics['val_ap'] == self.best_metrics.get('val_ap', 0) else ''}")
        print(
            f"   ⚖️  F1:        {metrics['val_f1']:>8.4f} {'🎯 BEST' if metrics['val_f1'] == self.best_metrics.get('val_f1', 0) else ''}")
        print(f"   🎯 Precision: {metrics['val_precision']:>8.4f}")
        print(f"   🔄 Recall:    {metrics['val_recall']:>8.4f}")
        print(f"   ⚖️  Bal Acc:   {metrics['val_balanced_accuracy']:>8.4f}")
        print(f"   📏 Threshold: {metrics['val_best_threshold']:>8.3f}")

        # Meilleures métriques globales
        self._print_best_metrics()

        print(f"{'=' * 80}")

    def _print_best_metrics(self):
        """Affiche les meilleures métriques globales"""
        if self.best_metrics:
            print(f"\n🏆 MEILLEURES MÉTRIQUES GLOBALES (Epoch {self.best_metrics['epoch']}):")
            print(f"   📊 Best AUC:  {self.best_metrics['val_auc']:.4f}")
            print(f"   🎯 Best AP:   {self.best_metrics['val_ap']:.4f}")
            print(f"   ⚖️  Best F1:   {self.best_metrics['val_f1']:.4f}")


class ComprehensiveTrainer:
    def __init__(self, config: OptimizedConfig):
        self.cfg = config
        self.device = self._setup_device()

        # Modèle
        self.model = OptimizedMultiExpertModel(
            embed_dim=config.embed_dim,
            use_checkpoint=config.use_checkpoint
        ).to(self.device)

        # Gestionnaires
        self.strategy_manager = ImbalanceStrategyManager(
            cancer_rate=0.0732,
            use_mixup=True,
            use_hard_mining=True,
            device=str(self.device)
        )

        # EMA
        self.ema = EMA(self.model, decay=0.999)

        # Logger de métriques
        self.metrics_logger = MetricsLogger()

        # Créer répertoire checkpoints
        Path(config.save_dir).mkdir(exist_ok=True)

        print("✅ Trainer initialisé avec métriques complètes")

    def _setup_device(self) -> torch.device:
        """Configure le device"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"🚀 GPU: {torch.cuda.get_device_name()}")
            print(f"   Memory: {torch.cuda.get_device_properties(device).total_memory / 1024 ** 3:.1f}GB")
            return device
        else:
            print("⚠️  CUDA non disponible, utilisation CPU")
            return torch.device("cpu")

    def setup_data(self, train_csv: str, val_csv: str, train_y_csv: str, val_y_csv: str, dicom_root: str):
        """Configure les données"""
        print("📊 Configuration des données...")

        # Datasets
        self.train_dataset = OptimizedDicomDataset(
            train_csv, dicom_root, train_y_csv, self.cfg, is_train=True
        )
        self.val_dataset = OptimizedDicomDataset(
            val_csv, dicom_root, val_y_csv, self.cfg, is_train=True
        )

        # DataLoaders avec sampling équilibré
        self.train_loader = self.strategy_manager.create_balanced_loader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers
        )

        # Validation sans sampling
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.cfg.batch_size * 2,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True
        )

        print(f"✅ Données chargées: {len(self.train_dataset)} train, {len(self.val_dataset)} val")

    def compute_detailed_metrics(self, preds: np.ndarray, targets: np.ndarray) -> dict:
        """Calcule toutes les métriques détaillées"""
        if len(np.unique(targets)) < 2:
            return {
                'auc': 0.5, 'average_precision': 0.5, 'f1': 0.0,
                'precision': 0.0, 'recall': 0.0, 'balanced_accuracy': 0.5,
                'best_threshold': 0.5, 'confusion_matrix': None
            }

        # Optimisation du threshold
        threshold_optimizer = ThresholdOptimizer(metric='f1')
        best_threshold, threshold_results = threshold_optimizer.optimize(targets, preds)

        # Prédictions binaires avec meilleur threshold
        best_preds = (preds >= best_threshold).astype(int)

        # Calcul de toutes les métriques
        auc = roc_auc_score(targets, preds)
        ap = average_precision_score(targets, preds)
        f1 = f1_score(targets, best_preds, zero_division=0)
        precision = precision_score(targets, best_preds, zero_division=0)
        recall = recall_score(targets, best_preds, zero_division=0)
        balanced_acc = balanced_accuracy_score(targets, best_preds)
        cm = confusion_matrix(targets, best_preds)

        return {
            'auc': auc,
            'average_precision': ap,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'balanced_accuracy': balanced_acc,
            'best_threshold': best_threshold,
            'confusion_matrix': cm
        }

    def train_epoch(self, optimizer: optim.Optimizer, epoch: int) -> dict:
        """Époque d'entraînement avec métriques détaillées"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_targets = []

        from tqdm import tqdm
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")

        for batch_idx, (img_high, img_low, labels, indices) in enumerate(pbar):
            img_high = img_high.to(self.device, non_blocking=True)
            img_low = img_low.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            indices = indices.to(self.device, non_blocking=True)

            # Mixup
            img_high, labels = self.strategy_manager.apply_mixup(img_high, labels)

            # Forward
            preds, gates, embeddings = self.model(img_high, img_low)
            loss_dict = self.strategy_manager.compute_loss(preds.squeeze(), labels)
            loss = loss_dict['total'] / self.cfg.grad_accum_steps

            # Backward
            loss.backward()

            # Hard mining
            self.strategy_manager.update_hard_mining(indices, preds.squeeze(), labels)

            # Gradient accumulation
            if (batch_idx + 1) % self.cfg.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                self.ema.update()

            # Métriques
            total_loss += loss.item() * self.cfg.grad_accum_steps
            all_preds.extend(preds.squeeze().detach().cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

            # Progress bar
            if batch_idx % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'loss': f"{loss.item() * self.cfg.grad_accum_steps:.4f}",
                    'lr': f"{current_lr:.2e}",
                    'mem': f"{torch.cuda.memory_allocated() / 1024 ** 3:.1f}GB"
                })

        # Calcul métriques train
        train_metrics = self.compute_detailed_metrics(np.array(all_preds), np.array(all_targets))
        train_metrics['loss'] = total_loss / len(self.train_loader)

        return train_metrics

    @torch.no_grad()
    def validate(self) -> dict:
        """Validation avec métriques complètes"""
        self.ema.apply_shadow()
        self.model.eval()

        all_preds = []
        all_targets = []
        all_gates = []

        from tqdm import tqdm
        pbar = tqdm(self.val_loader, desc="Validation")

        for img_high, img_low, labels in pbar:
            img_high = img_high.to(self.device, non_blocking=True)
            img_low = img_low.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            preds, gates, _ = self.model(img_high, img_low)

            all_preds.extend(preds.squeeze().cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            all_gates.extend(gates.cpu().numpy())

        self.ema.restore()

        # Métriques de validation
        val_metrics = self.compute_detailed_metrics(np.array(all_preds), np.array(all_targets))

        # Analyse des gates
        all_gates = np.array(all_gates)
        mean_gates = all_gates.mean(axis=0)
        gate_std = all_gates.std(axis=0)
        expert_names = ["Detector", "Texture", "Context", "Segment"]

        val_metrics['gate_analysis'] = {
            name: f"{mean:.3f} ± {std:.3f}"
            for name, mean, std in zip(expert_names, mean_gates, gate_std)
        }

        # Affichage matrice de confusion
        if val_metrics['confusion_matrix'] is not None:
            print(f"\n🎯 MATRICE DE CONFUSION:")
            print(f"   TN: {val_metrics['confusion_matrix'][0, 0]}, FP: {val_metrics['confusion_matrix'][0, 1]}")
            print(f"   FN: {val_metrics['confusion_matrix'][1, 0]}, TP: {val_metrics['confusion_matrix'][1, 1]}")

        return val_metrics

    def fit(self, train_csv: str, val_csv: str, train_y_csv: str, val_y_csv: str, dicom_root: str):
        """Pipeline d'entraînement complet avec métriques"""
        print("🚀 DÉMARRAGE ENTRAÎNEMENT AVEC MÉTRIQUES DÉTAILLÉES")
        print("=" * 80)

        # Setup données
        self.setup_data(train_csv, val_csv, train_y_csv, val_y_csv, dicom_root)

        # Optimizer
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
            betas=(0.9, 0.999)
        )

        # Scheduler
        from torch.optim.lr_scheduler import OneCycleLR
        steps_per_epoch = len(self.train_loader) // self.cfg.grad_accum_steps
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.cfg.lr,
            epochs=self.cfg.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.3,
            anneal_strategy='cos'
        )

        # Entraînement
        for epoch in range(self.cfg.epochs):
            epoch_start = time.time()

            # Dégeler backbones
            if epoch == self.cfg.freeze_epochs:
                print("\n🔓 Déblocage des backbones pré-entraînés...")
                self.model.unfreeze_backbones()
                for g in optimizer.param_groups:
                    g['lr'] = self.cfg.lr * 0.1

            # Entraînement
            train_metrics = self.train_epoch(optimizer, epoch)

            # Validation
            val_metrics = self.validate()

            # Timing
            epoch_time = time.time() - epoch_start

            # Learning rate actuel
            current_lr = optimizer.param_groups[0]['lr']

            # Log des métriques
            self.metrics_logger.log_epoch(
                epoch + 1, train_metrics, val_metrics, current_lr, epoch_time
            )

            # Sauvegarde best model
            if val_metrics['auc'] > self.metrics_logger.best_metrics.get('val_auc', 0):
                self.save_checkpoint(
                    f"{self.cfg.save_dir}/best_auc_{val_metrics['auc']:.4f}.pth",
                    epoch, val_metrics
                )

            # Scheduler step
            scheduler.step()

        # Sauvegarde finale
        print("\n💾 Sauvegarde du modèle final...")
        self.ema.apply_shadow()
        self.save_checkpoint(f"{self.cfg.save_dir}/final_ema.pth", self.cfg.epochs, val_metrics)

        # Résumé final
        self.print_final_summary()

    def save_checkpoint(self, path: str, epoch: int, metrics: dict):
        """Sauvegarde checkpoint"""
        torch.save({
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'ema_shadow': self.ema.shadow,
            'metrics': metrics,
            'config': self.cfg,
            'timestamp': datetime.now().isoformat()
        }, path)
        print(f"💾 Checkpoint sauvegardé: {path}")

    def print_final_summary(self):
        """Affiche le résumé final de l'entraînement"""
        best = self.metrics_logger.best_metrics
        print(f"\n{'=' * 80}")
        print(f"🎉 ENTRAÎNEMENT TERMINÉ - RÉSUMÉ FINAL")
        print(f"{'=' * 80}")
        print(f"🏆 MEILLEURES PERFORMANCES (Epoch {best['epoch']}):")
        print(f"   📊 AUC:  {best['val_auc']:.4f}")
        print(f"   🎯 AP:   {best['val_ap']:.4f}")
        print(f"   ⚖️  F1:   {best['val_f1']:.4f}")
        print(f"   🎯 Precision: {best['val_precision']:.4f}")
        print(f"   🔄 Recall:    {best['val_recall']:.4f}")
        print(f"   ⚖️  Bal Acc:   {best['val_balanced_accuracy']:.4f}")
        print(f"{'=' * 80}")


# ==========================================================
# CONFIGURATION ET LANCEMENT
# ==========================================================
def get_optimized_config():
    return OptimizedConfig(
        epochs=20,
        batch_size=8,  # Plus grand sur GPU
        grad_accum_steps=2,
        lr=1.5e-4,
        weight_decay=1e-4,
        high_res=(512, 512),
        low_res=(224, 224),
        embed_dim=256,
        use_checkpoint=True,
        freeze_epochs=4,
        num_workers=8,  # Plus de workers pour GPU
        pin_memory=True,
        save_dir="checkpoints_metrics"
    )


def main():
    """Lancement de l'entraînement"""
    print("🎯 ENTRAÎNEMENT AVEC MÉTRIQUES DÉTAILLÉES")
    print("=" * 80)

    # Configuration
    cfg = get_optimized_config()

    # Chemins (À ADAPTER)
    DATA_ROOT = "/Users/assadiabira/Bureau/Kaggle/Projet_kaggle/data"
    TRAIN_CSV = f"{DATA_ROOT}/csv/X_train.csv"
    TRAIN_Y_CSV = f"{DATA_ROOT}/csv/y_train.csv"
    VAL_CSV = f"{DATA_ROOT}/csv/X_val.csv"
    VAL_Y_CSV = f"{DATA_ROOT}/csv/y_val.csv"
    DICOM_ROOT = f"{DATA_ROOT}/train"

    # Vérification fichiers
    for path in [TRAIN_CSV, TRAIN_Y_CSV, VAL_CSV, VAL_Y_CSV]:
        if not Path(path).exists():
            print(f"❌ Fichier manquant: {path}")
            return

    # Trainer
    trainer = ComprehensiveTrainer(cfg)

    # Entraînement
    trainer.fit(
        train_csv=TRAIN_CSV,
        val_csv=VAL_CSV,
        train_y_csv=TRAIN_Y_CSV,
        val_y_csv=VAL_Y_CSV,
        dicom_root=DICOM_ROOT
    )


if __name__ == "__main__":
    main()
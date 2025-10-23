# train_final_mps_monitoring.py
import os
import time
import psutil
import GPUtil
from typing import Tuple, List, Optional, Dict
import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.metrics import f1_score
from tqdm import tqdm
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import multiprocessing as mp
import threading
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')


# ==========================================================
# MONITORING DES RESSOURCES
# ==========================================================
class ResourceMonitor:
    """Monitor CPU, RAM, GPU usage et ajuste dynamiquement"""

    def __init__(self, check_interval=5):
        self.check_interval = check_interval
        self.monitoring = False
        self.metrics = {
            'cpu_percent': [],
            'ram_percent': [],
            'gpu_memory': [],
            'batch_times': []
        }
        self.optimal_workers = max(1, mp.cpu_count() - 2)
        self.last_check = time.time()

    def start_monitoring(self):
        """Démarre le monitoring en arrière-plan"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print(f"🔍 Monitoring des ressources activé (intervalle: {self.check_interval}s)")

    def stop_monitoring(self):
        """Arrête le monitoring"""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=2)

    def _monitor_loop(self):
        """Boucle de monitoring des ressources"""
        while self.monitoring:
            try:
                # CPU
                cpu_percent = psutil.cpu_percent(interval=1)

                # RAM
                ram = psutil.virtual_memory()
                ram_percent = ram.percent

                # GPU (MPS)
                gpu_memory = self._get_gpu_memory()

                # Stocker les métriques
                self.metrics['cpu_percent'].append(cpu_percent)
                self.metrics['ram_percent'].append(ram_percent)
                self.metrics['gpu_memory'].append(gpu_memory)

                # Ajuster dynamiquement le nombre de workers
                self._adjust_workers(cpu_percent, ram_percent, gpu_memory)

                time.sleep(self.check_interval)

            except Exception as e:
                print(f"❌ Erreur monitoring: {e}")
                time.sleep(self.check_interval)

    def _get_gpu_memory(self):
        """Récupère l'utilisation mémoire GPU pour MPS"""
        try:
            if torch.backends.mps.is_available():
                # Pour MPS, on utilise torch.mps pour les infos mémoire
                return torch.mps.current_allocated_memory() / 1024 ** 3  # Convertir en GB
            else:
                return 0
        except:
            return 0

    def _adjust_workers(self, cpu_percent, ram_percent, gpu_memory):
        """Ajuste dynamiquement le nombre de workers optimal"""
        current_time = time.time()

        # Vérifier seulement toutes les 30 secondes
        if current_time - self.last_check < 30:
            return

        self.last_check = current_time

        # Logique d'ajustement
        new_workers = self.optimal_workers

        # Si CPU > 80%, réduire les workers
        if cpu_percent > 80:
            new_workers = max(1, self.optimal_workers - 2)
            print(f"⚠️  CPU élevé ({cpu_percent}%), réduction workers: {self.optimal_workers} → {new_workers}")

        # Si RAM > 85%, réduire les workers
        elif ram_percent > 85:
            new_workers = max(1, self.optimal_workers - 1)
            print(f"⚠️  RAM élevée ({ram_percent}%), réduction workers: {self.optimal_workers} → {new_workers}")

        # Si CPU < 50% et RAM < 70%, augmenter les workers
        elif cpu_percent < 50 and ram_percent < 70 and self.optimal_workers < (mp.cpu_count() - 1):
            new_workers = min(mp.cpu_count() - 1, self.optimal_workers + 1)
            print(f"✅ Ressources disponibles, augmentation workers: {self.optimal_workers} → {new_workers}")

        if new_workers != self.optimal_workers:
            self.optimal_workers = new_workers

    def record_batch_time(self, batch_time):
        """Enregistre le temps d'un batch pour analyse"""
        self.metrics['batch_times'].append(batch_time)

        # Si les batches deviennent trop lents, réduire la complexité
        if len(self.metrics['batch_times']) > 10:
            avg_batch_time = np.mean(self.metrics['batch_times'][-10:])
            if avg_batch_time > 30:  # Si > 30s par batch en moyenne
                print(f"🐌 Batch lent ({avg_batch_time:.1f}s), vérifiez les ressources")

    def get_resource_status(self):
        """Retourne le statut actuel des ressources"""
        if not self.metrics['cpu_percent']:
            return "No data"

        current_cpu = self.metrics['cpu_percent'][-1] if self.metrics['cpu_percent'] else 0
        current_ram = self.metrics['ram_percent'][-1] if self.metrics['ram_percent'] else 0
        current_gpu = self.metrics['gpu_memory'][-1] if self.metrics['gpu_memory'] else 0

        return f"CPU: {current_cpu:.1f}% | RAM: {current_ram:.1f}% | GPU: {current_gpu:.1f}GB | Workers: {self.optimal_workers}"

    def print_summary(self):
        """Affiche un résumé des ressources utilisées"""
        if not self.metrics['cpu_percent']:
            print("Aucune donnée de monitoring")
            return

        avg_cpu = np.mean(self.metrics['cpu_percent'])
        avg_ram = np.mean(self.metrics['ram_percent'])
        max_ram = np.max(self.metrics['ram_percent'])
        avg_batch_time = np.mean(self.metrics['batch_times']) if self.metrics['batch_times'] else 0

        print(f"\n📊 RÉSUMÉ DES RESSOURCES:")
        print(f"  CPU moyen: {avg_cpu:.1f}%")
        print(f"  RAM moyen: {avg_ram:.1f}% (max: {max_ram:.1f}%)")
        print(f"  Temps/batch moyen: {avg_batch_time:.2f}s")
        print(f"  Workers optimaux: {self.optimal_workers}")


# ==========================================================
# CONFIG AVEC ALLOCATION DYNAMIQUE
# ==========================================================
@dataclass
class DynamicConfig:
    epochs: int = 10  # ← RÉDUIT pour tests
    batch_size: int = 16
    lr: float = 1.5e-4
    weight_decay: float = 1e-4
    image_size: Tuple[int, int] = (512, 512)
    device: str = "mps"
    # Seront ajustés dynamiquement
    num_workers: int = 0
    prefetch_factor: int = 2
    persistent_workers: bool = True
    embed_dim: int = 256
    use_checkpoint: bool = False
    save_dir: str = "checkpoints_full"
    cache_size: int = 500
    focal_alpha: float = 0.75
    focal_gamma: float = 2.5
    fscore_threshold: float = 0.3

    def __post_init__(self):
        # Initialisation basée sur les ressources disponibles
        self._initialize_from_resources()

    def _initialize_from_resources(self):
        """Initialise la config basée sur les ressources système"""
        cpu_count = mp.cpu_count()
        ram_gb = psutil.virtual_memory().total / 1024 ** 3

        print(f"💻 Ressources système: {cpu_count} CPU, {ram_gb:.1f}GB RAM")

        # Ajuster le nombre de workers
        if cpu_count >= 8 and ram_gb >= 16:
            self.num_workers = 4
            self.batch_size = 16
            self.cache_size = 1000
        elif cpu_count >= 4 and ram_gb >= 8:
            self.num_workers = 2
            self.batch_size = 8
            self.cache_size = 500
        else:
            self.num_workers = 0
            self.batch_size = 4
            self.cache_size = 200

        print(f"⚙️  Configuration initiale: {self.num_workers} workers, batch_size={self.batch_size}")


# ==========================================================
# DATASET AVEC GESTION DYNAMIQUE DE MÉMOIRE
# ==========================================================
# ==========================================================
# MÉTRIQUES COMPLÈTES POUR DONNÉES DÉSÉQUILIBRÉES
# ==========================================================
class ClassificationMetrics:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.all_preds = []
        self.all_probs = []
        self.all_targets = []

    def update(self, probs, targets):
        """Met à jour les métriques avec de nouvelles prédictions"""
        # Conversion MPS -> CPU pour numpy
        if torch.is_tensor(probs):
            probs_cpu = probs.detach().cpu().numpy()
        else:
            probs_cpu = probs

        if torch.is_tensor(targets):
            targets_cpu = targets.detach().cpu().numpy()
        else:
            targets_cpu = targets

        self.all_probs.extend(probs_cpu)
        self.all_targets.extend(targets_cpu)
        preds = (np.array(probs_cpu) > self.threshold).astype(int)
        self.all_preds.extend(preds)

    def compute_all(self):
        """Calcule toutes les métriques importantes"""
        from sklearn.metrics import (roc_auc_score, precision_score, recall_score,
                                     f1_score, accuracy_score, confusion_matrix,
                                     average_precision_score, balanced_accuracy_score)

        probs = np.array(self.all_probs)
        preds = np.array(self.all_preds)
        targets = np.array(self.all_targets)

        # Vérifier qu'on a des données
        if len(targets) == 0 or len(np.unique(targets)) < 2:
            return self._empty_metrics()

        metrics = {}

        # Métriques de base
        metrics['accuracy'] = accuracy_score(targets, preds)
        metrics['precision'] = precision_score(targets, preds, zero_division=0)
        metrics['recall'] = recall_score(targets, preds, zero_division=0)
        metrics['f1'] = f1_score(targets, preds, zero_division=0)
        metrics['balanced_accuracy'] = balanced_accuracy_score(targets, preds)

        # Métriques pour données déséquilibrées
        try:
            metrics['auc_roc'] = roc_auc_score(targets, probs)
        except:
            metrics['auc_roc'] = 0.5

        try:
            metrics['auc_pr'] = average_precision_score(targets, probs)
        except:
            metrics['auc_pr'] = 0.0

        # Matrice de confusion
        cm = confusion_matrix(targets, preds)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            metrics['tn'] = tn
            metrics['fp'] = fp
            metrics['fn'] = fn
            metrics['tp'] = tp
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        else:
            metrics['tn'] = metrics['fp'] = metrics['fn'] = metrics['tp'] = 0
            metrics['specificity'] = 0

        # F-score à différents seuils
        metrics['f1_scores'] = self._compute_f1_at_thresholds(probs, targets)

        # Métriques additionnelles pour déséquilibre
        metrics = self._add_imbalance_metrics(metrics, targets)

        return metrics

    def _compute_f1_at_thresholds(self, probs, targets, thresholds=None):
        """Calcule F1 à différents seuils"""
        if thresholds is None:
            thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

        f1_scores = {}
        for threshold in thresholds:
            preds = (probs > threshold).astype(int)
            f1 = f1_score(targets, preds, zero_division=0)
            f1_scores[f'f1_th_{threshold}'] = f1

        return f1_scores

    def _add_imbalance_metrics(self, metrics, targets):
        """Ajoute des métriques spécifiques au déséquilibre"""
        from sklearn.metrics import cohen_kappa_score, matthews_corrcoef

        preds = np.array(self.all_preds)

        # Kappa de Cohen
        try:
            metrics['kappa'] = cohen_kappa_score(targets, preds)
        except:
            metrics['kappa'] = 0.0

        # Matthews Correlation Coefficient
        try:
            metrics['mcc'] = matthews_corrcoef(targets, preds)
        except:
            metrics['mcc'] = 0.0

        # Rapport de classe (pour déséquilibre)
        pos_ratio = np.mean(targets)
        metrics['class_ratio'] = pos_ratio
        metrics['imbalance_ratio'] = (1 - pos_ratio) / pos_ratio if pos_ratio > 0 else float('inf')

        return metrics

    def _empty_metrics(self):
        """Retourne des métriques par défaut si pas de données"""
        return {
            'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
            'balanced_accuracy': 0.0, 'auc_roc': 0.5, 'auc_pr': 0.0,
            'specificity': 0.0, 'kappa': 0.0, 'mcc': 0.0,
            'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0,
            'class_ratio': 0.0, 'imbalance_ratio': float('inf'),
            'f1_scores': {f'f1_th_{t}': 0.0 for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}
        }

    def print_detailed_report(self, prefix=""):
        """Affiche un rapport détaillé des métriques"""
        metrics = self.compute_all()

        print(f"\n{prefix} RAPPORT DÉTAILLÉ:")
        print(f"  {'=' * 50}")
        print(f"  📊 MÉTRIQUES DE BASE:")
        print(f"    Accuracy:        {metrics['accuracy']:.4f}")
        print(f"    Balanced Acc:    {metrics['balanced_accuracy']:.4f}")
        print(f"    Precision:       {metrics['precision']:.4f}")
        print(f"    Recall:          {metrics['recall']:.4f}")
        print(f"    F1-score:        {metrics['f1']:.4f}  ← PRINCIPAL")
        print(f"    Specificity:     {metrics['specificity']:.4f}")

        print(f"  📈 MÉTRIQUES DÉSÉQUILIBRE:")
        print(f"    AUC-ROC:         {metrics['auc_roc']:.4f}")
        print(f"    AUC-PR:          {metrics['auc_pr']:.4f}  ← IMPORTANT")
        print(f"    Kappa:           {metrics['kappa']:.4f}")
        print(f"    MCC:             {metrics['mcc']:.4f}")
        print(f"    Ratio positif:   {metrics['class_ratio']:.3f}")

        print(f"  🎯 MATRICE DE CONFUSION:")
        print(f"    TP: {metrics['tp']} | FP: {metrics['fp']}")
        print(f"    FN: {metrics['fn']} | TN: {metrics['tn']}")

        print(f"  🔧 F1-SCORES PAR SEUIL:")
        for th, f1 in metrics['f1_scores'].items():
            marker = " ← OPTIMAL" if f1 == max(metrics['f1_scores'].values()) else ""
            print(f"    {th}: {f1:.4f}{marker}")

class SmartDicomDataset(Dataset):
    def __init__(self, x_csv: str, dicom_root: str, y_csv: Optional[str] = None,
                 config: DynamicConfig = None, is_train: bool = True):
        self.df = pd.read_csv(x_csv)
        self.dicom_root = Path(dicom_root)
        self.is_train = is_train
        self.cfg = config or DynamicConfig()
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

        if is_train and y_csv:
            self.labels = pd.read_csv(y_csv)
            cancer_count = self.labels['cancer'].sum()
            total_count = len(self.labels)
            print(f"[Dataset] Classe 0: {total_count - cancer_count}, Classe 1: {cancer_count}")
            print(f"[Dataset] Ratio déséquilibre: {cancer_count / total_count:.3f}")
        else:
            self.labels = None

        # Validation des fichiers
        self.valid_paths = []
        self.valid_indices = []

        for idx, row in self.df.iterrows():
            expected = self.dicom_root / f"{row['patient_id']}_{row['image_id']}.dcm"
            if expected.exists():
                self.valid_paths.append(str(expected.resolve()))
                self.valid_indices.append(idx)

        # Réindexation
        self.df = self.df.iloc[self.valid_indices].reset_index(drop=True)
        if self.labels is not None:
            self.labels = self.labels.iloc[self.valid_indices].reset_index(drop=True)

        print(f"[Dataset] {len(self)} échantillons valides")
        print(f"[Dataset] Cache size: {self.cfg.cache_size}")

    def __len__(self) -> int:
        return len(self.valid_paths)

    def _load_dicom(self, path: str) -> np.ndarray:
        try:
            import pydicom
            ds = pydicom.dcmread(path, force=True)
            img = ds.pixel_array.astype(np.float32)

            # Optimisation: réduire la taille si nécessaire
            if img.shape[0] > 512 or img.shape[1] > 512:
                import cv2
                img = cv2.resize(img, (512, 512))

            if img.max() > img.min():
                img = (img - img.min()) / (img.max() - img.min())
            else:
                img = np.zeros_like(img)

            if img.ndim == 2:
                img = np.expand_dims(img, axis=-1)
            elif img.shape[-1] > 3:
                img = img[:, :, :3]

            return img
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return np.zeros((*self.cfg.image_size, 1), dtype=np.float32)

    def __getitem__(self, idx: int):
        path = self.valid_paths[idx]

        # Gestion intelligente du cache
        if path in self.cache:
            self.cache_hits += 1
            img = self.cache[path]
        else:
            self.cache_misses += 1
            img = self._load_dicom(path)

            # Nettoyer le cache si nécessaire (stratégie LRU simple)
            if len(self.cache) >= self.cfg.cache_size:
                # Supprimer un élément au hasard (simplifié)
                if self.cache:
                    self.cache.pop(next(iter(self.cache)))

            self.cache[path] = img

        img_t = torch.from_numpy(img).permute(2, 0, 1).float()

        if self.is_train and self.labels is not None:
            label_value = float(self.labels.iloc[idx]['cancer'])
            label = torch.tensor(label_value, dtype=torch.float32)
            return img_t, label, torch.tensor(idx, dtype=torch.long)

        return img_t

    def get_cache_stats(self):
        """Retourne les statistiques du cache"""
        total_access = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_access if total_access > 0 else 0
        return {
            'hits': self.cache_hits,
            'misses': self.cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self.cache)
        }


# ==========================================================
# TRAINER INTELLIGENT AVEC MONITORING
# ==========================================================
class SmartTrainer:
    def __init__(self, cfg: DynamicConfig):
        self.cfg = cfg
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        # Monitoring des ressources
        self.resource_monitor = ResourceMonitor()
        self.resource_monitor.start_monitoring()

        print(f"🚀 SmartTrainer initialisé sur {self.device}")
        print(f"⚙️  Configuration: {cfg.num_workers} workers, batch_size={cfg.batch_size}")

        # Modèle
        try:
            from multi_expert import OptimizedMultiExpertModel
            self.model = OptimizedMultiExpertModel(
                embed_dim=cfg.embed_dim,
                use_checkpoint=cfg.use_checkpoint
            ).to(self.device)
        except ImportError:
            # Fallback simple
            class SimpleModel(nn.Module):
                def __init__(self, embed_dim=256):
                    super().__init__()
                    self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
                    self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
                    self.pool = nn.AdaptiveAvgPool2d((1, 1))
                    self.fc = nn.Linear(64, 1)

                def forward(self, x):
                    if x.shape[1] == 3:
                        x = x.mean(dim=1, keepdim=True)
                    x = F.relu(self.conv1(x))
                    x = self.pool(F.relu(self.conv2(x)))
                    x = x.view(x.size(0), -1)
                    return self.fc(x)

            self.model = SimpleModel(cfg.embed_dim).to(self.device)
            print("Using fallback SimpleModel")

        # Optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        # Métriques
        self.train_metrics = ClassificationMetrics(threshold=cfg.fscore_threshold)
        self.val_metrics = ClassificationMetrics(threshold=cfg.fscore_threshold)

        # Historique
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_f1': [], 'val_f1': [],
            'train_auc_pr': [], 'val_auc_pr': [],
            'best_f1': 0.0, 'best_epoch': 0,
            'resource_usage': []
        }

    def create_dynamic_loader(self, dataset, is_train=True):
        """Crée un DataLoader avec paramètres dynamiques"""
        current_workers = self.resource_monitor.optimal_workers

        # Ajuster le batch size basé sur la mémoire disponible
        ram_percent = psutil.virtual_memory().percent
        dynamic_batch_size = self.cfg.batch_size

        if ram_percent > 80:
            dynamic_batch_size = max(4, self.cfg.batch_size // 2)
            print(f"⚠️  RAM élevée, réduction batch_size: {self.cfg.batch_size} → {dynamic_batch_size}")

        loader = DataLoader(
            dataset,
            batch_size=dynamic_batch_size,
            shuffle=is_train,
            num_workers=current_workers,
            pin_memory=True,
            persistent_workers=current_workers > 0,
            prefetch_factor=2 if current_workers > 0 else None,
            drop_last=is_train
        )

        return loader

    def setup_data(self, train_csv, val_csv, train_y_csv, val_y_csv, dicom_train_root, dicom_val_root):
        print("📂 Chargement des datasets avec optimisation...")

        self.train_dataset = SmartDicomDataset(
            train_csv, dicom_train_root, train_y_csv, self.cfg, True
        )
        self.val_dataset = SmartDicomDataset(
            val_csv, dicom_val_root, val_y_csv, self.cfg, True
        )

        # Loaders dynamiques
        self.train_loader = self.create_dynamic_loader(self.train_dataset, True)
        self.val_loader = self.create_dynamic_loader(self.val_dataset, False)

        print(f"[SmartTrainer] Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}")
        print(f"[SmartTrainer] Loaders: {len(self.train_loader)} batches (batch_size dynamique)")

    def train_epoch(self, epoch):
        self.model.train()
        self.train_metrics.reset()
        total_loss = 0
        batch_count = 0
        epoch_start = time.time()

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}")
        for batch_idx, batch in enumerate(pbar):
            batch_start = time.time()

            try:
                if len(batch) == 3:
                    imgs, labels, indices = batch
                else:
                    imgs, labels = batch
                    indices = None

                imgs, labels = imgs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(imgs)

                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs

                logits = logits.view(-1)

                with torch.no_grad():
                    probs = torch.sigmoid(logits)

                # Loss simple pour la démo
                loss = F.binary_cross_entropy_with_logits(logits, labels)

                loss.backward()
                self.optimizer.step()

                # Métriques
                self.train_metrics.update(probs, labels)

                total_loss += loss.item()
                batch_count += 1

                # Monitoring du temps de batch
                batch_time = time.time() - batch_start
                self.resource_monitor.record_batch_time(batch_time)

                # Mise à jour dynamique de la progress bar
                if batch_idx % 10 == 0:
                    resource_status = self.resource_monitor.get_resource_status()
                    pbar.set_postfix({
                        "loss": f"{loss.item():.4f}",
                        "batch_time": f"{batch_time:.1f}s",
                        "resources": resource_status[:30] + "..."
                    })

            except Exception as e:
                print(f"Batch {batch_idx} error: {e}")
                continue

        epoch_time = time.time() - epoch_start
        print(f"⏱️  Epoch {epoch + 1} terminée en {epoch_time:.1f}s")

        # Statistiques du cache
        cache_stats = self.train_dataset.get_cache_stats()
        print(f"📦 Cache: {cache_stats['hit_rate']:.1%} hit rate ({cache_stats['hits']}/{cache_stats['misses']})")

        # Calcul des métriques
        train_metrics = self.train_metrics.compute_all()
        avg_loss = total_loss / batch_count if batch_count > 0 else 0

        self.history['train_loss'].append(avg_loss)
        self.history['train_f1'].append(train_metrics['f1'])
        self.history['train_auc_pr'].append(train_metrics['auc_pr'])

        return avg_loss, train_metrics

    # ... (le reste des méthodes validate, print_metrics, train reste similaire)

    def train(self, train_csv, val_csv, train_y_csv, val_y_csv, dicom_train_root, dicom_val_root):
        try:
            self.setup_data(train_csv, val_csv, train_y_csv, val_y_csv, dicom_train_root, dicom_val_root)

            print("\n" + "=" * 70)
            print("🎯 ENTRAÎNEMENT INTELLIGENT AVEC MONITORING DES RESSOURCES")
            print("=" * 70)

            for epoch in range(self.cfg.epochs):
                print(f"\n{'=' * 50}")
                print(f"EPOCH {epoch + 1}/{self.cfg.epochs} - {datetime.now().strftime('%H:%M:%S')}")
                print(f"{'=' * 50}")

                # Entraînement
                train_loss, train_metrics = self.train_epoch(epoch)
                train_metrics['loss'] = train_loss

                # Validation
                val_loss, val_metrics = self.validate()
                val_metrics['loss'] = val_loss

                # Affichage des métriques
                self.print_metrics(train_metrics, "TRAIN")
                self.print_metrics(val_metrics, "VAL")

                # Sauvegarde conditionnelle
                current_f1 = val_metrics['f1']
                if current_f1 > self.history['best_f1']:
                    self.history['best_f1'] = current_f1
                    self.history['best_epoch'] = epoch
                    torch.save(self.model.state_dict(), f"best_model_f1_{current_f1:.4f}.pth")
                    print(f"🎯 NOUVEAU MEILLEUR MODÈLE! F1-score: {current_f1:.4f}")

                # Early stopping adaptatif
                if epoch - self.history['best_epoch'] > 3:  # Plus tolérant
                    print(f"🛑 Early stopping adaptatif à l'epoch {epoch + 1}")
                    break

            # Résumé final
            self.print_final_summary()

        finally:
            # Toujours arrêter le monitoring
            self.resource_monitor.stop_monitoring()
            self.resource_monitor.print_summary()

    def print_final_summary(self):
        print("\n" + "=" * 80)
        print("📊 RÉSUMÉ COMPLET DE L'ENTRAÎNEMENT")
        print("=" * 80)
        print(f"Meilleur F1-score: {self.history['best_f1']:.4f} (epoch {self.history['best_epoch'] + 1})")

        # Graphiques des ressources si matplotlib disponible
        try:
            self._plot_resource_usage()
        except:
            print("📈 Matplotlib non disponible pour les graphiques de ressources")

    def _plot_resource_usage(self):
        """Génère des graphiques de l'utilisation des ressources"""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # CPU et RAM
        if self.resource_monitor.metrics['cpu_percent']:
            axes[0, 0].plot(self.resource_monitor.metrics['cpu_percent'], label='CPU %')
            axes[0, 0].plot(self.resource_monitor.metrics['ram_percent'], label='RAM %')
            axes[0, 0].set_title('Utilisation CPU/RAM')
            axes[0, 0].legend()

        # Temps des batches
        if self.resource_monitor.metrics['batch_times']:
            axes[0, 1].plot(self.resource_monitor.metrics['batch_times'])
            axes[0, 1].set_title('Temps par batch (s)')

        # Métriques d'entraînement
        axes[1, 0].plot(self.history['train_loss'], label='Train Loss')
        axes[1, 0].plot(self.history['val_loss'], label='Val Loss')
        axes[1, 0].set_title('Loss')
        axes[1, 0].legend()

        axes[1, 1].plot(self.history['train_f1'], label='Train F1')
        axes[1, 1].plot(self.history['val_f1'], label='Val F1')
        axes[1, 1].set_title('F1-Score')
        axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig('training_monitoring.png', dpi=150, bbox_inches='tight')
        plt.close()

        print("📈 Graphiques de monitoring sauvegardés dans 'training_monitoring.png'")


# ==========================================================
# MAIN
# ==========================================================
# ==========================================================
# MAIN
# ==========================================================
def main():
    print("🚀 LANCEMENT AVEC SURVEILLANCE DES RESSOURCES")
    print("💻 Analyse des ressources système...")

    # Analyse détaillée du système
    cpu_count = mp.cpu_count()
    ram_gb = psutil.virtual_memory().total / 1024 ** 3
    available_ram_gb = psutil.virtual_memory().available / 1024 ** 3

    print(f"📊 Système détecté:")
    print(f"   CPU: {cpu_count} cores")
    print(f"   RAM: {ram_gb:.1f}GB total, {available_ram_gb:.1f}GB disponible")
    print(f"   MPS: {torch.backends.mps.is_available()}")

    # Configuration dynamique basée sur le système
    cfg = DynamicConfig()

    # Ajustements fins basés sur les ressources réelles
    if available_ram_gb < 4:
        print("⚠️  RAM limitée, configuration minimaliste activée")
        cfg.batch_size = 4
        cfg.cache_size = 100
        cfg.num_workers = 0
    elif available_ram_gb < 8:
        print("🔧 RAM modérée, configuration équilibrée")
        cfg.batch_size = 8
        cfg.cache_size = 300
        cfg.num_workers = min(2, cpu_count - 1)
    else:
        print("✅ Bonnes ressources, configuration optimale")
        cfg.batch_size = 16
        cfg.cache_size = 500
        cfg.num_workers = min(4, cpu_count - 1)

    print(f"⚙️  Configuration finale:")
    print(f"   - Batch size: {cfg.batch_size}")
    print(f"   - Workers: {cfg.num_workers}")
    print(f"   - Cache: {cfg.cache_size} images")
    print(f"   - Epochs: {cfg.epochs}")

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_ROOT = PROJECT_ROOT / "data"

    TRAIN_CSV = DATA_ROOT / "csv" / "X_train.csv"
    TRAIN_Y_CSV = DATA_ROOT / "csv" / "y_train.csv"
    VAL_CSV = DATA_ROOT / "csv" / "X_val.csv"
    VAL_Y_CSV = DATA_ROOT / "csv" / "y_val.csv"
    DICOM_TRAIN_ROOT = DATA_ROOT / "train"
    DICOM_VAL_ROOT = DATA_ROOT / "val"

    # Vérification des fichiers
    print("\n🔍 Vérification des fichiers...")
    missing_files = []
    for file_path in [TRAIN_CSV, TRAIN_Y_CSV, VAL_CSV, VAL_Y_CSV]:
        if not file_path.exists():
            missing_files.append(file_path)

    if missing_files:
        print("❌ Fichiers manquants:")
        for f in missing_files:
            print(f"   - {f}")
        return

    print("✅ Tous les fichiers nécessaires sont présents")

    # Analyse du dataset
    print("\n📈 Analyse du dataset...")
    try:
        train_labels = pd.read_csv(TRAIN_Y_CSV)
        val_labels = pd.read_csv(VAL_Y_CSV)

        train_pos = train_labels['cancer'].sum()
        val_pos = val_labels['cancer'].sum()

        print(f"📊 Distribution des classes:")
        print(f"   Train - Positifs: {train_pos}/{len(train_labels)} ({train_pos / len(train_labels):.3%})")
        print(f"   Val   - Positifs: {val_pos}/{len(val_labels)} ({val_pos / len(val_labels):.3%})")

        # Ajustement dynamique du nombre d'epochs basé sur la taille du dataset
        total_samples = len(train_labels) + len(val_labels)
        if total_samples > 10000:
            cfg.epochs = min(15, cfg.epochs)  # Réduire si très grand dataset
            print(f"📉 Grand dataset détecté, réduction à {cfg.epochs} epochs")

    except Exception as e:
        print(f"⚠️  Erreur lors de l'analyse du dataset: {e}")

    # Estimation du temps
    print("\n⏱️  Estimation du temps d'entraînement...")
    if cfg.num_workers > 0:
        estimated_batch_time = 5  # secondes avec parallélisme
    else:
        estimated_batch_time = 15  # secondes sans parallélisme

    total_batches = (len(train_labels) // cfg.batch_size) * cfg.epochs
    total_time_seconds = total_batches * estimated_batch_time
    hours = total_time_seconds // 3600
    minutes = (total_time_seconds % 3600) // 60

    print(f"   Temps estimé: {hours}h{minutes:02d}")
    print(f"   Début: {datetime.now().strftime('%H:%M:%S')}")

    # Démarrage de l'entraînement
    print("\n🎯 Démarrage de l'entraînement intelligent...")
    trainer = SmartTrainer(cfg)

    try:
        trainer.train(TRAIN_CSV, VAL_CSV, TRAIN_Y_CSV, VAL_Y_CSV, DICOM_TRAIN_ROOT, DICOM_VAL_ROOT)
    except KeyboardInterrupt:
        print("\n🛑 Entraînement interrompu par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n🧹 Nettoyage des ressources...")


if __name__ == "__main__":
    main()
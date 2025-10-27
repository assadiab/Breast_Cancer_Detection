"""
Training pipeline for hybrid ResNet-50 model.

This module handles the complete training and validation process including:
- Progressive fine-tuning across three phases
- Gradient accumulation for stable training
- Automatic learning rate scheduling
- Early stopping based on validation performance
- Model checkpointing and training metrics tracking

Training phases:
1. Feature extraction (frozen ResNet, train classifier only)
2. Partial fine-tuning (unfreeze later ResNet layers)
3. Full fine-tuning (unfreeze all layers)
"""
import time
import torch
import numpy as np
import pandas as pd
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, accuracy_score


class TrainingConfig:
    """
    Configuration class for the progressive fine-tuning training strategy.

    Defines learning rates, epoch distribution, and training parameters
    for the three-phase fine-tuning approach: feature extraction,
    partial fine-tuning, and full fine-tuning.
    """
    def __init__(self):
        self.learning_rates = {
            'phase1': 1e-4,
            'phase2': 5e-5,
            'phase3': 1e-6
        }
        self.phase_epochs = [5, 8, 10]
        self.total_epochs = sum(self.phase_epochs)
        self.early_stopping_patience = 6
        self.accumulation_steps = 2
        self.batch_size = 32


class Trainer:
    """
    Handles the complete training pipeline for our hybrid ResNet-50 model.

    Manages progressive fine-tuning across three phases with gradient accumulation,
    automatic threshold optimization, early stopping, and comprehensive metrics tracking.
    Saves best model checkpoints based on validation F1-score.

    Args:
        model: HybridResnet50 model instance
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        criterion: Loss function
        device: Computing device
    """
    def __init__(self, model, train_loader, val_loader, criterion, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.device = device
        self.results = []

    def train_one_epoch(self, optimizer, accumulation_steps=2):
        self.model.train()
        total_loss = 0
        all_preds, all_labels = [], []

        optimizer.zero_grad()

        for batch_idx, (imgs, metas, labels) in enumerate(self.train_loader):
            imgs = imgs.to(self.device)
            metas = metas.to(self.device)
            labels = labels.to(self.device).unsqueeze(1)

            logits = self.model(imgs, metas)
            loss = self.criterion(logits, labels) / accumulation_steps
            loss.backward()

            total_loss += loss.item() * accumulation_steps

            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            preds = torch.sigmoid(logits)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

            if (batch_idx + 1) % 20 == 0:
                print(f"  [{batch_idx+1}/{len(self.train_loader)}] Loss: {loss.item() * accumulation_steps:.4f}")

        if (batch_idx + 1) % accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss, np.array(all_preds), np.array(all_labels)

    def validate(self):
        self.model.eval()
        total_loss = 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for imgs, metas, labels in self.val_loader:
                imgs = imgs.to(self.device)
                metas = metas.to(self.device)
                labels = labels.to(self.device).unsqueeze(1)

                logits = self.model(imgs, metas)
                loss = self.criterion(logits, labels)

                total_loss += loss.item()
                preds = torch.sigmoid(logits)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(self.val_loader)
        return avg_loss, np.array(all_preds), np.array(all_labels)

    def calculate_optimal_threshold(self, labels, preds):
        best_threshold = 0.5
        best_f1 = 0

        for threshold in np.arange(0.1, 0.9, 0.05):
            preds_binary = (preds > threshold).astype(int)
            f1 = f1_score(labels, preds_binary, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        return best_threshold, best_f1

    def _calculate_metrics(self, labels, preds, threshold):
        preds_binary = (preds > threshold).astype(int)

        metrics = {
            'auc': roc_auc_score(labels, preds),
            'f1': f1_score(labels, preds_binary, zero_division=0),
            'recall': recall_score(labels, preds_binary, zero_division=0),
            'precision': precision_score(labels, preds_binary, zero_division=0),
            'accuracy': accuracy_score(labels, preds_binary)
        }

        return metrics, preds_binary

    def train_with_phases(self, training_config):
        current_phase = 1
        current_epoch_in_phase = 0
        phase_epochs = training_config.phase_epochs

        best_val_f1 = 0.0
        best_val_auc = 0.0
        best_val_loss = float('inf')
        patience_counter = 0
        optimal_threshold = 0.5

        from model import ModelManager
        model_manager = ModelManager(self.device)
        params_to_optimize = model_manager.setup_fine_tuning_phase(
            current_phase, training_config.learning_rates
        )

        optimizer = AdamW(params_to_optimize, weight_decay=0.01)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

        start_time = time.time()

        for epoch in range(training_config.total_epochs):
            epoch_start = time.time()

            if current_phase <= 3 and current_epoch_in_phase >= phase_epochs[current_phase-1]:
                if current_phase < 3:
                    current_phase += 1
                    current_epoch_in_phase = 0
                    params_to_optimize = model_manager.setup_fine_tuning_phase(
                        current_phase, training_config.learning_rates
                    )
                    optimizer = AdamW(params_to_optimize, weight_decay=0.01)
                    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

            current_epoch_in_phase += 1

            print(f"\nEpoch {epoch+1}/{training_config.total_epochs} | Phase {current_phase}.{current_epoch_in_phase}")

            print("Training...")
            train_loss, train_preds, train_labels = self.train_one_epoch(
                optimizer, training_config.accumulation_steps
            )

            print("Validation...")
            val_loss, val_preds, val_labels = self.validate()

            try:
                train_auc = roc_auc_score(train_labels, train_preds)
                val_auc = roc_auc_score(val_labels, val_preds)

                current_threshold, current_f1 = self.calculate_optimal_threshold(val_labels, val_preds)
                if current_f1 > best_val_f1:
                    optimal_threshold = current_threshold

                val_metrics, val_preds_binary = self._calculate_metrics(val_labels, val_preds, optimal_threshold)
                train_metrics, train_preds_binary = self._calculate_metrics(train_labels, train_preds, optimal_threshold)

                val_f1 = val_metrics['f1']
                train_f1 = train_metrics['f1']
                val_recall = val_metrics['recall']
                val_precision = val_metrics['precision']

            except Exception as e:
                print(f"Metrics calculation error: {e}")
                train_auc = val_auc = train_f1 = val_f1 = val_recall = val_precision = 0.0

            current_lr = optimizer.param_groups[0]['lr']
            epoch_time = time.time() - epoch_start

            print(f"\nResults:")
            print(f"  Train - Loss: {train_loss:.4f} | AUC: {train_auc:.4f} | F1: {train_f1:.4f}")
            print(f"  Val   - Loss: {val_loss:.4f} | AUC: {val_auc:.4f} | F1: {val_f1:.4f}")
            print(f"  Val   - Recall: {val_recall:.4f} | Precision: {val_precision:.4f}")
            print(f"  Optimal threshold: {optimal_threshold:.3f}")
            print(f"  Learning rate: {current_lr:.2e}")
            print(f"  Epoch time: {epoch_time:.1f}s")

            self.results.append({
                'epoch': epoch + 1,
                'phase': current_phase,
                'train_loss': train_loss,
                'train_auc': train_auc,
                'train_f1': train_f1,
                'val_loss': val_loss,
                'val_auc': val_auc,
                'val_f1': val_f1,
                'val_recall': val_recall,
                'val_precision': val_precision,
                'optimal_threshold': optimal_threshold,
                'learning_rate': current_lr,
                'epoch_time_s': epoch_time
            })

            scheduler.step(val_loss)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_val_auc = val_auc
                best_val_loss = val_loss
                patience_counter = 0

                torch.save({
                    'epoch': epoch,
                    'phase': current_phase,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_auc': val_auc,
                    'val_f1': val_f1,
                    'optimal_threshold': optimal_threshold,
                    'training_phase': current_phase
                }, "best_model_finetuned.pth")

                print(f"  Best model saved (F1: {val_f1:.4f})!")
            else:
                patience_counter += 1
                print(f"  Patience: {patience_counter}/{training_config.early_stopping_patience}")

            if patience_counter >= training_config.early_stopping_patience:
                print(f"Early stopping after {epoch+1} epochs!")
                break

        total_time = time.time() - start_time

        print(f"Fine-tuning completed!")
        print(f"  Best Val F1: {best_val_f1:.4f}")
        print(f"  Best Val AUC: {best_val_auc:.4f}")
        print(f"  Best Val Loss: {best_val_loss:.4f}")
        print(f"  Final optimal threshold: {optimal_threshold:.3f}")
        print(f"  Total time: {total_time/60:.1f} minutes")

        return self.results
"""
Model evaluation and performance analysis module.

This module provides tools to evaluate trained models on test data
and analyze performance through comprehensive metrics and visualizations.

Evaluation workflow:
1. Load trained model and data
2. Run inference on test set
3. Calculate performance metrics
4. Generate visualizations and reports

Core evaluation metrics:
    • ROC-AUC: Area Under Receiver Operating Characteristic Curve
    • F1-score: Harmonic mean of precision and recall
    • Recall: Sensitivity/True Positive Rate
    • Precision: Positive Predictive Value
    • Accuracy: Overall classification correctness

Note: This module is for POST-TRAINING evaluation only.
Training and validation loops are handled in train_FT.py.
"""
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix, accuracy_score


class ModelEvaluator:
    """
    Comprehensive evaluation suite for hybrid ResNet-50 model.

    Provides detailed performance analysis including metrics calculation,
    classification reports, confusion matrices, and ROC curve visualization.

    Args:
        model: Trained HybridResnet50 model
        val_loader: Validation DataLoader for evaluation
        device: Computing device
        criterion: Optional loss function for validation loss calculation
    """
    def __init__(self, model, val_loader, device, criterion=None):
        self.model = model
        self.val_loader = val_loader
        self.device = device
        self.criterion = criterion

    def evaluate(self, optimal_threshold=0.5):
        self.model.eval()
        all_preds, all_labels, all_losses = [], [], []

        with torch.no_grad():
            for imgs, metas, labels in self.val_loader:
                imgs = imgs.to(self.device)
                metas = metas.to(self.device)
                labels = labels.to(self.device).unsqueeze(1)

                logits = self.model(imgs, metas)

                if self.criterion:
                    loss = self.criterion(logits, labels)
                    all_losses.append(loss.item())

                preds = torch.sigmoid(logits)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        preds_binary = (all_preds > optimal_threshold).astype(int)

        metrics = {
            'auc': roc_auc_score(all_labels, all_preds),
            'f1': f1_score(all_labels, preds_binary, zero_division=0),
            'recall': recall_score(all_labels, preds_binary, zero_division=0),
            'precision': precision_score(all_labels, preds_binary, zero_division=0),
            'accuracy': accuracy_score(all_labels, preds_binary)
        }

        if all_losses:
            metrics['loss'] = sum(all_losses) / len(all_losses)

        return metrics, all_preds, all_labels, preds_binary

    def generate_classification_report(self, labels, preds_binary):
        return classification_report(labels, preds_binary, target_names=['Non-Cancer', 'Cancer'])

    def plot_confusion_matrix(self, labels, preds_binary, save_path=None):
        cm = confusion_matrix(labels, preds_binary)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Non-Cancer', 'Cancer'],
                   yticklabels=['Non-Cancer', 'Cancer'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def plot_roc_curve(self, labels, preds, save_path=None):
        fpr, tpr, _ = roc_curve(labels, preds)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def generate_detailed_report(self, optimal_threshold=0.5):
        metrics, all_preds, all_labels, preds_binary = self.evaluate(optimal_threshold)

        print("FINAL PERFORMANCE:")
        print(f"  AUC: {metrics['auc']:.4f}")
        print(f"  F1-Score: {metrics['f1']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")

        if 'loss' in metrics:
            print(f"  Loss: {metrics['loss']:.4f}")

        print("\nCLASSIFICATION REPORT:")
        print(self.generate_classification_report(all_labels, preds_binary))

        return metrics, all_preds, all_labels, preds_binary


class TrainingVisualizer:
    """
    Visualization and analysis tools for training results.

    Generates comprehensive training curves, saves results to CSV,
    and identifies top-performing epochs. Provides insights into
    model performance across different fine-tuning phases.

    Args:
        results_df: DataFrame containing training history and metrics
    """
    def __init__(self, results_df):
        self.results_df = results_df

    def plot_training_curves(self, save_path=None):
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))

        phase_colors = {1: 'blue', 2: 'orange', 3: 'green'}
        phase_labels = {
            1: 'Phase 1: Feature Extraction',
            2: 'Phase 2: Partial Fine-tuning',
            3: 'Phase 3: Full Fine-tuning'
        }

        for phase in [1, 2, 3]:
            phase_mask = self.results_df['phase'] == phase
            if phase_mask.any():
                axes[0,0].plot(self.results_df[phase_mask]['epoch'],
                              self.results_df[phase_mask]['train_loss'],
                              label=phase_labels[phase], marker='o', linewidth=2,
                              color=phase_colors[phase])
                axes[0,0].plot(self.results_df[phase_mask]['epoch'],
                              self.results_df[phase_mask]['val_loss'],
                              marker='o', linewidth=2, linestyle='--',
                              color=phase_colors[phase], alpha=0.7)

                axes[0,1].plot(self.results_df[phase_mask]['epoch'],
                              self.results_df[phase_mask]['val_auc'],
                              label=phase_labels[phase], marker='o', linewidth=2,
                              color=phase_colors[phase])

                axes[0,2].plot(self.results_df[phase_mask]['epoch'],
                              self.results_df[phase_mask]['val_f1'],
                              label=phase_labels[phase], marker='o', linewidth=2,
                              color=phase_colors[phase])

        axes[0,0].set_xlabel('Epoch')
        axes[0,0].set_ylabel('Loss')
        axes[0,0].set_title('Loss Curves by Phase')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)

        axes[0,1].axhline(y=0.75, color='red', linestyle=':', label='Target 0.75')
        axes[0,1].set_xlabel('Epoch')
        axes[0,1].set_ylabel('AUC')
        axes[0,1].set_title('Validation AUC by Phase')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        axes[0,1].set_ylim([0.4, 1])

        axes[0,2].axhline(y=0.40, color='red', linestyle=':', label='Target 0.40')
        axes[0,2].set_xlabel('Epoch')
        axes[0,2].set_ylabel('F1-Score')
        axes[0,2].set_title('Validation F1-Score by Phase')
        axes[0,2].legend()
        axes[0,2].grid(True, alpha=0.3)
        axes[0,2].set_ylim([0, 0.8])

        axes[1,0].plot(self.results_df['val_recall'], label='Recall', marker='o', linewidth=2, color='blue')
        axes[1,0].plot(self.results_df['val_precision'], label='Precision', marker='s', linewidth=2, color='purple')
        axes[1,0].set_xlabel('Epoch')
        axes[1,0].set_ylabel('Score')
        axes[1,0].set_title('Recall vs Precision')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        axes[1,0].set_ylim([0, 1])

        axes[1,1].plot(self.results_df['optimal_threshold'], marker='o', linewidth=2, color='red')
        axes[1,1].set_xlabel('Epoch')
        axes[1,1].set_ylabel('Optimal Threshold')
        axes[1,1].set_title('Optimal Threshold Evolution')
        axes[1,1].grid(True, alpha=0.3)
        axes[1,1].set_ylim([0.1, 0.9])

        axes[1,2].semilogy(self.results_df['learning_rate'], marker='o', linewidth=2, color='brown')
        axes[1,2].set_xlabel('Epoch')
        axes[1,2].set_ylabel('Learning Rate (log)')
        axes[1,2].set_title('Learning Rate Evolution')
        axes[1,2].grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def save_results(self, csv_path='training_results_finetuned.csv'):
        self.results_df.to_csv(csv_path, index=False)
        print(f"Results saved to: {csv_path}")

    def show_top_epochs(self, n=5):
        print(f"Top {n} epochs:")
        top_epochs = self.results_df.nlargest(n, 'val_f1')[['epoch', 'phase', 'val_auc', 'val_f1', 'val_recall', 'optimal_threshold']]
        print(top_epochs)
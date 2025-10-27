"""
Model definition and management for hybrid ResNet-50 mammography classification.

This module contains the core architecture and management classes for our hybrid model:
- HybridResnet50: Combines ResNet-50 image features with clinical metadata
- ModelManager: Handles model initialization and progressive fine-tuning strategies
- LossManager: Configures weighted loss functions for imbalanced data

The hybrid approach leverages both visual patterns from mammograms and clinical
context to improve cancer detection performance.
"""
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

class HybridResnet50(nn.Module):
    """
    Hybrid model combining ResNet-50 image features with clinical metadata.

    Architecture:
    - ResNet-50 backbone for image feature extraction (initally frozen)
    - MLP for metadata processing (age, density, site, machine)
    - Combined classifier using both image and metadata features

    Args:
        num_metadata_features (int): Number of metadata features (default: 4)
        freeze_resnet (bool): Whether to freeze ResNet weights (default: True)
    """
    def __init__(self, num_metadata_features=4, freeze_resnet=True):
        super().__init__()

        self.resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.resnet_features = nn.Sequential(*list(self.resnet.children())[:-1])

        if freeze_resnet:
            for param in self.resnet.parameters():
                param.requires_grad = False

        self.metadata_mlp = nn.Sequential(
            nn.Linear(num_metadata_features, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.classifier = nn.Sequential(
            nn.Linear(2048 + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, img, metadata):
        img_features = self.resnet_features(img).view(img.size(0), -1)
        meta_features = self.metadata_mlp(metadata)
        combined = torch.cat([img_features, meta_features], dim=1)
        logits = self.classifier(combined)
        return logits

    def predict_proba(self, img, metadata):
        logits = self.forward(img, metadata)
        return torch.sigmoid(logits)

    def get_trainable_parameters_count(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params


class ModelManager:
    """
    Manages the hybrid ResNet-50 model dev and fine-tuning strategies.

    Handles model initialization, parameter counting, and progressive fine-tuning
    across three phases: feature extraction, partial fine-tuning, and full fine-tuning.

    Args:
        device: Computing device (cuda/cpu/mps)
        num_metadata_features (int): Number of metadata features for the model
    """
    def __init__(self, device, num_metadata_features=4):
        self.device = device
        self.num_metadata_features = num_metadata_features
        self.model = None

    def initialize_model(self, freeze_resnet=True):
        self.model = HybridResnet50(
            num_metadata_features=self.num_metadata_features,
            freeze_resnet=freeze_resnet
        ).to(self.device)

        total_params, trainable_params = self.model.get_trainable_parameters_count()
        print(f"Model initialized on {self.device}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        return self.model

    def setup_fine_tuning_phase(self, phase, learning_rates):
        if phase == 1:
            for param in self.model.resnet.parameters():
                param.requires_grad = False

            params_to_optimize = [
                {'params': self.model.metadata_mlp.parameters(), 'lr': learning_rates['phase1']},
                {'params': self.model.classifier.parameters(), 'lr': learning_rates['phase1']}
            ]

        elif phase == 2:
            for param in self.model.resnet.parameters():
                param.requires_grad = False

            for param in self.model.resnet.layer3.parameters():
                param.requires_grad = True
            for param in self.model.resnet.layer4.parameters():
                param.requires_grad = True

            params_to_optimize = [
                {'params': self.model.resnet.layer3.parameters(), 'lr': learning_rates['phase2']},
                {'params': self.model.resnet.layer4.parameters(), 'lr': learning_rates['phase2']},
                {'params': self.model.metadata_mlp.parameters(), 'lr': learning_rates['phase2'] * 2},
                {'params': self.model.classifier.parameters(), 'lr': learning_rates['phase2'] * 2}
            ]

        else:
            for param in self.model.parameters():
                param.requires_grad = True

            params_to_optimize = [
                {'params': self.model.resnet.parameters(), 'lr': learning_rates['phase3']},
                {'params': self.model.metadata_mlp.parameters(), 'lr': learning_rates['phase3'] * 5},
                {'params': self.model.classifier.parameters(), 'lr': learning_rates['phase3'] * 5}
            ]

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Phase {phase}: {trainable_params:,} trainable parameters")

        return params_to_optimize


class LossManager:
    """
    Handles loss function configuration for imbalanced classification.

    Computes class weights based on dataset distribution and creates
    a weighted BCEWithLogitsLoss to address class imbalance in mammography data.
    """
    @staticmethod
    def create_weighted_loss(df, device):
        num_negative = (df['cancer'] == 0).sum()
        num_positive = (df['cancer'] == 1).sum()
        pos_weight = num_negative / num_positive

        pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

        print(f"Weighted loss configured:")
        print(f"Negative cases: {num_negative:,}")
        print(f"Positive cases: {num_positive:,}")
        print(f"pos_weight: {pos_weight:.3f}")

        return criterion, pos_weight
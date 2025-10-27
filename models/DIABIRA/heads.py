# mammo_heads_corrected.py
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import monai
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


class CorrectedRSNAMammoDetector(nn.Module):
    """Détecteur RSNA"""

    def __init__(self, out_dim=256):
        super().__init__()

        # Backbone avec global pooling EXPLICITE
        self.backbone = timm.create_model(
            "tf_efficientnetv2_s.in21k_ft_in1k",
            pretrained=True,
            in_chans=1,  # ✅ Grayscale
            num_classes=0,
            global_pool='avg'  # ✅ CRITIQUE: pooling global
        )

        self.feature_dim = self.backbone.num_features

        # Tête simple et efficace
        self.head = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, out_dim),
            nn.LayerNorm(out_dim)
        )

        logger.info(f" Loaded CORRECTED RSNA Detector (feature dim: {self.feature_dim})")

    def forward(self, x):
        features = self.backbone(x)  # ← (B, feature_dim) directement
        return self.head(features)


class CorrectedKaggleTextureHead(nn.Module):
    """Tête texture"""

    def __init__(self, out_dim=256):
        super().__init__()

        # Backbone simple avec pooling
        self.backbone = timm.create_model(
            "efficientnet_b1",  # Plus stable que B3
            pretrained=True,
            in_chans=1,
            num_classes=0,
            global_pool='avg'  # ✅ Pooling global
        )

        self.feature_dim = self.backbone.num_features

        # Tête légère
        self.head = nn.Sequential(
            nn.Linear(self.feature_dim, out_dim),
            nn.LayerNorm(out_dim)
        )

        logger.info(f"✅ Loaded CORRECTED Kaggle Texture (feature dim: {self.feature_dim})")

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


class CorrectedConvNextContextHead(nn.Module):
    """Contexte ConvNeXt"""

    def __init__(self, out_dim=256):
        super().__init__()

        self.backbone = timm.create_model(
            "convnext_small.in12k_ft_in1k",
            pretrained=True,
            in_chans=1,
            num_classes=0,
            global_pool='avg'
        )

        self.feature_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Linear(self.feature_dim, out_dim),
            nn.LayerNorm(out_dim)
        )

        logger.info(f"✅ Loaded ConvNext Context (feature dim: {self.feature_dim})")

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


class CorrectedDensityHead(nn.Module):
    """Densité ResNet50"""

    def __init__(self, out_dim=256):
        super().__init__()

        self.backbone = timm.create_model(
            "resnet50.a1_in1k",
            pretrained=True,
            in_chans=1,
            num_classes=0,
            global_pool='avg'
        )

        self.feature_dim = self.backbone.num_features

        self.head = nn.Sequential(
            nn.Linear(self.feature_dim, out_dim),
            nn.LayerNorm(out_dim)
        )

        logger.info(f"✅ Loaded Density Head (feature dim: {self.feature_dim})")

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


class CorrectMammoModel(nn.Module):
    def __init__(self, embed_dim=512, fusion_strategy="transformer"):  # ← AJOUTER fusion_strategy
        super().__init__()
        self.embed_dim = embed_dim

        logger.info(f"🏗️ Building CORRECT mammography model with {fusion_strategy} fusion...")

        # 4 têtes CORRECTEMENT adaptées
        self.detector = CorrectedRSNAMammoDetector(out_dim=embed_dim)
        self.texture = CorrectedKaggleTextureHead(out_dim=embed_dim)
        self.context = CorrectedConvNextContextHead(out_dim=embed_dim)
        self.density = CorrectedDensityHead(out_dim=embed_dim)

        # ✅ UTILISER LA FUSION SPÉCIFIÉE
        from models.DIABIRA.fusion import create_fusion_strategy
        self.fusion = create_fusion_strategy(fusion_strategy, embed_dim=embed_dim)

        logger.info(f"🎯 Model architecture with {fusion_strategy} fusion")

    def forward(self, images: torch.Tensor):
        det_emb = self.detector(images)
        tex_emb = self.texture(images)
        ctx_emb = self.context(images)
        den_emb = self.density(images)

        embeddings = torch.stack([det_emb, tex_emb, ctx_emb, den_emb], dim=1)
        pred, gates = self.fusion(embeddings)

        return pred, gates, embeddings

"""
def validate_model_adaptation():
    print("\n🔍 VALIDATING MODEL ADAPTATION TO DICOM 1-CHANNEL")
    print("=" * 60)

    model = CorrectMammoModel(embed_dim=512)

    # Test avec différentes tailles d'entrée DICOM typiques
    test_sizes = [
        (2, 1, 224, 224),  # Taille standard
        (2, 1, 256, 256),  # Taille légèrement plus grande
        (2, 1, 512, 512),  # Taille mammo haute résolution
        (2, 3, 224, 224),  # Test correction canal (3 → 1)
    ]

    for size in test_sizes:
        print(f"\n🧪 Testing input size: {size}")

        try:
            dummy_input = torch.randn(size)

            with torch.no_grad():
                pred, gates, embeddings = model(dummy_input)

            print(f"   ✅ SUCCESS - Output: {pred.shape}, Gates: {gates.shape}")
            print(f"   📊 Prediction range: [{pred.min():.3f}, {pred.max():.3f}]")

        except Exception as e:
            print(f"   ❌ FAILED: {e}")

    print(f"\n🎉 MODEL VALIDATION COMPLETED!")
"""

"""
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

    # Validation du modèle
    validate_model_adaptation()

    # Test simple
    print("\n🚀 QUICK TEST:")
    model = CorrectMammoModel(embed_dim=512)
    dummy_input = torch.randn(2, 1, 224, 224)

    with torch.no_grad():
        pred, gates, embeddings = model(dummy_input)

    print(f"✅ Final test successful!")
    print(f"   Input: {dummy_input.shape}")
    print(f"   Output: {pred.shape}")
    print(f"   Sample prediction: {pred[0].item():.4f}")
"""

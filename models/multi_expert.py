# multi_expert_optimized.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model
from typing import Optional, Tuple, List
from functools import lru_cache


class SpatialAttention(nn.Module):
    """Attention spatiale efficace pour features 2D"""

    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = torch.sigmoid(self.conv(x))
        return x * attn


class EfficientDetectorHead(nn.Module):
    """Detector optimisé avec depthwise separable convs"""

    def __init__(self, out_dim: int = 256):
        super().__init__()
        # Depthwise separable convolutions pour réduire params
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, groups=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, groups=32),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, groups=64),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.spatial_attn = SpatialAttention(128)
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Channel attention
        self.channel_attn = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.Sigmoid()
        )

        self.fc = nn.Sequential(
            nn.Linear(128, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Spatial + channel attention
        x = self.spatial_attn(x)
        x_pool = self.pool(x).flatten(1)
        ch_attn = self.channel_attn(x_pool).unsqueeze(-1).unsqueeze(-1)
        x = x * ch_attn

        x = self.pool(x).flatten(1)
        return self.fc(x)


class EfficientTextureHead(nn.Module):
    """EfficientNet avec gradient checkpointing"""

    def __init__(self, out_dim: int = 256, use_checkpoint: bool = True):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.backbone = create_model(
            "efficientnet_b0",
            pretrained=True,
            in_chans=1,
            num_classes=0,
            drop_rate=0.2
        )

        self.fc = nn.Sequential(
            nn.Linear(self.backbone.num_features, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.use_checkpoint:
            x = torch.utils.checkpoint.checkpoint(self.backbone, x, use_reentrant=False)
        else:
            x = self.backbone(x)
        return self.fc(x)


class EfficientContextHead(nn.Module):
    """Swin Transformer optimisé pour contexte global"""

    def __init__(self, out_dim: int = 256, use_checkpoint: bool = True):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        # Utiliser swin_tiny au lieu de swin_base pour MPS
        self.backbone = create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=True,
            in_chans=1,
            num_classes=0
        )

        self.fc = nn.Sequential(
            nn.Linear(self.backbone.num_features, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.use_checkpoint:
            x = torch.utils.checkpoint.checkpoint(self.backbone, x, use_reentrant=False)
        else:
            x = self.backbone(x)
        return self.fc(x)


class EfficientSegmentHead(nn.Module):
    """U-Net encoder léger et rapide"""

    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.enc1 = self._make_layer(1, 32)
        self.enc2 = self._make_layer(32, 64)
        self.enc3 = self._make_layer(64, 128)
        self.enc4 = self._make_layer(128, 256)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(256, out_dim),
            nn.LayerNorm(out_dim)
        )

    def _make_layer(self, in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


class CrossModalFusion(nn.Module):
    """Fusion cross-modale avec attention efficace"""

    def __init__(self, embed_dim: int = 256, num_heads: int = 4, num_experts: int = 4):
        super().__init__()
        self.num_experts = num_experts

        # Attention multi-head optimisée
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )

        # Gating mechanism pour pondération dynamique
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * num_experts, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_experts),
            nn.Softmax(dim=-1)
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(0.1)
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Classificateur final
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            embeddings: (B, num_experts, embed_dim)
        Returns:
            pred: (B, 1) probabilités
            attn_weights: (B, num_experts) poids d'attention
        """
        B, N, D = embeddings.shape

        # Cross-attention entre experts
        attn_out, _ = self.cross_attn(embeddings, embeddings, embeddings)
        attn_out = self.norm1(embeddings + attn_out)

        # Feed-forward
        ffn_out = self.ffn(attn_out)
        out = self.norm2(attn_out + ffn_out)

        # Gating pour fusion pondérée
        concat_emb = embeddings.reshape(B, -1)
        gates = self.gate(concat_emb)  # (B, num_experts)

        # Fusion pondérée
        weighted = (out * gates.unsqueeze(-1)).sum(dim=1)  # (B, D)

        # Classification
        pred = torch.sigmoid(self.classifier(weighted))

        return pred, gates


class OptimizedMultiExpertModel(nn.Module):
    """Modèle multi-expert optimisé pour MPS avec transfer learning"""

    def __init__(self, embed_dim: int = 256, use_checkpoint: bool = True):
        super().__init__()
        self.embed_dim = embed_dim

        # Têtes spécialisées
        self.detector = EfficientDetectorHead(out_dim=embed_dim)
        self.texture = EfficientTextureHead(out_dim=embed_dim, use_checkpoint=use_checkpoint)
        self.context = EfficientContextHead(out_dim=embed_dim, use_checkpoint=use_checkpoint)
        self.segment = EfficientSegmentHead(out_dim=embed_dim)

        # Fusion
        self.fusion = CrossModalFusion(embed_dim=embed_dim, num_heads=4, num_experts=4)

        # Cache pour low-res images
        self._low_res_cache = {}

    def _get_low_res(self, x: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        """Interpolation avec cache pour éviter recalculs"""
        cache_key = (x.shape, target_size)
        if cache_key not in self._low_res_cache:
            return F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return self._low_res_cache[cache_key]

    def forward(
            self,
            images: torch.Tensor,
            images_low_res: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            images: (B, 1, H, W) images haute résolution
            images_low_res: (B, 1, 224, 224) optionnel, pour contexte
        Returns:
            pred: (B, 1) prédictions
            gates: (B, num_experts) poids des experts
            embeddings: (B, num_experts, embed_dim) embeddings avant fusion
        """
        # Préparer low-res si nécessaire
        if images_low_res is None:
            images_low_res = F.interpolate(
                images, size=(224, 224),
                mode='bilinear', align_corners=False
            )

        # Extraire features en parallèle (sera optimisé par PyTorch)
        det_emb = self.detector(images)
        tex_emb = self.texture(images)
        ctx_emb = self.context(images_low_res)
        seg_emb = self.segment(images)

        # Stack embeddings
        embeddings = torch.stack([det_emb, tex_emb, ctx_emb, seg_emb], dim=1)

        # Fusion et prédiction
        pred, gates = self.fusion(embeddings)

        return pred, gates, embeddings

    def freeze_backbones(self):
        """Geler les backbones pré-entraînés pour fine-tuning"""
        for param in self.texture.backbone.parameters():
            param.requires_grad = False
        for param in self.context.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbones(self):
        """Dégeler pour fine-tuning complet"""
        for param in self.texture.backbone.parameters():
            param.requires_grad = True
        for param in self.context.backbone.parameters():
            param.requires_grad = True
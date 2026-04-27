"""
Multi-Head Expert Model — backbones v3 (poids médicaux spécialisés)

4 experts complémentaires :
  1. MammoscreenLesionExpert   — EfficientNetV2-S (ianpan/mammoscreen, RSNA breast cancer)
  2. XRayDenseNetTextureExpert — DenseNet121 (TorchXRayVision densenet121-res224-rsna)
  3. ResNetContextExpert       — ResNet50 (RadImageNet ou ImageNet fallback)
  4. ConvNextDensityExpert     — ConvNeXt-Small (timm ImageNet-21k, winner RSNA Kaggle)

Fusion : cross-attention multi-têtes + gating dynamique + MLP classifieur
Sortie : logit brut (pas de Sigmoid) → BCEWithLogitsLoss
"""

import logging
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import torchvision.models as tv_models

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers poids médicaux
# ─────────────────────────────────────────────────────────────────────────────

def _load_radImageNet(model: nn.Module, weights_path: str, first_conv_key: str) -> bool:
    """
    Charge les poids RadImageNet dans le modèle.
    Adapte le premier conv 3-canaux → 1-canal par moyennage.
    Retourne True si succès.
    """
    if not os.path.isfile(weights_path):
        logger.warning("RadImageNet weights not found: %s — fallback ImageNet", weights_path)
        return False

    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    if first_conv_key in state_dict:
        w = state_dict[first_conv_key]           # (out_ch, 3, kH, kW)
        state_dict[first_conv_key] = w.mean(dim=1, keepdim=True)  # (out_ch, 1, kH, kW)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    logger.info(
        "RadImageNet loaded from %s — %d missing, %d unexpected keys",
        os.path.basename(weights_path), len(missing), len(unexpected),
    )
    return True


def _imagenet_fallback_densenet(net: nn.Module) -> None:
    """Charge les poids ImageNet DenseNet121 et adapte le premier conv à 1 canal."""
    pretrained = tv_models.densenet121(weights=tv_models.DenseNet121_Weights.IMAGENET1K_V1)
    state = {k: v for k, v in pretrained.state_dict().items() if "features.conv0" not in k}
    net.load_state_dict(state, strict=False)
    with torch.no_grad():
        net.features.conv0.weight.copy_(
            pretrained.features.conv0.weight.mean(dim=1, keepdim=True)
        )


def _imagenet_fallback_resnet(net: nn.Module) -> None:
    """Charge les poids ImageNet ResNet50 et adapte le premier conv à 1 canal."""
    pretrained = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V2)
    state = {k: v for k, v in pretrained.state_dict().items() if "conv1" not in k}
    net.load_state_dict(state, strict=False)
    with torch.no_grad():
        net.conv1.weight.copy_(
            pretrained.conv1.weight.mean(dim=1, keepdim=True)
        )


def _load_mammoscreen_backbone(out_dim: int) -> Tuple[nn.Module, int]:
    """
    Télécharge et extrait le backbone EfficientNetV2-S de mammoscreen (ianpan/mammoscreen).

    mammoscreen = EfficientNetV2-S pré-entraîné sur CBIS-DDSM (ROI annotations cancer)
    puis fine-tuné sur RSNA breast cancer detection (notre dataset exact).
    AUROC moyen : 0.9451 sur RSNA holdout.

    Stratégie de chargement : on télécharge model.safetensors + modeling.py directement
    depuis HuggingFace Hub, puis on instancie MammoModel manuellement.
    Contourne les incompatibilités de transformers >= 4.48 avec le code mammoscreen.

    Retourne (backbone_module, feature_dim).
    """
    try:
        import sys
        import importlib.util
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        logger.info("Téléchargement mammoscreen (ianpan/mammoscreen)...")

        # Télécharger modeling.py + poids safetensors
        modeling_path = hf_hub_download("ianpan/mammoscreen", filename="modeling.py")
        weights_path  = hf_hub_download("ianpan/mammoscreen", filename="model.safetensors")

        # Importer dynamiquement MammoModel depuis les fichiers téléchargés.
        # modeling.py fait `from .configuration import MammoConfig` (import relatif)
        # → on crée un faux package "mammoscreen_pkg" dans sys.modules pour le résoudre.
        config_path = hf_hub_download("ianpan/mammoscreen", filename="configuration.py")

        pkg_name = "mammoscreen_pkg"
        if pkg_name not in sys.modules:
            # 1. Créer un module package vide
            import types
            pkg = types.ModuleType(pkg_name)
            pkg.__path__ = []
            pkg.__package__ = pkg_name
            sys.modules[pkg_name] = pkg

            # 2. Charger configuration.py comme mammoscreen_pkg.configuration
            cfg_spec = importlib.util.spec_from_file_location(
                f"{pkg_name}.configuration", config_path
            )
            cfg_mod = importlib.util.module_from_spec(cfg_spec)
            cfg_mod.__package__ = pkg_name
            sys.modules[f"{pkg_name}.configuration"] = cfg_mod
            cfg_spec.loader.exec_module(cfg_mod)

        # 3. Charger modeling.py comme mammoscreen_pkg.modeling
        mdl_spec = importlib.util.spec_from_file_location(
            f"{pkg_name}.modeling", modeling_path
        )
        mdl_mod = importlib.util.module_from_spec(mdl_spec)
        mdl_mod.__package__ = pkg_name
        sys.modules[f"{pkg_name}.modeling"] = mdl_mod
        mdl_spec.loader.exec_module(mdl_mod)

        MammoModel = mdl_mod.MammoModel

        # Instancier net0 (taille 2048×1024, pad_to_aspect_ratio=True)
        net0 = MammoModel(
            backbone="tf_efficientnetv2_s",
            image_size=(2048, 1024),
            pad_to_aspect_ratio=True,
            feature_dim=1280,
            dropout=0.1,
            num_classes=5,
            in_chans=1,
        )

        # Charger les poids safetensors en filtrant sur le préfixe "net0."
        state_dict = load_file(weights_path, device="cpu")
        net0_state = {
            k[len("net0."):]: v
            for k, v in state_dict.items()
            if k.startswith("net0.")
        }
        missing, unexpected = net0.load_state_dict(net0_state, strict=False)
        logger.info(
            "mammoscreen net0 chargé — %d missing, %d unexpected",
            len(missing), len(unexpected),
        )

        backbone = net0.backbone  # timm tf_efficientnetv2_s, in_chans=1, num_classes=0
        feat_dim = 1280
        logger.info("MammoscreenLesionExpert: poids RSNA chargés (feat_dim=%d)", feat_dim)
        return backbone, feat_dim

    except Exception as e:
        logger.warning("mammoscreen unavailable (%s) — fallback EfficientNetV2-S ImageNet-21k", e)
        backbone = timm.create_model(
            "tf_efficientnetv2_s.in21k_ft_in1k",
            pretrained=True,
            in_chans=1,
            num_classes=0,
            global_pool="avg",
        )
        return backbone, backbone.num_features


# ─────────────────────────────────────────────────────────────────────────────
# Expert 1 — EfficientNetV2-S (mammoscreen — entraîné RSNA exact)
# Rôle : détection lésions (masses, microcalcifications)
# ─────────────────────────────────────────────────────────────────────────────

class MammoscreenLesionExpert(nn.Module):
    """
    Expert 1 : EfficientNetV2-S backbone de mammoscreen.
    Poids entraînés sur CBIS-DDSM (ROI cancer) puis RSNA breast cancer.
    Entrée : (B, 1, H, W) float32, valeurs [0, 1].
    Le backbone mammoscreen s'attend à [-1, 1] → normalisation interne.
    """

    def __init__(self, out_dim: int = 512):
        super().__init__()

        backbone, feat_dim = _load_mammoscreen_backbone(out_dim)
        self.backbone = backbone
        self._has_global_pool = hasattr(backbone, "num_features")  # timm standard

        # mammoscreen normalise [0,255] → [-1,1] en interne.
        # Nos images sont en [0,1] → on les met à l'échelle [0,255] d'abord.
        self._needs_scale = True

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1) if not self._has_global_pool else nn.Identity(),
            nn.Flatten(),
            nn.Linear(feat_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

        # Si backbone timm avec global_pool="avg" → pas besoin de pool supplémentaire
        if self._has_global_pool:
            self.head = nn.Sequential(
                nn.Linear(feat_dim, out_dim),
                nn.LayerNorm(out_dim),
            )

        logger.info("MammoscreenLesionExpert ready (feat_dim=%d → %d)", feat_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Mettre à l'échelle [0,1] → [0,255] pour mammoscreen
        # La normalisation interne du backbone est [-1,1] = (x/255 - 0.5) * 2
        # Puisqu'on passe directement dans backbone.forward (pas preprocess), on simule ça.
        if self._needs_scale:
            x = x * 255.0  # [0,1] → [0,255]
            x = (x / 255.0 - 0.5) * 2.0  # [0,255] → [-1,1]

        feats = self.backbone(x)

        # Si sortie 4D (B, C, H, W) → pool spatial
        if feats.ndim == 4:
            feats = F.adaptive_avg_pool2d(feats, 1).flatten(1)

        return self.head(feats)


# ─────────────────────────────────────────────────────────────────────────────
# Expert 2 — DenseNet121 TorchXRayVision (entraîné RSNA chest X-ray)
# Rôle : texture fine du tissu mammaire
# ─────────────────────────────────────────────────────────────────────────────

class XRayDenseNetTextureExpert(nn.Module):
    """
    Expert 2 : DenseNet121 entraîné sur RSNA chest X-ray (TorchXRayVision).
    Même modalité radiologique (X-ray), même institution (RSNA).
    Distribution de pixels proche des mammographies.

    Entrée : (B, 1, H, W) float32, valeurs [0, 1].
    TorchXRayVision attend [-1024, 1024] → normalisation interne ici.
    """

    def __init__(self, out_dim: int = 512):
        super().__init__()

        try:
            import torchxrayvision as xrv
            xrv_model = xrv.models.DenseNet(weights="densenet121-res224-rsna")
            # Extraire les features (CNN sans classifieur)
            self.backbone = xrv_model.features
            feat_dim = 1024  # DenseNet121 features dimension
            logger.info("XRayDenseNetTextureExpert: TorchXRayVision RSNA loaded (feat_dim=%d)", feat_dim)
        except Exception as e:
            logger.warning("TorchXRayVision unavailable (%s) — fallback DenseNet121 ImageNet", e)
            net = tv_models.densenet121(weights=None)
            net.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            _imagenet_fallback_densenet(net)
            feat_dim = net.classifier.in_features
            net.classifier = nn.Identity()
            self.backbone = net.features
            logger.info("XRayDenseNetTextureExpert: ImageNet fallback (feat_dim=%d)", feat_dim)

        self.head = nn.Sequential(
            nn.Linear(feat_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TorchXRayVision attend les images en [-1024, 1024]
        # Nos images sont [0, 1] → on les mappe sur la plage attendue
        x = (x - 0.5) * 2048.0  # [0,1] → [-1024, 1024]

        feats = self.backbone(x)  # (B, 1024, H', W')
        feats = F.relu(feats, inplace=True)
        feats = F.adaptive_avg_pool2d(feats, 1).flatten(1)  # (B, 1024)

        return self.head(feats)


# ─────────────────────────────────────────────────────────────────────────────
# Expert 3 — ResNet50 (RadImageNet ou ImageNet fallback)
# Rôle : distribution anatomique spatiale, contexte global
# ─────────────────────────────────────────────────────────────────────────────

class ResNetContextExpert(nn.Module):
    """
    Expert 3 : ResNet50 avec poids RadImageNet (1.35M images médicales).
    Si les poids RadImageNet ne sont pas disponibles, fallback vers ImageNet.
    Téléchargement : https://drive.google.com/file/d/1RHt2GnuOYlc_gcoTETtBDSW73mFyRAtR/view
    → placer dans checkpoints/radImageNet/RadImageNet_pytorch_model.bin
    """

    def __init__(self, out_dim: int = 512, radImageNet_path: Optional[str] = None):
        super().__init__()

        net = tv_models.resnet50(weights=None)
        net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        loaded = False
        if radImageNet_path:
            loaded = _load_radImageNet(net, radImageNet_path, "conv1.weight")
        if not loaded:
            _imagenet_fallback_resnet(net)

        feat_dim = net.fc.in_features
        net.fc = nn.Identity()
        self.backbone = net

        self.head = nn.Sequential(
            nn.Linear(feat_dim, out_dim),
            nn.LayerNorm(out_dim),
        )
        src = "RadImageNet" if loaded else "ImageNet (fallback)"
        logger.info("ResNetContextExpert ready — weights: %s (feat_dim=%d)", src, feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


# ─────────────────────────────────────────────────────────────────────────────
# Expert 4 — ConvNeXt-Small (timm ImageNet-21k — winner RSNA Kaggle)
# Rôle : vision globale, classification densité BI-RADS
# ─────────────────────────────────────────────────────────────────────────────

class ConvNextDensityExpert(nn.Module):
    """
    Expert 4 : ConvNeXt-Small ImageNet-21k (winner RSNA Kaggle AUC 0.9433).
    Fort sur la vision globale de l'image → densité, asymétrie bilatérale.
    `in_chans=1` géré nativement par timm.
    """

    def __init__(self, out_dim: int = 512):
        super().__init__()
        self.backbone = timm.create_model(
            "convnext_small.in12k_ft_in1k",
            pretrained=True,
            in_chans=1,
            num_classes=0,
            global_pool="avg",
        )
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Linear(feat_dim, out_dim),
            nn.LayerNorm(out_dim),
        )
        logger.info("ConvNextDensityExpert ready (feat_dim=%d → %d)", feat_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


# ─────────────────────────────────────────────────────────────────────────────
# Expert-Aware Fusion
# Cross-attention multi-têtes + gating dynamique + classifieur MLP
# GELU partout, LayerNorm, pas de Sigmoid (BCEWithLogitsLoss)
# ─────────────────────────────────────────────────────────────────────────────

class ExpertAwareFusion(nn.Module):
    """
    Fusion des 4 experts par cross-attention et gating dynamique.

    Architecture :
      1. Cross-attention multi-têtes entre experts (4 heads)
      2. Projection MLP par expert : embed_dim → hidden_dim  [GELU, Dropout]
      3. Gating dynamique (poids softmax par expert)
      4. Fusion pondérée → vecteur embed_dim
      5. MLP classifieur : 512 → 256 → 128 → 1  [GELU, Dropout]
    """

    def __init__(self, embed_dim: int = 512, num_experts: int = 4, hidden_dim: int = 256):
        super().__init__()
        self.num_experts = num_experts

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=4, dropout=0.1, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.expert_proj = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
            )
            for _ in range(num_experts)
        ])

        gate_in = embed_dim * num_experts + hidden_dim * num_experts
        self.gate = nn.Sequential(
            nn.Linear(gate_in, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_experts),
            nn.Softmax(dim=-1),
        )

        clf_in = embed_dim + hidden_dim * num_experts
        self.classifier = nn.Sequential(
            nn.Linear(clf_in, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            # Pas de Sigmoid — BCEWithLogitsLoss pendant l'entraînement
        )

    def forward(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            embeddings: (B, num_experts, embed_dim)
        Returns:
            logit : (B, 1)  logit brut, pas de sigmoid
            gates  : (B, num_experts)  importance des experts
        """
        B, N, D = embeddings.shape

        # 1. Cross-attention entre experts + résidual
        attn_out, _ = self.cross_attention(embeddings, embeddings, embeddings)
        attended = self.norm(embeddings + attn_out)  # (B, N, D)

        # 2. Projection spécialisée par expert
        expert_feats = torch.stack(
            [self.expert_proj[i](attended[:, i, :]) for i in range(self.num_experts)],
            dim=1,
        )  # (B, N, hidden_dim)

        # 3. Gating dynamique
        flat_attended = attended.reshape(B, -1)        # (B, N*D)
        flat_expert = expert_feats.reshape(B, -1)      # (B, N*hidden_dim)
        gates = self.gate(torch.cat([flat_attended, flat_expert], dim=-1))  # (B, N)

        # 4. Fusion pondérée
        fused = (attended * gates.unsqueeze(-1)).sum(dim=1)  # (B, D)

        # 5. Classification
        logit = self.classifier(torch.cat([fused, flat_expert], dim=-1))  # (B, 1)

        return logit, gates


# ─────────────────────────────────────────────────────────────────────────────
# MultiHeadMammoModel — modèle principal OOP
# ─────────────────────────────────────────────────────────────────────────────

class MultiHeadMammoModel(nn.Module):
    """
    Multi-Head Expert Model v3 pour la détection du cancer du sein (RSNA).

    4 experts complémentaires fusionnés par cross-attention :
      1. MammoscreenLesionExpert   — EfficientNetV2-S (ianpan/mammoscreen, RSNA breast cancer)
      2. XRayDenseNetTextureExpert — DenseNet121 (TorchXRayVision densenet121-res224-rsna)
      3. ResNetContextExpert       — ResNet50 (RadImageNet ou ImageNet fallback)
      4. ConvNextDensityExpert     — ConvNeXt-Small (timm ImageNet-21k, winner RSNA Kaggle)

    Sortie : logit brut (pas de sigmoid) → utiliser BCEWithLogitsLoss.

    Args:
        embed_dim:         dimension des embeddings (512 par défaut)
        radImageNet_resnet: chemin vers les poids RadImageNet pour Expert 3
                            (None = fallback ImageNet)
    """

    def __init__(
        self,
        embed_dim: int = 512,
        radImageNet_resnet: Optional[str] = None,
    ):
        super().__init__()

        self.expert1 = MammoscreenLesionExpert(out_dim=embed_dim)
        self.expert2 = XRayDenseNetTextureExpert(out_dim=embed_dim)
        self.expert3 = ResNetContextExpert(out_dim=embed_dim, radImageNet_path=radImageNet_resnet)
        self.expert4 = ConvNextDensityExpert(out_dim=embed_dim)
        self.fusion = ExpertAwareFusion(embed_dim=embed_dim, num_experts=4)

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info("MultiHeadMammoModel v3 ready — %d trainable params", n_params)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, 1, H, W) image mammographie, valeurs float32 [0, 1]
        Returns:
            logit: (B, 1)  logit brut
            gates: (B, 4)  poids des experts (somme = 1)
        """
        e1 = self.expert1(x)
        e2 = self.expert2(x)
        e3 = self.expert3(x)
        e4 = self.expert4(x)

        embeddings = torch.stack([e1, e2, e3, e4], dim=1)  # (B, 4, embed_dim)
        logit, gates = self.fusion(embeddings)

        return logit, gates

    def freeze_backbones(self) -> None:
        """Phase 1 d'entraînement : geler tous les backbones (heads + fusion seulement)."""
        for expert in [self.expert1, self.expert2, self.expert3, self.expert4]:
            for p in expert.backbone.parameters():
                p.requires_grad = False
        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info("Backbones gelés — %d params entraînables", n_train)

    def unfreeze_backbones(self, last_n_blocks: int = 2) -> None:
        """Phase 2 : dégeler les N derniers blocs de chaque backbone."""
        def _unfreeze_last(module: nn.Module, n: int) -> None:
            children = list(module.children())
            for child in children[-n:]:
                for p in child.parameters():
                    p.requires_grad = True

        for expert in [self.expert1, self.expert2, self.expert3, self.expert4]:
            _unfreeze_last(expert.backbone, last_n_blocks)

        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info("Backbones partiellement dégelés (last %d) — %d params entraînables",
                    last_n_blocks, n_train)

    def unfreeze_all(self) -> None:
        """Phase 3 : entraînement end-to-end complet."""
        for p in self.parameters():
            p.requires_grad = True
        n_train = sum(p.numel() for p in self.parameters())
        logger.info("Tous les backbones dégelés — %d params entraînables", n_train)

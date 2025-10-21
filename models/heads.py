# models/heads.py
from typing import Optional
import torch
import torch.nn as nn
import torchvision
import timm
from monai.networks.nets import AttentionUnet

class DetectorHead(nn.Module):
    """
    Small wrapper around a detection backbone (Faster R-CNN w/ FPN) that returns
    a pooled embedding for fusion. If you have bbox annotations, train this separately.
    """
    def __init__(self, pretrained_backbone: bool = True, embedding_dim: int = 256):
        super().__init__()
        # torchvision's Faster R-CNN with ResNet50-FPN
        self.detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=pretrained_backbone, pretrained_backbone=False
        )
        # Replace predictor if necessary - we'll not use classifier here
        # Create a small projection of pooled features to embedding
        self.pool_proj = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim)
        )

    def forward(self, images, return_rois: bool = False):
        """
        images: list[Tensor] or Tensor batch (N,C,H,W)
        returns: tensor (N, embedding_dim)
        """
        # torchvision detectors expect list[Tensor]
        list_images = images if isinstance(images, list) else [img for img in images]
        # detector returns list of dicts (boxes, scores, labels)
        detections = self.detector(list_images)
        # As a simple strategy, we take feature maps from backbone via internal API if needed.
        # Here we fallback to a dummy: compute global features using a small conv on images.
        # (Practical implementation: extract ROI-pooled feature maps from backbone.)
        if not isinstance(images, list):
            x = torch.nn.functional.adaptive_avg_pool2d(images, 1).view(images.size(0), -1)
        else:
            # images list: compute per-image pooled features and stack
            x = torch.stack([torch.nn.functional.adaptive_avg_pool2d(img.unsqueeze(0), 1).view(-1) for img in images])
        # project to embedding
        emb = self.pool_proj(x)
        if return_rois:
            return emb, detections
        return emb

class TextureHead(nn.Module):
    """
    Efficient CNN backbone for texture/intensity features.
    """
    def __init__(self, model_name: str = "efficientnet_b0", pretrained: bool = True, embedding_dim: int = 256):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool='avg')
        feat_dim = self.backbone.num_features
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, embedding_dim)
        )

    def forward(self, x):
        features = self.backbone(x)  # (N, feat_dim)
        return self.proj(features)

class ContextHead(nn.Module):
    """
    Global context encoder. Using a Swin-like or other transformer backbone via timm.
    Intended to process downsampled full-image(s) or multi-views.
    If you pass a list of views per patient, aggregate accordingly.
    """
    def __init__(self, model_name: str = "swin_base_patch4_window7_224", pretrained: bool = True, embedding_dim: int = 256):
        super().__init__()
        # Use timm transformer backbone with global avgpool
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool='avg')
        feat_dim = self.backbone.num_features
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim)
        )

    def forward(self, x):
        """
        x: Tensor (N,C,H,W) or list of views -> if list, we aggregate mean of tokens
        """
        if isinstance(x, list):
            # each element is (N,C,H,W) stacked across N? assume list length = num_views
            # simple aggregation: average embeddings across views
            embs = [self.proj(self.backbone(v)) for v in x]  # list of (N,embed)
            stacked = torch.stack(embs, dim=0).mean(dim=0)
            return stacked
        else:
            return self.proj(self.backbone(x))

class SegmentationHead(nn.Module):
    """
    Attention U-Net for segmentation; outputs a segmentation mask and an embedding extracted
    from decoder bottleneck or pooled mask features.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 1, features: int = 32, embedding_dim: int = 256):
        super().__init__()
        self.unet = AttentionUnet(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=(features, features*2, features*4, features*8),
            strides=(2,2,2)
        )
        # small conv -> pool -> proj to embedding
        self.post = nn.Sequential(
            nn.Conv2d(out_channels, 8, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(8, embedding_dim)
        )

    def forward(self, x):
        mask = self.unet(x)  # (N, out_channels, H, W)
        emb = self.post(mask)
        return emb, mask

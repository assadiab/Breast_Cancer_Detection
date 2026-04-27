from typing import Tuple

import torch
import torch.nn as nn


class BaselineCNN(nn.Module):
    """
    CNN baseline pour la détection du cancer du sein (RSNA).

    Architecture simple 5 blocs conv → GlobalAvgPool → MLP classifier.
    Sert de référence pour comparer avec le Multi-Head Expert model.

    Input  : (B, 1, H, W)  image mammographie grayscale normalisée
    Output : (B, 1)         logit brut — utiliser BCEWithLogitsLoss
    """

    def __init__(self, in_channels: int = 1, dropout: float = 0.4):
        super().__init__()

        self.features = nn.Sequential(
            # Bloc 1 : 1 → 32
            _ConvBlock(in_channels, 32),
            nn.MaxPool2d(2, 2),

            # Bloc 2 : 32 → 64
            _ConvBlock(32, 64),
            nn.MaxPool2d(2, 2),

            # Bloc 3 : 64 → 128
            _ConvBlock(64, 128),
            nn.MaxPool2d(2, 2),

            # Bloc 4 : 128 → 256
            _ConvBlock(128, 256),
            nn.MaxPool2d(2, 2),

            # Bloc 5 : 256 → 512
            _ConvBlock(256, 512),
            nn.AdaptiveAvgPool2d(1),   # → (B, 512, 1, 1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, 1),
            # Pas de Sigmoid — BCEWithLogitsLoss
        )

        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)


class _ConvBlock(nn.Module):
    """Conv 3×3 → BN → GELU → Conv 3×3 → BN → GELU avec résidual projection."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )
        self.shortcut = (
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch))
            if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.shortcut(x)

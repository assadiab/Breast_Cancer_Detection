# models/fusion.py
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedAttentionFusion(nn.Module):
    """
    Simple gated attention fusion for N feature vectors.
    Each embedding passes through a gating scalar; attention computed over gates.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 512):
        super().__init__()
        self.attn_fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU()
        )

    def forward(self, embeddings: List[torch.Tensor]):
        """
        embeddings: list of tensors (N, D)
        returns: (N, output_dim) fused vector and attention weights (N, num_emb)
        """
        stacked = torch.stack(embeddings, dim=1)  # (N, M, D)
        N, M, D = stacked.shape
        # compute attention logits per embedding
        logits = self.attn_fc(stacked.view(N*M, D)).view(N, M)  # (N, M)
        weights = torch.softmax(logits, dim=1)  # (N, M)
        # weighted sum of projected embeddings
        proj = self.proj(stacked)  # (N, M, output_dim)
        weighted = (weights.unsqueeze(-1) * proj).sum(dim=1)  # (N, output_dim)
        return weighted, weights

class TransformerFusion(nn.Module):
    """
    Alternative: treat each embedding as a token and run a small TransformerEncoder.
    """
    def __init__(self, embed_dim: int, nhead: int = 4, num_layers: int = 2, output_dim: int = 512):
        super().__init__()
        self.input_proj = nn.Linear(embed_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=embed_dim*2, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.final_proj = nn.Linear(embed_dim, output_dim)

    def forward(self, embeddings: List[torch.Tensor]):
        # embeddings: list of (N, D) -> stack to (M, N, D) for transformer (seq_len, batch, dim)
        stacked = torch.stack(embeddings, dim=0)  # (M, N, D)
        M, N, D = stacked.shape
        x = self.input_proj(stacked)  # (M,N,D)
        x = self.transformer(x)  # (M,N,D)
        # average over tokens
        x = x.mean(dim=0)  # (N,D)
        out = self.final_proj(x)  # (N, output_dim)
        # produce uniform weights not provided — optional attention maps can be derived from self-attention hooks
        return out, None

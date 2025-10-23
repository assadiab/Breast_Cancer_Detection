# models/fusion_advanced.py
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
from typing import List, Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertAwareFusion(nn.Module):
    """
    Fusion avancée avec attention adaptée aux 4 experts mammographie
    - Prend en compte la spécialisation de chaque tête
    - Mécanisme d'attention contextuelle
    - Gating dynamique basé sur la confiance
    """

    def __init__(self, embed_dim: int = 512, num_experts: int = 4, hidden_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim

        # Attention par expert avec contexte
        self.expert_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )

        # Projection pour features d'experts
        self.expert_projection = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ) for _ in range(num_experts)
        ])

        # Gating network avec contexte global
        self.gate_network = nn.Sequential(
            nn.Linear(embed_dim * num_experts + hidden_dim * num_experts, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_experts),
            nn.Softmax(dim=-1)
        )

        # Classification finale
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim + hidden_dim * num_experts, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # Normalisation
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            embeddings: (B, num_experts, embed_dim) - Features des 4 experts

        Returns:
            pred: (B, 1) - Prédictions finales
            gates: (B, num_experts) - Poids d'attention des experts
        """
        B, N, D = embeddings.shape

        # 1. Self-attention entre experts
        attn_out, attn_weights = self.expert_attention(
            embeddings, embeddings, embeddings
        )
        attended_embeddings = self.layer_norm(embeddings + attn_out)

        # 2. Projection spécialisée par expert
        expert_features = []
        for i in range(self.num_experts):
            expert_feat = self.expert_projection[i](attended_embeddings[:, i, :])
            expert_features.append(expert_feat)

        expert_features = torch.stack(expert_features, dim=1)  # (B, N, hidden_dim)

        # 3. Concaténation pour gating
        flat_embeddings = attended_embeddings.reshape(B, -1)  # (B, N*D)
        flat_expert_features = expert_features.reshape(B, -1)  # (B, N*hidden_dim)

        gate_input = torch.cat([flat_embeddings, flat_expert_features], dim=-1)
        gates = self.gate_network(gate_input)  # (B, N)

        # 4. Fusion pondérée
        weighted_embeddings = (attended_embeddings * gates.unsqueeze(-1)).sum(dim=1)  # (B, D)

        # 5. Classification avec features enrichies
        classifier_input = torch.cat([weighted_embeddings, flat_expert_features], dim=-1)
        pred = self.classifier(classifier_input)

        return pred, gates


class CrossModalTransformerFusion(nn.Module):
    """
    Fusion Transformer avancée avec cross-attention entre experts
    """

    def __init__(self, embed_dim: int = 512, num_experts: int = 4, num_heads: int = 4):
        super().__init__()

        # Transformer encoder pour fusion cross-modale
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Query pour attention globale
        self.global_query = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Gating adaptatif
        self.adaptive_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, num_experts),
            nn.Softmax(dim=-1)
        )

        # Classification
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, D = embeddings.shape

        # 1. Transformer fusion
        transformer_out = self.transformer(embeddings)  # (B, N, D)

        # 2. Attention globale
        global_query = self.global_query.expand(B, -1, -1)
        attn_weights = F.softmax(
            torch.bmm(global_query, transformer_out.transpose(1, 2)).squeeze(1),
            dim=-1
        )  # (B, N)

        # 3. Fusion avec gating adaptatif
        weighted_avg = (transformer_out * attn_weights.unsqueeze(-1)).sum(dim=1)  # (B, D)

        # Features supplémentaires pour gating
        max_pool = transformer_out.max(dim=1)[0]  # (B, D)
        gate_input = torch.cat([weighted_avg, max_pool], dim=-1)

        gates = self.adaptive_gate(gate_input)  # (B, N)

        # 4. Fusion finale pondérée
        final_fusion = (transformer_out * gates.unsqueeze(-1)).sum(dim=1)
        pred = self.classifier(final_fusion)

        return pred, gates


class HierarchicalFusion(nn.Module):
    """
    Fusion hiérarchique inspirée des décisions médicales
    - Première fusion par paires d'experts complémentaires
    - Fusion hiérarchique progressive
    """

    def __init__(self, embed_dim: int = 512):
        super().__init__()

        # Paires d'experts complémentaires
        self.detector_texture_fusion = self._create_pair_fusion(embed_dim)
        self.context_density_fusion = self._create_pair_fusion(embed_dim)

        # Fusion finale
        self.final_fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # Gating hiérarchique
        self.hierarchical_gate = nn.Sequential(
            nn.Linear(embed_dim * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
            nn.Softmax(dim=-1)
        )

    def _create_pair_fusion(self, embed_dim: int):
        return nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, D = embeddings.shape

        # Extraire les 4 experts
        det_emb = embeddings[:, 0, :]  # Détecteur
        tex_emb = embeddings[:, 1, :]  # Texture
        ctx_emb = embeddings[:, 2, :]  # Contexte
        den_emb = embeddings[:, 3, :]  # Densité

        # 1. Fusion par paires complémentaires
        # Paire 1: Détecteur + Texture (lésions + détails)
        det_tex = torch.cat([det_emb, tex_emb], dim=-1)
        fused_det_tex = self.detector_texture_fusion(det_tex)

        # Paire 2: Contexte + Densité (global + risque)
        ctx_den = torch.cat([ctx_emb, den_emb], dim=-1)
        fused_ctx_den = self.context_density_fusion(ctx_den)

        # 2. Fusion finale
        final_input = torch.cat([fused_det_tex, fused_ctx_den], dim=-1)
        pred = self.final_fusion(final_input)

        # 3. Gating hiérarchique (pour monitoring)
        all_embeddings = torch.cat([det_emb, tex_emb, ctx_emb, den_emb], dim=-1)
        gates = self.hierarchical_gate(all_embeddings)

        return pred, gates


# Factory function pour choisir la fusion
def create_fusion_strategy(strategy: str = "expert_aware", **kwargs):
    """Factory pour créer la stratégie de fusion"""
    from models.fusion import GatedAttentionFusion
    strategies = {
        "expert_aware": ExpertAwareFusion,
        "transformer": CrossModalTransformerFusion,
        "hierarchical": HierarchicalFusion,
        "simple": lambda: GatedAttentionFusion(512, 128, 512)  # Votre original
    }

    if strategy not in strategies:
        raise ValueError(f"Stratégie {strategy} non supportée. Choisir parmi: {list(strategies.keys())}")

    return strategies[strategy](**kwargs)


"""
# Test de la fusion
if __name__ == "__main__":
    # Test avec données dummy
    batch_size = 2
    num_experts = 4
    embed_dim = 512

    dummy_embeddings = torch.randn(batch_size, num_experts, embed_dim)

    print("🧪 Testing fusion strategies...")

    for strategy_name in ["expert_aware", "transformer", "hierarchical"]:
        print(f"\n🔧 Testing {strategy_name}...")

        try:
            fusion_model = create_fusion_strategy(strategy_name)
            pred, gates = fusion_model(dummy_embeddings)

            print(f"✅ {strategy_name}: SUCCESS")
            print(f"   Pred: {pred.shape}, Gates: {gates.shape}")
            print(f"   Gate values: {[f'{g:.3f}' for g in gates[0].tolist()]}")

        except Exception as e:
            print(f"❌ {strategy_name}: FAILED - {e}")
"""

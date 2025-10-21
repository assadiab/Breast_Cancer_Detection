# tests/test_attention_detailed.py
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Ajouter le chemin des modèles
models_path = Path(__file__).parent.parent / "models"
sys.path.append(str(models_path))

from models.multi_expert import CrossModalFusion, OptimizedMultiExpertModel


def visualize_attention_matrix():
    print("🔍 MATRICE D'ATTENTION DÉTAILLÉE")
    print("=" * 60)

    device = torch.device("mps")

    # Créer un custom fusion module avec hook pour capturer l'attention
    class FusionWithHook(CrossModalFusion):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.attention_weights = None

        def forward(self, embeddings):
            # Hook pour capturer les poids d'attention
            def hook(module, input, output):
                self.attention_weights = output[1]  # attention weights

            # Register hook
            handle = self.cross_attn.register_forward_hook(hook)

            # Forward normal
            result = super().forward(embeddings)

            # Remove hook
            handle.remove()

            return result

    # Modèle avec hook
    fusion = FusionWithHook(embed_dim=256, num_heads=4, num_experts=4).to(device)
    fusion.eval()

    # Simuler des embeddings d'experts
    batch_size = 2
    expert_embeddings = torch.randn(batch_size, 4, 256, device=device)

    with torch.no_grad():
        pred, gates = fusion(expert_embeddings)

    if fusion.attention_weights is not None:
        print(f"🎯 Matrice d'attention capturée: {fusion.attention_weights.shape}")

        # Visualiser pour le premier échantillon, première tête
        expert_names = ["Detector", "Texture", "Context", "Segment"]

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        for head in range(4):  # 4 têtes d'attention
            attn_matrix = fusion.attention_weights[0, head].cpu().numpy()

            im = axes[head].imshow(attn_matrix, cmap='Blues', vmin=0, vmax=1)
            axes[head].set_title(f'Tête d\'Attention {head + 1}')
            axes[head].set_xticks(range(4))
            axes[head].set_yticks(range(4))
            axes[head].set_xticklabels(expert_names, rotation=45)
            axes[head].set_yticklabels(expert_names)

            # Annoter les valeurs
            for i in range(4):
                for j in range(4):
                    axes[head].text(j, i, f'{attn_matrix[i, j]:.2f}',
                                    ha='center', va='center', fontsize=8)

        plt.suptitle('Matrices d\'Attention - Qui écoute qui?', fontsize=16)
        plt.tight_layout()
        plt.savefig('attention_matrices_detailed.png', dpi=150, bbox_inches='tight')
        print("📈 Matrices d'attention sauvegardées: attention_matrices_detailed.png")

    return fusion.attention_weights


def test_expert_specialization():
    """Test avec des experts simulés ayant des spécialisations différentes"""
    print("\n" + "=" * 60)
    print("🎭 TEST AVEC EXPERTS SPÉCIALISÉS")
    print("=" * 60)

    device = torch.device("mps")
    fusion = CrossModalFusion(embed_dim=256, num_heads=4, num_experts=4).to(device)
    fusion.eval()

    # Créer des embeddings simulés avec des "patterns" différents
    batch_size = 3

    # Cas 1: Tous les experts similaires
    similar_experts = torch.randn(1, 4, 256, device=device).repeat(batch_size, 1, 1)

    # Cas 2: Experts très différents
    diverse_experts = torch.randn(batch_size, 4, 256, device=device)

    # Cas 3: Un expert dominant (simuler un cas où un expert est très important)
    dominant_expert = torch.randn(batch_size, 4, 256, device=device)
    dominant_expert[:, 1, :] += 5.0  # Rendre le Texture expert très actif

    test_cases = {
        "Experts similaires": similar_experts,
        "Experts diversifiés": diverse_experts,
        "Expert dominant (Texture)": dominant_expert
    }

    results = {}

    with torch.no_grad():
        for name, experts in test_cases.items():
            print(f"\n🔬 {name}:")
            pred, gates = fusion(experts)

            print(f"  Gates: {[f'{g:.3f}' for g in gates[0].cpu().numpy()]}")
            print(f"  Prédiction: {pred[0].item():.4f}")

            results[name] = {
                'gates': gates.cpu().numpy(),
                'pred': pred.cpu().numpy()
            }

    return results


def analyze_real_model_attention():
    """Analyse l'attention sur le vrai modèle avec données réelles"""
    print("\n" + "=" * 60)
    print("🧠 ANALYSE ATTENTION MODÈLE RÉEL")
    print("=" * 60)

    device = torch.device("mps")
    model = OptimizedMultiExpertModel(embed_dim=256).to(device)
    model.eval()

    # Différents types d'images simulées
    print("🎭 Test avec différents types d'images simulées:")

    # 1. Images "normales" (bruit aléatoire)
    normal_images = torch.randn(2, 1, 512, 512, device=device)
    normal_lowres = torch.randn(2, 1, 224, 224, device=device)

    # 2. Images avec texture forte (simuler anomalies)
    textured_images = torch.randn(2, 1, 512, 512, device=device)
    textured_images += torch.randn_like(textured_images) * 2  # Plus de texture

    with torch.no_grad():
        # Test images normales
        pred1, gates1, emb1 = model(normal_images, normal_lowres)
        print(f"\n📊 Images 'normales':")
        expert_names = ["Detector", "Texture", "Context", "Segment"]
        for i, name in enumerate(expert_names):
            print(f"  {name}: {gates1[0][i].item():.3f}")

        # Test images texturées
        pred2, gates2, emb2 = model(textured_images, normal_lowres)
        print(f"\n📊 Images 'texturées' (anomalies):")
        for i, name in enumerate(expert_names):
            print(f"  {name}: {gates2[0][i].item():.3f}")

        # Comparaison
        print(f"\n📈 Différence gates (texturé - normal):")
        for i, name in enumerate(expert_names):
            diff = gates2[0][i].item() - gates1[0][i].item()
            print(f"  {name}: {diff:+.3f}")


if __name__ == "__main__":
    # 1. Matrice d'attention détaillée
    attention_weights = visualize_attention_matrix()

    # 2. Test spécialisation experts
    results = test_expert_specialization()

    # 3. Analyse modèle réel
    analyze_real_model_attention()

    print("\n" + "=" * 60)
    print("🎉 ANALYSE TERMINÉE!")
    print("=" * 60)
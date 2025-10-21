# tests/tests_heads.py
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path

# Ajouter le chemin des modèles
models_path = Path(__file__).parent.parent / "models"
sys.path.append(str(models_path))

try:
    from models.multi_expert import (
        EfficientDetectorHead,
        EfficientTextureHead,
        EfficientContextHead,
        EfficientSegmentHead,
        OptimizedMultiExpertModel
    )

    print("✅ Modules chargés avec succès")
except ImportError as e:
    print(f"❌ Erreur import: {e}")
    print("Vérifiez le chemin vers multi_expert_optimized.py")
    sys.exit(1)


class HeadsTester:
    def __init__(self, device="mps"):
        self.device = torch.device(device)
        print(f"🖥️  Device utilisé: {self.device}")
        self.embed_dim = 256

        # Initialiser les têtes
        print("🧠 Initialisation des têtes...")
        self.detector_head = EfficientDetectorHead(out_dim=self.embed_dim)
        self.texture_head = EfficientTextureHead(out_dim=self.embed_dim, use_checkpoint=False)
        self.context_head = EfficientContextHead(out_dim=self.embed_dim, use_checkpoint=False)
        self.segment_head = EfficientSegmentHead(out_dim=self.embed_dim)

        # Mode évaluation
        self.detector_head.eval()
        self.texture_head.eval()
        self.context_head.eval()
        self.segment_head.eval()

        # Déplacer sur device
        self.detector_head.to(self.device)
        self.texture_head.to(self.device)
        self.context_head.to(self.device)
        self.segment_head.to(self.device)
        print("✅ Têtes déplacées sur device")

    def create_dummy_batch(self, batch_size=2, high_res=(512, 512), low_res=(224, 224)):
        """Crée un batch dummy pour tester"""
        high_res_images = torch.randn(batch_size, 1, *high_res).to(self.device)
        low_res_images = torch.randn(batch_size, 1, *low_res).to(self.device)
        return high_res_images, low_res_images

    def test_individual_heads(self):
        """Teste chaque tête individuellement"""
        print("\n🧪 TEST DES TÊTES INDIVIDUELLES")
        print("=" * 60)

        high_res, low_res = self.create_dummy_batch()

        # Test Detector Head
        print("\n1. 🎯 DETECTOR HEAD")
        try:
            with torch.no_grad():
                det_emb = self.detector_head(high_res)
            print(f"   ✅ Input: {high_res.shape} -> Output: {det_emb.shape}")
            print(f"   📊 Stats - Mean: {det_emb.mean().item():.4f}, Std: {det_emb.std().item():.4f}")
        except Exception as e:
            print(f"   ❌ Erreur: {e}")

        # Test Texture Head
        print("\n2. 🔍 TEXTURE HEAD (EfficientNet-B0)")
        try:
            with torch.no_grad():
                tex_emb = self.texture_head(high_res)
            print(f"   ✅ Input: {high_res.shape} -> Output: {tex_emb.shape}")
            print(f"   📊 Stats - Mean: {tex_emb.mean().item():.4f}, Std: {tex_emb.std().item():.4f}")

        except Exception as e:
            print(f"   ❌ Erreur: {e}")

        # Test Context Head
        print("\n3. 🌐 CONTEXT HEAD (Swin-Tiny)")
        try:
            with torch.no_grad():
                ctx_emb = self.context_head(low_res)
            print(f"   ✅ Input: {low_res.shape} -> Output: {ctx_emb.shape}")
            print(f"   📊 Stats - Mean: {ctx_emb.mean().item():.4f}, Std: {ctx_emb.std().item():.4f}")

        except Exception as e:
            print(f"   ❌ Erreur: {e}")

        # Test Segment Head
        print("\n4. 🎨 SEGMENT HEAD (U-Net light)")
        try:
            with torch.no_grad():
                seg_emb = self.segment_head(high_res)
            print(f"   ✅ Input: {high_res.shape} -> Output: {seg_emb.shape}")
            print(f"   📊 Stats - Mean: {seg_emb.mean().item():.4f}, Std: {seg_emb.std().item():.4f}")
        except Exception as e:
            print(f"   ❌ Erreur: {e}")

    def test_full_model(self):
        """Test du modèle complet avec fusion"""
        print("\n" + "=" * 60)
        print("🚀 TEST MODÈLE COMPLET AVEC FUSION")
        print("=" * 60)

        high_res, low_res = self.create_dummy_batch()

        # Créer modèle complet
        full_model = OptimizedMultiExpertModel(
            embed_dim=self.embed_dim,
            use_checkpoint=False
        ).to(self.device)
        full_model.eval()

        try:
            with torch.no_grad():
                preds, gates, embeddings = full_model(high_res, low_res)

            print(f"✅ Modèle complet fonctionnel!")
            print(f"📥 Input high_res: {high_res.shape}")
            print(f"📥 Input low_res: {low_res.shape}")
            print(f"📤 Output preds: {preds.shape} (probabilités)")
            print(f"📤 Output gates: {gates.shape} (poids experts)")
            print(f"📤 Output embeddings: {embeddings.shape} (features)")

            # Analyser les gates (attention weights)
            print(f"\n🎚️  GATES (poids des experts):")
            expert_names = ["Detector", "Texture", "Context", "Segment"]
            for i, name in enumerate(expert_names):
                gate_mean = gates[:, i].mean().item()
                print(f"   {name}: {gate_mean:.4f}")

            # Vérifier prédictions
            print(f"\n🔮 PRÉDICTIONS:")
            print(f"   Range: [{preds.min().item():.4f}, {preds.max().item():.4f}]")
            print(f"   Mean: {preds.mean().item():.4f}")

        except Exception as e:
            print(f"❌ Erreur modèle complet: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Lance tous les tests"""
    print("🧪 DÉMARRAGE DES TESTS DES TÊTES MULTI-EXPERTS")

    # Vérifier MPS
    if not torch.backends.mps.is_available():
        print("❌ MPS non disponible - utilisation CPU")
        device = "cpu"
    else:
        device = "mps"
        print("✅ MPS disponible")

    tester = HeadsTester(device=device)

    # Test individuel des têtes
    tester.test_individual_heads()

    # Test modèle complet
    tester.test_full_model()

    print("\n" + "=" * 60)
    print("🎉 TESTS TERMINÉS!")
    print("=" * 60)


if __name__ == "__main__":
    main()
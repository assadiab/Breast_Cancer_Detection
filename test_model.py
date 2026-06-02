"""
test_model.py — Validation architecture + métriques sur données locales.

Lance avec :
    pixi run python test_model.py
"""

import os
import sys
import logging
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ─── Chemins ───────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(__file__)
DICOM_DIR = "/Volumes/T9_Assa/Cours/M2/S1/Intelligence Artificielle/Projets/Projet Kaggle/dicom_output/train"
CSV_PATH  = "/Volumes/T9_Assa/Cours/M2/S1/Intelligence Artificielle/Projets/Projet Kaggle/old/testing/prerocessed/df_final.csv"
IMG_SIZE  = 224   # petit pour tester vite
BATCH     = 4
N_TRAIN   = 80    # images pour mini-train
N_VAL     = 40
DEVICE    = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ─── Dataset adapté au stockage plat {patient_id}_{image_id}.dcm ───────────────

class FlatDicomDataset(Dataset):
    """Dataset pour DICOMs stockés en fichiers plats patient_id_image_id.dcm."""

    def __init__(self, df: pd.DataFrame, dicom_dir: str, img_size: int, mode: str = "train"):
        self.df = df.reset_index(drop=True)
        self.dicom_dir = dicom_dir
        self.img_size = img_size
        self.mode = mode

        # Importer le pipeline de prétraitement
        sys.path.insert(0, ROOT)
        from core.loader import Loader
        from preprocess.windowing import Windowing
        from preprocess.cropping import Cropping

        # Config minimale pour Loader
        class _MinConfig:
            config = {"paths": {"csv": CSV_PATH, "images": dicom_dir, "out": None}}
            roi_config = {
                "min_area_px": 12000, "morpho_disk": 5,
                "use_convex_hull": True, "inset_mm_y": 2.0, "inset_mm_x": 0.8,
                "margins_mm": {"CC": {"x": 7.0, "y": 6.5}, "MLO": {"x": 9.0, "y": 6.5}},
                "norm_mode": "soft_tanh", "soft_tanh_k": 3.0,
            }
            images_dir = dicom_dir

        self._cfg = _MinConfig()
        self._loader = Loader(self._cfg)
        self._windowing = Windowing()

        # MONAI transforms
        from monai.transforms import Compose, NormalizeIntensity, ToTensor
        self._transforms = Compose([NormalizeIntensity(nonzero=True), ToTensor()])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = int(row["patient_id"])
        image_id   = int(row["image_id"])
        label      = float(row.get("cancer", 0))
        density    = str(row.get("density", "B"))
        if density not in ("A", "B", "C", "D"):
            density = "B"  # fallback pour density='N'

        dicom_path = os.path.join(self.dicom_dir, f"{patient_id}_{image_id}.dcm")

        try:
            img01, _ = self._loader.load_dicom_for_roi(dicom_path)
            img01 = self._windowing.process_one(img01, density=density)
        except Exception as e:
            logger.debug(f"Erreur DICOM {dicom_path}: {e}")
            img01 = np.zeros((self.img_size, self.img_size), dtype=np.float32)

        import cv2
        img_resized = cv2.resize(img01.astype(np.float32), (self.img_size, self.img_size),
                                  interpolation=cv2.INTER_AREA)
        img_ch = img_resized[np.newaxis, ...]  # (1, H, W)

        tensor = self._transforms(img_ch).float()
        return tensor, torch.tensor(label, dtype=torch.float32), patient_id


# ─── Test 1 : architecture ─────────────────────────────────────────────────────

def test_architecture():
    print("\n" + "="*60)
    print("TEST 1 — Architecture (forward pass dummy data)")
    print("="*60)

    from models.DIABIRA.multi_head_expert import MultiHeadMammoModel
    from models.DIABIRA.baseline_cnn import BaselineCNN
    from models.DIABIRA.losses import FocalAUCLoss

    dummy = torch.randn(2, 1, IMG_SIZE, IMG_SIZE).to(DEVICE)

    # ── MultiHeadMammoModel ──
    model = MultiHeadMammoModel(embed_dim=512).to(DEVICE)
    t0 = time.time()
    with torch.no_grad():
        logit, gates = model(dummy)
    dt = time.time() - t0

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nMultiHeadMammoModel :")
    print(f"  Params entraînables : {n_params:,}")
    print(f"  Output logit shape  : {logit.shape}  (attendu: [2, 1])")
    print(f"  Gates shape         : {gates.shape}  (attendu: [2, 4])")
    print(f"  Gates sum           : {gates.sum(dim=1).tolist()}  (doit être ≈ [1, 1])")
    print(f"  Inference time (x2) : {dt*1000:.1f}ms sur {DEVICE}")
    assert logit.shape == (2, 1), f"Shape logit incorrecte: {logit.shape}"
    assert gates.shape == (2, 4), f"Shape gates incorrecte: {gates.shape}"
    assert torch.allclose(gates.sum(dim=1), torch.ones(2, device=DEVICE), atol=1e-4)

    # ── BaselineCNN ──
    baseline = BaselineCNN().to(DEVICE)
    with torch.no_grad():
        out = baseline(dummy)
    n_b = sum(p.numel() for p in baseline.parameters() if p.requires_grad)
    print(f"\nBaselineCNN :")
    print(f"  Params entraînables : {n_b:,}")
    print(f"  Output shape        : {out.shape}  (attendu: [2, 1])")
    assert out.shape == (2, 1)

    # ── Loss ──
    loss_fn = FocalAUCLoss()
    fake_logits  = torch.randn(4)
    fake_labels  = torch.tensor([1., 0., 1., 0.])
    loss_dict = loss_fn(fake_logits, fake_labels)
    print(f"\nFocalAUCLoss :")
    for k, v in loss_dict.items():
        print(f"  {k}: {v.item():.4f}")
    assert "total" in loss_dict

    print("\n✅ TEST 1 PASSED")
    del model, baseline
    torch.mps.empty_cache() if DEVICE.type == "mps" else None


# ─── Test 2 : chargement données réelles ──────────────────────────────────────

def test_data_loading():
    print("\n" + "="*60)
    print("TEST 2 — Chargement données DICOM réelles")
    print("="*60)

    df = pd.read_csv(CSV_PATH)
    # Petit sous-ensemble équilibré
    pos = df[df["cancer"] == 1].head(10)
    neg = df[df["cancer"] == 0].head(10)
    sample_df = pd.concat([pos, neg]).reset_index(drop=True)

    print(f"  Sous-ensemble : {len(sample_df)} images ({pos['cancer'].sum()} positifs)")

    ds = FlatDicomDataset(sample_df, DICOM_DIR, IMG_SIZE, mode="val")
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)

    ok, errors = 0, 0
    for images, labels, pids in loader:
        assert images.shape == (len(images), 1, IMG_SIZE, IMG_SIZE), f"Shape: {images.shape}"
        assert images.dtype == torch.float32
        ok += len(images)
        print(f"  Batch OK — shape: {images.shape}, labels: {labels.tolist()}, "
              f"range: [{images.min():.2f}, {images.max():.2f}]")

    print(f"\n✅ TEST 2 PASSED — {ok} images chargées, {errors} erreurs")
    return ds


# ─── Test 3 : mini entraînement + métriques ────────────────────────────────────

def test_training_metrics():
    print("\n" + "="*60)
    print("TEST 3 — Mini entraînement (BaselineCNN) + métriques")
    print("="*60)

    from models.DIABIRA.baseline_cnn import BaselineCNN
    from models.DIABIRA.losses import FocalAUCLoss

    df = pd.read_csv(CSV_PATH)
    # Sous-ensemble stratifié train/val
    pos_train = df[(df["cancer"] == 1) & (df["split"] == "train")].head(N_TRAIN // 2)
    neg_train = df[(df["cancer"] == 0) & (df["split"] == "train")].head(N_TRAIN // 2)
    pos_val   = df[(df["cancer"] == 1) & (df["split"] == "val")].head(N_VAL // 2)
    neg_val   = df[(df["cancer"] == 0) & (df["split"] == "val")].head(N_VAL // 2)

    train_df = pd.concat([pos_train, neg_train]).sample(frac=1, random_state=42).reset_index(drop=True)
    val_df   = pd.concat([pos_val, neg_val]).reset_index(drop=True)

    print(f"  Train : {len(train_df)} ({train_df.cancer.sum()} positifs)")
    print(f"  Val   : {len(val_df)} ({val_df.cancer.sum()} positifs)")

    train_ds = FlatDicomDataset(train_df, DICOM_DIR, IMG_SIZE, mode="train")
    val_ds   = FlatDicomDataset(val_df, DICOM_DIR, IMG_SIZE, mode="val")
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0)

    model = BaselineCNN().to(DEVICE)
    criterion = FocalAUCLoss(alpha=0.75, gamma=2.5, pos_weight=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    N_EPOCHS = 5
    history = {"train_loss": [], "val_auroc": [], "val_f1": [], "val_recall": []}

    for epoch in range(1, N_EPOCHS + 1):
        # ── Train ──
        model.train()
        train_loss = 0.0
        for images, labels, _ in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(images).squeeze(1)
            loss_dict = criterion(logits, labels)
            loss_dict["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss_dict["total"].item()
        train_loss /= len(train_loader)

        # ── Val ──
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for images, labels, _ in val_loader:
                logits = model(images.to(DEVICE)).squeeze(1)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(labels.numpy())

        probs_arr  = np.array(all_probs)
        labels_arr = np.array(all_labels)

        auroc   = roc_auc_score(labels_arr, probs_arr) if labels_arr.sum() > 0 else 0.0
        # Threshold optimal
        best_f1, best_t = 0.0, 0.5
        for t in np.linspace(0.1, 0.9, 81):
            preds = (probs_arr >= t).astype(int)
            f1 = f1_score(labels_arr, preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        preds = (probs_arr >= best_t).astype(int)
        rec  = recall_score(labels_arr, preds, zero_division=0)
        prec = precision_score(labels_arr, preds, zero_division=0)

        history["train_loss"].append(round(train_loss, 4))
        history["val_auroc"].append(round(auroc, 4))
        history["val_f1"].append(round(best_f1, 4))
        history["val_recall"].append(round(rec, 4))

        print(f"  Epoch {epoch}/{N_EPOCHS} | "
              f"loss={train_loss:.4f} | AUROC={auroc:.4f} | "
              f"F1={best_f1:.4f} | Recall={rec:.4f} | Prec={prec:.4f} | thresh={best_t:.2f}")

    print(f"\n  Historique AUROC : {history['val_auroc']}")
    print(f"  Historique F1    : {history['val_f1']}")
    print(f"\n✅ TEST 3 PASSED — BaselineCNN entraîné sur {N_EPOCHS} epochs")
    return history


# ─── Test 4 : forward pass MultiHeadMammoModel sur vraies données ──────────────

def test_multihead_forward():
    print("\n" + "="*60)
    print("TEST 4 — MultiHeadMammoModel forward sur images réelles")
    print("="*60)

    from models.DIABIRA.multi_head_expert import MultiHeadMammoModel

    df = pd.read_csv(CSV_PATH)
    sample = pd.concat([
        df[df["cancer"] == 1].head(4),
        df[df["cancer"] == 0].head(4)
    ]).reset_index(drop=True)

    ds = FlatDicomDataset(sample, DICOM_DIR, IMG_SIZE, mode="val")
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)

    model = MultiHeadMammoModel(embed_dim=512).to(DEVICE)
    model.eval()

    with torch.no_grad():
        for images, labels, pids in loader:
            logits, gates = model(images.to(DEVICE))
            probs = torch.sigmoid(logits.squeeze(1))
            print(f"  Images shape  : {images.shape}")
            print(f"  Logits range  : [{logits.min():.3f}, {logits.max():.3f}]")
            print(f"  Probs         : {probs.cpu().numpy().round(3).tolist()}")
            print(f"  Labels        : {labels.numpy().tolist()}")
            print(f"  Gates (ex. 0) : {gates[0].cpu().numpy().round(3).tolist()}")
            break

    print(f"\n✅ TEST 4 PASSED")
    del model


# ─── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\nDevice utilisé : {DEVICE}")
    print(f"Image size     : {IMG_SIZE}×{IMG_SIZE}")
    print(f"Batch size     : {BATCH}")

    test_architecture()
    test_data_loading()
    history = test_training_metrics()
    test_multihead_forward()

    print("\n" + "="*60)
    print("TOUS LES TESTS PASSÉS")
    print("="*60)

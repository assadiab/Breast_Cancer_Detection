"""
train_compare.py — Comparaison dans les mêmes conditions que le rapport.

Stratégie : feature extraction + fusion training
  1. Backbones gelés → extraire embeddings 512-dim une seule fois (~2 min)
  2. Entraîner seulement ExpertAwareFusion + classifier sur ces embeddings
  3. Reporter ROC-AUC, PR-AUC, F1, Recall, Precision vs anciens résultats

Lancer : pixi run python train_compare.py
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
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# ─── Config ────────────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(__file__)
DICOM_DIR = "/Volumes/T9_Assa/Cours/M2/S1/Intelligence Artificielle/Projets/Projet Kaggle/dicom_output/train"
CSV_PATH  = "/Volumes/T9_Assa/Cours/M2/S1/Intelligence Artificielle/Projets/Projet Kaggle/old/testing/prerocessed/df_final.csv"
EMBED_DIR = "/tmp/mammo_embeddings"
IMG_SIZE  = 224
BATCH_EXTRACT = 16
BATCH_TRAIN   = 32
N_EPOCHS      = 30
LR            = 1e-3
PATIENCE      = 7
DEVICE        = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

sys.path.insert(0, ROOT)

# ─── Résultats de référence (rapport) ──────────────────────────────────────────
REFERENCE = {
    "EfficientNet-B0":        {"auroc": 0.63, "prauc": 0.14, "f1": 0.13, "recall": 0.18, "precision": 0.24},
    "ConvNeXt-Base":          {"auroc": 0.62, "prauc": 0.15, "f1": 0.20, "recall": 0.26, "precision": 0.15},
    "ResNet-50 + Meta":       {"auroc": 0.59, "prauc": 0.13, "f1": 0.19, "recall": 0.21, "precision": 0.17},
    "Multi-head v1 (rapport)":{"auroc": 0.66, "prauc": 0.15, "f1": 0.22, "recall": 0.23, "precision": 0.20},
}


# ─── Dataset pour chargement DICOM flat ────────────────────────────────────────

class FlatDicomDataset(Dataset):
    def __init__(self, df, dicom_dir, img_size):
        self.df = df.reset_index(drop=True)
        self.dicom_dir = dicom_dir
        self.img_size = img_size
        from core.loader import Loader
        from preprocess.windowing import Windowing
        class _Cfg:
            config = {"paths": {"csv": CSV_PATH, "images": dicom_dir, "out": None}}
            roi_config = {"min_area_px": 12000, "morpho_disk": 5, "use_convex_hull": True,
                          "inset_mm_y": 2.0, "inset_mm_x": 0.8,
                          "margins_mm": {"CC": {"x": 7.0, "y": 6.5}, "MLO": {"x": 9.0, "y": 6.5}},
                          "norm_mode": "soft_tanh", "soft_tanh_k": 3.0}
            images_dir = dicom_dir
        self._loader = Loader(_Cfg())
        self._wind = Windowing()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid, iid = int(row["patient_id"]), int(row["image_id"])
        density = str(row.get("density", "B"))
        if density not in ("A","B","C","D"):
            density = "B"
        label = float(row.get("cancer", 0))
        path = os.path.join(self.dicom_dir, f"{pid}_{iid}.dcm")
        try:
            import cv2
            img01, _ = self._loader.load_dicom_for_roi(path)
            img01 = self._wind.process_one(img01, density=density)
            img = cv2.resize(img01.astype(np.float32), (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
            if img.max() > img.min():
                img = (img - img.mean()) / (img.std() + 1e-6)
        except Exception:
            img = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        tensor = torch.from_numpy(img[np.newaxis]).float()
        return tensor, torch.tensor(label).float(), pid


# ─── Dataset pour embeddings pré-extraits ──────────────────────────────────────

class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels, patient_ids):
        self.embeddings = torch.from_numpy(embeddings).float()
        self.labels = torch.from_numpy(labels).float()
        self.pids = patient_ids

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx], self.pids[idx]


# ─── Fusion seule (entraînable) ────────────────────────────────────────────────

class FusionOnly(nn.Module):
    """ExpertAwareFusion standalone — s'entraîne sur embeddings pré-extraits."""
    def __init__(self, embed_dim=512, num_experts=4, hidden_dim=256):
        super().__init__()
        from models.DIABIRA.multi_head_expert import ExpertAwareFusion
        self.fusion = ExpertAwareFusion(embed_dim=embed_dim, num_experts=num_experts, hidden_dim=hidden_dim)

    def forward(self, x):  # x: (B, 4, 512)
        return self.fusion(x)


# ─── Étape 1 : extraction embeddings ──────────────────────────────────────────

def extract_embeddings(df, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    embed_path = os.path.join(save_dir, "embeddings.npy")
    label_path = os.path.join(save_dir, "labels.npy")
    pid_path   = os.path.join(save_dir, "pids.npy")

    if all(os.path.exists(p) for p in [embed_path, label_path, pid_path]):
        logger.info("Embeddings déjà extraits — chargement depuis cache")
        return (np.load(embed_path), np.load(label_path), np.load(pid_path))

    logger.info(f"Extraction embeddings sur {len(df)} images avec {DEVICE}...")
    import logging as _l; _l.disable(_l.WARNING)
    from models.DIABIRA.multi_head_expert import MultiHeadMammoModel
    _l.disable(_l.NOTSET)

    model = MultiHeadMammoModel(embed_dim=512).to(DEVICE)
    model.eval()

    ds = FlatDicomDataset(df, DICOM_DIR, IMG_SIZE)
    loader = DataLoader(ds, batch_size=BATCH_EXTRACT, shuffle=False, num_workers=0)

    all_embeds, all_labels, all_pids = [], [], []
    t0 = time.time()

    with torch.no_grad():
        for i, (images, labels, pids) in enumerate(loader):
            images = images.to(DEVICE)
            # Extraire les 4 embeddings séparément (avant fusion)
            e1 = model.expert1(images).cpu().numpy()
            e2 = model.expert2(images).cpu().numpy()
            e3 = model.expert3(images).cpu().numpy()
            e4 = model.expert4(images).cpu().numpy()
            embed = np.stack([e1, e2, e3, e4], axis=1)  # (B, 4, 512)
            all_embeds.append(embed)
            all_labels.append(labels.numpy())
            all_pids.append(pids.numpy() if isinstance(pids, torch.Tensor) else np.array(pids))

            if (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                pct = (i + 1) / len(loader)
                eta = elapsed / pct * (1 - pct)
                logger.info(f"  {i+1}/{len(loader)} batches | {elapsed:.0f}s écoulés | ETA {eta:.0f}s")

    embeddings = np.concatenate(all_embeds)    # (N, 4, 512)
    labels_arr = np.concatenate(all_labels)    # (N,)
    pids_arr   = np.concatenate(all_pids)      # (N,)

    np.save(embed_path, embeddings)
    np.save(label_path, labels_arr)
    np.save(pid_path, pids_arr)

    elapsed = time.time() - t0
    logger.info(f"Embeddings extraits en {elapsed:.0f}s — shape: {embeddings.shape}")
    del model
    return embeddings, labels_arr, pids_arr


# ─── Étape 2 : split patient-wise ─────────────────────────────────────────────

def patient_split(df, embeddings, labels, pids, val_ratio=0.20):
    all_pids = df["patient_id"].unique()
    pat_labels = df.groupby("patient_id")["cancer"].max().reindex(all_pids).values
    train_pids, val_pids = train_test_split(
        all_pids, test_size=val_ratio, stratify=pat_labels, random_state=42
    )
    train_mask = np.isin(pids, train_pids)
    val_mask   = np.isin(pids, val_pids)
    return (
        embeddings[train_mask], labels[train_mask], pids[train_mask],
        embeddings[val_mask],   labels[val_mask],   pids[val_mask],
    )


# ─── Étape 3 : entraînement fusion ────────────────────────────────────────────

def train_fusion(train_emb, train_lbl, val_emb, val_lbl, val_pids):
    from models.DIABIRA.losses import FocalAUCLoss

    # Weighted sampler pour gérer le déséquilibre
    pos_weight = (train_lbl == 0).sum() / max((train_lbl == 1).sum(), 1)
    weights = np.where(train_lbl == 1, pos_weight, 1.0)
    sampler = WeightedRandomSampler(torch.from_numpy(weights).float(), len(weights))

    train_ds = EmbeddingDataset(train_emb, train_lbl, np.zeros(len(train_lbl)))
    val_ds   = EmbeddingDataset(val_emb, val_lbl, val_pids)
    train_loader = DataLoader(train_ds, batch_size=BATCH_TRAIN, sampler=sampler, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_TRAIN, shuffle=False, num_workers=0)

    model = FusionOnly().to(DEVICE)
    criterion = FocalAUCLoss(alpha=0.75, gamma=2.5, focal_weight=0.7, auc_weight=0.3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=1e-6)

    best_auroc, best_metrics, no_improve = 0.0, {}, 0
    history = []

    logger.info(f"\nEntraînement fusion — {N_EPOCHS} epochs, lr={LR}, bs={BATCH_TRAIN}, device={DEVICE}")
    logger.info(f"Train: {len(train_ds)} | Val: {len(val_ds)} | pos_weight: {pos_weight:.1f}")

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for embeds, labels, _ in train_loader:
            embeds, labels = embeds.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logit, _ = model(embeds)
            loss = criterion(logit.squeeze(1), labels)["total"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        scheduler.step()

        # ── Évaluation ──
        model.eval()
        all_probs, all_lbls, all_pids_eval = [], [], []
        with torch.no_grad():
            for embeds, labels, pids_b in val_loader:
                logit, _ = model(embeds.to(DEVICE))
                probs = torch.sigmoid(logit.squeeze(1)).cpu().numpy()
                all_probs.extend(probs)
                all_lbls.extend(labels.numpy())
                all_pids_eval.extend(pids_b.numpy())

        probs_arr  = np.array(all_probs)
        labels_arr = np.array(all_lbls)
        pids_eval  = np.array(all_pids_eval)

        # Patient-level aggregation (max pooling)
        unique_p = np.unique(pids_eval)
        pat_probs  = np.array([probs_arr[pids_eval == p].max() for p in unique_p])
        pat_labels = np.array([labels_arr[pids_eval == p].max() for p in unique_p])

        auroc = roc_auc_score(pat_labels, pat_probs) if pat_labels.sum() > 0 else 0.0
        prauc = average_precision_score(pat_labels, pat_probs) if pat_labels.sum() > 0 else 0.0

        # Seuil optimal (max F1)
        best_f1, best_t = 0.0, 0.5
        for t in np.linspace(0.05, 0.95, 91):
            preds = (pat_probs >= t).astype(int)
            f1 = f1_score(pat_labels, preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        preds = (pat_probs >= best_t).astype(int)

        metrics = {
            "auroc":     round(auroc, 4),
            "prauc":     round(prauc, 4),
            "f1":        round(best_f1, 4),
            "recall":    round(recall_score(pat_labels, preds, zero_division=0), 4),
            "precision": round(precision_score(pat_labels, preds, zero_division=0), 4),
            "threshold": round(best_t, 2),
            "train_loss": round(train_loss, 4),
        }
        history.append(metrics)

        logger.info(
            f"Epoch {epoch:02d}/{N_EPOCHS} | loss={train_loss:.4f} | "
            f"AUROC={auroc:.4f} | PR-AUC={prauc:.4f} | F1={best_f1:.4f} | "
            f"Recall={metrics['recall']:.4f} | Prec={metrics['precision']:.4f} | "
            f"thresh={best_t:.2f} | LR={scheduler.get_last_lr()[0]:.2e}"
        )

        if auroc > best_auroc:
            best_auroc = auroc
            best_metrics = metrics.copy()
            best_metrics["epoch"] = epoch
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                logger.info(f"Early stopping epoch {epoch} (best AUROC={best_auroc:.4f})")
                break

    return best_metrics, history


# ─── Tableau comparatif ────────────────────────────────────────────────────────

def print_comparison(new_metrics):
    print("\n" + "="*80)
    print("COMPARAISON — Multi-Head v2 (nouveaux backbones) vs résultats rapport")
    print("="*80)
    header = f"{'Modèle':<30} {'ROC-AUC':>8} {'PR-AUC':>8} {'F1':>7} {'Recall':>8} {'Precision':>10}"
    print(header)
    print("-"*80)
    for name, m in REFERENCE.items():
        flag = " ◀ baseline" if "rapport" in name else ""
        print(f"{name:<30} {m['auroc']:>8.4f} {m['prauc']:>8.4f} {m['f1']:>7.4f} {m['recall']:>8.4f} {m['precision']:>10.4f}{flag}")
    print("-"*80)
    name = f"Multi-head v2 (epoch {new_metrics.get('epoch','?')})"
    m = new_metrics
    print(f"{name:<30} {m['auroc']:>8.4f} {m['prauc']:>8.4f} {m['f1']:>7.4f} {m['recall']:>8.4f} {m['precision']:>10.4f}  ◀ NOUVEAU")
    print("="*80)

    delta_auroc = new_metrics["auroc"] - REFERENCE["Multi-head v1 (rapport)"]["auroc"]
    delta_f1    = new_metrics["f1"]    - REFERENCE["Multi-head v1 (rapport)"]["f1"]
    print(f"\nDelta vs Multi-head v1 : AUROC {delta_auroc:+.4f} | F1 {delta_f1:+.4f}")

    if delta_auroc > 0:
        print(f"✅ Amélioration AUROC de {delta_auroc:+.4f} ({delta_auroc/REFERENCE['Multi-head v1 (rapport)']['auroc']*100:+.1f}%)")
    else:
        print(f"⚠️  Régression AUROC de {delta_auroc:.4f}")


# ─── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info(f"Device: {DEVICE} | Image size: {IMG_SIZE}x{IMG_SIZE}")

    # Charger CSV et filtrer images disponibles
    df = pd.read_csv(CSV_PATH)
    available_keys = {f.replace('.dcm','') for f in os.listdir(DICOM_DIR) if f.endswith('.dcm')}
    df['key'] = df['patient_id'].astype(str) + '_' + df['image_id'].astype(str)
    df = df[df['key'].isin(available_keys)].copy()
    logger.info(f"Dataset : {len(df)} images | {df.cancer.sum()} positifs ({df.cancer.mean()*100:.1f}%) | {df.patient_id.nunique()} patients")

    # Étape 1 : extraire embeddings
    embeddings, labels, pids = extract_embeddings(df, EMBED_DIR)

    # Étape 2 : split patient-wise 80/20
    train_emb, train_lbl, train_pids, val_emb, val_lbl, val_pids = patient_split(df, embeddings, labels, pids)
    logger.info(f"Split — Train: {len(train_lbl)} ({train_lbl.sum():.0f} pos) | Val: {len(val_lbl)} ({val_lbl.sum():.0f} pos)")

    # Étape 3 : entraîner fusion
    best_metrics, history = train_fusion(train_emb, train_lbl, val_emb, val_lbl, val_pids)

    # Résultats
    print_comparison(best_metrics)

    logger.info(f"\nMeilleure epoch : {best_metrics.get('epoch', '?')}")
    logger.info(f"Résultats complets : {best_metrics}")

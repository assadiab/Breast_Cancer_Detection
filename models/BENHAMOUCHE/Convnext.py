#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ConvNeXt+Signals Pipeline – Classification Cancer vs Non-Cancer à partir de mammographies DICOM.

Pipeline :
1. Préparation des DICOM annotés : nettoyage, parsing des .dcm
2. Fusion des CSV d'annotations (image_id -> label cancer / pas cancer) avec les chemins DICOM
   -> création d'un DataFrame final contenant dicom_path + label
   -> ce DataFrame est utilisé pour l'entraînement et la validation (les deux ont des labels)
3. Entraînement ConvNeXt avec cross-validation :
   - train_loader : apprentissage
   - val_loader   : évaluation (AUC, F1, MCC, courbe ROC, matrice de confusion)
   "val" a bien les labels.

   Deux variantes :
   - Variante A : "no-resize"  → pas de resize global, padding dynamique pour le GPU
   - Variante B : "resize512" → images redimensionnées en 512x512

   Les deux variantes :
   - utilisent BCEWithLogitsLoss + pos_weight pour gérer le déséquilibre de classes
   - utilisent WeightedRandomSampler
   - entraînent en k-fold (StratifiedKFold)
   - suivent AUC, F1, MCC, accuracy à chaque epoch
   - utilisent l'early stopping basé sur l'AUC de la validation
   - produisent courbe ROC + matrice de confusion sur la validation du fold

4. Sauvegarde du modèle (.pth)

5. Inférence sur un dossier d'images SANS labels (utilisé comme "test" final) :
   -> génération d'un CSV : file, prob_cancer, pred_class
   -> pas de métriques (pas de ground truth pour ces images)
"""

# ======================================================================
# IMPORTS
# ======================================================================

import os
import gc
import re
import shutil
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms as T

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score,
    matthews_corrcoef,
    accuracy_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)

import pydicom
from PIL import Image


# ======================================================================
# CONFIG
# ======================================================================

class Config:
    # Dossier contenant les DICOM annotés (train+val)
    LABELED_DATA_DIR = "/content/rsna_data/extracted"

    # Dossier contenant les DICOM sans labels (inférence finale)
    INFER_DATA_DIR   = "/content/rsna_data_unlabeled"

    # Chemins de sortie
    MODEL_PATH       = "/content/model_convnext_v2_best.pth"
    PREDICTION_CSV   = "/content/predictions_unlabeled.csv"

    # Paramètres data
    IMG_SIZE   = 512          # pour la variante "resize512"
    MAX_SIDE   = 1024         # limite max du plus grand côté avant downscale (variante no-resize)

    # Entraînement
    BATCH_SIZE    = 4
    NUM_WORKERS   = 2
    KFOLDS        = 5
    EPOCHS        = 15
    LR            = 1e-4
    WEIGHT_DECAY  = 1e-5
    PATIENCE      = 4         # patience pour l'early stopping
    SEED          = 42

    # Seuil décisionnel utilisé pour l'inférence sur les images sans labels
    THRESHOLD_INFER = 0.7

    # Device / AMP
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    USE_AMP = torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True


# Reproductibilité
torch.manual_seed(Config.SEED)
np.random.seed(Config.SEED)


# ======================================================================
# 1. OUTILS (DOWNLOAD / EXTRACTION)
# ======================================================================

def download_via_aria2(url: str, out_dir: str, out_name: str):
    """
    Construit une commande aria2c pour télécharger rapidement un gros zip/tar.
    """
    os.makedirs(out_dir, exist_ok=True)
    cmd = (
        f'aria2c -x16 -s16 -c --summary-interval=5 --console-log-level=warn '
        f'"{url}" -d "{out_dir}" -o "{out_name}"'
    )
    print("Commande suggérée :")
    print(cmd)


def extract_archive_if_needed(data_dir: str):
    """
    Cherche un zip/tar dans data_dir et l'extrait.
    """
    p = next(Path(data_dir).glob("*"), None)
    if p is None:
        print("Aucun fichier trouvé dans", data_dir)
        return

    if p.suffix == ".zip":
        print(f"Extraction ZIP: {p.name}")
        shutil.unpack_archive(str(p), extract_dir=data_dir)
    elif p.suffix in [".tar", ".gz", ".tgz"]:
        print(f"Extraction TAR: {p.name}")
        os.system(f'tar -xf "{p}" -C "{data_dir}"')
    else:
        print("Données déjà extraites ou format non-archive:", p.name)

    print(f"Contenu de {data_dir}:")
    for x in Path(data_dir).iterdir():
        print("   -", x)


# ======================================================================
# 2. NETTOYAGE DICOM + LISTAGE
# ======================================================================

def clean_macos_artifacts(root: Path):
    """
    Supprime les répertoires '__MACOSX' et les fichiers '._xxx.dcm' (artefacts macOS).
    """
    for bad_dir in root.rglob("__MACOSX"):
        print("Removing:", bad_dir)
        shutil.rmtree(bad_dir, ignore_errors=True)

    for bad_file in root.rglob("._*.dcm"):
        print("Removing artifact:", bad_file)
        try:
            bad_file.unlink()
        except Exception:
            pass


def list_dicoms(root: Path):
    """
    Retourne tous les chemins .dcm "propres" dans root.
    """
    dcm_paths = sorted([
        p for p in root.rglob("*.dcm")
        if "__MACOSX" not in str(p) and not p.name.startswith("._")
    ])
    print(f"Trouvé {len(dcm_paths)} DICOM sous {root}")
    if len(dcm_paths) > 0:
        print("   Exemples :", [p.name for p in dcm_paths[:5]])
    return dcm_paths


def probe_dicom_metadata(paths, n_show=3):
    """
    Affiche quelques métadonnées simples (PatientID, Modality, SeriesDescription)
    sans charger les pixels.
    """
    print("\nSonde DICOM :")
    for p in paths[:n_show]:
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
            pid = getattr(ds, "PatientID", "NA")
            mod = getattr(ds, "Modality", "NA")
            series = getattr(ds, "SeriesDescription", "NA")
            print(f" • {p.name:35s} | PatientID={pid} | Modality={mod} | Series={series}")
        except Exception as e:
            print(f"Erreur lecture {p.name} :", e)


# ======================================================================
# 3. TEST RAPIDE DE LECTURE DES PIXELS
# ======================================================================

def try_read_dicom_pixel(path: Path):
    """
    Essaie d'ouvrir le pixel_array avec pydicom puis SimpleITK.
    Retourne "pydicom", "sitk" ou None.
    """
    try:
        dcm = pydicom.dcmread(str(path), force=True)
        _ = dcm.pixelArray  # (certaines versions exposent pixelArray)
        return "pydicom"
    except Exception:
        pass

    try:
        import SimpleITK as sitk
        img = sitk.ReadImage(str(path))
        _ = sitk.GetArrayFromImage(img)
        return "sitk"
    except Exception:
        pass

    return None


def quick_read_report(dcm_list, sample_size=50):
    """
    Essaie de lire les pixels sur un sous-échantillon (~50 DICOM) pour estimer le taux de succès.
    """
    sample = dcm_list[:min(sample_size, len(dcm_list))]
    ok_pydi = 0
    ok_sitk = 0
    bad = 0
    for p in sample:
        m = try_read_dicom_pixel(p)
        if m == "pydicom":
            ok_pydi += 1
        elif m == "sitk":
            ok_sitk += 1
        else:
            bad += 1

    print(f"Lecture OK (pydicom)   : {ok_pydi}")
    print(f"Lecture OK (SimpleITK): {ok_sitk}")
    print(f"Echecs                : {bad} / {len(sample)}")


# ======================================================================
# 4. FUSION CSV <-> DICOMS (CONSTRUCTION DU DATAFRAME FINAL)
# ======================================================================

def merge_labels_with_dicoms(df_X, df_y, dicom_paths):
    """
    Construit une table finale exploitable par PyTorch.

    Entrées :
    - df_X : doit contenir au moins 'image_id'
    - df_y : doit contenir 'cancer' ou 'label'
    - dicom_paths : liste des chemins .dcm

    Sortie :
    DataFrame avec colonnes :
      image_id | dicom_path | label

    Hypothèse :
    l'ID image est récupérable dans le nom du DICOM :
    ex: '..._123456.dcm' -> image_id = '123456'
    """
    # Joint les infos X/y pour associer image_id -> label
    df_labels = pd.concat([df_X, df_y], axis=1)

    # Harmonisation du nom de la colonne de label
    if "cancer" in df_labels.columns and "label" not in df_labels.columns:
        df_labels = df_labels.rename(columns={"cancer": "label"})

    if "image_id" not in df_labels.columns:
        raise ValueError("df_X doit contenir une colonne 'image_id'.")

    df_labels["image_id"] = df_labels["image_id"].astype(str)

    # Associe chaque DICOM à un image_id (regex sur le nom du fichier)
    df_dicoms = pd.DataFrame({
        "image_id": [
            re.findall(r'_(\d+)', p.stem)[0] if len(re.findall(r'_(\d+)', p.stem)) > 0 else p.stem
            for p in dicom_paths
        ],
        "dicom_path": [str(p) for p in dicom_paths],
    })

    # Jointure pour ne garder que les DICOM qui ont un label connu
    df_final = df_dicoms.merge(df_labels, on="image_id", how="inner")

    print("\nAperçu fusion DICOMs + labels :")
    print(df_final.head())
    return df_final.reset_index(drop=True)


# ======================================================================
# 5. DATASETS PYTORCH
# ======================================================================

class DICOMDataset(Dataset):
    """
    Dataset pour l'entraînement et la validation.
    Chaque item:
        - lecture du DICOM
        - normalisation de l'image
        - application des transforms
        - renvoi (tensor_image, label_binaire)
    """

    def __init__(self, dataframe, transform=None):
        self.data = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        dcm_path = row["dicom_path"]
        label = torch.tensor(row["label"], dtype=torch.float32)

        try:
            ds = pydicom.dcmread(dcm_path, force=True)
            img = ds.pixel_array.astype(np.float32)

            # Normalisation -> [0,1], puis conversion en uint8 [0,255]
            img -= img.min()
            img /= (img.max() + 1e-6)
            img = (img * 255).astype(np.uint8)

        except Exception as e:
            print(f"Erreur lecture {dcm_path}: {e}")
            img = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.uint8)

        pil_img = Image.fromarray(img)

        if self.transform:
            pil_img = self.transform(pil_img)

        # Exemples sorties:
        # - resize512 : Tensor [1,512,512]
        # - no-resize : Tensor [1,H,W] (taille variable avant padding)
        return pil_img, label


class DICOMUnlabeledDataset(Dataset):
    """
    Dataset pour l'inférence finale (pas de label).
    Retourne (image_tensor, filename).
    """

    def __init__(self, paths, transform=None):
        self.paths = list(paths)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            ds = pydicom.dcmread(str(path), force=True)
            img = ds.pixel_array.astype(np.float32)
            img = (img - img.min()) / (img.max() - img.min() + 1e-6)
        except Exception:
            img = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

        pil_img = Image.fromarray((img * 255).astype(np.uint8)).convert("L")

        if self.transform:
            pil_img = self.transform(pil_img)

        return pil_img, str(Path(path).name)


# ======================================================================
# 6. TRANSFORMS
# ======================================================================

def get_transforms_resize512():
    """
    Variante "resize512":
    - Resize (512,512)
    - Grayscale (1 canal)
    - Augmentation légère sur le train
    - Normalisation mean=0.5,std=0.5
    """
    train_tfms = T.Compose([
        T.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        T.Grayscale(num_output_channels=1),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])

    val_tfms = T.Compose([
        T.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        T.Grayscale(num_output_channels=1),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])

    return train_tfms, val_tfms


def get_transforms_noresize():
    """
    Variante "no-resize":
    - pas de resize global
    - grayscale + normalisation
    - la gestion de la géométrie + padding est faite dans le collate
    """
    train_tfms = T.Compose([
        T.Grayscale(num_output_channels=1),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])

    val_tfms = T.Compose([
        T.Grayscale(num_output_channels=1),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])

    return train_tfms, val_tfms


# ======================================================================
# 7. COLLATE POUR "NO-RESIZE"
# ======================================================================

def _downscale_if_needed(img: torch.Tensor, max_side: int = Config.MAX_SIDE) -> torch.Tensor:
    """
    Downscale si l'image est trop grande pour que max(H,W) <= max_side.
    """
    _, h, w = img.shape
    if max(h, w) <= max_side:
        return img

    scale = max_side / float(max(h, w))
    nh, nw = int(h * scale), int(w * scale)
    return F.interpolate(
        img.unsqueeze(0),
        size=(nh, nw),
        mode="bilinear",
        align_corners=False
    ).squeeze(0)


def collate_pad(batch):
    """
    Collate pour la variante "no-resize":
    1. downscale chaque image si nécessaire
    2. padding pour aligner toutes les tailles d'images dans le batch
    """
    imgs, labels = zip(*batch)

    imgs = [img if torch.is_tensor(img) else T.ToTensor()(img) for img in imgs]
    imgs = [_downscale_if_needed(im) for im in imgs]

    max_h = max(im.shape[1] for im in imgs)
    max_w = max(im.shape[2] for im in imgs)
    padded = []
    for im in imgs:
        _, h, w = im.shape
        pad_h = max_h - h
        pad_w = max_w - w
        padded.append(
            F.pad(im, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
        )

    imgs_tensor = torch.stack(padded, 0)  # [B,1,Hmax,Wmax]
    labels_tensor = torch.tensor(labels, dtype=torch.float32)

    return imgs_tensor, labels_tensor


# ======================================================================
# 8. MODÈLES CONVNEXT
# ======================================================================

def create_convnext_model_freeze_all():
    """
    Variante "no-resize":
    - ConvNeXt-Base pré-entraîné ImageNet
    - gèle les features
    - tête MLP binaire custom
    """
    base = models.convnext_base(weights="IMAGENET1K_V1")

    for _, p in base.features.named_parameters():
        p.requires_grad = False

    n_features = base.classifier[2].in_features
    base.classifier = nn.Sequential(
        nn.Flatten(),
        nn.LayerNorm((n_features,)),
        nn.Linear(n_features, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 1),
    )

    return base.to(Config.DEVICE)


def create_convnext_model_partial_unfreeze():
    """
    Variante "resize512":
    - ConvNeXt-Base pré-entraîné
    - dégel sélectif des derniers blocs (7 et 8)
    """
    base = models.convnext_base(weights="IMAGENET1K_V1")

    for name, param in base.features.named_parameters():
        if "7" in name or "8" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    n_features = base.classifier[2].in_features
    base.classifier = nn.Sequential(
        nn.Flatten(),
        nn.LayerNorm((n_features,)),
        nn.Linear(n_features, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 1),
    )

    return base.to(Config.DEVICE)


# ======================================================================
# 9. METRIQUES + VISUALISATIONS (ROC, CONFUSION MATRIX)
# ======================================================================

def evaluate_model(y_true, y_prob, threshold=0.5):
    """
    Calcule f1_micro, f1_macro, MCC, accuracy, AUC pour un seuil donné.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "accuracy": accuracy_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_prob),
        "threshold": float(threshold),
    }


def find_best_threshold(y_true, y_prob, lo=0.1, hi=0.9, steps=17):
    """
    Cherche le seuil qui maximise le F1 macro (variante resize512).
    """
    best_t = 0.5
    best_f1 = -1.0
    for t in np.linspace(lo, hi, steps):
        y_pred = (y_prob >= t).astype(int)
        f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
        if f1m > best_f1:
            best_f1, best_t = f1m, t
    return float(best_t)


def find_best_threshold_for_f1(y_true, y_prob, lo=0.05, hi=0.95, steps=19):
    """
    Cherche le seuil qui maximise le F1 macro (variante no-resize).
    """
    best_t = 0.5
    best_f1 = -1.0
    for t in np.linspace(lo, hi, steps):
        y_pred = (y_prob >= t).astype(int)
        f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
        if f1m > best_f1:
            best_f1, best_t = f1m, t
    return float(best_t)


def plot_eval_details(y_true, y_prob, threshold, title_prefix="ConvNeXt+Signals"):
    """
    Affiche :
    - matrice de confusion (au meilleur seuil)
    - courbe ROC (TPR/FPR + AUC)
    sur la validation du fold.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    # Matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4,4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cbar=False,
        xticklabels=["No Cancer","Cancer"],
        yticklabels=["No Cancer","Cancer"]
    )
    plt.xlabel("Prédit")
    plt.ylabel("Réel")
    plt.title(f"{title_prefix} - Matrice de confusion (thr={threshold:.2f})")
    plt.tight_layout()
    plt.show()

    # Courbe ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"AUC = {auc_val:.3f}")
    plt.plot([0,1],[0,1],"r--")
    plt.xlabel("FPR (False Positive Rate)")
    plt.ylabel("TPR (True Positive Rate)")
    plt.title(f"{title_prefix} - Courbe ROC")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


# ======================================================================
# 10. ENTRAÎNEMENT D'UN FOLD (VARIANTE resize512)
# ======================================================================

def train_one_fold_resize512(
    fold,
    train_loader,
    val_loader,
    pos_weight_tensor
):
    """
    Entraînement pour la variante "resize512":
    - images déjà redimensionnées en [1,512,512]
    - duplication du canal -> [3,512,512] pour ConvNeXt
    - fine-tuning partiel (dégel sélectif)
    """
    model = create_convnext_model_partial_unfreeze()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.LR,
        weight_decay=Config.WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    scaler = torch.cuda.amp.GradScaler(enabled=Config.USE_AMP)

    best_auc = 0.0
    patience_counter = 0
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    for epoch in range(Config.EPOCHS):
        print(f"\n[Fold {fold}] Epoch {epoch+1}/{Config.EPOCHS}")
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for imgs, labels in tqdm(train_loader, desc=f"Train Fold {fold}", leave=False):
            imgs = imgs.to(Config.DEVICE, non_blocking=True)               # [B,1,H,W]
            labels = labels.unsqueeze(1).to(Config.DEVICE, non_blocking=True)  # [B,1]

            imgs = imgs.repeat(1, 3, 1, 1)  # ConvNeXt attend 3 canaux

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=Config.USE_AMP):
                outputs = model(imgs)        # [B,1] logits
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total

        # Validation (val contient les labels)
        model.eval()
        val_loss = 0.0
        val_probs = []
        val_trues = []

        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Val Fold {fold}", leave=False):
                imgs = imgs.to(Config.DEVICE, non_blocking=True)
                labels = labels.unsqueeze(1).to(Config.DEVICE, non_blocking=True)
                imgs = imgs.repeat(1, 3, 1, 1)

                with torch.cuda.amp.autocast(enabled=Config.USE_AMP):
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)

                val_loss += loss.item()

                probs = torch.sigmoid(outputs).squeeze(1).detach().cpu().numpy()
                val_probs.extend(probs.tolist())
                val_trues.extend(labels.squeeze(1).detach().cpu().numpy().tolist())

        val_loss /= len(val_loader)
        val_bin05 = (np.array(val_probs) >= 0.5).astype(int)
        val_acc = accuracy_score(val_trues, val_bin05)
        val_auc = roc_auc_score(val_trues, val_probs)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"TrainLoss={train_loss:.4f} | ValLoss={val_loss:.4f} | "
            f"TrainAcc={train_acc:.3f} | ValAcc(0.5)={val_acc:.3f} | AUC={val_auc:.3f}"
        )

        # Early stopping sur l'AUC de la validation
        if val_auc > best_auc + 1e-6:
            best_auc = val_auc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered (fold {fold}).")
                break

        torch.cuda.empty_cache()

    # Métriques finales sur la validation du fold
    best_th = find_best_threshold(val_trues, val_probs)
    metrics_best = evaluate_model(val_trues, val_probs, threshold=best_th)

    print(f"\n=== ConvNeXt+Signals — Fold {fold} (seuil F1-opt: {metrics_best['threshold']:.2f}) ===")
    print(metrics_best)
    print("\n--- Info @ seuil 0.50 ---")
    print(evaluate_model(val_trues, val_probs, threshold=0.5))

    # Visualisation finale : ROC + matrice de confusion
    plot_eval_details(val_trues, val_probs, threshold=best_th,
                      title_prefix=f"Fold {fold} (resize512)")

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return history, metrics_best


# ======================================================================
# 11. ENTRAÎNEMENT D'UN FOLD (VARIANTE no-resize)
# ======================================================================

def train_one_fold_noresize(
    fold,
    train_loader,
    val_loader,
    pos_weight_tensor
):
    """
    Variante "no-resize":
    - conserve la géométrie native des images
    - collate_pad() gère le downscale + padding pour batcher proprement
    - backbone ConvNeXt gelé, utilisé comme extracteur de features
    """
    model = create_convnext_model_freeze_all()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.LR,
        weight_decay=Config.WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    scaler = torch.cuda.amp.GradScaler(enabled=Config.USE_AMP)

    best_auc = 0.0
    patience_counter = 0
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    for epoch in range(Config.EPOCHS):
        print(f"\n[Fold {fold}] Epoch {epoch+1}/{Config.EPOCHS} (NO-RESIZE)")
        model.train()
        run_loss = 0.0
        correct = 0
        total = 0

        for imgs, labels in tqdm(train_loader, desc=f"Train {fold}", leave=False):
            imgs = imgs.to(Config.DEVICE)
            labels = labels.unsqueeze(1).to(Config.DEVICE)
            imgs = imgs.repeat(1, 3, 1, 1)  # [B,1,H,W] -> [B,3,H,W]

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=Config.USE_AMP):
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            run_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = run_loss / len(train_loader)
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        val_probs = []
        val_trues = []

        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Val {fold}", leave=False):
                imgs = imgs.to(Config.DEVICE)
                labels = labels.unsqueeze(1).to(Config.DEVICE)
                imgs = imgs.repeat(1, 3, 1, 1)

                with torch.cuda.amp.autocast(enabled=Config.USE_AMP):
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)

                val_loss += loss.item()
                probs = torch.sigmoid(outputs).squeeze(1).cpu().numpy()
                val_probs.extend(np.atleast_1d(probs).tolist())
                val_trues.extend(labels.squeeze(1).cpu().numpy().tolist())

        val_loss /= len(val_loader)
        val_acc = accuracy_score(val_trues, (np.array(val_probs) >= 0.5).astype(int))
        val_auc = roc_auc_score(val_trues, val_probs)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Loss={train_loss:.4f}/{val_loss:.4f} | "
            f"Acc={train_acc:.3f}/{val_acc:.3f} | AUC={val_auc:.3f}"
        )

        # Early stopping basé sur l'AUC validation
        if val_auc > best_auc + 1e-4:
            best_auc = val_auc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print("Early stopping (no AUC gain).")
                break

        torch.cuda.empty_cache()

    # Choix du meilleur seuil pour le F1 macro sur la validation
    best_t = find_best_threshold_for_f1(val_trues, val_probs)
    metrics_best = evaluate_model(val_trues, val_probs, threshold=best_t)
    print(f"\nFold {fold} done — Best F1 @ {best_t:.2f}: {metrics_best}")

    # Visualisation ROC + matrice de confusion
    plot_eval_details(val_trues, val_probs, threshold=best_t,
                      title_prefix=f"Fold {fold} (no-resize)")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return history, metrics_best


# ======================================================================
# 12. COURBES TRAIN / VAL
# ======================================================================

def plot_training_curves(history, title_prefix="ConvNeXt+Signals"):
    """
    Affiche loss train/val + accuracy train/val par epoch.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], "--o", label="Train")
    plt.plot(epochs, history["val_loss"], "-o", label="Val")
    plt.title(f"{title_prefix} — Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], "--o", label="Train Acc")
    plt.plot(epochs, history["val_acc"], "-o", label="Val Acc (0.5)")
    plt.title(f"{title_prefix} — Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()


# ======================================================================
# 13. CROSS-VALIDATION STRATIFIÉE
# ======================================================================

def run_cv_resize512(df_final):
    """
    Cross-validation k-fold pour la variante "resize512".
    """
    skf = StratifiedKFold(
        n_splits=Config.KFOLDS,
        shuffle=True,
        random_state=Config.SEED
    )

    train_tfms, val_tfms = get_transforms_resize512()

    all_metrics = []
    last_history = None

    for fold, (train_idx, val_idx) in enumerate(skf.split(df_final, df_final["label"])):
        train_df = df_final.iloc[train_idx].reset_index(drop=True)
        val_df   = df_final.iloc[val_idx].reset_index(drop=True)

        # Stats de classe (pour le pos_weight)
        counts = Counter(train_df["label"])
        n_pos = counts.get(1, 0)
        n_neg = counts.get(0, 0)
        print(f"\n[Fold {fold}] distribution train -> 0:{n_neg} | 1:{n_pos}")

        pos_weight_value = float(n_neg / max(1, n_pos)) if n_pos > 0 else 1.0
        pos_weight_tensor = torch.tensor(
            [pos_weight_value],
            device=Config.DEVICE,
            dtype=torch.float32
        )
        print(f"[Fold {fold}] pos_weight (BCE) = {pos_weight_value:.3f}")

        # WeightedRandomSampler pour équilibrer les classes
        class_weights = {cls: 1.0 / cnt for cls, cnt in counts.items()}
        sample_weights = [class_weights[y] for y in train_df["label"]]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

        train_ds = DICOMDataset(train_df, transform=train_tfms)
        val_ds   = DICOMDataset(val_df,   transform=val_tfms)

        loader_args = dict(
            batch_size=Config.BATCH_SIZE,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        train_loader = DataLoader(train_ds, sampler=sampler, **loader_args)
        val_loader   = DataLoader(val_ds, shuffle=False, **loader_args)

        history, metrics = train_one_fold_resize512(
            fold,
            train_loader,
            val_loader,
            pos_weight_tensor
        )

        all_metrics.append(metrics)
        last_history = history

    df_metrics = pd.DataFrame(all_metrics)
    print("\n=== Résultats Cross-Validation (ConvNeXt+Signals v2, resize512) ===")
    print(df_metrics.mean(numeric_only=True))

    plot_training_curves(last_history, title_prefix="ConvNeXt+Signals v2 (resize512)")
    return df_metrics


def run_cv_noresize(df_final):
    """
    Cross-validation pour la variante "no-resize".
    """
    df_use = df_final.copy()
    if "h" not in df_use.columns or "w" not in df_use.columns:
        hs = []
        ws = []
        for p in df_use["dicom_path"]:
            try:
                ds = pydicom.dcmread(p, force=True, stop_before_pixels=True)
                hs.append(int(getattr(ds, "Rows", 1024) or 1024))
                ws.append(int(getattr(ds, "Columns", 1024) or 1024))
            except Exception:
                hs.append(1024)
                ws.append(1024)
        df_use["h"] = hs
        df_use["w"] = ws

    # Tri par taille pour limiter les batchs énormes
    df_use = df_use.sort_values(["h", "w"]).reset_index(drop=True)

    skf = StratifiedKFold(
        n_splits=Config.KFOLDS,
        shuffle=True,
        random_state=Config.SEED
    )

    train_tfms, val_tfms = get_transforms_noresize()

    all_metrics = []
    last_history = None

    for fold, (train_idx, val_idx) in enumerate(skf.split(df_use, df_use["label"])):
        train_df = df_use.iloc[train_idx].reset_index(drop=True)
        val_df   = df_use.iloc[val_idx].reset_index(drop=True)

        counts = Counter(train_df["label"])
        n_pos = counts.get(1, 0)
        n_neg = counts.get(0, 0)
        print(f"\n[Fold {fold}] distribution train -> 0:{n_neg} | 1:{n_pos}")

        pos_weight_value = float(n_neg / max(1, n_pos)) if n_pos > 0 else 1.0
        pos_weight_tensor = torch.tensor(
            [pos_weight_value],
            device=Config.DEVICE,
            dtype=torch.float32
        )
        print(f"[Fold {fold}] pos_weight (BCE) = {pos_weight_value:.3f}")

        class_weights = {cls: 1.0 / cnt for cls, cnt in counts.items()}
        sample_weights = [class_weights[y] for y in train_df["label"]]
        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        train_ds = DICOMDataset(train_df, transform=train_tfms)
        val_ds   = DICOMDataset(val_df,   transform=val_tfms)

        loader_args = dict(
            batch_size=Config.BATCH_SIZE,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=collate_pad
        )

        train_loader = DataLoader(train_ds, sampler=sampler, **loader_args)
        val_loader   = DataLoader(val_ds, shuffle=False, **loader_args)

        history, metrics = train_one_fold_noresize(
            fold,
            train_loader,
            val_loader,
            pos_weight_tensor
        )

        all_metrics.append(metrics)
        last_history = history

    df_metrics = pd.DataFrame(all_metrics)
    print("\n=== Résultats Cross-Validation (ConvNeXt+Signals, NO-RESIZE) ===")
    print(df_metrics.mean(numeric_only=True))

    plot_training_curves(last_history, title_prefix="ConvNeXt+Signals (no-resize)")
    return df_metrics


# ======================================================================
# 14. SAUVEGARDE DU MODELE
# ======================================================================

def save_model_state(model, path=None):
    """
    Sauvegarde les poids .pth.
    """
    if path is None:
        path = Config.MODEL_PATH
    torch.save(model.state_dict(), path)
    print(f"Modèle sauvegardé -> {path}")


# ======================================================================
# 15. INFÉRENCE SUR DES IMAGES SANS LABEL
# ======================================================================

def create_convnext_model_for_inference():
    """
    Construit le même ConvNeXt (mêmes couches finales) pour l'inférence.
    Les poids entraînés (.pth) seront chargés ensuite.
    """
    base = models.convnext_base(weights=None)
    n_features = base.classifier[2].in_features
    base.classifier = nn.Sequential(
        nn.Flatten(),
        nn.LayerNorm((n_features,)),
        nn.Linear(n_features, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 1),
    )
    return base


def run_inference_on_unlabeled(
    model_path=None,
    data_dir=None,
    out_csv=None,
    threshold=None
):
    """
    Inférence sur un dossier DICOM sans labels (jeu externe).
    Génère un CSV : file, prob_cancer, pred_class.
    """
    if model_path is None:
        model_path = Config.MODEL_PATH
    if data_dir is None:
        data_dir = Config.INFER_DATA_DIR
    if out_csv is None:
        out_csv = Config.PREDICTION_CSV
    if threshold is None:
        threshold = Config.THRESHOLD_INFER

    # 1. Lister les DICOM à prédire
    dcm_paths = sorted([
        p for p in Path(data_dir).rglob("*.dcm")
        if "__MACOSX" not in str(p) and not Path(p).name.startswith("._")
    ])

    # 2. Dataset + DataLoader (mêmes préproc que resize512)
    infer_tfms = T.Compose([
        T.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        T.Grayscale(num_output_channels=1),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])

    infer_dataset = DICOMUnlabeledDataset(dcm_paths, transform=infer_tfms)
    infer_loader = DataLoader(
        infer_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False
    )

    # 3. Chargement du modèle entraîné
    print("Chargement du modèle pour inférence...")
    model = create_convnext_model_for_inference()
    model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
    model = model.to(Config.DEVICE)
    model.eval()
    print("Modèle prêt.")

    # 4. Prédiction
    predictions = []
    with torch.no_grad():
        for imgs, names in tqdm(infer_loader, desc="Inférence", ncols=90):
            imgs = imgs.to(Config.DEVICE)
            imgs = imgs.repeat(1, 3, 1, 1)  # grayscale -> 3 canaux pour ConvNeXt
            logits = model(imgs)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            probs = np.atleast_1d(probs)
            for name, prob in zip(names, probs):
                predictions.append({
                    "file": name,
                    "prob_cancer": float(prob),
                })

    # 5. Sauvegarde CSV
    df_pred = pd.DataFrame(predictions)
    df_pred["pred_class"] = (df_pred["prob_cancer"] > threshold).astype(int)
    df_pred.to_csv(out_csv, index=False)

    print("\nInférence terminée. Aperçu :")
    print(df_pred.head())
    print(f"Résultats sauvegardés -> {out_csv}")

    # Pas de métriques (pas de labels sur ce set)
    return df_pred


# ======================================================================
# 16. MAIN / EXEMPLES D'UTILISATION
# ======================================================================

if __name__ == "__main__":
    print("ConvNeXt+Signals pipeline initialisé.")
    print("Résumé :")
    print("- cross-validation: train/val (les 2 ont des labels)")
    print("- métriques + visualisations sur la validation")
    print("- export d'un modèle .pth")
    print("- inférence sur un dossier d'images sans labels, génération d'un CSV")

    # Exemple de préparation des DICOM labellisés
    # root_labeled = Path(Config.LABELED_DATA_DIR)
    # clean_macos_artifacts(root_labeled)
    # dicom_paths_labeled = list_dicoms(root_labeled)
    # probe_dicom_metadata(dicom_paths_labeled)

    # Exemple de création du DataFrame final image+label
    # X_df = pd.read_csv("X_train.csv")
    # y_df = pd.read_csv("y_train.csv")
    # df_final = merge_labels_with_dicoms(X_df, y_df, dicom_paths_labeled)
    # df_final contient: image_id, dicom_path, label
    # Ce df_final est utilisé pour train ET val dans la cross-val.

    # Cross-validation "resize512"
    # results_resize512 = run_cv_resize512(df_final)

    # Cross-validation "no-resize"
    # results_noresize = run_cv_noresize(df_final)

    # Sauvegarde manuelle d'un modèle entraîné
    # save_model_state(model, "/content/model_convnext_v2_best.pth")

    # Inférence finale sur le dossier sans labels
    # df_pred_unlabeled = run_inference_on_unlabeled()

    print("Script prêt. Adapter les chemins et décommenter les blocs nécessaires.")

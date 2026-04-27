"""
inference.py — Charge un checkpoint et prédit sur des images DICOM.

Usage standalone :
    python inference.py --checkpoint checkpoints/best.pt \\
                        --images data/raw \\
                        --output predictions.csv

Usage depuis Python :
    from inference import predict_folder, predict_single
"""

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Prédiction sur un dossier DICOM
# ─────────────────────────────────────────────────────────────

def predict_folder(
    images_dir: str,
    checkpoint_path: str,
    output_csv: str = "predictions.csv",
    device: str = "auto",
    img_size: int = 512,
    threshold: float = 0.5,
    batch_size: int = 8,
    num_workers: int = 2,
) -> pd.DataFrame:
    """
    Prédit le risque de cancer pour toutes les images DICOM trouvées
    dans images_dir/{patient_id}/{image_id}.dcm.

    Args:
        images_dir: dossier racine contenant les sous-dossiers patients
        checkpoint_path: path vers le fichier .pt sauvegardé par Trainer
        output_csv: fichier de sortie avec les prédictions
        device: "auto" | "cuda" | "mps" | "cpu"
        img_size: taille de resize (carré)
        threshold: seuil de décision binaire
        batch_size: taille des batchs d'inférence
        num_workers: workers DataLoader

    Returns:
        DataFrame avec colonnes patient_id, image_id, prob, pred
    """
    dev = _resolve_device(device)
    model = _load_model(checkpoint_path, dev)

    records = _discover_dicoms(images_dir)
    if not records:
        raise FileNotFoundError(f"Aucun DICOM trouvé dans {images_dir}")
    logger.info(f"{len(records)} images DICOM trouvées")

    ds = _InferenceDataset(records, images_dir, img_size)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=dev.type == "cuda")

    all_probs, all_pids, all_iids = [], [], []

    model.eval()
    with torch.no_grad():
        for images, patient_ids, image_ids in loader:
            logits, _ = model(images.to(dev))
            probs = torch.sigmoid(logits.squeeze(1)).cpu().numpy()
            all_probs.append(probs)
            all_pids.extend(patient_ids)
            all_iids.extend(image_ids)

    probs_arr = np.concatenate(all_probs)
    pids_arr = np.array(all_pids)

    # Patient-level aggregation
    unique_pids = np.unique(pids_arr)
    patient_rows = []
    for pid in unique_pids:
        mask = pids_arr == pid
        patient_prob = float(probs_arr[mask].max())
        patient_rows.append({
            "patient_id": pid,
            "prob": round(patient_prob, 4),
            "pred": int(patient_prob >= threshold),
        })

    df_patients = pd.DataFrame(patient_rows)
    df_images = pd.DataFrame({
        "patient_id": all_pids,
        "image_id": all_iids,
        "prob": probs_arr.round(4),
        "pred": (probs_arr >= threshold).astype(int),
    })

    # Sauvegarde des deux niveaux
    patient_out = output_csv.replace(".csv", "_patient_level.csv")
    image_out = output_csv.replace(".csv", "_image_level.csv")
    df_patients.to_csv(patient_out, index=False)
    df_images.to_csv(image_out, index=False)

    n_pos = df_patients["pred"].sum()
    logger.info(f"Résultats patient-level : {n_pos}/{len(df_patients)} positifs (threshold={threshold})")
    logger.info(f"Sauvegardé : {patient_out} | {image_out}")

    return df_patients


# ─────────────────────────────────────────────────────────────
# Prédiction sur une seule image
# ─────────────────────────────────────────────────────────────

def predict_single(
    dicom_path: str,
    checkpoint_path: str,
    device: str = "auto",
    img_size: int = 512,
    density: str = "B",
) -> dict:
    """
    Prédit le risque de cancer pour une seule image DICOM.

    Returns:
        dict avec prob (float), pred (0/1), gates (list des poids experts)
    """
    dev = _resolve_device(device)
    model = _load_model(checkpoint_path, dev)

    image = _load_single_dicom(dicom_path, img_size, density)
    tensor = torch.from_numpy(image).unsqueeze(0).to(dev)  # (1, 1, H, W)

    model.eval()
    with torch.no_grad():
        logit, gates = model(tensor)
        prob = torch.sigmoid(logit).item()

    return {
        "prob": round(prob, 4),
        "pred": int(prob >= 0.5),
        "gates": gates.squeeze(0).cpu().tolist(),
        "dicom_path": dicom_path,
    }


# ─────────────────────────────────────────────────────────────
# Helpers internes
# ─────────────────────────────────────────────────────────────

def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def _load_model(checkpoint_path: str, device: torch.device):
    from models.multi_head_expert import MultiHeadMammoModel

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint introuvable : {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    model = MultiHeadMammoModel(embed_dim=512)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    logger.info(f"Modèle chargé depuis {checkpoint_path} (epoch={ckpt.get('epoch', '?')}, AUROC={ckpt.get('auroc', '?')})")
    return model


def _discover_dicoms(images_dir: str) -> list:
    """Parcourt images_dir/{patient_id}/{image_id}.dcm et retourne la liste des records."""
    records = []
    for patient_id in os.listdir(images_dir):
        patient_path = os.path.join(images_dir, patient_id)
        if not os.path.isdir(patient_path):
            continue
        for fname in os.listdir(patient_path):
            if fname.endswith(".dcm"):
                image_id = fname.replace(".dcm", "")
                records.append({"patient_id": patient_id, "image_id": image_id})
    return records


def _load_single_dicom(dicom_path: str, img_size: int, density: str) -> np.ndarray:
    """Charge et prétraite un DICOM → numpy (1, H, W) float32."""
    import cv2
    from core.loader import Loader
    from core.configuration import Config

    # Config minimale pour utiliser le Loader
    images_dir = os.path.dirname(os.path.dirname(dicom_path))
    csv_path = os.path.join(images_dir, "train.csv")
    if not os.path.isfile(csv_path):
        csv_path = "/dev/null" if os.name != "nt" else "NUL"

    try:
        cfg = Config(csv_path=csv_path, images_dir=images_dir)
        loader = Loader(cfg)
        img01, _ = loader.load_dicom_for_roi(dicom_path)
    except Exception:
        # Fallback minimal si Config échoue
        import pydicom
        ds = pydicom.dcmread(dicom_path, force=True)
        img = ds.pixel_array.astype(np.float32)
        lo, hi = np.percentile(img, (0.5, 99.5))
        img01 = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)

    img_resized = cv2.resize(img01, (img_size, img_size), interpolation=cv2.INTER_AREA)
    # Normalisation simple
    img_resized = (img_resized - img_resized.mean()) / (img_resized.std() + 1e-6)
    return img_resized[np.newaxis, ...].astype(np.float32)


class _InferenceDataset(Dataset):
    """Dataset minimal pour l'inférence (pas de labels)."""

    def __init__(self, records: list, images_dir: str, img_size: int):
        self.records = records
        self.images_dir = images_dir
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        dicom_path = os.path.join(
            self.images_dir, str(rec["patient_id"]), f"{rec['image_id']}.dcm"
        )
        image = _load_single_dicom(dicom_path, self.img_size, density="B")
        return torch.from_numpy(image), rec["patient_id"], rec["image_id"]


# ─────────────────────────────────────────────────────────────
# CLI standalone
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="Inférence RSNA Breast Cancer")
    parser.add_argument("--checkpoint", required=True, help="Path checkpoint .pt")
    parser.add_argument("--images", required=True, help="Dossier DICOM")
    parser.add_argument("--output", default="predictions.csv")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    predict_folder(
        images_dir=args.images,
        checkpoint_path=args.checkpoint,
        output_csv=args.output,
        device=args.device,
        img_size=args.img_size,
        threshold=args.threshold,
        batch_size=args.batch_size,
    )

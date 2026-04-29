"""
main.py — Entry point train / eval / infer
Usage:
    python main.py train --csv data/train.csv --images data/raw --epochs 20
    python main.py eval  --csv data/train.csv --images data/raw --checkpoint checkpoints/best.pt
    python main.py infer --images data/raw --checkpoint checkpoints/best.pt --output predictions.csv
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Device helpers (CUDA / MPS / CPU)
# ─────────────────────────────────────────────────────────────

def get_device(device_str: str = "auto") -> torch.device:
    """Détecte automatiquement le meilleur device disponible et l'affiche."""
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            name = torch.cuda.get_device_name(0)
            mem  = torch.cuda.get_device_properties(0).total_memory // 1024 ** 3
            logger.info(f"GPU CUDA : {name} ({mem} GB)")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("GPU Apple Silicon MPS")
        else:
            device = torch.device("cpu")
            logger.info(f"CPU ({os.cpu_count()} cores)")
    else:
        device = torch.device(device_str)
        logger.info(f"Device forcé : {device}")
    return device


def get_num_workers(device: torch.device) -> int:
    """NUM_WORKERS optimal par device."""
    if device.type == "cuda":
        return 4
    if device.type == "mps":
        return 0   # multiprocessing instable sur MPS
    return min(4, os.cpu_count() or 1)


def get_batch_size(device: torch.device, override: int = None) -> int:
    """BATCH_SIZE recommandé par device (sauf override CLI)."""
    if override and override != 16:   # 16 = valeur par défaut CLI → on l'ignore
        return override
    if device.type == "cuda":
        return 32
    if device.type == "mps":
        return 8
    return 4


# ─────────────────────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────────────────────

def run_train(args: argparse.Namespace) -> None:
    from torch.utils.data import DataLoader
    from models.multi_head_expert import MultiHeadMammoModel
    from models.dataset import MammographyDataset, BalancedPatientSampler
    from models.trainer import Trainer

    device      = get_device(args.device)
    num_workers = get_num_workers(device)
    batch_size  = get_batch_size(device, args.batch_size)
    logger.info(f"batch_size={batch_size}  num_workers={num_workers}")

    df = _load_and_split_csv(args.csv)

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df   = df[df["split"] == "val"].reset_index(drop=True)

    train_ds = MammographyDataset(train_df, args.images,
                                  target_size=(args.img_size, args.img_size), mode="train")
    val_ds   = MammographyDataset(val_df,   args.images,
                                  target_size=(args.img_size, args.img_size), mode="val")

    train_labels = train_df["cancer"].values
    train_pids   = train_df["patient_id"].values
    sampler = BalancedPatientSampler(train_pids, train_labels, pos_weight=10.0)

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=pin,
                              prefetch_factor=2 if num_workers > 0 else None)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin)

    model = MultiHeadMammoModel(
        embed_dim=512,
        radImageNet_resnet=args.rad_resnet,
    )

    if args.freeze_phase1:
        model.freeze_backbones()
        logger.info("Phase 1 : backbones gelés")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=args.lr,
        n_epochs=args.epochs,
        patience=args.patience,
        checkpoint_dir=args.checkpoint_dir,
        use_amp=not args.no_amp,
    )

    history = trainer.train()
    logger.info(f"Entraînement terminé. Meilleur AUROC val : {max(history['val_auroc']):.4f}")


# ─────────────────────────────────────────────────────────────
# Eval
# ─────────────────────────────────────────────────────────────

def run_eval(args: argparse.Namespace) -> None:
    from torch.utils.data import DataLoader
    from models.multi_head_expert import MultiHeadMammoModel
    from models.dataset import MammographyDataset
    from sklearn.metrics import roc_auc_score, classification_report

    device      = get_device(args.device)
    num_workers = get_num_workers(device)
    batch_size  = get_batch_size(device, args.batch_size)

    df = _load_and_split_csv(args.csv)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    test_ds     = MammographyDataset(test_df, args.images,
                                     target_size=(args.img_size, args.img_size), mode="test")
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=device.type == "cuda")

    model = MultiHeadMammoModel()
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    all_probs, all_labels, all_pids = [], [], []

    with torch.no_grad():
        for images, labels, pids in test_loader:
            logits, _ = model(images.to(device))
            probs = torch.sigmoid(logits.squeeze(1)).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())
            all_pids.append(pids.numpy() if isinstance(pids, torch.Tensor) else np.array(pids))

    probs  = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    pids   = np.concatenate(all_pids)

    patient_probs, unique_pids = MammographyDataset.patient_level_aggregate(probs, pids, method="max")
    patient_labels = np.array([labels[pids == p].max() for p in unique_pids])

    auroc  = roc_auc_score(patient_labels, patient_probs)
    preds  = (patient_probs >= args.threshold).astype(int)

    logger.info(f"\nAUROC (patient-level) : {auroc:.4f}")
    logger.info(f"\n{classification_report(patient_labels, preds, target_names=['No cancer', 'Cancer'])}")


# ─────────────────────────────────────────────────────────────
# Infer
# ─────────────────────────────────────────────────────────────

def run_infer(args: argparse.Namespace) -> None:
    """Prédit sur un dossier d'images DICOM sans labels."""
    from inference import predict_folder
    predict_folder(
        images_dir=args.images,
        checkpoint_path=args.checkpoint,
        output_csv=args.output,
        device=args.device,
        img_size=args.img_size,
        threshold=args.threshold,
    )


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _load_and_split_csv(csv_path: str) -> pd.DataFrame:
    """Charge le CSV et ajoute une colonne 'split' patient-wise si absente."""
    df = pd.read_csv(csv_path)
    if "split" not in df.columns:
        patients = df["patient_id"].unique()
        labels_per_patient = df.groupby("patient_id")["cancer"].max().reindex(patients).values
        train_p, tmp_p = train_test_split(patients, test_size=0.30,
                                          stratify=labels_per_patient, random_state=42)
        tmp_labels = (df[df["patient_id"].isin(tmp_p)]
                      .groupby("patient_id")["cancer"].max().reindex(tmp_p).values)
        val_p, test_p = train_test_split(tmp_p, test_size=0.50,
                                         stratify=tmp_labels, random_state=42)
        df["split"] = "train"
        df.loc[df["patient_id"].isin(val_p),  "split"] = "val"
        df.loc[df["patient_id"].isin(test_p), "split"] = "test"
    return df


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RSNA Breast Cancer Detection — Multi-Head Expert Model"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--images",     required=True, help="Dossier DICOM")
    common.add_argument("--img-size",   type=int,   default=512)
    common.add_argument("--batch-size", type=int,   default=16,    help="0 = auto par device")
    common.add_argument("--workers",    type=int,   default=-1,    help="-1 = auto par device")
    common.add_argument("--device",     default="auto", choices=["auto", "cuda", "mps", "cpu"])
    common.add_argument("--threshold",  type=float, default=0.5)

    p_train = sub.add_parser("train", parents=[common])
    p_train.add_argument("--csv",            required=True)
    p_train.add_argument("--epochs",         type=int,   default=20)
    p_train.add_argument("--lr",             type=float, default=1e-4)
    p_train.add_argument("--patience",       type=int,   default=5)
    p_train.add_argument("--checkpoint-dir", default="checkpoints")
    p_train.add_argument("--rad-resnet",     default=None, help="Path RadImageNet ResNet50 .pt")
    p_train.add_argument("--freeze-phase1",  action="store_true")
    p_train.add_argument("--no-amp",         action="store_true", help="Désactiver mixed precision")

    p_eval = sub.add_parser("eval", parents=[common])
    p_eval.add_argument("--csv",        required=True)
    p_eval.add_argument("--checkpoint", required=True)

    p_infer = sub.add_parser("infer", parents=[common])
    p_infer.add_argument("--checkpoint", required=True)
    p_infer.add_argument("--output",     default="predictions.csv")

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args   = parser.parse_args()

    if args.command == "train":
        run_train(args)
    elif args.command == "eval":
        run_eval(args)
    elif args.command == "infer":
        run_infer(args)

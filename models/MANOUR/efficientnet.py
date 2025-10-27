''' 
1. Imports & Config
2. Dataset class (générique pour train/val/test)
3. Transforms
4. Model creation (EfficientNet)
5. Focal loss + class weights
6. Training loop (train + validation)
7. Evaluation sur test
8. Courbes + rapports finaux

'''


import sys
import gc
import warnings
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics import (
    f1_score, accuracy_score, roc_auc_score,
    classification_report, balanced_accuracy_score, matthews_corrcoef
)
from sklearn.metrics import average_precision_score
pr_auc = average_precision_score(targs, probs)
import pydicom
from PIL import Image

warnings.filterwarnings('ignore')

# ======================================================================
# CONFIGURATION
# ======================================================================
class Config:
    MODEL_NAME = "efficientnet_b0"
    PRETRAINED = True
    NUM_CLASSES = 1
    IMG_SIZE = 224
    BATCH_SIZE = 16 if torch.cuda.is_available() else 8
    NUM_WORKERS = 0
    EPOCHS = 8
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    FOCAL_LOSS_GAMMA = 2.5
    USE_CLASS_WEIGHTS = True
    USE_BALANCED_SAMPLING = True
    RESULTS_DIR = Path("/Users/manour/Downloads/kaggle/working/results")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ======================================================================
# CHEMINS DES DONNÉES
# ======================================================================
Train_PATH = "/Users/manour/Downloads/kaggle/working/train"
Val_PATH = "/Users/manour/Downloads/kaggle/working/validation"
Test_PATH = "/Users/manour/Downloads/kaggle/working/test"

X_TRAIN_PATH = "/Users/manour/Downloads/kaggle/working/X_train.csv"
Y_TRAIN_PATH = "/Users/manour/Downloads/kaggle/working/y_train.csv"
X_VAL_PATH = "/Users/manour/Downloads/kaggle/working/X_val.csv"
Y_VAL_PATH = "/Users/manour/Downloads/kaggle/working/y_val.csv"
X_TEST_PATH = "/Users/manour/Downloads/kaggle/working/X_test.csv"
Y_TEST_PATH = "/Users/manour/Downloads/kaggle/working/y_test.csv"

# ======================================================================
# DEVICE
# ======================================================================
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU détecté : {torch.cuda.get_device_name()}")
    torch.backends.cudnn.benchmark = True
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("GPU Apple Silicon (MPS) détecté")
else:
    device = torch.device("cpu")
    print("CPU utilisé")
print(f"Device actif : {device}")

def clear_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

# ======================================================================
# DATASET
# ======================================================================
class DICOMCancerDataset(Dataset):
    def __init__(self, dicom_dir, x_csv, y_csv, transform=None):
        self.dicom_dir = Path(dicom_dir)
        self.transform = transform
        self.samples = []
        self._load_and_match(x_csv, y_csv)

    def _load_and_match(self, x_csv, y_csv):
        x_df = pd.read_csv(x_csv)
        y_df = pd.read_csv(y_csv)
        x_df["image_id"] = x_df["image_id"].astype(str)
        y_df.columns = ["cancer"]
        combined = x_df.copy()
        combined["cancer"] = y_df["cancer"].values

        id_to_label = {str(r.image_id): int(r.cancer) for _, r in combined.iterrows()}
        dicoms = list(self.dicom_dir.glob("**/*.dcm"))
        matched = 0
        for f in dicoms:
            stem = f.stem.strip()
            if stem in id_to_label:
                self.samples.append({"path": str(f), "label": id_to_label[stem]})
                matched += 1
        print(f"{matched}/{len(dicoms)} fichiers DICOM appariés ({self.dicom_dir.name})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        try:
            dcm = pydicom.dcmread(s["path"], force=True)
            arr = dcm.pixel_array.astype(np.float32)
            arr = (arr - arr.min()) / max(arr.max() - arr.min(), 1e-8)
            arr = (arr * 255).astype(np.uint8)
            if len(arr.shape) == 2:
                arr = np.stack([arr] * 3, axis=-1)
            img = Image.fromarray(arr)
        except Exception:
            arr = np.ones((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8) * 128
            img = Image.fromarray(arr)
        if self.transform:
            img = self.transform(img)
        return img, s["label"]

# ======================================================================
# TRANSFORMS
# ======================================================================
def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

# ======================================================================
# MODÈLE
# ======================================================================
def create_model():
    clear_memory()
    model_fn = getattr(models, Config.MODEL_NAME)
    model = model_fn(pretrained=Config.PRETRAINED)
    num_f = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_f, Config.NUM_CLASSES)
    )
    return model.to(device)

# ======================================================================
# LOSS
# ======================================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2.5):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma
    def forward(self, inputs, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce)
        loss = self.alpha * (1 - pt) ** self.gamma * bce
        return loss.mean()

# ======================================================================
# OUTILS D'ÉQUILIBRAGE
# ======================================================================
def class_weights(dataset):
    labels = [s["label"] for s in dataset.samples]
    c = Counter(labels)
    print(f"Distribution : {c}")
    if Config.USE_CLASS_WEIGHTS and 1 in c:
        ratio = c[0] / c[1]
        pos_w = torch.tensor([ratio]).to(device)
    else:
        pos_w = torch.tensor([1.0]).to(device)
    return pos_w

def make_sampler(dataset):
    if not Config.USE_BALANCED_SAMPLING:
        return None
    labels = [s["label"] for s in dataset.samples]
    c = Counter(labels)
    w = {0: 1.0, 1: c[0]/max(c[1], 1)}
    sample_w = [w[l] for l in labels]
    return WeightedRandomSampler(sample_w, len(sample_w), replacement=True)

# ======================================================================
# ENTRAÎNEMENT ET VALIDATION
# ======================================================================
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x, y in tqdm(loader, desc="Train", leave=False):
        x, y = x.to(device), y.float().to(device)
        optimizer.zero_grad()
        out = model(x).squeeze()
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = (torch.sigmoid(out) > 0.5).float()
        correct += (preds == y).sum().item()
        total += y.size(0)
    return total_loss/len(loader), correct/total

def validate(model, loader, criterion):
    model.eval()
    preds, probs, targs, total_loss = [], [], [], 0
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Val", leave=False):
            x = x.to(device)
            yf = y.float().to(device)
            out = model(x).squeeze()
            loss = criterion(out, yf)
            total_loss += loss.item()
            p = torch.sigmoid(out).cpu().numpy()
            probs.extend(p)
            targs.extend(y.numpy())
            preds.extend((p > 0.5).astype(int))
    metrics = {
        "loss": total_loss/len(loader),
        "acc": accuracy_score(targs, preds),
        "bacc": balanced_accuracy_score(targs, preds),
        "auc": roc_auc_score(targs, probs) if len(set(targs)) > 1 else 0.5,
        "f1_macro": f1_score(targs, preds, average="macro", zero_division=0),
        "f1_micro": f1_score(targs, preds, average="micro", zero_division=0),
        "mcc": matthews_corrcoef(targs, preds)
    }
    return metrics


# ======================================================================
# BOUCLE D'ENTRAÎNEMENT COMPLÈTE
# ======================================================================
def train_model(train_ds, val_ds):
    sampler = make_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE,
                              sampler=sampler, shuffle=(sampler is None),
                              num_workers=Config.NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE,
                            shuffle=False, num_workers=Config.NUM_WORKERS)

    model = create_model()
    pos_w = class_weights(train_ds)
    criterion = FocalLoss(alpha=1, gamma=Config.FOCAL_LOSS_GAMMA)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE,
                            weight_decay=Config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)

    history = {"train_loss": [], "train_acc": [], "val_loss": [],
               "val_acc": [], "val_auc": [], "val_f1": [], "val_mcc": []}

    best_auc, best_state = 0.0, None

    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{Config.EPOCHS}")
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_metrics = validate(model, val_loader, criterion)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        for k in ["loss", "acc", "auc"]:
            history[f"val_{k}"].append(val_metrics[k])
        history["val_f1"].append(val_metrics["f1_macro"])
        history["val_mcc"].append(val_metrics["mcc"])

        print(f"Train Loss {tr_loss:.4f} | Train Acc {tr_acc:.4f}")
        print(f"Val Loss {val_metrics['loss']:.4f} | Val AUC {val_metrics['auc']:.4f} | "
              f"F1 {val_metrics['f1_macro']:.4f} | MCC {val_metrics['mcc']:.4f}")

        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            best_state = model.state_dict().copy()

    if best_state:
        model.load_state_dict(best_state)
        torch.save(model.state_dict(), Config.RESULTS_DIR / "best_model.pth")
        print(f"Meilleur modèle sauvegardé (AUC={best_auc:.3f})")

    plot_history(history, Config.RESULTS_DIR / "training_curves.png")
    return model, history


# ======================================================================
# VISUALISATION
# ======================================================================
def plot_history(history, path):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train")
    plt.plot(history["val_loss"], label="Val")
    plt.title("Loss")
    plt.legend(); plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history["val_auc"], label="AUC")
    plt.plot(history["val_f1"], label="F1 Macro")
    plt.plot(history["val_mcc"], label="MCC")
    plt.title("Validation Metrics")
    plt.legend(); plt.grid(True)

    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Courbes sauvegardées dans {path}")


# ======================================================================
# ÉVALUATION FINALE SUR LE TEST
# =====================================================================
from sklearn.metrics import average_precision_score, precision_score, recall_score

def evaluate_test(model, test_ds):
    test_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE,
                             shuffle=False, num_workers=Config.NUM_WORKERS)
    model.eval()
    preds, probs, targs = [], [], []
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Test"):
            x = x.to(device)
            out = model(x).squeeze()
            p = torch.sigmoid(out).cpu().numpy()
            probs.extend(p)
            targs.extend(y.numpy())
            preds.extend((p > 0.5).astype(int))

    report = classification_report(targs, preds,
                                   target_names=["Non-Cancer", "Cancer"],
                                   zero_division=0, output_dict=True)
    auc = roc_auc_score(targs, probs) if len(set(targs)) > 1 else 0.5
    pr_auc = average_precision_score(targs, probs)
    precision = precision_score(targs, preds, zero_division=0)
    recall = recall_score(targs, preds, zero_division=0)
    f1 = f1_score(targs, preds, average="macro")
    acc = accuracy_score(targs, preds)
    bacc = balanced_accuracy_score(targs, preds)
    mcc = matthews_corrcoef(targs, preds)

    results = {
        "Accuracy": acc,
        "Balanced Accuracy": bacc,
        "ROC-AUC": auc,
        "PR-AUC": pr_auc,
        "Precision": precision,
        "Recall": recall,
        "F1 Macro": f1,
        "MCC": mcc
    }
    print("\nRésultats finaux sur le jeu de test :")
    for k, v in results.items():
        print(f"{k:<20} {v:.4f}")

    pd.DataFrame(report).T.to_csv(Config.RESULTS_DIR / "classification_report.csv")
    print(f"Rapport de classification sauvegardé : {Config.RESULTS_DIR/'classification_report.csv'}")
    return results


# ======================================================================
# MAIN
# ======================================================================
def main():
    print("=== Entraînement EfficientNet sur train/val/test ===")
    train_ds = DICOMCancerDataset(Train_PATH, X_TRAIN_PATH, Y_TRAIN_PATH, transform=get_transforms(True))
    val_ds = DICOMCancerDataset(Val_PATH, X_VAL_PATH, Y_VAL_PATH, transform=get_transforms(False))
    test_ds = DICOMCancerDataset(Test_PATH, X_TEST_PATH, Y_TEST_PATH, transform=get_transforms(False))

    model, hist = train_model(train_ds, val_ds)
    results = evaluate_test(model, test_ds)

    pd.DataFrame([results]).to_csv(Config.RESULTS_DIR / "test_results.csv", index=False)
    print(f"Résultats enregistrés : {Config.RESULTS_DIR/'test_results.csv'}")

if __name__ == "__main__":
    main()
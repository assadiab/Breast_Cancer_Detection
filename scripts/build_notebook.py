#!/usr/bin/env python3
"""Génère le notebook RSNA Mammo-CLIP propre (entraînement uniquement, pas de rebuild cache)."""
import json, sys

cells = []

def code(src):
    cells.append({"cell_type": "code", "execution_count": None,
                  "metadata": {}, "outputs": [], "source": src})

def md(src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src})

# ════════════════════════════════════════════════════════════════════════════
md("""# RSNA Breast Cancer Detection — Mammo-CLIP B5

**Approche** : encodeur EfficientNet-B5 pré-entraîné sur de vraies mammographies (Mammo-CLIP, datasets UPMC + VinDr), fine-tuné sur le cache RSNA pré-construit (47k images).

- Cache PNG pré-construit (pas de reconstruction → 9h dédiées à l'entraînement)
- Crop sein automatique on-the-fly
- Loss BCE + WeightedRandomSampler (déséquilibre ~2%)
- Métrique pF1 niveau sein (métrique officielle RSNA) + AUROC
- AMP, LR différentiel, cosine schedule, TTA""")

# ── Cell 1 : GPU diagnostic ──────────────────────────────────────────────────
code("""import subprocess
_smi = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                      capture_output=True, text=True)
print("GPU:", _smi.stdout.strip())
assert 'T4' in _smi.stdout or 'P100' in _smi.stdout or 'A100' in _smi.stdout, "GPU non détecté"
if 'T4' not in _smi.stdout:
    print("⚠️  ATTENTION : GPU != T4 — vérifier l'accélérateur dans l'UI Kaggle")""")

# ── Cell 2 : installs + imports ──────────────────────────────────────────────
code("""# efficientnet_pytorch (lukemelas) : naming _conv_stem/_blocks = clés EXACTES du checkpoint Mammo-CLIP
import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'efficientnet_pytorch==0.7.1'], check=False)

import os, json, time, gc, math, random
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from efficientnet_pytorch import EfficientNet
from sklearn.metrics import roc_auc_score, average_precision_score

SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP = DEVICE.type == 'cuda'
# Normalisation EXACTE de Mammo-CLIP pour RSNA (sur images [0,1])
MC_MEAN, MC_STD = 0.3089279, 0.25053555408335154
print(f"torch {torch.__version__} | device {DEVICE} | AMP={USE_AMP}")""")

# ── Cell 3 : config + détection des chemins ──────────────────────────────────
code("""# ── Détection automatique du cache pré-construit + des CSVs de split ─────────
print("=== /kaggle/input/ ===")
for _it in sorted(os.listdir('/kaggle/input')):
    print("  ", _it, '→', os.listdir(os.path.join('/kaggle/input', _it))[:6])

def _find_dir_with(pattern, roots=('/kaggle/input',)):
    for root in roots:
        for dp, dn, fn in os.walk(root):
            if any(f.endswith(pattern) for f in fn):
                return dp
    return None

# Cache PNG : dossier contenant des .png (le plus gros)
_cache_candidates = []
for dp, dn, fn in os.walk('/kaggle/input'):
    n_png = sum(1 for f in fn if f.endswith('.png'))
    if n_png > 100:
        _cache_candidates.append((n_png, dp))
_cache_candidates.sort(reverse=True)
assert _cache_candidates, "Aucun cache PNG trouvé dans /kaggle/input"
CACHE_DIR = _cache_candidates[0][1]
print(f"\\n✅ CACHE_DIR = {CACHE_DIR} ({_cache_candidates[0][0]} PNGs)")

# CSVs de split
_csv_dir = _find_dir_with('X_train.csv')
assert _csv_dir, "X_train.csv introuvable"
print(f"✅ CSV split dir = {_csv_dir}")

# Poids Mammo-CLIP B5 (dataset monté OU download HF)
MAMMOCLIP_CKPT = None
for dp, dn, fn in os.walk('/kaggle/input'):
    for f in fn:
        if f.endswith('.tar') and 'b5' in f.lower():
            MAMMOCLIP_CKPT = os.path.join(dp, f)
print(f"✅ Mammo-CLIP ckpt (dataset) = {MAMMOCLIP_CKPT}")

# ── Hyperparamètres (recette Mammo-CLIP RSNA : lr 5e-5, BCE+pos_weight, full FT) ─
IMG_SIZE     = 512
BATCH_SIZE   = 16
ACCUM_STEPS  = 2       # batch effectif = 32
NUM_WORKERS  = 4
N_EPOCHS     = 12
PATIENCE     = 4
LR_BACKBONE  = 5e-5    # full fine-tune comme Mammo-CLIP
LR_HEAD      = 5e-4
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 1

WORK = '/kaggle/working'
CKPT_DIR = f'{WORK}/checkpoints'; os.makedirs(CKPT_DIR, exist_ok=True)
RES_DIR  = f'{WORK}/results';     os.makedirs(RES_DIR,  exist_ok=True)
FIG_DIR  = f'{WORK}/figures';     os.makedirs(FIG_DIR,  exist_ok=True)
print("✅ Config prête")""")

# ── Cell 4 : chargement des splits ───────────────────────────────────────────
code("""# Charge les splits pré-calculés (patient-level, pas de fuite)
X_train = pd.read_csv(f"{_csv_dir}/X_train.csv")
Y_train = pd.read_csv(f"{_csv_dir}/Y_train.csv")
X_val   = pd.read_csv(f"{_csv_dir}/X_val.csv")
Y_val   = pd.read_csv(f"{_csv_dir}/Y_val.csv")
X_test  = pd.read_csv(f"{_csv_dir}/X_test.csv")
Y_test  = pd.read_csv(f"{_csv_dir}/Y_test.csv")

for name, X, Y in [('train',X_train,Y_train),('val',X_val,Y_val),('test',X_test,Y_test)]:
    pos = int(Y['cancer'].sum())
    print(f"{name:5s}: {len(X):6d} images | cancer+ = {pos:4d} ({pos/len(X)*100:.2f}%)")""")

# ── Cell 5 : Dataset avec crop sein + augmentation ──────────────────────────
code('''def crop_breast(img, thr=0.06):
    """Crop la bounding-box du sein (enlève le fond noir). img float [0,1] HxW."""
    mask = img > thr
    if mask.sum() < 200:
        return img
    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    # marge 2%
    h, w = img.shape
    my, mx = int(0.02*h), int(0.02*w)
    y0 = max(0, y0-my); y1 = min(h, y1+my)
    x0 = max(0, x0-mx); x1 = min(w, x1+mx)
    return img[y0:y1, x0:x1]


class MammoDataset(Dataset):
    def __init__(self, X_df, y_df, cache_dir, img_size=512, augment=False):
        _df  = X_df.reset_index(drop=True)
        _lbl = y_df.reset_index(drop=True)['cancer'].values.astype(np.float32)
        # filtre les images réellement présentes dans le cache
        paths = _df.apply(lambda r: f"{cache_dir}/{r['patient_id']}_{r['image_id']}.png", axis=1).values
        mask  = np.array([os.path.exists(p) for p in paths])
        if not mask.all():
            print(f"  ⚠️ {(~mask).sum()}/{len(mask)} images absentes du cache (filtrées)")
        self.df       = _df[mask].reset_index(drop=True)
        self.labels   = _lbl[mask]
        self.paths    = paths[mask]
        self.img_size = img_size
        self.augment  = augment

    def __len__(self):
        return len(self.df)

    def _aug(self, img):
        if random.random() < 0.5:
            img = np.fliplr(img).copy()
        if random.random() < 0.5:
            img = np.flipud(img).copy()
        if random.random() < 0.4:
            ang = random.uniform(-12, 12)
            h, w = img.shape
            M = cv2.getRotationMatrix2D((w/2, h/2), ang, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
        if random.random() < 0.4:
            img = np.clip(img * random.uniform(0.85, 1.15), 0, 1)
        if random.random() < 0.3:
            img = np.power(np.clip(img, 0, 1), random.uniform(0.8, 1.25))
        return img

    def __getitem__(self, idx):
        img = cv2.imread(self.paths[idx], cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((self.img_size, self.img_size), np.float32)
        else:
            img = img.astype(np.float32) / 255.0
            img = crop_breast(img)
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        if self.augment:
            img = self._aug(img)
        # Normalisation Mammo-CLIP + réplication 3 canaux (le checkpoint attend du RGB)
        img = (np.clip(img, 0, 1) - MC_MEAN) / MC_STD
        t = torch.from_numpy(np.ascontiguousarray(img)).float().unsqueeze(0).repeat(3, 1, 1)
        return t, torch.tensor(self.labels[idx], dtype=torch.float32)

print("✅ MammoDataset défini")''')

# ── Cell 6 : Modèle Mammo-CLIP B5 (efficientnet_pytorch lukemelas) ────────────
code('''def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0/p)

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__(); self.p = p; self.eps = eps
    def forward(self, x):
        return gem(x, self.p, self.eps).flatten(1)

class MammoCLIPModel(nn.Module):
    """EfficientNet-B5 lukemelas (poids Mammo-CLIP) + GeM + tête de classification."""
    def __init__(self, ckpt_path=None, drop=0.3):
        super().__init__()
        # même construction que Mammo-CLIP : EfficientNet.from_name('efficientnet-b5', num_classes=1)
        self.encoder = EfficientNet.from_name('efficientnet-b5', num_classes=1)
        n_features = 2048  # B5 _conv_head out
        self.pool  = GeM()

        if ckpt_path and os.path.exists(ckpt_path):
            self._load_mammoclip(ckpt_path)
        else:
            print("⚠️  Pas de checkpoint Mammo-CLIP — backbone non pré-entraîné !")

        self.head = nn.Sequential(
            nn.Linear(n_features, 512), nn.BatchNorm1d(512), nn.SiLU(),
            nn.Dropout(drop), nn.Linear(512, 1)
        )

    def _load_mammoclip(self, path):
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        raw = ckpt.get('model', ckpt)
        sd = {}
        for k, v in raw.items():
            if k.startswith('image_encoder.'):
                sd[k[len('image_encoder.'):]] = v
        miss, unexp = self.encoder.load_state_dict(sd, strict=False)
        n_loaded = len(sd) - len([k for k in sd if k in unexp])
        print(f"✅ Mammo-CLIP : {n_loaded}/{len(sd)} tenseurs chargés | manquants={len(miss)} inattendus={len(unexp)}")
        if n_loaded < 200:
            print(f"   ⚠️ Peu de tenseurs chargés — vérifier la compatibilité ! Exemples manquants: {miss[:3]}")

    def forward(self, x):
        feat = self.encoder.extract_features(x)   # [B, 2048, H', W']
        feat = self.pool(feat)                     # [B, 2048]
        return self.head(feat).squeeze(-1)

# Téléchargement HF si pas de dataset monté (le .tar fait ~1.6GB)
if MAMMOCLIP_CKPT is None:
    try:
        from huggingface_hub import hf_hub_download
        MAMMOCLIP_CKPT = hf_hub_download(
            repo_id="shawn24/Mammo-CLIP",
            filename="Pre-trained-checkpoints/b5-model-best-epoch-7.tar",
            cache_dir="/kaggle/working/hf")
        print(f"✅ Mammo-CLIP téléchargé : {MAMMOCLIP_CKPT}")
    except Exception as e:
        print(f"⚠️ Échec download HF : {e}")

model = MammoCLIPModel(ckpt_path=MAMMOCLIP_CKPT).to(DEVICE)
n_par = sum(p.numel() for p in model.parameters())
print(f"✅ Modèle : {n_par/1e6:.1f}M paramètres")''')

# ── Cell 7 : Loss, sampler, métriques ────────────────────────────────────────
code('''# ── Sampler pondéré (équilibre les batchs) ───────────────────────────────────
def make_sampler(labels, pos_mult=3.0):
    """Sur-échantillonne les positifs (mais pas 1:1 pour garder de la variété)."""
    w = np.where(labels == 1, pos_mult, 1.0)
    return WeightedRandomSampler(torch.tensor(w, dtype=torch.float), len(w), replacement=True)

# ── pF1 probabiliste (métrique officielle RSNA) ──────────────────────────────
def pfbeta(labels, preds, beta=1.0):
    labels = np.asarray(labels, np.float64); preds = np.asarray(preds, np.float64)
    ctp = (preds * labels).sum()
    cfp = (preds * (1 - labels)).sum()
    p = ctp / (ctp + cfp + 1e-9)
    r = ctp / (labels.sum() + 1e-9)
    if p + r == 0: return 0.0
    return (1 + beta**2) * p * r / (beta**2 * p + r + 1e-9)

def breast_aggregate(df, probs, labels):
    """Agrège au niveau (patient, laterality) : prob = moyenne, label = max."""
    g = df.copy()
    g['prob'] = probs; g['lab'] = labels
    agg = g.groupby(['patient_id', 'laterality']).agg(prob=('prob','mean'), lab=('lab','max'))
    return agg['prob'].values, agg['lab'].values

def best_threshold_f1(labels, probs):
    best_f1, best_t = 0, 0.5
    for t in np.arange(0.02, 0.9, 0.02):
        pred = (probs >= t).astype(int)
        tp = ((pred==1)&(labels==1)).sum(); fp=((pred==1)&(labels==0)).sum(); fn=((pred==0)&(labels==1)).sum()
        f1 = 2*tp/(2*tp+fp+fn+1e-9)
        if f1 > best_f1: best_f1, best_t = f1, t
    return best_f1, best_t

def compute_metrics(df, probs, labels):
    probs = np.asarray(probs); labels = np.asarray(labels)
    auroc = roc_auc_score(labels, probs) if len(np.unique(labels))>1 else 0.5
    bp, bl = breast_aggregate(df, probs, labels)
    pf1 = pfbeta(bl, bp)
    f1, thr = best_threshold_f1(bl, bp)
    return {'auroc': round(float(auroc),4), 'pf1_breast': round(float(pf1),4),
            'f1_breast': round(float(f1),4), 'threshold': round(float(thr),3),
            'auroc_breast': round(float(roc_auc_score(bl,bp)) if len(np.unique(bl))>1 else 0.5, 4)}

print("✅ Métriques + sampler définis")''')

# ── Cell 8 : DataLoaders ─────────────────────────────────────────────────────
code("""ds_train = MammoDataset(X_train, Y_train, CACHE_DIR, IMG_SIZE, augment=True)
ds_val   = MammoDataset(X_val,   Y_val,   CACHE_DIR, IMG_SIZE, augment=False)
ds_test  = MammoDataset(X_test,  Y_test,  CACHE_DIR, IMG_SIZE, augment=False)

sampler = make_sampler(ds_train.labels, pos_mult=3.0)
_pf = {'prefetch_factor': 2} if NUM_WORKERS > 0 else {}
train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=NUM_WORKERS, pin_memory=True, drop_last=True, **_pf)
val_loader   = DataLoader(ds_val,   batch_size=BATCH_SIZE*2, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)
test_loader  = DataLoader(ds_test,  batch_size=BATCH_SIZE*2, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)
print(f"✅ Loaders : train={len(ds_train)} val={len(ds_val)} test={len(ds_test)}")""")

# ── Cell 9 : Boucle d'entraînement ───────────────────────────────────────────
code('''# Optimiseur : LR différentiel backbone vs tête
backbone_params = list(model.encoder.parameters()) + list(model.pool.parameters())
head_params     = list(model.head.parameters())
optimizer = torch.optim.AdamW([
    {'params': backbone_params, 'lr': LR_BACKBONE},
    {'params': head_params,     'lr': LR_HEAD},
], weight_decay=WEIGHT_DECAY)

# pos_weight léger (le sampler fait déjà l'essentiel)
_pos = ds_train.labels.sum(); _neg = len(ds_train.labels) - _pos
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([min(_neg/_pos, 5.0)]).to(DEVICE))

steps_per_epoch = len(train_loader) // ACCUM_STEPS
total_steps = steps_per_epoch * N_EPOCHS
warmup_steps = steps_per_epoch * WARMUP_EPOCHS
def lr_lambda(step):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * prog))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)

def run_eval(loader, df):
    model.eval(); probs, labs = [], []
    with torch.no_grad():
        for X, y in tqdm(loader, desc='eval', leave=False):
            X = X.to(DEVICE, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=USE_AMP):
                logit = model(X)
            probs.extend(torch.sigmoid(logit).float().cpu().numpy().tolist())
            labs.extend(y.numpy().tolist())
    return compute_metrics(df, probs, labs), probs, labs

history = []
best_pf1, best_state, no_improve = -1, None, 0
print(f"🚀 Entraînement : {N_EPOCHS} epochs × {steps_per_epoch} steps (batch eff. {BATCH_SIZE*ACCUM_STEPS})")
t_start = time.time()

for epoch in range(1, N_EPOCHS+1):
    model.train(); optimizer.zero_grad()
    run_loss, nb = 0.0, 0
    for i, (X, y) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch}/{N_EPOCHS}', leave=False)):
        X, y = X.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        with torch.amp.autocast('cuda', enabled=USE_AMP):
            logit = model(X)
            loss = criterion(logit, y) / ACCUM_STEPS
        scaler.scale(loss).backward()
        if (i+1) % ACCUM_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer); scaler.update()
            optimizer.zero_grad(); scheduler.step()
        run_loss += loss.item()*ACCUM_STEPS; nb += 1
    train_loss = run_loss/max(nb,1)

    metrics, _, _ = run_eval(val_loader, X_val.assign(cancer=Y_val['cancer'].values))
    row = {'epoch': epoch, 'train_loss': round(train_loss,4),
           'lr': round(optimizer.param_groups[0]['lr'],6), **metrics}
    history.append(row)
    print(f"E{epoch:2d} | loss={train_loss:.4f} | AUROC={metrics['auroc']:.4f} "
          f"| pF1_breast={metrics['pf1_breast']:.4f} | F1_breast={metrics['f1_breast']:.4f}")

    if metrics['pf1_breast'] > best_pf1:
        best_pf1 = metrics['pf1_breast']
        best_state = {k: v.cpu().clone() for k,v in model.state_dict().items()}
        torch.save(best_state, f"{CKPT_DIR}/mammoclip_b5_best.pth")
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f"⏹ Early stopping epoch {epoch}")
            break

if best_state: model.load_state_dict(best_state)
print(f"✅ Entraînement terminé en {(time.time()-t_start)/60:.1f} min — best pF1 val = {best_pf1:.4f}")''')

# ── Cell 10 : Évaluation test + TTA ──────────────────────────────────────────
code('''# TTA : moyenne image originale + flip horizontal
def run_eval_tta(loader, df):
    model.eval(); probs, labs = [], []
    with torch.no_grad():
        for X, y in tqdm(loader, desc='test+TTA', leave=False):
            X = X.to(DEVICE, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=USE_AMP):
                p1 = torch.sigmoid(model(X))
                p2 = torch.sigmoid(model(torch.flip(X, dims=[3])))
            p = ((p1+p2)/2).float().cpu().numpy()
            probs.extend(p.tolist()); labs.extend(y.numpy().tolist())
    return compute_metrics(df, probs, labs), probs, labs

df_test_full = X_test.assign(cancer=Y_test['cancer'].values)
test_metrics, test_probs, test_labs = run_eval_tta(test_loader, df_test_full)
print("📊 TEST — Mammo-CLIP B5")
for k, v in test_metrics.items():
    print(f"  {k:14s}: {v}")

# Sauvegarde
results = {'model': 'mammoclip_b5', 'best_pf1_val': best_pf1,
           'history': history, 'test_metrics': test_metrics,
           'config': {'img_size': IMG_SIZE, 'batch': BATCH_SIZE*ACCUM_STEPS,
                      'n_epochs': N_EPOCHS, 'lr_backbone': LR_BACKBONE, 'lr_head': LR_HEAD}}
with open(f"{RES_DIR}/mammoclip_b5.json", 'w') as f:
    json.dump(results, f, indent=2)
print(f"✅ Résultats → {RES_DIR}/mammoclip_b5.json")''')

# ── Cell 11 : Courbes ────────────────────────────────────────────────────────
code('''import matplotlib.pyplot as plt
ep = [h['epoch'] for h in history]
fig, ax = plt.subplots(1, 2, figsize=(13,4))
ax[0].plot(ep, [h['train_loss'] for h in history], 'o-', label='train loss')
ax[0].set_title('Loss'); ax[0].set_xlabel('epoch'); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].plot(ep, [h['auroc'] for h in history], 'o-', label='AUROC')
ax[1].plot(ep, [h['pf1_breast'] for h in history], 's-', label='pF1 breast')
ax[1].plot(ep, [h['f1_breast'] for h in history], '^-', label='F1 breast')
ax[1].set_title('Métriques val'); ax[1].set_xlabel('epoch'); ax[1].legend(); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig(f"{FIG_DIR}/mammoclip_b5_curves.png", dpi=110, bbox_inches='tight')
plt.close()

from sklearn.metrics import roc_curve
bp, bl = breast_aggregate(df_test_full, test_probs, test_labs)
fpr, tpr, _ = roc_curve(bl, bp)
plt.figure(figsize=(5,5))
plt.plot(fpr, tpr, label=f"AUROC={test_metrics['auroc_breast']:.3f}")
plt.plot([0,1],[0,1],'k--',alpha=.3); plt.xlabel('FPR'); plt.ylabel('TPR')
plt.title('ROC test (niveau sein)'); plt.legend(); plt.grid(alpha=.3)
plt.savefig(f"{FIG_DIR}/mammoclip_b5_roc.png", dpi=110, bbox_inches='tight'); plt.close()
print("✅ Figures sauvegardées")
print("\\n🎉 NOTEBOOK TERMINÉ")''')

# ════════════════════════════════════════════════════════════════════════════
nb = {
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.0"},
        "kaggle": {"accelerator": "nvidiaTeslaT4", "isGpuEnabled": True,
                   "isInternetEnabled": True, "language": "python", "sourceType": "notebook"}
    },
    "nbformat": 4, "nbformat_minor": 4, "cells": cells
}

out = sys.argv[1] if len(sys.argv) > 1 else "notebooks/rsna-mammoclip-assa.ipynb"
with open(out, 'w') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f"✅ Notebook généré : {out} ({len(cells)} cellules)")

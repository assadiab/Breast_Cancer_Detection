# 🎗️ Breast Cancer Detection

AI-assisted breast cancer detection on screening mammography — [RSNA Kaggle Competition](https://www.kaggle.com/competitions/rsna-breast-cancer-detection).

**Collaborative project — M2 Biologie-Informatique, Université Paris Cité**  
Encadrants : Tatiana Galochkina · Frédéric Guyon · Jean-Christophe Gelly

---

## Author

**Assa Diabira** — M2 Biologie-Informatique, Université Paris Cité  
Multi-Head Expert model: 4 medical backbones + cross-attention fusion

---

## Project Structure

```
Breast_Cancer_Detection/
├── main.py                        ← entry point : train / eval / infer
├── inference.py                   ← load model + predict on DICOM files
├── requirements.txt
├── README.md
├── .gitignore
│
├── core/                          ← shared utilities
│   ├── configuration.py
│   ├── dataset_manager.py
│   └── loader.py
│
├── preprocess/                    ← DICOM preprocessing pipeline
│   ├── pipeline.py                ← PreprocessPipeline (5 modes)
│   ├── cropping.py                ← ROI crop + pectoral muscle removal
│   ├── windowing.py               ← adaptive windowing by BI-RADS density
│   └── resampler.py               ← isotropic resampling
│
├── models/                        ← Multi-Head Expert model
│   ├── multi_head_expert.py       ← 4 expert backbones + fusion
│   ├── baseline_cnn.py            ← baseline for comparison
│   ├── losses.py                  ← FocalAUCLoss (70% Focal + 30% AUC)
│   ├── trainer.py                 ← Trainer class (OOP)
│   └── dataset.py                 ← MammographyDataset (PyTorch)
│
├── notebooks/
│   ├── eda.ipynb                  ← Exploratory Data Analysis
│   ├── preprocessing_benchmark.ipynb
│   ├── training_baseline.ipynb
│   └── training_multihead.ipynb
│
└── results/
    ├── metrics/                   ← JSON metrics from Kaggle runs
    └── figures/                   ← ROC curves, confusion matrices
```

---

## Model Architecture — Multi-Head Expert (DIABIRA v3)

```
DICOM image
    ↓ PreprocessPipeline (crop → adaptive windowing → resize)
float32 [1, H, W]  (grayscale, values in [0, 1])
    ↓
┌─────────────────────────────────────────────────────────────┐
│                    4 EXPERT BACKBONES                        │
│  1. EfficientNetV2-S  mammoscreen  (RSNA breast, AUC 0.945) │
│  2. DenseNet121       TorchXRayVision RSNA X-ray            │
│  3. ResNet50          RadImageNet (1.35M medical images)     │
│  4. ConvNeXt-Small    ImageNet-21k (RSNA Kaggle winner)      │
└─────────────────────────────────────────────────────────────┘
    ↓ each → 512-dim embedding
Expert-Aware Fusion:
  cross-attention (4 heads) → per-expert MLP → dynamic gating (softmax)
    ↓
MLP classifier: 512 → 256 → 128 → 1  [GELU, Dropout]
    ↓
BCEWithLogitsLoss  (no Sigmoid on model output)
```

106M parameters total. Freeze backbones for phase 1 → 6.3M trainable.

---

## Installation

```bash
git clone https://github.com/assadiab/Breast_Cancer_Detection.git
cd Breast_Cancer_Detection
pip install -r requirements.txt
```

Key dependencies: `torch`, `torchvision`, `timm`, `torchxrayvision`, `monai`, `huggingface_hub`, `safetensors`, `albumentations`, `pydicom`, `opencv-python`, `scikit-learn`.

**Optional — RadImageNet weights for Expert 3 (ResNet50):**
Download from [Google Drive](https://drive.google.com/file/d/1RHt2GnuOYlc_gcoTETtBDSW73mFyRAtR/view) and place in `checkpoints/radImageNet/`.  
Expert 1 (mammoscreen) and Expert 2 (TorchXRayVision) download automatically on first run.

---

## Training

### Quick start

```bash
# Multi-Head Expert model
python main.py train \
  --csv train.csv \
  --images-dir /path/to/dicom/images \
  --preprocess-mode full \
  --epochs 20 --batch-size 8 --lr 1e-4 \
  --checkpoint-dir checkpoints/

# Baseline CNN
python main.py train \
  --csv train.csv \
  --images-dir /path/to/dicom/images \
  --model baseline --epochs 30 \
  --checkpoint-dir checkpoints/baseline/
```

### Three-phase training (recommended)

```python
from models.multi_head_expert import MultiHeadMammoModel
from models.trainer import Trainer

model = MultiHeadMammoModel(embed_dim=512)

# Phase 1 — frozen backbones (6.3M trainable params)
model.freeze_backbones()
Trainer(model, train_loader, val_loader, device, lr=1e-3, n_epochs=10).train()

# Phase 2 — unfreeze last 2 blocks
model.unfreeze_backbones(last_n_blocks=2)
Trainer(model, train_loader, val_loader, device, lr=1e-4, n_epochs=10).train()

# Phase 3 — full fine-tuning (106M params)
model.unfreeze_all()
Trainer(model, train_loader, val_loader, device, lr=5e-5, n_epochs=10).train()
```

### Preprocessing modes

```python
from preprocess.pipeline import PreprocessPipeline

pipeline = PreprocessPipeline(config, mode="full", target_hw=(1024, 512))
# modes: "raw" | "crop_only" | "window_only" | "full" | "full_iso"

img = pipeline.process_one(patient_id, image_id, laterality, view, density, dicom_path)
# returns float32 numpy (H, W), values in [0, 1]
```

### Evaluation & inference

```bash
# Evaluate on validation set
python main.py eval \
  --csv val.csv \
  --images-dir /path/to/dicom \
  --checkpoint checkpoints/best_model.pth

# Predict on new DICOM files
python main.py infer \
  --images-dir /path/to/new/dicoms \
  --checkpoint checkpoints/best_model.pth \
  --output predictions.csv
```

---

## Preprocessing Pipeline

Three stages applied before model input:

**1. ROI Cropping** — Otsu thresholding → morphological ops → pectoral muscle removal (Hough lines, MLO views) → standardized left orientation → bounding box with mm margins.

**2. Adaptive Windowing** — Density-aware: percentile clipping → gamma correction → CLAHE (relative tile size) → weighted fusion. Tuned per BI-RADS density (A/B/C/D).

**3. Resize** to target resolution (default 1024×512).

---

## Results

| Model | ROC-AUC | PR-AUC | F1 | Recall |
|-------|---------|--------|-----|--------|
| EfficientNet-B0 | 0.63 | 0.14 | 0.13 | 0.18 |
| ConvNeXt-Base | 0.62 | 0.15 | 0.20 | 0.26 |
| ResNet-50 + Meta | 0.59 | 0.13 | 0.19 | 0.21 |
| **Multi-Head v1** | **0.66** | **0.15** | **0.22** | **0.23** |
| **Multi-Head v3** | *in progress — Kaggle GPU* | — | — | — |

Metrics from Kaggle GPU runs → [`results/metrics/`](results/metrics/)

---

## Dataset

[RSNA Screening Mammography Breast Cancer Detection](https://www.kaggle.com/competitions/rsna-breast-cancer-detection) — 54,706 DICOM images, 11,913 patients, 2.12% cancer rate.

Patient-wise stratified split (no data leakage): 70% train / 15% val / 15% test.  
Class imbalance: `pos_weight=13.7` + FocalLoss (γ=2.5, α=0.75) + patient-aware oversampling.

Data is not versioned here (DICOM files too large for GitHub).

---

## References

- mammoscreen : https://huggingface.co/ianpan/mammoscreen
- TorchXRayVision : https://github.com/mlmed/torchxrayvision
- RadImageNet : https://pubs.rsna.org/doi/full/10.1148/ryai.210315
- RSNA Kaggle winner (ConvNeXt) : https://pmc.ncbi.nlm.nih.gov/articles/PMC11048882/

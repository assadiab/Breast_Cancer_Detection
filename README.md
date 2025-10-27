# 🎗️ RSNA Breast Cancer Detection

## 📋 Overview

This project provides innovative solutions for the [RSNA Screening Mammography Breast Cancer Detection competition](https://www.kaggle.com/competitions/rsna-breast-cancer-detection) organized by the Radiological Society of North America (RSNA) in partnership with Kaggle.

Breast cancer is the most common cancer worldwide. In 2020, there were 2.3 million new diagnoses and 685,000 deaths. Our solutions aim to develop AI systems capable of assisting radiologists in early breast cancer detection on screening mammograms.

### 👥 Team Collaboration

This is a **collaborative project** with four different model architectures, each developed by a team member:

1. **DIABIRA** - Multi-Head Ensemble Architecture
2. **ABBASI** - Advanced Deep Learning Approach
3. **MANOUR** - Specialized CNN Model
4. **BENHAMOUCHE** - Hybrid Architecture


## 🗂️ Project Structure

```
rsna-breast-cancer-detection/
├── README.md
├── pixi.toml                      # Pixi configuration
├── .gitignore
│── EDA.ipynb                      # Exploratory Data Analysis
│
├── core/                          # Core utilities and shared modules
│   ├── __init__.py
│   ├── config.py                 # Configuration management
│   ├── data_loader.py            # Data loading utilities
│   └── utils.py                  # General utilities
│
├── preprocess/                    # 📐 Preprocessing modules
│   ├── __init__.py
│   ├── dicom_processor.py        # DICOM loading and processing
│   ├── normalization.py          # Intensity normalization
│   ├── augmentation.py           # Data augmentation
│   ├── cleaning.py               # Data cleaning strategies
│   └── splitter.py               # Train/val/test splitting
│
├── models/                        # 🧠 Model architectures (4 approaches)
│   │
│   ├── DIABIRA/                  # 1️⃣ Primary Model - Multi-Head Ensemble
│   │   ├── __init__.py
│   │   ├── multi_head_model.py   # Main architecture
│   │   ├── heads/                # Specialized heads
│   │   │   ├── efficient_net.py
│   │   │   ├── convnext.py
│   │   │   └── swin_transformer.py
│   │   ├── fusion.py             # Fusion mechanism
│   │   ├── losses.py             # Adaptive Focal Loss
│   │   ├── train.py              # Training script
│   │   └── README.md             # Model-specific documentation
│   │
│   ├── ABBASI/                   # 2️⃣ Second Model
│   │   ├── __init__.py
│   │   ├── model.py
│   │   ├── train.py
│   │   └── README.md
│   │
│   ├── MANOUR/                   # 3️⃣ Third Model
│   │   ├── __init__.py
│   │   ├── model.py
│   │   ├── train.py
│   │   └── README.md
│   │
│   └── BENHAMOUCHE/              # 4️⃣ Fourth Model
│       ├── __init__.py
│       ├── model.py
│       ├── train.py
│       └── README.md
│
├── data/                          # Data (not versioned)
│   ├── raw/                      # Original DICOM files
│   │   ├── train_images/
│   │   └── test_images/
│   ├── processed/                # Preprocessed data
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── eda/                      # EDA outputs
│
├── scripts/                       # Utility scripts
│   ├── download_data.py          # Download from Kaggle
│   ├── run_preprocessing.py      # Run preprocessing pipeline
│   ├── compare_models.py         # Compare all 4 models
│   └── create_submission.py      # Create submission file
│
├── checkpoints/                   # Model checkpoints (not versioned)
│   ├── DIABIRA/
│   ├── ABBASI/
│   ├── MANOUR/
│   └── BENHAMOUCHE/
│
├── runs/                          # TensorBoard logs (not versioned)
├── results/                       # Results and analysis
└── submissions/                   # Submission files
```

## 📊 Exploratory Data Analysis (EDA)

Before building models, we conducted comprehensive exploratory data analysis documented in `notebooks/EDA.ipynb`:


### EDA Notebook Contents

The `notebooks/EDA.ipynb` includes:
1. **Data Loading & Validation**: CSV and DICOM file structure
2. **Statistical Analysis**: Distributions, correlations, chi-square tests
3. **Visualization**: Class balance, age distribution, site comparison
4. **Image Analysis**: Sample images, intensity histograms, view types
5. **Consistency Checks**: CSV-to-image mapping verification
6. **Insights & Recommendations**: Preprocessing strategy, model requirements

## 🔧 Core Modules

The `core/` directory contains shared utilities used across all models:

### config.py
Configuration management for experiments:
```python
from core.config import Config

config = Config()
config.batch_size = 16
config.learning_rate = 1e-4
config.device = 'auto'  # Auto-detect GPU/CPU
```

### data_loader.py
Unified data loading interface:
```python
from core.data_loader import get_dataloaders

train_loader, val_loader, test_loader = get_dataloaders(
    data_dir='data/processed',
    batch_size=16,
    num_workers=8
)
```

### metrics.py
Comprehensive evaluation metrics:
```python
from core.metrics import calculate_metrics

metrics = calculate_metrics(y_true, y_pred, y_proba)
# Returns: f1, precision, recall, auc_roc, auc_pr, specificity
```

### visualization.py
Plotting and visualization tools:
```python
from core.visualization import plot_training_curves, plot_confusion_matrix

plot_training_curves(history, save_path='results/training.png')
plot_confusion_matrix(y_true, y_pred, save_path='results/cm.png')
```

## 📐 Preprocessing Pipeline

The `preprocess/` directory contains modular preprocessing components:

### 1. DICOM Processing (`dicom_processor.py`)

Load and decode DICOM medical images:
```python
from preprocess.dicom_processor import DICOMProcessor

processor = DICOMProcessor()
pixel_array = processor.load_dicom(dicom_path)
# Handles: MONOCHROME1/2, decompression, metadata extraction
```

**Features:**
- ✅ Automatic MONOCHROME1 inversion
- ✅ DICOM metadata extraction
- ✅ Support for compressed formats
- ✅ Batch processing capabilities

### 2. Normalization (`normalization.py`)

Intensity normalization for mammography:
```python
from preprocess.normalization import normalize_mammogram

normalized = normalize_mammogram(
    image,
    window_center='auto',  # Auto-detect optimal window
    window_width='auto',
    target_range=(0, 255)
)
```

**Techniques:**
- 🎚️ Windowing/Level adjustment
- 📊 Histogram equalization
- 🔢 Z-score normalization
- 🎯 Percentile-based clipping

### 3. Data Augmentation (`augmentation.py`)

Adaptive augmentation based on class:
```python
from preprocess.augmentation import get_augmentation_pipeline

# Different augmentation for cancer vs non-cancer
aug_positive = get_augmentation_pipeline(is_cancer=True, intensity='high')
aug_negative = get_augmentation_pipeline(is_cancer=False, intensity='low')
```

**Transformations:**
- 🔄 Rotation (±15°)
- ↔️ Horizontal/Vertical flips
- 🎨 Contrast adjustment
- 🔍 Random zoom
- 🌊 Elastic deformation (for cancer cases)

### 4. Data Cleaning (`cleaning.py`)

Strategic undersampling to handle imbalance:
```python
from preprocess.cleaning import clean_dataset

df_cleaned = clean_dataset(
    df_original,
    neg_fraction=0.25,  # Keep 25% of negative patients
    keep_all_cancer=True,  # Never remove cancer cases
    random_seed=42
)
```

**Strategy:**
- ✅ Keep **ALL** cancer patients (483 patients)
- 📉 Sample **25%** of non-cancer patients
- 🎯 Result: 2.12% → 7.32% cancer rate
- 💾 Reduce dataset from 293GB → 83GB

### 5. Train/Val/Test Splitting (`splitter.py`)

Patient-level stratified splitting to prevent data leakage:
```python
from preprocess.splitter import split_dataset

train_df, val_df, test_df = split_dataset(
    df_cleaned,
    train_size=0.70,
    val_size=0.15,
    test_size=0.15,
    stratify_by='patient_id',  # CRITICAL: No patient in multiple sets
    random_seed=42
)
```

**Key Features:**
- 👤 **Patient-level split** (prevents data leakage)
- ⚖️ Stratified by cancer status
- 📊 Maintains class balance across splits
- 🔒 Reproducible with fixed random seed

### Preprocessing Statistics

| Stage          | Patients | Cancer Patients | Images | Cancer Images | Size (GB) | Cancer Rate |
|----------------|----------|-----------------|--------|---------------|-----------|-------------|
| **Initial**    | 11,913   | 486 (4.08%)    | 54,706 | 1,158 (2.12%) | 293.10    | 2.12%       |
| **Cleaned**    | 3,344    | 483 (14.44%)   | 13,379 | 979 (7.32%)   | 82.87     | 7.32%       |
| **Train**      | 2,340    | 338 (14.44%)   | 9,363  | 685 (7.32%)   | 57.99     | 7.32%       |
| **Validation** | 502      | 73 (14.54%)    | 2,008  | 150 (7.47%)   | 12.35     | 7.47%       |
| **Test**       | 502      | 72 (14.34%)    | 2,008  | 144 (7.17%)   | 12.52     | 7.17%       |

## 🧠 Model 1: DIABIRA - Multi-Head Ensemble Architecture

Our primary model uses a multi-head ensemble approach with specialized experts.

### Architecture Overview

```
                    ┌─────────────────────────────┐
                    │   DICOM Image (1 channel)  │
                    └─────────────┬───────────────┘
                                  │
                    ┌─────────────┴───────────────┐
                    │    DICOM Preprocessing      │
                    │  (from preprocess/ module)  │
                    └─────────────┬───────────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        │                         │                         │
   ┌────▼────┐             ┌─────▼──────┐          ┌──────▼──────┐
   │  Head 1 │             │   Head 2   │          │   Head 3    │
   │EfficientNet│          │EfficientNet│          │  ConvNeXt   │
   │   V2-S  │             │    B1      │          │   Small     │
   │(Detection)│           │ (Texture)  │          │ (Context)   │
   └────┬────┘             └─────┬──────┘          └──────┬──────┘
        │                         │                         │
        │                    ┌────▼────┐                    │
        │                    │  Head 4 │                    │
        │                    │  Swin   │                    │
        │                    │Transform│                    │
        │                    │ (Global)│                    │
        │                    └────┬────┘                    │
        │                         │                         │
        └─────────────────────────┼─────────────────────────┘
                                  │
                    ┌─────────────▼───────────────┐
                    │  Fusion Module              │
                    │  • Multi-head Attention     │
                    │  • Adaptive Weighting       │
                    └─────────────┬───────────────┘
                                  │
                    ┌─────────────▼───────────────┐
                    │  Final Prediction           │
                    │  (Cancer Probability)       │
                    └─────────────────────────────┘
```

### The 4 Specialized Heads

1. **RSNA Detector Head** (`heads/efficient_net.py`)
   - Backbone: Pre-trained EfficientNetV2-S
   - Specialization: Localized lesion detection
   - Focus: Microcalcifications and suspicious masses

2. **Kaggle Texture Head** (`heads/efficient_net.py`)
   - Backbone: EfficientNet-B1
   - Specialization: Textural property analysis
   - Focus: Homogeneity, granularity, local contrasts

3. **ConvNeXt Context Head** (`heads/convnext.py`)
   - Backbone: ConvNeXt-Small
   - Specialization: Large-scale contextual information
   - Focus: Spatial distribution, density, symmetry

4. **Swin Global Vision Head** (`heads/swin_transformer.py`)
   - Backbone: Swin Transformer
   - Specialization: Hierarchical global understanding
   - Focus: Long-range relationships, overall view

### Key Features

- **Adaptive Focal Loss**: Handles class imbalance
- **Multi-head Attention**: Dynamic expert weighting
- **Progressive Training**: Warmup → Fine-tuning → Refinement
- **GPU Optimization**: Automatic resource detection

For detailed information, see [`models/DIABIRA/README.md`](models/DIABIRA/README.md)

## 🤖 Other Models

### Model 2: ABBASI
See [`models/ABBASI/README.md`](models/ABBASI/README.md) for architecture details and results.

### Model 3: MANOUR
See [`models/MANOUR/README.md`](models/MANOUR/README.md) for architecture details and results.

### Model 4: BENHAMOUCHE
See [`models/BENHAMOUCHE/README.md`](models/BENHAMOUCHE/README.md) for architecture details and results.

## 🚀 Installation

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for NVIDIA GPU) or Metal (for Mac M-series)
- 16GB+ RAM
- 50GB+ disk space
- Pixi package manager

### Installation with Pixi (Recommended)

#### Install Pixi

**macOS/Linux:**
```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

**Windows:**
```powershell
iwr -useb https://pixi.sh/install.ps1 | iex
```

#### Setup Project

```bash
# Clone the repository
git clone https://github.com/your-username/rsna-breast-cancer-detection.git
cd rsna-breast-cancer-detection

# Install dependencies with Pixi
pixi install

# Activate the environment
pixi shell

# Verify installation
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

## 🎮 Usage

### 1. Explore the Data (EDA)

```bash
# Launch Jupyter notebook
pixi run notebook

# Open notebooks/EDA.ipynb
# Explore data distributions, visualizations, and insights
```

### 2. Preprocess the Data

```bash
# Run full preprocessing pipeline
pixi run preprocess-clean

# Or run step-by-step
python -m preprocess.dicom_processor --input data/raw/train_images
python -m preprocess.cleaning --neg_fraction 0.25
python -m preprocess.splitter --train_size 0.70
```

### 3. Train Models

#### Train DIABIRA Model (Primary)
```bash
# Using Pixi task
pixi run train-diabira

# Or manually
cd models/DIABIRA
python train.py --data_dir ../../data/processed --epochs 30
```

#### Train Other Models
```bash
# ABBASI model
cd models/ABBASI
python train.py

# MANOUR model
cd models/MANOUR
python train.py

# BENHAMOUCHE model
cd models/BENHAMOUCHE
python train.py
```

### 4. Compare Models

```bash
# Run model comparison script
python scripts/compare_models.py \
    --models DIABIRA ABBASI MANOUR BENHAMOUCHE \
    --metrics f1 auc_roc auc_pr \
    --output results/comparison.csv
```

### 5. Create Submission

```bash
# Generate predictions from best model
python scripts/create_submission.py \
    --model DIABIRA \
    --checkpoint checkpoints/DIABIRA/best_model.pth \
    --output submissions/submission.csv
```

## 📈 Results and Metrics

### Evaluation Metrics
- **F1-Score**: Balance between precision and recall
- **AUC-ROC**: Overall classification performance
- **AUC-PR**: Particularly suited for imbalanced classes
- **Sensitivity** (Recall): True positive detection rate
- **Specificity**: True negative identification rate

### Model Comparison (Example)

| Model       | F1-Score | AUC-ROC | AUC-PR | Sensitivity | Specificity | Training Time |
|-------------|----------|---------|--------|-------------|-------------|---------------|
| DIABIRA     | 0.512    | 0.891   | 0.389  | 0.654       | 0.921       | 8h 23m        |
| ABBASI      | TBD      | TBD     | TBD    | TBD         | TBD         | TBD           |
| MANOUR      | TBD      | TBD     | TBD    | TBD         | TBD         | TBD           |
| BENHAMOUCHE | TBD      | TBD     | TBD    | TBD         | TBD         | TBD           |

*Note: Fill in results as models are trained*

## 🎲 Strategies for Imbalanced Classes

All models implement strategies to handle the severe class imbalance:

### 1. Adaptive Focal Loss
```python
# Penalizes errors on minority class more heavily
Loss = -α(1-p)^γ log(p)
```

### 2. Adaptive Data Augmentation
- **For positive cases (cancer)**: Aggressive augmentation
- **For negative cases**: Minimal augmentation

### 3. Stratified Resampling
- Oversampling positive cases
- Light undersampling of negative cases
- Patient-stratified validation

### 4. Optimized Decision Threshold
- Threshold calibration to maximize F1-score
- Precision-Recall curve optimization

## ⚙️ GPU/CPU Optimizations

Training pipeline automatically detects and optimizes resource utilization:

### Automatic Detection
- Device (MPS for Mac M-series, CUDA for NVIDIA, CPU otherwise)
- Number of available GPUs
- GPU and CPU RAM
- Optimal number of workers
- Optimal batch size

### Implemented Optimizations
- ✅ Multi-GPU Parallelization
- ✅ Asynchronous Data Loading
- ✅ Pin Memory (CUDA)
- ✅ Gradient Accumulation
- ✅ Mixed Precision Training
- ✅ Validation Caching
- ✅ Gradient Checkpointing

## 🤝 Contributing

This is a collaborative project. Each team member maintains their model in a separate directory.

### Adding a New Model

1. Create directory: `models/YOUR_NAME/`
2. Implement model architecture
3. Create training script
4. Add README with architecture details
5. Update main README with model description

### Code Guidelines
- Documented and commented code
- Follow PEP 8 conventions
- Use shared utilities from `core/` and `preprocess/`
- Update model comparison table with results

## 👥 Team Members

- **Assa DIABIRA** - Multi-Head Ensemble Architecture
- **ABBASI** - Advanced Deep Learning Approach
- **MANOUR** - Specialized CNN Model  
- **BENHAMOUCHE** - Hybrid Architecture

## 📝 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **RSNA** for organizing the competition and providing the dataset
- **Kaggle** for the competition platform
- Mammography screening programs in Australia and the US for the data
- PyTorch and timm communities for model implementations

## 📚 References

### Scientific Papers
- [Deep Learning for Breast Cancer Detection](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10077079/)
- [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
- [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298)
- [Swin Transformer: Hierarchical Vision Transformer](https://arxiv.org/abs/2103.14030)

### Competition and Datasets
- [RSNA Screening Mammography Breast Cancer Detection](https://www.kaggle.com/competitions/rsna-breast-cancer-detection)
- [RSNA Official Challenge Page](https://www.rsna.org/rsnai/ai-image-challenge/screening-mammography-breast-cancer-detection-ai-challenge)
- [1st Place Solution](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/discussion/392449)

## 📧 Contact

For questions or suggestions, feel free to open an issue or contact team members.

---

⭐ If this project was useful to you, don't forget to give it a star!

🎗️ **Together against breast cancer** - Every early detection counts.
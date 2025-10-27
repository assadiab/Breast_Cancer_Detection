# 🎗️ Breast Cancer Detection

## Overview

This project provides innovative solutions for the [RSNA Screening Mammography Breast Cancer Detection competition](https://www.kaggle.com/competitions/rsna-breast-cancer-detection) organized by the Radiological Society of North America (RSNA) in partnership with Kaggle.

Breast cancer is the most common cancer worldwide. In 2020, there were 2.3 million new diagnoses and 685,000 deaths. Our solutions aim to develop AI systems capable of assisting radiologists in early breast cancer detection on screening mammograms.

### Team Collaboration

This is a **collaborative project** with four different model architectures.
Each team member maintains their model in a separate directory under **[models/](models)**.

1. **[DIABIRA](models/DIABIRA)** - Multi-Head Ensemble Architecture
2. **[ABBASI](models/ABBASI)** - Advanced Deep Learning Approach
3. **[MANOUR](models/MANOUR)** - Specialized CNN Model
4. **[BENHAMOUCHE](models/BENHAMOUCHE)** - Hybrid Architecture

### Objectives

- **Early Detection**: Identify cancer signs on screening mammograms
- **High Accuracy**: Maximize sensitivity while minimizing false positives
- **Robustness**: Handle severe class imbalance (~7% positive cases)
- **Performance**: Optimize GPU/CPU resource utilization for efficient training
- **Comparison**: Evaluate multiple architectural approaches

## Project Structure

```
Breast_Cancer_Detection/
├── README.md
├── pixi.toml                      # Pixi configuration
├── .gitignore
├── rsna_data-eda.ipynb           # Exploratory Data Analysis
│
├── core/                          # Core utilities and shared modules
│   ├── __init__.py
│   ├── configuration.py          # Configuration management
│   ├── dataset_manager.py        # Dataset management utilities
│   └── loader.py                 # Data loading utilities
│
├── preprocess/                    # Preprocessing modules
│   ├── __init__.py
│   ├── cropping.py               # Image cropping utilities
│   ├── resampler.py              # Resampling operations
│   └── windowing.py              # Intensity windowing
│
├── models/                        # Model architectures (4 approaches)
│   ├── DIABIRA/                
│   ├── ABBASI/                  
│   ├── MANOUR/                 
│   └── BENHAMOUCHE/           
│
└──data/                          # Data (not versioned)
    ├── raw/                      # Original DICOM files
    └── processed/                # Preprocessed data
```

## Exploratory Data Analysis (EDA)

Comprehensive exploratory data analysis is documented in **[rsna_data-eda.ipynb](rsna_data-eda.ipynb)**:
1. **Data Loading & Validation**: CSV and DICOM file structure
2. **Statistical Analysis**: Distributions, correlations, chi-square tests
3. **Visualization**: Class balance, age distribution, site comparison
4. **Image Analysis**: Sample images, intensity histograms, view types
5. **Consistency Checks**: CSV-to-image mapping verification
6. **Insights & Recommendations**: Preprocessing strategy, model requirements

### Key Findings

**Dataset Overview:**
-  **54,706 images** from **11,913 patients**
-  Multiple sites and imaging machines
-  **4 images per patient** on average (MLO and CC views, left and right breasts)

**Class Imbalance Challenge:**
-  Initial cancer rate: **2.12%** (1,158 cancer images / 54,706 total)
-  Only **4.08%** of patients have cancer (486 / 11,913)
-  Major challenge requiring specialized techniques

## Usage

### 1. Installation with Pixi
```bash
# Clone the repository
git clone https://github.com/assadiabira/Breast_Cancer_Detection.git
cd Breast_Cancer_Detection

# Install dependencies with Pixi
pixi install

# Activate the environment
pixi shell
```

### 2. Explore the Data (EDA)

```bash
# Launch Jupyter notebook
jupyter notebook rsna_data-eda.ipynb

# Or with Pixi
pixi run jupyter notebook rsna_data-eda.ipynb
```

### 3. Preprocess the Data

```python
# Example preprocessing pipeline
from preprocess.windowing import apply_window
from preprocess.cropping import crop_breast_region
from preprocess.resampler import resample_image

# Load DICOM
import pydicom
dcm = pydicom.dcmread('path/to/image.dcm')
image = dcm.pixel_array

# Apply preprocessing
windowed = apply_window(image)
cropped = crop_breast_region(windowed)
resampled = resample_image(cropped, target_size=(224, 224))
```

### Competition and Datasets
- [RSNA Screening Mammography Breast Cancer Detection](https://www.kaggle.com/competitions/rsna-breast-cancer-detection)
- [RSNA Official Challenge Page](https://www.rsna.org/rsnai/ai-image-challenge/screening-mammography-breast-cancer-detection-ai-challenge)

# Image de base PyTorch + CUDA (GPU). Pour du CPU-only, remplacer par python:3.11-slim.
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

# Dépendances système pour OpenCV / GDCM
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work

# Dépendances Python (cache de layer : on copie d'abord requirements.txt)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code du projet
COPY . .

# Par défaut : régénère le notebook d'entraînement depuis le script source
CMD ["python", "scripts/build_notebook_multihead.py", "kaggle/train_multihead/rsna-mammoclip-multihead.ipynb"]

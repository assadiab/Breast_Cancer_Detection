#!/bin/bash

# Configuration MPS
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export KMP_DUPLICATE_LIB_OK=TRUE

echo "🚀 Démarrage avec GPU Apple MPS..."
echo ""

pixi run python /Users/assadiabira/Bureau/Kaggle/Projet_kaggle/models/train.py
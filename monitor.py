#!/usr/bin/env python3
import pandas as pd
import time
import os

while True:
    os.system('clear')
    print("📊 TRAINING LIVE MONITOR")
    print("=" * 80)

    try:
        df = pd.read_csv('checkpoints/training_history.csv')

        # Dernière ligne
        last = df.iloc[-1]

        print(f"\n🏃 Epoch: {int(last['epoch'])}")
        print(f"⏱️  Time: {last['epoch_time']:.1f}s")
        print(f"\n📉 Loss: Train={last['train_loss']:.4f} | Val={last['val_loss']:.4f}")
        print(f"🎯 F1: {last['val_f1']:.4f}")
        print(f"📊 Precision: {last['val_precision']:.4f} | Recall: {last['val_recall']:.4f}")
        print(f"📈 AUC-ROC: {last['val_auc_roc']:.4f}")

        # Best
        best_f1 = df['val_f1'].max()
        best_epoch = df.loc[df['val_f1'].idxmax(), 'epoch']
        print(f"\n⭐ Best F1: {best_f1:.4f} (Epoch {int(best_epoch)})")

        # Dernières 5 epochs
        if len(df) >= 5:
            print(f"\n📈 Last 5 F1: {df['val_f1'].tail(5).tolist()}")

        print(f"\n{df.tail(5).to_string(index=False)}")

    except FileNotFoundError:
        print("⏳ Waiting for training to start...")
    except Exception as e:
        print(f"Error: {e}")

    print(f"\n🔄 Refreshing every 5s... (Ctrl+C to stop)")
    time.sleep(5)
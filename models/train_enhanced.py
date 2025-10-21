# train_enhanced.py
from enhanced_losses import WeightedFocalLoss, AUCMLoss
from progressive_training import ProgressiveTrainer


class EnhancedTrainer(Trainer):
    def __init__(self, config):
        super().__init__(config)
        self.progressive_trainer = ProgressiveTrainer(self.model, config)

        # Calcul des poids pour déséquilibre (basé sur vos stats)
        cancer_rate = 0.0732  # 7.32% d'images avec cancer
        self.class_weights = torch.tensor([
            1.0,  # Poids classe 0 (normal)
            1.0 / cancer_rate  # Poids classe 1 (cancer)
        ]).to(self.device)

    def setup_loss(self, loss_name):
        """Configure la loss selon le stage"""
        if loss_name == 'focal':
            return WeightedFocalLoss(
                alpha=0.25,
                gamma=2.0,
                class_weights=self.class_weights
            )
        elif loss_name == 'aucm':
            return AUCMLoss(imratio=0.0732)
        else:
            return nn.BCEWithLogitsLoss()

    def enhanced_fit(self, train_csv, val_csv, train_y_csv, val_y_csv, dicom_root):
        """Training progressif avec stages"""

        # Chargement des données
        train_ds = OptimizedDicomDataset(train_csv, dicom_root, train_y_csv, self.cfg, True)
        val_ds = OptimizedDicomDataset(val_csv, dicom_root, val_y_csv, self.cfg, True)

        # DataLoader avec weighted sampling pour déséquilibre
        weights = [2.0 if label == 1 else 0.5 for label in train_ds.labels['cancer']]
        sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))

        train_loader = DataLoader(
            train_ds,
            batch_size=self.cfg.batch_size,
            sampler=sampler,  # Utiliser sampler au lieu de shuffle
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory
        )

        # Training par stages
        stages = self.progressive_trainer.training_stages()

        for stage_idx, stage_config in enumerate(stages):
            print(f"\n{'=' * 50}")
            print(f"STAGE {stage_idx + 1}/{len(stages)}")
            print(f"{'=' * 50}")

            # Configuration du stage
            self.progressive_trainer.setup_stage(stage_config)
            self.criterion = self.setup_loss(stage_config['loss'])

            # Optimizer pour le stage
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=stage_config['lr'],
                weight_decay=self.cfg.weight_decay
            )

            # Entraînement du stage
            for epoch in range(stage_config['epochs']):
                train_loss = self.train_epoch(train_loader, optimizer, epoch)
                val_loss = self.validate(val_loader)

                print(f"Stage {stage_idx + 1} Epoch {epoch}: "
                      f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

                # Sauvegarde best model
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.save_checkpoint(
                        f"{self.cfg.save_dir}/best_stage{stage_idx + 1}.pth",
                        epoch, val_loss
                    )


# Configuration finale
def get_optimized_config():
    return OptimizedConfig(
        epochs=15,
        batch_size=4,
        grad_accum_steps=4,
        lr=2e-4,
        high_res=(512, 512),
        low_res=(224, 224),
        embed_dim=256,
        use_checkpoint=True,
        freeze_epochs=3,
        focal_alpha=0.25,
        focal_gamma=2.0,
        save_dir="checkpoints_enhanced"
    )


if __name__ == "__main__":
    cfg = get_optimized_config()
    trainer = EnhancedTrainer(cfg)

    # Chemins (à adapter)
    DATA_ROOT = "/Users/assadiabira/Bureau/Kaggle/Projet_kaggle/data"

    trainer.enhanced_fit(
        train_csv=f"{DATA_ROOT}/csv/X_train.csv",
        val_csv=f"{DATA_ROOT}/csv/X_val.csv",
        train_y_csv=f"{DATA_ROOT}/csv/y_train.csv",
        val_y_csv=f"{DATA_ROOT}/csv/y_val.csv",
        dicom_root=f"{DATA_ROOT}/train"
    )
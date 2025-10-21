# progressive_training.py
class ProgressiveTrainer:
    def __init__(self, model, config):
        self.model = model
        self.cfg = config
        self.current_stage = 0

    def training_stages(self):
        """Stages progressifs pour stabiliser l'entraînement"""
        stages = [
            {
                'name': 'Warmup - Features bas niveau',
                'epochs': 3,
                'frozen_heads': ['context', 'texture'],  # Geler backbones lourds
                'lr': 1e-4,
                'loss': 'focal'
            },
            {
                'name': 'Unfreeze contextuel',
                'epochs': 5,
                'frozen_heads': [],
                'lr': 5e-5,
                'loss': 'focal'
            },
            {
                'name': 'Fine-tuning complet',
                'epochs': 7,
                'frozen_heads': [],
                'lr': 1e-5,
                'loss': 'aucm'  # Passer à AUC-M pour optimisation finale
            }
        ]
        return stages

    def setup_stage(self, stage_config):
        """Configure le modèle pour le stage actuel"""
        print(f"\n🎯 Stage: {stage_config['name']}")

        # Geler/dégeler les têtes
        for head_name in ['detector', 'texture', 'context', 'segment']:
            head = getattr(self.model, head_name)
            for param in head.parameters():
                param.requires_grad = head_name not in stage_config['frozen_heads']

        # Compter les paramètres entraînables
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Params entraînables: {trainable_params:,}")
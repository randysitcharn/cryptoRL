"""
training - Module pour l'entraînement des modèles.

Contient:
- train_foundation: Script d'entraînement MAE
"""

from src.training.train_foundation import train, TrainingConfig

__all__ = ['train', 'TrainingConfig']

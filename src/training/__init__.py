"""
training - Module pour l'entraînement des modèles.

Contient:
- train_foundation: Script d'entraînement MAE (Phase 2)
- train_agent: Script d'entraînement TQC + Foundation (Phase 3)
- env: Environnement de trading RL
"""

from src.training.train_foundation import train as train_foundation, TrainingConfig
from src.training.env import CryptoTradingEnv
from src.training.train_agent import train as train_agent, TrainingConfig as AgentTrainingConfig

__all__ = [
    'train_foundation',
    'train_agent',
    'TrainingConfig',
    'AgentTrainingConfig',
    'CryptoTradingEnv',
]

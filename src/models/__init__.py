"""
models - Module pour les modèles ML/DL.

Contient:
- CryptoMAE: Masked Auto-Encoder pour pré-entraînement
- TQC Agent: Factory pour l'agent RL (requiert sb3_contrib)
"""

# Foundation models (toujours disponibles)
from src.models.foundation import CryptoMAE, SinusoidalPositionalEncoding

__all__ = ['CryptoMAE', 'SinusoidalPositionalEncoding']

# RL agents (optionnels, requièrent sb3_contrib)
try:
    from src.models.agent import create_tqc_agent, create_agent
    from src.models.transformer_policy import TransformerFeatureExtractor
    from src.models.rl_adapter import FoundationFeatureExtractor
    from src.models.callbacks import TensorBoardStepCallback
    from src.models.tqc_dropout_policy import (
        TQCDropoutPolicy,
        DropoutActor,
        DropoutCritic,
        create_mlp_with_dropout,
    )
    from src.models.robust_actor import RobustDropoutActor

    __all__.extend([
        'create_tqc_agent',
        'create_agent',
        'TransformerFeatureExtractor',
        'FoundationFeatureExtractor',
        'TensorBoardStepCallback',
        'TQCDropoutPolicy',
        'DropoutActor',
        'DropoutCritic',
        'create_mlp_with_dropout',
        'RobustDropoutActor',
    ])
except ImportError:
    pass  # sb3_contrib not installed

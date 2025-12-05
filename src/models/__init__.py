"""
models - Module pour les agents RL.

Contient la factory TQC avec Transformer et les architectures neuronales custom.
"""

from src.models.agent import create_tqc_agent, create_agent
from src.models.transformer_policy import TransformerFeatureExtractor
from src.models.callbacks import TensorBoardStepCallback

__all__ = ['create_tqc_agent', 'create_agent', 'TransformerFeatureExtractor', 'TensorBoardStepCallback']

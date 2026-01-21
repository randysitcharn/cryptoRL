# -*- coding: utf-8 -*-
"""
base.py - Base configuration with shared parameters.

Provides common settings used across training, tuning, and evaluation.
"""

from dataclasses import dataclass
import torch


# =============================================================================
# Global Settings
# =============================================================================

SEED: int = 42


def get_device() -> torch.device:
    """Detect optimal device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


DEVICE: torch.device = get_device()


# =============================================================================
# Path Configuration
# =============================================================================

@dataclass
class PathConfig:
    """Common paths used across the project."""

    data_path: str = "data/processed_data.parquet"
    encoder_path: str = "weights/pretrained_encoder.pth"
    checkpoint_dir: str = "weights/checkpoints/"
    tensorboard_log: str = "logs/tensorboard/"
    results_dir: str = "results/"


# =============================================================================
# Environment Configuration
# =============================================================================

@dataclass
class EnvConfig:
    """Trading environment parameters."""

    # Window and structure
    window_size: int = 64
    episode_length: int = 2048

    # Transaction costs
    commission: float = 0.0006  # 0.06% per trade
    slippage: float = 0.0001    # 0.01% slippage

    # Data split
    train_ratio: float = 0.8

    # Reward function
    reward_scaling: float = 30.0
    downside_coef: float = 1.0
    upside_coef: float = 0.0
    action_discretization: float = 0.1

    # Volatility scaling
    target_volatility: float = 0.01
    vol_window: int = 24
    max_leverage: float = 5.0


# =============================================================================
# Foundation Model Configuration
# =============================================================================

@dataclass
class FoundationModelConfig:
    """Foundation model (Transformer encoder) architecture."""

    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    freeze_encoder: bool = True

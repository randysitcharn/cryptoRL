# -*- coding: utf-8 -*-
"""
base.py - Base configuration with shared parameters.

Provides common settings used across training, tuning, and evaluation.
"""

from dataclasses import dataclass
import torch

from src.config.constants import (
    MAE_D_MODEL, MAE_N_HEADS, MAE_N_LAYERS, MAE_DROPOUT,
    DEFAULT_TARGET_VOLATILITY, DEFAULT_VOL_WINDOW, DEFAULT_MAX_LEVERAGE,
)


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

    # Volatility scaling (Single Source of Truth: constants.py)
    target_volatility: float = DEFAULT_TARGET_VOLATILITY
    vol_window: int = DEFAULT_VOL_WINDOW
    max_leverage: float = DEFAULT_MAX_LEVERAGE


# =============================================================================
# Foundation Model Configuration
# =============================================================================

@dataclass
class FoundationModelConfig:
    """Foundation model (Transformer encoder) architecture."""

    d_model: int = MAE_D_MODEL
    n_heads: int = MAE_N_HEADS
    n_layers: int = MAE_N_LAYERS
    dropout: float = MAE_DROPOUT
    freeze_encoder: bool = True

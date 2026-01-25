# -*- coding: utf-8 -*-
"""
config - Centralized configuration module for cryptoRL.

Exports all configuration classes and global settings.

Usage:
    from src.config import SEED, DEVICE, TQCTrainingConfig
    from src.config import FoundationTrainingConfig
"""

# Global settings (backward compatibility with src.config)
from src.config.base import (
    SEED,
    DEVICE,
    get_device,
    PathConfig,
    EnvConfig,
    FoundationModelConfig,
)

# Shared constants
from src.config.constants import (
    OHLCV_COLS,
    EXCLUDE_COLS,
    CRYPTO_TICKERS,
    MACRO_TICKERS,
    TICKER_MAPPING,
    MAX_LOOKBACK_WINDOW,
    DEFAULT_PURGE_WINDOW,
    HMM_FEATURE_PREFIXES,
    HMM_CONTEXT_COLS,
    MAE_D_MODEL,
    MAE_N_HEADS,
    MAE_N_LAYERS,
    MAE_DIM_FEEDFORWARD,
    MAE_DROPOUT,
)

# Training configurations
from src.config.training import (
    TQCTrainingConfig,
    WFOTrainingConfig,
    FoundationTrainingConfig,
    linear_schedule,
)

# Backward compatibility alias
TrainingConfig = TQCTrainingConfig

__all__ = [
    # Global
    "SEED",
    "DEVICE",
    "get_device",
    # Constants
    "OHLCV_COLS",
    "EXCLUDE_COLS",
    "CRYPTO_TICKERS",
    "MACRO_TICKERS",
    "TICKER_MAPPING",
    "MAX_LOOKBACK_WINDOW",
    "DEFAULT_PURGE_WINDOW",
    "HMM_FEATURE_PREFIXES",
    "HMM_CONTEXT_COLS",
    "MAE_D_MODEL",
    "MAE_N_HEADS",
    "MAE_N_LAYERS",
    "MAE_DIM_FEEDFORWARD",
    "MAE_DROPOUT",
    # Base configs
    "PathConfig",
    "EnvConfig",
    "FoundationModelConfig",
    # Training configs
    "TQCTrainingConfig",
    "WFOTrainingConfig",
    "TrainingConfig",  # Alias
    "FoundationTrainingConfig",
    "linear_schedule",
]

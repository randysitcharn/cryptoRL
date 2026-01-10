# -*- coding: utf-8 -*-
"""
config - Centralized configuration module for cryptoRL.

Exports all configuration classes and global settings.

Usage:
    from src.config import SEED, DEVICE, TQCTrainingConfig
    from src.config import TuningConfig, FoundationTrainingConfig
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
)

# Training configurations
from src.config.training import (
    TQCTrainingConfig,
    TuningConfig,
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
    # Base configs
    "PathConfig",
    "EnvConfig",
    "FoundationModelConfig",
    # Training configs
    "TQCTrainingConfig",
    "TrainingConfig",  # Alias
    "TuningConfig",
    "FoundationTrainingConfig",
    "linear_schedule",
]

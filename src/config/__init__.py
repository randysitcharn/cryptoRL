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

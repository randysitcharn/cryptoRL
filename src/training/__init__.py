# -*- coding: utf-8 -*-
"""
training - Training module for cryptoRL.

Provides:
- CryptoTradingEnv: Gymnasium environment for trading
- Callbacks: TensorBoard, logging, curriculum learning
- Training functions: train_agent, train_foundation
"""

from src.training.env import CryptoTradingEnv
from src.training.train_agent import train as train_agent
from src.training.train_foundation import train as train_foundation
from src.training.callbacks import (
    TensorBoardStepCallback,
    StepLoggingCallback,
    DetailTensorboardCallback,
    CurriculumFeesCallback,
    get_next_run_dir,
)

# Re-export configs from centralized config module
from src.config import (
    TQCTrainingConfig,
    FoundationTrainingConfig,
)

__all__ = [
    # Environment
    "CryptoTradingEnv",
    # Training functions
    "train_agent",
    "train_foundation",
    # Callbacks
    "TensorBoardStepCallback",
    "StepLoggingCallback",
    "DetailTensorboardCallback",
    "CurriculumFeesCallback",
    "get_next_run_dir",
    # Configs
    "TQCTrainingConfig",
    "FoundationTrainingConfig",
]

# -*- coding: utf-8 -*-
"""
training - Training module for cryptoRL.

Provides:
- CryptoTradingEnv: Gymnasium environment for trading
- Callbacks: TensorBoard, logging, curriculum learning
- Training functions: train_agent, train_foundation
"""

# Core environment (no heavy dependencies)
from src.training.env import CryptoTradingEnv

__all__ = [
    # Environment
    "CryptoTradingEnv",
    # Training functions (lazy loaded)
    "train_agent",
    "train_foundation",
    # Callbacks (lazy loaded)
    "TensorBoardStepCallback",
    "StepLoggingCallback",
    "DetailTensorboardCallback",
    "CurriculumFeesCallback",
    "get_next_run_dir",
    # Configs
    "TQCTrainingConfig",
    "FoundationTrainingConfig",
]


def __getattr__(name):
    """Lazy import for heavy dependencies (sb3_contrib, torch.utils.tensorboard)."""
    if name == "train_agent":
        from src.training.train_agent import train
        return train
    elif name == "train_foundation":
        from src.training.train_foundation import train
        return train
    elif name in ("TensorBoardStepCallback", "StepLoggingCallback",
                  "DetailTensorboardCallback", "CurriculumFeesCallback", "get_next_run_dir"):
        from src.training import callbacks
        return getattr(callbacks, name)
    elif name in ("TQCTrainingConfig", "FoundationTrainingConfig"):
        from src import config
        return getattr(config, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

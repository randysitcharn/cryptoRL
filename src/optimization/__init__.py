# -*- coding: utf-8 -*-
"""
optimization - Hyperparameter tuning module for cryptoRL.

Provides Optuna-based hyperparameter optimization for TQC agents.
"""

from src.optimization.tune import (
    TrialPruningCallback,
    run_optimization,
)

# Re-export TuningConfig for convenience
from src.config import TuningConfig

__all__ = [
    "TuningConfig",
    "TrialPruningCallback",
    "run_optimization",
]

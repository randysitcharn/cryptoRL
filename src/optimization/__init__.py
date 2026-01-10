# -*- coding: utf-8 -*-
"""
optimization - Hyperparameter tuning module for cryptoRL.

Provides Optuna-based hyperparameter optimization for TQC agents.
"""

# Re-export TuningConfig (no heavy deps)
from src.config import TuningConfig

__all__ = [
    "TuningConfig",
    "TrialPruningCallback",  # lazy
    "run_optimization",  # lazy
]


def __getattr__(name):
    """Lazy import for heavy dependencies (sb3_contrib, optuna)."""
    if name == "TrialPruningCallback":
        from src.optimization.tune import TrialPruningCallback
        return TrialPruningCallback
    elif name == "run_optimization":
        from src.optimization.tune import run_optimization
        return run_optimization
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

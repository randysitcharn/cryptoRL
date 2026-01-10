# -*- coding: utf-8 -*-
"""
Evaluation module for backtesting and model validation.

Provides unified interfaces for model evaluation, backtesting,
and visualization of trading strategies.
"""

# Light imports only (no sb3_contrib dependency)
from src.evaluation.config import EvaluationConfig
from src.evaluation.visualize import EvaluationVisualizer

__all__ = [
    # Config
    "EvaluationConfig",
    "BacktestConfig",  # Legacy (lazy)
    # Runner (lazy loaded)
    "BacktestRunner",
    # Visualization
    "EvaluationVisualizer",
    "Backtester",  # lazy
    # Functions (lazy loaded)
    "run_backtest",
]


def __getattr__(name):
    """Lazy import for heavy dependencies (sb3_contrib)."""
    if name == "BacktestRunner":
        from src.evaluation.runner import BacktestRunner
        return BacktestRunner
    elif name == "run_backtest":
        from src.evaluation.backtest import run_backtest
        return run_backtest
    elif name == "Backtester":
        from src.evaluation.backtest import Backtester
        return Backtester
    elif name == "BacktestConfig":
        from src.evaluation.backtest import BacktestConfig
        return BacktestConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

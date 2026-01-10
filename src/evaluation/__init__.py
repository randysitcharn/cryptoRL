# -*- coding: utf-8 -*-
"""
Evaluation module for backtesting and model validation.

Provides unified interfaces for model evaluation, backtesting,
and visualization of trading strategies.
"""

from src.evaluation.config import EvaluationConfig
from src.evaluation.runner import BacktestRunner
from src.evaluation.visualize import EvaluationVisualizer
from src.evaluation.backtest import BacktestConfig, run_backtest, Backtester

__all__ = [
    # Config
    "EvaluationConfig",
    "BacktestConfig",  # Legacy alias
    # Runner
    "BacktestRunner",
    "Backtester",
    # Visualization
    "EvaluationVisualizer",
    # Functions
    "run_backtest",
]

# -*- coding: utf-8 -*-
"""
utils - Utility functions for cryptoRL.

Provides:
- Financial metrics (Sharpe, Sortino, max drawdown, etc.)
- Reproducibility utilities (seed_everything)
- Hardware auto-detection and adaptive configuration
"""

from src.utils.metrics import (
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_total_return,
    calculate_buy_hold_return,
    calculate_win_rate,
    calculate_profit_factor,
)

from src.utils.reproducibility import seed_everything

from src.utils.hardware import (
    HardwareManager,
    HardwareSpecs,
    OptimalConfig,
    get_hardware_summary,
)

__all__ = [
    # Metrics
    "calculate_sharpe_ratio",
    "calculate_sortino_ratio",
    "calculate_max_drawdown",
    "calculate_total_return",
    "calculate_buy_hold_return",
    "calculate_win_rate",
    "calculate_profit_factor",
    # Reproducibility
    "seed_everything",
    # Hardware
    "HardwareManager",
    "HardwareSpecs",
    "OptimalConfig",
    "get_hardware_summary",
]

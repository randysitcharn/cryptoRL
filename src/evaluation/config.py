# -*- coding: utf-8 -*-
"""
config.py - Evaluation configuration classes.

Provides unified configuration for backtesting and model evaluation.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class EvaluationConfig:
    """
    Unified configuration for model evaluation and backtesting.

    Supports both CSV and Parquet data formats, and provides options
    for data splitting, visualization, and output.
    """

    # --- Data ---
    data_path: str = "data/processed_data.parquet"
    data_format: str = "parquet"  # "parquet" or "csv"

    # --- Model ---
    model_path: str = "weights/checkpoints/best_model.zip"

    # --- Environment ---
    window_size: int = 64
    commission: float = 0.0006
    slippage: float = 0.0001
    initial_balance: float = 10_000.0

    # --- Data Split ---
    train_ratio: float = 0.8
    val_ratio: float = 0.15  # Only used with TimeSeriesSplitter
    eval_split: str = "test"  # "train", "val", or "test"
    use_time_series_splitter: bool = False  # Use TimeSeriesSplitter (for CSV)
    purge_window: int = 50  # Purge window for TimeSeriesSplitter

    # --- Evaluation ---
    deterministic: bool = True
    verbose: bool = True

    # --- Output ---
    output_dir: str = "results/"
    save_plots: bool = True
    plot_types: List[str] = field(default_factory=lambda: [
        "portfolio_drawdown",
        "actions_distribution",
        "strategy_comparison",
        "returns_distribution",
        "position_timeline",
    ])

    # --- Backward Compatibility Aliases ---
    @classmethod
    def from_backtest_config(cls, config) -> "EvaluationConfig":
        """Create EvaluationConfig from legacy BacktestConfig."""
        return cls(
            data_path=config.data_path,
            data_format="parquet",
            model_path=config.model_path,
            window_size=config.window_size,
            commission=config.commission,
            initial_balance=config.initial_balance,
            train_ratio=config.train_ratio,
            output_dir=config.output_dir,
            eval_split="test",
            use_time_series_splitter=False,
            plot_types=["portfolio_drawdown", "position_timeline"],
        )

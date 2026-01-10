# -*- coding: utf-8 -*-
"""
backtest.py - Out-of-Sample Validation of TQC Agent.

Provides backward-compatible interface for backtesting.
Uses the unified BacktestRunner internally.

Usage:
    python -m src.evaluation.backtest
"""

from dataclasses import dataclass
from typing import Optional

from src.evaluation.config import EvaluationConfig
from src.evaluation.runner import BacktestRunner


# ============================================================================
# Legacy Configuration (backward compatibility)
# ============================================================================

@dataclass
class BacktestConfig:
    """
    Configuration for backtesting.

    This class is kept for backward compatibility.
    New code should use EvaluationConfig directly.
    """

    # Paths
    data_path: str = "data/processed_data.parquet"
    model_path: str = "weights/checkpoints/best_model.zip"
    output_dir: str = "results/"

    # Environment
    window_size: int = 64
    commission: float = 0.0006
    train_ratio: float = 0.8

    # Backtest
    initial_balance: float = 10_000.0

    def to_evaluation_config(self) -> EvaluationConfig:
        """Convert to EvaluationConfig."""
        return EvaluationConfig(
            data_path=self.data_path,
            data_format="parquet",
            model_path=self.model_path,
            window_size=self.window_size,
            commission=self.commission,
            initial_balance=self.initial_balance,
            train_ratio=self.train_ratio,
            output_dir=self.output_dir,
            eval_split="test",
            use_time_series_splitter=False,
            plot_types=["combined"],  # Single combined plot like original
            verbose=True,
        )


# ============================================================================
# Backtest Runner (backward compatibility wrapper)
# ============================================================================

def run_backtest(config: Optional[BacktestConfig] = None) -> dict:
    """
    Run backtest on validation data.

    This function is kept for backward compatibility.
    New code should use BacktestRunner directly.

    Args:
        config: Backtest configuration (uses defaults if None).

    Returns:
        Dictionary with backtest results and KPIs.
    """
    if config is None:
        config = BacktestConfig()

    # Convert to unified config
    eval_config = config.to_evaluation_config()

    # Run using unified runner
    runner = BacktestRunner(eval_config)
    results = runner.run()

    # Return in legacy format for compatibility
    return {
        'total_return': results['metrics'].get('total_return', 0),
        'buy_hold_return': results['metrics'].get('buy_hold_return', 0),
        'sharpe_ratio': results['metrics'].get('sharpe_ratio', 0),
        'max_drawdown': results['metrics'].get('max_drawdown', 0),
        'total_trades': results['metrics'].get('total_trades', 0),
        'total_commission': results['metrics'].get('total_commission', 0),
        'final_nav': results['metrics'].get('final_nav', 0),
        'initial_nav': results['metrics'].get('initial_nav', 0),
        'steps': results['metrics'].get('steps', 0),
        'nav_history': results['history'].get('nav', []),
        'price_history': results['history'].get('prices', []),
        'action_history': results['history'].get('actions', []),
        'position_history': results['history'].get('positions', []),
    }


# ============================================================================
# Convenience Class (new API)
# ============================================================================

class Backtester:
    """
    Convenience wrapper for BacktestRunner.

    Provides a simpler interface for common backtesting scenarios.
    """

    def __init__(
        self,
        data_path: str = "data/processed_data.parquet",
        model_path: str = "weights/checkpoints/best_model.zip",
        output_dir: str = "results/",
        **kwargs
    ):
        """
        Initialize backtester.

        Args:
            data_path: Path to data file.
            model_path: Path to trained model.
            output_dir: Directory for output.
            **kwargs: Additional arguments passed to EvaluationConfig.
        """
        self.config = EvaluationConfig(
            data_path=data_path,
            model_path=model_path,
            output_dir=output_dir,
            **kwargs
        )
        self.runner = BacktestRunner(self.config)
        self.results = None

    def run(self) -> dict:
        """Run backtest and return results."""
        self.results = self.runner.run()
        return self.results

    @property
    def metrics(self) -> dict:
        """Get metrics from last run."""
        if self.results is None:
            raise ValueError("No results available. Call run() first.")
        return self.results['metrics']

    @property
    def history(self) -> dict:
        """Get history from last run."""
        if self.results is None:
            raise ValueError("No results available. Call run() first.")
        return self.results['history']


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    results = run_backtest()

# -*- coding: utf-8 -*-
"""
runner.py - Unified backtest runner for model evaluation.

Provides a single interface for running backtests with different configurations.
Supports both CSV and Parquet data formats, and multiple visualization options.
"""

from typing import Dict, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
from sb3_contrib import TQC

from src.training.env import CryptoTradingEnv
from src.evaluation.config import EvaluationConfig
from src.evaluation.visualize import EvaluationVisualizer
from src.utils.metrics import (
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_total_return,
    calculate_buy_hold_return,
    calculate_win_rate,
)


class BacktestRunner:
    """
    Unified backtest runner for model evaluation.

    Handles data loading, environment creation, model loading,
    backtest execution, metrics calculation, and visualization.
    """

    def __init__(self, config: EvaluationConfig):
        """
        Initialize backtest runner.

        Args:
            config: Evaluation configuration.
        """
        self.config = config
        self.visualizer = EvaluationVisualizer(output_dir=config.output_dir)

        # State
        self.model: Optional[TQC] = None
        self.env: Optional[CryptoTradingEnv] = None
        self.history: Dict = {}
        self.metrics: Dict[str, float] = {}

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and split data based on configuration.

        Returns:
            Tuple of (full_df, eval_df) DataFrames.
        """
        if self.config.verbose:
            print(f"[INFO] Loading data from {self.config.data_path}...")

        # Load based on format
        if self.config.data_format == "csv":
            df = pd.read_csv(self.config.data_path, index_col=0, parse_dates=True)
        else:
            df = pd.read_parquet(self.config.data_path)

        if self.config.verbose:
            print(f"[OK] Loaded {len(df)} rows")

        # Split data
        if self.config.use_time_series_splitter:
            # Use TimeSeriesSplitter for proper train/val/test splits
            from src.data_engineering.splitter import TimeSeriesSplitter

            splitter = TimeSeriesSplitter(df)
            train_df, val_df, test_df = splitter.split_data(
                train_ratio=self.config.train_ratio,
                val_ratio=self.config.val_ratio,
                purge_window=self.config.purge_window
            )

            if self.config.eval_split == "train":
                eval_df = train_df
            elif self.config.eval_split == "val":
                eval_df = val_df
            else:
                eval_df = test_df
        else:
            # Simple ratio split
            split_idx = int(len(df) * self.config.train_ratio)
            eval_df = df.iloc[split_idx:].reset_index(drop=True)

        if self.config.verbose:
            print(f"[OK] Using {self.config.eval_split.upper()} split: {len(eval_df)} rows")

        return df, eval_df

    def create_environment(self, df: pd.DataFrame = None) -> CryptoTradingEnv:
        """
        Create trading environment.

        Args:
            df: DataFrame to use (for CSV mode). If None, uses parquet path.

        Returns:
            CryptoTradingEnv instance.
        """
        if df is not None:
            # DataFrame mode (legacy CSV support)
            env = CryptoTradingEnv(
                df=df,
                window_size=self.config.window_size,
                commission=self.config.commission,
                slippage=self.config.slippage,
                initial_balance=self.config.initial_balance,
                random_start=False,
                episode_length=None,
            )
        else:
            # Parquet mode with index slicing
            full_df = pd.read_parquet(self.config.data_path)
            n_total = len(full_df)
            split_idx = int(n_total * self.config.train_ratio)

            env = CryptoTradingEnv(
                parquet_path=self.config.data_path,
                start_idx=split_idx,
                end_idx=n_total,
                window_size=self.config.window_size,
                commission=self.config.commission,
                initial_balance=self.config.initial_balance,
                random_start=False,
                episode_length=None,
            )

        self.env = env
        return env

    def load_model(self) -> TQC:
        """
        Load trained model.

        Returns:
            Loaded TQC model.
        """
        if self.config.verbose:
            print(f"[INFO] Loading model from {self.config.model_path}...")

        self.model = TQC.load(self.config.model_path)

        if self.config.verbose:
            print("[OK] Model loaded")

        return self.model

    def run_backtest(self, env: CryptoTradingEnv = None, model: TQC = None) -> Dict:
        """
        Execute backtest.

        Args:
            env: Environment to use (uses self.env if None).
            model: Model to use (uses self.model if None).

        Returns:
            History dictionary with backtest results.
        """
        if env is None:
            env = self.env
        if model is None:
            model = self.model

        if env is None or model is None:
            raise ValueError("Environment and model must be loaded before running backtest")

        if self.config.verbose:
            print("[INFO] Running backtest (deterministic=True)...")

        # Initialize history
        history = {
            'timestamps': [],
            'nav': [],
            'actions': [],
            'returns': [],
            'drawdowns': [],
            'prices': [],
            'positions': [],
        }

        obs, info = env.reset()

        # Record initial state
        history['nav'].append(info.get('nav', info.get('portfolio_value', self.config.initial_balance)))
        history['prices'].append(info.get('price', 0))
        history['positions'].append(info.get('position_pct', 0))
        history['actions'].append(0.0)
        history['returns'].append(0.0)
        history['drawdowns'].append(0.0)

        # Try to get timestamp
        try:
            if hasattr(env, 'current_step'):
                history['timestamps'].append(env.current_step)
        except:
            history['timestamps'].append(0)

        done = False
        step_count = 0
        peak_nav = history['nav'][0]

        while not done:
            # Deterministic prediction
            action, _ = model.predict(obs, deterministic=self.config.deterministic)

            # Execute step
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Get current NAV
            current_nav = info.get('nav', info.get('portfolio_value', 0))

            # Calculate drawdown
            peak_nav = max(peak_nav, current_nav)
            drawdown = (peak_nav - current_nav) / peak_nav if peak_nav > 0 else 0

            # Record history
            history['nav'].append(current_nav)
            history['prices'].append(info.get('price', 0))
            history['actions'].append(float(action[0]))
            history['returns'].append(info.get('return', info.get('log_return', 0)))
            history['drawdowns'].append(drawdown)

            # Get position
            if 'position_pct' in info:
                history['positions'].append(info['position_pct'])
            elif 'asset_holdings' in info and info.get('portfolio_value', 0) > 0:
                asset_value = info['asset_holdings'] * info.get('price', 0)
                position_pct = asset_value / info['portfolio_value']
                history['positions'].append(position_pct)
            else:
                history['positions'].append(0)

            # Timestamp
            try:
                if hasattr(env, 'current_step'):
                    history['timestamps'].append(env.current_step)
                else:
                    history['timestamps'].append(step_count + 1)
            except:
                history['timestamps'].append(step_count + 1)

            step_count += 1

            # Progress update
            if self.config.verbose and step_count % 500 == 0:
                trades = info.get('total_trades', 0)
                pos = info.get('position_pct', 0)
                print(f"      Step {step_count:5d} | NAV: ${current_nav:,.2f} | "
                      f"Position: {pos:+.2f} | Trades: {trades}")

        # Store last info for metrics
        self._last_info = info

        if self.config.verbose:
            print(f"[OK] Backtest complete: {step_count} steps")

        self.history = history
        return history

    def calculate_metrics(self, history: Dict = None) -> Dict[str, float]:
        """
        Calculate performance metrics.

        Args:
            history: History dict (uses self.history if None).

        Returns:
            Dictionary of performance metrics.
        """
        if history is None:
            history = self.history

        nav = np.array(history['nav'])
        prices = np.array(history['prices'])
        returns = np.array(history['returns'])

        # Core metrics
        total_return = calculate_total_return(nav)
        buy_hold_return = calculate_buy_hold_return(prices)
        sharpe = calculate_sharpe_ratio(returns)
        sortino = calculate_sortino_ratio(returns)
        max_dd = calculate_max_drawdown(nav)
        win_rate = calculate_win_rate(returns)

        metrics = {
            'total_return': total_return,
            'buy_hold_return': buy_hold_return,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'initial_nav': nav[0],
            'final_nav': nav[-1],
            'steps': len(nav) - 1,
        }

        # Add trading metrics if available
        if hasattr(self, '_last_info') and self._last_info:
            metrics['total_trades'] = self._last_info.get('total_trades', 0)
            metrics['total_commission'] = self._last_info.get('total_commission', 0)

        self.metrics = metrics
        return metrics

    def print_metrics(self, metrics: Dict[str, float] = None) -> None:
        """Print formatted metrics summary."""
        if metrics is None:
            metrics = self.metrics

        print("\n" + "=" * 60)
        print("PERFORMANCE METRICS")
        print("=" * 60)

        print(f"\n  Strategy Performance:")
        print(f"    Initial NAV:    ${metrics.get('initial_nav', 0):,.2f}")
        print(f"    Final NAV:      ${metrics.get('final_nav', 0):,.2f}")
        print(f"    Total Return:   {metrics.get('total_return', 0)*100:+.2f}%")

        print(f"\n  Benchmark:")
        print(f"    Buy & Hold:     {metrics.get('buy_hold_return', 0)*100:+.2f}%")
        alpha = (metrics.get('total_return', 0) - metrics.get('buy_hold_return', 0)) * 100
        print(f"    Alpha:          {alpha:+.2f}%")

        print(f"\n  Risk Metrics:")
        print(f"    Sharpe Ratio:   {metrics.get('sharpe_ratio', 0):.3f}")
        print(f"    Sortino Ratio:  {metrics.get('sortino_ratio', 0):.3f}")
        print(f"    Max Drawdown:   {metrics.get('max_drawdown', 0)*100:.2f}%")
        print(f"    Win Rate:       {metrics.get('win_rate', 0)*100:.2f}%")

        if 'total_trades' in metrics:
            print(f"\n  Trading Activity:")
            print(f"    Total Trades:   {metrics.get('total_trades', 0)}")
            print(f"    Commission:     ${metrics.get('total_commission', 0):.2f}")

        print("=" * 60)

    def generate_plots(
        self,
        history: Dict = None,
        metrics: Dict[str, float] = None
    ) -> list:
        """
        Generate visualization plots.

        Args:
            history: History dict.
            metrics: Metrics dict.

        Returns:
            List of saved plot paths.
        """
        if history is None:
            history = self.history
        if metrics is None:
            metrics = self.metrics

        if not self.config.save_plots:
            return []

        return self.visualizer.plot_all(
            history=history,
            metrics=metrics,
            plot_types=self.config.plot_types
        )

    def run(self) -> Dict:
        """
        Execute full evaluation pipeline.

        Returns:
            Dictionary with metrics and history.
        """
        print("=" * 70)
        print("BACKTEST EVALUATION")
        print("=" * 70)

        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        # Load data and create environment
        if self.config.data_format == "csv" or self.config.use_time_series_splitter:
            full_df, eval_df = self.load_data()
            self.create_environment(df=eval_df)
        else:
            self.create_environment()

        # Load model
        self.load_model()

        # Run backtest
        history = self.run_backtest()

        # Calculate metrics
        metrics = self.calculate_metrics(history)

        # Print results
        if self.config.verbose:
            self.print_metrics(metrics)

        # Generate plots
        if self.config.save_plots:
            print("\n[INFO] Generating plots...")
            self.generate_plots(history, metrics)

        print("\n" + "=" * 70)
        print("BACKTEST COMPLETE")
        print("=" * 70)

        return {
            'metrics': metrics,
            'history': history,
        }

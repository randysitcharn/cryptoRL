# -*- coding: utf-8 -*-
"""
visualize.py - Visualization functions for backtesting results.

Provides comprehensive plotting capabilities for model evaluation:
- Portfolio value and drawdown
- Actions distribution
- Strategy vs Buy & Hold comparison
- Returns distribution
- Position timeline
"""

from typing import Dict, List, Optional
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


class EvaluationVisualizer:
    """
    Visualization engine for backtest results.

    Generates publication-quality plots for model evaluation.
    """

    def __init__(self, output_dir: str = "results/", style: str = "seaborn-v0_8-darkgrid"):
        """
        Initialize visualizer.

        Args:
            output_dir: Directory to save plots.
            style: Matplotlib style to use.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.style = style

    def _apply_style(self):
        """Apply matplotlib style."""
        try:
            plt.style.use(self.style)
        except OSError:
            plt.style.use('seaborn-v0_8')

    def plot_portfolio_drawdown(
        self,
        timestamps: List,
        nav: np.ndarray,
        drawdowns: np.ndarray,
        metrics: Dict[str, float],
        filename: str = "portfolio_drawdown.png"
    ) -> str:
        """
        Plot portfolio value and drawdown.

        Args:
            timestamps: Time index.
            nav: Net Asset Value history.
            drawdowns: Drawdown history.
            metrics: Performance metrics dict.
            filename: Output filename.

        Returns:
            Path to saved plot.
        """
        self._apply_style()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        # NAV
        ax1.plot(timestamps, nav, label='Strategy NAV', color='#2E86AB', linewidth=1.5)
        ax1.axhline(y=nav[0], color='gray', linestyle='--', alpha=0.5, label='Initial')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.set_title(f'Portfolio Value | Total Return: {metrics.get("total_return", 0)*100:.2f}%')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        # Drawdown
        dd_pct = drawdowns * 100 if np.max(drawdowns) <= 1 else drawdowns
        ax2.fill_between(timestamps, dd_pct, 0, color='#E74C3C', alpha=0.5)
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_xlabel('Date')
        ax2.set_title(f'Drawdown | Max: {metrics.get("max_drawdown", 0)*100:.2f}%')
        ax2.grid(True, alpha=0.3)
        ax2.invert_yaxis()

        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150)
        plt.close()

        return str(output_path)

    def plot_actions_distribution(
        self,
        actions: np.ndarray,
        filename: str = "actions_distribution.png"
    ) -> str:
        """
        Plot histogram of actions taken.

        Args:
            actions: Action history.
            filename: Output filename.

        Returns:
            Path to saved plot.
        """
        self._apply_style()

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(actions, bins=50, color='#3498DB', edgecolor='white', alpha=0.8)
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Neutral')
        ax.axvline(x=np.mean(actions), color='green', linestyle='-',
                   linewidth=2, label=f'Mean: {np.mean(actions):.3f}')
        ax.set_xlabel('Action (Position Target)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution des Actions [-1=Cash, +1=Full Position]')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150)
        plt.close()

        return str(output_path)

    def plot_strategy_comparison(
        self,
        timestamps: List,
        nav: np.ndarray,
        prices: np.ndarray,
        initial_balance: float = 10000.0,
        filename: str = "strategy_vs_buyhold.png"
    ) -> str:
        """
        Plot strategy vs Buy & Hold comparison.

        Args:
            timestamps: Time index.
            nav: Strategy NAV history.
            prices: Price history.
            initial_balance: Initial portfolio value.
            filename: Output filename.

        Returns:
            Path to saved plot.
        """
        self._apply_style()

        # Calculate Buy & Hold NAV
        initial_price = prices[0]
        shares = initial_balance / initial_price
        bh_nav = shares * prices

        fig, ax = plt.subplots(figsize=(14, 6))

        ax.plot(timestamps, nav, label='Strategy', color='#2E86AB', linewidth=1.5)
        ax.plot(timestamps, bh_nav, label='Buy & Hold', color='#E67E22', linewidth=1.5, linestyle='--')
        ax.axhline(y=nav[0], color='gray', linestyle=':', alpha=0.5)

        # Annotate final values
        strategy_return = (nav[-1] / nav[0] - 1) * 100
        bh_return = (bh_nav[-1] / bh_nav[0] - 1) * 100

        ax.annotate(f'Strategy: {strategy_return:+.2f}%',
                    xy=(timestamps[-1], nav[-1]), fontsize=10,
                    xytext=(10, 10), textcoords='offset points')
        ax.annotate(f'B&H: {bh_return:+.2f}%',
                    xy=(timestamps[-1], bh_nav[-1]), fontsize=10,
                    xytext=(10, -20), textcoords='offset points')

        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value ($)')
        ax.set_title('Strategy vs Buy & Hold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150)
        plt.close()

        return str(output_path)

    def plot_returns_distribution(
        self,
        returns: np.ndarray,
        metrics: Dict[str, float],
        filename: str = "returns_distribution.png"
    ) -> str:
        """
        Plot histogram of returns.

        Args:
            returns: Return history.
            metrics: Performance metrics dict.
            filename: Output filename.

        Returns:
            Path to saved plot.
        """
        self._apply_style()

        fig, ax = plt.subplots(figsize=(10, 6))

        # Remove zeros for cleaner histogram
        nonzero_returns = returns[returns != 0]

        ax.hist(nonzero_returns * 100, bins=50, color='#9B59B6', edgecolor='white', alpha=0.8)
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Zero')

        if len(nonzero_returns) > 0:
            ax.axvline(x=np.mean(nonzero_returns) * 100, color='green', linestyle='-',
                       linewidth=2, label=f'Mean: {np.mean(nonzero_returns) * 100:.4f}%')

        ax.set_xlabel('Log Return (%)')
        ax.set_ylabel('Frequency')
        sharpe = metrics.get('sharpe_ratio', 0)
        sortino = metrics.get('sortino_ratio', 0)
        ax.set_title(f'Distribution des Returns | Sharpe: {sharpe:.2f} | Sortino: {sortino:.2f}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150)
        plt.close()

        return str(output_path)

    def plot_position_timeline(
        self,
        timestamps: List,
        positions: np.ndarray,
        filename: str = "position_over_time.png"
    ) -> str:
        """
        Plot position over time.

        Args:
            timestamps: Time index.
            positions: Position history (as fraction 0-1 or percentage).
            filename: Output filename.

        Returns:
            Path to saved plot.
        """
        self._apply_style()

        fig, ax = plt.subplots(figsize=(14, 4))

        # Normalize to percentage if needed
        positions_pct = positions * 100 if np.max(np.abs(positions)) <= 1 else positions

        ax.fill_between(timestamps, positions_pct, 0, color='#1ABC9C', alpha=0.6)
        ax.plot(timestamps, positions_pct, color='#16A085', linewidth=0.5)
        ax.set_xlabel('Date')
        ax.set_ylabel('Position (%)')
        ax.set_title('Position en Crypto au cours du temps (% du portfolio)')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150)
        plt.close()

        return str(output_path)

    def plot_combined(
        self,
        nav_history: np.ndarray,
        price_history: np.ndarray,
        position_history: np.ndarray,
        metrics: Dict[str, float],
        filename: str = "backtest_results.png"
    ) -> str:
        """
        Generate combined plot (backtest.py style).

        Two subplots:
        1. NAV vs Buy & Hold (normalized to 100)
        2. Position over time

        Args:
            nav_history: NAV history.
            price_history: Price history.
            position_history: Position history.
            metrics: Performance metrics.
            filename: Output filename.

        Returns:
            Path to saved plot.
        """
        self._apply_style()

        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # Normalize to 100 for comparison
        nav_normalized = nav_history / nav_history[0] * 100
        price_normalized = price_history / price_history[0] * 100

        total_return = metrics.get('total_return', 0)
        buy_hold_return = metrics.get('buy_hold_return', 0)
        sharpe_ratio = metrics.get('sharpe_ratio', 0)
        max_drawdown = metrics.get('max_drawdown', 0)
        total_trades = metrics.get('total_trades', 0)

        # Subplot 1: NAV vs Buy & Hold
        ax1 = axes[0]
        ax1.plot(nav_normalized, label=f'TQC Agent ({total_return*100:+.2f}%)',
                 color='blue', linewidth=1.5)
        ax1.plot(price_normalized, label=f'Buy & Hold ({buy_hold_return*100:+.2f}%)',
                 color='orange', linewidth=1.5, alpha=0.7)
        ax1.set_title('Out-of-Sample Performance: TQC Agent vs Buy & Hold', fontsize=14)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Value (Normalized to 100)')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=100, color='gray', linestyle='--', alpha=0.5)

        # Add performance annotation
        textstr = f'Sharpe: {sharpe_ratio:.3f}\nMax DD: {max_drawdown*100:.1f}%\nTrades: {total_trades}'
        ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Subplot 2: Position over time
        ax2 = axes[1]
        ax2.fill_between(range(len(position_history)), position_history, 0,
                         alpha=0.5, color='green', where=np.array(position_history) > 0, label='Long')
        ax2.fill_between(range(len(position_history)), position_history, 0,
                         alpha=0.5, color='red', where=np.array(position_history) < 0, label='Short')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_title('Position Over Time', fontsize=14)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Position [-1, +1]')
        ax2.set_ylim(-1.1, 1.1)
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        return str(output_path)

    def plot_all(
        self,
        history: Dict,
        metrics: Dict[str, float],
        plot_types: Optional[List[str]] = None
    ) -> List[str]:
        """
        Generate all requested plots.

        Args:
            history: Backtest history dict.
            metrics: Performance metrics.
            plot_types: List of plot types to generate.

        Returns:
            List of paths to saved plots.
        """
        if plot_types is None:
            plot_types = [
                "portfolio_drawdown",
                "actions_distribution",
                "strategy_comparison",
                "returns_distribution",
                "position_timeline",
            ]

        saved_plots = []

        timestamps = history.get('timestamps', list(range(len(history['nav']))))
        nav = np.array(history['nav'])
        prices = np.array(history['prices'])
        actions = np.array(history.get('actions', []))
        returns = np.array(history.get('returns', []))
        positions = np.array(history.get('positions', []))
        drawdowns = np.array(history.get('drawdowns', np.zeros(len(nav))))

        for plot_type in plot_types:
            try:
                if plot_type == "portfolio_drawdown":
                    path = self.plot_portfolio_drawdown(timestamps, nav, drawdowns, metrics)
                elif plot_type == "actions_distribution" and len(actions) > 0:
                    path = self.plot_actions_distribution(actions)
                elif plot_type == "strategy_comparison":
                    path = self.plot_strategy_comparison(timestamps, nav, prices)
                elif plot_type == "returns_distribution" and len(returns) > 0:
                    path = self.plot_returns_distribution(returns, metrics)
                elif plot_type == "position_timeline" and len(positions) > 0:
                    path = self.plot_position_timeline(timestamps, positions)
                elif plot_type == "combined":
                    path = self.plot_combined(nav, prices, positions, metrics)
                else:
                    continue

                saved_plots.append(path)
                print(f"[OK] Saved: {path}")

            except Exception as e:
                print(f"[WARN] Failed to generate {plot_type}: {e}")

        return saved_plots

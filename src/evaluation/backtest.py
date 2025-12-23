# -*- coding: utf-8 -*-
"""
backtest.py - Out-of-Sample Validation of TQC Agent.

Evaluates the best trained model on validation data (20% hold-out).
Uses deterministic=True for reproducible evaluation.

Usage:
    python -m src.evaluation.backtest
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sb3_contrib import TQC
from src.training.env import CryptoTradingEnv


# ============================================================================
# Configuration
# ============================================================================

class BacktestConfig:
    """Configuration for backtesting."""

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


# ============================================================================
# KPI Calculations
# ============================================================================

def calculate_sharpe_ratio(returns: np.ndarray, periods_per_year: int = 252 * 24) -> float:
    """
    Calculate annualized Sharpe Ratio.

    Args:
        returns: Array of log returns.
        periods_per_year: Number of periods per year (default: hourly data).

    Returns:
        Annualized Sharpe Ratio.
    """
    if len(returns) < 2 or np.std(returns) == 0:
        return 0.0
    return float(np.mean(returns) / np.std(returns) * np.sqrt(periods_per_year))


def calculate_max_drawdown(nav_series: np.ndarray) -> float:
    """
    Calculate maximum drawdown.

    Args:
        nav_series: Array of NAV values.

    Returns:
        Maximum drawdown as a positive percentage (e.g., 0.15 = 15%).
    """
    peak = np.maximum.accumulate(nav_series)
    drawdown = (peak - nav_series) / peak
    return float(np.max(drawdown))


def calculate_total_return(nav_series: np.ndarray) -> float:
    """
    Calculate total return.

    Args:
        nav_series: Array of NAV values.

    Returns:
        Total return as a decimal (e.g., 0.25 = 25%).
    """
    return float((nav_series[-1] - nav_series[0]) / nav_series[0])


def calculate_buy_hold_return(prices: np.ndarray) -> float:
    """
    Calculate Buy & Hold return.

    Args:
        prices: Array of prices.

    Returns:
        Buy & Hold return as a decimal.
    """
    return float((prices[-1] - prices[0]) / prices[0])


# ============================================================================
# Backtest Runner
# ============================================================================

def run_backtest(config: BacktestConfig = None) -> dict:
    """
    Run backtest on validation data.

    Args:
        config: Backtest configuration.

    Returns:
        Dictionary with backtest results and KPIs.
    """
    if config is None:
        config = BacktestConfig()

    print("=" * 70)
    print("OUT-OF-SAMPLE BACKTEST")
    print("=" * 70)

    # Create output directory
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    # ==================== Load Data ====================
    print("\n[1/4] Loading data and creating validation environment...")

    df = pd.read_parquet(config.data_path)
    n_total = len(df)
    split_idx = int(n_total * config.train_ratio)

    print(f"      Total samples: {n_total:,}")
    print(f"      Train samples: {split_idx:,} (0 to {split_idx-1})")
    print(f"      Val samples: {n_total - split_idx:,} ({split_idx} to {n_total-1})")

    # Create validation environment
    val_env = CryptoTradingEnv(
        parquet_path=config.data_path,
        start_idx=split_idx,
        end_idx=n_total,
        window_size=config.window_size,
        commission=config.commission,
        initial_balance=config.initial_balance,
        random_start=False,  # Sequential for validation
        episode_length=None,  # Full validation set
    )

    # ==================== Load Model ====================
    print(f"\n[2/4] Loading model from {config.model_path}...")
    model = TQC.load(config.model_path)
    print("      Model loaded successfully")

    # ==================== Run Backtest ====================
    print("\n[3/4] Running backtest (deterministic=True)...")

    # Initialize tracking
    nav_history = []
    price_history = []
    action_history = []
    returns_history = []
    position_history = []

    obs, info = val_env.reset()
    nav_history.append(info['nav'])
    price_history.append(info['price'])
    position_history.append(info['position_pct'])

    done = False
    step_count = 0

    while not done:
        # Deterministic prediction (no exploration noise)
        action, _ = model.predict(obs, deterministic=True)

        # Execute step
        obs, reward, terminated, truncated, info = val_env.step(action)
        done = terminated or truncated

        # Record history
        nav_history.append(info['nav'])
        price_history.append(info['price'])
        action_history.append(float(action[0]))
        returns_history.append(info['return'])
        position_history.append(info['position_pct'])

        step_count += 1

        # Progress update every 500 steps
        if step_count % 500 == 0:
            print(f"      Step {step_count:5d} | NAV: ${info['nav']:,.2f} | "
                  f"Position: {info['position_pct']:+.2f} | Trades: {info['total_trades']}")

    # Convert to numpy arrays
    nav_history = np.array(nav_history)
    price_history = np.array(price_history)
    action_history = np.array(action_history)
    returns_history = np.array(returns_history)
    position_history = np.array(position_history)

    print(f"\n      Backtest completed: {step_count} steps")

    # ==================== Calculate KPIs ====================
    print("\n[4/4] Calculating KPIs...")

    total_return = calculate_total_return(nav_history)
    buy_hold_return = calculate_buy_hold_return(price_history)
    sharpe_ratio = calculate_sharpe_ratio(returns_history)
    max_drawdown = calculate_max_drawdown(nav_history)
    total_trades = info['total_trades']
    total_commission = info['total_commission']

    # Results dictionary
    results = {
        'total_return': total_return,
        'buy_hold_return': buy_hold_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'total_trades': total_trades,
        'total_commission': total_commission,
        'final_nav': nav_history[-1],
        'initial_nav': nav_history[0],
        'steps': step_count,
        'nav_history': nav_history,
        'price_history': price_history,
        'action_history': action_history,
        'position_history': position_history,
    }

    # ==================== Print Results ====================
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)

    print(f"\n  Strategy Performance:")
    print(f"    Initial NAV:    ${config.initial_balance:,.2f}")
    print(f"    Final NAV:      ${nav_history[-1]:,.2f}")
    print(f"    Total Return:   {total_return * 100:+.2f}%")

    print(f"\n  Buy & Hold Benchmark:")
    print(f"    B&H Return:     {buy_hold_return * 100:+.2f}%")
    print(f"    Alpha:          {(total_return - buy_hold_return) * 100:+.2f}%")

    print(f"\n  Risk Metrics:")
    print(f"    Sharpe Ratio:   {sharpe_ratio:.3f}")
    print(f"    Max Drawdown:   {max_drawdown * 100:.2f}%")

    print(f"\n  Trading Activity:")
    print(f"    Total Trades:   {total_trades}")
    print(f"    Commission:     ${total_commission:.2f}")
    print(f"    Avg Position:   {np.mean(position_history):+.2f}")

    # ==================== Generate Visualization ====================
    print("\n" + "=" * 70)
    print("Generating visualization...")

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Normalize to 100 for comparison
    nav_normalized = nav_history / nav_history[0] * 100
    price_normalized = price_history / price_history[0] * 100

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

    # Subplot 2: Actions / Positions over time
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

    # Save figure
    output_path = Path(config.output_dir) / "backtest_results.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"      Saved to: {output_path}")

    plt.close()

    print("\n" + "=" * 70)
    print("BACKTEST COMPLETE")
    print("=" * 70)

    return results


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    results = run_backtest()

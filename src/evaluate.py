# -*- coding: utf-8 -*-
"""
evaluate.py - Script d'evaluation et de backtesting pour l'agent TQC.

Charge un modele entraine et genere des graphiques de performance:
- Portfolio Value (NAV) au cours du temps
- Drawdown en % depuis le peak
- Distribution des actions [-1, 1]
- Comparaison Strategy vs Buy & Hold
- Distribution des log-returns

Metriques calculees:
- Total Return (%)
- Sharpe Ratio (annualized)
- Sortino Ratio (annualized)
- Max Drawdown (%)
- Win Rate (%)
"""

import os
import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sb3_contrib import TQC

from src.env.trading_env import CryptoTradingEnv
from src.data_engineering.splitter import TimeSeriesSplitter


def load_model(model_path: str) -> TQC:
    """Charge un modele TQC sauvegarde."""
    print(f"[INFO] Loading model from {model_path}...")
    model = TQC.load(model_path)
    print("[OK] Model loaded")
    return model


def run_backtest(
    model: TQC,
    df: pd.DataFrame,
    window_size: int = 64
) -> Dict[str, List]:
    """
    Execute un backtest sur les donnees fournies.

    Args:
        model: Agent TQC entraine.
        df: DataFrame de test avec features.
        window_size: Taille de fenetre pour l'env.

    Returns:
        Dict avec historique: nav, actions, returns, drawdowns, prices, timestamps.
    """
    print(f"[INFO] Running backtest on {len(df)} rows...")

    env = CryptoTradingEnv(df, window_size=window_size)

    # Historique
    history = {
        'timestamps': [],
        'nav': [],
        'actions': [],
        'returns': [],
        'drawdowns': [],
        'prices': [],
        'positions': [],  # Position en % du portfolio
    }

    obs, info = env.reset()
    done = False

    # Initial state
    history['timestamps'].append(df.index[env.current_step])
    history['nav'].append(info['portfolio_value'])
    history['prices'].append(info['price'])
    history['actions'].append(0.0)
    history['returns'].append(0.0)
    history['drawdowns'].append(0.0)
    history['positions'].append(0.0)

    while not done:
        # Predict action
        action, _ = model.predict(obs, deterministic=True)

        # Execute step
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Record history
        history['timestamps'].append(df.index[env.current_step])
        history['nav'].append(info['portfolio_value'])
        history['prices'].append(info['price'])
        history['actions'].append(info['action'])
        history['returns'].append(info['log_return'])
        history['drawdowns'].append(info['max_drawdown'])

        # Calculate position % (asset value / total portfolio)
        asset_value = info['asset_holdings'] * info['price']
        position_pct = asset_value / info['portfolio_value'] if info['portfolio_value'] > 0 else 0
        history['positions'].append(position_pct)

    print(f"[OK] Backtest complete: {len(history['nav'])} steps")
    return history


def calculate_metrics(history: Dict[str, List]) -> Dict[str, float]:
    """
    Calcule les metriques de performance.

    Returns:
        Dict avec: total_return, sharpe, sortino, max_drawdown, win_rate.
    """
    nav = np.array(history['nav'])
    returns = np.array(history['returns'])

    # Total Return
    total_return = (nav[-1] / nav[0] - 1) * 100

    # Annualized Sharpe Ratio (assume daily data, 252 trading days)
    # Sharpe = mean(returns) / std(returns) * sqrt(252)
    if len(returns) > 1 and np.std(returns) > 0:
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Sortino Ratio (only downside deviation)
    negative_returns = returns[returns < 0]
    if len(negative_returns) > 1:
        downside_std = np.std(negative_returns)
        sortino = np.mean(returns) / downside_std * np.sqrt(252) if downside_std > 0 else 0.0
    else:
        sortino = 0.0

    # Max Drawdown
    max_drawdown = max(history['drawdowns']) * 100

    # Win Rate (% of positive returns)
    positive_returns = returns[returns > 0]
    win_rate = len(positive_returns) / len(returns) * 100 if len(returns) > 0 else 0.0

    metrics = {
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
    }

    return metrics


def calculate_buy_and_hold(history: Dict[str, List], initial_balance: float = 10000.0) -> np.ndarray:
    """Calcule la NAV Buy & Hold pour comparaison."""
    prices = np.array(history['prices'])
    initial_price = prices[0]

    # B&H: acheter au debut et garder
    shares = initial_balance / initial_price
    bh_nav = shares * prices

    return bh_nav


def plot_results(
    history: Dict[str, List],
    metrics: Dict[str, float],
    save_dir: str = "results/"
) -> None:
    """
    Genere et sauvegarde tous les graphiques.

    Args:
        history: Historique du backtest.
        metrics: Metriques calculees.
        save_dir: Dossier de sauvegarde.
    """
    os.makedirs(save_dir, exist_ok=True)

    timestamps = history['timestamps']
    nav = np.array(history['nav'])
    actions = np.array(history['actions'])
    returns = np.array(history['returns'])
    drawdowns = np.array(history['drawdowns'])
    bh_nav = calculate_buy_and_hold(history)

    # Style
    plt.style.use('seaborn-v0_8-darkgrid')

    # ============================================================
    # Figure 1: Portfolio Value + Drawdown (2 subplots)
    # ============================================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # NAV
    ax1.plot(timestamps, nav, label='Strategy NAV', color='#2E86AB', linewidth=1.5)
    ax1.axhline(y=nav[0], color='gray', linestyle='--', alpha=0.5, label='Initial')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.set_title(f'Portfolio Value | Total Return: {metrics["total_return"]:.2f}%')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Drawdown
    ax2.fill_between(timestamps, drawdowns * 100, 0, color='#E74C3C', alpha=0.5)
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_xlabel('Date')
    ax2.set_title(f'Drawdown | Max: {metrics["max_drawdown"]:.2f}%')
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()  # Drawdown negatif en bas

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'portfolio_drawdown.png'), dpi=150)
    plt.close()
    print(f"[OK] Saved: {save_dir}/portfolio_drawdown.png")

    # ============================================================
    # Figure 2: Actions Distribution (histogram)
    # ============================================================
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
    plt.savefig(os.path.join(save_dir, 'actions_distribution.png'), dpi=150)
    plt.close()
    print(f"[OK] Saved: {save_dir}/actions_distribution.png")

    # ============================================================
    # Figure 3: Strategy vs Buy & Hold
    # ============================================================
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
    plt.savefig(os.path.join(save_dir, 'strategy_vs_buyhold.png'), dpi=150)
    plt.close()
    print(f"[OK] Saved: {save_dir}/strategy_vs_buyhold.png")

    # ============================================================
    # Figure 4: Returns Distribution
    # ============================================================
    fig, ax = plt.subplots(figsize=(10, 6))

    # Remove zeros for cleaner histogram
    nonzero_returns = returns[returns != 0]

    ax.hist(nonzero_returns * 100, bins=50, color='#9B59B6', edgecolor='white', alpha=0.8)
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Zero')
    ax.axvline(x=np.mean(nonzero_returns) * 100, color='green', linestyle='-',
               linewidth=2, label=f'Mean: {np.mean(nonzero_returns) * 100:.4f}%')

    ax.set_xlabel('Log Return (%)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Distribution des Returns | Sharpe: {metrics["sharpe_ratio"]:.2f} | Sortino: {metrics["sortino_ratio"]:.2f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'returns_distribution.png'), dpi=150)
    plt.close()
    print(f"[OK] Saved: {save_dir}/returns_distribution.png")

    # ============================================================
    # Figure 5: Position over time
    # ============================================================
    fig, ax = plt.subplots(figsize=(14, 4))

    positions = np.array(history['positions']) * 100
    ax.fill_between(timestamps, positions, 0, color='#1ABC9C', alpha=0.6)
    ax.plot(timestamps, positions, color='#16A085', linewidth=0.5)
    ax.set_xlabel('Date')
    ax.set_ylabel('Position (%)')
    ax.set_title('Position en Crypto au cours du temps (% du portfolio)')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'position_over_time.png'), dpi=150)
    plt.close()
    print(f"[OK] Saved: {save_dir}/position_over_time.png")


def print_metrics(metrics: Dict[str, float]) -> None:
    """Affiche les metriques de maniere formatee."""
    print("\n" + "=" * 50)
    print("PERFORMANCE METRICS")
    print("=" * 50)
    print(f"  Total Return:    {metrics['total_return']:>10.2f} %")
    print(f"  Sharpe Ratio:    {metrics['sharpe_ratio']:>10.2f}")
    print(f"  Sortino Ratio:   {metrics['sortino_ratio']:>10.2f}")
    print(f"  Max Drawdown:    {metrics['max_drawdown']:>10.2f} %")
    print(f"  Win Rate:        {metrics['win_rate']:>10.2f} %")
    print("=" * 50)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate a trained TQC model')
    parser.add_argument('--model', type=str, default='models/best_model.zip',
                        help='Path to the trained model')
    parser.add_argument('--data', type=str, default='data/processed/BTC-USD_processed.csv',
                        help='Path to processed data')
    parser.add_argument('--output', type=str, default='results/',
                        help='Output directory for plots')
    parser.add_argument('--window', type=int, default=64,
                        help='Window size for Transformer')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Which data split to evaluate on')
    args = parser.parse_args()

    print("=" * 60)
    print("CryptoRL - Model Evaluation & Backtesting")
    print("=" * 60)

    # Load data
    print(f"\n[INFO] Loading data from {args.data}...")
    df = pd.read_csv(args.data, index_col=0, parse_dates=True)
    print(f"[OK] Loaded {len(df)} rows")

    # Split data
    print("\n[INFO] Splitting data...")
    splitter = TimeSeriesSplitter(df)
    train_df, val_df, test_df = splitter.split_data(
        train_ratio=0.7,
        val_ratio=0.15,
        purge_window=50
    )

    # Select split
    if args.split == 'train':
        eval_df = train_df
        split_name = "TRAIN"
    elif args.split == 'val':
        eval_df = val_df
        split_name = "VALIDATION"
    else:
        eval_df = test_df
        split_name = "TEST"

    print(f"[OK] Using {split_name} split: {len(eval_df)} rows")
    print(f"     From: {eval_df.index[0]} to {eval_df.index[-1]}")

    # Load model
    model = load_model(args.model)

    # Run backtest
    history = run_backtest(model, eval_df, window_size=args.window)

    # Calculate metrics
    metrics = calculate_metrics(history)
    print_metrics(metrics)

    # Generate plots
    output_dir = os.path.join(args.output, args.split)
    print(f"\n[INFO] Generating plots in {output_dir}/...")
    plot_results(history, metrics, save_dir=output_dir)

    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)
    print(f"\nPlots saved to: {output_dir}/")
    print("  - portfolio_drawdown.png")
    print("  - actions_distribution.png")
    print("  - strategy_vs_buyhold.png")
    print("  - returns_distribution.png")
    print("  - position_over_time.png")


if __name__ == "__main__":
    main()

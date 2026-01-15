#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_segment.py - Post-Mortem Analysis for WFO Segments.

Runs a comprehensive backtest on the held-out test period and generates
a clean metric report.

Usage:
    python scripts/analyze_segment.py --segment 0
    python scripts/analyze_segment.py --segment 0 --verbose
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from sb3_contrib import TQC

from src.training.env import CryptoTradingEnv
from src.models.rl_adapter import FoundationFeatureExtractor
from src.utils.metrics import (
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_total_return,
    calculate_win_rate,
)


def analyze_segment(segment_id: int, verbose: bool = True) -> dict:
    """
    Run post-mortem analysis on a WFO segment.

    Args:
        segment_id: Segment number (0, 1, 2, ...)
        verbose: Print detailed output.

    Returns:
        Dictionary with all metrics.
    """
    # Paths
    base_dir = ROOT_DIR
    data_dir = base_dir / "data" / "wfo" / f"segment_{segment_id}"
    weights_dir = base_dir / "weights" / "wfo" / f"segment_{segment_id}"

    test_path = data_dir / "test.parquet"
    model_path = weights_dir / "tqc.zip"
    encoder_path = weights_dir / "encoder.pth"

    # Validate paths
    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found: {test_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not encoder_path.exists():
        raise FileNotFoundError(f"Encoder not found: {encoder_path}")

    if verbose:
        print("=" * 60)
        print(f"POST-MORTEM ANALYSIS - Segment {segment_id}")
        print("=" * 60)
        print(f"\n[1/4] Loading test data: {test_path}")

    # Load test data
    test_df = pd.read_parquet(test_path)
    if verbose:
        print(f"      Shape: {test_df.shape}")
        print(f"      Period: {test_df.index[0]} to {test_df.index[-1]}")

    # Create environment
    if verbose:
        print(f"\n[2/4] Creating evaluation environment...")

    env = CryptoTradingEnv(
        df=test_df,
        window_size=64,
        commission=0.0006,  # Lower commission for eval (realistic)
        episode_length=len(test_df) - 64,  # Full test period
        random_start=False,  # Deterministic start for eval
    )

    # Load model
    if verbose:
        print(f"\n[3/4] Loading model: {model_path}")

    custom_objects = {
        "policy_kwargs": {
            "features_extractor_class": FoundationFeatureExtractor,
            "features_extractor_kwargs": {
                "encoder_path": str(encoder_path),
                "d_model": 256,
                "freeze_encoder": True,
            },
        }
    }

    model = TQC.load(str(model_path), env=env, custom_objects=custom_objects)

    # Run backtest
    if verbose:
        print(f"\n[4/4] Running backtest on test period...")

    obs, _ = env.reset()
    done = False

    nav_history = [env.nav]
    action_history = []
    position_history = [0.0]
    returns = []

    step = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        nav_history.append(env.nav)
        action_history.append(float(action[0]))
        position_history.append(env.position)

        # Calculate step return
        if len(nav_history) > 1:
            ret = (nav_history[-1] - nav_history[-2]) / nav_history[-2]
            returns.append(ret)

        step += 1
        if verbose and step % 500 == 0:
            print(f"      Step {step}: NAV={env.nav:.2f}, Pos={env.position:.2f}")

    # Calculate metrics
    returns = np.array(returns)
    nav_history = np.array(nav_history)

    total_return = calculate_total_return(nav_history)
    sharpe_ratio = calculate_sharpe_ratio(returns, annualize=True)
    max_drawdown = calculate_max_drawdown(nav_history)
    win_rate = calculate_win_rate(returns)

    # Count trades (position changes)
    positions = np.array(position_history)
    position_changes = np.abs(np.diff(positions))
    num_trades = np.sum(position_changes > 0.05)  # Threshold for "trade"

    # Calculate additional metrics
    final_nav = nav_history[-1]
    initial_nav = nav_history[0]

    # Sortino Ratio (downside only)
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 0:
        downside_std = np.std(downside_returns) * np.sqrt(8760)  # Annualized
        sortino_ratio = (np.mean(returns) * 8760) / downside_std if downside_std > 0 else 0
    else:
        sortino_ratio = float('inf')

    # Compile results
    metrics = {
        'segment_id': segment_id,
        'total_return_pct': total_return * 100,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown_pct': max_drawdown * 100,
        'win_rate_pct': win_rate * 100,
        'num_trades': int(num_trades),
        'final_nav': final_nav,
        'initial_nav': initial_nav,
        'test_steps': step,
        'avg_position': np.mean(np.abs(positions)),
    }

    # Print report
    if verbose:
        print("\n" + "=" * 60)
        print("METRIC REPORT")
        print("=" * 60)
        print(f"""
┌─────────────────────────────────────────────────────────┐
│  SEGMENT {segment_id} - TEST PERIOD BACKTEST                     │
├─────────────────────────────────────────────────────────┤
│  Total Return:      {metrics['total_return_pct']:>8.2f}%                        │
│  Sharpe Ratio:      {metrics['sharpe_ratio']:>8.2f}                          │
│  Sortino Ratio:     {metrics['sortino_ratio']:>8.2f}                          │
│  Max Drawdown:      {metrics['max_drawdown_pct']:>8.2f}%                        │
│  Win Rate:          {metrics['win_rate_pct']:>8.2f}%                        │
│  Number of Trades:  {metrics['num_trades']:>8d}                          │
├─────────────────────────────────────────────────────────┤
│  Initial NAV:       {metrics['initial_nav']:>10.2f}                      │
│  Final NAV:         {metrics['final_nav']:>10.2f}                      │
│  Test Steps:        {metrics['test_steps']:>10d}                      │
│  Avg |Position|:    {metrics['avg_position']:>10.4f}                      │
└─────────────────────────────────────────────────────────┘
""")

        # Verdict
        print("VERDICT:", end=" ")
        if metrics['total_return_pct'] > 0 and metrics['sharpe_ratio'] > 0.5:
            print("✅ PROFITABLE (Return > 0, Sharpe > 0.5)")
        elif metrics['total_return_pct'] > 0:
            print("🟡 MARGINAL (Return > 0, but low Sharpe)")
        else:
            print("❌ UNPROFITABLE (Return < 0)")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Post-Mortem Analysis for WFO Segments")
    parser.add_argument("--segment", type=int, required=True, help="Segment ID to analyze")
    parser.add_argument("--verbose", action="store_true", default=True, help="Verbose output")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")

    args = parser.parse_args()

    verbose = not args.quiet

    try:
        metrics = analyze_segment(args.segment, verbose=verbose)

        if args.quiet:
            # JSON output for scripting
            import json
            print(json.dumps(metrics, indent=2))

    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        print("[INFO] Make sure the WFO training for this segment is complete.")
        sys.exit(1)


if __name__ == "__main__":
    main()

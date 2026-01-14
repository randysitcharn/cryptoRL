#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_regime_performance.py - Analyze WFO performance by HMM regime.

Calculates Win Rate, Sharpe, PnL/Hour, Exposure per regime (Crash, Downtrend, Range, Uptrend).
Can regenerate history from saved TQC models for existing segments.

Usage:
    # Regenerate history for all segments with saved TQC models
    python scripts/analyze_regime_performance.py --regenerate

    # Analyze all segments
    python scripts/analyze_regime_performance.py

    # Analyze single segment
    python scripts/analyze_regime_performance.py --segment 0
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

REGIME_NAMES = {0: 'Crash', 1: 'Downtrend', 2: 'Range', 3: 'Uptrend'}
REGIME_COLORS = {
    'Crash': '#e74c3c',
    'Downtrend': '#f39c12',
    'Range': '#3498db',
    'Uptrend': '#2ecc71'
}


def regenerate_history(segment_id: int, data_dir: str, weights_dir: str, output_dir: str) -> pd.DataFrame:
    """
    Regenerate history by replaying backtest with saved TQC model.
    Uses existing weights to recreate step-by-step history.

    Args:
        segment_id: Segment ID to regenerate
        data_dir: Directory with segment data (data/wfo/)
        weights_dir: Directory with saved weights (weights/wfo/)
        output_dir: Output directory for history files (results/history/)

    Returns:
        DataFrame with step-by-step history
    """
    from sb3_contrib import TQC
    from src.training.env import CryptoTradingEnv
    from src.training.wrappers import RiskManagementWrapper

    # Paths
    test_path = f"{data_dir}/segment_{segment_id}/test.parquet"
    train_path = f"{data_dir}/segment_{segment_id}/train.parquet"
    tqc_path = f"{weights_dir}/segment_{segment_id}/tqc.zip"

    # Check if files exist
    if not os.path.exists(tqc_path):
        print(f"  [SKIP] Segment {segment_id}: No TQC model found at {tqc_path}")
        return None

    if not os.path.exists(test_path):
        print(f"  [SKIP] Segment {segment_id}: No test data found at {test_path}")
        return None

    print(f"[Segment {segment_id}] Regenerating history from {tqc_path}...")

    # Calculate baseline_vol from train data
    if os.path.exists(train_path):
        train_df = pd.read_parquet(train_path)
        baseline_vol = train_df['BTC_Close'].pct_change().std()
    else:
        baseline_vol = 0.01  # Fallback

    # Create environment
    env = CryptoTradingEnv(
        parquet_path=test_path,
        window_size=64,
        commission=0.0004,
        episode_length=None,
        random_start=False,
        target_volatility=0.15,
        vol_window=168,
        max_leverage=1.0,
        price_column='BTC_Close',
    )

    # Wrap with Risk Management
    env = RiskManagementWrapper(
        env,
        vol_window=24,
        vol_threshold=3.0,
        max_drawdown=0.10,
        cooldown_steps=12,
        augment_obs=False,
        baseline_vol=baseline_vol,
    )

    # Load model
    model = TQC.load(tqc_path)

    # Run backtest collecting full history
    obs, _ = env.reset()
    done = False

    rewards = []
    navs = []
    positions = []
    actions_history = []
    prices = []
    circuit_breakers = []

    step_count = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        rewards.append(reward)
        navs.append(info.get('nav', 10000))
        positions.append(info.get('position_pct', 0))
        actions_history.append(float(action[0]))
        prices.append(info.get('price', 0))
        circuit_breakers.append(info.get('circuit_breaker', False))

        step_count += 1
        if step_count % 500 == 0:
            print(f"    Step {step_count}...")

    # Save history
    history_df = pd.DataFrame({
        'step': range(len(rewards)),
        'reward': rewards,
        'nav': navs,
        'position': positions,
        'action': actions_history,
        'price': prices,
        'circuit_breaker': circuit_breakers,
    })

    os.makedirs(output_dir, exist_ok=True)
    history_path = f"{output_dir}/segment_{segment_id}_history.parquet"
    history_df.to_parquet(history_path)
    print(f"  [OK] History saved: {history_path} ({len(history_df)} steps)")

    return history_df


def regenerate_all_histories(data_dir: str, weights_dir: str, output_dir: str, segments: list = None):
    """Regenerate history for all segments with saved TQC models."""
    print("=" * 70)
    print("  REGENERATING HISTORIES FROM SAVED TQC MODELS")
    print("=" * 70)

    # Find all segment directories with TQC models
    weights_path = Path(weights_dir)
    if not weights_path.exists():
        print(f"[ERROR] Weights directory not found: {weights_dir}")
        return

    segment_dirs = sorted([d for d in weights_path.iterdir()
                          if d.is_dir() and d.name.startswith('segment_')])

    if not segment_dirs:
        print(f"[ERROR] No segment directories found in {weights_dir}")
        return

    # Filter to specific segments if provided
    if segments:
        segment_dirs = [d for d in segment_dirs
                       if int(d.name.split('_')[1]) in segments]

    print(f"Found {len(segment_dirs)} segments with saved models.\n")

    regenerated = 0
    failed = []
    for seg_dir in segment_dirs:
        segment_id = int(seg_dir.name.split('_')[1])
        try:
            result = regenerate_history(segment_id, data_dir, weights_dir, output_dir)
            if result is not None:
                regenerated += 1
        except Exception as e:
            print(f"  [ERROR] Segment {segment_id} failed: {e}")
            failed.append(segment_id)

    print(f"\n[DONE] Regenerated {regenerated}/{len(segment_dirs)} segment histories.")
    if failed:
        print(f"[WARNING] Failed segments: {failed}")


def load_segment_data(segment_id: int, data_dir: str, history_dir: str):
    """Load test data (HMM) and history (agent actions)."""
    test_path = f"{data_dir}/segment_{segment_id}/test.parquet"
    history_path = f"{history_dir}/segment_{segment_id}_history.parquet"

    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test data not found: {test_path}")
    if not os.path.exists(history_path):
        raise FileNotFoundError(f"History not found: {history_path}")

    test_df = pd.read_parquet(test_path)
    history_df = pd.read_parquet(history_path)

    return test_df, history_df


def merge_regime_data(test_df: pd.DataFrame, history_df: pd.DataFrame) -> pd.DataFrame:
    """Merge HMM regimes with agent history."""
    # Get dominant regime (argmax of Prob_0-3)
    prob_cols = ['Prob_0', 'Prob_1', 'Prob_2', 'Prob_3']

    # Check if prob columns exist
    if not all(col in test_df.columns for col in prob_cols):
        raise ValueError(f"Test data missing regime probability columns: {prob_cols}")

    test_df = test_df.copy()

    # Fill NaN values (warmup period) with forward fill, then use default regime 2 (Range)
    for col in prob_cols:
        test_df[col] = test_df[col].ffill().fillna(0.25)  # Equal prob if no data

    # Get dominant regime
    test_df['regime'] = test_df[prob_cols].idxmax(axis=1).str[-1].astype(int)
    test_df['regime_name'] = test_df['regime'].map(REGIME_NAMES)
    test_df['regime_confidence'] = test_df[prob_cols].max(axis=1)

    # Align lengths (history may be shorter due to window_size)
    min_len = min(len(test_df), len(history_df))

    # Take last min_len from test_df (to skip warmup/context rows)
    # and first min_len from history_df
    merged = pd.DataFrame({
        'step': range(min_len),
        'regime': test_df['regime'].values[-min_len:],
        'regime_name': test_df['regime_name'].values[-min_len:],
        'regime_confidence': test_df['regime_confidence'].values[-min_len:],
        'reward': history_df['reward'].values[:min_len],
        'nav': history_df['nav'].values[:min_len],
        'position': history_df['position'].values[:min_len],
        'action': history_df['action'].values[:min_len],
        'price': history_df['price'].values[:min_len],
    })

    # Calculate returns
    merged['return'] = merged['nav'].pct_change().fillna(0)
    merged['trade_occurred'] = merged['position'].diff().abs() > 0.01

    return merged


def calculate_metrics_per_regime(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate metrics grouped by regime."""
    results = []

    for regime in sorted(df['regime'].unique()):
        mask = df['regime'] == regime
        regime_df = df[mask]

        if len(regime_df) < 2:
            continue

        returns = regime_df['return'].values
        positions = regime_df['position'].values

        # Win Rate: % of positive returns
        win_rate = (returns > 0).mean() * 100

        # Sharpe (annualized hourly)
        if returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(8760)
        else:
            sharpe = 0.0

        # Avg PnL per hour (%)
        avg_pnl_hour = returns.mean() * 100

        # Exposure: % time with position != 0
        exposure = (np.abs(positions) > 0.01).mean() * 100

        # Number of trades in this regime
        trades = regime_df['trade_occurred'].sum()

        # Hours in regime
        hours = len(regime_df)

        # Total return in this regime (non-contiguous, just sum of returns)
        total_return = returns.sum() * 100

        results.append({
            'regime': regime,
            'regime_name': REGIME_NAMES[regime],
            'hours': hours,
            'pct_time': hours / len(df) * 100,
            'win_rate': win_rate,
            'sharpe': sharpe,
            'avg_pnl_hour': avg_pnl_hour,
            'exposure': exposure,
            'trades': int(trades),
            'total_return': total_return,
        })

    return pd.DataFrame(results)


def print_report(metrics_df: pd.DataFrame, segment_id: int = None):
    """Print formatted report."""
    title = "PERFORMANCE PAR REGIME HMM"
    if segment_id is not None:
        title += f" - Segment {segment_id}"

    print("\n" + "=" * 75)
    print(f"  {title}")
    print("=" * 75)
    print(f"{'Regime':<12} {'Hours':>6} {'%Time':>7} {'WinRate':>8} {'Sharpe':>8} {'PnL/h':>9} {'Exposure':>9} {'Trades':>7}")
    print("-" * 75)

    for _, row in metrics_df.iterrows():
        print(f"{row['regime_name']:<12} {row['hours']:>6.0f} {row['pct_time']:>6.1f}% "
              f"{row['win_rate']:>7.1f}% {row['sharpe']:>+8.2f} {row['avg_pnl_hour']:>+8.4f}% "
              f"{row['exposure']:>8.1f}% {row['trades']:>7.0f}")

    print("=" * 75)


def plot_regime_performance(metrics_df: pd.DataFrame, output_path: str):
    """Generate bar chart of Sharpe by regime."""
    # Sort by regime order
    metrics_df = metrics_df.sort_values('regime')

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    colors = [REGIME_COLORS[name] for name in metrics_df['regime_name']]

    # Sharpe by regime
    ax1 = axes[0]
    bars = ax1.bar(metrics_df['regime_name'], metrics_df['sharpe'],
                   color=colors, edgecolor='black', linewidth=1.5)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_ylabel('Sharpe Ratio (Annualized)', fontsize=11)
    ax1.set_title('Sharpe par Regime', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='x', rotation=15)
    for bar, val in zip(bars, metrics_df['sharpe']):
        ypos = bar.get_height() + 0.05 if val >= 0 else bar.get_height() - 0.15
        ax1.text(bar.get_x() + bar.get_width() / 2, ypos,
                 f'{val:+.2f}', ha='center', va='bottom' if val >= 0 else 'top',
                 fontsize=10, fontweight='bold')

    # Win Rate by regime
    ax2 = axes[1]
    bars = ax2.bar(metrics_df['regime_name'], metrics_df['win_rate'],
                   color=colors, edgecolor='black', linewidth=1.5)
    ax2.axhline(y=50, color='red', linestyle='--', linewidth=1.5, label='50% (random)')
    ax2.set_ylabel('Win Rate (%)', fontsize=11)
    ax2.set_title('Win Rate par Regime', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='x', rotation=15)
    ax2.legend(loc='lower right')
    ax2.set_ylim(0, 100)

    # Exposure by regime
    ax3 = axes[2]
    bars = ax3.bar(metrics_df['regime_name'], metrics_df['exposure'],
                   color=colors, edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('Exposure (%)', fontsize=11)
    ax3.set_title('Exposition par Regime', fontsize=12, fontweight='bold')
    ax3.tick_params(axis='x', rotation=15)
    ax3.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n[OK] Plot saved: {output_path}")


def analyze_single_segment(segment_id: int, data_dir: str, history_dir: str, output_dir: str):
    """Analyze a single segment."""
    try:
        test_df, history_df = load_segment_data(segment_id, data_dir, history_dir)
        merged = merge_regime_data(test_df, history_df)
        metrics = calculate_metrics_per_regime(merged)
        print_report(metrics, segment_id)
        plot_path = f"{output_dir}/regime_performance_seg{segment_id}.png"
        plot_regime_performance(metrics, plot_path)
        return metrics
    except Exception as e:
        print(f"[ERROR] Segment {segment_id} failed: {e}")
        return None


def analyze_all_segments(data_dir: str, history_dir: str, output_dir: str):
    """Analyze all available segments."""
    print("=" * 75)
    print("  ANALYZING PERFORMANCE BY HMM REGIME")
    print("=" * 75)

    # Find all history files
    if not os.path.exists(history_dir):
        print(f"[ERROR] History directory not found: {history_dir}")
        return None

    history_files = sorted([f for f in os.listdir(history_dir)
                            if f.endswith('_history.parquet')])

    if not history_files:
        print(f"[ERROR] No history files found in {history_dir}")
        print("Run with --regenerate first to create history files from saved TQC models.")
        return None

    print(f"Found {len(history_files)} segment histories.\n")

    all_metrics = []

    for hf in history_files:
        segment_id = int(hf.split('_')[1])
        try:
            test_df, history_df = load_segment_data(segment_id, data_dir, history_dir)
            merged = merge_regime_data(test_df, history_df)
            metrics = calculate_metrics_per_regime(merged)
            metrics['segment_id'] = segment_id
            all_metrics.append(metrics)
            print_report(metrics, segment_id)
        except Exception as e:
            print(f"[WARNING] Segment {segment_id} failed: {e}")

    if all_metrics:
        # Aggregate across all segments
        combined = pd.concat(all_metrics, ignore_index=True)

        # Weighted average by hours
        agg_metrics = combined.groupby('regime_name').apply(
            lambda x: pd.Series({
                'hours': x['hours'].sum(),
                'win_rate': np.average(x['win_rate'], weights=x['hours']),
                'sharpe': np.average(x['sharpe'], weights=x['hours']),
                'avg_pnl_hour': np.average(x['avg_pnl_hour'], weights=x['hours']),
                'exposure': np.average(x['exposure'], weights=x['hours']),
                'trades': x['trades'].sum(),
                'total_return': x['total_return'].sum(),
            })
        ).reset_index()

        agg_metrics['regime'] = agg_metrics['regime_name'].map(
            {v: k for k, v in REGIME_NAMES.items()})
        agg_metrics['pct_time'] = agg_metrics['hours'] / agg_metrics['hours'].sum() * 100

        print("\n" + "=" * 75)
        print("  AGREGATED ACROSS ALL SEGMENTS (Weighted by Hours)")
        print_report(agg_metrics)

        # Additional summary
        print("\n  SUMMARY:")
        total_hours = agg_metrics['hours'].sum()
        print(f"    Total hours analyzed: {total_hours:.0f}")
        print(f"    Total segments: {len(all_metrics)}")
        print(f"    Total trades: {agg_metrics['trades'].sum():.0f}")

        best_regime = agg_metrics.loc[agg_metrics['sharpe'].idxmax()]
        worst_regime = agg_metrics.loc[agg_metrics['sharpe'].idxmin()]
        print(f"    Best regime:  {best_regime['regime_name']} (Sharpe: {best_regime['sharpe']:+.2f})")
        print(f"    Worst regime: {worst_regime['regime_name']} (Sharpe: {worst_regime['sharpe']:+.2f})")

        # Save plot
        os.makedirs(output_dir, exist_ok=True)
        plot_path = f"{output_dir}/regime_performance.png"
        plot_regime_performance(agg_metrics, plot_path)

        # Save CSV
        csv_path = f"{output_dir}/regime_metrics.csv"
        combined.to_csv(csv_path, index=False)
        print(f"[OK] Detailed metrics saved: {csv_path}")

        # Save aggregated
        agg_csv_path = f"{output_dir}/regime_metrics_aggregated.csv"
        agg_metrics.to_csv(agg_csv_path, index=False)
        print(f"[OK] Aggregated metrics saved: {agg_csv_path}")

        return agg_metrics

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Analyze WFO performance by HMM regime",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Regenerate history for all segments
    python scripts/analyze_regime_performance.py --regenerate

    # Analyze all segments
    python scripts/analyze_regime_performance.py

    # Analyze single segment
    python scripts/analyze_regime_performance.py --segment 0
        """
    )
    parser.add_argument("--data-dir", default="data/wfo",
                        help="Directory with segment test data")
    parser.add_argument("--weights-dir", default="weights/wfo",
                        help="Directory with saved TQC models")
    parser.add_argument("--history-dir", default="results/history",
                        help="Directory with/for segment history files")
    parser.add_argument("--output-dir", default="results",
                        help="Output directory for plots/CSV")
    parser.add_argument("--segment", type=int, default=None,
                        help="Analyze single segment (optional)")
    parser.add_argument("--regenerate", action="store_true",
                        help="Regenerate history from saved TQC models")
    parser.add_argument("--segments", type=str, default=None,
                        help="Comma-separated list of segments to regenerate (e.g., '0,1,2,3,4,5')")

    args = parser.parse_args()

    # Parse segments list
    segments_list = None
    if args.segments:
        segments_list = [int(s.strip()) for s in args.segments.split(',')]

    if args.regenerate:
        # Regenerate histories from saved models
        regenerate_all_histories(
            data_dir=args.data_dir,
            weights_dir=args.weights_dir,
            output_dir=args.history_dir,
            segments=segments_list
        )
        print()

    if args.segment is not None:
        # Single segment analysis
        analyze_single_segment(
            segment_id=args.segment,
            data_dir=args.data_dir,
            history_dir=args.history_dir,
            output_dir=args.output_dir
        )
    else:
        # All segments analysis
        analyze_all_segments(
            data_dir=args.data_dir,
            history_dir=args.history_dir,
            output_dir=args.output_dir
        )


if __name__ == "__main__":
    main()

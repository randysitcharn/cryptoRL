#!/usr/bin/env python3
"""
visualize_hmm_per_segment.py - Visualisation HMM par segment WFO.

Genere une image par segment montrant BTC price avec les regimes HMM colores.

Usage:
    python scripts/visualize_hmm_per_segment.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.data_engineering.manager import RegimeDetector
from src.data_engineering.features import FeatureEngineer


# Configuration
REGIME_COLORS = ['#d62728', '#ff7f0e', '#7f7f7f', '#2ca02c']
REGIME_LABELS = ['Crash', 'Downtrend', 'Range', 'Uptrend']
OUTPUT_DIR = "results/hmm_segments"


def load_and_prepare_data():
    """Charge les donnees historiques et applique le feature engineering."""
    data_path = "data/raw_historical/multi_asset_historical.csv"

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    print(f"Loaded data: {df.shape[0]} rows")

    engineer = FeatureEngineer()
    df = engineer.engineer_features(df)
    df = df.dropna()
    print(f"After feature engineering: {len(df)} rows")

    return df


def simulate_wfo_segments(df, train_rows=8640, test_rows=2160, step_rows=2160, max_segments=13):
    """Simule le decoupage WFO (train 12 mois, test 3 mois)."""
    segments = []
    n_rows = len(df)
    segment_id = 0
    start_idx = 0

    while start_idx + train_rows + test_rows <= n_rows and segment_id < max_segments:
        train_end = start_idx + train_rows
        test_end = train_end + test_rows

        train_df = df.iloc[start_idx:train_end].copy()
        test_df = df.iloc[train_end:test_end].copy()

        segments.append({
            'id': segment_id,
            'train_df': train_df,
            'test_df': test_df,
            'train_start': train_df.index.min(),
            'train_end': train_df.index.max(),
            'test_start': test_df.index.min(),
            'test_end': test_df.index.max()
        })

        segment_id += 1
        start_idx += step_rows

    return segments


def collect_hmm_metrics(segments):
    """Collecte les metriques HMM pour chaque segment."""
    results = []

    for seg in segments:
        print(f"\n--- Segment {seg['id']} ---")

        detector = RegimeDetector(n_components=4, n_mix=2, random_state=42)

        try:
            result_df = detector.fit_predict(seg['train_df'], segment_id=seg['id'])

            prob_cols = ['Prob_0', 'Prob_1', 'Prob_2', 'Prob_3']
            valid_rows = result_df[prob_cols].dropna()

            if len(valid_rows) == 0:
                continue

            probs = valid_rows[prob_cols].values
            dominant = probs.argmax(axis=1)

            proportions = [(dominant == i).sum() / len(dominant) for i in range(4)]

            btc_close = result_df.loc[valid_rows.index, 'BTC_Close'].values
            log_returns = np.zeros(len(btc_close))
            log_returns[1:] = np.log(btc_close[1:] / btc_close[:-1])

            mean_returns = []
            for state in range(4):
                mask = dominant == state
                if mask.sum() > 0:
                    mean_returns.append(log_returns[mask].mean() * 100)
                else:
                    mean_returns.append(0.0)

            n_active = sum(1 for p in proportions if p >= 0.05)
            separation = np.std(mean_returns)

            results.append({
                'segment_id': seg['id'],
                'n_active': n_active,
                'proportions': proportions,
                'mean_returns': mean_returns,
                'separation': separation,
                'train_start': seg['train_start'],
                'train_end': seg['train_end'],
                'result_df': result_df,
                'valid_rows': valid_rows
            })

        except Exception as e:
            print(f"  Error: {e}")

    return results


def plot_segment_timeline(segment_result, output_dir):
    """Genere une visualisation pour un segment."""
    seg_id = segment_result['segment_id']
    result_df = segment_result['result_df']
    valid_rows = segment_result['valid_rows']
    train_start = segment_result['train_start']
    train_end = segment_result['train_end']
    n_active = segment_result['n_active']
    separation = segment_result['separation']
    proportions = segment_result['proportions']

    fig, ax = plt.subplots(figsize=(16, 6))

    # 1. Plot BTC price
    btc_close = result_df['BTC_Close']
    ax.plot(btc_close.index, btc_close.values, 'k-', linewidth=0.8, alpha=0.8)

    # 2. Color by regime
    prob_cols = ['Prob_0', 'Prob_1', 'Prob_2', 'Prob_3']
    probs = valid_rows[prob_cols].values
    dominant = probs.argmax(axis=1)

    for i, (idx, _) in enumerate(valid_rows.iterrows()):
        if i < len(dominant):
            regime = dominant[i]
            ax.axvspan(idx, idx + pd.Timedelta(hours=1), alpha=0.3,
                      color=REGIME_COLORS[regime], linewidth=0)

    # 3. Info box
    props_str = " / ".join([f"{p:.0%}" for p in proportions])
    info_text = (
        f"Segment {seg_id}\n"
        f"Train: {train_start.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')}\n"
        f"Active states: {n_active}/4\n"
        f"Separation: {separation*100:.3f}%\n"
        f"Props: {props_str}"
    )
    ax.text(0.02, 0.97, info_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # 4. Legend
    patches = [mpatches.Patch(color=REGIME_COLORS[i], alpha=0.5, label=REGIME_LABELS[i])
               for i in range(4)]
    ax.legend(handles=patches, loc='upper right')

    # 5. Formatting
    ax.set_xlabel('Date')
    ax.set_ylabel('BTC Price (USD)')
    ax.set_title(f'Segment {seg_id} - BTC Price with HMM Regimes')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # 6. Save
    output_path = os.path.join(output_dir, f"segment_{seg_id:02d}_regimes.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  [OK] Saved: {output_path}")


def main():
    print("=" * 70)
    print("HMM PER-SEGMENT VISUALIZATION")
    print("=" * 70)

    # 1. Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. Load data
    print("\n[1/4] Loading data...")
    df = load_and_prepare_data()

    # 3. Simulate WFO segments
    print("\n[2/4] Simulating WFO segments...")
    segments = simulate_wfo_segments(df)
    print(f"  Generated {len(segments)} segments")

    # 4. Collect HMM metrics
    print("\n[3/4] Collecting HMM metrics...")
    results = collect_hmm_metrics(segments)

    # 5. Generate visualizations
    print("\n[4/4] Generating per-segment visualizations...")
    for r in results:
        plot_segment_timeline(r, OUTPUT_DIR)

    # Summary
    print("\n" + "=" * 70)
    print(f"COMPLETE - {len(results)} visualizations generated:")
    print(f"  Output: {OUTPUT_DIR}/segment_XX_regimes.png")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())

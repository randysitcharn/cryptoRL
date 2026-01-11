#!/usr/bin/env python3
"""
hmm_wfo_viz.py - HMM Walk-Forward Optimization avec Visualisation.

Entraîne le HMM sur chaque segment WFO (12 mois train, 3 mois test)
et génère des visualisations des prédictions de régimes.

Usage:
    python scripts/hmm_wfo_viz.py [--segments N] [--output-dir DIR]

Output:
    - results/hmm_wfo/segment_X_regimes.png (visualisation par segment)
    - results/hmm_wfo/hmm_wfo_summary.png (vue d'ensemble)
    - results/hmm_wfo/hmm_wfo_metrics.csv (métriques)
"""

import os
import sys
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

from src.data_engineering.manager import RegimeDetector
from src.data_engineering.features import FeatureEngineer


# ============================================================================
# Configuration
# ============================================================================

REGIME_COLORS = ['#d62728', '#ff7f0e', '#7f7f7f', '#2ca02c']  # Crash, Down, Range, Up
REGIME_LABELS = ['Crash', 'Downtrend', 'Range', 'Uptrend']
REGIME_CMAP = ListedColormap(REGIME_COLORS)

# WFO Configuration
TRAIN_MONTHS = 12
TEST_MONTHS = 3
STEP_MONTHS = 3
HOURS_PER_MONTH = 720  # ~30 jours * 24h

TRAIN_ROWS = TRAIN_MONTHS * HOURS_PER_MONTH  # 8640
TEST_ROWS = TEST_MONTHS * HOURS_PER_MONTH    # 2160
STEP_ROWS = STEP_MONTHS * HOURS_PER_MONTH    # 2160


# ============================================================================
# Data Loading
# ============================================================================

def load_data(data_path: str = None) -> pd.DataFrame:
    """Charge et prépare les données."""
    if data_path is None:
        # Try multiple paths
        paths = [
            "data/raw_training_data.parquet",
            "data/raw_historical/multi_asset_historical.csv",
        ]
        for p in paths:
            if os.path.exists(p):
                data_path = p
                break

    if data_path is None or not os.path.exists(data_path):
        raise FileNotFoundError("No data file found. Run data pipeline first.")

    print(f"Loading data from: {data_path}")

    if data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)

    print(f"  Raw shape: {df.shape}")

    # Check if features already exist
    if 'BTC_LogRet' not in df.columns:
        print("  Applying feature engineering...")
        engineer = FeatureEngineer()
        df = engineer.engineer_features(df)

    df = df.dropna()
    print(f"  Final shape: {df.shape}")

    return df


# ============================================================================
# WFO Segmentation
# ============================================================================

def create_wfo_segments(df: pd.DataFrame, max_segments: int = None) -> list:
    """Crée les segments WFO (12 mois train, 3 mois test, step 3 mois)."""
    segments = []
    n_rows = len(df)
    segment_id = 0
    start_idx = 0

    while start_idx + TRAIN_ROWS + TEST_ROWS <= n_rows:
        if max_segments and segment_id >= max_segments:
            break

        train_end = start_idx + TRAIN_ROWS
        test_end = train_end + TEST_ROWS

        segments.append({
            'id': segment_id,
            'train_start_idx': start_idx,
            'train_end_idx': train_end,
            'test_start_idx': train_end,
            'test_end_idx': test_end,
            'train_start': df.index[start_idx],
            'train_end': df.index[train_end - 1],
            'test_start': df.index[train_end],
            'test_end': df.index[test_end - 1],
        })

        segment_id += 1
        start_idx += STEP_ROWS

    print(f"Created {len(segments)} WFO segments")
    return segments


# ============================================================================
# HMM Training & Prediction
# ============================================================================

def train_hmm_segment(df: pd.DataFrame, segment: dict) -> dict:
    """Entraîne HMM sur train et prédit sur test."""
    seg_id = segment['id']

    train_df = df.iloc[segment['train_start_idx']:segment['train_end_idx']].copy()
    test_df = df.iloc[segment['test_start_idx']:segment['test_end_idx']].copy()

    print(f"\n[Segment {seg_id}] Train: {segment['train_start'].date()} -> {segment['train_end'].date()}")
    print(f"            Test:  {segment['test_start'].date()} -> {segment['test_end'].date()}")

    # Train HMM
    detector = RegimeDetector(n_components=4, n_mix=2, n_iter=200, random_state=42)

    try:
        train_result = detector.fit_predict(train_df)

        # Extract probabilities from train result
        prob_cols = ['Prob_0', 'Prob_1', 'Prob_2', 'Prob_3']
        proba_train = train_result[prob_cols].dropna().values

        # Predict on test (with context buffer for HMM warmup)
        context_rows = 336  # 2 weeks buffer
        context_start = max(0, segment['test_start_idx'] - context_rows)
        test_with_context = df.iloc[context_start:segment['test_end_idx']].copy()

        test_result = detector.predict(test_with_context)

        # Remove context from results
        actual_test_len = segment['test_end_idx'] - segment['test_start_idx']
        test_result = test_result.iloc[-actual_test_len:]
        proba_test = test_result[prob_cols].dropna().values

        # Get dominant regime
        regime_train = np.argmax(proba_train, axis=1)
        regime_test = np.argmax(proba_test, axis=1)

        # Compute metrics
        regime_counts_train = np.bincount(regime_train, minlength=4)
        regime_counts_test = np.bincount(regime_test, minlength=4)

        metrics = {
            'segment_id': seg_id,
            'train_start': segment['train_start'],
            'train_end': segment['train_end'],
            'test_start': segment['test_start'],
            'test_end': segment['test_end'],
            'train_rows': len(train_df),
            'test_rows': len(test_df),
        }

        for i, label in enumerate(REGIME_LABELS):
            metrics[f'train_{label.lower()}_pct'] = regime_counts_train[i] / len(regime_train) * 100
            metrics[f'test_{label.lower()}_pct'] = regime_counts_test[i] / len(regime_test) * 100

        # Align dataframes with regime arrays (drop NaN rows)
        train_df_valid = train_result.dropna(subset=prob_cols).copy()
        test_df_valid = test_result.dropna(subset=prob_cols).copy()

        return {
            'segment': segment,
            'metrics': metrics,
            'train_df': train_df_valid,
            'test_df': test_df_valid,
            'regime_train': regime_train,
            'regime_test': regime_test,
            'proba_train': proba_train,
            'proba_test': proba_test,
            'detector': detector,
            'success': True,
        }

    except Exception as e:
        print(f"  [ERROR] Segment {seg_id} failed: {e}")
        return {
            'segment': segment,
            'success': False,
            'error': str(e),
        }


# ============================================================================
# Visualization
# ============================================================================

def plot_segment_regimes(result: dict, output_dir: str) -> str:
    """Génère la visualisation des régimes pour un segment."""
    if not result['success']:
        return None

    seg_id = result['segment']['id']
    train_df = result['train_df']
    test_df = result['test_df']
    regime_train = result['regime_train']
    regime_test = result['regime_test']

    fig, axes = plt.subplots(3, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [2, 2, 1]})

    # ========== Plot 1: Train Period ==========
    ax1 = axes[0]
    price_train = train_df['BTC_Close'].values

    # Background colors for regimes
    for i in range(len(regime_train) - 1):
        ax1.axvspan(i, i+1, alpha=0.3, color=REGIME_COLORS[regime_train[i]], linewidth=0)

    ax1.plot(price_train, color='black', linewidth=0.8, label='BTC Price')
    ax1.set_title(f"Segment {seg_id} - TRAIN ({result['segment']['train_start'].date()} -> {result['segment']['train_end'].date()})",
                  fontsize=12, fontweight='bold')
    ax1.set_ylabel('BTC Price (USD)')
    ax1.set_xlim(0, len(price_train))
    ax1.grid(True, alpha=0.3)

    # ========== Plot 2: Test Period ==========
    ax2 = axes[1]
    price_test = test_df['BTC_Close'].values

    for i in range(len(regime_test) - 1):
        ax2.axvspan(i, i+1, alpha=0.3, color=REGIME_COLORS[regime_test[i]], linewidth=0)

    ax2.plot(price_test, color='black', linewidth=0.8, label='BTC Price')
    ax2.set_title(f"Segment {seg_id} - TEST ({result['segment']['test_start'].date()} -> {result['segment']['test_end'].date()})",
                  fontsize=12, fontweight='bold')
    ax2.set_ylabel('BTC Price (USD)')
    ax2.set_xlim(0, len(price_test))
    ax2.grid(True, alpha=0.3)

    # ========== Plot 3: Regime Distribution ==========
    ax3 = axes[2]

    metrics = result['metrics']
    x = np.arange(4)
    width = 0.35

    train_pcts = [metrics[f'train_{l.lower()}_pct'] for l in REGIME_LABELS]
    test_pcts = [metrics[f'test_{l.lower()}_pct'] for l in REGIME_LABELS]

    bars1 = ax3.bar(x - width/2, train_pcts, width, label='Train', color=[c + '80' for c in REGIME_COLORS])
    bars2 = ax3.bar(x + width/2, test_pcts, width, label='Test', color=REGIME_COLORS)

    ax3.set_ylabel('% du temps')
    ax3.set_xticks(x)
    ax3.set_xticklabels(REGIME_LABELS)
    ax3.legend()
    ax3.set_ylim(0, 100)
    ax3.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, val in zip(bars1, train_pcts):
        if val > 5:
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.0f}%',
                    ha='center', va='bottom', fontsize=8)
    for bar, val in zip(bars2, test_pcts):
        if val > 5:
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.0f}%',
                    ha='center', va='bottom', fontsize=8)

    # Legend for regimes
    patches = [mpatches.Patch(color=c, label=l, alpha=0.5) for c, l in zip(REGIME_COLORS, REGIME_LABELS)]
    fig.legend(handles=patches, loc='upper right', bbox_to_anchor=(0.99, 0.99), ncol=4)

    plt.tight_layout()

    # Save
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"segment_{seg_id}_regimes.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  [Plot] Saved: {plot_path}")
    return plot_path


def plot_summary(results: list, output_dir: str) -> str:
    """Génère une vue d'ensemble de tous les segments."""
    successful = [r for r in results if r['success']]
    if not successful:
        return None

    n_segments = len(successful)

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    # ========== Plot 1: Regime Timeline ==========
    ax1 = axes[0]

    for i, result in enumerate(successful):
        seg_id = result['segment']['id']

        # Test regimes (more interesting to visualize)
        regime_test = result['regime_test']
        n_test = len(regime_test)

        # Create colored bar
        for j in range(n_test):
            ax1.barh(seg_id, 1, left=j, color=REGIME_COLORS[regime_test[j]], height=0.8)

    ax1.set_xlabel('Hours in Test Period')
    ax1.set_ylabel('Segment')
    ax1.set_title('Régimes HMM par Segment (Période Test)', fontsize=12, fontweight='bold')
    ax1.set_yticks(range(n_segments))
    ax1.set_yticklabels([f"Seg {r['segment']['id']}" for r in successful])
    ax1.invert_yaxis()

    # Legend
    patches = [mpatches.Patch(color=c, label=l) for c, l in zip(REGIME_COLORS, REGIME_LABELS)]
    ax1.legend(handles=patches, loc='upper right', ncol=4)

    # ========== Plot 2: Regime Distribution Over Time ==========
    ax2 = axes[1]

    segment_ids = [r['segment']['id'] for r in successful]

    # Stack bar chart
    bottoms = np.zeros(n_segments)
    for regime_idx, (color, label) in enumerate(zip(REGIME_COLORS, REGIME_LABELS)):
        pcts = [r['metrics'][f'test_{label.lower()}_pct'] for r in successful]
        ax2.bar(segment_ids, pcts, bottom=bottoms, color=color, label=label)
        bottoms += pcts

    ax2.set_xlabel('Segment')
    ax2.set_ylabel('% du Temps')
    ax2.set_title('Distribution des Régimes par Segment (Test)', fontsize=12, fontweight='bold')
    ax2.set_xticks(segment_ids)
    ax2.legend(loc='upper right', ncol=4)
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save
    plot_path = os.path.join(output_dir, "hmm_wfo_summary.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n[Summary Plot] Saved: {plot_path}")
    return plot_path


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='HMM WFO avec Visualisation')
    parser.add_argument('--segments', type=int, default=None, help='Nombre max de segments')
    parser.add_argument('--output-dir', type=str, default='results/hmm_wfo', help='Dossier output')
    parser.add_argument('--data', type=str, default=None, help='Chemin vers les données')
    args = parser.parse_args()

    print("=" * 70)
    print("HMM WALK-FORWARD OPTIMIZATION - Visualisation des Régimes")
    print("=" * 70)
    print(f"  Train: {TRAIN_MONTHS} mois ({TRAIN_ROWS} rows)")
    print(f"  Test:  {TEST_MONTHS} mois ({TEST_ROWS} rows)")
    print(f"  Step:  {STEP_MONTHS} mois ({STEP_ROWS} rows)")
    print(f"  Output: {args.output_dir}")
    print()

    # Load data
    df = load_data(args.data)

    # Create segments
    segments = create_wfo_segments(df, max_segments=args.segments)

    if not segments:
        print("ERROR: No segments could be created. Check data length.")
        return

    # Process each segment
    results = []
    all_metrics = []

    for segment in segments:
        result = train_hmm_segment(df, segment)
        results.append(result)

        if result['success']:
            # Generate segment plot
            plot_path = plot_segment_regimes(result, args.output_dir)
            result['plot_path'] = plot_path
            all_metrics.append(result['metrics'])

    # Summary visualization
    plot_summary(results, args.output_dir)

    # Save metrics CSV
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        csv_path = os.path.join(args.output_dir, "hmm_wfo_metrics.csv")
        metrics_df.to_csv(csv_path, index=False)
        print(f"\n[Metrics] Saved: {csv_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("RÉSUMÉ")
    print("=" * 70)

    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]

    print(f"  Segments traités: {len(successful)}/{len(results)}")
    if failed:
        print(f"  Segments échoués: {[r['segment']['id'] for r in failed]}")

    print(f"\n  Fichiers générés dans: {args.output_dir}/")
    print(f"    - {len(successful)} images de segments")
    print(f"    - 1 image résumé")
    print(f"    - 1 fichier CSV métriques")


if __name__ == "__main__":
    main()

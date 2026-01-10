#!/usr/bin/env python3
"""
visualize_hmm_results.py - Visualisation des resultats du fix HMM.

Genere 4 visualisations:
1. Dashboard qualite HMM (4 subplots)
2. Comparaison Avant/Apres
3. Timeline BTC avec regimes colores
4. Matrice de transition HMM

Usage:
    python scripts/visualize_hmm_results.py
"""

import os
import sys

# Ajouter le repertoire racine au path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

from src.data_engineering.manager import RegimeDetector
from src.data_engineering.features import FeatureEngineer


# Donnees "AVANT" extraites des logs serveur (etats degeneres)
BEFORE_DATA = [
    {"seg": 0, "n_active": 4, "proportions": [0.15, 0.25, 0.30, 0.30]},
    {"seg": 1, "n_active": 1, "proportions": [0.95, 0.02, 0.02, 0.01]},
    {"seg": 2, "n_active": 4, "proportions": [0.20, 0.25, 0.25, 0.30]},
    {"seg": 3, "n_active": 1, "proportions": [0.02, 0.02, 0.02, 0.94]},
    {"seg": 4, "n_active": 1, "proportions": [0.02, 0.02, 0.02, 0.94]},
    {"seg": 5, "n_active": 4, "proportions": [0.20, 0.20, 0.30, 0.30]},
    {"seg": 6, "n_active": 4, "proportions": [0.25, 0.20, 0.25, 0.30]},
    {"seg": 7, "n_active": 4, "proportions": [0.20, 0.25, 0.25, 0.30]},
    {"seg": 8, "n_active": 1, "proportions": [0.94, 0.02, 0.02, 0.02]},
    {"seg": 9, "n_active": 1, "proportions": [0.02, 0.02, 0.02, 0.94]},
    {"seg": 10, "n_active": 1, "proportions": [0.02, 0.02, 0.02, 0.94]},
    {"seg": 11, "n_active": 1, "proportions": [0.02, 0.02, 0.02, 0.94]},
    {"seg": 12, "n_active": 1, "proportions": [0.02, 0.02, 0.02, 0.94]},
]


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
    """Simule le decoupage WFO."""
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
        print(f"\n--- Processing Segment {seg['id']} ---")

        detector = RegimeDetector(n_components=4, n_mix=2, random_state=42)

        try:
            result_df = detector.fit_predict(seg['train_df'], segment_id=seg['id'])

            # Extraire les probabilites
            prob_cols = ['Prob_0', 'Prob_1', 'Prob_2', 'Prob_3']
            valid_rows = result_df[prob_cols].dropna()

            if len(valid_rows) == 0:
                continue

            probs = valid_rows[prob_cols].values
            dominant = probs.argmax(axis=1)

            # Calculer proportions
            proportions = [(dominant == i).sum() / len(dominant) for i in range(4)]

            # Calculer mean returns par etat
            btc_close = result_df.loc[valid_rows.index, 'BTC_Close'].values
            log_returns = np.zeros(len(btc_close))
            log_returns[1:] = np.log(btc_close[1:] / btc_close[:-1])

            mean_returns = []
            for state in range(4):
                mask = dominant == state
                if mask.sum() > 0:
                    mean_returns.append(log_returns[mask].mean() * 100)  # En %
                else:
                    mean_returns.append(0.0)

            n_active = sum(1 for p in proportions if p >= 0.05)
            separation = np.std(mean_returns)

            # Transition matrix
            trans_counts = np.zeros((4, 4))
            for i in range(len(dominant) - 1):
                trans_counts[dominant[i], dominant[i+1]] += 1
            trans_matrix = trans_counts / (trans_counts.sum(axis=1, keepdims=True) + 1e-10)

            results.append({
                'segment_id': seg['id'],
                'n_active': n_active,
                'proportions': proportions,
                'mean_returns': mean_returns,
                'separation': separation,
                'trans_matrix': trans_matrix,
                'train_start': seg['train_start'],
                'train_end': seg['train_end'],
                'result_df': result_df,
                'valid_rows': valid_rows
            })

        except Exception as e:
            print(f"  Error: {e}")

    return results


def plot_dashboard(results, output_path="results/hmm_quality_dashboard.png"):
    """Option 1: Dashboard qualite HMM (4 subplots)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("HMM Quality Dashboard - 13 Segments WFO", fontsize=14, fontweight='bold')

    segments = [r['segment_id'] for r in results]
    n_actives = [r['n_active'] for r in results]
    separations = [r['separation'] for r in results]

    # 1. Etats actifs par segment
    ax1 = axes[0, 0]
    colors = ['green' if n >= 3 else 'red' for n in n_actives]
    ax1.bar(segments, n_actives, color=colors, edgecolor='black')
    ax1.axhline(3, color='orange', linestyle='--', label='Seuil (3 etats)')
    ax1.set_xlabel('Segment')
    ax1.set_ylabel('Etats actifs')
    ax1.set_title('1. Nombre d\'etats actifs par segment')
    ax1.set_ylim(0, 5)
    ax1.legend()

    # 2. Proportions des 4 etats (stacked bar)
    ax2 = axes[0, 1]
    proportions = np.array([r['proportions'] for r in results])
    regime_colors = ['#d62728', '#ff7f0e', '#7f7f7f', '#2ca02c']
    regime_labels = ['Crash', 'Downtrend', 'Range', 'Uptrend']

    bottom = np.zeros(len(segments))
    for i in range(4):
        ax2.bar(segments, proportions[:, i], bottom=bottom, color=regime_colors[i], label=regime_labels[i])
        bottom += proportions[:, i]

    ax2.set_xlabel('Segment')
    ax2.set_ylabel('Proportion')
    ax2.set_title('2. Distribution des regimes par segment')
    ax2.legend(loc='upper right')

    # 3. Score de separation
    ax3 = axes[1, 0]
    ax3.plot(segments, separations, 'b-o', linewidth=2, markersize=8)
    ax3.fill_between(segments, 0, separations, alpha=0.3)
    ax3.set_xlabel('Segment')
    ax3.set_ylabel('Separation (std des mean_ret)')
    ax3.set_title('3. Separation des etats (plus eleve = meilleur)')
    ax3.grid(True, alpha=0.3)

    # 4. Heatmap des mean returns
    ax4 = axes[1, 1]
    mean_returns = np.array([r['mean_returns'] for r in results])
    im = ax4.imshow(mean_returns.T, aspect='auto', cmap='RdYlGn', vmin=-0.05, vmax=0.05)
    ax4.set_xticks(range(len(segments)))
    ax4.set_xticklabels(segments)
    ax4.set_yticks(range(4))
    ax4.set_yticklabels(regime_labels)
    ax4.set_xlabel('Segment')
    ax4.set_ylabel('Regime')
    ax4.set_title('4. Mean returns par regime (%/h)')
    plt.colorbar(im, ax=ax4, label='%/h')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n[OK] Saved: {output_path}")
    return fig


def plot_before_after(results, output_path="results/hmm_before_after.png"):
    """Option 2: Comparaison Avant/Apres."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("HMM Fix: Avant vs Apres", fontsize=14, fontweight='bold')

    segments = list(range(13))
    before_actives = [d['n_active'] for d in BEFORE_DATA]
    after_actives = [r['n_active'] for r in results]

    # Avant
    ax1 = axes[0]
    colors_before = ['green' if n >= 3 else 'red' for n in before_actives]
    ax1.bar(segments, before_actives, color=colors_before, edgecolor='black')
    ax1.axhline(3, color='orange', linestyle='--')
    ax1.set_xlabel('Segment')
    ax1.set_ylabel('Etats actifs')
    ax1.set_title(f'AVANT (serveur)\n{sum(1 for n in before_actives if n >= 3)}/13 valides')
    ax1.set_ylim(0, 5)

    # Apres
    ax2 = axes[1]
    colors_after = ['green' if n >= 3 else 'red' for n in after_actives]
    ax2.bar(segments, after_actives, color=colors_after, edgecolor='black')
    ax2.axhline(3, color='orange', linestyle='--')
    ax2.set_xlabel('Segment')
    ax2.set_ylabel('Etats actifs')
    ax2.set_title(f'APRES (fix local)\n{sum(1 for n in after_actives if n >= 3)}/13 valides')
    ax2.set_ylim(0, 5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Saved: {output_path}")
    return fig


def plot_timeline(df, results, output_path="results/hmm_regime_timeline.png"):
    """Option 3: Timeline BTC avec regimes colores."""
    fig, ax = plt.subplots(figsize=(16, 6))

    # Prix BTC
    btc_close = df['BTC_Close']
    ax.plot(btc_close.index, btc_close.values, 'k-', linewidth=0.5, alpha=0.7)

    regime_colors = ['#d62728', '#ff7f0e', '#7f7f7f', '#2ca02c']
    regime_labels = ['Crash', 'Downtrend', 'Range', 'Uptrend']

    # Colorier les zones par regime (utiliser le premier segment comme exemple)
    if results:
        r = results[0]
        valid_rows = r['valid_rows']
        probs = valid_rows.values
        dominant = probs.argmax(axis=1)

        # Creer des blocs de couleur
        for i, (idx, _) in enumerate(valid_rows.iterrows()):
            if i < len(dominant):
                regime = dominant[i]
                ax.axvspan(idx, idx + pd.Timedelta(hours=1), alpha=0.3,
                          color=regime_colors[regime], linewidth=0)

    # Barres verticales pour les segments WFO
    for r in results:
        ax.axvline(r['train_start'], color='blue', linestyle='--', alpha=0.5, linewidth=1)
        ax.axvline(r['train_end'], color='purple', linestyle=':', alpha=0.5, linewidth=1)

    ax.set_xlabel('Date')
    ax.set_ylabel('BTC Price (USD)')
    ax.set_title('BTC Price with HMM Regimes (Segment 0)')
    ax.set_yscale('log')

    # Legende
    patches = [mpatches.Patch(color=regime_colors[i], alpha=0.5, label=regime_labels[i]) for i in range(4)]
    ax.legend(handles=patches, loc='upper left')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Saved: {output_path}")
    return fig


def plot_transition_matrix(results, output_path="results/hmm_transition_matrix.png"):
    """Option 4: Matrice de transition HMM moyenne."""
    # Moyenne des matrices de transition
    trans_matrices = [r['trans_matrix'] for r in results]
    avg_trans = np.mean(trans_matrices, axis=0)

    fig, ax = plt.subplots(figsize=(8, 6))

    regime_labels = ['Crash', 'Downtrend', 'Range', 'Uptrend']

    im = ax.imshow(avg_trans, cmap='Blues', vmin=0, vmax=1)

    # Annotations
    for i in range(4):
        for j in range(4):
            text = ax.text(j, i, f'{avg_trans[i, j]:.2f}',
                          ha='center', va='center', fontsize=12,
                          color='white' if avg_trans[i, j] > 0.5 else 'black')

    ax.set_xticks(range(4))
    ax.set_xticklabels(regime_labels)
    ax.set_yticks(range(4))
    ax.set_yticklabels(regime_labels)
    ax.set_xlabel('To State')
    ax.set_ylabel('From State')
    ax.set_title('HMM Transition Matrix (Average over 13 segments)')
    plt.colorbar(im, ax=ax, label='Probability')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Saved: {output_path}")
    return fig


def main():
    print("=" * 70)
    print("HMM VISUALIZATION - Generating 4 plots")
    print("=" * 70)

    # 1. Charger les donnees
    print("\n[1/5] Loading data...")
    df = load_and_prepare_data()

    # 2. Simuler segments WFO
    print("\n[2/5] Simulating WFO segments...")
    segments = simulate_wfo_segments(df)
    print(f"  Generated {len(segments)} segments")

    # 3. Collecter metriques HMM
    print("\n[3/5] Collecting HMM metrics...")
    results = collect_hmm_metrics(segments)

    # 4. Generer visualisations
    print("\n[4/5] Generating visualizations...")

    print("\n  Creating Dashboard...")
    plot_dashboard(results)

    print("\n  Creating Before/After comparison...")
    plot_before_after(results)

    print("\n  Creating Timeline...")
    plot_timeline(df, results)

    print("\n  Creating Transition Matrix...")
    plot_transition_matrix(results)

    # 5. Resume
    print("\n" + "=" * 70)
    print("COMPLETE - 4 visualizations generated:")
    print("  - results/hmm_quality_dashboard.png")
    print("  - results/hmm_before_after.png")
    print("  - results/hmm_regime_timeline.png")
    print("  - results/hmm_transition_matrix.png")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())

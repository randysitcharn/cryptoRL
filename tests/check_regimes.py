"""
check_regimes.py - Visualisation des régimes de marché détectés par le HMM.

Ce script:
1. Charge les données processées (Parquet)
2. Plot le prix BTC avec fond coloré par régime dominant
3. Vérifie la cohérence des 3 régimes (Bull, Bear, Range)

Couleurs:
- Vert: Bull (return positif)
- Rouge: Bear (return négatif)
- Gris: Range (intermédiaire)
"""

import sys
import os

# Ajouter le chemin racine pour les imports
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def load_data(parquet_path: str) -> pd.DataFrame:
    """
    Charge les données depuis le fichier Parquet.

    Args:
        parquet_path: Chemin du fichier Parquet.

    Returns:
        DataFrame des données processées.
    """
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Date range: {df.index.min()} to {df.index.max()}")

    return df


def get_dominant_regime(df: pd.DataFrame) -> pd.Series:
    """
    Calcule le régime dominant (argmax des probabilités).

    Args:
        df: DataFrame avec colonnes Prob_Bear, Prob_Range, Prob_Bull.

    Returns:
        Series avec le nom du régime dominant.
    """
    prob_cols = ['Prob_Bear', 'Prob_Range', 'Prob_Bull']

    # Vérifier que les colonnes existent
    missing = [col for col in prob_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing regime probability columns: {missing}")

    probs = df[prob_cols].values
    dominant_idx = np.argmax(probs, axis=1)

    # Mapper index vers nom
    regime_map = {0: 'Bear', 1: 'Range', 2: 'Bull'}
    dominant = pd.Series([regime_map[i] for i in dominant_idx], index=df.index)

    return dominant


def get_regime_statistics(df: pd.DataFrame) -> dict:
    """
    Calcule les statistiques par régime.

    Args:
        df: DataFrame avec colonnes Prob_Bear, Prob_Range, Prob_Bull.

    Returns:
        Dict avec statistiques par régime.
    """
    dominant = get_dominant_regime(df)
    stats = {}

    for regime in ['Bear', 'Range', 'Bull']:
        mask = dominant == regime
        count = mask.sum()

        if count > 0:
            # Utiliser les variations de prix comme proxy pour les returns
            if 'BTC_Close' in df.columns:
                prices = df.loc[mask, 'BTC_Close']
                returns = prices.pct_change().dropna()
                mean_ret = returns.mean() if len(returns) > 0 else 0
                vol = returns.std() if len(returns) > 0 else 0
            else:
                mean_ret = 0
                vol = 0

            stats[regime] = {
                'count': count,
                'pct': 100 * count / len(df),
                'mean_ret': mean_ret,
                'vol': vol
            }
        else:
            stats[regime] = {'count': 0, 'pct': 0, 'mean_ret': 0, 'vol': 0}

    print("\nRegime statistics:")
    for regime in ['Bear', 'Range', 'Bull']:
        s = stats[regime]
        print(f"  {regime}: count={s['count']} ({s['pct']:.1f}%), "
              f"mean_ret={s['mean_ret']:.6f}, vol={s['vol']:.6f}")

    return stats


def plot_regime_visualization(
    df: pd.DataFrame,
    output_path: str
):
    """
    Génère le graphique avec fond coloré par régime.

    Args:
        df: DataFrame avec BTC_Close et Prob_Bear, Prob_Range, Prob_Bull.
        output_path: Chemin de sauvegarde du graphique.
    """
    # Créer le dossier de sortie
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Obtenir les régimes dominants
    dominant = get_dominant_regime(df)

    # Couleurs par régime (3 états)
    regime_colors = {
        'Bull': '#2ecc71',    # Vert
        'Bear': '#e74c3c',    # Rouge
        'Range': '#95a5a6',   # Gris
    }

    # Créer la figure
    fig, ax = plt.subplots(figsize=(16, 8))

    # Vérifier si BTC_Close existe (non scalé)
    if 'BTC_Close' in df.columns:
        price_col = 'BTC_Close'
    else:
        # Utiliser une autre colonne comme proxy
        price_col = df.columns[0]
        print(f"Warning: BTC_Close not found, using {price_col}")

    # Plot le prix
    ax.plot(df.index, df[price_col], color='black', linewidth=0.8, alpha=0.9)

    # Colorier le fond par segments de régime
    # Trouver les changements de régime (comparaison avec valeur précédente)
    regime_changes = (dominant != dominant.shift(1)).fillna(True)
    change_indices = df.index[regime_changes].tolist()

    # Ajouter le début et la fin
    if df.index[0] not in change_indices:
        change_indices.insert(0, df.index[0])
    change_indices.append(df.index[-1])

    # Dessiner les rectangles colorés
    ymin, ymax = df[price_col].min(), df[price_col].max()
    padding = (ymax - ymin) * 0.05

    for i in range(len(change_indices) - 1):
        start = change_indices[i]
        end = change_indices[i + 1]

        # Trouver le régime pour ce segment (dominant contient déjà le nom)
        regime_name = dominant.loc[start]
        color = regime_colors.get(regime_name, '#ffffff')

        ax.axvspan(start, end, alpha=0.3, color=color, linewidth=0)

    # Configuration du graphique
    ax.set_title('BTC Price with Market Regime Detection (GMM-HMM 3 States)',
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Price (USD)', fontsize=11)
    ax.grid(True, alpha=0.3)

    # Légende (3 régimes)
    legend_patches = [
        mpatches.Patch(color=regime_colors['Bull'], alpha=0.5, label='Bull'),
        mpatches.Patch(color=regime_colors['Range'], alpha=0.5, label='Range'),
        mpatches.Patch(color=regime_colors['Bear'], alpha=0.5, label='Bear'),
    ]
    ax.legend(handles=legend_patches, loc='upper left', fontsize=10)

    # Formatter l'axe X
    fig.autofmt_xdate()

    # Sauvegarder
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n[PLOT] Saved to: {output_path}")


def print_regime_statistics(df: pd.DataFrame):
    """
    Affiche les statistiques détaillées des régimes.

    Args:
        df: DataFrame avec colonnes Prob_Bear, Prob_Range, Prob_Bull.
    """
    dominant = get_dominant_regime(df)

    print("\n" + "=" * 60)
    print("REGIME STATISTICS")
    print("=" * 60)

    total = len(df)

    for regime in ['Bear', 'Range', 'Bull']:
        count = (dominant == regime).sum()
        pct = 100 * count / total

        print(f"\n[{regime}]")
        print(f"  Count: {count} ({pct:.1f}%)")

        # Trouver les périodes continues
        mask = dominant == regime
        if mask.any():
            regime_dates = df.index[mask]
            if len(regime_dates) > 0:
                print(f"  First occurrence: {regime_dates[0]}")
                print(f"  Last occurrence: {regime_dates[-1]}")


def check_bull_periods(df: pd.DataFrame):
    """
    Vérifie si le régime Bull s'active lors des périodes de hausse connues.

    Events à vérifier:
    - BTC Rally Feb-Mar 2024
    - BTC Rally Nov-Dec 2024

    Args:
        df: DataFrame avec colonnes Prob_Bear, Prob_Range, Prob_Bull.
    """
    dominant = get_dominant_regime(df)

    print("\n" + "=" * 60)
    print("BULL PERIOD VERIFICATION")
    print("=" * 60)

    events = [
        ('BTC Rally Feb-Mar 2024', '2024-02-01', '2024-03-31'),
        ('BTC Rally Nov-Dec 2024', '2024-11-01', '2024-12-31'),
    ]

    for event_name, start, end in events:
        try:
            mask = (df.index >= start) & (df.index <= end)
            if mask.sum() == 0:
                print(f"\n[{event_name}] No data available for this period")
                continue

            period_regimes = dominant[mask]
            bull_count = (period_regimes == 'Bull').sum()
            bull_pct = 100 * bull_count / len(period_regimes)

            status = "OK" if bull_pct > 30 else "CHECK"

            print(f"\n[{event_name}] ({start} to {end})")
            print(f"  Bull regime: {bull_count}/{len(period_regimes)} hours ({bull_pct:.1f}%)")
            print(f"  Status: {status}")

        except Exception as e:
            print(f"\n[{event_name}] Error: {e}")


def main():
    """
    Script principal de visualisation.
    """
    print("=" * 60)
    print("REGIME VISUALIZATION CHECK (3 States)")
    print("=" * 60)

    # Chemins
    parquet_path = os.path.join(ROOT_DIR, "data/processed_data.parquet")
    output_path = os.path.join(ROOT_DIR, "logs/regime_visualization.png")

    # 1. Charger les données
    print("\n[1] Loading processed data...")
    try:
        df = load_data(parquet_path)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        print("\nPlease run the DataManager pipeline first:")
        print("  python -c \"from src.data_engineering import DataManager; DataManager().pipeline()\"")
        return

    # 2. Calculer les statistiques par régime
    print("\n[2] Computing regime statistics...")
    get_regime_statistics(df)

    # 3. Afficher les statistiques détaillées
    print_regime_statistics(df)

    # 4. Vérifier les périodes de hausse
    check_bull_periods(df)

    # 5. Générer le graphique
    print("\n[5] Generating visualization...")
    plot_regime_visualization(df, output_path)

    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

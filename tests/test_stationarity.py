"""
test_stationarity.py - Test du module FeatureEngineer.

Ce script:
1. Charge les données multi-actifs
2. Applique le Feature Engineering (FFD, volatilité, Z-Score)
3. Affiche un rapport de stationnarité (d optimal, p-value ADF)
4. Génère un graphique comparant BTC_Close vs BTC_Fracdiff
"""

import sys
import os

# Ajouter le chemin racine pour les imports
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

# Import direct du module features (évite les imports transitifs via __init__.py)
import importlib.util
spec = importlib.util.spec_from_file_location(
    "features",
    os.path.join(ROOT_DIR, "src", "data_engineering", "features.py")
)
features_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(features_module)
FeatureEngineer = features_module.FeatureEngineer


def run_adf_test(series: pd.Series, name: str) -> dict:
    """
    Exécute le test ADF sur une série.

    Args:
        series: Série temporelle à tester.
        name: Nom de la série pour l'affichage.

    Returns:
        Dict avec les résultats ADF.
    """
    series_clean = series.dropna()

    if len(series_clean) < 100:
        return {'name': name, 'statistic': None, 'p_value': None, 'stationary': False}

    try:
        result = adfuller(series_clean, maxlag=1, regression='c')
        return {
            'name': name,
            'statistic': result[0],
            'p_value': result[1],
            'stationary': result[1] < 0.05
        }
    except Exception as e:
        print(f"[ERROR] ADF test failed for {name}: {e}")
        return {'name': name, 'statistic': None, 'p_value': None, 'stationary': False}


def generate_comparison_plot(
    df_original: pd.DataFrame,
    df_features: pd.DataFrame,
    output_path: str = "logs/stationarity_comparison.png"
):
    """
    Génère un graphique comparant BTC_Close (non-stationnaire) vs BTC_Fracdiff (stationnaire).

    Args:
        df_original: DataFrame original avec BTC_Close.
        df_features: DataFrame enrichi avec BTC_Fracdiff.
        output_path: Chemin de sauvegarde du graphique.
    """
    # Créer le dossier logs si nécessaire
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Plot 1: BTC_Close (non-stationnaire)
    ax1 = axes[0]
    btc_close = df_original['BTC_Close'].dropna()
    ax1.plot(btc_close.index, btc_close.values, color='blue', linewidth=0.8)
    ax1.set_title('BTC_Close (Non-Stationary)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Price (USD)')
    ax1.grid(True, alpha=0.3)

    # Ajouter le résultat ADF
    adf_close = run_adf_test(btc_close, 'BTC_Close')
    ax1.text(
        0.02, 0.95,
        f"ADF p-value: {adf_close['p_value']:.4f}\nStatus: {'Stationary' if adf_close['stationary'] else 'Non-Stationary'}",
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    # Plot 2: BTC_Fracdiff (stationnaire)
    ax2 = axes[1]
    if 'BTC_Fracdiff' in df_features.columns:
        btc_fracdiff = df_features['BTC_Fracdiff'].dropna()
        ax2.plot(btc_fracdiff.index, btc_fracdiff.values, color='green', linewidth=0.8)
        ax2.set_title('BTC_Fracdiff (Stationary - Fractional Differentiation)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Fracdiff Value')
        ax2.grid(True, alpha=0.3)

        # Ajouter le résultat ADF
        adf_fracdiff = run_adf_test(btc_fracdiff, 'BTC_Fracdiff')
        ax2.text(
            0.02, 0.95,
            f"ADF p-value: {adf_fracdiff['p_value']:.4f}\nStatus: {'Stationary' if adf_fracdiff['stationary'] else 'Non-Stationary'}",
            transform=ax2.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5)
        )
    else:
        ax2.text(0.5, 0.5, 'BTC_Fracdiff not found', ha='center', va='center', fontsize=14)

    ax2.set_xlabel('Date')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n[PLOT] Saved comparison plot to: {output_path}")


def main():
    """
    Script principal de test.
    """
    print("=" * 60)
    print("STATIONARITY TEST - FeatureEngineer Module")
    print("=" * 60)

    # 1. Charger les données
    data_path = "data/processed/multi_asset.csv"
    print(f"\n[1] Loading data from {data_path}...")

    if not os.path.exists(data_path):
        print(f"[ERROR] File not found: {data_path}")
        print("Please run the MultiAssetDownloader first.")
        return

    df_original = pd.read_csv(data_path, index_col=0, parse_dates=True)
    print(f"    Loaded: {df_original.shape[0]} rows, {df_original.shape[1]} columns")

    # 2. Appliquer le Feature Engineering
    print("\n[2] Applying Feature Engineering...")
    engineer = FeatureEngineer(
        ffd_window=100,
        d_range=(0.0, 1.0, 0.05),
        vol_window=24,
        zscore_window=720
    )

    df_features = engineer.engineer_features(df_original.copy())

    # 3. Afficher le rapport de stationnarité
    print("\n[3] Stationarity Report:")
    print(engineer.get_stationarity_report())

    # 4. Rapport détaillé
    print("\n[4] Detailed ADF Results:")
    print("-" * 60)

    # Test ADF sur les séries originales (Close)
    print("\n--- Original Close Series (Expected: Non-Stationary) ---")
    for asset in ['BTC', 'ETH', 'SPX', 'DXY']:
        col = f"{asset}_Close"
        if col in df_original.columns:
            result = run_adf_test(df_original[col], col)
            status = "Stationary" if result['stationary'] else "Non-Stationary"
            print(f"  {col}: p-value = {result['p_value']:.6f} ({status})")

    # Test ADF sur les séries Fracdiff
    print("\n--- Fracdiff Series (Expected: Stationary) ---")
    for asset in ['BTC', 'ETH', 'SPX', 'DXY']:
        col = f"{asset}_Fracdiff"
        if col in df_features.columns:
            result = run_adf_test(df_features[col], col)
            d_opt = engineer.optimal_d.get(asset, 'N/A')
            status = "Stationary" if result['stationary'] else "Non-Stationary"
            print(f"  {col}: d={d_opt}, p-value = {result['p_value']:.6f} ({status})")

    # 5. Générer le graphique
    print("\n[5] Generating comparison plot...")
    generate_comparison_plot(df_original, df_features)

    # 6. Afficher les nouvelles colonnes
    print("\n[6] New Feature Columns:")
    new_cols = [c for c in df_features.columns if c not in df_original.columns]
    for col in sorted(new_cols):
        print(f"    - {col}")

    # 7. Statistiques finales
    print("\n[7] Final Statistics:")
    print(f"    Original columns: {len(df_original.columns)}")
    print(f"    Final columns: {len(df_features.columns)}")
    print(f"    New features: {len(new_cols)}")
    print(f"    Final rows: {len(df_features)}")

    # 8. Vérifier les NaN
    nan_count = df_features.isna().sum().sum()
    print(f"    NaN values: {nan_count}")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

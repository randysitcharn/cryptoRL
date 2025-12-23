"""
debug_eth_stationarity.py - Audit de la stationnarité ETH.

Compare 3 valeurs de d pour la Différenciation Fractionnaire:
- d=0.10 (ancien, problématique - corrélation trop haute)
- d=0.30 (nouveau plancher min_d_floor)
- d=0.40 (standard littérature)

Métriques analysées:
- ADF Statistic + Critical Value 1%
- Corrélation Pearson avec le prix brut (DANGER si > 0.90)
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

# Configuration
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)


def get_weights_ffd(d: float, threshold: float = 1e-5) -> np.ndarray:
    """
    Calcule les poids FFD (Fixed-Width Window Fractional Differentiation).

    Formule: w_k = -w_{k-1} * (d - k + 1) / k
    """
    weights = [1.0]
    k = 1

    while True:
        w_k = -weights[-1] * (d - k + 1) / k
        if abs(w_k) < threshold:
            break
        weights.append(w_k)
        k += 1
        if k > 500:  # Limite de sécurité
            break

    return np.array(weights[::-1])


def apply_ffd(series: pd.Series, d: float) -> pd.Series:
    """
    Applique la Différenciation Fractionnaire (FFD).
    """
    weights = get_weights_ffd(d)
    width = len(weights)

    result = pd.Series(index=series.index, dtype=np.float64)

    for i in range(width - 1, len(series)):
        window = series.iloc[i - width + 1:i + 1].values
        if len(window) == width:
            result.iloc[i] = np.dot(weights, window)

    return result


def run_adf_detailed(series: pd.Series) -> dict:
    """
    Exécute le test ADF et retourne les détails complets.
    """
    series_clean = series.dropna()

    if len(series_clean) < 100:
        return {
            'statistic': None,
            'p_value': None,
            'critical_1pct': None,
            'critical_5pct': None,
            'stationary': False
        }

    try:
        result = adfuller(series_clean, maxlag=1, regression='c')
        return {
            'statistic': result[0],
            'p_value': result[1],
            'critical_1pct': result[4]['1%'],
            'critical_5pct': result[4]['5%'],
            'stationary': result[1] < 0.05
        }
    except Exception as e:
        print(f"[ERROR] ADF failed: {e}")
        return {
            'statistic': None,
            'p_value': None,
            'critical_1pct': None,
            'critical_5pct': None,
            'stationary': False
        }


def compute_correlation(series1: pd.Series, series2: pd.Series) -> float:
    """
    Calcule la corrélation de Pearson entre deux séries.
    """
    # Aligner les indices et supprimer les NaN
    df = pd.DataFrame({'s1': series1, 's2': series2}).dropna()

    if len(df) < 10:
        return np.nan

    return df['s1'].corr(df['s2'])


def generate_plot(
    eth_close: pd.Series,
    fracdiff_010: pd.Series,
    fracdiff_030: pd.Series,
    fracdiff_040: pd.Series,
    output_path: str
):
    """
    Génère le graphique comparatif avec 4 subplots.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    # Plot 1: Prix Brut
    ax1 = axes[0]
    eth_clean = eth_close.dropna()
    ax1.plot(eth_clean.index, eth_clean.values, color='blue', linewidth=0.8)
    ax1.set_title('ETH_Close (Raw Price - Non-Stationary)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Price (USD)')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Fracdiff d=0.10
    ax2 = axes[1]
    fd010_clean = fracdiff_010.dropna()
    ax2.plot(fd010_clean.index, fd010_clean.values, color='red', linewidth=0.8)
    ax2.set_title('ETH_Fracdiff (d=0.10) - PROBLEMATIC: Corr > 0.99', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Fracdiff Value')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # Plot 3: Fracdiff d=0.30
    ax3 = axes[2]
    fd030_clean = fracdiff_030.dropna()
    ax3.plot(fd030_clean.index, fd030_clean.values, color='orange', linewidth=0.8)
    ax3.set_title('ETH_Fracdiff (d=0.30) - NEW FLOOR (min_d_floor)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Fracdiff Value')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # Plot 4: Fracdiff d=0.40
    ax4 = axes[3]
    fd040_clean = fracdiff_040.dropna()
    ax4.plot(fd040_clean.index, fd040_clean.values, color='green', linewidth=0.8)
    ax4.set_title('ETH_Fracdiff (d=0.40) - Standard Literature Value', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Fracdiff Value')
    ax4.set_xlabel('Date')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n[PLOT] Saved to: {output_path}")


def print_d_report(d_value: float, adf: dict, corr: float, label: str):
    """
    Affiche le rapport pour une valeur de d.
    """
    print(f"\n[d={d_value:.2f}] ({label})")
    print(f"  ADF Stat: {adf['statistic']:.4f} | Critical 1%: {adf['critical_1pct']:.4f} | P-Value: {adf['p_value']:.6f}")

    if adf['statistic'] < adf['critical_1pct']:
        print(f"  ADF Check: PASS (Stat < Critical 1%)")
    else:
        print(f"  ADF Check: FAIL (Stat > Critical 1%)")

    if corr > 0.90:
        danger = "DANGER - Almost identical to raw price!"
    elif corr > 0.80:
        danger = "WARNING - High correlation"
    else:
        danger = "OK"
    print(f"  Correlation with Price: {corr:.4f} ({danger})")


def main():
    """
    Script principal d'audit.
    """
    # 1. Charger les données
    data_path = os.path.join(ROOT_DIR, "data/processed/multi_asset.csv")

    if not os.path.exists(data_path):
        print(f"[ERROR] File not found: {data_path}")
        return

    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    eth_close = df['ETH_Close']

    print("=" * 60)
    print("ETH STATIONARITY AUDIT (3 Comparisons)")
    print("=" * 60)
    print(f"\nData loaded: {len(eth_close)} rows")

    # 2. Générer les versions transformées
    print("\n[1] Applying Fractional Differentiation...")
    fracdiff_010 = apply_ffd(eth_close, d=0.10)
    fracdiff_030 = apply_ffd(eth_close, d=0.30)
    fracdiff_040 = apply_ffd(eth_close, d=0.40)

    print(f"    d=0.10: {fracdiff_010.dropna().shape[0]} valid values")
    print(f"    d=0.30: {fracdiff_030.dropna().shape[0]} valid values")
    print(f"    d=0.40: {fracdiff_040.dropna().shape[0]} valid values")

    # 3. Calculer les métriques
    print("\n[2] Computing ADF statistics...")

    adf_raw = run_adf_detailed(eth_close)
    adf_010 = run_adf_detailed(fracdiff_010)
    adf_030 = run_adf_detailed(fracdiff_030)
    adf_040 = run_adf_detailed(fracdiff_040)

    print("\n[3] Computing correlations...")
    corr_010 = compute_correlation(eth_close, fracdiff_010)
    corr_030 = compute_correlation(eth_close, fracdiff_030)
    corr_040 = compute_correlation(eth_close, fracdiff_040)

    # 4. Afficher le rapport
    print("\n")
    print("=" * 60)
    print("--- AUDIT REPORT: ETH ---")
    print("=" * 60)

    # Prix brut (référence)
    print(f"\n[RAW PRICE]")
    print(f"  ADF Stat: {adf_raw['statistic']:.4f} | Critical 1%: {adf_raw['critical_1pct']:.4f} | P-Value: {adf_raw['p_value']:.4f}")
    status_raw = "STATIONARY" if adf_raw['stationary'] else "NON-STATIONARY"
    print(f"  Status: {status_raw}")

    # d=0.10
    print_d_report(0.10, adf_010, corr_010, "OLD - Problematic")

    # d=0.30
    print_d_report(0.30, adf_030, corr_030, "NEW FLOOR - min_d_floor")

    # d=0.40
    print_d_report(0.40, adf_040, corr_040, "Standard Literature")

    # Tableau récapitulatif
    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    print(f"\n{'d':<8} {'ADF Stat':<12} {'P-Value':<12} {'Corr':<10} {'Status':<20}")
    print("-" * 60)
    print(f"{'0.10':<8} {adf_010['statistic']:<12.4f} {adf_010['p_value']:<12.6f} {corr_010:<10.4f} {'DANGER (corr>0.90)' if corr_010 > 0.90 else 'OK':<20}")
    print(f"{'0.30':<8} {adf_030['statistic']:<12.4f} {adf_030['p_value']:<12.6f} {corr_030:<10.4f} {'DANGER (corr>0.90)' if corr_030 > 0.90 else 'OK':<20}")
    print(f"{'0.40':<8} {adf_040['statistic']:<12.4f} {adf_040['p_value']:<12.6f} {corr_040:<10.4f} {'DANGER (corr>0.90)' if corr_040 > 0.90 else 'OK':<20}")

    # Conclusion
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)

    if corr_030 < 0.90:
        print("\n[OK] d=0.30 (min_d_floor) is VALID!")
        print(f"  Correlation {corr_030:.4f} is below the 0.90 danger threshold.")
        print("  The series is properly transformed and stationary.")
    else:
        print("\n[WARNING] d=0.30 still has high correlation!")
        print("  Consider increasing min_d_floor to 0.40.")

    print("\n")

    # 5. Générer le graphique
    print("[4] Generating comparison plot...")
    output_path = os.path.join(ROOT_DIR, "logs/eth_audit_comparison.png")
    generate_plot(eth_close, fracdiff_010, fracdiff_030, fracdiff_040, output_path)

    print("\n" + "=" * 60)
    print("AUDIT COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

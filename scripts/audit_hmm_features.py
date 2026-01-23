"""
Audit HMM Features Engineering - P1.1
Tests: stationnarité, look-ahead bias, multicollinéarité, clipping, windows
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from src.data_engineering.manager import RegimeDetector
from src.data_engineering.loader import MultiAssetDownloader

def load_test_data():
    """Charge les données de test"""
    print("=" * 80)
    print("AUDIT HMM FEATURES - P1.1")
    print("=" * 80)
    
    # Essayer de charger depuis raw_historical
    historical_path = "data/raw_historical/multi_asset_historical.csv"
    if os.path.exists(historical_path):
        print(f"Loading from {historical_path}...")
        df = pd.read_csv(historical_path, index_col=0, parse_dates=True)
    else:
        print("Downloading data...")
        downloader = MultiAssetDownloader()
        df = downloader.download_multi_asset()
    
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    return df

def test_stationarity(series, name, alpha=0.05):
    """Test de stationnarité (ADF et KPSS)"""
    results = {}
    
    # ADF Test (H0: non-stationary)
    try:
        adf_result = adfuller(series.dropna(), autolag='AIC')
        results['adf'] = {
            'statistic': adf_result[0],
            'pvalue': adf_result[1],
            'is_stationary': adf_result[1] < alpha,
            'critical_values': adf_result[4]
        }
    except Exception as e:
        results['adf'] = {'error': str(e)}
    
    # KPSS Test (H0: stationary)
    try:
        kpss_result = kpss(series.dropna(), regression='c', nlags='auto')
        results['kpss'] = {
            'statistic': kpss_result[0],
            'pvalue': kpss_result[1],
            'is_stationary': kpss_result[1] > alpha,
            'critical_values': kpss_result[3]
        }
    except Exception as e:
        results['kpss'] = {'error': str(e)}
    
    # Consensus
    if 'adf' in results and 'kpss' in results:
        if 'is_stationary' in results['adf'] and 'is_stationary' in results['kpss']:
            results['consensus'] = results['adf']['is_stationary'] and results['kpss']['is_stationary']
        else:
            results['consensus'] = None
    else:
        results['consensus'] = None
    
    return results

def test_lookahead_bias(df, detector):
    """Test de look-ahead bias dans les features"""
    print("\n" + "=" * 80)
    print("TEST 2: Look-Ahead Bias Detection")
    print("=" * 80)
    
    issues = []
    
    # Vérifier que les rolling windows utilisent min_periods correctement
    features = detector._compute_hmm_features(df)
    
    # Test: vérifier que les premières valeurs sont NaN si window pas remplie
    for feature in detector.HMM_FEATURES:
        feature_series = features[feature]
        first_valid_idx = feature_series.first_valid_index()
        
        # HMM_Trend, HMM_Vol, HMM_RiskOnOff: window=168h
        if feature in ['HMM_Trend', 'HMM_Vol', 'HMM_RiskOnOff']:
            expected_nan = 168
        # HMM_Momentum: window=14h
        elif feature == 'HMM_Momentum':
            expected_nan = 14
        # HMM_VolRatio: max(24, 168) = 168h
        elif feature == 'HMM_VolRatio':
            expected_nan = 168
        else:
            expected_nan = None
        
        if expected_nan is not None:
            nan_count = feature_series.isna().sum()
            if nan_count < expected_nan:
                issues.append(f"{feature}: Only {nan_count} NaN values, expected at least {expected_nan}")
            elif nan_count > expected_nan + 10:  # Tolérance
                issues.append(f"{feature}: {nan_count} NaN values, expected ~{expected_nan}")
    
    # Test: vérifier qu'on n'utilise pas de données futures
    # Simuler un point dans le temps et vérifier que seules les données passées sont utilisées
    test_idx = 500
    test_data = df.iloc[:test_idx].copy()
    test_features = detector._compute_hmm_features(test_data)
    
    # Comparer avec le calcul sur tout le dataset
    full_features = detector._compute_hmm_features(df)
    
    for feature in detector.HMM_FEATURES:
        if test_idx < len(full_features):
            val_partial = test_features[feature].iloc[-1]
            val_full = full_features[feature].iloc[test_idx]
            
            if pd.notna(val_partial) and pd.notna(val_full):
                diff = abs(val_partial - val_full)
                if diff > 1e-6:
                    issues.append(f"{feature}: Look-ahead bias detected (diff={diff:.2e})")
    
    if issues:
        print("❌ ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
        return False, issues
    else:
        print("✅ No look-ahead bias detected")
        return True, []

def test_multicollinearity(features_df):
    """Test de multicollinéarité (VIF)"""
    print("\n" + "=" * 80)
    print("TEST 3: Multicollinearity (VIF)")
    print("=" * 80)
    
    # Sélectionner uniquement les features HMM valides
    hmm_cols = [col for col in features_df.columns if col.startswith('HMM_')]
    valid_data = features_df[hmm_cols].dropna()
    
    if len(valid_data) < 100:
        print(f"⚠️  Not enough data for VIF: {len(valid_data)} samples")
        return None
    
    # Calculer VIF
    vif_data = pd.DataFrame()
    vif_data["Feature"] = hmm_cols
    vif_data["VIF"] = [variance_inflation_factor(valid_data.values, i) 
                       for i in range(len(hmm_cols))]
    
    print("\nVIF Results:")
    print(vif_data.to_string(index=False))
    
    # Identifier les problèmes (VIF > 5 = multicollinéarité modérée, > 10 = forte)
    high_vif = vif_data[vif_data['VIF'] > 5]
    if len(high_vif) > 0:
        print(f"\n⚠️  Features with VIF > 5 (multicollinearity):")
        print(high_vif.to_string(index=False))
    
    very_high_vif = vif_data[vif_data['VIF'] > 10]
    if len(very_high_vif) > 0:
        print(f"\n❌ Features with VIF > 10 (strong multicollinearity):")
        print(very_high_vif.to_string(index=False))
        return False, vif_data
    
    print("\n✅ No strong multicollinearity detected (all VIF < 10)")
    return True, vif_data

def test_correlation_matrix(features_df):
    """Matrice de corrélation entre features"""
    print("\n" + "=" * 80)
    print("TEST 4: Correlation Matrix")
    print("=" * 80)
    
    hmm_cols = [col for col in features_df.columns if col.startswith('HMM_')]
    valid_data = features_df[hmm_cols].dropna()
    
    corr_matrix = valid_data.corr()
    
    print("\nCorrelation Matrix:")
    print(corr_matrix.round(3).to_string())
    
    # Identifier les corrélations élevées (> 0.7)
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:
                high_corr_pairs.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_val
                ))
    
    if high_corr_pairs:
        print(f"\n⚠️  High correlations (|r| > 0.7):")
        for feat1, feat2, corr in high_corr_pairs:
            print(f"  {feat1} <-> {feat2}: {corr:.3f}")
    else:
        print("\n✅ No high correlations detected (all |r| < 0.7)")
    
    return corr_matrix

def test_clipping_justification(features_df, detector):
    """Valider que les clips sont justifiés"""
    print("\n" + "=" * 80)
    print("TEST 5: Clipping Justification")
    print("=" * 80)
    
    issues = []
    clipping_rules = {
        'HMM_Trend': (-0.05, 0.05),
        'HMM_Vol': (0, 0.2),
        'HMM_Momentum': (0, 1),
        'HMM_RiskOnOff': (-0.02, 0.02),
        'HMM_VolRatio': (0.2, 5.0)
    }
    
    for feature, (clip_min, clip_max) in clipping_rules.items():
        if feature not in features_df.columns:
            continue
        
        series = features_df[feature].dropna()
        if len(series) == 0:
            continue
        
        # Statistiques avant clipping
        pct_clipped_min = (series < clip_min).sum() / len(series) * 100
        pct_clipped_max = (series > clip_max).sum() / len(series) * 100
        pct_clipped_total = pct_clipped_min + pct_clipped_max
        
        # Vérifier si le clipping est trop agressif (> 5% des valeurs)
        if pct_clipped_total > 5:
            issues.append(
                f"{feature}: {pct_clipped_total:.2f}% of values would be clipped "
                f"({pct_clipped_min:.2f}% below {clip_min}, {pct_clipped_max:.2f}% above {clip_max})"
            )
        
        # Vérifier les valeurs extrêmes
        min_val = series.min()
        max_val = series.max()
        
        print(f"\n{feature}:")
        print(f"  Range: [{min_val:.6f}, {max_val:.6f}]")
        print(f"  Clip bounds: [{clip_min}, {clip_max}]")
        print(f"  Would clip: {pct_clipped_total:.2f}% of values")
        print(f"  Percentiles: 1%={series.quantile(0.01):.6f}, 99%={series.quantile(0.99):.6f}")
    
    if issues:
        print("\n⚠️  CLIPPING ISSUES:")
        for issue in issues:
            print(f"  - {issue}")
        return False, issues
    else:
        print("\n✅ Clipping appears reasonable (< 5% of values clipped)")
        return True, []

def main():
    """Exécute tous les tests d'audit"""
    # Charger les données
    df = load_test_data()
    
    # Initialiser le détecteur
    detector = RegimeDetector(n_components=4)
    
    # Calculer les features
    print("\n" + "=" * 80)
    print("Computing HMM features...")
    print("=" * 80)
    features_df = detector._compute_hmm_features(df)
    
    # TEST 1: Stationnarité
    print("\n" + "=" * 80)
    print("TEST 1: Stationarity Tests (ADF + KPSS)")
    print("=" * 80)
    
    stationarity_results = {}
    for feature in detector.HMM_FEATURES:
        if feature not in features_df.columns:
            continue
        
        series = features_df[feature].dropna()
        if len(series) < 100:
            print(f"\n⚠️  {feature}: Not enough data ({len(series)} samples)")
            continue
        
        print(f"\n{feature}:")
        results = test_stationarity(series, feature)
        stationarity_results[feature] = results
        
        if 'adf' in results and 'pvalue' in results['adf']:
            adf_pval = results['adf']['pvalue']
            adf_stat = results['adf']['statistic']
            print(f"  ADF: statistic={adf_stat:.4f}, p-value={adf_pval:.4f}, "
                  f"stationary={'✅' if adf_pval < 0.05 else '❌'}")
        
        if 'kpss' in results and 'pvalue' in results['kpss']:
            kpss_pval = results['kpss']['pvalue']
            kpss_stat = results['kpss']['statistic']
            print(f"  KPSS: statistic={kpss_pval:.4f}, p-value={kpss_pval:.4f}, "
                  f"stationary={'✅' if kpss_pval > 0.05 else '❌'}")
        
        if 'consensus' in results and results['consensus'] is not None:
            print(f"  Consensus: {'✅ Stationary' if results['consensus'] else '❌ Non-stationary'}")
    
    # TEST 2: Look-ahead bias
    lookahead_ok, lookahead_issues = test_lookahead_bias(df, detector)
    
    # TEST 3: Multicollinéarité
    multicoll_ok, vif_data = test_multicollinearity(features_df)
    
    # TEST 4: Correlation matrix
    corr_matrix = test_correlation_matrix(features_df)
    
    # TEST 5: Clipping justification
    clipping_ok, clipping_issues = test_clipping_justification(features_df, detector)
    
    # SYNTHÈSE
    print("\n" + "=" * 80)
    print("SYNTHÈSE - FINDINGS")
    print("=" * 80)
    
    findings = {
        'P0': [],
        'P1': [],
        'P2': []
    }
    
    # Analyser les résultats de stationnarité
    non_stationary = [f for f, r in stationarity_results.items() 
                     if 'consensus' in r and r['consensus'] is False]
    if non_stationary:
        findings['P1'].append(f"Non-stationary features: {', '.join(non_stationary)}")
    
    # Look-ahead bias
    if not lookahead_ok:
        findings['P0'].append(f"Look-ahead bias detected: {len(lookahead_issues)} issues")
    
    # Multicollinéarité
    if multicoll_ok is False:
        high_vif_features = vif_data[vif_data['VIF'] > 10]['Feature'].tolist()
        findings['P1'].append(f"Strong multicollinearity (VIF > 10): {', '.join(high_vif_features)}")
    elif multicoll_ok is None:
        findings['P2'].append("Could not compute VIF (insufficient data)")
    
    # Clipping
    if not clipping_ok:
        findings['P2'].append(f"Clipping may be too aggressive: {len(clipping_issues)} features")
    
    # Afficher les findings
    print("\nFINDINGS BY PRIORITY:")
    for priority in ['P0', 'P1', 'P2']:
        if findings[priority]:
            print(f"\n{priority} (Critical/Important/Minor):")
            for finding in findings[priority]:
                print(f"  - {finding}")
        else:
            print(f"\n{priority}: ✅ No issues")
    
    # Score global
    total_issues = sum(len(v) for v in findings.values())
    if total_issues == 0:
        score = 10
    elif len(findings['P0']) == 0 and len(findings['P1']) == 0:
        score = 8 - len(findings['P2']) * 0.5
    elif len(findings['P0']) == 0:
        score = 6 - len(findings['P1']) * 0.5
    else:
        score = max(0, 4 - len(findings['P0']) * 1.0)
    
    print(f"\n{'=' * 80}")
    print(f"SCORE GLOBAL: {score:.1f}/10")
    print(f"{'=' * 80}")
    
    return {
        'stationarity': stationarity_results,
        'lookahead': {'ok': lookahead_ok, 'issues': lookahead_issues},
        'multicollinearity': {'ok': multicoll_ok, 'vif': vif_data},
        'correlation': corr_matrix,
        'clipping': {'ok': clipping_ok, 'issues': clipping_issues},
        'findings': findings,
        'score': score
    }

if __name__ == "__main__":
    results = main()

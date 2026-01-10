#!/usr/bin/env python3
"""
test_hmm_fix.py - Test de validation du fix HMM.

Simule le découpage WFO sur les données historiques et vérifie
que le HMM produit des états distincts pour chaque segment.

Usage:
    python scripts/test_hmm_fix.py
"""

import os
import sys

# Ajouter le répertoire racine au path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from src.data_engineering.manager import RegimeDetector
from src.data_engineering.features import FeatureEngineer


def load_historical_data(data_path: str = "data/raw_historical/multi_asset_historical.csv") -> pd.DataFrame:
    """Charge les données historiques."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    print(f"Loaded data: {df.shape[0]} rows, {df.index.min()} to {df.index.max()}")
    return df


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Applique le feature engineering nécessaire pour HMM."""
    engineer = FeatureEngineer()
    df = engineer.engineer_features(df)
    return df


def simulate_wfo_segments(
    df: pd.DataFrame,
    train_rows: int = 8640,  # 12 mois
    test_rows: int = 2160,   # 3 mois
    step_rows: int = 2160    # 3 mois
) -> list:
    """
    Simule le découpage WFO.

    Returns:
        Liste de tuples (segment_id, train_df, test_df)
    """
    segments = []
    n_rows = len(df)

    segment_id = 0
    start_idx = 0

    while start_idx + train_rows + test_rows <= n_rows:
        train_end = start_idx + train_rows
        test_end = train_end + test_rows

        train_df = df.iloc[start_idx:train_end].copy()
        test_df = df.iloc[train_end:test_end].copy()

        segments.append((segment_id, train_df, test_df))

        segment_id += 1
        start_idx += step_rows

        # Limiter à 13 segments comme dans le WFO original
        if segment_id >= 13:
            break

    return segments


def test_hmm_on_segment(segment_id: int, train_df: pd.DataFrame) -> dict:
    """
    Teste le HMM sur un segment.

    Returns:
        Dict avec les métriques de qualité
    """
    detector = RegimeDetector(n_components=4, n_mix=2, random_state=42)

    try:
        result_df = detector.fit_predict(train_df, segment_id=segment_id)

        # Extraire les métriques de qualité
        prob_cols = ['Prob_0', 'Prob_1', 'Prob_2', 'Prob_3']
        valid_rows = result_df[prob_cols].dropna()

        if len(valid_rows) == 0:
            return {
                'segment_id': segment_id,
                'success': False,
                'error': 'No valid HMM predictions',
                'n_active_states': 0,
                'state_proportions': [0, 0, 0, 0],
                'is_valid': False
            }

        # Calculer les proportions d'états
        probs = valid_rows[prob_cols].values
        dominant = probs.argmax(axis=1)

        proportions = []
        for state in range(4):
            prop = (dominant == state).sum() / len(dominant)
            proportions.append(prop)

        n_active = sum(1 for p in proportions if p >= 0.05)

        return {
            'segment_id': segment_id,
            'success': True,
            'n_active_states': n_active,
            'state_proportions': proportions,
            'is_valid': n_active >= 3
        }

    except Exception as e:
        return {
            'segment_id': segment_id,
            'success': False,
            'error': str(e),
            'n_active_states': 0,
            'state_proportions': [0, 0, 0, 0],
            'is_valid': False
        }


def main():
    print("=" * 70)
    print("TEST HMM FIX - Validation des états distincts")
    print("=" * 70)

    # 1. Charger les données
    print("\n[1/4] Loading historical data...")
    try:
        df = load_historical_data()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Please ensure data/raw_historical/multi_asset_historical.csv exists")
        return 1

    # 2. Feature engineering
    print("\n[2/4] Applying feature engineering...")
    df = apply_feature_engineering(df)
    df = df.dropna()
    print(f"  After cleanup: {len(df)} rows")

    # 3. Simuler WFO segments
    print("\n[3/4] Simulating WFO segments...")
    segments = simulate_wfo_segments(df)
    print(f"  Generated {len(segments)} segments")

    # 4. Tester HMM sur chaque segment
    print("\n[4/4] Testing HMM on each segment...")
    print("-" * 70)

    results = []
    for segment_id, train_df, test_df in segments:
        print(f"\n--- Segment {segment_id} ---")
        print(f"  Train: {len(train_df)} rows ({train_df.index.min()} to {train_df.index.max()})")

        result = test_hmm_on_segment(segment_id, train_df)
        results.append(result)

        if result['success']:
            props_str = ", ".join([f"{p:.1%}" for p in result['state_proportions']])
            status = "[OK] VALID" if result['is_valid'] else "[X] INVALID"
            print(f"  Result: {status} (n_active={result['n_active_states']}, props=[{props_str}])")
        else:
            print(f"  Result: [X] ERROR - {result.get('error', 'Unknown')}")

    # Résumé
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    n_valid = sum(1 for r in results if r['is_valid'])
    n_total = len(results)
    success_rate = n_valid / n_total * 100 if n_total > 0 else 0

    print(f"\nSegments tested: {n_total}")
    print(f"Valid HMM: {n_valid}/{n_total} ({success_rate:.0f}%)")

    # Détails par segment
    print("\nDetails:")
    for r in results:
        status = "[OK]" if r['is_valid'] else "[X]"
        print(f"  Seg {r['segment_id']:2d}: {status} n_active={r['n_active_states']}")

    # Critères de succès
    print("\n" + "-" * 70)
    if success_rate >= 80:
        print("[OK] SUCCESS: >= 80% segments have valid HMM")
        return 0
    else:
        print("[X] FAILURE: < 80% segments have valid HMM")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

#!/usr/bin/env python3
"""
regenerate_data.py - Régénère data/processed_data.parquet avec le nouveau pipeline

Utilise DataManager avec DataProcessor unifié pour régénérer les données
avec les correctifs de normalisation (MIN_IQR=1e-3, HMM_Trend en Z-Score).
"""

import os
import sys

# Add project root to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.data_engineering.manager import DataManager

if __name__ == "__main__":
    print("=" * 70)
    print("RÉGÉNÉRATION DES DONNÉES AVEC NOUVEAU PIPELINE")
    print("=" * 70)
    print("\nLe nouveau pipeline utilise:")
    print("  - DataProcessor unifié (MIN_IQR=1e-2)")
    print("  - HMM_Trend en Z-Score glissant")
    print("  - Patch de sécurité pour éviter l'explosion des MACD")
    print()
    
    # Créer DataManager
    manager = DataManager()
    
    # Exécuter le pipeline complet
    # Note: train_end_idx=None pour legacy mode (avertissement mais OK pour dataset global)
    # Pour un vrai leak-free, il faudrait spécifier train_end_idx
    df = manager.pipeline(
        save_path="data/processed_data.parquet",
        scaler_path="data/scaler.pkl",
        use_cached_data=True,
        train_end_idx=None  # Legacy mode pour dataset global (OK pour audit)
    )
    
    print("\n" + "=" * 70)
    print("RÉGÉNÉRATION TERMINÉE")
    print("=" * 70)
    print(f"\nDonnées sauvegardées:")
    print(f"  - data/processed_data.parquet ({len(df)} lignes, {len(df.columns)} colonnes)")
    print(f"  - data/scaler.pkl")
    print(f"\nPlage de dates: {df.index[0]} à {df.index[-1]}")
    print(f"\nAperçu des colonnes:")
    print(df.columns.tolist()[:10], "...")

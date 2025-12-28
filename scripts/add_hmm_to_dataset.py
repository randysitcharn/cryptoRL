#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
add_hmm_to_dataset.py - Ajoute les colonnes HMM aux datasets existants.

Ce script:
1. Charge le dataset de training
2. Entraîne le HMM (fit_predict) et sauvegarde le modèle
3. Charge le dataset de test
4. Applique le HMM pré-entraîné (predict) pour éviter data leakage
5. Sauvegarde les deux datasets enrichis

Usage:
    python scripts/add_hmm_to_dataset.py
"""

import os
import sys

# Ajouter le chemin racine pour les imports
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import pandas as pd
from src.data_engineering.manager import RegimeDetector


def main():
    print("=" * 60)
    print("ADD HMM TO DATASET")
    print("=" * 60)

    # Paths
    train_path = os.path.join(ROOT_DIR, "data/processed_data.parquet")
    test_path = os.path.join(ROOT_DIR, "data/processed_data_test.parquet")
    hmm_path = os.path.join(ROOT_DIR, "data/hmm_model.pkl")

    # =========================================================================
    # 1. Training Set: fit_predict
    # =========================================================================
    print("\n[1/4] Loading training dataset...")
    df_train = pd.read_parquet(train_path)
    print(f"  Shape: {df_train.shape}")
    print(f"  Columns: {len(df_train.columns)}")

    # Supprimer les anciennes colonnes HMM si présentes
    old_hmm_cols = [c for c in df_train.columns if c.startswith('Prob_') or c.startswith('HMM_')]
    if old_hmm_cols:
        print(f"  Removing old HMM columns: {old_hmm_cols}")
        df_train = df_train.drop(columns=old_hmm_cols)
        print(f"  New shape: {df_train.shape}")

    print("\n[2/4] Training HMM on training set (4 states)...")
    detector = RegimeDetector(n_components=4, n_mix=2, n_iter=200)
    df_train = detector.fit_predict(df_train)

    # Sauvegarder le HMM
    detector.save(hmm_path)

    # Drop NaN rows created by HMM features (168h window)
    initial_rows = len(df_train)
    df_train = df_train.dropna()
    dropped = initial_rows - len(df_train)
    print(f"  Dropped {dropped} rows with NaN ({len(df_train)} remaining)")

    # Sauvegarder le training set enrichi
    df_train.to_parquet(train_path)
    print(f"  Saved: {train_path}")
    print(f"  New shape: {df_train.shape}")

    # =========================================================================
    # 2. Test Set: predict (no refit)
    # =========================================================================
    print("\n[3/4] Loading test dataset...")
    if not os.path.exists(test_path):
        print(f"  WARNING: Test set not found at {test_path}, skipping")
    else:
        df_test = pd.read_parquet(test_path)
        print(f"  Shape: {df_test.shape}")

        # Supprimer les anciennes colonnes HMM si présentes
        old_hmm_cols = [c for c in df_test.columns if c.startswith('Prob_') or c.startswith('HMM_')]
        if old_hmm_cols:
            print(f"  Removing old HMM columns: {old_hmm_cols}")
            df_test = df_test.drop(columns=old_hmm_cols)
            print(f"  New shape: {df_test.shape}")

        print("\n[4/4] Applying HMM to test set (predict only, no refit)...")
        df_test = detector.predict(df_test)

        # Drop NaN rows
        initial_rows = len(df_test)
        df_test = df_test.dropna()
        dropped = initial_rows - len(df_test)
        print(f"  Dropped {dropped} rows with NaN ({len(df_test)} remaining)")

        # Sauvegarder le test set enrichi
        df_test.to_parquet(test_path)
        print(f"  Saved: {test_path}")
        print(f"  New shape: {df_test.shape}")

    # =========================================================================
    # Validation
    # =========================================================================
    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)

    # Recharger et vérifier
    df_check = pd.read_parquet(train_path)
    prob_cols = ['Prob_0', 'Prob_1', 'Prob_2', 'Prob_3']
    regime_labels = ['Crash', 'Downtrend', 'Range', 'Uptrend']

    print("\nTraining set:")
    print(f"  Shape: {df_check.shape}")
    print(f"  Prob columns present: {all(c in df_check.columns for c in prob_cols)}")
    for i, col in enumerate(prob_cols):
        if col in df_check.columns:
            print(f"    {col} ({regime_labels[i]}): min={df_check[col].min():.4f}, max={df_check[col].max():.4f}")

    if os.path.exists(test_path):
        df_check_test = pd.read_parquet(test_path)
        print("\nTest set:")
        print(f"  Shape: {df_check_test.shape}")
        print(f"  Prob columns present: {all(c in df_check_test.columns for c in prob_cols)}")

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()

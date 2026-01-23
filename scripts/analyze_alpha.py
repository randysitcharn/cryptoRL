# -*- coding: utf-8 -*-
"""
analyze_alpha.py - Feature Alpha Potential Audit.

Analyzes the predictive power of our features using a simple XGBoost probe model.
The Oracle Test proved the architecture works - now we need to know if our features
have any signal.

Usage:
    python scripts/analyze_alpha.py
    python scripts/analyze_alpha.py --data data/processed_data.parquet
"""

import argparse
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    from sklearn.ensemble import RandomForestClassifier


def load_data(data_path: str) -> pd.DataFrame:
    """Load and prepare data."""
    print(f"\n[1/5] Loading data from: {data_path}")

    if data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)

    print(f"  Shape: {df.shape}")
    print(f"  Columns: {len(df.columns)}")

    return df


def prepare_features_and_target(
    df: pd.DataFrame,
    target_col: str = "BTC_Close",
    lookahead: int = 1
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Prepare feature matrix X and target y.

    Target = Sign(LogReturn_t+1): +1 if price goes up, 0 otherwise.
    """
    print(f"\n[2/5] Preparing features and target...")

    # Calculate future log return
    prices = df[target_col].values
    log_returns = np.log(prices[lookahead:] / prices[:-lookahead])

    # Target: 1 if return > 0, else 0
    target = (log_returns > 0).astype(int)

    # Align features (drop last 'lookahead' rows)
    df_aligned = df.iloc[:-lookahead].copy()

    # Select feature columns (exclude OHLCV price columns)
    exclude_patterns = ['_Open', '_High', '_Low', '_Close', '_Volume']
    feature_cols = [
        col for col in df_aligned.columns
        if not any(pattern in col for pattern in exclude_patterns)
    ]

    # Also exclude any target leakage (future returns)
    feature_cols = [col for col in feature_cols if 'target' not in col.lower()]

    # If no features found, use all numeric columns except prices
    if len(feature_cols) == 0:
        print("  [WARNING] No engineered features found, using raw data")
        feature_cols = df_aligned.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in feature_cols if not any(p in col for p in exclude_patterns)]

    X = df_aligned[feature_cols].copy()
    y = pd.Series(target, index=df_aligned.index)

    # Handle NaN/Inf
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    print(f"  Features: {len(feature_cols)}")
    print(f"  Samples: {len(y)}")
    print(f"  Target distribution: UP={y.sum()} ({y.mean()*100:.1f}%), DOWN={len(y)-y.sum()} ({(1-y.mean())*100:.1f}%)")

    return X, y, feature_cols


def train_probe_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Tuple[object, float, np.ndarray]:
    """Train a quick probe model to measure feature predictive power."""
    print(f"\n[3/5] Training probe model...")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if HAS_XGBOOST:
        print("  Using XGBoost Classifier")
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
            use_label_encoder=False,
            eval_metric='logloss'
        )
    else:
        print("  Using RandomForest (XGBoost not installed)")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
            n_jobs=-1
        )

    model.fit(X_train_scaled, y_train)

    # Predictions
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    # Feature importances
    if HAS_XGBOOST:
        importances = model.feature_importances_
    else:
        importances = model.feature_importances_

    print(f"  Training complete.")

    return model, accuracy, importances


def compute_baselines(y_test: pd.Series) -> dict:
    """Compute baseline accuracies."""
    print(f"\n[4/5] Computing baselines...")

    # Majority class baseline
    majority_class = y_test.mode()[0]
    majority_acc = (y_test == majority_class).mean()
    majority_label = "UP" if majority_class == 1 else "DOWN"

    # Random baseline (50%)
    random_acc = 0.5

    baselines = {
        'random': random_acc,
        'majority': majority_acc,
        'majority_label': majority_label
    }

    print(f"  Random baseline: {random_acc*100:.1f}%")
    print(f"  Majority baseline ({majority_label}): {majority_acc*100:.1f}%")

    return baselines


def report_results(
    accuracy: float,
    baselines: dict,
    importances: np.ndarray,
    feature_cols: List[str],
    top_n: int = 10
):
    """Print final report."""
    print("\n" + "=" * 70)
    print("ALPHA POTENTIAL AUDIT - RESULTS")
    print("=" * 70)

    # Accuracy comparison
    gain_vs_random = (accuracy - baselines['random']) * 100
    gain_vs_majority = (accuracy - baselines['majority']) * 100

    print(f"\n>>> BASELINE Accuracy: {baselines['majority']*100:.1f}% (Always bet {baselines['majority_label']})")
    print(f">>> MODEL Accuracy:    {accuracy*100:.1f}% (Gain: {gain_vs_majority:+.1f}%)")

    # Interpretation
    print("\n" + "-" * 70)
    if gain_vs_majority > 2.0:
        print("[GOOD] Model beats baseline by >2% - Features have SOME predictive power")
    elif gain_vs_majority > 0.5:
        print("[WEAK] Model barely beats baseline - Features have MINIMAL signal")
    else:
        print("[BAD] Model doesn't beat baseline - Features are PURE NOISE")
    print("-" * 70)

    # Feature importance ranking
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances
    }).sort_values('importance', ascending=False)

    print(f"\n>>> TOP {top_n} FEATURES (Most Predictive):")
    for i, row in importance_df.head(top_n).iterrows():
        idx = importance_df.index.get_loc(i) + 1
        print(f"    {idx:2d}. {row['feature']:40s} ({row['importance']:.4f})")

    print(f"\n>>> BOTTOM {top_n} FEATURES (Useless - Importance ~0):")
    for i, row in importance_df.tail(top_n).iterrows():
        idx = len(importance_df) - importance_df.index.get_loc(i)
        print(f"    {idx:2d}. {row['feature']:40s} ({row['importance']:.4f})")

    # Summary stats
    print(f"\n>>> FEATURE IMPORTANCE DISTRIBUTION:")
    print(f"    Mean:   {importances.mean():.4f}")
    print(f"    Std:    {importances.std():.4f}")
    print(f"    Max:    {importances.max():.4f}")
    print(f"    Min:    {importances.min():.4f}")
    print(f"    Zero:   {(importances < 0.001).sum()} features with importance < 0.001")

    print("\n" + "=" * 70)

    return importance_df


def main():
    parser = argparse.ArgumentParser(description="Feature Alpha Potential Audit")
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed_data.parquet",
        help="Path to data file"
    )
    parser.add_argument(
        "--target",
        type=str,
        default="BTC_Close",
        help="Target price column"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set ratio"
    )
    parser.add_argument(
        "--lookahead",
        type=int,
        default=1,
        help="Prediction horizon (steps ahead)"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("ALPHA POTENTIAL AUDIT")
    print("=" * 70)
    print(f"Data: {args.data}")
    print(f"Target: {args.target}")
    print(f"Lookahead: {args.lookahead} step(s)")
    print(f"Test size: {args.test_size*100:.0f}%")

    # Load data
    df = load_data(args.data)

    # Prepare features and target
    X, y, feature_cols = prepare_features_and_target(
        df,
        target_col=args.target,
        lookahead=args.lookahead
    )

    # Train/test split (chronological - no shuffle for time series!)
    split_idx = int(len(X) * (1 - args.test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    # Compute baselines
    baselines = compute_baselines(y_test)

    # Train probe model
    model, accuracy, importances = train_probe_model(
        X_train, y_train, X_test, y_test
    )

    # Report results
    print(f"\n[5/5] Generating report...")
    importance_df = report_results(
        accuracy, baselines, importances, feature_cols
    )

    # Save importance to CSV
    output_path = "logs/feature_importance.csv"
    Path("logs").mkdir(exist_ok=True)
    importance_df.to_csv(output_path, index=False)
    print(f"\nFeature importance saved to: {output_path}")

    return accuracy, baselines, importance_df


if __name__ == "__main__":
    main()

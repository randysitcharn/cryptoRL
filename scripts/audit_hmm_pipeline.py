# -*- coding: utf-8 -*-
"""
audit_hmm_pipeline.py - HMM and Pipeline Signal Analysis (Unified)

Combines all signal analysis:
1. HMM State Analysis: Analyzes what the HMM hidden states capture
2. Pipeline Signal Loss Audit: Tests each stage with XGBoost
3. Supervised MAE Test: Compares MAE with auxiliary prediction head

Usage:
    # Run all analyses on real data
    python scripts/audit_hmm_pipeline.py

    # HMM audit only
    python scripts/audit_hmm_pipeline.py --mode hmm

    # Pipeline signal audit only
    python scripts/audit_hmm_pipeline.py --mode pipeline

    # Quick test with synthetic data
    python scripts/audit_hmm_pipeline.py --mode synthetic --signal 0.3
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
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from typing import Tuple, Dict, Optional, List
from collections import Counter

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    from sklearn.ensemble import RandomForestClassifier

# Project imports
from src.data_engineering.features import FeatureEngineer
from src.data_engineering.manager import RegimeDetector, DataManager
from src.models.foundation import CryptoMAE


# ============================================================================
# SHARED UTILITIES
# ============================================================================

def train_xgboost(X_train, y_train, X_test, y_test) -> Dict:
    """Train XGBoost/RandomForest and return metrics."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if HAS_XGBOOST:
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
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
            n_jobs=-1
        )

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred, average='weighted'),
        'model': model
    }


def _has_required_features(df: pd.DataFrame) -> bool:
    """Vérifie si toutes les features de momentum requises sont présentes."""
    required_raw = ['BTC_RSI_14', 'BTC_MACD_Hist', 'BTC_ADX_14']
    required_hmm = ['HMM_RSI_14', 'HMM_MACD_Hist', 'HMM_ADX_14']

    has_raw = all(col in df.columns for col in required_raw)
    has_hmm = all(col in df.columns for col in required_hmm)

    return has_raw and has_hmm


def load_or_create_data(
    segment_id: int = 0,
    force_retrain: bool = False
) -> pd.DataFrame:
    """Load data, retraining if necessary."""
    # Try segment first
    segment_path = f"data/wfo/segment_{segment_id}/train.parquet"
    if os.path.exists(segment_path) and not force_retrain:
        print(f"  Checking segment: {segment_path}")
        df = pd.read_parquet(segment_path)
        if _has_required_features(df):
            print(f"  [OK] Segment has all required features")
            return df

    # Try processed_data.parquet
    processed_path = "data/processed_data.parquet"
    if os.path.exists(processed_path) and not force_retrain:
        print(f"  Checking: {processed_path}")
        df = pd.read_parquet(processed_path)
        if _has_required_features(df):
            print(f"  [OK] Loaded parquet with all features")
            return df
        print(f"  [WARNING] Missing features, retraining...")

    # Retrain
    print(f"\n  [RETRAINING] Processing data...")
    manager = DataManager()
    df = manager.pipeline(
        save_path=processed_path,
        scaler_path="data/scaler.pkl",
        use_cached_data=True
    )

    if _has_required_features(df):
        print(f"  [OK] Data processed with all features")
    else:
        print(f"  [ERROR] Features still missing after retraining!")

    return df


# ============================================================================
# HMM STATE ANALYSIS
# ============================================================================

def get_dominant_state(df: pd.DataFrame) -> np.ndarray:
    """Get dominant HMM state from probabilities."""
    prob_cols = [c for c in df.columns if c.startswith('Prob_')]
    if len(prob_cols) == 0:
        raise ValueError("No Prob_* columns found")
    probs = df[prob_cols].values
    return np.argmax(probs, axis=1)


def compute_state_durations(states: np.ndarray, target_state: int) -> List[int]:
    """Compute durations of consecutive periods in target state."""
    durations = []
    current_duration = 0
    for state in states:
        if state == target_state:
            current_duration += 1
        else:
            if current_duration > 0:
                durations.append(current_duration)
            current_duration = 0
    if current_duration > 0:
        durations.append(current_duration)
    return durations


def compute_state_statistics(
    df: pd.DataFrame,
    states: np.ndarray,
    price_col: str = 'BTC_Close'
) -> pd.DataFrame:
    """Compute statistics for each HMM state."""
    prices = df[price_col].values
    returns = np.zeros(len(prices))
    returns[:-1] = np.log(prices[1:] / prices[:-1])

    future_returns = np.zeros(len(prices))
    future_returns[:-1] = returns[1:]

    unique_states = sorted(np.unique(states))
    results = []

    for state in unique_states:
        mask = states == state
        count = mask.sum()
        if count == 0:
            continue

        valid_mask = mask.copy()
        valid_mask[-1] = False
        win_rate = (future_returns[valid_mask] > 0).mean() if valid_mask.sum() > 0 else 0.5

        state_returns = returns[mask]
        avg_abs_return = np.abs(state_returns).mean()
        mean_return = state_returns.mean()

        durations = compute_state_durations(states, state)
        avg_duration = np.mean(durations) if len(durations) > 0 else 0

        results.append({
            'State': state,
            'Count': count,
            'Pct': count / len(states) * 100,
            'Win_Rate': win_rate * 100,
            'Avg_Abs_Ret': avg_abs_return * 100,
            'Mean_Ret': mean_return * 100,
            'Avg_Duration': avg_duration,
        })

    return pd.DataFrame(results)


def auto_describe_state(row: pd.Series) -> str:
    """Auto-generate state description."""
    vol = row['Avg_Abs_Ret']
    win_rate = row['Win_Rate']
    mean_ret = row['Mean_Ret']

    vol_desc = "Low Vol" if vol < 0.05 else ("Med Vol" if vol < 0.15 else "High Vol")

    if abs(win_rate - 50) < 2:
        dir_desc = "Neutral"
    elif win_rate > 52:
        dir_desc = "Bullish" if mean_ret > 0 else "Contrarian"
    else:
        dir_desc = "Bearish" if mean_ret < 0 else "Contrarian"

    return f"{vol_desc} / {dir_desc}"


def run_hmm_audit(segment_id: int = 0):
    """Run the full HMM audit."""
    print("=" * 70)
    print("HMM INTERNAL AUDIT")
    print("=" * 70)
    print(f"Segment: {segment_id}")

    # Analyze features
    print("\n" + "=" * 70)
    print("HMM FEATURE ANALYSIS")
    print("=" * 70)

    features = RegimeDetector.HMM_FEATURES
    print(f"\nFeatures used by HMM ({len(features)}):")
    for i, feat in enumerate(features, 1):
        if 'Trend' in feat:
            desc = "Rolling mean of log-returns (directional)"
        elif 'Vol' in feat and 'Ratio' not in feat:
            desc = "Parkinson volatility (log-transformed)"
        elif 'RSI_14' in feat:
            desc = "RSI-14 (directional)"
        elif 'MACD_Hist' in feat:
            desc = "MACD Histogram (directional)"
        elif 'ADX_14' in feat:
            desc = "ADX-14 (trend strength)"
        else:
            desc = "Unknown"
        print(f"  {i}. {feat}: {desc}")

    # Load data
    print("\n" + "=" * 70)
    print("DATA LOADING")
    print("=" * 70)
    df = load_or_create_data(segment_id)
    print(f"  Shape: {df.shape}")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")

    # Get states
    print("\n" + "=" * 70)
    print("STATE ANALYSIS")
    print("=" * 70)

    try:
        states = get_dominant_state(df)
        print(f"  Total samples: {len(states)}")
        print(f"  Unique states: {len(np.unique(states))}")
    except ValueError as e:
        print(f"  [ERROR] {e}")
        return

    # Statistics
    stats = compute_state_statistics(df, states)
    stats['Description'] = stats.apply(auto_describe_state, axis=1)

    print("\n>>> STATE STATISTICS:")
    print("-" * 90)
    print(f"{'State':^6} | {'Count':^7} | {'Pct':^6} | {'Win Rate':^10} | {'Avg |Ret|':^10} | {'Mean Ret':^10} | {'Avg Dur':^8} | {'Description':^18}")
    print("-" * 90)

    for _, row in stats.iterrows():
        print(f"{int(row['State']):^6} | {int(row['Count']):^7} | {row['Pct']:^5.1f}% | {row['Win_Rate']:^9.1f}% | {row['Avg_Abs_Ret']:^9.3f}% | {row['Mean_Ret']:^+9.4f}% | {row['Avg_Duration']:^7.1f}h | {row['Description']:^18}")

    print("-" * 90)

    # Directional analysis
    max_win_deviation = max(abs(row['Win_Rate'] - 50) for _, row in stats.iterrows())

    print("\n>>> DIRECTIONAL ANALYSIS:")
    if max_win_deviation < 2:
        print("  [!] ALERT: ALL states have Win Rate ~ 50% -> DIRECTIONALLY BLIND")
    elif max_win_deviation < 5:
        print(f"  [~] WEAK: Max deviation = {max_win_deviation:.1f}%")
    else:
        print(f"  [OK] GOOD: Max deviation = {max_win_deviation:.1f}%")

    # Volatility analysis
    vol_range = stats['Avg_Abs_Ret'].max() / (stats['Avg_Abs_Ret'].min() + 1e-10)
    print("\n>>> VOLATILITY ANALYSIS:")
    if vol_range > 3:
        print(f"  [OK] GOOD: Volatility range = {vol_range:.1f}x")
    else:
        print(f"  [~] WEAK: Volatility range = {vol_range:.1f}x")

    print("=" * 70)
    return stats


# ============================================================================
# PIPELINE SIGNAL AUDIT
# ============================================================================

def prepare_raw_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    """Extract raw technical features (Test A)."""
    exclude_patterns = ['_Open', '_High', '_Low', '_Close', '_Volume']
    hmm_patterns = ['Prob_', 'HMM_']

    feature_cols = []
    for col in df.columns:
        if any(p in col for p in exclude_patterns):
            continue
        if any(p in col for p in hmm_patterns):
            continue
        if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            feature_cols.append(col)

    X = df[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    momentum_features = ['_RSI_14', '_MACD_Hist', '_ADX_14']
    found_momentum = [col for col in feature_cols if any(mf in col for mf in momentum_features)]

    if found_momentum:
        print(f"      [OK] Momentum features: {len(found_momentum)}")
    else:
        print(f"      [WARNING] No momentum features found")

    return X, feature_cols


def prepare_hmm_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    """Extract HMM probability features (Test B)."""
    hmm_cols = [col for col in df.columns if col.startswith('Prob_')]
    if len(hmm_cols) == 0:
        raise ValueError("No HMM probability columns found")

    X = df[hmm_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.25)
    return X, hmm_cols


def create_target(df: pd.DataFrame, price_col: str = "BTC_Close") -> pd.Series:
    """Create target: Sign(LogReturn_t+1)."""
    prices = df[price_col].values
    log_returns = np.log(prices[1:] / prices[:-1])
    target = (log_returns > 0).astype(int)
    return pd.Series(target, index=df.index[:-1])


def run_pipeline_audit(
    segment_id: int = 0,
    skip_mae: bool = False,
    device: str = "cpu",
    force_retrain: bool = False
):
    """Run the pipeline signal audit."""
    print("=" * 70)
    print("PIPELINE SIGNAL LOSS AUDIT")
    print("=" * 70)
    print(f"Segment: {segment_id}, Device: {device}")

    # Load data
    print(f"\n[1/5] Loading data...")
    df = load_or_create_data(segment_id=segment_id, force_retrain=force_retrain)
    print(f"  Shape: {df.shape}")

    # Analyze momentum features
    print(f"\n[2/5] Analyzing momentum features...")
    for indicator in ['RSI', 'MACD', 'ADX']:
        cols = [c for c in df.columns if f'_{indicator}' in c or f'_{indicator}_' in c]
        if cols:
            col = cols[0]
            series = df[col].dropna()
            print(f"  {indicator}: mean={series.mean():.2f}, std={series.std():.2f}, range=[{series.min():.2f}, {series.max():.2f}]")

    # Create target
    print(f"\n[3/5] Creating target...")
    target = create_target(df)
    df_aligned = df.iloc[:-1].copy()
    print(f"  Target: UP={target.sum()} ({target.mean()*100:.1f}%)")

    # Prepare features
    print(f"\n[4/5] Preparing feature sets...")

    print("  [A] Raw technical features...")
    X_raw, raw_cols = prepare_raw_features(df_aligned)
    print(f"      {len(raw_cols)} features")

    print("  [B] HMM probabilities...")
    try:
        X_hmm, hmm_cols = prepare_hmm_features(df_aligned)
        print(f"      {len(hmm_cols)} features")
        has_hmm = True
    except ValueError as e:
        print(f"      [SKIP] {e}")
        has_hmm = False

    # Train/test split
    print(f"\n[5/5] Running XGBoost tests...")
    split_ratio = 0.8
    split_idx = int(len(target) * split_ratio)

    # Test A
    print("  Training Test A (Raw Features)...")
    X_train_a, X_test_a = X_raw.iloc[:split_idx], X_raw.iloc[split_idx:]
    y_train_a, y_test_a = target.iloc[:split_idx], target.iloc[split_idx:]
    results_a = train_xgboost(X_train_a, y_train_a, X_test_a, y_test_a)

    # Test B
    results_b = None
    if has_hmm:
        print("  Training Test B (HMM Probs)...")
        X_train_b, X_test_b = X_hmm.iloc[:split_idx], X_hmm.iloc[split_idx:]
        results_b = train_xgboost(X_train_b, y_train_a, X_test_b, y_test_a)

    # Results
    baseline_acc = max(y_test_a.mean(), 1 - y_test_a.mean())

    print("\n" + "=" * 70)
    print("PIPELINE SIGNAL AUDIT RESULTS")
    print("=" * 70)

    print(f"\n>>> BASELINE (Majority): {baseline_acc*100:.1f}%")

    gain_a = (results_a['accuracy'] - baseline_acc) * 100
    emoji_a = "[OK]" if gain_a > 2.0 else ("[WEAK]" if gain_a > 0.5 else "[FAIL]")
    print(f"\n>>> TEST A (Raw Features): {results_a['accuracy']*100:.1f}% ({emoji_a} {gain_a:+.1f}%)")

    if results_b:
        gain_b = (results_b['accuracy'] - baseline_acc) * 100
        emoji_b = "[OK]" if gain_b > 2.0 else ("[WEAK]" if gain_b > 0.5 else "[FAIL]")
        print(f">>> TEST B (HMM Probs):    {results_b['accuracy']*100:.1f}% ({emoji_b} {gain_b:+.1f}%)")

    print("=" * 70)

    return {'baseline': baseline_acc, 'test_a': results_a, 'test_b': results_b}


# ============================================================================
# SUPERVISED MAE COMPARISON (SYNTHETIC DATA)
# ============================================================================

def create_synthetic_data(
    n_samples: int = 10000,
    n_features: int = 35,
    signal_strength: float = 0.3
) -> Tuple:
    """Create synthetic data with known signal."""
    np.random.seed(42)

    features = np.random.randn(n_samples, n_features).astype(np.float32)
    base_returns = np.random.randn(n_samples) * 0.02
    signal = np.sign(features[:-1, 0]) * signal_strength * 0.02
    returns = base_returns.copy()
    returns[1:] += signal
    close_prices = 100 * np.exp(np.cumsum(returns)).astype(np.float32)
    directions = (returns[1:] > 0).astype(np.float32)

    return features, close_prices, directions


def train_supervised_mae(
    train_loader: DataLoader,
    val_loader: DataLoader,
    input_dim: int,
    device: torch.device,
    epochs: int = 20,
    aux_weight: float = 0.5,
    mask_ratio: float = 0.15
) -> Tuple[CryptoMAE, dict]:
    """Train MAE with supervised auxiliary loss."""
    model = CryptoMAE(
        input_dim=input_dim,
        d_model=64,
        n_heads=4,
        n_layers=2
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    history = {'val_acc': []}

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            pred, target_recon, mask, pred_logits = model(x, mask_ratio=mask_ratio)
            recon_loss = F.mse_loss(pred[mask], target_recon)
            aux_loss = F.binary_cross_entropy_with_logits(pred_logits, y)
            loss = recon_loss + aux_weight * aux_loss

            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                _, _, _, pred_logits = model(x, mask_ratio=mask_ratio)
                pred_dir = (torch.sigmoid(pred_logits) > 0.5).float()
                correct += (pred_dir == y).sum().item()
                total += y.numel()

        val_acc = correct / total
        history['val_acc'].append(val_acc)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:2d}/{epochs} | Val Acc: {val_acc*100:.1f}%")

    return model, history


def run_synthetic_comparison(
    n_samples: int = 10000,
    signal_strength: float = 0.3,
    seq_len: int = 64,
    epochs: int = 20,
    device: str = "cpu"
):
    """Run MAE vs XGBoost comparison on synthetic data."""
    print("=" * 70)
    print("SUPERVISED MAE vs XGBOOST (Synthetic Data)")
    print("=" * 70)
    print(f"\nConfig: samples={n_samples}, signal={signal_strength}, epochs={epochs}")

    # Create data
    features, close_prices, directions = create_synthetic_data(
        n_samples=n_samples,
        n_features=35,
        signal_strength=signal_strength
    )
    print(f"  Data: {features.shape}, Class balance: {directions.mean()*100:.1f}% up")

    # XGBoost (flat features)
    X_flat = features[:-1]
    y_flat = directions
    split_idx = int(len(X_flat) * 0.8)

    X_train, X_test = X_flat[:split_idx], X_flat[split_idx:]
    y_train, y_test = y_flat[:split_idx], y_flat[split_idx:]

    print("\n[XGBoost]")
    xgb_results = train_xgboost(X_train, y_train, X_test, y_test)
    print(f"  Accuracy: {xgb_results['accuracy']*100:.1f}%")

    # MAE (sequences)
    sequences = []
    targets = []
    for i in range(len(features) - seq_len):
        sequences.append(features[i:i + seq_len])
        close_future = close_prices[i + seq_len]
        close_current = close_prices[i + seq_len - 1]
        targets.append(1.0 if close_future > close_current else 0.0)

    sequences = torch.tensor(np.array(sequences), dtype=torch.float32)
    targets = torch.tensor(np.array(targets), dtype=torch.float32).unsqueeze(1)

    split_seq = int(len(sequences) * 0.8)
    train_dataset = TensorDataset(sequences[:split_seq], targets[:split_seq])
    test_dataset = TensorDataset(sequences[split_seq:], targets[split_seq:])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    print(f"\n[Supervised MAE]")
    device_torch = torch.device(device)
    model, history = train_supervised_mae(
        train_loader=train_loader,
        val_loader=test_loader,
        input_dim=35,
        device=device_torch,
        epochs=epochs
    )

    # Evaluate
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device_torch), y.to(device_torch)
            _, _, _, pred_logits = model(x, mask_ratio=0.0)
            pred_dir = (torch.sigmoid(pred_logits) > 0.5).float()
            all_preds.extend(pred_dir.cpu().numpy().flatten())
            all_targets.extend(y.cpu().numpy().flatten())

    mae_acc = accuracy_score(all_targets, all_preds)
    print(f"  Final Accuracy: {mae_acc*100:.1f}%")

    # Comparison
    baseline_acc = max(y_test.mean(), 1 - y_test.mean())
    diff = (mae_acc - xgb_results['accuracy']) * 100

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Baseline:  {baseline_acc*100:.1f}%")
    print(f"  XGBoost:   {xgb_results['accuracy']*100:.1f}%")
    print(f"  MAE:       {mae_acc*100:.1f}%")
    print(f"  MAE vs XGB: {diff:+.1f}%")

    if diff > 2:
        print("  [OK] MAE beats XGBoost!")
    elif diff > -2:
        print("  [OK] MAE matches XGBoost")
    else:
        print("  [FAIL] XGBoost still better")

    print("=" * 70)
    return {'baseline': baseline_acc, 'xgboost': xgb_results['accuracy'], 'mae': mae_acc}


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="HMM and Pipeline Signal Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["hmm", "pipeline", "both", "synthetic"],
        default="both",
        help="Analysis mode: hmm, pipeline, both, or synthetic (MAE test)"
    )
    parser.add_argument("--segment", type=int, default=0, help="WFO segment ID")
    parser.add_argument("--skip-mae", action="store_true", help="Skip MAE in pipeline mode")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force-retrain", action="store_true", help="Force data retraining")
    parser.add_argument("--signal", type=float, default=0.3, help="Signal strength (synthetic mode)")
    parser.add_argument("--epochs", type=int, default=20, help="MAE epochs (synthetic mode)")
    args = parser.parse_args()

    if args.mode == "synthetic":
        run_synthetic_comparison(
            signal_strength=args.signal,
            epochs=args.epochs,
            device=args.device
        )
    elif args.mode in ["hmm", "both"]:
        run_hmm_audit(segment_id=args.segment)
        if args.mode == "both":
            print("\n")

    if args.mode in ["pipeline", "both"]:
        run_pipeline_audit(
            segment_id=args.segment,
            skip_mae=args.skip_mae,
            device=args.device,
            force_retrain=args.force_retrain,
        )


if __name__ == "__main__":
    main()

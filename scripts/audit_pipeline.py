# -*- coding: utf-8 -*-
"""
audit_pipeline.py - Pipeline Signal Analysis (Unified)

Combines all signal analysis:
1. HMM State Analysis: Analyzes what the HMM hidden states capture
2. Pipeline Signal Loss Audit: Tests each stage with XGBoost
3. Supervised MAE Test: Compares MAE with auxiliary prediction head
4. Normalization Audit: Analyzes feature normalization/clipping at each stage

Usage:
    # Run all analyses on real data
    python scripts/audit_pipeline.py

    # HMM audit only
    python scripts/audit_pipeline.py --mode hmm

    # Pipeline signal audit only
    python scripts/audit_pipeline.py --mode pipeline

    # Normalization audit only
    python scripts/audit_pipeline.py --check-normalization

    # Quick test with synthetic data
    python scripts/audit_pipeline.py --mode synthetic --signal 0.3
"""

import argparse
import sys
import os
import logging
import warnings
import json
import time
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn.utils as torch_utils
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from typing import Tuple, Dict, Optional, List
from collections import Counter
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, silhouette_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
try:
    from sklearn.manifold import TSNE
    HAS_TSNE = True
except ImportError:
    HAS_TSNE = False

# Additional imports for HMM audit
from scipy import stats as scipy_stats
from scipy.signal import correlate
try:
    from scipy.stats import entropy as scipy_entropy
    from sklearn.metrics import mutual_info_score
    HAS_MUTUAL_INFO = True
except ImportError:
    HAS_MUTUAL_INFO = False

# Financial metrics
from src.utils.metrics import calculate_sharpe_ratio, calculate_max_drawdown

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
from src.data.dataset import CryptoDataset

# TQC audit imports
try:
    from sb3_contrib import TQC
    HAS_TQC = True
except ImportError:
    HAS_TQC = False
    print("[WARNING] sb3_contrib not available. TQC audit will be disabled.")

from src.training.batch_env import BatchCryptoEnv

# FiLM audit imports
try:
    from src.models.rl_adapter import FoundationFeatureExtractor, HMM_CONTEXT_SIZE
    from src.models.layers import FiLMLayer
    from gymnasium import spaces
    HAS_FILM = True
except ImportError as e:
    HAS_FILM = False
    print(f"[WARNING] FiLM imports not available: {e}")

# Import normalization audit
# Add scripts directory to path for import
scripts_dir = os.path.dirname(os.path.abspath(__file__))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)
from audit_normalization import run_normalization_audit

# Configure logger for MAE training
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.propagate = False  # Prevent duplicate logs

# ============================================================================
# SHARED UTILITIES
# ============================================================================

def get_debug_log_path() -> str:
    """Get path to debug log file."""
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, "debug_audit.log")


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
    """
    Get dominant HMM state from probabilities.
    
    Priorité:
    1. HMM_Prob_* (Belief States Forward-Only)
    2. Prob_* (Forward-Backward, legacy)
    """
    # Chercher d'abord les belief states Forward-Only
    prob_cols = [c for c in df.columns if c.startswith('HMM_Prob_')]
    
    # Fallback sur Prob_* si HMM_Prob_* n'existent pas
    if len(prob_cols) == 0:
        prob_cols = [c for c in df.columns if c.startswith('Prob_')]
    
    if len(prob_cols) == 0:
        raise ValueError("No HMM probability columns found (neither HMM_Prob_* nor Prob_*)")
    
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
# HMM COMPREHENSIVE AUDIT FUNCTIONS
# ============================================================================

def analyze_hmm_training_dynamics(
    detector: RegimeDetector,
    df: pd.DataFrame,
    output_dir: Optional[str] = None
) -> Dict:
    """
    Analyze HMM training dynamics: EM convergence and quality metrics.
    
    Includes seed robustness test.
    
    Returns:
        Dict with training history, quality metrics, and robustness results
    """
    results = {}
    
    # Extract EM convergence history
    if detector.hmm is not None and hasattr(detector.hmm, 'monitor_'):
        history = detector.hmm.monitor_.history if hasattr(detector.hmm.monitor_, 'history') else []
        results['em_history'] = history
        results['n_iterations'] = len(history)
        results['converged'] = detector.hmm.monitor_.converged if hasattr(detector.hmm.monitor_, 'converged') else False
        results['final_log_likelihood'] = history[-1] if len(history) > 0 else None
        
        # Calculate deltas
        if len(history) > 1:
            deltas = [history[i] - history[i-1] for i in range(1, len(history))]
            results['deltas'] = deltas
            results['avg_delta'] = np.mean(deltas) if deltas else 0.0
        else:
            results['deltas'] = []
            results['avg_delta'] = 0.0
    else:
        results['em_history'] = []
        results['n_iterations'] = 0
        results['converged'] = False
        results['final_log_likelihood'] = None
    
    # Quality metrics
    if detector.hmm is not None:
        transmat = detector.hmm.transmat_
        
        # Transition matrix entropy
        transmat_entropy = -np.sum(transmat * np.log(transmat + 1e-10)) / detector.n_components
        results['transition_entropy'] = transmat_entropy
        
        # Diagonal average (persistence)
        diag_avg = np.diag(transmat).mean()
        results['diagonal_avg'] = diag_avg
        
        # K-Means inertia
        if detector.kmeans is not None:
            results['kmeans_inertia'] = detector.kmeans.inertia_
        else:
            results['kmeans_inertia'] = None
    else:
        results['transition_entropy'] = None
        results['diagonal_avg'] = None
        results['kmeans_inertia'] = None
    
    # State proportions
    try:
        states = get_dominant_state(df)
        unique, counts = np.unique(states, return_counts=True)
        state_proportions = {int(state): count / len(states) for state, count in zip(unique, counts)}
        results['state_proportions'] = state_proportions
        
        # Separation score (from state statistics)
        stats = compute_state_statistics(df, states)
        mean_returns = stats['Mean_Ret'].values
        separation_score = np.std(mean_returns) if len(mean_returns) > 1 else 0.0
        results['separation_score'] = separation_score
    except (ValueError, KeyError):
        results['state_proportions'] = {}
        results['separation_score'] = 0.0
    
    # Seed Robustness Test
    print("  Testing seed robustness (training 3 HMMs with different seeds)...")
    robustness_results = _test_seed_robustness(df, detector.n_components)
    results['seed_robustness'] = robustness_results
    
    # Generate plot
    if output_dir:
        _plot_hmm_training_dynamics(results, output_dir)
    
    return results


def _test_seed_robustness(
    df: pd.DataFrame,
    n_components: int = 4,
    n_tests: int = 3
) -> Dict:
    """
    Test HMM stability by training multiple HMMs with different seeds.
    
    Returns:
        Dict with overlap metrics and stability assessment
    """
    try:
        from hmmlearn.hmm import GMMHMM
    except ImportError:
        return {'overlap': 0.0, 'stable': False, 'error': 'hmmlearn not available'}
    
    # Extract HMM features
    detector_temp = RegimeDetector(n_components=n_components, n_iter=50)  # Faster for test
    df_with_features = detector_temp._compute_hmm_features(df)
    
    features_raw = df_with_features[RegimeDetector.HMM_FEATURES].values
    valid_mask = np.isfinite(features_raw).all(axis=1)
    features_valid = features_raw[valid_mask]
    
    if len(features_valid) < 100:
        return {'overlap': 0.0, 'stable': False, 'error': 'Not enough valid samples'}
    
    # Scale features
    from src.data_engineering.processor import DataProcessor
    hmm_processor = DataProcessor(config={'min_iqr': 1.0, 'clip_range': (-5, 5)})
    features_df = pd.DataFrame(features_valid, columns=RegimeDetector.HMM_FEATURES)
    hmm_processor.fit(features_df)
    features_scaled_df = hmm_processor.transform(features_df)
    features_scaled = features_scaled_df.values
    
    # Train multiple HMMs
    all_states = []
    seeds = [42, 123, 456]
    
    for seed in seeds[:n_tests]:
        try:
            detector_test = RegimeDetector(
                n_components=n_components,
                n_iter=50,
                random_state=seed
            )
            
            # K-Means warm start
            kmeans = KMeans(n_clusters=n_components, random_state=seed, n_init=10)
            kmeans.fit(features_scaled)
            detector_test.kmeans = kmeans
            
            # Fit HMM
            hmm = GMMHMM(
                n_components=n_components,
                n_mix=2,
                n_iter=50,
                random_state=seed,
                covariance_type='diag',
                init_params='stc',
                min_covar=1e-3,
            )
            hmm._init(features_scaled, None)
            
            # Inject K-Means centers
            kmeans_centers = kmeans.cluster_centers_
            np.random.seed(seed)
            for i in range(n_components):
                for j in range(2):  # n_mix=2
                    noise = np.random.normal(0, 0.1, size=kmeans_centers.shape[1])
                    hmm.means_[i, j, :] = kmeans_centers[i] + noise
            
            hmm.fit(features_scaled)
            detector_test.hmm = hmm
            detector_test._is_fitted = True
            
            # Store scaler for _sort_states_by_trend (needed for inverse_transform if used)
            detector_test.scaler = hmm_processor.get_scaler()
            
            # Apply transition penalty (if needed)
            detector_test._apply_transition_penalty()
            
            # IMPORTANT: Apply state sorting by Trend to ensure stable ordering
            detector_test._sort_states_by_trend()
            
            # Predict states (already in sorted order after _sort_states_by_trend)
            states = hmm.predict(features_scaled)
            all_states.append(states)
        except Exception as e:
            print(f"    [WARNING] Seed {seed} failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if len(all_states) < 2:
        return {'overlap': 0.0, 'stable': False, 'error': 'Failed to train multiple HMMs'}
    
    # Calculate pairwise overlaps (Jaccard similarity on state sequences)
    overlaps = []
    for i in range(len(all_states)):
        for j in range(i + 1, len(all_states)):
            states_i = all_states[i]
            states_j = all_states[j]
            
            # Jaccard similarity: intersection / union
            # But we need to account for label switching - use dominant state matching
            # Simple approach: count exact matches
            matches = (states_i == states_j).sum()
            overlap = matches / len(states_i) if len(states_i) > 0 else 0.0
            overlaps.append(overlap)
    
    avg_overlap = np.mean(overlaps) if overlaps else 0.0
    min_overlap = np.min(overlaps) if overlaps else 0.0
    
    return {
        'overlap': avg_overlap,
        'min_overlap': min_overlap,
        'overlaps': overlaps,
        'stable': avg_overlap >= 0.80,
        'n_tests': len(all_states)
    }


def _plot_hmm_training_dynamics(results: Dict, output_dir: str):
    """Plot HMM training dynamics: convergence and quality metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: EM Convergence
    ax = axes[0, 0]
    if results['em_history']:
        history = results['em_history']
        epochs = range(1, len(history) + 1)
        ax.plot(epochs, history, 'b-', linewidth=0.7, label='Log-Likelihood')
        ax.set_xlabel('EM Iteration')
        ax.set_ylabel('Log-Likelihood')
        ax.set_title('EM Convergence', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if results.get('converged'):
            ax.text(0.05, 0.95, 'CONVERGED', transform=ax.transAxes,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='green', alpha=0.3))
        else:
            ax.text(0.05, 0.95, 'NOT CONVERGED', transform=ax.transAxes,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
    else:
        ax.text(0.5, 0.5, 'No EM history available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('EM Convergence', fontweight='bold')
    
    # Plot 2: Delta Log-Likelihood
    ax = axes[0, 1]
    if results.get('deltas'):
        deltas = results['deltas']
        epochs = range(2, len(deltas) + 2)
        ax.plot(epochs, deltas, 'g-', linewidth=0.7)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('EM Iteration')
        ax.set_ylabel('Delta Log-Likelihood')
        ax.set_title('Convergence Rate', fontweight='bold')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No delta data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Convergence Rate', fontweight='bold')
    
    # Plot 3: Quality Metrics
    ax = axes[1, 0]
    metrics = []
    labels = []
    if results.get('separation_score') is not None:
        metrics.append(results['separation_score'])
        labels.append('Separation\nScore')
    if results.get('transition_entropy') is not None:
        metrics.append(results['transition_entropy'])
        labels.append('Transition\nEntropy')
    if results.get('diagonal_avg') is not None:
        metrics.append(results['diagonal_avg'])
        labels.append('Diagonal\nAvg')
    if results.get('kmeans_inertia') is not None:
        metrics.append(results['kmeans_inertia'] / 100)  # Normalize for display
        labels.append('K-Means\nInertia\n(x100)')
    
    if metrics:
        colors = ['blue', 'green', 'orange', 'red'][:len(metrics)]
        ax.bar(range(len(metrics)), metrics, color=colors, alpha=0.7)
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel('Value')
        ax.set_title('HMM Quality Metrics', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, 'No quality metrics available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('HMM Quality Metrics', fontweight='bold')
    
    # Plot 4: Seed Robustness
    ax = axes[1, 1]
    robustness = results.get('seed_robustness', {})
    if robustness and 'overlap' in robustness:
        overlap = robustness['overlap']
        stable = robustness.get('stable', False)
        
        ax.barh([0], [overlap], color='green' if stable else 'red', alpha=0.7)
        ax.axvline(x=0.80, color='orange', linestyle='--', linewidth=0.7, label='Threshold (80%)')
        ax.set_xlim(0, 1.0)
        ax.set_ylim(-0.5, 0.5)
        ax.set_xlabel('Overlap (Jaccard Similarity)')
        ax.set_title(f'Seed Robustness\n({"STABLE" if stable else "UNSTABLE"})', fontweight='bold')
        ax.text(overlap + 0.02, 0, f'{overlap:.1%}', va='center', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_yticks([])
    else:
        ax.text(0.5, 0.5, 'Seed robustness test failed', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Seed Robustness', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hmm_training_dynamics.svg'), format='svg', bbox_inches='tight')
    plt.close()


def analyze_regime_detection_quality(
    detector: RegimeDetector,
    df: pd.DataFrame,
    output_dir: Optional[str] = None
) -> Dict:
    """
    Analyze regime detection quality: time-series, transition matrix, durations, and financial fingerprint.
    
    Returns:
        Dict with all regime quality metrics
    """
    results = {}
    
    # Get states
    try:
        states = get_dominant_state(df)
        results['states'] = states
        results['n_states'] = len(np.unique(states))
    except ValueError as e:
        print(f"  [ERROR] Cannot get states: {e}")
        return results
    
    # State statistics (existing function)
    stats = compute_state_statistics(df, states)
    stats['Description'] = stats.apply(auto_describe_state, axis=1)
    results['state_statistics'] = stats
    
    # Transition matrix
    if detector.hmm is not None:
        transmat = detector.hmm.transmat_
        results['transition_matrix'] = transmat
        
        # Persistence analysis
        diag_values = np.diag(transmat)
        results['diagonal_values'] = diag_values
        results['sticky_states'] = np.where(diag_values > 0.8)[0].tolist()
        results['volatile_states'] = np.where(diag_values < 0.5)[0].tolist()
    else:
        results['transition_matrix'] = None
    
    # Regime durations
    durations_by_state = {}
    for state in np.unique(states):
        durations = compute_state_durations(states, int(state))
        durations_by_state[int(state)] = durations
    results['durations_by_state'] = durations_by_state
    
    # Financial Fingerprint
    financial_metrics = _calculate_financial_fingerprint(df, states)
    results['financial_fingerprint'] = financial_metrics
    
    # Generate plots
    if output_dir:
        _plot_hmm_regime_timeline(df, states, output_dir)
        if results['transition_matrix'] is not None:
            _plot_hmm_transition_matrix(results['transition_matrix'], output_dir)
        _plot_hmm_regime_durations(durations_by_state, output_dir)
        _plot_hmm_financial_fingerprint(financial_metrics, output_dir)
    
    return results


def _calculate_financial_fingerprint(
    df: pd.DataFrame,
    states: np.ndarray,
    price_col: str = 'BTC_Close'
) -> pd.DataFrame:
    """
    Calculate financial metrics for each HMM state.
    
    Returns:
        DataFrame with Annualized Return, Sharpe Ratio, Max Drawdown, Volatility per state
    """
    prices = df[price_col].values
    returns = np.zeros(len(prices))
    returns[1:] = np.log(prices[1:] / prices[:-1])
    
    # Calculate NAV series for drawdown
    navs = 10000 * np.exp(np.cumsum(returns))
    
    unique_states = sorted(np.unique(states))
    financial_data = []
    
    periods_per_year = 252 * 24  # Hourly data
    
    for state in unique_states:
        mask = states == state
        state_returns = returns[mask]
        
        if len(state_returns) < 2:
            continue
        
        # Annualized Return
        mean_return = np.mean(state_returns)
        annualized_return = mean_return * periods_per_year * 100  # Convert to percentage
        
        # Sharpe Ratio
        sharpe = calculate_sharpe_ratio(state_returns, periods_per_year=periods_per_year)
        
        # Max Drawdown
        state_navs = navs[mask]
        if len(state_navs) > 1:
            peak = np.maximum.accumulate(state_navs)
            drawdown = (peak - state_navs) / peak
            max_dd = drawdown.max() * 100
        else:
            max_dd = 0.0
        
        # Volatility (Parkinson if available, else std of returns)
        if 'BTC_Parkinson' in df.columns:
            state_vol = df.loc[df.index[mask], 'BTC_Parkinson'].mean()
        else:
            state_vol = np.std(state_returns) * np.sqrt(periods_per_year) * 100
        
        financial_data.append({
            'State': int(state),
            'Annualized_Return': annualized_return,
            'Sharpe_Ratio': sharpe,
            'Max_Drawdown': max_dd,
            'Volatility': state_vol,
            'Count': mask.sum()
        })
    
    return pd.DataFrame(financial_data)


def _plot_hmm_regime_timeline(
    df: pd.DataFrame,
    states: np.ndarray,
    output_dir: str,
    n_periods: int = 3
):
    """Plot time-series with HMM states overlaid."""
    prices = df['BTC_Close'].values if 'BTC_Close' in df.columns else df.iloc[:, 0].values
    
    # Select representative periods
    total_len = len(prices)
    period_size = total_len // n_periods
    
    fig, axes = plt.subplots(n_periods, 1, figsize=(16, 4 * n_periods))
    if n_periods == 1:
        axes = [axes]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(np.unique(states))))
    state_colors = {int(s): colors[i] for i, s in enumerate(sorted(np.unique(states)))}
    
    for period_idx in range(n_periods):
        start_idx = period_idx * period_size
        end_idx = min((period_idx + 1) * period_size, total_len)
        
        ax = axes[period_idx]
        period_prices = prices[start_idx:end_idx]
        period_states = states[start_idx:end_idx]
        period_dates = df.index[start_idx:end_idx]
        
        # Plot price
        ax.plot(range(len(period_prices)), period_prices, 'k-', linewidth=0.5, alpha=0.7, label='Price')
        
        # Color background by state
        current_state = period_states[0]
        start_seg = 0
        for i in range(1, len(period_states)):
            if period_states[i] != current_state:
                # Fill segment
                ax.axvspan(start_seg, i, alpha=0.2, color=state_colors[int(current_state)])
                current_state = period_states[i]
                start_seg = i
        # Last segment
        ax.axvspan(start_seg, len(period_states), alpha=0.2, color=state_colors[int(current_state)])
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Price')
        ax.set_title(f'Regime Timeline - Period {period_idx + 1}\n({period_dates[0]} to {period_dates[-1]})', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hmm_regime_timeline.svg'), format='svg', bbox_inches='tight')
    plt.close()


def _plot_hmm_transition_matrix(transmat: np.ndarray, output_dir: str):
    """Plot transition matrix as heatmap."""
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(transmat, annot=True, fmt='.3f', cmap='Blues', 
                square=True, linewidths=0.5, cbar_kws={'label': 'Transition Probability'})
    
    plt.xlabel('To State')
    plt.ylabel('From State')
    plt.title('HMM Transition Matrix', fontweight='bold', fontsize=14)
    
    # Highlight diagonal (persistence)
    for i in range(len(transmat)):
        plt.gca().add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='red', lw=2))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hmm_transition_matrix.svg'), format='svg', bbox_inches='tight')
    plt.close()


def _plot_hmm_regime_durations(durations_by_state: Dict, output_dir: str):
    """Plot histogram of regime durations by state."""
    n_states = len(durations_by_state)
    fig, axes = plt.subplots(1, n_states, figsize=(5 * n_states, 4))
    if n_states == 1:
        axes = [axes]
    
    for idx, (state, durations) in enumerate(sorted(durations_by_state.items())):
        ax = axes[idx]
        if len(durations) > 0:
            ax.hist(durations, bins=20, alpha=0.7, edgecolor='black')
            ax.axvline(np.mean(durations), color='red', linestyle='--', linewidth=0.7, label=f'Mean: {np.mean(durations):.1f}h')
            ax.set_xlabel('Duration (hours)')
            ax.set_ylabel('Frequency')
            ax.set_title(f'State {state} Durations', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'State {state} Durations', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hmm_regime_durations.svg'), format='svg', bbox_inches='tight')
    plt.close()


def _plot_hmm_financial_fingerprint(financial_df: pd.DataFrame, output_dir: str):
    """Plot financial fingerprint: Return, Sharpe, Drawdown, Volatility by state."""
    if len(financial_df) == 0:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    states = financial_df['State'].values
    
    # Plot 1: Annualized Return
    ax = axes[0, 0]
    ax.bar(states, financial_df['Annualized_Return'], color='green', alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('State')
    ax.set_ylabel('Annualized Return (%)')
    ax.set_title('Annualized Return by State', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Sharpe Ratio
    ax = axes[0, 1]
    colors_sharpe = ['green' if s > 0 else 'red' for s in financial_df['Sharpe_Ratio']]
    ax.bar(states, financial_df['Sharpe_Ratio'], color=colors_sharpe, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('State')
    ax.set_ylabel('Sharpe Ratio')
    ax.set_title('Sharpe Ratio by State', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Max Drawdown
    ax = axes[1, 0]
    ax.bar(states, financial_df['Max_Drawdown'], color='red', alpha=0.7)
    ax.set_xlabel('State')
    ax.set_ylabel('Max Drawdown (%)')
    ax.set_title('Max Drawdown by State', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Volatility
    ax = axes[1, 1]
    ax.bar(states, financial_df['Volatility'], color='orange', alpha=0.7)
    ax.set_xlabel('State')
    ax.set_ylabel('Volatility (%)')
    ax.set_title('Volatility by State', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hmm_financial_fingerprint.svg'), format='svg', bbox_inches='tight')
    plt.close()


def analyze_hmm_feature_space(
    detector: RegimeDetector,
    df: pd.DataFrame,
    output_dir: Optional[str] = None,
    use_tsne: bool = False
) -> Dict:
    """
    Analyze HMM feature space: PCA/t-SNE projection and feature attribution.
    
    Returns:
        Dict with embeddings, projections, separability, and feature attribution
    """
    results = {}
    
    # Extract HMM features
    df_with_features = detector._compute_hmm_features(df)
    features_raw = df_with_features[RegimeDetector.HMM_FEATURES].values
    valid_mask = np.isfinite(features_raw).all(axis=1)
    features_valid = features_raw[valid_mask]
    
    if len(features_valid) < 10:
        print("  [WARNING] Not enough valid features for analysis")
        return results
    
    # Get states aligned with valid features
    try:
        states = get_dominant_state(df_with_features)
        states_valid = states[valid_mask]
    except ValueError:
        states_valid = None
    
    # Scale features (same as HMM training)
    from src.data_engineering.processor import DataProcessor
    hmm_processor = DataProcessor(config={'min_iqr': 1.0, 'clip_range': (-5, 5)})
    features_df = pd.DataFrame(features_valid, columns=RegimeDetector.HMM_FEATURES)
    hmm_processor.fit(features_df)
    features_scaled_df = hmm_processor.transform(features_df)
    features_scaled = features_scaled_df.values
    
    results['features'] = features_scaled
    results['feature_names'] = RegimeDetector.HMM_FEATURES
    
    # PCA projection
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features_scaled)
    results['pca_projection'] = features_pca
    results['pca_explained_variance'] = pca.explained_variance_ratio_.sum()
    
    # t-SNE projection (optional)
    if use_tsne and HAS_TSNE:
        print("  Computing t-SNE (this may take a while)...")
        n_samples_tsne = min(1000, len(features_scaled))
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        features_tsne = tsne.fit_transform(features_scaled[:n_samples_tsne])
        results['tsne_projection'] = features_tsne
        results['tsne_states'] = states_valid[:n_samples_tsne] if states_valid is not None else None
    
    # Separability analysis
    if states_valid is not None and len(np.unique(states_valid)) > 1:
        silhouette = silhouette_score(features_scaled, states_valid)
        results['silhouette_score'] = silhouette
    else:
        results['silhouette_score'] = None
    
    # Feature Attribution (Radar Charts)
    feature_attribution = _calculate_feature_attribution(features_valid, states_valid, RegimeDetector.HMM_FEATURES)
    results['feature_attribution'] = feature_attribution
    
    # Generate plots
    if output_dir:
        _plot_hmm_feature_space_pca(features_pca, states_valid, output_dir)
        if use_tsne and HAS_TSNE and 'tsne_projection' in results:
            _plot_hmm_feature_space_tsne(results['tsne_projection'], results.get('tsne_states'), output_dir)
        _plot_hmm_feature_attribution(feature_attribution, RegimeDetector.HMM_FEATURES, output_dir)
    
    return results


def _calculate_feature_attribution(
    features: np.ndarray,
    states: Optional[np.ndarray],
    feature_names: List[str]
) -> Dict:
    """
    Calculate feature attribution: which features characterize each state.
    
    Returns:
        Dict with mean feature values per state and deviations from global mean
    """
    if states is None:
        return {}
    
    # Global mean and std for each feature
    global_means = np.mean(features, axis=0)
    global_stds = np.std(features, axis=0) + 1e-10
    
    attribution = {}
    unique_states = sorted(np.unique(states))
    
    for state in unique_states:
        mask = states == state
        state_features = features[mask]
        
        if len(state_features) == 0:
            continue
        
        # Mean feature values for this state
        state_means = np.mean(state_features, axis=0)
        
        # Z-scores (deviation from global mean in standard deviations)
        z_scores = (state_means - global_means) / global_stds
        
        # Identify significant deviations
        significant_features = []
        for i, (feat_name, z_score) in enumerate(zip(feature_names, z_scores)):
            if abs(z_score) > 2.0:  # Significant deviation
                significant_features.append({
                    'feature': feat_name,
                    'z_score': z_score,
                    'state_mean': state_means[i],
                    'global_mean': global_means[i]
                })
        
        attribution[int(state)] = {
            'mean_values': state_means.tolist(),
            'z_scores': z_scores.tolist(),
            'significant_features': significant_features
        }
    
    return attribution


def _plot_hmm_feature_space_pca(
    features_pca: np.ndarray,
    states: Optional[np.ndarray],
    output_dir: str
):
    """Plot PCA projection colored by HMM states."""
    fig, axes = plt.subplots(1, 2 if states is not None else 1, figsize=(14, 6))
    if states is None:
        axes = [axes]
    
    # Plot 1: Colored by state
    if states is not None:
        ax = axes[0]
        unique_states = sorted(np.unique(states))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_states)))
        
        for state, color in zip(unique_states, colors):
            mask = states == state
            ax.scatter(features_pca[mask, 0], features_pca[mask, 1],
                      c=[color], label=f'State {state}', alpha=0.6, s=20)
        
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title('HMM Feature Space - PCA (Colored by States)', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 2: Colored by direction (if we have price data)
    if len(axes) > 1:
        ax = axes[1]
        # Use first PC as proxy for direction
        median_pc1 = np.median(features_pca[:, 0])
        up_mask = features_pca[:, 0] > median_pc1
        ax.scatter(features_pca[up_mask, 0], features_pca[up_mask, 1],
                  c='green', label='UP', alpha=0.6, s=20)
        ax.scatter(features_pca[~up_mask, 0], features_pca[~up_mask, 1],
                  c='red', label='DOWN', alpha=0.6, s=20)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title('HMM Feature Space - PCA (Colored by Direction)', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hmm_feature_space_pca.svg'), format='svg', bbox_inches='tight')
    plt.close()


def _plot_hmm_feature_space_tsne(
    features_tsne: np.ndarray,
    states: Optional[np.ndarray],
    output_dir: str
):
    """Plot t-SNE projection colored by HMM states."""
    fig, axes = plt.subplots(1, 2 if states is not None else 1, figsize=(14, 6))
    if states is None:
        axes = [axes]
    
    # Plot 1: Colored by state
    if states is not None:
        ax = axes[0]
        unique_states = sorted(np.unique(states))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_states)))
        
        for state, color in zip(unique_states, colors):
            mask = states == state
            ax.scatter(features_tsne[mask, 0], features_tsne[mask, 1],
                      c=[color], label=f'State {state}', alpha=0.6, s=20)
        
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.set_title('HMM Feature Space - t-SNE (Colored by States)', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 2: Colored by direction
    if len(axes) > 1:
        ax = axes[1]
        median_tsne1 = np.median(features_tsne[:, 0])
        up_mask = features_tsne[:, 0] > median_tsne1
        ax.scatter(features_tsne[up_mask, 0], features_tsne[up_mask, 1],
                  c='green', label='UP', alpha=0.6, s=20)
        ax.scatter(features_tsne[~up_mask, 0], features_tsne[~up_mask, 1],
                  c='red', label='DOWN', alpha=0.6, s=20)
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.set_title('HMM Feature Space - t-SNE (Colored by Direction)', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hmm_feature_space_tsne.svg'), format='svg', bbox_inches='tight')
    plt.close()


def _plot_hmm_feature_attribution(
    attribution: Dict,
    feature_names: List[str],
    output_dir: str
):
    """Plot radar charts showing feature attribution per state."""
    if not attribution:
        return
    
    n_states = len(attribution)
    fig, axes = plt.subplots(1, n_states, figsize=(6 * n_states, 6), subplot_kw=dict(projection='polar'))
    if n_states == 1:
        axes = [axes]
    
    # Number of features
    n_features = len(feature_names)
    angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    for idx, (state, state_data) in enumerate(sorted(attribution.items())):
        ax = axes[idx]
        
        # Normalize feature values to [0, 1] for radar chart
        mean_values = np.array(state_data['mean_values'])
        # Normalize to [0, 1] range
        min_val, max_val = mean_values.min(), mean_values.max()
        if max_val > min_val:
            normalized = (mean_values - min_val) / (max_val - min_val)
        else:
            normalized = np.ones_like(mean_values) * 0.5
        
        values = normalized.tolist()
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=0.7, label=f'State {state}')
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(feature_names, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_title(f'State {state} Feature Profile', fontweight='bold', pad=20)
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hmm_feature_attribution.svg'), format='svg', bbox_inches='tight')
    plt.close()


def analyze_hmm_predictive_power(
    df: pd.DataFrame,
    output_dir: Optional[str] = None
) -> Dict:
    """
    Analyze HMM predictive power: correlations, mutual information, calibration, and lag analysis.
    
    Returns:
        Dict with all predictive power metrics
    """
    results = {}
    
    # Get probabilities (priorité: HMM_Prob_* puis Prob_*)
    prob_cols = [c for c in df.columns if c.startswith('HMM_Prob_')]
    if len(prob_cols) == 0:
        prob_cols = [c for c in df.columns if c.startswith('Prob_')]
    if len(prob_cols) == 0:
        print("  [ERROR] No Prob_* columns found")
        return results
    
    probs = df[prob_cols].values
    
    # Get prices and returns
    if 'BTC_Close' not in df.columns:
        print("  [ERROR] BTC_Close column not found")
        return results
    
    prices = df['BTC_Close'].values
    returns = np.zeros(len(prices))
    returns[1:] = np.log(prices[1:] / prices[:-1])
    
    # Future returns at different horizons
    future_returns_1h = np.zeros(len(returns))
    future_returns_1h[:-1] = returns[1:]
    
    future_returns_24h = np.zeros(len(returns))
    future_returns_24h[:-24] = np.log(prices[24:] / prices[:-24]) if len(prices) > 24 else 0
    
    future_returns_168h = np.zeros(len(returns))
    future_returns_168h[:-168] = np.log(prices[168:] / prices[:-168]) if len(prices) > 168 else 0
    
    # Correlations with future returns
    correlations = {}
    horizons = ['1h', '24h', '168h']
    future_returns_dict = {
        '1h': future_returns_1h,
        '24h': future_returns_24h,
        '168h': future_returns_168h
    }
    
    for prob_idx, prob_col in enumerate(prob_cols):
        prob_values = probs[:, prob_idx]
        state_correlations = {}
        
        for horizon in horizons:
            future_ret = future_returns_dict[horizon]
            # Align lengths
            min_len = min(len(prob_values), len(future_ret))
            valid_mask = np.isfinite(prob_values[:min_len]) & np.isfinite(future_ret[:min_len])
            
            if valid_mask.sum() > 10:
                corr = np.corrcoef(prob_values[:min_len][valid_mask], future_ret[:min_len][valid_mask])[0, 1]
                state_correlations[horizon] = corr if np.isfinite(corr) else 0.0
            else:
                state_correlations[horizon] = 0.0
        
        correlations[prob_col] = state_correlations
    
    results['correlations'] = correlations
    
    # Mutual Information
    if HAS_MUTUAL_INFO:
        mutual_info = {}
        for prob_idx, prob_col in enumerate(prob_cols):
            prob_values = probs[:, prob_idx]
            mi_values = {}
            
            for horizon in horizons:
                future_ret = future_returns_dict[horizon]
                min_len = min(len(prob_values), len(future_ret))
                valid_mask = np.isfinite(prob_values[:min_len]) & np.isfinite(future_ret[:min_len])
                
                if valid_mask.sum() > 10:
                    # Discretize for mutual information
                    try:
                        prob_discrete = pd.cut(prob_values[:min_len][valid_mask], bins=10, labels=False, duplicates='drop')
                        ret_discrete = pd.cut(future_ret[:min_len][valid_mask], bins=10, labels=False, duplicates='drop')
                        
                        # Remove NaN
                        valid_both = ~(pd.isna(prob_discrete) | pd.isna(ret_discrete))
                        if valid_both.sum() > 10:
                            mi = mutual_info_score(prob_discrete[valid_both], ret_discrete[valid_both])
                            mi_values[horizon] = mi
                        else:
                            mi_values[horizon] = 0.0
                    except Exception as e:
                        # Fallback if discretization fails
                        mi_values[horizon] = 0.0
                else:
                    mi_values[horizon] = 0.0
            
            mutual_info[prob_col] = mi_values
        
        results['mutual_information'] = mutual_info
    
    # Calibration Analysis (Reliability Diagram)
    calibration_results = _analyze_calibration(probs, future_returns_1h, prob_cols)
    results['calibration'] = calibration_results
    
    # Lag Analysis
    lag_results = _analyze_lag(probs, prices, prob_cols)
    results['lag_analysis'] = lag_results
    
    # Generate plots
    if output_dir:
        _plot_hmm_predictive_power(correlations, results.get('mutual_information'), output_dir)
        _plot_hmm_calibration(calibration_results, prob_cols, output_dir)
        _plot_hmm_lag_analysis(lag_results, prob_cols, output_dir)
    
    return results


def _analyze_calibration(
    probs: np.ndarray,
    future_returns: np.ndarray,
    prob_cols: List[str],
    n_bins: int = 10
) -> Dict:
    """Analyze calibration: reliability diagram."""
    calibration = {}
    
    # Create binary target: positive return
    target = (future_returns > 0).astype(float)
    
    for prob_idx, prob_col in enumerate(prob_cols):
        prob_values = probs[:, prob_idx]
        
        # Align lengths
        min_len = min(len(prob_values), len(target))
        prob_aligned = prob_values[:min_len]
        target_aligned = target[:min_len]
        
        # Remove NaN
        valid_mask = np.isfinite(prob_aligned) & np.isfinite(target_aligned)
        if valid_mask.sum() < 10:
            continue
        
        prob_valid = prob_aligned[valid_mask]
        target_valid = target_aligned[valid_mask]
        
        # Bin probabilities
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(prob_valid, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        # Calculate observed frequency per bin
        bin_centers = []
        observed_freq = []
        predicted_freq = []
        
        for i in range(n_bins):
            bin_mask = bin_indices == i
            if bin_mask.sum() > 0:
                bin_centers.append((bins[i] + bins[i+1]) / 2)
                observed_freq.append(target_valid[bin_mask].mean())
                predicted_freq.append(prob_valid[bin_mask].mean())
        
        calibration[prob_col] = {
            'bin_centers': bin_centers,
            'observed_freq': observed_freq,
            'predicted_freq': predicted_freq
        }
    
    return calibration


def _analyze_lag(
    probs: np.ndarray,
    prices: np.ndarray,
    prob_cols: List[str],
    max_lag: int = 24
) -> Dict:
    """
    Analyze lag: correlation between Prob_* and real metrics at different time lags.
    
    Tests lags from -max_lag (predictive) to +max_lag (reactive).
    """
    lag_results = {}
    
    # Calculate real volatility (Parkinson if available, else use returns)
    returns = np.zeros(len(prices))
    returns[1:] = np.log(prices[1:] / prices[:-1])
    volatility = np.abs(returns)  # Simple volatility proxy
    
    # Test different lags
    lags = list(range(-max_lag, max_lag + 1))
    
    for prob_idx, prob_col in enumerate(prob_cols):
        prob_values = probs[:, prob_idx]
        
        # Calculate correlation at each lag
        correlations = []
        for lag in lags:
            if lag < 0:
                # Predictive: compare prob[t] with metric[t+|lag|]
                prob_shifted = prob_values[:len(volatility) + lag]
                metric_shifted = volatility[-lag:] if lag < 0 else volatility
            elif lag > 0:
                # Reactive: compare prob[t] with metric[t-lag]
                prob_shifted = prob_values[lag:]
                metric_shifted = volatility[:len(prob_shifted)]
            else:
                # Coincident
                min_len = min(len(prob_values), len(volatility))
                prob_shifted = prob_values[:min_len]
                metric_shifted = volatility[:min_len]
            
            # Align lengths
            min_len = min(len(prob_shifted), len(metric_shifted))
            if min_len < 10:
                correlations.append(0.0)
                continue
            
            prob_aligned = prob_shifted[:min_len]
            metric_aligned = metric_shifted[:min_len]
            
            # Remove NaN
            valid_mask = np.isfinite(prob_aligned) & np.isfinite(metric_aligned)
            if valid_mask.sum() < 10:
                correlations.append(0.0)
                continue
            
            corr = np.corrcoef(prob_aligned[valid_mask], metric_aligned[valid_mask])[0, 1]
            correlations.append(corr if np.isfinite(corr) else 0.0)
        
        # Find optimal lag (maximum correlation)
        optimal_lag_idx = np.argmax(np.abs(correlations))
        optimal_lag = lags[optimal_lag_idx]
        
        lag_results[prob_col] = {
            'lags': lags,
            'correlations': correlations,
            'optimal_lag': optimal_lag,
            'optimal_correlation': correlations[optimal_lag_idx],
            'is_predictive': optimal_lag < 0,
            'lead_time': abs(optimal_lag) if optimal_lag < 0 else 0
        }
    
    return lag_results


def _plot_hmm_predictive_power(
    correlations: Dict,
    mutual_info: Optional[Dict],
    output_dir: str
):
    """Plot predictive power: correlations and mutual information."""
    n_plots = 2 if mutual_info else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 6))
    if n_plots == 1:
        axes = [axes]
    
    # Plot 1: Correlations
    ax = axes[0]
    prob_cols = list(correlations.keys())
    horizons = ['1h', '24h', '168h']
    x = np.arange(len(prob_cols))
    width = 0.25
    
    for i, horizon in enumerate(horizons):
        values = [correlations[col].get(horizon, 0.0) for col in prob_cols]
        ax.bar(x + i * width, values, width, label=f'{horizon} ahead', alpha=0.7)
    
    ax.set_xlabel('HMM State Probability')
    ax.set_ylabel('Correlation with Future Returns')
    ax.set_title('Predictive Power: Correlations', fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(prob_cols, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Plot 2: Mutual Information
    if mutual_info:
        ax = axes[1]
        for i, horizon in enumerate(horizons):
            values = [mutual_info.get(col, {}).get(horizon, 0.0) for col in prob_cols]
            ax.bar(x + i * width, values, width, label=f'{horizon} ahead', alpha=0.7)
        
        ax.set_xlabel('HMM State Probability')
        ax.set_ylabel('Mutual Information')
        ax.set_title('Predictive Power: Mutual Information', fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(prob_cols, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hmm_predictive_power.svg'), format='svg', bbox_inches='tight')
    plt.close()


def _plot_hmm_calibration(
    calibration: Dict,
    prob_cols: List[str],
    output_dir: str
):
    """Plot reliability diagram (calibration)."""
    n_states = len(calibration)
    fig, axes = plt.subplots(1, n_states, figsize=(6 * n_states, 6))
    if n_states == 1:
        axes = [axes]
    
    for idx, (prob_col, calib_data) in enumerate(calibration.items()):
        ax = axes[idx]
        
        bin_centers = calib_data['bin_centers']
        observed = calib_data['observed_freq']
        predicted = calib_data['predicted_freq']
        
        ax.plot(bin_centers, observed, 'o-', label='Observed', linewidth=0.7, markersize=8)
        ax.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration', linewidth=0.7, alpha=0.7)
        ax.plot(bin_centers, predicted, 's--', label='Predicted', linewidth=0.5, alpha=0.5)
        
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('Observed Frequency')
        ax.set_title(f'Calibration: {prob_col}', fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hmm_calibration.svg'), format='svg', bbox_inches='tight')
    plt.close()


def _plot_hmm_lag_analysis(
    lag_results: Dict,
    prob_cols: List[str],
    output_dir: str
):
    """Plot lag analysis: correlation vs lag for each state."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, prob_col in enumerate(prob_cols[:4]):  # Max 4 states
        if prob_col not in lag_results:
            continue
        
        ax = axes[idx]
        lag_data = lag_results[prob_col]
        lags = lag_data['lags']
        correlations = lag_data['correlations']
        optimal_lag = lag_data['optimal_lag']
        
        ax.plot(lags, correlations, 'b-', linewidth=0.7, marker='o', markersize=4)
        ax.axvline(x=optimal_lag, color='red', linestyle='--', linewidth=0.7, 
                  label=f'Optimal: {optimal_lag}h')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        
        # Color regions
        ax.axvspan(-24, 0, alpha=0.1, color='green', label='Predictive')
        ax.axvspan(0, 24, alpha=0.1, color='red', label='Reactive')
        
        ax.set_xlabel('Lag (hours)')
        ax.set_ylabel('Correlation')
        ax.set_title(f'{prob_col}\nOptimal Lag: {optimal_lag}h', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(prob_cols), 4):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hmm_lag_analysis.svg'), format='svg', bbox_inches='tight')
    plt.close()


def run_hmm_audit_comprehensive(
    segment_id: int = 0,
    force_retrain: bool = False,
    use_tsne: bool = False,
    output_dir: Optional[str] = None
) -> Dict:
    """
    Run comprehensive HMM audit with all analyses.
    
    Args:
        segment_id: WFO segment ID
        force_retrain: Force HMM retraining
        use_tsne: Also generate t-SNE plots (slower)
        output_dir: Output directory (if None, creates timestamped directory)
    
    Returns:
        Dict with all audit results
    """
    print("=" * 70)
    print("HMM COMPREHENSIVE AUDIT")
    print("=" * 70)
    print(f"Segment: {segment_id}, Force Retrain: {force_retrain}, t-SNE: {use_tsne}")
    
    # Create output directory
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_base = os.path.join("results", "hmm_audit")
        os.makedirs(output_base, exist_ok=True)
        output_dir = os.path.join(output_base, timestamp)
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}")
    
    results = {
        'output_dir': output_dir,
        'segment_id': segment_id,
        'config': {}
    }
    
    # Load data
    print("\n[1/7] Loading data...")
    df = load_or_create_data(segment_id=segment_id, force_retrain=force_retrain)
    print(f"  Shape: {df.shape}")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")
    
    # Check if HMM states exist (HMM_Prob_* ou Prob_*)
    has_hmm = any(col.startswith('HMM_Prob_') or col.startswith('Prob_') for col in df.columns)
    
    # Train or use existing HMM
    print(f"\n[2/7] Preparing HMM...")
    if not has_hmm or force_retrain:
        print("  Training new HMM...")
        detector = RegimeDetector(n_components=4)
        df = detector.fit_predict(df, segment_id=segment_id)
        results['hmm_trained'] = True
    else:
        print("  Using existing HMM states from data...")
        # Try to load saved HMM model
        detector = None
        hmm_paths = [
            f"models/wfo/segment_{segment_id}/hmm.pkl",
            f"models/wfo/hmm.pkl",
            "models/hmm.pkl"
        ]
        
        for hmm_path in hmm_paths:
            if os.path.exists(hmm_path):
                try:
                    print(f"  Loading HMM from {hmm_path}...")
                    detector = RegimeDetector.load(hmm_path)
                    results['hmm_trained'] = False
                    results['hmm_loaded_from'] = hmm_path
                    print("  [OK] HMM loaded successfully")
                    break
                except Exception as e:
                    print(f"  [WARNING] Failed to load HMM from {hmm_path}: {e}")
                    continue
        
        if detector is None:
            print("  [NOTE] No saved HMM found - some analyses will be limited")
            print("  [TIP] Use --hmm-retrain to train a new HMM for full analysis")
            results['hmm_trained'] = False
    
    results['config'] = {
        'n_components': 4,
        'features': RegimeDetector.HMM_FEATURES,
        'has_hmm_object': detector is not None
    }
    
    # Training Dynamics Analysis
    print(f"\n[3/7] Analyzing training dynamics...")
    if detector is not None:
        training_results = analyze_hmm_training_dynamics(detector, df, output_dir)
        results['training_dynamics'] = training_results
    else:
        print("  [SKIP] Requires HMM detector object")
        results['training_dynamics'] = {}
    
    # Regime Detection Quality Analysis
    print(f"\n[4/7] Analyzing regime detection quality...")
    if detector is not None:
        regime_results = analyze_regime_detection_quality(detector, df, output_dir)
    else:
        # Limited analysis without detector
        try:
            states = get_dominant_state(df)
            stats = compute_state_statistics(df, states)
            financial_metrics = _calculate_financial_fingerprint(df, states)
            
            regime_results = {
                'state_statistics': stats,
                'financial_fingerprint': financial_metrics
            }
            
            # Generate plots that don't require detector
            _plot_hmm_financial_fingerprint(financial_metrics, output_dir)
            _plot_hmm_regime_timeline(df, states, output_dir)
        except ValueError as e:
            print(f"  [ERROR] {e}")
            regime_results = {}
    
    results['regime_detection'] = regime_results
    
    # Feature Space Analysis
    print(f"\n[5/7] Analyzing feature space...")
    if detector is not None:
        feature_results = analyze_hmm_feature_space(detector, df, output_dir, use_tsne=use_tsne)
        results['feature_space'] = feature_results
    else:
        print("  [SKIP] Requires HMM detector object")
        results['feature_space'] = {}
    
    # Predictive Power Analysis
    print(f"\n[6/7] Analyzing predictive power...")
    predictive_results = analyze_hmm_predictive_power(df, output_dir)
    results['predictive_power'] = predictive_results
    
    # Generate Report
    print(f"\n[7/7] Generating report...")
    report_path = generate_hmm_audit_report(results, output_dir)
    results['report_path'] = report_path
    
    print("\n" + "=" * 70)
    print("HMM AUDIT COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {output_dir}")
    print(f"Report: {report_path}")
    print("=" * 70)
    
    return results


def generate_hmm_audit_report(results: Dict, output_dir: str) -> str:
    """Generate comprehensive text report."""
    report_path = os.path.join(output_dir, "hmm_audit_report.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("HMM COMPREHENSIVE AUDIT REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Header
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Segment ID: {results.get('segment_id', 'N/A')}\n")
        f.write(f"Output Directory: {results.get('output_dir', 'N/A')}\n\n")
        
        config = results.get('config', {})
        f.write("CONFIGURATION\n")
        f.write("-" * 80 + "\n")
        for key, value in config.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        # Section 1: Training Dynamics
        f.write("=" * 80 + "\n")
        f.write("SECTION 1: TRAINING DYNAMICS\n")
        f.write("=" * 80 + "\n\n")
        
        training = results.get('training_dynamics', {})
        if training:
            if training.get('em_history'):
                f.write("EM Convergence:\n")
                f.write(f"  Iterations: {training.get('n_iterations', 0)}\n")
                f.write(f"  Converged: {training.get('converged', False)}\n")
                if training.get('final_log_likelihood') is not None:
                    f.write(f"  Final Log-Likelihood: {training['final_log_likelihood']:.4f}\n")
                f.write("\n")
            
            f.write("Quality Metrics:\n")
            if training.get('separation_score') is not None:
                f.write(f"  Separation Score: {training['separation_score']:.6f}\n")
            if training.get('transition_entropy') is not None:
                f.write(f"  Transition Entropy: {training['transition_entropy']:.4f}\n")
            if training.get('diagonal_avg') is not None:
                f.write(f"  Diagonal Average: {training['diagonal_avg']:.4f}\n")
            if training.get('kmeans_inertia') is not None:
                f.write(f"  K-Means Inertia: {training['kmeans_inertia']:.2f}\n")
            f.write("\n")
            
            # Seed Robustness
            robustness = training.get('seed_robustness', {})
            if robustness:
                f.write("Seed Robustness Test:\n")
                f.write(f"  Average Overlap: {robustness.get('overlap', 0.0):.1%}\n")
                f.write(f"  Minimum Overlap: {robustness.get('min_overlap', 0.0):.1%}\n")
                f.write(f"  Stable: {robustness.get('stable', False)}\n")
                if not robustness.get('stable', True):
                    f.write("  [!] ALERT: HMM is unstable (overlap < 80%) - dangerous for RL\n")
                f.write("\n")
        else:
            f.write("  [INFO] Training dynamics not available (HMM object required)\n\n")
        
        # Section 2: Regime Detection Quality
        f.write("=" * 80 + "\n")
        f.write("SECTION 2: REGIME DETECTION QUALITY\n")
        f.write("=" * 80 + "\n\n")
        
        regime = results.get('regime_detection', {})
        if regime.get('state_statistics') is not None:
            stats = regime['state_statistics']
            f.write("State Statistics:\n")
            for _, row in stats.iterrows():
                f.write(f"  State {int(row['State'])}: Count={int(row['Count'])}, "
                       f"Pct={row['Pct']:.1f}%, Win Rate={row['Win_Rate']:.1f}%, "
                       f"Mean Ret={row['Mean_Ret']:.4f}%, Avg Duration={row['Avg_Duration']:.1f}h\n")
            f.write("\n")
        
        # Financial Fingerprint
        financial = regime.get('financial_fingerprint')
        if financial is not None and len(financial) > 0:
            f.write("Financial Fingerprint:\n")
            for _, row in financial.iterrows():
                f.write(f"  State {int(row['State'])}:\n")
                f.write(f"    Annualized Return: {row['Annualized_Return']:.2f}%\n")
                f.write(f"    Sharpe Ratio: {row['Sharpe_Ratio']:.2f}\n")
                f.write(f"    Max Drawdown: {row['Max_Drawdown']:.2f}%\n")
                f.write(f"    Volatility: {row['Volatility']:.2f}%\n")
            f.write("\n")
            
            # Check if states are financially distinct
            if len(financial) > 1:
                sharpe_range = financial['Sharpe_Ratio'].max() - financial['Sharpe_Ratio'].min()
                return_range = financial['Annualized_Return'].max() - financial['Annualized_Return'].min()
                if sharpe_range < 0.5 and return_range < 10:
                    f.write("  [!] ALERT: States have similar financial metrics - HMM may separate noise\n")
                else:
                    f.write("  [OK] States are financially distinct\n")
            f.write("\n")
        
        # Section 3: Feature Space Structure
        f.write("=" * 80 + "\n")
        f.write("SECTION 3: FEATURE SPACE STRUCTURE\n")
        f.write("=" * 80 + "\n\n")
        
        feature_space = results.get('feature_space', {})
        if feature_space:
            f.write(f"Features: {len(feature_space.get('feature_names', []))}\n")
            if feature_space.get('pca_explained_variance') is not None:
                f.write(f"PCA Explained Variance: {feature_space['pca_explained_variance']*100:.2f}%\n")
            if feature_space.get('silhouette_score') is not None:
                silhouette = feature_space['silhouette_score']
                f.write(f"Silhouette Score: {silhouette:.4f}\n")
                if silhouette < 0.3:
                    f.write("  [!] SPAGHETTI PLOT: Low separability (silhouette < 0.3)\n")
                    f.write("      States are not well separated in feature space\n")
                elif silhouette < 0.5:
                    f.write("  [~] WEAK SEPARABILITY: Moderate clustering (silhouette < 0.5)\n")
                else:
                    f.write("  [OK] GOOD SEPARABILITY: Clear clustering by states\n")
            f.write("\n")
            
            # Feature Attribution
            attribution = feature_space.get('feature_attribution', {})
            if attribution:
                f.write("Feature Attribution (Significant Deviations):\n")
                for state, state_data in sorted(attribution.items()):
                    sig_features = state_data.get('significant_features', [])
                    if sig_features:
                        f.write(f"  State {state}:\n")
                        for feat_info in sig_features:
                            f.write(f"    {feat_info['feature']}: z-score={feat_info['z_score']:.2f}\n")
                f.write("\n")
        else:
            f.write("  [INFO] Feature space analysis not available (HMM object required)\n\n")
        
        # Section 4: Predictive Power
        f.write("=" * 80 + "\n")
        f.write("SECTION 4: PREDICTIVE POWER\n")
        f.write("=" * 80 + "\n\n")
        
        predictive = results.get('predictive_power', {})
        correlations = predictive.get('correlations', {})
        if correlations:
            f.write("Correlations with Future Returns:\n")
            for prob_col, corr_dict in correlations.items():
                f.write(f"  {prob_col}:\n")
                for horizon, corr in corr_dict.items():
                    f.write(f"    {horizon} ahead: {corr:.4f}\n")
            f.write("\n")
        
        lag_analysis = predictive.get('lag_analysis', {})
        if lag_analysis:
            f.write("Lag Analysis (Latency of Detection):\n")
            for prob_col, lag_data in lag_analysis.items():
                optimal_lag = lag_data.get('optimal_lag', 0)
                lead_time = lag_data.get('lead_time', 0)
                is_predictive = lag_data.get('is_predictive', False)
                
                f.write(f"  {prob_col}:\n")
                f.write(f"    Optimal Lag: {optimal_lag}h\n")
                if is_predictive:
                    f.write(f"    Lead Time: {lead_time}h (PREDICTIVE)\n")
                else:
                    f.write(f"    Lead Time: {lead_time}h (REACTIVE)\n")
                
                if optimal_lag > 4:
                    f.write(f"    [!] WARNING: High lag ({optimal_lag}h) - may be useless for high-frequency trading\n")
            f.write("\n")
        
        calibration = predictive.get('calibration', {})
        if calibration:
            f.write("Calibration Analysis:\n")
            f.write("  See reliability diagram in hmm_calibration.png\n")
            f.write("  Perfect calibration: observed frequency = predicted probability\n")
            f.write("\n")
        
        # Section 5: Recommendations
        f.write("=" * 80 + "\n")
        f.write("SECTION 5: RECOMMENDATIONS\n")
        f.write("=" * 80 + "\n\n")
        
        recommendations = []
        
        # Check training dynamics
        if training:
            robustness = training.get('seed_robustness', {})
            if robustness and not robustness.get('stable', True):
                recommendations.append("- HMM is unstable (seed robustness < 80%) - consider fixing random seed or increasing training iterations")
        
        # Check financial fingerprint
        if financial is not None and len(financial) > 1:
            sharpe_range = financial['Sharpe_Ratio'].max() - financial['Sharpe_Ratio'].min()
            if sharpe_range < 0.5:
                recommendations.append("- States have similar financial profiles - HMM may not capture distinct market regimes")
        
        # Check separability
        if feature_space.get('silhouette_score') is not None:
            if feature_space['silhouette_score'] < 0.3:
                recommendations.append("- Low feature space separability - consider feature engineering or different HMM configuration")
        
        # Check lag
        if lag_analysis:
            high_lag_states = [col for col, data in lag_analysis.items() if data.get('optimal_lag', 0) > 4]
            if high_lag_states:
                recommendations.append(f"- High detection lag detected for {len(high_lag_states)} state(s) - may limit trading frequency")
        
        if not recommendations:
            recommendations.append("- HMM appears healthy. Continue monitoring during full training.")
        
        for i, rec in enumerate(recommendations, 1):
            f.write(f"{i}. {rec}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    return report_path


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
    """
    Extract HMM probability features (Test B).
    
    Priorité:
    1. HMM_Prob_* (Belief States Forward-Only, pas de look-ahead bias)
    2. Prob_* (Forward-Backward, legacy)
    """
    # Chercher d'abord les belief states Forward-Only (préférées)
    hmm_cols = [col for col in df.columns if col.startswith('HMM_Prob_')]
    
    # Fallback sur Prob_* si HMM_Prob_* n'existent pas (compatibilité legacy)
    if len(hmm_cols) == 0:
        hmm_cols = [col for col in df.columns if col.startswith('Prob_')]
    
    if len(hmm_cols) == 0:
        raise ValueError("No HMM probability columns found (neither HMM_Prob_* nor Prob_*)")

    X = df[hmm_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.25)
    
    # Log quel type de colonnes on utilise
    if hmm_cols[0].startswith('HMM_Prob_'):
        print(f"      [INFO] Using Forward-Only belief states (HMM_Prob_*)")
    else:
        print(f"      [INFO] Using Forward-Backward probabilities (Prob_*)")
    
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
    """
    Train MAE with supervised auxiliary loss.
    
    Returns:
        model: Trained CryptoMAE model
        history: Dict with complete training history including:
            - train_total, train_recon, train_aux (losses)
            - val_total, val_recon, val_aux, val_accuracy
            - gradient_norms (per epoch)
    """
    # #region agent log
    debug_log_path = get_debug_log_path()
    try:
        with open(debug_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({
                'sessionId': 'debug-session',
                'runId': 'run1',
                'hypothesisId': 'B',
                'location': 'audit_pipeline.py:2106',
                'message': 'train_supervised_mae entry',
                'data': {'input_dim': input_dim, 'epochs': epochs, 'device': str(device)},
                'timestamp': int(time.time() * 1000)
            }) + '\n')
    except Exception:
        pass
    # #endregion
    
    # #region agent log
    try:
        with open(debug_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({
                'sessionId': 'debug-session',
                'runId': 'run1',
                'hypothesisId': 'B',
                'location': 'audit_pipeline.py:2125',
                'message': 'Before model creation',
                'data': {},
                'timestamp': int(time.time() * 1000)
            }) + '\n')
    except Exception:
        pass
    # #endregion
    model = CryptoMAE(
        input_dim=input_dim,
        d_model=64,
        n_heads=4,
        n_layers=2
    )
    # #region agent log
    try:
        with open(debug_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({
                'sessionId': 'debug-session',
                'runId': 'run1',
                'hypothesisId': 'B',
                'location': 'audit_pipeline.py:2130',
                'message': 'After model creation, before .to(device)',
                'data': {},
                'timestamp': int(time.time() * 1000)
            }) + '\n')
    except Exception:
        pass
    # #endregion
    model = model.to(device)
    # #region agent log
    try:
        with open(debug_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({
                'sessionId': 'debug-session',
                'runId': 'run1',
                'hypothesisId': 'B',
                'location': 'audit_pipeline.py:2132',
                'message': 'After model.to(device)',
                'data': {},
                'timestamp': int(time.time() * 1000)
            }) + '\n')
    except Exception:
        pass
    # #endregion

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # #region agent log
    try:
        with open(debug_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({
                'sessionId': 'debug-session',
                'runId': 'run1',
                'hypothesisId': 'B',
                'location': 'audit_pipeline.py:2134',
                'message': 'After optimizer creation',
                'data': {},
                'timestamp': int(time.time() * 1000)
            }) + '\n')
    except Exception:
        pass
    # #endregion
    
    # Complete history tracking
    history = {
        'train_total': [],
        'train_recon': [],
        'train_aux': [],
        'val_total': [],
        'val_recon': [],
        'val_aux': [],
        'val_accuracy': [],
        'gradient_norms': []
    }

    # #region agent log
    try:
        with open(debug_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({
                'sessionId': 'debug-session',
                'runId': 'run1',
                'hypothesisId': 'C',
                'location': 'audit_pipeline.py:2146',
                'message': 'Before epochs loop',
                'data': {'epochs': epochs},
                'timestamp': int(time.time() * 1000)
            }) + '\n')
    except Exception:
        pass
    # #endregion
    
    for epoch in range(epochs):
        # #region agent log
        try:
            with open(debug_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({
                    'sessionId': 'debug-session',
                    'runId': 'run1',
                    'hypothesisId': 'C',
                    'location': 'audit_pipeline.py:2147',
                    'message': 'Epoch start',
                    'data': {'epoch': epoch + 1, 'total_epochs': epochs},
                    'timestamp': int(time.time() * 1000)
                }) + '\n')
        except Exception:
            pass
        # #endregion
        # Training phase
        model.train()
        epoch_train_total = 0.0
        epoch_train_recon = 0.0
        epoch_train_aux = 0.0
        epoch_grad_norms = []
        n_train_batches = 0

        # #region agent log
        try:
            with open(debug_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({
                    'sessionId': 'debug-session',
                    'runId': 'run1',
                    'hypothesisId': 'A',
                    'location': 'audit_pipeline.py:2155',
                    'message': 'Before train_loader iteration',
                    'data': {'epoch': epoch + 1},
                    'timestamp': int(time.time() * 1000)
                }) + '\n')
        except Exception:
            pass
        # #endregion
        
        batch_iter = iter(train_loader)
        batch_idx = 0
        while True:
            # #region agent log
            if batch_idx <= 5:
                try:
                    with open(debug_log_path, 'a', encoding='utf-8') as f:
                        f.write(json.dumps({
                            'sessionId': 'debug-session',
                            'runId': 'run1',
                            'hypothesisId': 'A',
                            'location': 'audit_pipeline.py:2338',
                            'message': 'Before next() call on iterator',
                            'data': {'epoch': epoch + 1, 'batch_idx': batch_idx},
                            'timestamp': int(time.time() * 1000)
                        }) + '\n')
                        f.flush()
                except Exception:
                    pass
            # #endregion
            try:
                x, y = next(batch_iter)
            except StopIteration:
                break
            batch_idx += 1
            # #region agent log
            if n_train_batches == 0:
                try:
                    with open(debug_log_path, 'a', encoding='utf-8') as f:
                        f.write(json.dumps({
                            'sessionId': 'debug-session',
                            'runId': 'run1',
                            'hypothesisId': 'A',
                            'location': 'audit_pipeline.py:2156',
                            'message': 'First batch loaded',
                            'data': {'epoch': epoch + 1, 'x_shape': list(x.shape), 'y_shape': list(y.shape)},
                            'timestamp': int(time.time() * 1000)
                        }) + '\n')
                except Exception:
                    pass
            # #endregion
            # #region agent log
            if n_train_batches == 0:
                try:
                    with open(debug_log_path, 'a', encoding='utf-8') as f:
                        f.write(json.dumps({
                            'sessionId': 'debug-session',
                            'runId': 'run1',
                            'hypothesisId': 'C',
                            'location': 'audit_pipeline.py:2157',
                            'message': 'Before .to(device)',
                            'data': {'epoch': epoch + 1},
                            'timestamp': int(time.time() * 1000)
                        }) + '\n')
                except Exception:
                    pass
            # #endregion
            x, y = x.to(device), y.to(device)
            # S'assurer que y a shape (batch, 1) pour compatibilité avec pred_logits
            if y.dim() == 1:
                y = y.unsqueeze(1)
            # #region agent log
            if n_train_batches == 0:
                try:
                    with open(debug_log_path, 'a', encoding='utf-8') as f:
                        f.write(json.dumps({
                            'sessionId': 'debug-session',
                            'runId': 'run1',
                            'hypothesisId': 'C',
                            'location': 'audit_pipeline.py:2159',
                            'message': 'After .to(device), before forward',
                            'data': {'epoch': epoch + 1},
                            'timestamp': int(time.time() * 1000)
                        }) + '\n')
                except Exception:
                    pass
            # #endregion
            optimizer.zero_grad()

            # #region agent log
            if n_train_batches == 0:
                try:
                    with open(debug_log_path, 'a', encoding='utf-8') as f:
                        f.write(json.dumps({
                            'sessionId': 'debug-session',
                            'runId': 'run1',
                            'hypothesisId': 'C',
                            'location': 'audit_pipeline.py:2161',
                            'message': 'Before model forward',
                            'data': {'epoch': epoch + 1},
                            'timestamp': int(time.time() * 1000)
                        }) + '\n')
                except Exception:
                    pass
            # #endregion
            pred, target_recon, mask, pred_logits = model(x, mask_ratio=mask_ratio)
            # #region agent log
            if n_train_batches == 0:
                try:
                    with open(debug_log_path, 'a', encoding='utf-8') as f:
                        f.write(json.dumps({
                            'sessionId': 'debug-session',
                            'runId': 'run1',
                            'hypothesisId': 'C',
                            'location': 'audit_pipeline.py:2162',
                            'message': 'After model forward',
                            'data': {'epoch': epoch + 1},
                            'timestamp': int(time.time() * 1000)
                        }) + '\n')
                except Exception:
                    pass
            # #endregion
            recon_loss = F.mse_loss(pred[mask], target_recon)
            aux_loss = F.binary_cross_entropy_with_logits(pred_logits, y)
            loss = recon_loss + aux_weight * aux_loss

            # #region agent log
            if n_train_batches == 0:
                try:
                    with open(debug_log_path, 'a', encoding='utf-8') as f:
                        f.write(json.dumps({
                            'sessionId': 'debug-session',
                            'runId': 'run1',
                            'hypothesisId': 'C',
                            'location': 'audit_pipeline.py:2164',
                            'message': 'Before backward',
                            'data': {'epoch': epoch + 1},
                            'timestamp': int(time.time() * 1000)
                        }) + '\n')
                except Exception:
                    pass
            # #endregion
            loss.backward()
            # #region agent log
            if n_train_batches == 0:
                try:
                    with open(debug_log_path, 'a', encoding='utf-8') as f:
                        f.write(json.dumps({
                            'sessionId': 'debug-session',
                            'runId': 'run1',
                            'hypothesisId': 'C',
                            'location': 'audit_pipeline.py:2167',
                            'message': 'After backward, before optimizer step',
                            'data': {'epoch': epoch + 1},
                            'timestamp': int(time.time() * 1000)
                        }) + '\n')
                except Exception:
                    pass
            # #endregion
            
            # Track gradient norm before clipping
            grad_norm = torch_utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
            epoch_grad_norms.append(grad_norm.item())
            
            optimizer.step()
            # #region agent log
            if n_train_batches == 0:
                try:
                    with open(debug_log_path, 'a', encoding='utf-8') as f:
                        f.write(json.dumps({
                            'sessionId': 'debug-session',
                            'runId': 'run1',
                            'hypothesisId': 'C',
                            'location': 'audit_pipeline.py:2171',
                            'message': 'After optimizer step',
                            'data': {'epoch': epoch + 1},
                            'timestamp': int(time.time() * 1000)
                        }) + '\n')
                except Exception:
                    pass
            # #endregion

            epoch_train_total += loss.item()
            epoch_train_recon += recon_loss.item()
            epoch_train_aux += aux_loss.item()
            n_train_batches += 1
            
            # #region agent log
            if n_train_batches % 100 == 0 or n_train_batches <= 5:
                try:
                    with open(debug_log_path, 'a', encoding='utf-8') as f:
                        f.write(json.dumps({
                            'sessionId': 'debug-session',
                            'runId': 'run1',
                            'hypothesisId': 'A',
                            'location': 'audit_pipeline.py:2175',
                            'message': 'Batch processed',
                            'data': {'epoch': epoch + 1, 'batch': n_train_batches, 'total_batches': len(train_loader)},
                            'timestamp': int(time.time() * 1000)
                        }) + '\n')
                        f.flush()  # Force flush to ensure log is written
                except Exception:
                    pass
            # #endregion
            
            # Print progress every 100 batches to show the script is working
            if n_train_batches % 100 == 0:
                print(f"    Batch {n_train_batches}/{len(train_loader)} (Epoch {epoch+1}/{epochs})", flush=True)
            
            # #region agent log
            if n_train_batches <= 5:
                try:
                    with open(debug_log_path, 'a', encoding='utf-8') as f:
                        f.write(json.dumps({
                            'sessionId': 'debug-session',
                            'runId': 'run1',
                            'hypothesisId': 'A',
                            'location': 'audit_pipeline.py:2338',
                            'message': 'After batch processed, before next iteration',
                            'data': {'epoch': epoch + 1, 'batch': n_train_batches},
                            'timestamp': int(time.time() * 1000)
                        }) + '\n')
                        f.flush()  # Force flush to ensure log is written
                except Exception:
                    pass
            # #endregion

        # Average training metrics
        # #region agent log
        try:
            with open(debug_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({
                    'sessionId': 'debug-session',
                    'runId': 'run1',
                    'hypothesisId': 'C',
                    'location': 'audit_pipeline.py:2177',
                    'message': 'Training phase complete',
                    'data': {'epoch': epoch + 1, 'n_batches': n_train_batches},
                    'timestamp': int(time.time() * 1000)
                }) + '\n')
        except Exception:
            pass
        # #endregion
        history['train_total'].append(epoch_train_total / n_train_batches)
        history['train_recon'].append(epoch_train_recon / n_train_batches)
        history['train_aux'].append(epoch_train_aux / n_train_batches)
        history['gradient_norms'].append(np.mean(epoch_grad_norms))

        # Validation phase
        # #region agent log
        try:
            with open(debug_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({
                    'sessionId': 'debug-session',
                    'runId': 'run1',
                    'hypothesisId': 'C',
                    'location': 'audit_pipeline.py:2183',
                    'message': 'Starting validation phase',
                    'data': {'epoch': epoch + 1},
                    'timestamp': int(time.time() * 1000)
                }) + '\n')
        except Exception:
            pass
        # #endregion
        model.eval()
        epoch_val_total = 0.0
        epoch_val_recon = 0.0
        epoch_val_aux = 0.0
        correct, total = 0, 0
        n_val_batches = 0

        with torch.no_grad():
            # #region agent log
            try:
                with open(debug_log_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps({
                        'sessionId': 'debug-session',
                        'runId': 'run1',
                        'hypothesisId': 'A',
                        'location': 'audit_pipeline.py:2192',
                        'message': 'Before val_loader iteration',
                        'data': {'epoch': epoch + 1},
                        'timestamp': int(time.time() * 1000)
                    }) + '\n')
            except Exception:
                pass
            # #endregion
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                # S'assurer que y a shape (batch, 1) pour compatibilité avec pred_logits
                if y.dim() == 1:
                    y = y.unsqueeze(1)
                pred, target_recon, mask, pred_logits = model(x, mask_ratio=mask_ratio)
                
                recon_loss = F.mse_loss(pred[mask], target_recon)
                aux_loss = F.binary_cross_entropy_with_logits(pred_logits, y)
                loss = recon_loss + aux_weight * aux_loss

                pred_dir = (torch.sigmoid(pred_logits) > 0.5).float()
                correct += (pred_dir == y).sum().item()
                total += y.numel()

                epoch_val_total += loss.item()
                epoch_val_recon += recon_loss.item()
                epoch_val_aux += aux_loss.item()
                n_val_batches += 1

        val_acc = correct / total if total > 0 else 0.0
        history['val_total'].append(epoch_val_total / n_val_batches)
        history['val_recon'].append(epoch_val_recon / n_val_batches)
        history['val_aux'].append(epoch_val_aux / n_val_batches)
        history['val_accuracy'].append(val_acc)

        # Log progression for every epoch
        # #region agent log
        try:
            with open(debug_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({
                    'sessionId': 'debug-session',
                    'runId': 'run1',
                    'hypothesisId': 'D',
                    'location': 'audit_pipeline.py:2215',
                    'message': 'Before logger.info call',
                    'data': {'epoch': epoch + 1},
                    'timestamp': int(time.time() * 1000)
                }) + '\n')
        except Exception:
            pass
        # #endregion
        try:
            logger.info(
                f"MAE Epoch {epoch+1:2d}/{epochs} | "
                f"Train Total: {history['train_total'][-1]:.6f} "
                f"(Recon: {history['train_recon'][-1]:.6f}, Aux: {history['train_aux'][-1]:.6f}) | "
                f"Val Total: {history['val_total'][-1]:.6f} "
                f"(Recon: {history['val_recon'][-1]:.6f}, Aux: {history['val_aux'][-1]:.6f}) | "
                f"Val Acc: {val_acc*100:.2f}% | "
                f"Grad Norm: {history['gradient_norms'][-1]:.6f}"
            )
        except Exception as e:
            # #region agent log
            try:
                with open(debug_log_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps({
                        'sessionId': 'debug-session',
                        'runId': 'run1',
                        'hypothesisId': 'D',
                        'location': 'audit_pipeline.py:2216',
                        'message': 'Logger.info exception',
                        'data': {'epoch': epoch + 1, 'error': str(e)},
                        'timestamp': int(time.time() * 1000)
                    }) + '\n')
            except Exception:
                pass
            # #endregion

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:2d}/{epochs} | "
                  f"Train: {history['train_total'][-1]:.4f} "
                  f"(R:{history['train_recon'][-1]:.4f} A:{history['train_aux'][-1]:.4f}) | "
                  f"Val: {history['val_total'][-1]:.4f} "
                  f"(Acc:{val_acc*100:.1f}%) | "
                  f"Grad: {history['gradient_norms'][-1]:.4f}")

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
            # S'assurer que y a shape (batch, 1) pour compatibilité avec pred_logits
            if y.dim() == 1:
                y = y.unsqueeze(1)
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
# MAE AUDIT FUNCTIONS
# ============================================================================

def analyze_classification_metrics(
    model: CryptoMAE,
    test_loader: DataLoader,
    device: torch.device
) -> Dict:
    """
    Analyze classification metrics for the supervised prediction head.
    
    Returns:
        Dict with:
            - accuracy: Overall accuracy
            - confusion_matrix: (TP, TN, FP, FN)
            - precision: TP / (TP + FP)
            - recall: TP / (TP + FN)
            - f1_score: Harmonic mean of precision and recall
    """
    model.eval()
    
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for x, y_true in test_loader:
            x = x.to(device)
            y_true = y_true.to(device)
            
            # S'assurer que y_true a shape (batch, 1)
            if y_true.dim() == 1:
                y_true = y_true.unsqueeze(1)
            
            # Forward pass - récupérer les 4 valeurs
            pred, target_recon, mask, pred_logits = model(x, mask_ratio=0.0)
            
            # Stockage pour métriques globales
            all_logits.append(pred_logits.detach().cpu())
            all_labels.append(y_true.detach().cpu())
    
    # Concatenate all predictions
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Convert to predictions
    pred_probs = torch.sigmoid(all_logits)
    pred_classes = (pred_probs > 0.5).float()
    
    # Calculate metrics
    accuracy = (pred_classes == all_labels).float().mean().item()
    
    # Confusion matrix
    TP = ((pred_classes == 1) & (all_labels == 1)).sum().item()
    TN = ((pred_classes == 0) & (all_labels == 0)).sum().item()
    FP = ((pred_classes == 1) & (all_labels == 0)).sum().item()
    FN = ((pred_classes == 0) & (all_labels == 1)).sum().item()
    
    # Precision, Recall, F1
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': {'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN},
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'n_samples': len(all_labels)
    }


def analyze_reconstruction_quality(
    model: CryptoMAE,
    test_loader: DataLoader,
    feature_names: List[str],
    device: torch.device,
    n_samples: int = 5,
    output_dir: Optional[str] = None
) -> Dict:
    """
    Analyze reconstruction quality: time-series visualization and feature breakdown.
    
    Returns:
        Dict with:
            - reconstruction_samples: List of (original, reconstructed) for visualization
            - feature_mse: Dict mapping feature_name -> MSE
            - best_features: List of top features (lowest MSE)
            - worst_features: List of worst features (highest MSE)
    """
    model.eval()
    
    # Collect samples for visualization
    samples_to_viz = []
    all_original = []
    all_reconstructed = []
    all_masks = []
    
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            if batch_idx >= n_samples:
                break
            x = x.to(device)
            
            pred, target_recon, mask, _ = model(x, mask_ratio=0.15)
            
            # Store for visualization (first sequence in batch)
            samples_to_viz.append({
                'original': x[0].cpu().numpy(),  # (seq_len, n_features)
                'reconstructed': pred[0].cpu().numpy(),
                'mask': mask[0].cpu().numpy()
            })
            
            # Collect all for feature breakdown
            all_original.append(x.cpu().numpy())
            all_reconstructed.append(pred.cpu().numpy())
            all_masks.append(mask.cpu().numpy())
    
    # Concatenate all batches
    all_original = np.concatenate(all_original, axis=0)  # (n_total, seq_len, n_features)
    all_reconstructed = np.concatenate(all_reconstructed, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)
    
    # Calculate MSE per feature (only on masked positions)
    feature_mse = {}
    for feat_idx, feat_name in enumerate(feature_names):
        # Extract masked positions for this feature
        masked_original = all_original[all_masks, feat_idx]
        masked_reconstructed = all_reconstructed[all_masks, feat_idx]
        
        mse = np.mean((masked_original - masked_reconstructed) ** 2)
        feature_mse[feat_name] = mse
    
    # Sort features by MSE
    sorted_features = sorted(feature_mse.items(), key=lambda x: x[1])
    best_features = [name for name, _ in sorted_features[:5]]
    worst_features = [name for name, _ in sorted_features[-5:]]
    
    # Generate visualizations
    if output_dir:
        _plot_reconstruction_samples(samples_to_viz, feature_names, output_dir)
        _plot_feature_breakdown(feature_mse, output_dir)
    
    return {
        'reconstruction_samples': samples_to_viz,
        'feature_mse': feature_mse,
        'best_features': best_features,
        'worst_features': worst_features,
        'overall_mse': np.mean(list(feature_mse.values()))
    }


def _plot_reconstruction_samples(
    samples: List[Dict],
    feature_names: List[str],
    output_dir: str
):
    """Plot time-series reconstruction for selected features."""
    # Select important features to visualize
    important_features = []
    for feat in feature_names:
        if any(ind in feat for ind in ['RSI_14', 'MACD_Hist', 'ADX_14', 'Vol_', 'Parkinson', 'GK']):
            important_features.append(feat)
    
    if len(important_features) > 6:
        important_features = important_features[:6]
    
    n_samples = len(samples)
    n_features = len(important_features)
    
    fig, axes = plt.subplots(n_samples, n_features, figsize=(4*n_features, 3*n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    if n_features == 1:
        axes = axes.reshape(-1, 1)
    
    for sample_idx, sample in enumerate(samples):
        original = sample['original']  # (seq_len, n_features)
        reconstructed = sample['reconstructed']
        mask = sample['mask']  # (seq_len,)
        
        for feat_idx, feat_name in enumerate(important_features):
            if feat_name not in feature_names:
                continue
            feat_col_idx = feature_names.index(feat_name)
            
            ax = axes[sample_idx, feat_idx]
            seq_len = len(original)
            time_steps = np.arange(seq_len)
            
            # Plot original
            ax.plot(time_steps, original[:, feat_col_idx], 'b-', label='Original', linewidth=0.5, alpha=0.7)
            
            # Plot reconstructed
            ax.plot(time_steps, reconstructed[:, feat_col_idx], 'orange', label='Reconstructed', 
                   linewidth=0.5, linestyle='--', alpha=0.7)
            
            # Highlight masked positions
            masked_positions = np.where(mask)[0]
            if len(masked_positions) > 0:
                ax.scatter(masked_positions, original[masked_positions, feat_col_idx], 
                          c='red', s=30, marker='x', label='Masked', zorder=5)
            
            ax.set_title(f'{feat_name}\n(Sample {sample_idx+1})', fontsize=9)
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Value')
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'reconstruction_samples.svg'), format='svg', bbox_inches='tight')
    plt.close()


def _plot_feature_breakdown(feature_mse: Dict[str, float], output_dir: str):
    """Plot bar chart of MSE by feature."""
    sorted_features = sorted(feature_mse.items(), key=lambda x: x[1], reverse=True)
    features, mses = zip(*sorted_features)
    
    plt.figure(figsize=(12, 8))
    colors = ['red' if mse > np.median(list(feature_mse.values())) else 'blue' 
              for mse in mses]
    plt.barh(range(len(features)), mses, color=colors, alpha=0.7)
    plt.yticks(range(len(features)), features, fontsize=8)
    plt.xlabel('MSE (Reconstruction Error)', fontsize=10)
    plt.title('Reconstruction Quality by Feature\n(Lower is Better)', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_breakdown.svg'), format='svg', bbox_inches='tight')
    plt.close()


def analyze_latent_space(
    model: CryptoMAE,
    test_loader: DataLoader,
    hmm_states: Optional[np.ndarray] = None,
    device: torch.device = None,
    n_samples: int = 2000,
    output_dir: Optional[str] = None,
    use_tsne: bool = False
) -> Dict:
    """
    Analyze latent space structure with PCA/t-SNE and HMM state coloring.
    
    Args:
        model: Trained CryptoMAE model
        test_loader: DataLoader for test data
        hmm_states: Optional array of HMM states (same length as test data)
        device: Torch device
        n_samples: Number of samples to analyze
        output_dir: Directory to save plots
        use_tsne: If True, also generate t-SNE plot (slower)
    
    Returns:
        Dict with embeddings, projections, and separability metrics
    """
    model.eval()
    
    embeddings_list = []
    directions_list = []
    states_list = []
    
    with torch.no_grad():
        sample_count = 0
        for batch_idx, (x, y) in enumerate(test_loader):
            if sample_count >= n_samples:
                break
            
            x = x.to(device)
            # Encode without masking
            encoded = model.encode(x)  # (batch, seq_len, d_model)
            # Global average pooling
            pooled = encoded.mean(dim=1)  # (batch, d_model)
            
            batch_size = x.shape[0]
            embeddings_list.append(pooled.cpu().numpy())
            directions_list.extend(y.cpu().numpy().flatten())
            
            # Extract HMM states if available
            if hmm_states is not None:
                start_idx = sample_count
                end_idx = min(sample_count + batch_size, len(hmm_states))
                if start_idx < len(hmm_states):
                    states_list.extend(hmm_states[start_idx:end_idx].tolist())
            
            sample_count += batch_size
    
    embeddings = np.concatenate(embeddings_list, axis=0)  # (n_samples, d_model)
    directions = np.array(directions_list[:len(embeddings)])
    
    # PCA projection
    pca = PCA(n_components=2)
    embeddings_pca = pca.fit_transform(embeddings)
    
    results = {
        'embeddings': embeddings,
        'embeddings_pca': embeddings_pca,
        'pca_explained_variance': pca.explained_variance_ratio_.sum(),
        'directions': directions
    }
    
    # Calculate separability metrics
    if hmm_states is not None and len(states_list) == len(embeddings):
        states_array = np.array(states_list[:len(embeddings)])
        results['hmm_states'] = states_array
        
        # Silhouette score
        if len(np.unique(states_array)) > 1:
            silhouette = silhouette_score(embeddings, states_array)
            results['silhouette_score'] = silhouette
        else:
            results['silhouette_score'] = None
    
    # Generate plots
    if output_dir:
        _plot_latent_space_pca(embeddings_pca, directions, 
                               results.get('hmm_states'), output_dir)
        
        # Generate target-colored plot (supervised latent space)
        _plot_latent_space_target(embeddings_pca, directions, output_dir)
        
        if use_tsne and HAS_TSNE:
            print("  Computing t-SNE (this may take a while)...")
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            embeddings_tsne = tsne.fit_transform(embeddings[:min(1000, len(embeddings))])
            results['embeddings_tsne'] = embeddings_tsne
            
            _plot_latent_space_tsne(embeddings_tsne, 
                                   directions[:len(embeddings_tsne)],
                                   results.get('hmm_states')[:len(embeddings_tsne)] if 'hmm_states' in results else None,
                                   output_dir)
            
            # Generate target-colored t-SNE plot
            _plot_latent_space_target(embeddings_tsne, directions[:len(embeddings_tsne)], 
                                     output_dir, use_tsne=True)
    
    return results


def _plot_latent_space_pca(
    embeddings_pca: np.ndarray,
    directions: np.ndarray,
    hmm_states: Optional[np.ndarray],
    output_dir: str
):
    """Plot PCA projection colored by HMM states or directions."""
    fig, axes = plt.subplots(1, 2 if hmm_states is not None else 1, figsize=(14, 6))
    if hmm_states is None:
        axes = [axes]
    
    # Plot 1: Colored by direction
    ax = axes[0]
    up_mask = directions == 1.0
    ax.scatter(embeddings_pca[up_mask, 0], embeddings_pca[up_mask, 1], 
              c='green', label='UP', alpha=0.6, s=20)
    ax.scatter(embeddings_pca[~up_mask, 0], embeddings_pca[~up_mask, 1], 
              c='red', label='DOWN', alpha=0.6, s=20)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('Latent Space PCA - Colored by Market Direction', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Colored by HMM states
    if hmm_states is not None:
        ax = axes[1]
        unique_states = sorted(np.unique(hmm_states))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_states)))
        
        for state, color in zip(unique_states, colors):
            mask = hmm_states == state
            ax.scatter(embeddings_pca[mask, 0], embeddings_pca[mask, 1],
                      c=[color], label=f'State {state}', alpha=0.6, s=20)
        
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title('Latent Space PCA - Colored by HMM States', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'latent_space_pca.svg'), format='svg', bbox_inches='tight')
    plt.close()


def _plot_latent_space_target(
    embeddings: np.ndarray,
    directions: np.ndarray,
    output_dir: str,
    use_tsne: bool = False
):
    """
    Plot latent space colored by real targets (Rouge = Baisse future, Vert = Hausse future).
    
    This visualization shows if the model has successfully separated geometrically
    the zones of market rise and fall.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Color by target: Rouge = Baisse (0), Vert = Hausse (1)
    up_mask = directions == 1.0
    ax.scatter(embeddings[up_mask, 0], embeddings[up_mask, 1], 
              c='green', label='Hausse future (Target=1)', alpha=0.6, s=30)
    ax.scatter(embeddings[~up_mask, 0], embeddings[~up_mask, 1], 
              c='red', label='Baisse future (Target=0)', alpha=0.6, s=30)
    
    method = 't-SNE' if use_tsne else 'PCA'
    ax.set_xlabel(f'{method} Component 1', fontsize=12)
    ax.set_ylabel(f'{method} Component 2', fontsize=12)
    ax.set_title(f'Supervised Latent Space ({method}) - Colored by Real Targets\n'
                f'Rouge=Baisse future, Vert=Hausse future', 
                fontweight='bold', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = 'mae_latent_space_target_tsne.png' if use_tsne else 'mae_latent_space_target.png'
    plt.savefig(os.path.join(output_dir, filename), format='png', dpi=150, bbox_inches='tight')
    plt.close()


def _plot_latent_space_tsne(
    embeddings_tsne: np.ndarray,
    directions: np.ndarray,
    hmm_states: Optional[np.ndarray],
    output_dir: str
):
    """Plot t-SNE projection colored by HMM states or directions."""
    fig, axes = plt.subplots(1, 2 if hmm_states is not None else 1, figsize=(14, 6))
    if hmm_states is None:
        axes = [axes]
    
    # Plot 1: Colored by direction
    ax = axes[0]
    up_mask = directions == 1.0
    ax.scatter(embeddings_tsne[up_mask, 0], embeddings_tsne[up_mask, 1], 
              c='green', label='UP', alpha=0.6, s=20)
    ax.scatter(embeddings_tsne[~up_mask, 0], embeddings_tsne[~up_mask, 1], 
              c='red', label='DOWN', alpha=0.6, s=20)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title('Latent Space t-SNE - Colored by Market Direction', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Colored by HMM states
    if hmm_states is not None:
        ax = axes[1]
        unique_states = sorted(np.unique(hmm_states))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_states)))
        
        for state, color in zip(unique_states, colors):
            mask = hmm_states == state
            ax.scatter(embeddings_tsne[mask, 0], embeddings_tsne[mask, 1],
                      c=[color], label=f'State {state}', alpha=0.6, s=20)
        
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.set_title('Latent Space t-SNE - Colored by HMM States', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'latent_space_tsne.svg'), format='svg', bbox_inches='tight')
    plt.close()


def plot_training_dynamics(history: Dict, output_dir: str):
    """Plot training dynamics: loss curves and gradient norms."""
    epochs = range(1, len(history['train_total']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Total Loss (Train vs Val)
    ax = axes[0, 0]
    ax.plot(epochs, history['train_total'], 'b-', label='Train', linewidth=2)
    ax.plot(epochs, history['val_total'], 'r-', label='Val', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.set_title('Total Loss (Train vs Val)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Reconstruction Loss (Train vs Val)
    ax = axes[0, 1]
    ax.plot(epochs, history['train_recon'], 'b-', label='Train', linewidth=0.7)
    ax.plot(epochs, history['val_recon'], 'r-', label='Val', linewidth=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Reconstruction Loss')
    ax.set_title('Reconstruction Loss (Train vs Val)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Auxiliary Loss (Train vs Val)
    ax = axes[1, 0]
    ax.plot(epochs, history['train_aux'], 'b-', label='Train', linewidth=0.7)
    ax.plot(epochs, history['val_aux'], 'r-', label='Val', linewidth=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Auxiliary Loss')
    ax.set_title('Auxiliary Loss (Train vs Val)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Gradient Norms
    ax = axes[1, 1]
    ax.plot(epochs, history['gradient_norms'], 'g-', linewidth=0.7)
    ax.axhline(y=10.0, color='r', linestyle='--', linewidth=0.7, label='Explosion Threshold')
    ax.axhline(y=1e-6, color='orange', linestyle='--', linewidth=0.7, label='Vanishing Threshold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Gradient Norm')
    ax.set_title('Gradient Norms', fontweight='bold')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_dynamics.svg'), format='svg', bbox_inches='tight')
    plt.close()


def run_mae_audit(
    mode: str = "real",
    segment_id: int = 0,
    encoder_path: Optional[str] = None,
    epochs: int = 20,
    device: str = "cpu",
    force_retrain: bool = False,
    use_tsne: bool = False,
    signal_strength: float = 0.3,
    n_samples: int = 10000
) -> Dict:
    """
    Run comprehensive MAE audit with all analyses.
    
    Args:
        mode: "synthetic" or "real"
        segment_id: WFO segment ID (for real mode)
        encoder_path: Path to pre-trained encoder (optional)
        epochs: Number of training epochs if training needed
        device: Device to use
        force_retrain: Force data retraining
        use_tsne: Also generate t-SNE plots (slower)
        signal_strength: Signal strength for synthetic data
        n_samples: Number of samples for synthetic data
    
    Returns:
        Dict with all audit results
    """
    # #region agent log
    debug_log_path = get_debug_log_path()
    try:
        with open(debug_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({
                'sessionId': 'debug-session',
                'runId': 'run1',
                'hypothesisId': 'F',
                'location': 'audit_pipeline.py:3033',
                'message': 'run_mae_audit entry',
                'data': {'mode': mode, 'segment_id': segment_id, 'device': device, 'epochs': epochs},
                'timestamp': int(time.time() * 1000)
            }) + '\n')
    except Exception:
        pass
    # #endregion
    print("=" * 70)
    print("MAE COMPREHENSIVE AUDIT")
    print("=" * 70)
    print(f"Mode: {mode}, Device: {device}, Epochs: {epochs}")
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_base = os.path.join("results", "mae_audit")
    os.makedirs(output_base, exist_ok=True)
    output_dir = os.path.join(output_base, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    device_torch = torch.device(device)
    results = {
        'output_dir': output_dir,
        'mode': mode,
        'config': {}
    }
    
    # Prepare data and model
    if mode == "synthetic":
        print("\n[1/6] Creating synthetic data...")
        features, close_prices, directions = create_synthetic_data(
            n_samples=n_samples,
            n_features=35,
            signal_strength=signal_strength
        )
        
        seq_len = 64
        sequences = []
        targets = []
        for i in range(len(features) - seq_len):
            sequences.append(features[i:i + seq_len])
            close_future = close_prices[i + seq_len]
            close_current = close_prices[i + seq_len - 1]
            targets.append(1.0 if close_future > close_current else 0.0)
        
        sequences = torch.tensor(np.array(sequences), dtype=torch.float32)
        targets = torch.tensor(np.array(targets), dtype=torch.float32).unsqueeze(1)
        
        split_idx = int(len(sequences) * 0.8)
        train_dataset = TensorDataset(sequences[:split_idx], targets[:split_idx])
        test_dataset = TensorDataset(sequences[split_idx:], targets[split_idx:])
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        input_dim = 35
        feature_names = [f"Feature_{i}" for i in range(input_dim)]
        hmm_states = None
        
        results['config'] = {
            'input_dim': input_dim,
            'd_model': 64,
            'n_heads': 4,
            'n_layers': 2,
            'epochs': epochs,
            'signal_strength': signal_strength
        }
        
    else:  # real mode
        print("\n[1/6] Loading real data...")
        # #region agent log
        debug_log_path = get_debug_log_path()
        try:
            with open(debug_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({
                    'sessionId': 'debug-session',
                    'runId': 'run1',
                    'hypothesisId': 'F',
                    'location': 'audit_pipeline.py:3123',
                    'message': 'Before load_or_create_data',
                    'data': {'segment_id': segment_id, 'force_retrain': force_retrain},
                    'timestamp': int(time.time() * 1000)
                }) + '\n')
        except Exception:
            pass
        # #endregion
        df = load_or_create_data(segment_id=segment_id, force_retrain=force_retrain)
        # #region agent log
        try:
            with open(debug_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({
                    'sessionId': 'debug-session',
                    'runId': 'run1',
                    'hypothesisId': 'F',
                    'location': 'audit_pipeline.py:3124',
                    'message': 'After load_or_create_data',
                    'data': {'df_shape': list(df.shape) if df is not None else None},
                    'timestamp': int(time.time() * 1000)
                }) + '\n')
        except Exception:
            pass
        # #endregion
        
        # Get HMM states if available
        try:
            hmm_states = get_dominant_state(df)
            print(f"  Found HMM states: {len(np.unique(hmm_states))} unique states")
        except ValueError:
            hmm_states = None
            print("  No HMM states found")
        
        # Create dataset
        seq_len = 64
        train_path = f"data/wfo/segment_{segment_id}/train.parquet" if os.path.exists(f"data/wfo/segment_{segment_id}/train.parquet") else "data/processed_data.parquet"
        # #region agent log
        try:
            with open(debug_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({
                    'sessionId': 'debug-session',
                    'runId': 'run1',
                    'hypothesisId': 'F',
                    'location': 'audit_pipeline.py:3137',
                    'message': 'Before CryptoDataset creation',
                    'data': {'train_path': train_path, 'seq_len': seq_len},
                    'timestamp': int(time.time() * 1000)
                }) + '\n')
        except Exception:
            pass
        # #endregion
        train_dataset = CryptoDataset(
            parquet_path=train_path,
            seq_len=seq_len,
            return_targets=True
        )
        # #region agent log
        try:
            with open(debug_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({
                    'sessionId': 'debug-session',
                    'runId': 'run1',
                    'hypothesisId': 'F',
                    'location': 'audit_pipeline.py:3141',
                    'message': 'After CryptoDataset creation',
                    'data': {'dataset_len': len(train_dataset) if train_dataset else None},
                    'timestamp': int(time.time() * 1000)
                }) + '\n')
        except Exception:
            pass
        # #endregion
        
        # Use subset for test
        test_size = min(2000, len(train_dataset) // 5)
        test_indices = list(range(len(train_dataset) - test_size, len(train_dataset)))
        train_indices = list(range(len(train_dataset) - test_size))
        
        test_dataset_subset = torch.utils.data.Subset(train_dataset, test_indices)
        train_dataset_subset = torch.utils.data.Subset(train_dataset, train_indices)
        
        # #region agent log
        debug_log_path = get_debug_log_path()
        try:
            with open(debug_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({
                    'sessionId': 'debug-session',
                    'runId': 'run1',
                    'hypothesisId': 'E',
                    'location': 'audit_pipeline.py:2862',
                    'message': 'Before DataLoader creation',
                    'data': {
                        'train_subset_len': len(train_dataset_subset),
                        'test_subset_len': len(test_dataset_subset)
                    },
                    'timestamp': int(time.time() * 1000)
                }) + '\n')
        except Exception:
            pass
        # #endregion
        train_loader = DataLoader(train_dataset_subset, batch_size=64, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_dataset_subset, batch_size=64, shuffle=False, num_workers=0)
        # #region agent log
        try:
            with open(debug_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({
                    'sessionId': 'debug-session',
                    'runId': 'run1',
                    'hypothesisId': 'E',
                    'location': 'audit_pipeline.py:2864',
                    'message': 'After DataLoader creation',
                    'data': {},
                    'timestamp': int(time.time() * 1000)
                }) + '\n')
        except Exception:
            pass
        # #endregion
        
        input_dim = train_dataset.n_features
        feature_names = train_dataset.get_feature_names()
        
        # Align HMM states with test dataset
        # HMM states are per row in dataframe, but sequences start at seq_len
        # Each sequence corresponds to the last timestep of the window
        if hmm_states is not None:
            # For each test sequence index, find corresponding HMM state
            # Sequence at index i in dataset corresponds to row (seq_len + i) in dataframe
            hmm_states_aligned = []
            for idx in test_indices:
                # The sequence ending at position idx corresponds to HMM state at position (seq_len - 1 + idx)
                # But since dataset starts at seq_len, we need to adjust
                df_idx = seq_len - 1 + idx
                if df_idx < len(hmm_states):
                    hmm_states_aligned.append(hmm_states[df_idx])
                else:
                    hmm_states_aligned.append(0)  # Default state
            hmm_states_aligned = np.array(hmm_states_aligned)
        else:
            hmm_states_aligned = None
        
        results['config'] = {
            'input_dim': input_dim,
            'd_model': 64,
            'n_heads': 4,
            'n_layers': 2,
            'epochs': epochs,
            'segment_id': segment_id
        }
        results['feature_names'] = feature_names
    
    # Load or train model
    print(f"\n[2/6] Preparing model...")
    # #region agent log
    try:
        with open(debug_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({
                'sessionId': 'debug-session',
                'runId': 'run1',
                'hypothesisId': 'F',
                'location': 'audit_pipeline.py:3197',
                'message': 'Before model preparation check',
                'data': {'encoder_path': encoder_path, 'encoder_exists': encoder_path and os.path.exists(encoder_path) if encoder_path else False},
                'timestamp': int(time.time() * 1000)
            }) + '\n')
    except Exception:
        pass
    # #endregion
    if encoder_path and os.path.exists(encoder_path):
        print(f"  Loading pre-trained encoder from {encoder_path}")
        model = CryptoMAE(input_dim=input_dim, d_model=64, n_heads=4, n_layers=2).to(device_torch)
        encoder_state = torch.load(encoder_path, map_location=device_torch, weights_only=False)
        model.embedding.load_state_dict(encoder_state['embedding'])
        model.pos_encoder.load_state_dict(encoder_state['pos_encoder'])
        model.encoder.load_state_dict(encoder_state['encoder'])
        
        # Create dummy history (no training)
        history = {
            'train_total': [],
            'train_recon': [],
            'train_aux': [],
            'val_total': [],
            'val_recon': [],
            'val_aux': [],
            'val_accuracy': [],
            'gradient_norms': []
        }
        results['encoder_path'] = encoder_path
    else:
        print(f"  Training new model...")
        # #region agent log
        debug_log_path = get_debug_log_path()
        try:
            with open(debug_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({
                    'sessionId': 'debug-session',
                    'runId': 'run1',
                    'hypothesisId': 'A',
                    'location': 'audit_pipeline.py:2919',
                    'message': 'Before train_supervised_mae call',
                    'data': {
                        'input_dim': input_dim,
                        'epochs': epochs,
                        'device': str(device_torch),
                        'train_loader_len': len(train_loader),
                        'val_loader_len': len(test_loader)
                    },
                    'timestamp': int(time.time() * 1000)
                }) + '\n')
        except Exception:
            pass
        # #endregion
        model, history = train_supervised_mae(
            train_loader=train_loader,
            val_loader=test_loader,
            input_dim=input_dim,
            device=device_torch,
            epochs=epochs
        )
        # #region agent log
        try:
            with open(debug_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({
                    'sessionId': 'debug-session',
                    'runId': 'run1',
                    'hypothesisId': 'A',
                    'location': 'audit_pipeline.py:2927',
                    'message': 'After train_supervised_mae call',
                    'data': {'history_keys': list(history.keys())},
                    'timestamp': int(time.time() * 1000)
                }) + '\n')
        except Exception:
            pass
        # #endregion
        results['encoder_path'] = None
    
    results['training_history'] = history
    
    # Training Dynamics Analysis
    print(f"\n[3/6] Analyzing training dynamics...")
    plot_training_dynamics(history, output_dir)
    
    # Detect overfitting
    if len(history['val_total']) > 5:
        train_trend = np.mean(history['train_total'][-5:]) - np.mean(history['train_total'][:5])
        val_trend = np.mean(history['val_total'][-5:]) - np.mean(history['val_total'][:5])
        overfitting = val_trend > 0 and train_trend < 0
        results['overfitting_detected'] = overfitting
    else:
        results['overfitting_detected'] = False
    
    # Gradient analysis
    if history['gradient_norms']:
        max_grad = max(history['gradient_norms'])
        min_grad = min(history['gradient_norms'])
        gradient_explosion = max_grad > 10.0
        gradient_vanishing = min_grad < 1e-6
        results['gradient_analysis'] = {
            'max_norm': max_grad,
            'min_norm': min_grad,
            'explosion_detected': gradient_explosion,
            'vanishing_detected': gradient_vanishing
        }
    
    # Classification Metrics Analysis (Supervised Head)
    print(f"\n[4/7] Analyzing classification metrics (supervised head)...")
    classification_results = analyze_classification_metrics(
        model=model,
        test_loader=test_loader,
        device=device_torch
    )
    results['classification'] = classification_results
    print(f"  Accuracy: {classification_results['accuracy']*100:.2f}%")
    print(f"  Precision: {classification_results['precision']*100:.2f}%")
    print(f"  Recall: {classification_results['recall']*100:.2f}%")
    print(f"  F1 Score: {classification_results['f1_score']*100:.2f}%")
    
    # Reconstruction Quality Analysis
    print(f"\n[5/7] Analyzing reconstruction quality...")
    recon_results = analyze_reconstruction_quality(
        model=model,
        test_loader=test_loader,
        feature_names=feature_names,
        device=device_torch,
        n_samples=5,
        output_dir=output_dir
    )
    results['reconstruction'] = recon_results
    
    # Latent Space Analysis
    print(f"\n[6/7] Analyzing latent space structure...")
    # For real mode, we need to collect HMM states aligned with test sequences
    hmm_states_for_latent = None
    if mode == "real" and hmm_states_aligned is not None:
        hmm_states_for_latent = hmm_states_aligned
    
    latent_results = analyze_latent_space(
        model=model,
        test_loader=test_loader,
        hmm_states=hmm_states_for_latent,
        device=device_torch,
        n_samples=2000,
        output_dir=output_dir,
        use_tsne=use_tsne
    )
    results['latent_space'] = latent_results
    
    # Generate Report
    print(f"\n[7/7] Generating report...")
    report_path = generate_mae_audit_report(results, output_dir)
    results['report_path'] = report_path
    
    print("\n" + "=" * 70)
    print("MAE AUDIT COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {output_dir}")
    print(f"Report: {report_path}")
    print("=" * 70)
    
    return results


def generate_mae_audit_report(results: Dict, output_dir: str) -> str:
    """Generate comprehensive text report."""
    report_path = os.path.join(output_dir, "mae_audit_report.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("MAE COMPREHENSIVE AUDIT REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Header
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Mode: {results['mode']}\n")
        f.write(f"Output Directory: {results['output_dir']}\n\n")
        
        config = results['config']
        f.write("CONFIGURATION\n")
        f.write("-" * 80 + "\n")
        for key, value in config.items():
            f.write(f"  {key}: {value}\n")
        if 'encoder_path' in results and results['encoder_path']:
            f.write(f"  encoder_path: {results['encoder_path']}\n")
        f.write("\n")

        # Feature list
        if 'feature_names' in results:
            feature_names = results['feature_names']
            f.write("INPUT FEATURES\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Total: {len(feature_names)} features\n\n")
            for i, name in enumerate(feature_names):
                f.write(f"  {i+1:2d}. {name}\n")
            f.write("\n")
        
        # Section 1: Training Dynamics
        f.write("=" * 80 + "\n")
        f.write("SECTION 1: TRAINING DYNAMICS\n")
        f.write("=" * 80 + "\n\n")
        
        history = results['training_history']
        if history['train_total']:
            f.write("Loss History:\n")
            f.write(f"  Final Train Loss: {history['train_total'][-1]:.6f}\n")
            f.write(f"  Final Val Loss: {history['val_total'][-1]:.6f}\n")
            f.write(f"  Final Train Recon: {history['train_recon'][-1]:.6f}\n")
            f.write(f"  Final Val Recon: {history['val_recon'][-1]:.6f}\n")
            if history['val_accuracy']:
                f.write(f"  Final Val Accuracy: {history['val_accuracy'][-1]*100:.2f}%\n")
            f.write("\n")
            
            # Overfitting detection
            if results.get('overfitting_detected'):
                f.write("  [!] OVERFITTING DETECTED: Val loss increasing while train loss decreasing\n")
            else:
                f.write("  [OK] No overfitting detected\n")
            f.write("\n")
            
            # Gradient analysis
            if 'gradient_analysis' in results:
                grad_analysis = results['gradient_analysis']
                f.write("Gradient Analysis:\n")
                f.write(f"  Max Gradient Norm: {grad_analysis['max_norm']:.6f}\n")
                f.write(f"  Min Gradient Norm: {grad_analysis['min_norm']:.6f}\n")
                if grad_analysis['explosion_detected']:
                    f.write("  [!] GRADIENT EXPLOSION DETECTED (norm > 10.0)\n")
                elif grad_analysis['vanishing_detected']:
                    f.write("  [!] GRADIENT VANISHING DETECTED (norm < 1e-6)\n")
                else:
                    f.write("  [OK] Gradient norms stable\n")
                f.write("\n")
        else:
            f.write("  [INFO] Model loaded from checkpoint, no training history available\n\n")
        
        # Section 2: Supervised Head Metrics (Classification)
        f.write("=" * 80 + "\n")
        f.write("SECTION 2: SUPERVISED HEAD METRICS\n")
        f.write("=" * 80 + "\n\n")
        
        if 'classification' in results:
            cls = results['classification']
            f.write("Classification Performance:\n")
            f.write(f"  Accuracy: {cls['accuracy']*100:.2f}%\n")
            f.write(f"  Precision: {cls['precision']*100:.2f}%\n")
            f.write(f"  Recall: {cls['recall']*100:.2f}%\n")
            f.write(f"  F1 Score: {cls['f1_score']*100:.2f}%\n")
            f.write(f"  Number of Samples: {cls['n_samples']}\n\n")
            
            # Confusion Matrix
            cm = cls['confusion_matrix']
            f.write("Confusion Matrix:\n")
            f.write(f"  True Positives (TP):  {cm['TP']}\n")
            f.write(f"  True Negatives (TN):  {cm['TN']}\n")
            f.write(f"  False Positives (FP): {cm['FP']}\n")
            f.write(f"  False Negatives (FN): {cm['FN']}\n\n")
            
            # Interpretation
            baseline = max(cm['TP'] + cm['FN'], cm['TN'] + cm['FP']) / cls['n_samples']
            gain = cls['accuracy'] - baseline
            f.write("Interpretation:\n")
            if cls['accuracy'] < 0.52:
                f.write("  [!] Accuracy ≈ 50%: Model hasn't learned (coin flip)\n")
            elif cls['accuracy'] < 0.60:
                f.write(f"  [OK] Accuracy {cls['accuracy']*100:.1f}%: Beginning of signal (Alpha detected)\n")
                f.write(f"  Gain over baseline: {gain*100:+.2f}%\n")
            else:
                f.write(f"  [!] Accuracy > 60%: Suspicious - possible overfitting or data leakage\n")
                f.write(f"  Gain over baseline: {gain*100:+.2f}%\n")
            f.write("\n")
        else:
            f.write("  [INFO] Classification metrics not available\n\n")
        
        # Section 3: Reconstruction Quality
        f.write("=" * 80 + "\n")
        f.write("SECTION 3: RECONSTRUCTION QUALITY\n")
        f.write("=" * 80 + "\n\n")
        
        recon = results['reconstruction']
        f.write(f"Overall MSE: {recon['overall_mse']:.6f}\n\n")
        
        f.write("Top 5 Best Features (Lowest MSE):\n")
        for i, feat in enumerate(recon['best_features'], 1):
            mse = recon['feature_mse'][feat]
            f.write(f"  {i}. {feat}: {mse:.6f}\n")
        f.write("\n")
        
        f.write("Top 5 Worst Features (Highest MSE):\n")
        for i, feat in enumerate(recon['worst_features'], 1):
            mse = recon['feature_mse'][feat]
            f.write(f"  {i}. {feat}: {mse:.6f}\n")
        f.write("\n")
        
        # Lazy model detection
        feature_mses = list(recon['feature_mse'].values())
        mse_std = np.std(feature_mses)
        if mse_std < 0.01:
            f.write("  [!] LAZY MODEL DETECTED: Very low variance in reconstruction errors\n")
            f.write("      Model may be predicting mean values instead of learning patterns\n")
        else:
            f.write(f"  [OK] Feature reconstruction variance: {mse_std:.6f}\n")
        f.write("\n")
        
        # Section 4: Latent Space Structure
        f.write("=" * 80 + "\n")
        f.write("SECTION 4: LATENT SPACE STRUCTURE\n")
        f.write("=" * 80 + "\n\n")
        
        latent = results['latent_space']
        f.write(f"Embedding Dimensions: {latent['embeddings'].shape}\n")
        f.write(f"PCA Explained Variance: {latent['pca_explained_variance']*100:.2f}%\n\n")
        
        if 'silhouette_score' in latent and latent['silhouette_score'] is not None:
            silhouette = latent['silhouette_score']
            f.write(f"Silhouette Score (HMM States): {silhouette:.4f}\n")
            if silhouette < 0.3:
                f.write("  [!] SPAGHETTI PLOT: Low separability (silhouette < 0.3)\n")
                f.write("      MAE may be myopic - embeddings don't capture regime structure\n")
            elif silhouette < 0.5:
                f.write("  [~] WEAK SEPARABILITY: Moderate clustering (silhouette < 0.5)\n")
            else:
                f.write("  [OK] GOOD SEPARABILITY: Clear clustering by HMM states\n")
            f.write("\n")
        else:
            f.write("  [INFO] HMM states not available for separability analysis\n\n")
        
        # Section 4: Recommendations
        f.write("=" * 80 + "\n")
        f.write("SECTION 4: RECOMMENDATIONS\n")
        f.write("=" * 80 + "\n\n")
        
        recommendations = []
        
        if results.get('overfitting_detected'):
            recommendations.append("- Consider early stopping or regularization to reduce overfitting")
        
        if 'gradient_analysis' in results:
            if results['gradient_analysis']['explosion_detected']:
                recommendations.append("- Reduce learning rate or add gradient clipping to prevent explosion")
            elif results['gradient_analysis']['vanishing_detected']:
                recommendations.append("- Increase learning rate or use residual connections to prevent vanishing")
        
        if recon['overall_mse'] > 1.0:
            recommendations.append("- High reconstruction error: consider increasing model capacity or training longer")
        
        if mse_std < 0.01:
            recommendations.append("- Lazy model detected: increase mask ratio or add regularization to force learning")
        
        if 'silhouette_score' in latent and latent['silhouette_score'] is not None:
            if latent['silhouette_score'] < 0.3:
                recommendations.append("- Low latent space separability: consider supervised pre-training or larger model")
        
        if not recommendations:
            recommendations.append("- Model appears healthy. Continue monitoring during full training.")
        
        for i, rec in enumerate(recommendations, 1):
            f.write(f"{i}. {rec}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    return report_path


# ============================================================================
# TQC COMPREHENSIVE AUDIT
# ============================================================================

def analyze_tqc_quantiles(
    model: TQC,
    test_env: BatchCryptoEnv,
    test_df: pd.DataFrame,
    n_steps: int = 1000
) -> Dict:
    """
    F.1 Inter-Quantile Range (IQR) vs HMM_Entropy
    
    Test: When HMM_Entropy is high (market uncertainty), 
    TQC's IQR should increase (model uncertainty).
    
    If not: TQC is overconfident (dangerous).
    
    Args:
        model: Loaded TQC model
        test_env: Test environment
        n_steps: Number of steps to collect data
    
    Returns:
        Dict with IQR analysis metrics
    """
    print(f"\n[F] Analyzing TQC Quantiles (IQR vs HMM_Entropy)...")
    
    device = next(model.policy.parameters()).device
    model.policy.eval()
    
    hmm_entropies = []
    iqrs = []
    q_values_mean = []
    q_values_std = []
    
    obs, info = test_env.gym_reset()
    done = False
    step_count = 0
    
    # Get starting index in dataframe (account for window_size)
    window_size = test_env.window_size if hasattr(test_env, 'window_size') else 64
    df_start_idx = window_size
    
    while not done and step_count < n_steps:
        # Get action from policy
        action, _ = model.predict(obs, deterministic=True)
        
        # Extract HMM_Entropy from dataframe
        hmm_entropy = None
        if test_df is not None and (df_start_idx + step_count) < len(test_df):
            row = test_df.iloc[df_start_idx + step_count]
            if 'HMM_Entropy' in row:
                hmm_entropy = row['HMM_Entropy']
        
        # Get quantiles from critic
        with torch.no_grad():
            obs_tensor = {
                'market': torch.tensor(obs['market'], dtype=torch.float32).unsqueeze(0).to(device),
                'position': torch.tensor(obs['position'], dtype=torch.float32).unsqueeze(0).to(device),
                'w_cost': torch.tensor(obs['w_cost'], dtype=torch.float32).unsqueeze(0).to(device)
            }
            action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0).to(device)
            
            # Get quantiles from critic
            # TQC critic returns (batch, n_critics, n_quantiles)
            quantiles = model.policy.critic(obs_tensor, action_tensor)
            # Shape: (batch, n_critics, n_quantiles)
            
            # Average across critics
            quantiles_flat = quantiles.mean(dim=1)  # (batch, n_quantiles)
            quantiles_flat = quantiles_flat.squeeze(0).cpu().numpy()  # (n_quantiles,)
            
            # Calculate IQR (Q90 - Q10)
            n_quantiles = len(quantiles_flat)
            q10_idx = int(0.10 * n_quantiles)
            q90_idx = int(0.90 * n_quantiles)
            iqr = quantiles_flat[q90_idx] - quantiles_flat[q10_idx]
            
            # Calculate mean and std of Q-values
            q_mean = quantiles_flat.mean()
            q_std = quantiles_flat.std()
            
            iqrs.append(iqr)
            q_values_mean.append(q_mean)
            q_values_std.append(q_std)
            
            # For HMM_Entropy, we'll need to extract from the data
            # For now, set to None and we'll fill it later from the dataframe
            if hmm_entropy is not None:
                hmm_entropies.append(hmm_entropy)
            else:
                hmm_entropies.append(None)
        
        # Step environment
        obs, reward, terminated, truncated, info = test_env.gym_step(action)
        done = terminated or truncated
        step_count += 1
    
    # Convert to numpy arrays
    iqrs = np.array(iqrs)
    q_values_mean = np.array(q_values_mean)
    q_values_std = np.array(q_values_std)
    
    # Calculate correlation if we have HMM_Entropy
    correlation = None
    overconfidence_flag = False
    
    if len([h for h in hmm_entropies if h is not None]) > 10:
        hmm_entropies_valid = np.array([h for h in hmm_entropies if h is not None])
        iqrs_valid = np.array([iqrs[i] for i, h in enumerate(hmm_entropies) if h is not None])
        
        if len(hmm_entropies_valid) > 10:
            correlation = np.corrcoef(hmm_entropies_valid, iqrs_valid)[0, 1]
            
            # Check for overconfidence: correlation should be positive (high entropy → high IQR)
            # If correlation < 0.3, TQC ignores market uncertainty
            if correlation < 0.3:
                overconfidence_flag = True
    
    results = {
        'iqrs': iqrs,
        'q_values_mean': q_values_mean,
        'q_values_std': q_values_std,
        'hmm_entropies': hmm_entropies,
        'correlation': correlation,
        'overconfidence_detected': overconfidence_flag,
        'n_steps': step_count
    }
    
    print(f"  Collected {step_count} steps")
    if correlation is not None:
        print(f"  Correlation (HMM_Entropy, IQR): {correlation:.4f}")
        if overconfidence_flag:
            print(f"  [!] OVERCONFIDENCE DETECTED: Correlation < 0.3")
        else:
            print(f"  [OK] Correlation >= 0.3 (TQC responds to market uncertainty)")
    
    return results


def analyze_tqc_calibration(
    model: TQC,
    test_env: BatchCryptoEnv,
    test_df: pd.DataFrame,
    n_steps: int = 1000
) -> Dict:
    """
    A.1 Reliability Diagram: Q-values vs Actual Returns
    A.2 Entropy Correlation: HMM_Entropy vs Q_value_std
    
    Args:
        model: Loaded TQC model
        test_env: Test environment
        n_steps: Number of steps to collect data
    
    Returns:
        Dict with calibration metrics
    """
    print(f"\n[A] Analyzing TQC Calibration...")
    
    device = next(model.policy.parameters()).device
    model.policy.eval()
    
    q_values = []
    actual_returns = []
    hmm_entropies = []
    q_value_stds = []
    actions = []
    
    obs, info = test_env.gym_reset()
    done = False
    step_count = 0
    initial_nav = info.get('nav', 10000.0)
    prev_nav = initial_nav
    
    # Get starting index in dataframe (account for window_size)
    window_size = test_env.window_size if hasattr(test_env, 'window_size') else 64
    df_start_idx = window_size
    
    while not done and step_count < n_steps:
        # Get action and Q-value
        action, _ = model.predict(obs, deterministic=True)
        
        # Get Q-value from critic
        with torch.no_grad():
            obs_tensor = {
                'market': torch.tensor(obs['market'], dtype=torch.float32).unsqueeze(0).to(device),
                'position': torch.tensor(obs['position'], dtype=torch.float32).unsqueeze(0).to(device),
                'w_cost': torch.tensor(obs['w_cost'], dtype=torch.float32).unsqueeze(0).to(device)
            }
            action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0).to(device)
            
            quantiles = model.policy.critic(obs_tensor, action_tensor)
            quantiles_flat = quantiles.mean(dim=1).squeeze(0).cpu().numpy()
            
            q_mean = quantiles_flat.mean()
            q_std = quantiles_flat.std()
            
            q_values.append(q_mean)
            q_value_stds.append(q_std)
        
        actions.append(action[0])
        
        # Extract HMM_Entropy from dataframe
        hmm_entropy = None
        if test_df is not None and (df_start_idx + step_count) < len(test_df):
            row = test_df.iloc[df_start_idx + step_count]
            if 'HMM_Entropy' in row:
                hmm_entropy = row['HMM_Entropy']
        hmm_entropies.append(hmm_entropy)
        
        # Step environment
        obs, reward, terminated, truncated, info = test_env.gym_step(action)
        done = terminated or truncated
        
        # Calculate actual return
        nav = info.get('nav', prev_nav)
        actual_return = (nav - prev_nav) / prev_nav if prev_nav > 0 else 0.0
        actual_returns.append(actual_return)
        prev_nav = nav
        
        step_count += 1
    
    # Convert to numpy
    q_values = np.array(q_values)
    actual_returns = np.array(actual_returns)
    q_value_stds = np.array(q_value_stds)
    hmm_entropies = np.array([h if h is not None else np.nan for h in hmm_entropies])
    
    # A.1 Reliability Diagram
    # Bin Q-values into deciles
    n_bins = 10
    q_bins = np.percentile(q_values, np.linspace(0, 100, n_bins + 1))
    bin_indices = np.digitize(q_values, q_bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    bin_means_q = []
    bin_means_return = []
    bin_win_rates = []
    
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_means_q.append(q_values[mask].mean())
            bin_means_return.append(actual_returns[mask].mean())
            bin_win_rates.append((actual_returns[mask] > 0).mean())
        else:
            bin_means_q.append(np.nan)
            bin_means_return.append(np.nan)
            bin_win_rates.append(np.nan)
    
    # Calculate ECE (Expected Calibration Error)
    ece = np.nanmean([abs(bin_means_q[i] - bin_means_return[i]) 
                      for i in range(n_bins) if not np.isnan(bin_means_q[i])])
    
    # Calculate Brier Score (simplified - treating as regression)
    brier_score = np.mean((q_values - actual_returns) ** 2)
    
    # A.2 Entropy Correlation
    entropy_correlation = None
    if not np.all(np.isnan(hmm_entropies)):
        valid_mask = ~np.isnan(hmm_entropies)
        if valid_mask.sum() > 10:
            entropy_correlation = np.corrcoef(
                hmm_entropies[valid_mask],
                q_value_stds[valid_mask]
            )[0, 1]
    
    results = {
        'q_values': q_values,
        'actual_returns': actual_returns,
        'q_value_stds': q_value_stds,
        'hmm_entropies': hmm_entropies,
        'bin_means_q': bin_means_q,
        'bin_means_return': bin_means_return,
        'bin_win_rates': bin_win_rates,
        'ece': ece,
        'brier_score': brier_score,
        'entropy_correlation': entropy_correlation,
        'n_steps': step_count
    }
    
    print(f"  Collected {step_count} steps")
    print(f"  ECE (Expected Calibration Error): {ece:.4f}")
    print(f"  Brier Score: {brier_score:.4f}")
    if entropy_correlation is not None:
        print(f"  Correlation (HMM_Entropy, Q_std): {entropy_correlation:.4f}")
    
    return results


def analyze_tqc_action_distribution(
    model: TQC,
    test_env: BatchCryptoEnv,
    n_steps: int = 1000
) -> Dict:
    """
    D.1 Action Histogram, Saturation, Entropy
    
    Args:
        model: Loaded TQC model
        test_env: Test environment
        n_steps: Number of steps to collect data
    
    Returns:
        Dict with action distribution metrics
    """
    print(f"\n[D] Analyzing TQC Action Distribution...")
    
    actions = []
    
    obs, info = test_env.gym_reset()
    done = False
    step_count = 0
    
    while not done and step_count < n_steps:
        action, _ = model.predict(obs, deterministic=True)
        actions.append(action[0])
        
        obs, reward, terminated, truncated, info = test_env.gym_step(action)
        done = terminated or truncated
        step_count += 1
    
    actions = np.array(actions)
    
    # Calculate saturation (% of actions at ±1.0 or close)
    saturation_threshold = 0.95
    saturated = np.abs(np.abs(actions) - 1.0) < (1.0 - saturation_threshold)
    saturation_pct = saturated.mean() * 100
    
    # Calculate action entropy (discretized)
    n_bins = 21  # Match action discretization
    hist, _ = np.histogram(actions, bins=n_bins, range=(-1.0, 1.0))
    hist = hist / hist.sum()  # Normalize
    hist = hist[hist > 0]  # Remove zeros
    action_entropy = -np.sum(hist * np.log(hist))
    
    # Statistics
    action_mean = actions.mean()
    action_std = actions.std()
    action_min = actions.min()
    action_max = actions.max()
    
    results = {
        'actions': actions,
        'saturation_pct': saturation_pct,
        'action_entropy': action_entropy,
        'action_mean': action_mean,
        'action_std': action_std,
        'action_min': action_min,
        'action_max': action_max,
        'n_steps': step_count
    }
    
    print(f"  Collected {step_count} steps")
    print(f"  Saturation (% actions near ±1.0): {saturation_pct:.1f}%")
    print(f"  Action Entropy: {action_entropy:.4f}")
    print(f"  Action Range: [{action_min:.3f}, {action_max:.3f}]")
    print(f"  Action Mean: {action_mean:.3f}, Std: {action_std:.3f}")
    
    if saturation_pct > 95:
        print(f"  [!] POLICY COLLAPSE DETECTED: >95% actions saturated")
    if action_entropy < 1.0:
        print(f"  [!] LOW EXPLORATION: Action entropy < 1.0")
    
    return results


def analyze_tqc_attribution(
    model: TQC,
    test_env: BatchCryptoEnv,
    n_samples: int = 1000
) -> Dict:
    """
    B.1 Gradient Attribution: ∇_features Q(s, a)
    B.2 Feature Importance Ranking
    
    Args:
        model: Loaded TQC model
        test_env: Test environment
        n_samples: Number of samples for attribution
    
    Returns:
        Dict with feature importance metrics
    """
    print(f"\n[B] Analyzing TQC Attribution...")
    
    device = next(model.policy.parameters()).device
    model.policy.train()  # Need gradients
    
    # Sample observations
    observations = []
    actions = []
    
    obs, info = test_env.gym_reset()
    done = False
    step_count = 0
    
    while not done and step_count < n_samples:
        action, _ = model.predict(obs, deterministic=True)
        observations.append({
            'market': obs['market'].copy(),
            'position': obs['position'].copy(),
            'w_cost': obs['w_cost'].copy()
        })
        actions.append(action[0])
        
        obs, reward, terminated, truncated, info = test_env.gym_step(action)
        done = terminated or truncated
        step_count += 1
    
    # Calculate gradients for a subset (gradient computation is expensive)
    sample_size = min(100, len(observations))
    indices = np.random.choice(len(observations), sample_size, replace=False)
    
    feature_gradients = []
    
    for idx in indices:
        obs_sample = observations[idx]
        action_sample = actions[idx]

        # Create tensors with requires_grad
        market_tensor = torch.tensor(obs_sample['market'], dtype=torch.float32).unsqueeze(0).to(device)
        market_tensor.requires_grad_(True)
        position_tensor = torch.tensor(obs_sample['position'], dtype=torch.float32).unsqueeze(0).to(device)
        position_tensor.requires_grad_(True)
        w_cost_tensor = torch.tensor(obs_sample['w_cost'], dtype=torch.float32).unsqueeze(0).to(device)
        w_cost_tensor.requires_grad_(True)

        obs_tensor = {
            'market': market_tensor,
            'position': position_tensor,
            'w_cost': w_cost_tensor
        }
        action_tensor = torch.tensor([action_sample], dtype=torch.float32).unsqueeze(0).to(device)

        try:
            # Get Q-value - try different critic interfaces
            if hasattr(model.policy.critic, 'forward'):
                quantiles = model.policy.critic(obs_tensor, action_tensor)
            else:
                # Fallback: use actor to get action value
                quantiles = model.policy.critic.q1_forward(obs_tensor, action_tensor)

            q_value = quantiles.mean()  # Average across critics and quantiles

            # Zero gradients before backward
            if market_tensor.grad is not None:
                market_tensor.grad.zero_()
            if position_tensor.grad is not None:
                position_tensor.grad.zero_()
            if w_cost_tensor.grad is not None:
                w_cost_tensor.grad.zero_()

            # Compute gradients
            q_value.backward(retain_graph=True)

            # Extract gradients with None check
            market_grad = market_tensor.grad.abs().mean().item() if market_tensor.grad is not None else 0.0
            position_grad = position_tensor.grad.abs().mean().item() if position_tensor.grad is not None else 0.0
            w_cost_grad = w_cost_tensor.grad.abs().mean().item() if w_cost_tensor.grad is not None else 0.0

            feature_gradients.append({
                'market': market_grad,
                'position': position_grad,
                'w_cost': w_cost_grad
            })
        except Exception as e:
            # Skip this sample if gradient computation fails
            if len(feature_gradients) == 0:
                print(f"  [WARNING] Gradient computation failed: {e}")
            continue
    
    # Aggregate gradients (with fallback for empty list)
    if len(feature_gradients) == 0:
        print(f"  [WARNING] No gradients computed - critic may not support gradient flow")
        avg_gradients = {'market': 0.0, 'position': 0.0, 'w_cost': 0.0}
        feature_ranking = [('market', 0.0), ('position', 0.0), ('w_cost', 0.0)]
    else:
        avg_gradients = {
            'market': np.mean([g['market'] for g in feature_gradients]),
            'position': np.mean([g['position'] for g in feature_gradients]),
            'w_cost': np.mean([g['w_cost'] for g in feature_gradients])
        }
        feature_ranking = sorted(avg_gradients.items(), key=lambda x: x[1], reverse=True)

    results = {
        'feature_gradients': avg_gradients,
        'feature_ranking': feature_ranking,
        'n_samples': len(feature_gradients)
    }
    
    print(f"  Analyzed {sample_size} samples")
    print(f"  Feature Importance Ranking:")
    for i, (feature, importance) in enumerate(feature_ranking, 1):
        print(f"    {i}. {feature}: {importance:.6f}")
    
    return results


def analyze_tqc_value_add(
    model: TQC,
    test_env: BatchCryptoEnv,
    test_df: pd.DataFrame,
    baseline_strategy: str = "naive_mae"
) -> Dict:
    """
    C.1 Baseline Comparison: Naive MAE vs TQC vs Oracle
    C.2 Regime-Specific Performance
    
    Args:
        model: Loaded TQC model
        test_env: Test environment
        test_df: Test dataframe with HMM features
        baseline_strategy: Baseline strategy to use
    
    Returns:
        Dict with value add metrics
    """
    print(f"\n[C] Analyzing TQC Value Add...")
    
    # C.1 Run backtest with TQC
    tqc_returns = []
    tqc_actions = []
    tqc_hmm_states = []
    
    obs, info = test_env.gym_reset()
    done = False
    step_count = 0
    initial_nav = info.get('nav', 10000.0)
    prev_nav = initial_nav
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        tqc_actions.append(action[0])
        
        # Get HMM state (dominant probability)
        if 'hmm_entropy' in obs or test_df is not None:
            # Extract from dataframe if available
            if test_df is not None and step_count < len(test_df):
                hmm_probs = [
                    test_df.iloc[step_count].get('HMM_Prob_0', 0),
                    test_df.iloc[step_count].get('HMM_Prob_1', 0),
                    test_df.iloc[step_count].get('HMM_Prob_2', 0),
                    test_df.iloc[step_count].get('HMM_Prob_3', 0)
                ]
                dominant_state = np.argmax(hmm_probs)
                tqc_hmm_states.append(dominant_state)
            else:
                tqc_hmm_states.append(None)
        else:
            tqc_hmm_states.append(None)
        
        obs, reward, terminated, truncated, info = test_env.gym_step(action)
        done = terminated or truncated
        
        nav = info.get('nav', prev_nav)
        ret = (nav - prev_nav) / prev_nav if prev_nav > 0 else 0.0
        tqc_returns.append(ret)
        prev_nav = nav
        step_count += 1
    
    tqc_returns = np.array(tqc_returns)
    
    # Calculate TQC metrics
    tqc_sharpe = calculate_sharpe_ratio(tqc_returns) if len(tqc_returns) > 1 else 0.0
    tqc_total_return = (prev_nav / initial_nav - 1.0) * 100
    tqc_max_dd = calculate_max_drawdown(tqc_returns)
    tqc_win_rate = (tqc_returns > 0).mean() * 100
    
    # C.2 Regime-Specific Performance
    regime_metrics = {}
    if any(s is not None for s in tqc_hmm_states):
        for state in range(4):
            state_mask = np.array([s == state for s in tqc_hmm_states])
            if state_mask.sum() > 10:
                state_returns = tqc_returns[state_mask]
                regime_metrics[f'state_{state}'] = {
                    'sharpe': calculate_sharpe_ratio(state_returns) if len(state_returns) > 1 else 0.0,
                    'win_rate': (state_returns > 0).mean() * 100,
                    'avg_return': state_returns.mean() * 100,
                    'n_trades': state_mask.sum()
                }
    
    results = {
        'tqc_sharpe': tqc_sharpe,
        'tqc_total_return': tqc_total_return,
        'tqc_max_dd': tqc_max_dd,
        'tqc_win_rate': tqc_win_rate,
        'tqc_returns': tqc_returns,
        'regime_metrics': regime_metrics,
        'n_steps': step_count
    }
    
    print(f"  TQC Performance:")
    print(f"    Sharpe Ratio: {tqc_sharpe:.4f}")
    print(f"    Total Return: {tqc_total_return:.2f}%")
    print(f"    Max Drawdown: {tqc_max_dd:.2f}%")
    print(f"    Win Rate: {tqc_win_rate:.1f}%")
    
    if regime_metrics:
        print(f"  Regime-Specific Performance:")
        for state, metrics in regime_metrics.items():
            print(f"    {state}: Sharpe={metrics['sharpe']:.4f}, WinRate={metrics['win_rate']:.1f}%")
    
    return results


def analyze_tqc_convergence(
    tensorboard_log_dir: str
) -> Dict:
    """
    E.1 Training Curves Analysis from TensorBoard
    
    Args:
        tensorboard_log_dir: Path to TensorBoard log directory
    
    Returns:
        Dict with convergence analysis
    """
    print(f"\n[E] Analyzing TQC Convergence...")
    
    # Try to parse TensorBoard events
    # This is a simplified version - full implementation would use tensorboard library
    results = {
        'log_dir': tensorboard_log_dir,
        'available': os.path.exists(tensorboard_log_dir),
        'note': 'Full TensorBoard parsing requires tensorboard library'
    }
    
    if os.path.exists(tensorboard_log_dir):
        print(f"  TensorBoard log directory found: {tensorboard_log_dir}")
        print(f"  [INFO] Full parsing requires tensorboard library (not implemented here)")
    else:
        print(f"  [WARNING] TensorBoard log directory not found: {tensorboard_log_dir}")
    
    return results


def run_tqc_audit(
    segment_id: int = 0,
    tqc_path: Optional[str] = None,
    test_data_path: Optional[str] = None,
    encoder_path: Optional[str] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    n_samples: int = 1000,
    output_dir: Optional[str] = None
) -> Dict:
    """
    Run comprehensive TQC audit with all analyses (A-F).
    
    Args:
        segment_id: WFO segment ID (used to auto-detect paths if not provided)
        tqc_path: Path to TQC model (auto-detected from segment if None)
        test_data_path: Path to test data parquet (auto-detected if None)
        encoder_path: Path to MAE encoder (auto-detected if None)
        device: Device to use
        n_samples: Number of samples for attribution analysis
        output_dir: Output directory (auto-generated if None)
    
    Returns:
        Dict with all audit results
    """
    if not HAS_TQC:
        raise ImportError("sb3_contrib not available. Cannot run TQC audit.")
    
    print("=" * 70)
    print("TQC COMPREHENSIVE AUDIT")
    print("=" * 70)
    print(f"Segment: {segment_id}, Device: {device}")
    
    # Auto-detect paths from segment_id
    if tqc_path is None:
        tqc_path = f"weights/wfo/segment_{segment_id}/tqc.zip"
    if test_data_path is None:
        test_data_path = f"data/processed/wfo/segment_{segment_id}/test.parquet"
    if encoder_path is None:
        encoder_path = f"weights/wfo/segment_{segment_id}/encoder.pth"
    
    # Validate paths exist
    for name, path in [("TQC", tqc_path), ("Test Data", test_data_path)]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} not found: {path}")
    
    # Create output directory
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_base = os.path.join("results", "tqc_audit")
        os.makedirs(output_base, exist_ok=True)
        output_dir = os.path.join(output_base, timestamp)
        os.makedirs(output_dir, exist_ok=True)
    
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}")
    
    results = {
        'output_dir': output_dir,
        'segment_id': segment_id,
        'tqc_path': tqc_path,
        'test_data_path': test_data_path,
        'config': {
            'device': device,
            'n_samples': n_samples
        }
    }
    
    # Load model
    print(f"\n[1/7] Loading TQC model...")
    model = TQC.load(tqc_path, device=device)
    print(f"  Model loaded from: {tqc_path}")
    
    # Load test data
    print(f"\n[2/7] Loading test data...")
    test_df = pd.read_parquet(test_data_path)
    print(f"  Test data shape: {test_df.shape}")
    
    # Create test environment
    print(f"\n[3/7] Creating test environment...")
    test_env = BatchCryptoEnv(
        parquet_path=test_data_path,
        n_envs=1,
        device=device,
        window_size=64,  # Default window size
        episode_length=len(test_df) - 64 - 1,
        initial_balance=10000.0,
        commission=0.001,
        slippage=0.0001,
        price_column='BTC_Close',
        random_start=False
    )
    test_env.set_training_mode(False)  # Disable observation noise
    
    # Run all analyses
    print(f"\n[4/7] Running analyses...")
    
    # Section F: Quantiles
    results['quantiles'] = analyze_tqc_quantiles(model, test_env, test_df, n_steps=n_samples)
    
    # Section A: Calibration
    results['calibration'] = analyze_tqc_calibration(model, test_env, test_df, n_steps=n_samples)
    
    # Section D: Action Distribution
    results['action_distribution'] = analyze_tqc_action_distribution(model, test_env, n_steps=n_samples)
    
    # Section C: Value Add
    results['value_add'] = analyze_tqc_value_add(model, test_env, test_df, baseline_strategy="naive_mae")
    
    # Section B: Attribution
    results['attribution'] = analyze_tqc_attribution(model, test_env, n_samples=min(n_samples, 100))
    
    # Section E: Convergence (requires TensorBoard log)
    tensorboard_log = f"logs/tensorboard_tqc/segment_{segment_id}"
    results['convergence'] = analyze_tqc_convergence(tensorboard_log)
    
    # Generate plots
    print(f"\n[5/7] Generating plots...")
    _generate_tqc_plots(results, plots_dir)
    
    # Generate report
    print(f"\n[6/7] Generating report...")
    report_path = generate_tqc_audit_report(results, output_dir)
    
    # Save metrics
    print(f"\n[7/7] Saving metrics...")
    metrics_path = os.path.join(output_dir, "metrics.json")
    # Convert numpy arrays to lists for JSON serialization
    metrics_serializable = _serialize_results(results)
    with open(metrics_path, 'w') as f:
        json.dump(metrics_serializable, f, indent=2)
    
    print("\n" + "=" * 70)
    print("TQC AUDIT COMPLETE")
    print("=" * 70)
    print(f"Report: {report_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Plots: {plots_dir}")
    
    return results


def _serialize_results(results: Dict) -> Dict:
    """Convert numpy arrays to lists for JSON serialization."""
    def _convert(value):
        if isinstance(value, dict):
            return {k: _convert(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [_convert(item) for item in value]
        elif isinstance(value, tuple):
            return tuple(_convert(item) for item in value)
        elif isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, (np.integer, np.int64, np.int32)):
            return int(value)
        elif isinstance(value, (np.floating, np.float64, np.float32)):
            return float(value)
        elif isinstance(value, (np.bool_,)):
            return bool(value)
        else:
            return value
    return _convert(results)


def _generate_tqc_plots(results: Dict, plots_dir: str):
    """Generate all plots for TQC audit."""
    # F. Quantiles: IQR vs HMM_Entropy
    if 'quantiles' in results and results['quantiles'].get('correlation') is not None:
        plt.figure(figsize=(10, 6))
        hmm_entropies = results['quantiles']['hmm_entropies']
        iqrs = results['quantiles']['iqrs']
        valid_mask = np.array([h is not None for h in hmm_entropies])
        if valid_mask.sum() > 0:
            plt.scatter([h for h, v in zip(hmm_entropies, valid_mask) if v],
                       [iqrs[i] for i, v in enumerate(valid_mask) if v],
                       alpha=0.5)
            plt.xlabel('HMM_Entropy')
            plt.ylabel('IQR (Q90 - Q10)')
            plt.title('TQC Quantiles: IQR vs HMM_Entropy')
            plt.grid(True, alpha=0.3)
            corr = results['quantiles']['correlation']
            plt.text(0.05, 0.95, f'Correlation: {corr:.4f}', 
                    transform=plt.gca().transAxes, verticalalignment='top')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "quantiles_iqr_vs_hmm_entropy.png"), dpi=150)
            plt.close()
    
    # A. Calibration: Reliability Diagram
    if 'calibration' in results:
        plt.figure(figsize=(10, 6))
        bin_means_q = results['calibration']['bin_means_q']
        bin_means_return = results['calibration']['bin_means_return']
        valid_bins = [i for i, q in enumerate(bin_means_q) if not np.isnan(q)]
        if valid_bins:
            valid_q_values = [bin_means_q[i] for i in valid_bins]
            valid_return_values = [bin_means_return[i] for i in valid_bins]
            plt.plot(valid_q_values, valid_return_values, 'o-', label='Actual')
            plt.plot([min(valid_q_values), max(valid_q_values)],
                    [min(valid_q_values), max(valid_q_values)],
                    '--', label='Perfect Calibration')
            plt.xlabel('Predicted Q-value (Binned)')
            plt.ylabel('Actual Return')
            plt.title('TQC Calibration: Reliability Diagram')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "calibration_reliability_diagram.png"), dpi=150)
            plt.close()
    
    # D. Action Distribution: Histogram
    if 'action_distribution' in results:
        plt.figure(figsize=(10, 6))
        actions = results['action_distribution']['actions']
        plt.hist(actions, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Action Value')
        plt.ylabel('Frequency')
        plt.title('TQC Action Distribution')
        plt.axvline(0, color='red', linestyle='--', alpha=0.5, label='Zero')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "action_distribution_histogram.png"), dpi=150)
        plt.close()


def generate_tqc_audit_report(
    results: Dict,
    output_dir: str
) -> str:
    """
    Generate comprehensive TQC audit report (Markdown + plots).
    
    Sections:
    - A. Calibration (ECE, Brier, Entropy Correlation)
    - B. Attribution (Feature Importance, HMM Sensitivity)
    - C. Value Add (Sharpe Delta, Regime Performance)
    - D. Action Distribution (Saturation, Entropy)
    - E. Convergence (Training Curves)
    - F. Quantiles (IQR vs HMM_Entropy correlation)
    """
    report_path = os.path.join(output_dir, "report.md")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# TQC Comprehensive Audit Report\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Segment ID**: {results['segment_id']}\n\n")
        f.write("=" * 80 + "\n\n")
        
        # Section F: Quantiles
        f.write("## F. Quantiles Analysis (IQR vs HMM_Entropy)\n\n")
        if 'quantiles' in results:
            quantiles = results['quantiles']
            f.write(f"- **Correlation**: {quantiles.get('correlation', 'N/A')}\n")
            f.write(f"- **Overconfidence Detected**: {quantiles.get('overconfidence_detected', False)}\n")
            f.write(f"- **Steps Analyzed**: {quantiles.get('n_steps', 0)}\n\n")
            if quantiles.get('overconfidence_detected'):
                f.write("⚠️ **WARNING**: TQC shows overconfidence (correlation < 0.3).\n")
                f.write("The model does not increase uncertainty when market uncertainty is high.\n\n")
        
        # Section A: Calibration
        f.write("## A. Calibration Analysis\n\n")
        if 'calibration' in results:
            cal = results['calibration']
            f.write(f"- **ECE (Expected Calibration Error)**: {cal.get('ece', 'N/A'):.4f}\n")
            f.write(f"- **Brier Score**: {cal.get('brier_score', 'N/A'):.4f}\n")
            f.write(f"- **Entropy Correlation**: {cal.get('entropy_correlation', 'N/A')}\n")
            f.write(f"- **Steps Analyzed**: {cal.get('n_steps', 0)}\n\n")
        
        # Section B: Attribution
        f.write("## B. Attribution Analysis\n\n")
        if 'attribution' in results:
            attr = results['attribution']
            f.write("**Feature Importance Ranking**:\n\n")
            for i, (feature, importance) in enumerate(attr.get('feature_ranking', []), 1):
                f.write(f"{i}. {feature}: {importance:.6f}\n")
            f.write(f"\n**Samples Analyzed**: {attr.get('n_samples', 0)}\n\n")
        
        # Section C: Value Add
        f.write("## C. Value Add Analysis\n\n")
        if 'value_add' in results:
            va = results['value_add']
            f.write("**TQC Performance**:\n\n")
            f.write(f"- **Sharpe Ratio**: {va.get('tqc_sharpe', 'N/A'):.4f}\n")
            f.write(f"- **Total Return**: {va.get('tqc_total_return', 'N/A'):.2f}%\n")
            f.write(f"- **Max Drawdown**: {va.get('tqc_max_dd', 'N/A'):.2f}%\n")
            f.write(f"- **Win Rate**: {va.get('tqc_win_rate', 'N/A'):.1f}%\n\n")
            
            if 'regime_metrics' in va and va['regime_metrics']:
                f.write("**Regime-Specific Performance**:\n\n")
                for state, metrics in va['regime_metrics'].items():
                    f.write(f"- **{state}**: Sharpe={metrics['sharpe']:.4f}, WinRate={metrics['win_rate']:.1f}%\n")
                f.write("\n")
        
        # Section D: Action Distribution
        f.write("## D. Action Distribution Analysis\n\n")
        if 'action_distribution' in results:
            ad = results['action_distribution']
            f.write(f"- **Saturation %**: {ad.get('saturation_pct', 'N/A'):.1f}%\n")
            f.write(f"- **Action Entropy**: {ad.get('action_entropy', 'N/A'):.4f}\n")
            f.write(f"- **Action Range**: [{ad.get('action_min', 'N/A'):.3f}, {ad.get('action_max', 'N/A'):.3f}]\n")
            f.write(f"- **Action Mean**: {ad.get('action_mean', 'N/A'):.3f}, Std: {ad.get('action_std', 'N/A'):.3f}\n\n")
            if ad.get('saturation_pct', 0) > 95:
                f.write("⚠️ **WARNING**: Policy collapse detected (>95% actions saturated).\n\n")
        
        # Section E: Convergence
        f.write("## E. Convergence Analysis\n\n")
        if 'convergence' in results:
            conv = results['convergence']
            f.write(f"- **TensorBoard Log**: {conv.get('log_dir', 'N/A')}\n")
            f.write(f"- **Available**: {conv.get('available', False)}\n")
            f.write(f"- **Note**: {conv.get('note', 'N/A')}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    return report_path


# ============================================================================
# FiLM AUDIT
# ============================================================================

def audit_film_mechanics(
    segment_id: int = 0,
    tqc_path: Optional[str] = None,
    encoder_path: Optional[str] = None,
    test_data_path: Optional[str] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_dir: Optional[str] = None,
    n_samples: int = 1000
) -> Dict:
    """
    Audit FiLM (Feature-wise Linear Modulation) mechanics with 3 critical tests:
    
    A. Static Analysis (Weights): Inspect param_generator weights
    B. Dynamic Analysis (Activations): Capture Gamma and Beta for real data
    C. Sensitivity Test (Crash Test): Test with pure HMM contexts
    
    Args:
        segment_id: WFO segment ID (used to auto-detect paths if not provided)
        tqc_path: Path to TQC model (auto-detected from segment if None)
        encoder_path: Path to MAE encoder (auto-detected if None)
        test_data_path: Path to test data parquet (auto-detected if None)
        device: Device to use
        output_dir: Output directory (auto-generated if None)
        n_samples: Number of samples for dynamic analysis
    
    Returns:
        Dict with all audit results
    """
    if not HAS_FILM:
        raise ImportError("FiLM imports not available. Cannot run FiLM audit.")
    if not HAS_TQC:
        raise ImportError("sb3_contrib not available. Cannot load TQC model.")
    
    print("=" * 70)
    print("FiLM MECHANICS AUDIT")
    print("=" * 70)
    print(f"Segment: {segment_id}, Device: {device}")
    
    # Auto-detect paths from segment_id
    if tqc_path is None:
        tqc_path = f"weights/wfo/segment_{segment_id}/tqc.zip"
    if encoder_path is None:
        encoder_path = f"weights/wfo/segment_{segment_id}/encoder.pth"
    if test_data_path is None:
        test_data_path = f"data/processed/wfo/segment_{segment_id}/test.parquet"
    
    # Validate paths exist
    for name, path in [("TQC", tqc_path), ("Test Data", test_data_path)]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} not found: {path}")
    
    # Create output directory
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_base = os.path.join("results", "film_audit")
        os.makedirs(output_base, exist_ok=True)
        output_dir = os.path.join(output_base, timestamp)
        os.makedirs(output_dir, exist_ok=True)
    
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}")
    
    results = {
        'output_dir': output_dir,
        'segment_id': segment_id,
        'tqc_path': tqc_path,
        'test_data_path': test_data_path,
        'config': {
            'device': device,
            'n_samples': n_samples
        }
    }
    
    # Load TQC model
    print(f"\n[1/4] Loading TQC model...")
    model = TQC.load(tqc_path, device=device)
    print(f"  Model loaded from: {tqc_path}")
    
    # Extract FoundationFeatureExtractor
    feature_extractor = model.policy.features_extractor
    if not isinstance(feature_extractor, FoundationFeatureExtractor):
        raise TypeError(f"Expected FoundationFeatureExtractor, got {type(feature_extractor)}")
    
    if feature_extractor.film_layer is None:
        raise ValueError("FiLM layer is None! Model does not use FiLM.")
    
    film_layer = feature_extractor.film_layer
    print(f"  FiLM layer found: feature_dim={film_layer.feature_dim}, context_dim={film_layer.context_dim}")
    
    # Load test data
    print(f"\n[2/4] Loading test data...")
    test_df = pd.read_parquet(test_data_path)
    print(f"  Test data shape: {test_df.shape}")
    
    # Create test environment for data loading
    test_env = BatchCryptoEnv(
        parquet_path=test_data_path,
        n_envs=1,
        device=device,
        window_size=64,
        episode_length=len(test_df) - 64 - 1,
        initial_balance=10000.0,
        commission=0.001,
        slippage=0.0001,
        price_column='BTC_Close',
        random_start=False
    )
    test_env.set_training_mode(False)
    
    # ========================================================================
    # A. STATIC ANALYSIS (WEIGHTS)
    # ========================================================================
    print(f"\n[3/4] A. Static Analysis (Weights)...")
    
    # Get the MLP that generates gamma and beta
    param_generator = film_layer.mlp
    
    # Extract weights and biases from the last layer (output layer)
    last_layer = param_generator[-1]  # Last Linear layer
    weight = last_layer.weight.data  # (2 * feature_dim, hidden_dim)
    bias = last_layer.bias.data  # (2 * feature_dim,)
    
    # Split into gamma and beta parts
    feature_dim = film_layer.feature_dim
    gamma_weight = weight[:feature_dim, :]  # (feature_dim, hidden_dim)
    beta_weight = weight[feature_dim:, :]   # (feature_dim, hidden_dim)
    gamma_bias = bias[:feature_dim]         # (feature_dim,)
    beta_bias = bias[feature_dim:]          # (feature_dim,)
    
    # Statistics
    gamma_weight_mean = gamma_weight.mean().item()
    gamma_weight_std = gamma_weight.std().item()
    gamma_bias_mean = gamma_bias.mean().item()
    gamma_bias_std = gamma_bias.std().item()
    
    beta_weight_mean = beta_weight.mean().item()
    beta_weight_std = beta_weight.std().item()
    beta_bias_mean = beta_bias.mean().item()
    beta_bias_std = beta_bias.std().item()
    
    # Check if initialization is still visible (gamma_bias should be close to 1.0 initially)
    init_gamma_visible = abs(gamma_bias_mean - 1.0) < 0.5  # Allow some drift
    init_beta_visible = abs(beta_bias_mean - 0.0) < 0.5
    
    print(f"  Gamma weights: mean={gamma_weight_mean:.6f}, std={gamma_weight_std:.6f}")
    print(f"  Gamma bias: mean={gamma_bias_mean:.6f}, std={gamma_bias_std:.6f}")
    print(f"    → Initialization visible (≈1.0): {init_gamma_visible}")
    print(f"  Beta weights: mean={beta_weight_mean:.6f}, std={beta_weight_std:.6f}")
    print(f"  Beta bias: mean={beta_bias_mean:.6f}, std={beta_bias_std:.6f}")
    print(f"    → Initialization visible (≈0.0): {init_beta_visible}")
    
    # Check for weight explosion
    weight_exploded = gamma_weight_std > 10.0 or beta_weight_std > 10.0
    if weight_exploded:
        print(f"  ⚠️ WARNING: Weight explosion detected!")
    
    results['static_analysis'] = {
        'gamma_weight_mean': gamma_weight_mean,
        'gamma_weight_std': gamma_weight_std,
        'gamma_bias_mean': gamma_bias_mean,
        'gamma_bias_std': gamma_bias_std,
        'beta_weight_mean': beta_weight_mean,
        'beta_weight_std': beta_weight_std,
        'beta_bias_mean': beta_bias_mean,
        'beta_bias_std': beta_bias_std,
        'init_gamma_visible': init_gamma_visible,
        'init_beta_visible': init_beta_visible,
        'weight_exploded': weight_exploded
    }
    
    # ========================================================================
    # B. DYNAMIC ANALYSIS (ACTIVATIONS)
    # ========================================================================
    print(f"\n[4/4] B. Dynamic Analysis (Activations)...")
    
    # Collect samples from test environment
    obs_list = []
    n_collected = 0
    obs = test_env.reset()
    
    while n_collected < n_samples:
        # Extract market observations (BatchCryptoEnv returns dict with numpy arrays)
        if isinstance(obs, dict):
            market_obs_np = obs["market"]  # (n_envs, seq_len, n_features) or (seq_len, n_features)
        else:
            raise ValueError(f"Unexpected observation type: {type(obs)}")
        
        # Convert to torch tensor
        if isinstance(market_obs_np, np.ndarray):
            market_obs = torch.from_numpy(market_obs_np).to(device).float()
        else:
            market_obs = market_obs_np.to(device).float()
        
        # Handle both single env and batch cases
        if market_obs.dim() == 2:
            # Single env: (seq_len, n_features) -> (1, seq_len, n_features)
            market_obs = market_obs.unsqueeze(0)
        
        obs_list.append(market_obs)
        n_collected += market_obs.shape[0]
        
        # Step environment
        action = test_env.action_space.sample()
        obs, _, done, _ = test_env.step(action)
        if isinstance(done, np.ndarray) and done.all():
            obs = test_env.reset()
        elif isinstance(done, bool) and done:
            obs = test_env.reset()
    
    # Concatenate all observations
    all_market_obs = torch.cat(obs_list, dim=0)  # (B, seq_len, n_features)
    batch_size = min(all_market_obs.shape[0], n_samples)
    all_market_obs = all_market_obs[:batch_size]
    
    # Extract HMM context (last 5 columns at last timestep)
    hmm_context = all_market_obs[:, -1, -HMM_CONTEXT_SIZE:].float()  # (B, 5)
    
    # Forward pass through FiLM MLP to get gamma and beta
    with torch.no_grad():
        params = film_layer.mlp(hmm_context)  # (B, 2 * feature_dim)
        gamma, beta = params.chunk(2, dim=-1)  # each (B, feature_dim)
    
    # Convert to numpy for analysis
    gamma_np = gamma.cpu().numpy().flatten()
    beta_np = beta.cpu().numpy().flatten()
    
    # Statistics
    gamma_mean = float(gamma_np.mean())
    gamma_std = float(gamma_np.std())
    gamma_min = float(gamma_np.min())
    gamma_max = float(gamma_np.max())
    gamma_median = float(np.median(gamma_np))
    
    beta_mean = float(beta_np.mean())
    beta_std = float(beta_np.std())
    beta_min = float(beta_np.min())
    beta_max = float(beta_np.max())
    beta_median = float(np.median(beta_np))
    
    # Check if gamma is centered around 1.0 (not collapsed to 0)
    gamma_centered = abs(gamma_mean - 1.0) < 0.5
    gamma_not_collapsed = gamma_std > 0.01  # Should have some variance
    
    print(f"  Gamma: mean={gamma_mean:.6f}, std={gamma_std:.6f}, "
          f"min={gamma_min:.6f}, max={gamma_max:.6f}, median={gamma_median:.6f}")
    print(f"    → Centered around 1.0: {gamma_centered}")
    print(f"    → Not collapsed (std > 0.01): {gamma_not_collapsed}")
    print(f"  Beta: mean={beta_mean:.6f}, std={beta_std:.6f}, "
          f"min={beta_min:.6f}, max={beta_max:.6f}, median={beta_median:.6f}")
    
    # Plot histograms
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Gamma histogram
    axes[0].hist(gamma_np, bins=50, alpha=0.7, edgecolor='black')
    axes[0].axvline(1.0, color='r', linestyle='--', label='Initial (1.0)')
    axes[0].axvline(gamma_mean, color='g', linestyle='--', label=f'Mean ({gamma_mean:.3f})')
    axes[0].set_xlabel('Gamma Value')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'Gamma Distribution (mean={gamma_mean:.3f}, std={gamma_std:.3f})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Beta histogram
    axes[1].hist(beta_np, bins=50, alpha=0.7, edgecolor='black', color='orange')
    axes[1].axvline(0.0, color='r', linestyle='--', label='Initial (0.0)')
    axes[1].axvline(beta_mean, color='g', linestyle='--', label=f'Mean ({beta_mean:.3f})')
    axes[1].set_xlabel('Beta Value')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'Beta Distribution (mean={beta_mean:.3f}, std={beta_std:.3f})')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "film_gamma_dist.png"), dpi=150)
    plt.close()
    print(f"  Saved: film_gamma_dist.png")
    
    results['dynamic_analysis'] = {
        'gamma_mean': gamma_mean,
        'gamma_std': gamma_std,
        'gamma_min': gamma_min,
        'gamma_max': gamma_max,
        'gamma_median': gamma_median,
        'beta_mean': beta_mean,
        'beta_std': beta_std,
        'beta_min': beta_min,
        'beta_max': beta_max,
        'beta_median': beta_median,
        'gamma_centered': gamma_centered,
        'gamma_not_collapsed': gamma_not_collapsed
    }
    
    # ========================================================================
    # C. SENSITIVITY TEST (CRASH TEST)
    # ========================================================================
    print(f"\n[5/4] C. Sensitivity Test (Crash Test)...")
    
    # Get a fixed MAE embedding sample
    # We need to encode a sample through the MAE encoder first
    sample_market = all_market_obs[0:1]  # (1, seq_len, n_features)
    
    # Encode through MAE (frozen encoder)
    # Handle float16/float32 conversion if needed
    with torch.no_grad():
        encoder_dtype = next(feature_extractor.mae.embedding.parameters()).dtype
        if encoder_dtype == torch.float16:
            if sample_market.is_cuda:
                with torch.amp.autocast('cuda'):
                    encoded = feature_extractor.mae.encode(sample_market.half())
                encoded = encoded.float()
            else:
                # CPU doesn't support float16, convert to float32
                encoded = feature_extractor.mae.encode(sample_market.float())
        else:
            encoded = feature_extractor.mae.encode(sample_market.float())  # (1, seq_len, d_model)
    
    # Create pure HMM contexts (one-hot for 4 states + entropy)
    n_states = 4
    contexts = []
    
    # State 0: [1, 0, 0, 0, entropy] - Crash pur
    entropy_val = 0.0  # Low entropy (certain state)
    context_0 = torch.tensor([[1.0, 0.0, 0.0, 0.0, entropy_val]], device=device, dtype=torch.float32)
    contexts.append(context_0)
    
    # State 1: [0, 1, 0, 0, entropy]
    context_1 = torch.tensor([[0.0, 1.0, 0.0, 0.0, entropy_val]], device=device, dtype=torch.float32)
    contexts.append(context_1)
    
    # State 2: [0, 0, 1, 0, entropy]
    context_2 = torch.tensor([[0.0, 0.0, 1.0, 0.0, entropy_val]], device=device, dtype=torch.float32)
    contexts.append(context_2)
    
    # State 3: [0, 0, 0, 1, entropy] - Pump pur
    context_3 = torch.tensor([[0.0, 0.0, 0.0, 1.0, entropy_val]], device=device, dtype=torch.float32)
    contexts.append(context_3)
    
    # Repeat the encoded sample for each context
    encoded_repeated = encoded.repeat(n_states, 1, 1)  # (n_states, seq_len, d_model)
    contexts_tensor = torch.cat(contexts, dim=0)  # (n_states, 5)
    
    # Forward pass through FiLM
    with torch.no_grad():
        modulated_outputs = film_layer(encoded_repeated, contexts_tensor)  # (n_states, seq_len, d_model)
    
    # Flatten for distance calculation (or use mean over sequence)
    # Option 1: Flatten completely
    modulated_flat = modulated_outputs.view(n_states, -1)  # (n_states, seq_len * d_model)
    
    # Option 2: Mean over sequence (more interpretable)
    modulated_mean = modulated_outputs.mean(dim=1)  # (n_states, d_model)
    
    # Calculate pairwise distances
    dist_matrix_flat = torch.cdist(modulated_flat, modulated_flat, p=2)  # (n_states, n_states)
    dist_matrix_mean = torch.cdist(modulated_mean, modulated_mean, p=2)  # (n_states, n_states)
    
    # Key metric: Distance between State 0 (Crash) and State 3 (Pump)
    sensitivity_score_flat = dist_matrix_flat[0, 3].item()
    sensitivity_score_mean = dist_matrix_mean[0, 3].item()
    
    # Mean distance across all pairs (excluding diagonal)
    mask = ~torch.eye(n_states, dtype=torch.bool, device=device)
    mean_distance_flat = dist_matrix_flat[mask].mean().item()
    mean_distance_mean = dist_matrix_mean[mask].mean().item()
    
    print(f"  Sensitivity Score (State 0 vs State 3):")
    print(f"    → Flattened: {sensitivity_score_flat:.6f}")
    print(f"    → Mean over sequence: {sensitivity_score_mean:.6f}")
    print(f"  Mean pairwise distance (all states):")
    print(f"    → Flattened: {mean_distance_flat:.6f}")
    print(f"    → Mean over sequence: {mean_distance_mean:.6f}")
    
    # Alert if score is too low (FiLM disconnected)
    threshold = 0.01
    film_disconnected = sensitivity_score_mean < threshold
    
    if film_disconnected:
        print(f"  ⚠️ ALERT: FiLM appears disconnected! (score < {threshold})")
        print(f"     The context is not significantly changing the output.")
    else:
        print(f"  ✓ FiLM is reactive to context changes.")
    
    # Plot sensitivity matrix
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Heatmap for flattened distances
    im1 = axes[0].imshow(dist_matrix_flat.cpu().numpy(), cmap='viridis', aspect='auto')
    axes[0].set_title('Sensitivity Matrix (Flattened)')
    axes[0].set_xlabel('State')
    axes[0].set_ylabel('State')
    axes[0].set_xticks(range(n_states))
    axes[0].set_yticks(range(n_states))
    axes[0].set_xticklabels(['Crash', 'State1', 'State2', 'Pump'])
    axes[0].set_yticklabels(['Crash', 'State1', 'State2', 'Pump'])
    plt.colorbar(im1, ax=axes[0])
    
    # Add text annotations
    for i in range(n_states):
        for j in range(n_states):
            text = axes[0].text(j, i, f'{dist_matrix_flat[i, j].item():.3f}',
                              ha="center", va="center", color="white" if dist_matrix_flat[i, j] > dist_matrix_flat.max() / 2 else "black")
    
    # Heatmap for mean distances
    im2 = axes[1].imshow(dist_matrix_mean.cpu().numpy(), cmap='viridis', aspect='auto')
    axes[1].set_title('Sensitivity Matrix (Mean over Sequence)')
    axes[1].set_xlabel('State')
    axes[1].set_ylabel('State')
    axes[1].set_xticks(range(n_states))
    axes[1].set_yticks(range(n_states))
    axes[1].set_xticklabels(['Crash', 'State1', 'State2', 'Pump'])
    axes[1].set_yticklabels(['Crash', 'State1', 'State2', 'Pump'])
    plt.colorbar(im2, ax=axes[1])
    
    # Add text annotations
    for i in range(n_states):
        for j in range(n_states):
            text = axes[1].text(j, i, f'{dist_matrix_mean[i, j].item():.3f}',
                              ha="center", va="center", color="white" if dist_matrix_mean[i, j] > dist_matrix_mean.max() / 2 else "black")
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "film_sensitivity_matrix.png"), dpi=150)
    plt.close()
    print(f"  Saved: film_sensitivity_matrix.png")
    
    results['sensitivity_test'] = {
        'sensitivity_score_flat': sensitivity_score_flat,
        'sensitivity_score_mean': sensitivity_score_mean,
        'mean_distance_flat': mean_distance_flat,
        'mean_distance_mean': mean_distance_mean,
        'film_disconnected': film_disconnected,
        'threshold': threshold,
        'dist_matrix_flat': dist_matrix_flat.cpu().numpy().tolist(),
        'dist_matrix_mean': dist_matrix_mean.cpu().numpy().tolist()
    }
    
    # ========================================================================
    # GENERATE REPORT
    # ========================================================================
    print(f"\n[6/4] Generating report...")
    report_path = os.path.join(output_dir, "film_audit_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("FiLM MECHANICS AUDIT REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Segment ID: {segment_id}\n")
        f.write(f"TQC Path: {tqc_path}\n")
        f.write(f"Test Data: {test_data_path}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("A. STATIC ANALYSIS (WEIGHTS)\n")
        f.write("=" * 80 + "\n\n")
        static = results['static_analysis']
        f.write(f"Gamma Weights:\n")
        f.write(f"  Mean: {static['gamma_weight_mean']:.6f}\n")
        f.write(f"  Std:  {static['gamma_weight_std']:.6f}\n")
        f.write(f"Gamma Bias:\n")
        f.write(f"  Mean: {static['gamma_bias_mean']:.6f}\n")
        f.write(f"  Std:  {static['gamma_bias_std']:.6f}\n")
        f.write(f"  Initialization visible (≈1.0): {static['init_gamma_visible']}\n\n")
        f.write(f"Beta Weights:\n")
        f.write(f"  Mean: {static['beta_weight_mean']:.6f}\n")
        f.write(f"  Std:  {static['beta_weight_std']:.6f}\n")
        f.write(f"Beta Bias:\n")
        f.write(f"  Mean: {static['beta_bias_mean']:.6f}\n")
        f.write(f"  Std:  {static['beta_bias_std']:.6f}\n")
        f.write(f"  Initialization visible (≈0.0): {static['init_beta_visible']}\n\n")
        if static['weight_exploded']:
            f.write("⚠️ WARNING: Weight explosion detected!\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("B. DYNAMIC ANALYSIS (ACTIVATIONS)\n")
        f.write("=" * 80 + "\n\n")
        dynamic = results['dynamic_analysis']
        f.write(f"Gamma Statistics:\n")
        f.write(f"  Mean:   {dynamic['gamma_mean']:.6f}\n")
        f.write(f"  Std:    {dynamic['gamma_std']:.6f}\n")
        f.write(f"  Min:    {dynamic['gamma_min']:.6f}\n")
        f.write(f"  Max:    {dynamic['gamma_max']:.6f}\n")
        f.write(f"  Median: {dynamic['gamma_median']:.6f}\n")
        f.write(f"  Centered around 1.0: {dynamic['gamma_centered']}\n")
        f.write(f"  Not collapsed (std > 0.01): {dynamic['gamma_not_collapsed']}\n\n")
        f.write(f"Beta Statistics:\n")
        f.write(f"  Mean:   {dynamic['beta_mean']:.6f}\n")
        f.write(f"  Std:    {dynamic['beta_std']:.6f}\n")
        f.write(f"  Min:    {dynamic['beta_min']:.6f}\n")
        f.write(f"  Max:    {dynamic['beta_max']:.6f}\n")
        f.write(f"  Median: {dynamic['beta_median']:.6f}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("C. SENSITIVITY TEST (CRASH TEST)\n")
        f.write("=" * 80 + "\n\n")
        sensitivity = results['sensitivity_test']
        f.write(f"Sensitivity Score (State 0 vs State 3):\n")
        f.write(f"  Flattened: {sensitivity['sensitivity_score_flat']:.6f}\n")
        f.write(f"  Mean over sequence: {sensitivity['sensitivity_score_mean']:.6f}\n\n")
        f.write(f"Mean pairwise distance (all states):\n")
        f.write(f"  Flattened: {sensitivity['mean_distance_flat']:.6f}\n")
        f.write(f"  Mean over sequence: {sensitivity['mean_distance_mean']:.6f}\n\n")
        if sensitivity['film_disconnected']:
            f.write(f"⚠️ ALERT: FiLM appears disconnected! (score < {sensitivity['threshold']})\n")
            f.write("   The context is not significantly changing the output.\n\n")
        else:
            f.write("✓ FiLM is reactive to context changes.\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    print(f"  Report saved: {report_path}")
    
    # Save metrics JSON
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(_serialize_results(results), f, indent=2)
    print(f"  Metrics saved: {metrics_path}")
    
    print("\n" + "=" * 70)
    print("FiLM AUDIT COMPLETE")
    print("=" * 70)
    print(f"Report: {report_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Plots: {plots_dir}")
    
    return results


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
        choices=["hmm", "pipeline", "both", "synthetic", "mae", "tqc", "film"],
        default="both",
        help="Analysis mode: hmm, pipeline, both, synthetic (old MAE test), mae (comprehensive MAE audit), tqc (comprehensive TQC audit), or film (FiLM mechanics audit)"
    )
    parser.add_argument("--segment", type=int, default=0, help="WFO segment ID")
    parser.add_argument("--skip-mae", action="store_true", help="Skip MAE in pipeline mode")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force-retrain", action="store_true", help="Force data retraining")
    parser.add_argument("--signal", type=float, default=0.3, help="Signal strength (synthetic mode)")
    parser.add_argument("--epochs", type=int, default=20, help="MAE epochs (synthetic/mae mode)")
    
    # New MAE audit arguments
    parser.add_argument("--mae-mode", type=str, choices=["synthetic", "real"], default="real",
                        help="MAE audit mode: synthetic or real data")
    parser.add_argument("--mae-encoder-path", type=str, default=None,
                        help="Path to pre-trained MAE encoder (optional)")
    parser.add_argument("--mae-segment", type=int, default=0,
                        help="WFO segment ID for MAE audit (real mode)")
    parser.add_argument("--mae-tsne", action="store_true",
                        help="Also generate t-SNE plots (slower)")
    parser.add_argument("--mae-samples", type=int, default=10000,
                        help="Number of samples for synthetic MAE audit")
    
    # New HMM comprehensive audit arguments
    parser.add_argument("--hmm-mode", type=str, choices=["real", "test"], default="real",
                        help="HMM audit mode: real (WFO data) or test (synthetic)")
    parser.add_argument("--hmm-segment", type=int, default=0,
                        help="WFO segment ID for HMM audit (real mode)")
    parser.add_argument("--hmm-retrain", action="store_true",
                        help="Force HMM retraining (otherwise uses existing if available)")
    parser.add_argument("--hmm-tsne", action="store_true",
                        help="Also generate t-SNE plots for HMM feature space (slower)")
    
    parser.add_argument("--check-normalization", action="store_true",
                        help="Run normalization/clipping audit for HMM, MAE, TQC")
    
    # TQC audit arguments
    parser.add_argument("--tqc-segment", type=int, default=0,
                        help="WFO segment ID for TQC audit")
    parser.add_argument("--tqc-path", type=str, default=None,
                        help="Path to TQC model (auto-detected from segment if None)")
    parser.add_argument("--tqc-test-data", type=str, default=None,
                        help="Path to test data parquet (auto-detected if None)")
    parser.add_argument("--tqc-samples", type=int, default=1000,
                        help="Number of samples for attribution analysis")
    
    # FiLM audit arguments
    parser.add_argument("--film-segment", type=int, default=0,
                        help="WFO segment ID for FiLM audit")
    parser.add_argument("--film-path", type=str, default=None,
                        help="Path to TQC model for FiLM audit (auto-detected from segment if None)")
    parser.add_argument("--film-test-data", type=str, default=None,
                        help="Path to test data parquet for FiLM audit (auto-detected if None)")
    parser.add_argument("--film-samples", type=int, default=1000,
                        help="Number of samples for dynamic analysis")
    
    args = parser.parse_args()

    if args.mode == "mae":
        # New comprehensive MAE audit
        run_mae_audit(
            mode=args.mae_mode,
            segment_id=args.mae_segment,
            encoder_path=args.mae_encoder_path,
            epochs=args.epochs,
            device=args.device,
            force_retrain=args.force_retrain,
            use_tsne=args.mae_tsne,
            signal_strength=args.signal,
            n_samples=args.mae_samples
        )
    elif args.mode == "synthetic":
        # Old synthetic comparison (kept for backward compatibility)
        run_synthetic_comparison(
            signal_strength=args.signal,
            epochs=args.epochs,
            device=args.device
        )
    elif args.mode in ["hmm", "both"]:
        # New comprehensive HMM audit
        run_hmm_audit_comprehensive(
            segment_id=args.hmm_segment,
            force_retrain=args.hmm_retrain,
            use_tsne=args.hmm_tsne
        )
        if args.mode == "both":
            print("\n")

    if args.mode in ["pipeline", "both"]:
        run_pipeline_audit(
            segment_id=args.segment,
            skip_mae=args.skip_mae,
            device=args.device,
            force_retrain=args.force_retrain,
        )
    
    if args.check_normalization:
        run_normalization_audit(
            segment_id=args.segment,
            output_dir=None,  # Auto-generate unique directory with timestamp
            device=args.device,
            force_retrain=args.force_retrain
        )
    
    if args.mode == "tqc":
        run_tqc_audit(
            segment_id=args.tqc_segment,
            tqc_path=args.tqc_path,
            test_data_path=args.tqc_test_data,
            encoder_path=args.mae_encoder_path,  # Réutiliser argument MAE
            device=args.device,
            n_samples=args.tqc_samples
        )
    
    if args.mode == "film":
        audit_film_mechanics(
            segment_id=args.film_segment,
            tqc_path=args.film_path,
            test_data_path=args.film_test_data,
            encoder_path=args.mae_encoder_path,  # Réutiliser argument MAE
            device=args.device,
            n_samples=args.film_samples
        )


if __name__ == "__main__":
    main()

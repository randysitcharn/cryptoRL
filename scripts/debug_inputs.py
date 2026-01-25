#!/usr/bin/env python3
"""
debug_inputs.py - Diagnostic script for BatchCryptoEnv observations.

Analyzes raw observations from BatchCryptoEnv to detect:
- Constant features (std=0) → "Dead Features"
- NaN/Inf values → Data corruption
- Saturation (abs > 10.0) → Missing normalization
- Vanishing (abs < 1e-6) → Scaling issues

This script does NOT require a trained model - it just runs random actions
to inspect the environment's observation outputs.

Usage:
    python scripts/debug_inputs.py [--segment-id 0] [--n-steps 1000] [--use-test-data]
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch

# Add project root to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.training.batch_env import BatchCryptoEnv
from src.config import WFOTrainingConfig


def find_segment_data_path(segment_id: int, use_test_data: bool = False) -> str:
    """
    Find the data path for a given segment.
    
    Args:
        segment_id: Segment identifier (0, 1, 2, ...)
        use_test_data: If True, use test.parquet instead of train.parquet
        
    Returns:
        Path to the parquet file.
        
    Raises:
        FileNotFoundError: If the segment data file doesn't exist.
    """
    data_type = "test" if use_test_data else "train"
    segment_path = f"data/wfo/segment_{segment_id}/{data_type}.parquet"
    
    if not os.path.exists(segment_path):
        # Fallback to processed_data.parquet if segment doesn't exist
        fallback_path = "data/processed_data.parquet"
        if os.path.exists(fallback_path):
            print(f"[WARNING] Segment {segment_id} not found, using fallback: {fallback_path}")
            return fallback_path
        else:
            raise FileNotFoundError(
                f"Segment data not found: {segment_path}\n"
                f"Fallback also not found: {fallback_path}\n"
                f"Please run WFO pipeline first or ensure data exists."
            )
    
    return segment_path


def analyze_tensor(
    tensor: np.ndarray,
    name: str,
    feature_idx: Optional[int] = None
) -> Dict[str, Any]:
    """
    Analyze a tensor for anomalies.
    
    Args:
        tensor: Input tensor (any shape).
        name: Feature name for reporting.
        feature_idx: Optional feature index (for per-feature analysis).
        
    Returns:
        Dictionary with statistics and anomaly flags.
    """
    # Flatten for analysis
    flat = tensor.flatten()
    
    # Basic statistics
    mean = float(np.mean(flat))
    std = float(np.std(flat))
    min_val = float(np.min(flat))
    max_val = float(np.max(flat))
    
    # Anomaly detection
    n_nan = int(np.isnan(flat).sum())
    n_inf = int(np.isinf(flat).sum())
    is_dead = (std == 0.0) and (n_nan == 0) and (n_inf == 0)
    
    # Saturation check: abs > 10.0 suggests missing normalization
    n_saturated = int((np.abs(flat[np.isfinite(flat)]) > 10.0).sum()) if np.isfinite(flat).any() else 0
    is_saturated = n_saturated > 0

    # Vanishing check: abs < 1e-6 suggests scaling issue
    finite_flat = flat[np.isfinite(flat)]
    n_vanishing = int((np.abs(finite_flat) < 1e-6).sum()) if len(finite_flat) > 0 else 0
    is_vanishing = (n_vanishing > 0) and (std > 0)  # Only flag if there's variation
    
    # Critical anomaly: NaN, Inf, or Dead Feature
    has_critical = (n_nan > 0) or (n_inf > 0) or is_dead
    
    return {
        "name": name,
        "feature_idx": feature_idx,
        "shape": tensor.shape,
        "mean": mean,
        "std": std,
        "min": min_val,
        "max": max_val,
        "n_nan": n_nan,
        "n_inf": n_inf,
        "is_dead": is_dead,
        "n_saturated": n_saturated,
        "is_saturated": is_saturated,
        "n_vanishing": n_vanishing,
        "is_vanishing": is_vanishing,
        "has_critical": has_critical,
    }


def print_analysis_table(results: List[Dict[str, Any]], group_name: str):
    """
    Print a formatted analysis table.
    
    Args:
        results: List of analysis dictionaries.
        group_name: Name of the feature group (e.g., "market", "position").
    """
    print(f"\n{'=' * 100}")
    print(f"ANALYSIS: {group_name.upper()}")
    print(f"{'=' * 100}")
    
    # Table header
    header = f"{'Feature':<30} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12} {'NaN':>6} {'Inf':>6} {'Status':<20}"
    print(header)
    print("-" * 100)
    
    # Table rows
    for r in results:
        # Build status string
        status_parts = []
        if r["has_critical"]:
            if r["n_nan"] > 0:
                status_parts.append(f"NaN({r['n_nan']})")
            if r["n_inf"] > 0:
                status_parts.append(f"Inf({r['n_inf']})")
            if r["is_dead"]:
                status_parts.append("DEAD")
        if r["is_saturated"]:
            status_parts.append(f"SAT({r['n_saturated']})")
        if r["is_vanishing"]:
            status_parts.append(f"VAN({r['n_vanishing']})")
        
        status = ", ".join(status_parts) if status_parts else "OK"
        
        # Format values
        mean_str = f"{r['mean']:>12.6f}" if np.isfinite(r['mean']) else "NaN"
        std_str = f"{r['std']:>12.6f}" if np.isfinite(r['std']) else "NaN"
        min_str = f"{r['min']:>12.6f}" if np.isfinite(r['min']) else "NaN"
        max_str = f"{r['max']:>12.6f}" if np.isfinite(r['max']) else "NaN"
        
        name = r["name"]
        if r["feature_idx"] is not None:
            name = f"{name}[{r['feature_idx']}]"
        
        row = f"{name:<30} {mean_str} {std_str} {min_str} {max_str} {r['n_nan']:>6} {r['n_inf']:>6} {status:<20}"
        print(row)
    
    print("-" * 100)


def main():
    parser = argparse.ArgumentParser(description="Debug BatchCryptoEnv observations")
    parser.add_argument(
        "--segment-id",
        type=int,
        default=0,
        help="Segment ID to use (default: 0)"
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=1000,
        help="Number of steps to run (default: 1000)"
    )
    parser.add_argument(
        "--use-test-data",
        action="store_true",
        help="Use test.parquet instead of train.parquet"
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=32,
        help="Number of parallel environments (default: 32)"
    )
    
    args = parser.parse_args()
    
    print("=" * 100)
    print("BATCHCRYPTOENV OBSERVATION DIAGNOSTIC")
    print("=" * 100)
    print(f"Segment ID: {args.segment_id}")
    print(f"Data type: {'test' if args.use_test_data else 'train'}")
    print(f"Steps: {args.n_steps}")
    print(f"Environments: {args.n_envs}")
    print("=" * 100)
    
    # Load configuration
    print("\n[1/4] Loading WFOTrainingConfig...")
    config = WFOTrainingConfig()
    print(f"  Window size: {config.window_size}")
    print(f"  Episode length: {config.episode_length}")
    
    # Find data path
    print("\n[2/4] Finding segment data...")
    data_path = find_segment_data_path(args.segment_id, args.use_test_data)
    print(f"  Data path: {data_path}")
    
    # Create environment (validation/test mode: no noise, sequential start)
    print("\n[3/4] Creating BatchCryptoEnv (validation mode)...")
    env = BatchCryptoEnv(
        parquet_path=data_path,
        n_envs=args.n_envs,
        device="cuda" if torch.cuda.is_available() else "cpu",
        window_size=config.window_size,
        episode_length=config.episode_length,
        initial_balance=10_000.0,
        commission=config.commission,
        slippage=0.0001,
        reward_scaling=config.reward_scaling,
        downside_coef=config.downside_coef,
        upside_coef=config.upside_coef,
        action_discretization=config.action_discretization,
        target_volatility=config.target_volatility,
        vol_window=config.vol_window,
        max_leverage=config.max_leverage,
        price_column='BTC_Close',
        observation_noise=0.0,  # Disable noise for validation
        random_start=False,  # Sequential start for reproducibility
    )
    
    # Disable training mode (no noise injection)
    env.set_training_mode(False)
    
    print(f"  Device: {env.device}")
    print(f"  Features: {env.n_features}")
    print(f"  Feature names: {env.feature_names[:5]}... (showing first 5)")
    
    # Reset environment
    print("\n[4/4] Running random policy loop...")
    obs = env.reset()
    
    # Collect observations
    all_observations = {
        "market": [],
        "position": [],
        "w_cost": [],
    }
    
    for step in range(args.n_steps):
        # Sample random action
        actions = env.action_space.sample()
        
        # Step environment
        env.step_async(actions)
        obs, rewards, dones, infos = env.step_wait()
        
        # Collect observations
        all_observations["market"].append(obs["market"])
        all_observations["position"].append(obs["position"])
        all_observations["w_cost"].append(obs["w_cost"])
        
        if (step + 1) % 100 == 0:
            print(f"  Step {step + 1}/{args.n_steps}...")
    
    print("\n" + "=" * 100)
    print("ANALYZING OBSERVATIONS")
    print("=" * 100)
    
    # Concatenate all observations
    market_all = np.concatenate(all_observations["market"], axis=0)  # (n_steps * n_envs, window_size, n_features)
    position_all = np.concatenate(all_observations["position"], axis=0)  # (n_steps * n_envs, 1)
    w_cost_all = np.concatenate(all_observations["w_cost"], axis=0)  # (n_steps * n_envs, 1)
    
    # Analyze market features (per feature dimension)
    print("\nAnalyzing market features (per feature dimension)...")
    market_results = []
    n_features = market_all.shape[2]
    
    for feat_idx in range(n_features):
        # Extract feature across all timesteps and envs
        feature_data = market_all[:, :, feat_idx]  # (n_steps * n_envs, window_size)
        
        result = analyze_tensor(
            feature_data,
            name=env.feature_names[feat_idx] if feat_idx < len(env.feature_names) else f"feature_{feat_idx}",
            feature_idx=feat_idx
        )
        market_results.append(result)
    
    print_analysis_table(market_results, "market")
    
    # Analyze position tensor
    print("\nAnalyzing position tensor...")
    position_result = analyze_tensor(position_all, name="position")
    print_analysis_table([position_result], "position")
    
    # Analyze w_cost tensor
    print("\nAnalyzing w_cost tensor...")
    w_cost_result = analyze_tensor(w_cost_all, name="w_cost")
    print_analysis_table([w_cost_result], "w_cost")
    
    # Summary: Check for critical anomalies
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    
    critical_issues = []
    
    # Check market features
    for r in market_results:
        if r["has_critical"]:
            issue = f"Market[{r['name']}]: "
            if r["n_nan"] > 0:
                issue += f"NaN({r['n_nan']}) "
            if r["n_inf"] > 0:
                issue += f"Inf({r['n_inf']}) "
            if r["is_dead"]:
                issue += "DEAD (std=0) "
            critical_issues.append(issue.strip())
    
    # Check position
    if position_result["has_critical"]:
        issue = "Position: "
        if position_result["n_nan"] > 0:
            issue += f"NaN({position_result['n_nan']}) "
        if position_result["n_inf"] > 0:
            issue += f"Inf({position_result['n_inf']}) "
        if position_result["is_dead"]:
            issue += "DEAD (std=0) "
        critical_issues.append(issue.strip())
    
    # Check w_cost
    if w_cost_result["has_critical"]:
        issue = "w_cost: "
        if w_cost_result["n_nan"] > 0:
            issue += f"NaN({w_cost_result['n_nan']}) "
        if w_cost_result["n_inf"] > 0:
            issue += f"Inf({w_cost_result['n_inf']}) "
        if w_cost_result["is_dead"]:
            issue += "DEAD (std=0) "
        critical_issues.append(issue.strip())
    
    # Print summary
    if critical_issues:
        print("\n❌ CRITICAL ANOMALIES DETECTED:")
        for issue in critical_issues:
            print(f"  - {issue}")
        print("\n⚠️  WARNINGS (non-critical):")
        
        # Check saturation
        saturated_features = [r for r in market_results if r["is_saturated"]]
        if saturated_features:
            print(f"  - {len(saturated_features)} market features with saturation (abs > 10.0)")
        
        if position_result["is_saturated"]:
            print("  - Position tensor with saturation")
        if w_cost_result["is_saturated"]:
            print("  - w_cost tensor with saturation")
        
        # Check vanishing
        vanishing_features = [r for r in market_results if r["is_vanishing"]]
        if vanishing_features:
            print(f"  - {len(vanishing_features)} market features with vanishing values (abs < 1e-6)")
        
        if position_result["is_vanishing"]:
            print("  - Position tensor with vanishing values")
        if w_cost_result["is_vanishing"]:
            print("  - w_cost tensor with vanishing values")
    else:
        print("\n✅ No critical anomalies detected!")
        
        # Still check warnings
        warnings = []
        saturated_features = [r for r in market_results if r["is_saturated"]]
        if saturated_features:
            warnings.append(f"{len(saturated_features)} market features with saturation")
        
        vanishing_features = [r for r in market_results if r["is_vanishing"]]
        if vanishing_features:
            warnings.append(f"{len(vanishing_features)} market features with vanishing values")
        
        if warnings:
            print("\n⚠️  WARNINGS:")
            for w in warnings:
                print(f"  - {w}")
        else:
            print("✅ No warnings!")
    
    print("\n" + "=" * 100)
    
    # Raise error if critical anomalies found
    if critical_issues:
        raise ValueError(
            f"CRITICAL ANOMALIES DETECTED: {len(critical_issues)} issues found.\n"
            f"Pipeline stopped. Please fix data preprocessing or feature engineering.\n"
            f"Issues:\n" + "\n".join(f"  - {issue}" for issue in critical_issues)
        )
    
    print("✅ Diagnostic completed successfully!")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] User cancelled.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

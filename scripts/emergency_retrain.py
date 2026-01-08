#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
emergency_retrain.py - Emergency Fine-Tuning for TQC Agent

Quick retraining script for when market regimes shift and the RiskManagementWrapper
blocks trades. Fine-tunes an existing model on recent data with reduced learning rate
to prevent catastrophic forgetting.

CRITICAL: All environment parameters are imported from TrainingConfig to ensure
mathematical equivalence with the original training environment.

Usage:
    python scripts/emergency_retrain.py \
        --model_path weights/segment_0/tqc.zip \
        --data_path data/processed_data.parquet \
        --steps 20000 \
        --lr_factor 0.1

Author: ML Ops
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sb3_contrib import TQC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.training.env import CryptoTradingEnv
from src.training.train_agent import TrainingConfig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Emergency Fine-Tuning for TQC Agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the stable model (.zip)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to dataset (includes crash data)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20_000,
        help="Fine-tuning steps (~10-15%% of original 150k)",
    )
    parser.add_argument(
        "--lr_factor",
        type=float,
        default=0.1,
        help="Learning rate multiplier (0.1 = 10%% of original)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="weights/emergency",
        help="Output directory for fine-tuned model",
    )
    parser.add_argument(
        "--eval_rows",
        type=int,
        default=720,
        help="Rows from END of dataset for crash eval (720 = 1 month)",
    )
    return parser.parse_args()


def create_finetune_env(data_path: str, cfg: TrainingConfig) -> DummyVecEnv:
    """
    Create training environment for fine-tuning.

    Uses TrainingConfig values to ensure mathematical equivalence.
    - VolScaling: ON (from config)
    - CircuitBreaker: OFF (agent must learn from crash data)
    - Curriculum: OFF (full costs immediately)
    """
    env = CryptoTradingEnv(
        parquet_path=data_path,
        window_size=cfg.window_size,
        commission=cfg.commission,
        reward_scaling=cfg.reward_scaling,
        downside_coef=cfg.downside_coef,
        upside_coef=cfg.upside_coef,
        churn_coef=cfg.churn_coef,
        smooth_coef=cfg.smooth_coef,
        action_discretization=cfg.action_discretization,
        target_volatility=cfg.target_volatility,
        vol_window=cfg.vol_window,
        max_leverage=cfg.max_leverage,
        random_start=True,
        episode_length=cfg.episode_length,
    )
    env = Monitor(env)
    # Lambda with default arg to avoid late binding issue
    return DummyVecEnv([lambda e=env: e])


def create_eval_env(
    data_path: str,
    cfg: TrainingConfig,
    eval_rows: int,
) -> tuple[CryptoTradingEnv, int, int]:
    """
    Create evaluation environment targeting the END of the dataset (crash period).

    Args:
        data_path: Path to parquet data.
        cfg: TrainingConfig for environment parameters.
        eval_rows: Number of rows from end for evaluation.

    Returns:
        Tuple of (env, start_idx, end_idx) for logging.
    """
    # Calculate indices to target crash period at END of dataset
    df = pd.read_parquet(data_path)
    total_rows = len(df)

    # Need window_size extra rows for observation buffer
    crash_start_idx = max(0, total_rows - eval_rows - cfg.window_size)
    crash_end_idx = total_rows

    env = CryptoTradingEnv(
        parquet_path=data_path,
        window_size=cfg.window_size,
        commission=cfg.commission,
        reward_scaling=cfg.reward_scaling,
        downside_coef=cfg.downside_coef,
        upside_coef=cfg.upside_coef,
        churn_coef=cfg.churn_coef,
        smooth_coef=cfg.smooth_coef,
        action_discretization=cfg.action_discretization,
        target_volatility=cfg.target_volatility,
        vol_window=cfg.vol_window,
        max_leverage=cfg.max_leverage,
        start_idx=crash_start_idx,
        end_idx=crash_end_idx,
        random_start=False,  # Deterministic for comparison
        episode_length=None,  # Full crash period
    )

    return env, crash_start_idx, crash_end_idx


def evaluate_model(model: TQC, env: CryptoTradingEnv) -> dict:
    """
    Run a single evaluation episode and return metrics.

    Uses deterministic policy, no circuit breaker.
    """
    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    steps = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated

    return {
        "nav": info.get("nav", 10000.0),
        "total_trades": info.get("total_trades", 0),
        "total_reward": total_reward,
        "steps": steps,
    }


def print_config_verification(cfg: TrainingConfig, new_lr: float):
    """Print config values for safety verification."""
    print("[Config Verification - Single Source of Truth]")
    print(f"  reward_scaling:        {cfg.reward_scaling} (from TrainingConfig)")
    print(f"  downside_coef:         {cfg.downside_coef} (from TrainingConfig)")
    print(f"  churn_coef:            {cfg.churn_coef} (from TrainingConfig)")
    print(f"  smooth_coef:           {cfg.smooth_coef} (from TrainingConfig)")
    print(f"  commission:            {cfg.commission} (from TrainingConfig)")
    print(f"  action_discretization: {cfg.action_discretization} (from TrainingConfig)")
    print(f"  target_volatility:     {cfg.target_volatility} (from TrainingConfig)")
    print(f"  vol_window:            {cfg.vol_window} (from TrainingConfig)")
    print(f"  max_leverage:          {cfg.max_leverage} (from TrainingConfig)")
    print(f"  learning_rate:         {cfg.learning_rate:.0e} -> {new_lr:.0e}")
    print()


def main():
    """Main entry point."""
    args = parse_args()

    # Validate inputs
    if not os.path.exists(args.model_path):
        print(f"[ERROR] Model not found: {args.model_path}")
        sys.exit(1)
    if not os.path.exists(args.data_path):
        print(f"[ERROR] Data not found: {args.data_path}")
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load config (SINGLE SOURCE OF TRUTH)
    cfg = TrainingConfig()

    # Calculate new learning rate from config
    new_lr = cfg.learning_rate * args.lr_factor

    print("=" * 70)
    print("[Emergency Fine-Tuning]")
    print("=" * 70)
    print(f"  Model: {args.model_path}")
    print(f"  Data: {args.data_path}")
    print(f"  Steps: {args.steps:,}")
    print(f"  LR Factor: {args.lr_factor} ({args.lr_factor*100:.0f}% of original)")
    print(f"  Eval Rows: {args.eval_rows} (from END of dataset)")
    print()

    # Print config verification
    print_config_verification(cfg, new_lr)

    # Load model
    print("[Loading model...]")
    model = TQC.load(args.model_path)

    # Create eval environment targeting crash period (END of dataset)
    print("[Creating crash period evaluation environment...]")
    eval_env, start_idx, end_idx = create_eval_env(args.data_path, cfg, args.eval_rows)
    print(f"  Dataset rows: {end_idx:,}")
    print(f"  Eval range: rows {start_idx:,} - {end_idx:,} (last {args.eval_rows} + {cfg.window_size} window)")
    print()

    # Evaluate BEFORE fine-tuning on crash period
    print("[Evaluating BEFORE fine-tuning (crash period)...]")
    before_metrics = evaluate_model(model, eval_env)
    print(f"  [Before] NAV: ${before_metrics['nav']:,.2f} | "
          f"Trades: {before_metrics['total_trades']} | "
          f"Reward: {before_metrics['total_reward']:.2f}")
    print()

    # Update learning rate (prevent catastrophic forgetting)
    model.learning_rate = new_lr
    model.lr_schedule = lambda _: new_lr

    # Create fine-tuning environment (full dataset)
    print("[Creating fine-tuning environment...]")
    print("  VolScaling: ON")
    print("  CircuitBreaker: OFF (must learn from crash)")
    print("  Curriculum: OFF (full costs)")
    train_env = create_finetune_env(args.data_path, cfg)

    # Attach environment to model
    model.set_env(train_env)

    # Fine-tune
    print()
    print(f"[Fine-tuning for {args.steps:,} steps...]")
    model.learn(
        total_timesteps=args.steps,
        reset_num_timesteps=False,  # Continue TensorBoard logging
        progress_bar=True,
    )

    # Re-create eval env (env state may be corrupted after training)
    eval_env, _, _ = create_eval_env(args.data_path, cfg, args.eval_rows)

    # Evaluate AFTER fine-tuning on crash period
    print()
    print("[Evaluating AFTER fine-tuning (crash period)...]")
    after_metrics = evaluate_model(model, eval_env)
    print(f"  [After]  NAV: ${after_metrics['nav']:,.2f} | "
          f"Trades: {after_metrics['total_trades']} | "
          f"Reward: {after_metrics['total_reward']:.2f}")

    # Compute improvement
    nav_change = after_metrics['nav'] - before_metrics['nav']
    nav_pct = (nav_change / before_metrics['nav']) * 100
    reward_change = after_metrics['total_reward'] - before_metrics['total_reward']
    print()
    print("[Improvement]")
    print(f"  NAV Change:    ${nav_change:+,.2f} ({nav_pct:+.2f}%)")
    print(f"  Reward Change: {reward_change:+.2f}")

    # Save fine-tuned model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(args.output_dir, f"emergency_model_{timestamp}.zip")
    model.save(output_path)

    print()
    print("=" * 70)
    print(f"[Saved] {output_path}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())

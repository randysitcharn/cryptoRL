# -*- coding: utf-8 -*-
"""
check_activity.py - Agent Behavioral Audit.

Checks if the agent is making varied decisions or suffering from
"Mode Collapse" (stuck on a single action value).

Usage:
    python -m src.evaluation.check_activity
"""

import numpy as np
import pandas as pd

from sb3_contrib import TQC
from src.training.env import CryptoTradingEnv


# ============================================================================
# Configuration
# ============================================================================

class AuditConfig:
    """Configuration for behavioral audit."""

    data_path: str = "data/processed_data.parquet"
    model_path: str = "weights/checkpoints/best_model.zip"

    window_size: int = 64
    commission: float = 0.0006
    train_ratio: float = 0.8

    audit_steps: int = 1000


# ============================================================================
# Audit Functions
# ============================================================================

def run_behavioral_audit(config: AuditConfig = None) -> dict:
    """
    Run behavioral audit on the agent.

    Args:
        config: Audit configuration.

    Returns:
        Dictionary with audit results.
    """
    if config is None:
        config = AuditConfig()

    print("=" * 70)
    print("AGENT BEHAVIORAL AUDIT")
    print("=" * 70)

    # ==================== Load Environment ====================
    print(f"\n[1/3] Loading validation environment...")

    df = pd.read_parquet(config.data_path)
    n_total = len(df)
    split_idx = int(n_total * config.train_ratio)

    val_env = CryptoTradingEnv(
        parquet_path=config.data_path,
        start_idx=split_idx,
        end_idx=n_total,
        window_size=config.window_size,
        commission=config.commission,
        random_start=False,
        episode_length=None,
    )

    # ==================== Load Model ====================
    print(f"[2/3] Loading model from {config.model_path}...")
    model = TQC.load(config.model_path)
    print("      Model loaded successfully")

    # ==================== Run Audit ====================
    print(f"\n[3/3] Running {config.audit_steps} steps (deterministic=True)...")

    actions = []
    obs, _ = val_env.reset()

    for step in range(config.audit_steps):
        action, _ = model.predict(obs, deterministic=True)
        actions.append(float(action[0]))

        obs, reward, terminated, truncated, info = val_env.step(action)

        if terminated or truncated:
            obs, _ = val_env.reset()

        if (step + 1) % 200 == 0:
            print(f"      Step {step + 1}/{config.audit_steps}")

    actions = np.array(actions)

    # ==================== Statistical Analysis ====================
    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS")
    print("=" * 70)

    # Basic stats
    mean_action = np.mean(actions)
    std_action = np.std(actions)
    min_action = np.min(actions)
    max_action = np.max(actions)

    print(f"\n  Basic Statistics:")
    print(f"    Mean Action:     {mean_action:+.4f}")
    print(f"    Std Deviation:   {std_action:.4f}")
    print(f"    Min Action:      {min_action:+.4f}")
    print(f"    Max Action:      {max_action:+.4f}")
    print(f"    Range:           {max_action - min_action:.4f}")

    # ==================== Action Distribution ====================
    print(f"\n  Action Distribution (Histogram):")

    strong_long = np.sum(actions > 0.5) / len(actions) * 100
    weak_long = np.sum((actions > 0.1) & (actions <= 0.5)) / len(actions) * 100
    neutral = np.sum((actions >= -0.1) & (actions <= 0.1)) / len(actions) * 100
    weak_short = np.sum((actions >= -0.5) & (actions < -0.1)) / len(actions) * 100
    strong_short = np.sum(actions < -0.5) / len(actions) * 100

    print(f"    Strong Long  (> +0.5):  {strong_long:5.1f}% {'#' * int(strong_long / 2)}")
    print(f"    Weak Long    (+0.1~+0.5): {weak_long:5.1f}% {'#' * int(weak_long / 2)}")
    print(f"    Neutral      (-0.1~+0.1): {neutral:5.1f}% {'#' * int(neutral / 2)}")
    print(f"    Weak Short   (-0.5~-0.1): {weak_short:5.1f}% {'#' * int(weak_short / 2)}")
    print(f"    Strong Short (< -0.5):  {strong_short:5.1f}% {'#' * int(strong_short / 2)}")

    # ==================== Frozen State Test ====================
    print(f"\n  Frozen State Analysis:")

    # Calculate action changes between consecutive steps
    action_changes = np.abs(np.diff(actions))

    frozen_threshold = 0.01
    frozen_count = np.sum(action_changes < frozen_threshold)
    frozen_pct = frozen_count / len(action_changes) * 100

    small_change_threshold = 0.05
    small_change_count = np.sum(action_changes < small_change_threshold)
    small_change_pct = small_change_count / len(action_changes) * 100

    significant_changes = np.sum(action_changes >= 0.1)
    significant_pct = significant_changes / len(action_changes) * 100

    print(f"    Frozen steps (Δ < 0.01):      {frozen_pct:5.1f}% ({frozen_count}/{len(action_changes)})")
    print(f"    Small changes (Δ < 0.05):     {small_change_pct:5.1f}%")
    print(f"    Significant changes (Δ ≥ 0.1): {significant_pct:5.1f}%")
    print(f"    Mean change per step:         {np.mean(action_changes):.4f}")
    print(f"    Max change in single step:    {np.max(action_changes):.4f}")

    # ==================== Diagnosis ====================
    print("\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)

    issues = []

    # Check for mode collapse
    if std_action < 0.05:
        issues.append("MODE COLLAPSE: Std < 0.05 - Agent outputs nearly constant actions")

    # Check for frozen state
    if frozen_pct > 90:
        issues.append(f"FROZEN STATE: {frozen_pct:.1f}% of actions change < 0.01")

    # Check for extreme bias
    if abs(mean_action) > 0.8:
        direction = "LONG" if mean_action > 0 else "SHORT"
        issues.append(f"EXTREME BIAS: Mean action = {mean_action:+.2f} (always {direction})")

    # Check for lack of exploration
    if (max_action - min_action) < 0.5:
        issues.append(f"LIMITED RANGE: Actions only span {max_action - min_action:.2f} (should be ~2.0)")

    # Check for no neutral positions
    if neutral < 5:
        issues.append(f"NO NEUTRAL: Only {neutral:.1f}% neutral positions")

    if issues:
        print("\n  ISSUES DETECTED:")
        for i, issue in enumerate(issues, 1):
            print(f"    {i}. {issue}")
        verdict = "UNHEALTHY"
    else:
        print("\n  No major issues detected.")
        verdict = "HEALTHY"

    print(f"\n  VERDICT: {verdict}")

    # ==================== Sample Actions ====================
    print("\n" + "=" * 70)
    print("SAMPLE ACTIONS (first 20 steps)")
    print("=" * 70)
    print(f"\n  {actions[:20].round(4).tolist()}")

    print("\n" + "=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)

    return {
        "mean": mean_action,
        "std": std_action,
        "min": min_action,
        "max": max_action,
        "frozen_pct": frozen_pct,
        "strong_long_pct": strong_long,
        "strong_short_pct": strong_short,
        "neutral_pct": neutral,
        "verdict": verdict,
        "actions": actions,
    }


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    results = run_behavioral_audit()

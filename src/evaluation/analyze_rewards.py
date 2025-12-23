# -*- coding: utf-8 -*-
"""
analyze_rewards.py - Reward Scaling Analysis.

Analyzes the DSR reward distribution to check for saturation issues.
Captures both raw DSR values (before tanh) and final rewards (after tanh).

Usage:
    python -m src.evaluation.analyze_rewards
"""

import numpy as np
import pandas as pd
from typing import List, Tuple

from src.training.env import CryptoTradingEnv


# ============================================================================
# Modified Environment to Capture Raw DSR
# ============================================================================

class AnalysisEnv(CryptoTradingEnv):
    """Environment that captures raw DSR values before tanh squashing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.raw_dsr_history: List[float] = []
        self.final_reward_history: List[float] = []

    def _calculate_reward(self, step_return: float) -> float:
        """Override to capture raw DSR before tanh."""
        # 1. Update EMAs
        delta_A = step_return - self.A
        delta_B = step_return ** 2 - self.B
        self.A += self.eta * delta_A
        self.B += self.eta * delta_B

        # 2. Variance avec floor
        variance = self.B - self.A ** 2
        sigma = np.sqrt(max(variance, 1e-4))

        # 3. Calcul DSR brut
        dsr = (delta_A - 0.5 * self.A * delta_B) / sigma

        # CAPTURE: Raw DSR before scaling
        self.raw_dsr_history.append(dsr)

        # 4. Squashing par tanh (Factor 1.0)
        reward = float(np.tanh(dsr * 1.0))

        # CAPTURE: Final reward after tanh
        self.final_reward_history.append(reward)

        return reward

    def reset(self, *args, **kwargs):
        """Reset and clear history."""
        self.raw_dsr_history = []
        self.final_reward_history = []
        return super().reset(*args, **kwargs)


# ============================================================================
# Analysis Functions
# ============================================================================

def run_reward_analysis(n_steps: int = 1000, use_random: bool = True) -> dict:
    """
    Run reward distribution analysis.

    Args:
        n_steps: Number of steps to run.
        use_random: If True, use random actions. If False, try to load model.

    Returns:
        Dictionary with analysis results.
    """
    print("=" * 70)
    print("REWARD SCALING ANALYSIS")
    print("=" * 70)

    # ==================== Create Environment ====================
    print(f"\n[1/3] Creating analysis environment...")

    df = pd.read_parquet("data/processed_data.parquet")
    n_total = len(df)
    split_idx = int(n_total * 0.8)

    env = AnalysisEnv(
        parquet_path="data/processed_data.parquet",
        start_idx=split_idx,
        end_idx=n_total,
        window_size=64,
        commission=0.0006,
        random_start=False,
        episode_length=None,
    )

    # ==================== Run Steps ====================
    print(f"\n[2/3] Running {n_steps} steps...")

    if use_random:
        print("      Using RANDOM actions")
    else:
        print("      Using PRETRAINED model")
        try:
            from sb3_contrib import TQC
            model = TQC.load("weights/checkpoints/best_model.zip")
        except:
            print("      [FALLBACK] Model not found, using random")
            use_random = True

    obs, _ = env.reset()
    all_raw_dsr = []
    all_final_rewards = []

    for step in range(n_steps):
        if use_random:
            action = env.action_space.sample()
        else:
            action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            # Save history before reset
            all_raw_dsr.extend(env.raw_dsr_history)
            all_final_rewards.extend(env.final_reward_history)
            obs, _ = env.reset()

        if (step + 1) % 200 == 0:
            print(f"      Step {step + 1}/{n_steps}")

    # Add remaining history
    all_raw_dsr.extend(env.raw_dsr_history)
    all_final_rewards.extend(env.final_reward_history)

    raw_dsr = np.array(all_raw_dsr)
    final_rewards = np.array(all_final_rewards)

    # ==================== Analysis ====================
    print(f"\n[3/3] Analyzing {len(final_rewards)} reward samples...")

    print("\n" + "=" * 70)
    print("RAW DSR (before tanh)")
    print("=" * 70)

    print(f"\n  Statistics:")
    print(f"    Mean:     {np.mean(raw_dsr):+.4f}")
    print(f"    Std:      {np.std(raw_dsr):.4f}")
    print(f"    Min:      {np.min(raw_dsr):+.4f}")
    print(f"    Max:      {np.max(raw_dsr):+.4f}")

    # Check extreme values
    extreme_pos = np.sum(raw_dsr > 10) / len(raw_dsr) * 100
    extreme_neg = np.sum(raw_dsr < -10) / len(raw_dsr) * 100
    print(f"\n  Extreme Values:")
    print(f"    DSR > +10:  {extreme_pos:.2f}%")
    print(f"    DSR < -10:  {extreme_neg:.2f}%")

    print("\n" + "=" * 70)
    print("FINAL REWARDS (after tanh with factor 1.0)")
    print("=" * 70)

    print(f"\n  Statistics:")
    print(f"    Mean:     {np.mean(final_rewards):+.4f}")
    print(f"    Std:      {np.std(final_rewards):.4f}")
    print(f"    Min:      {np.min(final_rewards):+.4f}")
    print(f"    Max:      {np.max(final_rewards):+.4f}")

    # Saturation analysis
    sat_pos = np.sum(final_rewards > 0.95) / len(final_rewards) * 100
    sat_neg = np.sum(final_rewards < -0.95) / len(final_rewards) * 100
    sat_total = sat_pos + sat_neg

    near_zero = np.sum(np.abs(final_rewards) < 0.01) / len(final_rewards) * 100

    print(f"\n  Saturation Analysis:")
    print(f"    Rewards > +0.95:   {sat_pos:.2f}%")
    print(f"    Rewards < -0.95:   {sat_neg:.2f}%")
    print(f"    TOTAL SATURATED:   {sat_total:.2f}%")
    print(f"    Near zero (<0.01): {near_zero:.2f}%")

    # Distribution histogram
    print(f"\n  Distribution (Histogram):")
    bins = [(-1.0, -0.5), (-0.5, -0.1), (-0.1, 0.1), (0.1, 0.5), (0.5, 1.0)]
    for low, high in bins:
        count = np.sum((final_rewards >= low) & (final_rewards < high))
        pct = count / len(final_rewards) * 100
        bar = "#" * int(pct / 2)
        label = f"[{low:+.1f}, {high:+.1f})"
        print(f"    {label}: {pct:5.1f}% {bar}")

    # ==================== Diagnosis ====================
    print("\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)

    std = np.std(final_rewards)
    issues = []
    recommendations = []

    if sat_total > 50:
        issues.append(f"OVER-SATURATION: {sat_total:.1f}% rewards at ±1")
        recommendations.append("Reduce scaling factor (1.0 → 0.5 or lower)")

    if near_zero > 80:
        issues.append(f"UNDER-SCALED: {near_zero:.1f}% rewards near zero")
        recommendations.append("Increase scaling factor (1.0 → 2.0 or higher)")

    if std < 0.1:
        issues.append(f"LOW VARIANCE: Std = {std:.4f} (should be 0.3-0.5)")
        recommendations.append("Check if DSR is computing correctly")

    if std > 0.7:
        issues.append(f"HIGH VARIANCE: Std = {std:.4f} (may cause instability)")
        recommendations.append("Consider reducing scaling factor")

    if 0.2 <= std <= 0.5 and sat_total < 20 and near_zero < 50:
        print("\n  STATUS: OPTIMAL SCALING")
        print(f"  Std = {std:.4f} is in ideal range [0.2, 0.5]")
        print(f"  Saturation = {sat_total:.1f}% is acceptable (<20%)")
    else:
        print("\n  ISSUES DETECTED:")
        for i, issue in enumerate(issues, 1):
            print(f"    {i}. {issue}")

        print("\n  RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations, 1):
            print(f"    {i}. {rec}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    return {
        "raw_dsr_mean": np.mean(raw_dsr),
        "raw_dsr_std": np.std(raw_dsr),
        "reward_mean": np.mean(final_rewards),
        "reward_std": std,
        "saturation_pct": sat_total,
        "near_zero_pct": near_zero,
        "raw_dsr": raw_dsr,
        "final_rewards": final_rewards,
    }


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Reward Scaling Analysis")
    parser.add_argument("--steps", type=int, default=1000, help="Number of steps")
    parser.add_argument("--random", action="store_true", help="Use random actions")

    args = parser.parse_args()

    results = run_reward_analysis(n_steps=args.steps, use_random=args.random)

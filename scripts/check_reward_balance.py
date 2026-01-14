#!/usr/bin/env python3
"""
check_reward_balance.py - Check reward component balance with random actions.

Diagnoses whether reward penalties (churn, smooth) dominate or are dominated by PnL.
This helps ensure the agent has a balanced learning signal.

Usage:
    python scripts/check_reward_balance.py
    python scripts/check_reward_balance.py --data data/wfo/segment_0/train.parquet
"""

import argparse
import torch
import numpy as np
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.batch_env import BatchCryptoEnv


def main():
    parser = argparse.ArgumentParser(description="Check reward component balance")
    parser.add_argument("--data", type=str, default="data/processed_data.parquet",
                        help="Path to parquet data file")
    parser.add_argument("--price-col", type=str, default=None,
                        help="Price column name (auto-detected if not provided)")
    parser.add_argument("--n-envs", type=int, default=16,
                        help="Number of parallel environments")
    parser.add_argument("--n-steps", type=int, default=100,
                        help="Number of steps to run")
    parser.add_argument("--churn-coef", type=float, default=1.0,
                        help="Churn penalty coefficient")
    parser.add_argument("--smooth-coef", type=float, default=0.005,
                        help="Smoothness penalty coefficient")
    args = parser.parse_args()

    print("=" * 60)
    print("  REWARD BALANCE CHECK")
    print("=" * 60)

    # Check data file exists
    if not Path(args.data).exists():
        print(f"\n  ERROR: Data file not found: {args.data}")
        print("  Try: --data data/wfo/segment_0/train.parquet")
        return

    # Auto-detect price column
    import pandas as pd
    df = pd.read_parquet(args.data)
    if args.price_col:
        price_col = args.price_col
    else:
        # Auto-detect: 'close', 'BTC_Close', 'price', etc.
        candidates = ['close', 'BTC_Close', 'price', 'Close']
        price_col = None
        for col in candidates:
            if col in df.columns:
                price_col = col
                break
        if price_col is None:
            print(f"\n  ERROR: Could not auto-detect price column.")
            print(f"  Available columns: {df.columns.tolist()[:10]}")
            print(f"  Use --price-col to specify.")
            return

    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Device: {device}")
    print(f"  Data: {args.data}")
    print(f"  Price col: {price_col}")
    print(f"  N_envs: {args.n_envs}")
    print(f"  N_steps: {args.n_steps}")
    print(f"  churn_coef: {args.churn_coef}")
    print(f"  smooth_coef: {args.smooth_coef}")

    # Create env
    try:
        env = BatchCryptoEnv(
            parquet_path=args.data,
            price_column=price_col,
            n_envs=args.n_envs,
            device=device,
            churn_coef=args.churn_coef,
            smooth_coef=args.smooth_coef,
        )
    except Exception as e:
        print(f"\n  ERROR creating environment: {e}")
        import traceback
        traceback.print_exc()
        return

    # Collect stats
    pnl_list, churn_list, smooth_list = [], [], []

    env.reset()
    for step in range(args.n_steps):
        # Random actions
        actions = np.random.uniform(-1, 1, size=(args.n_envs, 1)).astype(np.float32)
        obs, rewards, dones, infos = env.step(actions)

        metrics = env.get_global_metrics()
        pnl_list.append(metrics["avg_rew_pnl"])
        churn_list.append(metrics["avg_rew_churn"])
        smooth_list.append(metrics["avg_rew_smooth"])

    # Statistics
    pnl = np.array(pnl_list)
    churn = np.array(churn_list)
    smooth = np.array(smooth_list)

    print("\n  COMPONENT STATISTICS ({} random steps)".format(args.n_steps))
    print("-" * 60)
    print(f"{'Component':<15} {'Mean':>12} {'Min':>12} {'Max':>12} {'Std':>12}")
    print("-" * 60)
    print(f"{'PnL':<15} {pnl.mean():>+12.4f} {pnl.min():>+12.4f} {pnl.max():>+12.4f} {pnl.std():>12.4f}")
    print(f"{'Churn':<15} {churn.mean():>+12.4f} {churn.min():>+12.4f} {churn.max():>+12.4f} {churn.std():>12.4f}")
    print(f"{'Smooth':<15} {smooth.mean():>+12.4f} {smooth.min():>+12.4f} {smooth.max():>+12.4f} {smooth.std():>12.4f}")
    print("-" * 60)

    # Total reward
    total = pnl + churn + smooth
    print(f"{'TOTAL':<15} {total.mean():>+12.4f} {total.min():>+12.4f} {total.max():>+12.4f} {total.std():>12.4f}")
    print("-" * 60)

    # Ratios
    print("\n  AMPLITUDE RATIOS")
    print("-" * 60)
    total_penalty = abs(churn.mean()) + abs(smooth.mean())
    pnl_abs = abs(pnl.mean())
    ratio = pnl_abs / total_penalty if total_penalty > 0 else float('inf')

    churn_smooth_ratio = abs(churn.mean()) / abs(smooth.mean()) if smooth.mean() != 0 else float('inf')

    print(f"  |PnL| / |Penalties| = {ratio:.2f}")
    print(f"  |Churn| / |Smooth|  = {churn_smooth_ratio:.2f}")

    # Interpretation
    print("\n  DIAGNOSTIC")
    print("-" * 60)

    if ratio < 0.5:
        print("  [WARNING] Penalties dominate PnL significantly!")
        print("     L'agent risque d'apprendre a NE PAS trader.")
        print("     Suggestion: Reduire churn_coef ou smooth_coef.")
    elif ratio < 1:
        print("  [ATTENTION] Penalties slightly dominate PnL.")
        print("     L'agent sera penalise pour chaque trade.")
        print("     Peut etre OK si on veut un agent conservateur.")
    elif ratio > 10:
        print("  [WARNING] PnL domine les penalties!")
        print("     Les penalties n'ont pas d'effet significatif.")
        print("     Suggestion: Augmenter churn_coef ou smooth_coef.")
    elif ratio > 5:
        print("  [ATTENTION] PnL domine les penalties.")
        print("     L'agent peut ignorer les couts de transaction.")
    else:
        print("  [OK] Balance correcte: PnL et penalties sont du meme ordre.")
        print("     L'agent peut apprendre a trader de maniere rentable")
        print("     tout en etant penalise pour le churn excessif.")

    # Signal-to-noise
    snr = abs(pnl.mean()) / pnl.std() if pnl.std() > 0 else 0
    print(f"\n  Signal-to-Noise (PnL): {snr:.3f}")
    if snr < 0.1:
        print("     [WARNING] SNR tres faible - signal noye dans le bruit.")
    elif snr < 0.5:
        print("     [OK] SNR raisonnable pour RL.")
    else:
        print("     [EXCELLENT] SNR eleve - signal clair.")

    env.close()
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
run_oracle_test.py - Run the Oracle Test to validate TQC architecture.

If the model can profit on OBVIOUS signals, the architecture works.
If not, the architecture is broken.

Usage:
    python scripts/run_oracle_test.py
    python scripts/run_oracle_test.py --timesteps 100000
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from pathlib import Path

from src.training.batch_env import BatchCryptoEnv
from src.training.train_agent import create_policy_kwargs
from src.config.training import TQCTrainingConfig
from sb3_contrib import TQC

# Check for TQCDropoutPolicy
try:
    from src.models.tqc_dropout_policy import TQCDropoutPolicy
    USE_DROPOUT_POLICY = True
except ImportError:
    USE_DROPOUT_POLICY = False
    print("[WARNING] TQCDropoutPolicy not found, using default TQC policy")


def run_oracle_test(
    timesteps: int = 100_000,
    n_envs: int = 64,
    target_volatility: float = 0.20,  # More permissive for oracle
    log_dir: str = "logs/oracle_test",
):
    """Run the Oracle Test."""

    print("=" * 70)
    print("ORACLE TEST - Architecture Validation")
    print("=" * 70)
    print(f"\nConfig:")
    print(f"  Timesteps: {timesteps:,}")
    print(f"  Envs: {n_envs}")
    print(f"  Target Vol: {target_volatility}")
    print(f"  Log dir: {log_dir}")

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    # Check oracle data exists
    oracle_path = "data/oracle_dataset.parquet"
    if not os.path.exists(oracle_path):
        print(f"\nERROR: Oracle data not found at {oracle_path}")
        print("Run: python scripts/make_oracle_data.py")
        return

    print(f"\n[1/4] Creating Oracle Environment...")

    # Create training env with oracle data
    train_env = BatchCryptoEnv(
        parquet_path=oracle_path,
        price_column="BTC_Close",
        n_envs=n_envs,
        device=device,
        window_size=32,  # Smaller window for simplicity
        episode_length=500,  # Shorter episodes
        initial_balance=10_000.0,
        commission=0.0001,  # Low commission
        slippage=0.0,
        target_volatility=target_volatility,
        max_leverage=3.0,  # Allow bigger positions
        observation_noise=0.0,  # No noise - we want to overfit!
        enable_domain_randomization=False,  # No randomization
        action_discretization=0.0,  # Continuous actions
    )

    # Create eval env (same settings)
    eval_env = BatchCryptoEnv(
        parquet_path=oracle_path,
        price_column="BTC_Close",
        n_envs=4,
        device=device,
        window_size=32,
        episode_length=500,
        initial_balance=10_000.0,
        commission=0.0001,
        slippage=0.0,
        target_volatility=target_volatility,
        max_leverage=3.0,
        observation_noise=0.0,
        enable_domain_randomization=False,
        action_discretization=0.0,
        random_start=False,  # Deterministic for eval
    )

    print(f"  Observation space: {train_env.observation_space}")
    print(f"  Action space: {train_env.action_space}")

    print(f"\n[2/4] Creating TQC Model...")

    # Simple config - no fancy features, just raw learning
    config = TQCTrainingConfig()
    config.learning_rate = 3e-4
    config.buffer_size = 100_000
    config.batch_size = 256
    config.gamma = 0.99
    config.tau = 0.005
    config.ent_coef = "auto"  # Let it converge naturally
    config.train_freq = 1
    config.gradient_steps = 1
    config.use_sde = False  # Disable gSDE for this test
    config.n_critics = 2
    config.n_quantiles = 25
    config.top_quantiles_to_drop = 2

    # Simple network
    policy_kwargs = dict(
        net_arch=dict(pi=[64, 64], qf=[64, 64]),
        n_critics=config.n_critics,
        n_quantiles=config.n_quantiles,
    )

    # Select policy
    policy_class = "MlpPolicy"

    model = TQC(
        policy=policy_class,
        env=train_env,
        learning_rate=config.learning_rate,
        buffer_size=config.buffer_size,
        batch_size=config.batch_size,
        gamma=config.gamma,
        tau=config.tau,
        ent_coef=config.ent_coef,
        train_freq=config.train_freq,
        gradient_steps=config.gradient_steps,
        top_quantiles_to_drop_per_net=config.top_quantiles_to_drop,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=log_dir,
        device=device,
    )

    print(f"  Policy: {policy_class}")
    print(f"  Parameters: {sum(p.numel() for p in model.policy.parameters()):,}")

    print(f"\n[3/4] Training on Oracle Data...")
    print("-" * 70)

    # Custom callback to monitor positions
    from stable_baselines3.common.callbacks import BaseCallback

    class OracleMonitorCallback(BaseCallback):
        def __init__(self, check_freq=5000, verbose=1):
            super().__init__(verbose)
            self.check_freq = check_freq
            self.positions = []
            self.rewards = []

        def _on_step(self) -> bool:
            if self.num_timesteps % self.check_freq == 0:
                # Get recent positions from env
                env = self.training_env.envs[0] if hasattr(self.training_env, 'envs') else self.training_env
                if hasattr(env, 'position_pcts'):
                    pos = env.position_pcts.mean().item()
                    nav = env._get_navs().mean().item()
                    self.positions.append(abs(pos))

                    print(f"Step {self.num_timesteps:>7} | "
                          f"Pos: {pos:>+6.3f} | "
                          f"NAV: {nav:>10.2f} | "
                          f"Mean |Pos|: {np.mean(self.positions[-10:]):.3f}")
            return True

    callback = OracleMonitorCallback(check_freq=5000)

    # Train
    model.learn(
        total_timesteps=timesteps,
        callback=callback,
        progress_bar=True,
    )

    print("-" * 70)
    print(f"\n[4/4] Evaluation...")

    # Evaluate
    obs = eval_env.reset()
    total_reward = 0
    positions = []
    navs = []

    for _ in range(500):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        total_reward += reward.mean()
        positions.append(eval_env.position_pcts.mean().item())
        navs.append(eval_env._get_navs().mean().item())

        if done.any():
            break

    # Calculate metrics
    mean_position = np.mean(np.abs(positions))
    final_nav = navs[-1] if navs else 10000
    returns = np.diff(navs) / np.array(navs[:-1]) if len(navs) > 1 else [0]
    sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252 * 24)  # Hourly

    print(f"\n" + "=" * 70)
    print("ORACLE TEST RESULTS")
    print("=" * 70)
    print(f"  Mean |Position|: {mean_position:.4f}")
    print(f"  Final NAV: {final_nav:.2f}")
    print(f"  Total Return: {(final_nav/10000 - 1)*100:.2f}%")
    print(f"  Sharpe Ratio: {sharpe:.2f}")
    print("=" * 70)

    # Verdict
    print("\nVERDICT:")
    if mean_position > 0.5 and sharpe > 1.0:
        print("  [SUCCESS] The model CAN learn with clear signals!")
        print("  -> Problem is in FEATURES, not architecture")
    elif mean_position > 0.3:
        print("  [PARTIAL] Model trades but not optimally")
        print("  -> Architecture may need tuning")
    else:
        print("  [FAILURE] Model stuck at zero even with perfect signal")
        print("  -> Architecture is BROKEN")

    # Save model
    model.save(f"{log_dir}/oracle_model")
    print(f"\nModel saved to: {log_dir}/oracle_model.zip")

    return model, sharpe, mean_position


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Oracle Test")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--n-envs", type=int, default=64)
    parser.add_argument("--target-vol", type=float, default=0.20)
    args = parser.parse_args()

    run_oracle_test(
        timesteps=args.timesteps,
        n_envs=args.n_envs,
        target_volatility=args.target_vol,
    )

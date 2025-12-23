# -*- coding: utf-8 -*-
"""
tune.py - Optuna Hyperparameter Tuning for TQC Agent (GENIUS CATCHER MODE).

Tracks BEST mean reward during training, not final value.
This catches "ephemeral geniuses" - agents that peak early then collapse.

Key features:
- EvalCallback tracks best_mean_reward throughout training
- Returns peak performance, not final (catastrophic forgetting resistant)
- Patient pruning (20k warmup) to let slow strategies emerge

Usage:
    python -m src.optimization.tune --n-trials 50
"""

import os
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from sb3_contrib import TQC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from torch.optim import AdamW

# Suppress warnings during optimization
warnings.filterwarnings("ignore", category=UserWarning)

from src.config import DEVICE, SEED
from src.models.rl_adapter import FoundationFeatureExtractor
from src.training.env import CryptoTradingEnv


# ============================================================================
# Configuration
# ============================================================================

class TuningConfig:
    """Configuration for hyperparameter tuning."""

    # Paths
    data_path: str = "data/processed_data.parquet"
    encoder_path: str = "weights/pretrained_encoder.pth"
    output_dir: str = "results/"
    study_db: str = "sqlite:///optuna_study.db"

    # Environment
    window_size: int = 64
    commission: float = 0.0006
    train_ratio: float = 0.8
    episode_length: int = 2048

    # Foundation Model (must match pretrained encoder)
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2

    # Fixed TQC parameters (not tuned)
    buffer_size: int = 100_000
    top_quantiles_to_drop: int = 2
    n_critics: int = 2
    n_quantiles: int = 25
    net_arch: list = [256, 256]
    freeze_encoder: bool = True
    use_sde: bool = True
    sde_sample_freq: int = -1
    use_sde_at_warmup: bool = True

    # Tuning parameters (GENIUS CATCHER MODE)
    trial_timesteps: int = 60_000  # Longer trials for slow strategies
    n_trials: int = 50
    eval_freq: int = 5000         # Eval every 5k steps
    n_eval_episodes: int = 1      # Fast: 1 episode per eval


# ============================================================================
# Pruning Callback
# ============================================================================

class TrialPruningCallback(BaseCallback):
    """
    Callback for Optuna pruning integration.
    Reports intermediate values and checks for pruning.
    """

    def __init__(
        self,
        trial: optuna.Trial,
        eval_freq: int = 5000,
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.trial = trial
        self.eval_freq = eval_freq
        self.episode_rewards = []
        self.is_pruned = False

    def _on_step(self) -> bool:
        # Collect episode rewards
        if self.locals.get("infos"):
            for info in self.locals["infos"]:
                if "episode" in info:
                    self.episode_rewards.append(info["episode"]["r"])

        # Report to Optuna every eval_freq steps
        if self.num_timesteps % self.eval_freq == 0 and self.episode_rewards:
            mean_reward = np.mean(self.episode_rewards[-10:])

            # Check for NaN/Inf (explosion)
            if not np.isfinite(mean_reward):
                self.is_pruned = True
                raise optuna.TrialPruned()

            # Report intermediate value
            self.trial.report(mean_reward, self.num_timesteps)

            # Check if trial should be pruned
            if self.trial.should_prune():
                self.is_pruned = True
                raise optuna.TrialPruned()

        return True


# ============================================================================
# Helper Functions
# ============================================================================

def calculate_sharpe_ratio(returns: np.ndarray, periods_per_year: int = 252 * 24) -> float:
    """Calculate annualized Sharpe Ratio."""
    if len(returns) < 2 or np.std(returns) == 0:
        return 0.0
    sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(periods_per_year))
    return sharpe if np.isfinite(sharpe) else -100.0


def create_environments(config: TuningConfig):
    """Create training and validation environments (both vectorized)."""
    train_env, val_env = CryptoTradingEnv.create_train_val_envs(
        parquet_path=config.data_path,
        train_ratio=config.train_ratio,
        window_size=config.window_size,
        commission=config.commission,
        episode_length=config.episode_length,
    )

    # Training env with Monitor for episode stats
    train_env_monitored = Monitor(train_env)
    train_vec_env = DummyVecEnv([lambda: train_env_monitored])

    # Validation env vectorized for EvalCallback
    val_env_monitored = Monitor(val_env)
    val_vec_env = DummyVecEnv([lambda: val_env_monitored])

    return train_vec_env, val_vec_env


def create_policy_kwargs(config: TuningConfig) -> dict:
    """Create policy kwargs with FoundationFeatureExtractor."""
    return dict(
        features_extractor_class=FoundationFeatureExtractor,
        features_extractor_kwargs=dict(
            encoder_path=config.encoder_path,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            freeze_encoder=config.freeze_encoder,
        ),
        net_arch=config.net_arch,
        n_critics=config.n_critics,
        n_quantiles=config.n_quantiles,
        optimizer_class=AdamW,
        optimizer_kwargs=dict(
            weight_decay=1e-4,
            eps=1e-5,
        ),
    )


# ============================================================================
# Objective Function
# ============================================================================

def objective(trial: optuna.Trial, config: TuningConfig) -> float:
    """
    Optuna objective function (GENIUS CATCHER).

    Samples hyperparameters, trains TQC for trial_timesteps,
    and returns the BEST mean reward achieved during training.
    This catches ephemeral geniuses that peak early then collapse.
    """
    # ==================== Sample Hyperparameters (SMART & SAFE) ====================
    # LR capped at 1e-4 (signal is now x10 stronger with reward_scaling=1.0)
    lr = trial.suggest_float("lr", 1e-6, 1e-4, log=True)
    # Long-term horizon
    gamma = trial.suggest_categorical("gamma", [0.98, 0.99, 0.995, 0.999])
    # Fixed tau for stability
    tau = 0.005
    # Exploration-friendly entropy
    ent_coef = trial.suggest_categorical("ent_coef", ["auto", 0.005, 0.01])
    # Simple gradient steps
    gradient_steps = trial.suggest_categorical("gradient_steps", [1, 2])
    # Batch size for stability
    batch_size = trial.suggest_categorical("batch_size", [128, 256])

    print(f"\n{'='*60}")
    print(f"Trial {trial.number} (GENIUS CATCHER)")
    print(f"{'='*60}")
    print(f"  lr={lr:.2e}, gamma={gamma}, tau={tau}")
    print(f"  ent_coef={ent_coef}, gradient_steps={gradient_steps}, batch={batch_size}")

    try:
        # ==================== Create Environments ====================
        train_env, val_vec_env = create_environments(config)

        # ==================== Create Model ====================
        policy_kwargs = create_policy_kwargs(config)

        model = TQC(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=lr,
            buffer_size=config.buffer_size,
            batch_size=batch_size,
            gamma=gamma,
            tau=tau,
            ent_coef=ent_coef,
            train_freq=1,
            gradient_steps=gradient_steps,
            top_quantiles_to_drop_per_net=config.top_quantiles_to_drop,
            use_sde=config.use_sde,
            sde_sample_freq=config.sde_sample_freq,
            use_sde_at_warmup=config.use_sde_at_warmup,
            policy_kwargs=policy_kwargs,
            verbose=0,
            seed=SEED,
            device=DEVICE,
        )

        # ==================== Callbacks ====================
        # EvalCallback: tracks best_mean_reward on validation env
        eval_callback = EvalCallback(
            val_vec_env,
            eval_freq=config.eval_freq,           # Evaluate every 5k steps
            n_eval_episodes=config.n_eval_episodes,  # Fast: 1 episode per eval
            deterministic=True,                   # Deterministic policy for eval
            verbose=0,
        )

        # Pruning callback for Optuna integration
        pruning_callback = TrialPruningCallback(
            trial=trial,
            eval_freq=10000,  # Report every 10k steps (matches pruner interval)
        )

        # Combine callbacks
        callbacks = CallbackList([eval_callback, pruning_callback])

        # ==================== Train ====================
        model.learn(
            total_timesteps=config.trial_timesteps,
            callback=callbacks,
            progress_bar=False,
        )

        # ==================== Return BEST Mean Reward ====================
        # This is the key: we return the PEAK performance, not final
        best_reward = eval_callback.best_mean_reward

        print(f"  -> Best Mean Reward: {best_reward:.4f}")

        # Handle NaN/Inf
        if not np.isfinite(best_reward):
            return -100.0

        return best_reward

    except optuna.TrialPruned:
        print("  -> PRUNED (divergence detected)")
        raise

    except Exception as e:
        print(f"  -> ERROR: {e}")
        return -100.0


# ============================================================================
# Main Optimization Loop
# ============================================================================

def run_optimization(config: TuningConfig = None, n_trials: int = None) -> optuna.Study:
    """
    Run Optuna hyperparameter optimization.

    Args:
        config: Tuning configuration.
        n_trials: Number of trials (overrides config).

    Returns:
        Optuna Study object.
    """
    if config is None:
        config = TuningConfig()

    if n_trials is not None:
        config.n_trials = n_trials

    print("=" * 70)
    print("OPTUNA HYPERPARAMETER TUNING (GENIUS CATCHER)")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Trial timesteps: {config.trial_timesteps:,}")
    print(f"  Number of trials: {config.n_trials}")
    print(f"  Eval freq: every {config.eval_freq:,} steps")
    print(f"  Eval episodes: {config.n_eval_episodes}")
    print(f"  Device: {DEVICE}")
    print(f"\n  Mode: Returns BEST mean reward (not final)")

    # Create output directory
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    # ==================== Create Study (SMART & SAFE) ====================
    sampler = TPESampler(seed=SEED)
    # Patient pruning: don't judge before 20k steps
    pruner = MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=20000,  # Wait 20k steps before pruning
        interval_steps=10000,  # Check every 10k steps
    )

    study = optuna.create_study(
        study_name="tqc_genius_catcher",  # Tracks best_mean_reward (not final)
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=config.study_db,
        load_if_exists=True,
    )

    # ==================== Optimize ====================
    print(f"\nStarting optimization...")
    print(f"{'='*70}\n")

    study.optimize(
        lambda trial: objective(trial, config),
        n_trials=config.n_trials,
        catch=(Exception,),
        show_progress_bar=True,
    )

    # ==================== Results ====================
    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)

    print(f"\nBest Trial: {study.best_trial.number}")
    print(f"Best Mean Reward: {study.best_value:.4f}")
    print(f"\nBest Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # ==================== Save Results ====================
    # Save all trials to CSV
    df = study.trials_dataframe()
    csv_path = Path(config.output_dir) / "study_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    # Save best params to separate file
    best_params_path = Path(config.output_dir) / "best_params.txt"
    with open(best_params_path, "w") as f:
        f.write(f"Best Trial: {study.best_trial.number}\n")
        f.write(f"Best Mean Reward: {study.best_value:.4f}\n\n")
        f.write("Best Hyperparameters:\n")
        for key, value in study.best_params.items():
            f.write(f"  {key}: {value}\n")
    print(f"Best params saved to: {best_params_path}")

    return study


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Optuna Hyperparameter Tuning for TQC (GENIUS CATCHER MODE)")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of trials")
    parser.add_argument("--timesteps", type=int, default=60_000, help="Steps per trial (60k for slow strategies)")

    args = parser.parse_args()

    config = TuningConfig()
    config.trial_timesteps = args.timesteps

    study = run_optimization(config, n_trials=args.n_trials)

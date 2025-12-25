# -*- coding: utf-8 -*-
"""
train_agent.py - TQC Agent Training with Foundation Model.

Trains a TQC agent using the pre-trained CryptoMAE encoder as feature extractor.
The encoder is frozen by default to preserve learned market representations.

Usage:
    python -m src.training.train_agent
"""

import os
from typing import Callable, Union

from sb3_contrib import TQC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
import torch
import numpy as np

from src.training.clipped_optimizer import ClippedAdamW


class StepLoggingCallback(BaseCallback):
    """
    Callback pour logger à chaque N steps (console + TensorBoard).
    """
    def __init__(self, log_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.last_episode_info = {}

    def _on_step(self) -> bool:
        # Collecter les infos d'épisode si disponibles
        if self.locals.get("infos"):
            for info in self.locals["infos"]:
                if "episode" in info:
                    ep_reward = info["episode"]["r"]
                    ep_length = info["episode"]["l"]
                    self.episode_rewards.append(ep_reward)
                    self.episode_lengths.append(ep_length)
                    self.last_episode_info = {
                        "reward": ep_reward,
                        "length": ep_length,
                    }

                # Collecter les infos de trading (NAV, position, etc.)
                if "nav" in info:
                    self.last_episode_info["nav"] = info["nav"]
                if "position_pct" in info:
                    self.last_episode_info["position"] = info["position_pct"]
                if "total_trades" in info:
                    self.last_episode_info["trades"] = info["total_trades"]

        # Logger tous les log_freq steps
        if self.num_timesteps % self.log_freq == 0:
            # Calculer les stats des derniers épisodes
            if self.episode_rewards:
                mean_reward = np.mean(self.episode_rewards[-10:])
                mean_length = np.mean(self.episode_lengths[-10:])
            else:
                mean_reward = 0
                mean_length = 0

            # Log to TensorBoard
            self.logger.record("custom/mean_reward_10ep", mean_reward)
            self.logger.record("custom/mean_length_10ep", mean_length)

            if self.last_episode_info:
                if "nav" in self.last_episode_info:
                    self.logger.record("custom/nav", self.last_episode_info["nav"])
                if "position" in self.last_episode_info:
                    self.logger.record("custom/position", self.last_episode_info["position"])
                if "trades" in self.last_episode_info:
                    self.logger.record("custom/total_trades", self.last_episode_info["trades"])

            # Force dump to TensorBoard
            self.logger.dump(self.num_timesteps)

            # Console log
            fps = self.model.logger.name_to_value.get("time/fps", 0)
            nav = self.last_episode_info.get("nav", 0)
            pos = self.last_episode_info.get("position", 0)

            print(f"Step {self.num_timesteps:>7} | "
                  f"Reward: {mean_reward:>8.2f} | "
                  f"NAV: {nav:>10.2f} | "
                  f"Pos: {pos:>+5.2f} | "
                  f"FPS: {fps:>5.0f}")

        return True


from src.config import DEVICE, SEED
from src.models.rl_adapter import FoundationFeatureExtractor
from src.training.env import CryptoTradingEnv


# ============================================================================
# Configuration
# ============================================================================

class TrainingConfig:
    """Training configuration for TQC with Foundation Model."""

    # Paths
    data_path: str = "data/processed_data.parquet"
    encoder_path: str = "weights/pretrained_encoder.pth"
    save_path: str = "weights/tqc_agent_final.zip"
    tensorboard_log: str = "logs/tensorboard_tqc/"
    checkpoint_dir: str = "weights/checkpoints/"

    # Environment
    window_size: int = 64
    commission: float = 0.0006  # 0.06%
    train_ratio: float = 0.8
    episode_length: int = 2048  # Épisodes plus courts pour tracking des rewards

    # Foundation Model (must match pretrained encoder)
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2

    # TQC Hyperparameters (Ultra Safe Baseline)
    total_timesteps: int = 500_000
    learning_rate: float = 5e-6  # Ultra Safe: priorité stabilité
    buffer_size: int = 200_000
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.05
    ent_coef: Union[str, float] = "auto"  # Auto-tuned entropy
    train_freq: int = 1
    gradient_steps: int = 1  # Ultra Safe: 1 update per step
    top_quantiles_to_drop: int = 2
    n_critics: int = 2
    n_quantiles: int = 25

    # Policy Network
    net_arch: list = [256, 256]
    freeze_encoder: bool = True

    # gSDE (State-Dependent Exploration)
    use_sde: bool = True
    sde_sample_freq: int = -1  # -1 = resample noise once per episode
    use_sde_at_warmup: bool = True

    # Callbacks
    eval_freq: int = 5_000  # Same as tuning - catches peak performance
    checkpoint_freq: int = 50_000
    log_freq: int = 100  # Log every N steps


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    Args:
        initial_value: Initial learning rate.

    Returns:
        Callable that maps progress (1.0 -> 0.0) to learning rate.
    """
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


def create_policy_kwargs(config: TrainingConfig) -> dict:
    """
    Create policy kwargs with FoundationFeatureExtractor.

    Args:
        config: Training configuration.

    Returns:
        Dict of policy keyword arguments.
    """
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
        optimizer_class=ClippedAdamW,
        optimizer_kwargs=dict(
            max_grad_norm=0.5,  # Gradient clipping intégré
            weight_decay=1e-4,
            eps=1e-5,
        ),
    )


def create_environments(config: TrainingConfig):
    """
    Create training and evaluation environments.

    Args:
        config: Training configuration.

    Returns:
        Tuple of (train_vec_env, eval_vec_env).
    """
    # Create train/val environments
    train_env, val_env = CryptoTradingEnv.create_train_val_envs(
        parquet_path=config.data_path,
        train_ratio=config.train_ratio,
        window_size=config.window_size,
        commission=config.commission,
        episode_length=config.episode_length,
    )

    # Wrap in Monitor for episode tracking
    train_env_monitored = Monitor(train_env)
    val_env_monitored = Monitor(val_env)

    # Vectorize for SB3
    train_vec_env = DummyVecEnv([lambda: train_env_monitored])
    eval_vec_env = DummyVecEnv([lambda: val_env_monitored])

    return train_vec_env, eval_vec_env


def create_callbacks(config: TrainingConfig, eval_env) -> list:
    """
    Create training callbacks.

    Args:
        config: Training configuration.
        eval_env: Evaluation environment.

    Returns:
        List of callbacks.
    """
    callbacks = []

    # Step logging callback
    step_callback = StepLoggingCallback(log_freq=config.log_freq)
    callbacks.append(step_callback)

    # Note: Gradient clipping is now handled by ClippedAdamW optimizer

    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=config.checkpoint_dir,
        log_path="logs/",
        eval_freq=config.eval_freq,
        deterministic=True,
        render=False,
        verbose=1,
    )
    callbacks.append(eval_callback)

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=config.checkpoint_freq,
        save_path=config.checkpoint_dir,
        name_prefix="tqc_foundation",
        verbose=1,
    )
    callbacks.append(checkpoint_callback)

    return callbacks


def train(config: TrainingConfig = None) -> TQC:
    """
    Train TQC agent with Foundation Model feature extractor.

    Args:
        config: Training configuration (uses default if None).

    Returns:
        Trained TQC model.
    """
    if config is None:
        config = TrainingConfig()

    print("=" * 70)
    print("TQC + Foundation Model Training")
    print("=" * 70)

    # Create directories
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.tensorboard_log, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # ==================== Environment Setup ====================
    print("\n[1/4] Creating environments...")
    train_env, eval_env = create_environments(config)

    obs_shape = train_env.observation_space.shape
    print(f"      Observation space: {obs_shape}")
    print(f"      Action space: {train_env.action_space.shape}")
    print(f"      Device: {DEVICE}")

    # ==================== Policy Setup ====================
    print("\n[2/4] Configuring policy with FoundationFeatureExtractor...")
    policy_kwargs = create_policy_kwargs(config)

    print(f"      Encoder: {config.encoder_path}")
    print(f"      Frozen: {config.freeze_encoder}")
    print(f"      Net arch: {config.net_arch}")
    print(f"      features_dim: {obs_shape[0]} * {config.d_model} = {obs_shape[0] * config.d_model}")
    print(f"      gSDE: {config.use_sde} (sample_freq={config.sde_sample_freq})")

    # ==================== Model Creation ====================
    print("\n[3/4] Creating TQC model...")
    model = TQC(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=linear_schedule(config.learning_rate),
        buffer_size=config.buffer_size,
        batch_size=config.batch_size,
        gamma=config.gamma,
        tau=config.tau,
        ent_coef=config.ent_coef,
        train_freq=config.train_freq,
        gradient_steps=config.gradient_steps,
        top_quantiles_to_drop_per_net=config.top_quantiles_to_drop,
        use_sde=config.use_sde,
        sde_sample_freq=config.sde_sample_freq,
        use_sde_at_warmup=config.use_sde_at_warmup,
        policy_kwargs=policy_kwargs,
        tensorboard_log=config.tensorboard_log,
        verbose=1,
        seed=SEED,
        device=DEVICE,
    )

    print(f"      Total parameters: {sum(p.numel() for p in model.policy.parameters()):,}")
    print(f"      Trainable parameters: {sum(p.numel() for p in model.policy.parameters() if p.requires_grad):,}")

    # ==================== Training ====================
    print("\n[4/4] Starting training...")
    print(f"      Total timesteps: {config.total_timesteps:,}")
    print(f"      Eval frequency: {config.eval_freq:,}")
    print(f"      Checkpoint frequency: {config.checkpoint_freq:,}")
    print("\n" + "=" * 70)

    callbacks = create_callbacks(config, eval_env)

    model.learn(
        total_timesteps=config.total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    # ==================== Save Final Model ====================
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)

    model.save(config.save_path)
    print(f"\nModel saved to: {config.save_path}")

    print("\nArtifacts:")
    print(f"  - Final model: {config.save_path}")
    print(f"  - Best model: {config.checkpoint_dir}best_model.zip")
    print(f"  - Checkpoints: {config.checkpoint_dir}")
    print(f"  - TensorBoard: {config.tensorboard_log}")

    print("\nTo view training progress:")
    print(f"  tensorboard --logdir={config.tensorboard_log}")

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train TQC agent with Foundation Model")
    parser.add_argument("--timesteps", type=int, default=500_000, help="Total timesteps")
    parser.add_argument("--log-freq", type=int, default=100, help="Log frequency (steps)")
    parser.add_argument("--eval-freq", type=int, default=5_000, help="Eval frequency (steps)")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate (Ultra Safe)")

    args = parser.parse_args()

    config = TrainingConfig()
    config.total_timesteps = args.timesteps
    config.log_freq = args.log_freq
    config.eval_freq = args.eval_freq
    config.learning_rate = args.lr

    train(config)

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

# CUDA Optimization: Auto-tune convolutions for repeated input sizes
torch.backends.cudnn.benchmark = True

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
        self.episode_trades = []  # Track trades per episode
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
                    # Capture final trades at episode end
                    if "total_trades" in info:
                        self.episode_trades.append(info["total_trades"])
                    self.last_episode_info = {
                        "reward": ep_reward,
                        "length": ep_length,
                    }

                # Collecter les infos de trading (NAV, position - updated every step)
                if "nav" in info:
                    self.last_episode_info["nav"] = info["nav"]
                if "position_pct" in info:
                    self.last_episode_info["position"] = info["position_pct"]

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

            # Log trades of last completed episode
            if self.episode_trades:
                self.logger.record("custom/trades_per_episode", self.episode_trades[-1])

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


class DetailTensorboardCallback(BaseCallback):
    """
    Callback pour logger les composantes du reward dans TensorBoard.
    Permet de visualiser: log_return, penalty_vol, churn_penalty, total_raw.
    Inclut des stats épisode pour analyse du coefficient de churn optimal.
    Capture aussi les métriques de diagnostic pour l'analyse post-training.
    """
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        # Accumulateurs pour stats épisode
        self.episode_churn_penalties = []
        self.episode_log_returns = []
        self.episode_position_deltas = []

        # Métriques de diagnostic (accumulées sur tout le training)
        self.all_actions = []
        self.entropy_values = []
        self.critic_losses = []
        self.actor_losses = []
        self.churn_ratios = []

        # Gradient norms pour diagnostic de maximum local
        self.actor_grad_norms = []
        self.critic_grad_norms = []

    def _on_step(self) -> bool:
        # Track actions for saturation analysis
        if self.locals.get("actions") is not None:
            self.all_actions.extend(np.abs(self.locals["actions"]).flatten())

        # Récupérer les infos depuis l'environnement
        if self.locals.get("infos"):
            info = self.locals["infos"][0]

            # Log reward components (every step) - use record_mean for smooth curves
            if "rewards/log_return" in info:
                self.logger.record_mean("rewards/log_return", info["rewards/log_return"])
                self.episode_log_returns.append(info["rewards/log_return"])
            if "rewards/penalty_vol" in info:
                self.logger.record_mean("rewards/penalty_vol", info["rewards/penalty_vol"])
            if "rewards/churn_penalty" in info:
                self.logger.record_mean("rewards/churn_penalty", info["rewards/churn_penalty"])
                self.episode_churn_penalties.append(info["rewards/churn_penalty"])
            if "rewards/smoothness_penalty" in info:
                self.logger.record_mean("rewards/smoothness_penalty", info["rewards/smoothness_penalty"])
            if "rewards/position_delta" in info:
                self.logger.record_mean("rewards/position_delta", info["rewards/position_delta"])
                self.episode_position_deltas.append(info["rewards/position_delta"])
            if "rewards/total_raw" in info:
                self.logger.record_mean("rewards/total_raw", info["rewards/total_raw"])
            if "rewards/scaled" in info:
                self.logger.record_mean("rewards/scaled", info["rewards/scaled"])

            # À la fin d'un épisode, log les stats agrégées
            if "episode" in info:
                if self.episode_churn_penalties:
                    total_churn = sum(self.episode_churn_penalties)
                    total_log_ret = sum(self.episode_log_returns)
                    total_delta = sum(self.episode_position_deltas)

                    self.logger.record("churn/episode_total_penalty", total_churn)
                    self.logger.record("churn/episode_total_log_return", total_log_ret)
                    self.logger.record("churn/episode_total_position_delta", total_delta)

                    # Ratio churn/return (pour calibrer le coefficient)
                    if abs(total_log_ret) > 1e-8:
                        ratio = abs(total_churn / total_log_ret)
                        self.logger.record("churn/penalty_to_return_ratio", ratio)
                        self.churn_ratios.append(ratio)

                # Reset accumulateurs épisode
                self.episode_churn_penalties = []
                self.episode_log_returns = []
                self.episode_position_deltas = []

        # Capture training metrics from model logger (safe access)
        try:
            if hasattr(self.model, 'logger') and self.model.logger is not None:
                name_to_value = self.model.logger.name_to_value
                if "train/ent_coef" in name_to_value:
                    self.entropy_values.append(name_to_value["train/ent_coef"])
                if "train/critic_loss" in name_to_value:
                    self.critic_losses.append(name_to_value["train/critic_loss"])
                if "train/actor_loss" in name_to_value:
                    self.actor_losses.append(name_to_value["train/actor_loss"])
        except (KeyError, AttributeError):
            pass

        # Log gradient norms pour diagnostic de convergence/maximum local
        try:
            if hasattr(self.model, 'actor') and self.model.actor is not None:
                actor_grad_norm = self._compute_grad_norm(self.model.actor)
                if actor_grad_norm is not None and actor_grad_norm > 0:
                    self.logger.record_mean("grad/actor_norm", actor_grad_norm)
                    self.actor_grad_norms.append(actor_grad_norm)

            if hasattr(self.model, 'critic') and self.model.critic is not None:
                critic_grad_norm = self._compute_grad_norm(self.model.critic)
                if critic_grad_norm is not None and critic_grad_norm > 0:
                    self.logger.record_mean("grad/critic_norm", critic_grad_norm)
                    self.critic_grad_norms.append(critic_grad_norm)
        except Exception:
            pass

        return True

    def _compute_grad_norm(self, model) -> float:
        """Compute the L2 norm of gradients for a model."""
        total_norm = 0.0
        n_params = 0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
                n_params += 1
        if n_params == 0:
            return None
        return total_norm ** 0.5

    def get_training_metrics(self) -> dict:
        """Return diagnostic metrics at end of training."""
        return {
            "action_saturation": float(np.mean(self.all_actions)) if self.all_actions else 0.0,
            "avg_entropy": float(np.mean(self.entropy_values)) if self.entropy_values else 0.0,
            "avg_critic_loss": float(np.mean(self.critic_losses)) if self.critic_losses else 0.0,
            "avg_actor_loss": float(np.mean(self.actor_losses)) if self.actor_losses else 0.0,
            "avg_churn_ratio": float(np.mean(self.churn_ratios)) if self.churn_ratios else 0.0,
            "avg_actor_grad_norm": float(np.mean(self.actor_grad_norms)) if self.actor_grad_norms else 0.0,
            "avg_critic_grad_norm": float(np.mean(self.critic_grad_norms)) if self.critic_grad_norms else 0.0,
        }


import re
import glob

from src.config import DEVICE, SEED
from src.models.rl_adapter import FoundationFeatureExtractor
from src.training.env import CryptoTradingEnv
from src.training.wrappers import RiskManagementWrapper


def find_latest_checkpoint(checkpoint_dir: str) -> str:
    """
    Find the latest checkpoint in the checkpoint directory.

    Looks for files matching pattern: tqc_foundation_XXXXXX_steps.zip
    Returns the one with the highest step number.

    Args:
        checkpoint_dir: Path to checkpoint directory

    Returns:
        Path to the latest checkpoint, or None if not found
    """
    pattern = os.path.join(checkpoint_dir, "tqc_foundation_*_steps.zip")
    checkpoints = glob.glob(pattern)

    if not checkpoints:
        return None

    # Extract step numbers and sort
    def extract_steps(path):
        match = re.search(r"_(\d+)_steps\.zip$", path)
        return int(match.group(1)) if match else 0

    checkpoints.sort(key=extract_steps, reverse=True)
    return checkpoints[0]


# ============================================================================
# Configuration
# ============================================================================

class TrainingConfig:
    """Training configuration for TQC with Foundation Model."""

    # Paths
    data_path: str = "data/processed_data.parquet"
    eval_data_path: str = "data/processed_data_eval.parquet"  # Separate eval dataset
    encoder_path: str = "weights/pretrained_encoder.pth"
    save_path: str = "weights/tqc_agent_final.zip"
    tensorboard_log: str = "logs/tensorboard_tqc/"
    checkpoint_dir: str = "weights/checkpoints/"

    # Run name (for TensorBoard)
    name: str = None  # If set, appears in TensorBoard

    # Environment (aggressive regularization for anti-overfitting)
    window_size: int = 64
    commission: float = 0.0015  # 0.15% - Higher cost during training (3.75x penalty)
    train_ratio: float = 0.8
    episode_length: int = 2048  # Épisodes plus courts pour tracking des rewards
    eval_episode_length: int = 720  # 1 month eval (30 days * 24h)
    reward_scaling: float = 30.0  # Amplify signal for tanh (optimal range)
    downside_coef: float = 10.0  # Sortino downside penalty coefficient
    upside_coef: float = 0.0  # Symmetric upside bonus coefficient
    action_discretization: float = 0.1  # Discretize actions (0.1 = 21 positions)
    churn_coef: float = 1.0  # Doubled: stronger anti-churn penalty
    smooth_coef: float = 1.0  # Quadratic smoothness penalty coefficient

    # Foundation Model (must match pretrained encoder)
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2

    # TQC Hyperparameters (aggressive regularization)
    total_timesteps: int = 150_000  # Reduced to prevent overfitting
    learning_rate: float = 6e-5  # With floor at 10% (6e-6 minimum)
    buffer_size: int = 200_000
    batch_size: int = 1024  # Larger batch for gradient smoothing
    gamma: float = 0.95  # Favor short-term rewards
    tau: float = 0.005  # Slow soft update to prevent catastrophic forgetting
    ent_coef: Union[str, float] = 0.05  # Fixed high entropy (force exploration)
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

    # Resume training (Lightweight mode - no buffer save/load)
    load_model_path: str = None  # Path to .zip model to resume from
    load_latest: bool = False  # Auto-select latest checkpoint from checkpoint_dir
    resume_learning_starts: int = 10_000  # Warmup steps to refill buffer before training

    # Curriculum Learning
    use_curriculum: bool = True  # Enable curriculum learning for fees/smoothness
    curriculum_warmup_steps: int = 50_000  # Steps to reach target values

    # Risk Management (Circuit Breaker)
    use_risk_management: bool = True
    risk_vol_window: int = 24  # Rolling window for volatility (hours)
    risk_vol_threshold: float = 3.0  # Trigger if vol > 3x baseline
    risk_max_drawdown: float = 0.10  # Trigger if DD > 10%
    risk_cooldown: int = 12  # Force HOLD for 12 steps after trigger
    risk_augment_obs: bool = False  # Don't change obs space by default


def linear_schedule(initial_value: float, floor_ratio: float = 0.1) -> Callable[[float], float]:
    """
    Linear learning rate schedule with floor.

    Decays linearly from initial_value to floor (10% of initial by default).
    Prevents learning collapse at end of training.

    Args:
        initial_value: Initial learning rate.
        floor_ratio: Minimum LR as ratio of initial (default: 0.1 = 10%).

    Returns:
        Callable that maps progress (1.0 -> 0.0) to learning rate.
    """
    floor = initial_value * floor_ratio
    def func(progress_remaining: float) -> float:
        return max(floor, progress_remaining * initial_value)
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
    # Curriculum Learning: start with 0 fees/smoothness if enabled
    if config.use_curriculum:
        initial_commission = 0.0
        initial_smooth = 0.0
    else:
        initial_commission = config.commission
        initial_smooth = config.smooth_coef

    # Create training environment (full training dataset)
    train_env = CryptoTradingEnv(
        parquet_path=config.data_path,
        window_size=config.window_size,
        commission=initial_commission,
        episode_length=config.episode_length,
        reward_scaling=config.reward_scaling,
        downside_coef=config.downside_coef,
        upside_coef=config.upside_coef,
        action_discretization=config.action_discretization,
        churn_coef=config.churn_coef,
        smooth_coef=initial_smooth,
        random_start=True,
    )

    # Create eval environment (separate eval dataset, 1 month)
    # Note: Eval env uses target values (no curriculum) for consistent evaluation
    val_env = CryptoTradingEnv(
        parquet_path=config.eval_data_path,
        window_size=config.window_size,
        commission=config.commission,
        episode_length=config.eval_episode_length,  # 720h = 1 month for faster eval
        reward_scaling=config.reward_scaling,
        downside_coef=config.downside_coef,
        upside_coef=config.upside_coef,
        action_discretization=config.action_discretization,
        churn_coef=config.churn_coef,
        smooth_coef=config.smooth_coef,
        random_start=False,  # Sequential for evaluation
    )

    # Wrap with Risk Management (Circuit Breaker)
    if config.use_risk_management:
        train_env = RiskManagementWrapper(
            train_env,
            vol_window=config.risk_vol_window,
            vol_threshold=config.risk_vol_threshold,
            max_drawdown=config.risk_max_drawdown,
            cooldown_steps=config.risk_cooldown,
            augment_obs=config.risk_augment_obs,
        )
        # Calibrate baseline volatility on train env
        train_env.calibrate_baseline(n_steps=2000)

        val_env = RiskManagementWrapper(
            val_env,
            vol_window=config.risk_vol_window,
            vol_threshold=config.risk_vol_threshold,
            max_drawdown=config.risk_max_drawdown,
            cooldown_steps=config.risk_cooldown,
            augment_obs=config.risk_augment_obs,
        )
        # Share calibration from train env
        val_env.baseline_vol = train_env.baseline_vol

    # Wrap in Monitor for episode tracking
    train_env_monitored = Monitor(train_env)
    val_env_monitored = Monitor(val_env)

    # Vectorize for SB3
    train_vec_env = DummyVecEnv([lambda: train_env_monitored])
    eval_vec_env = DummyVecEnv([lambda: val_env_monitored])

    return train_vec_env, eval_vec_env


def create_callbacks(config: TrainingConfig, eval_env) -> tuple[list, DetailTensorboardCallback]:
    """
    Create training callbacks.

    Args:
        config: Training configuration.
        eval_env: Evaluation environment.

    Returns:
        Tuple of (callbacks_list, detail_callback).
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

    # Detail Tensorboard callback for reward components
    detail_callback = DetailTensorboardCallback(verbose=0)
    callbacks.append(detail_callback)

    # Curriculum Learning callback
    if config.use_curriculum:
        from src.training.curriculum_callback import CurriculumFeesCallback
        curriculum_callback = CurriculumFeesCallback(
            target_fee_rate=config.commission,
            target_smooth_coef=config.smooth_coef,
            warmup_steps=config.curriculum_warmup_steps,
            verbose=1
        )
        callbacks.append(curriculum_callback)

    return callbacks, detail_callback


def train(config: TrainingConfig = None) -> tuple[TQC, dict]:
    """
    Train TQC agent with Foundation Model feature extractor.

    Args:
        config: Training configuration (uses default if None).

    Returns:
        Tuple of (Trained TQC model, training metrics dict).
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
    # Use custom name for TensorBoard if provided
    tb_log_name = config.name if config.name else "TQC"

    if config.load_model_path:
        # ========== LIGHTWEIGHT RESUME MODE ==========
        print(f"\n[3/4] Loading model from {config.load_model_path}...")
        print("      Mode: Lightweight Resume (no buffer)")

        model = TQC.load(
            config.load_model_path,
            env=train_env,
            device=DEVICE,
            tensorboard_log=config.tensorboard_log,
        )

        # CRITICAL: Cold Start Warmup - fill buffer before training
        model.learning_starts = config.resume_learning_starts
        print(f"      learning_starts set to {config.resume_learning_starts} (Cold Start Warmup)")

        # Reapply learning rate schedule
        model.learning_rate = linear_schedule(config.learning_rate)
        model.lr_schedule = linear_schedule(config.learning_rate)

        # Reapply entropy coefficient (can be overwritten by load)
        if config.ent_coef != "auto":
            model.ent_coef = config.ent_coef
            print(f"      ent_coef set to {config.ent_coef}")
        else:
            print("      ent_coef: auto (using loaded value)")

        print(f"      Flow: Load -> Empty Buffer -> Play {config.resume_learning_starts} steps -> Resume Training")

        # TensorBoard continuity warning
        if config.name is None:
            # List existing TensorBoard runs
            tb_runs = glob.glob(os.path.join(config.tensorboard_log, "*"))
            tb_names = [os.path.basename(r).rsplit("_", 1)[0] for r in tb_runs if os.path.isdir(r)]
            tb_names = list(set(tb_names))  # Deduplicate
            if tb_names:
                print(f"\n      [WARNING] No --name provided. TensorBoard will create a new curve.")
                print(f"      To continue an existing curve, use one of: {tb_names}")
    else:
        # ========== NEW TRAINING MODE ==========
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

    callbacks, detail_callback = create_callbacks(config, eval_env)

    # reset_num_timesteps=False continues TensorBoard from previous run
    is_resume = config.load_model_path is not None
    model.learn(
        total_timesteps=config.total_timesteps,
        callback=callbacks,
        tb_log_name=tb_log_name,
        progress_bar=True,
        reset_num_timesteps=not is_resume,  # False for resume, True for fresh
    )

    # ==================== Capture Training Metrics ====================
    training_metrics = detail_callback.get_training_metrics()

    print("\n[Training Diagnostics]")
    print(f"  Action Saturation: {training_metrics['action_saturation']:.3f}")
    print(f"  Avg Entropy: {training_metrics['avg_entropy']:.4f}")
    print(f"  Avg Critic Loss: {training_metrics['avg_critic_loss']:.4f}")
    print(f"  Avg Actor Loss: {training_metrics['avg_actor_loss']:.4f}")
    print(f"  Avg Churn Ratio: {training_metrics['avg_churn_ratio']:.3f}")
    print(f"  Avg Actor Grad Norm: {training_metrics['avg_actor_grad_norm']:.4f}")
    print(f"  Avg Critic Grad Norm: {training_metrics['avg_critic_grad_norm']:.4f}")

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

    return model, training_metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train TQC agent with Foundation Model")
    parser.add_argument("--timesteps", type=int, default=500_000, help="Total timesteps")
    parser.add_argument("--log-freq", type=int, default=100, help="Log frequency (steps)")
    parser.add_argument("--eval-freq", type=int, default=5_000, help="Eval frequency (steps)")
    parser.add_argument("--lr", type=float, default=6e-5, help="Learning rate (with 10% floor)")
    parser.add_argument("--checkpoint-dir", type=str, default="weights/checkpoints/", help="Checkpoint directory")
    parser.add_argument("--tau", type=float, default=None, help="Soft update coefficient (override config)")
    parser.add_argument("--downside-coef", type=float, default=None, help="Sortino downside penalty coefficient")
    parser.add_argument("--upside-coef", type=float, default=None, help="Symmetric upside bonus coefficient")
    parser.add_argument("--action-disc", type=float, default=None, help="Action discretization (0.1 = 21 positions, 0 = disabled)")
    parser.add_argument("--churn-coef", type=float, default=None, help="Churn penalty coefficient (0.1 = 10x amplified trading cost)")
    parser.add_argument("--smooth-coef", type=float, default=None, help="Smoothness penalty coefficient (quadratic penalty on position changes)")
    parser.add_argument("--ent-coef", type=float, default=None, help="Entropy coefficient (None = auto)")
    parser.add_argument("--name", type=str, default=None, help="Run name (appears in TensorBoard)")
    parser.add_argument("--gradient-steps", type=int, default=None, help="Gradient steps per update (default: 1)")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (default: 256)")
    parser.add_argument("--gamma", type=float, default=None, help="Discount factor (default: 0.99)")
    parser.add_argument("--load-model", type=str, default=None,
                        help="Path to .zip model to resume training from (Lightweight mode: no buffer)")
    parser.add_argument("--load-latest", action="store_true",
                        help="Auto-select latest checkpoint from checkpoint_dir")

    args = parser.parse_args()

    config = TrainingConfig()
    config.total_timesteps = args.timesteps
    config.log_freq = args.log_freq
    config.eval_freq = args.eval_freq
    config.learning_rate = args.lr
    config.checkpoint_dir = args.checkpoint_dir
    if args.tau is not None:
        config.tau = args.tau
    if args.downside_coef is not None:
        config.downside_coef = args.downside_coef
    if args.upside_coef is not None:
        config.upside_coef = args.upside_coef
    if args.action_disc is not None:
        config.action_discretization = args.action_disc
    if args.churn_coef is not None:
        config.churn_coef = args.churn_coef
    if args.smooth_coef is not None:
        config.smooth_coef = args.smooth_coef
    if args.ent_coef is not None:
        config.ent_coef = args.ent_coef
    if args.name is not None:
        config.name = args.name
    if args.gradient_steps is not None:
        config.gradient_steps = args.gradient_steps
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.gamma is not None:
        config.gamma = args.gamma
    if args.load_model is not None:
        config.load_model_path = args.load_model
    if args.load_latest:
        config.load_latest = True

    # Resolve --load-latest to actual checkpoint path
    if config.load_latest:
        latest = find_latest_checkpoint(config.checkpoint_dir)
        if latest:
            config.load_model_path = latest
            print(f"[Auto] Latest checkpoint found: {latest}")
        else:
            print(f"[Warning] No checkpoints found in {config.checkpoint_dir}")

    train(config)

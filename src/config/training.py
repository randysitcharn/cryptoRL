# -*- coding: utf-8 -*-
"""
training.py - Training configurations for TQC agent and hyperparameter tuning.

Centralizes all training-related settings for consistency across the codebase.
"""

from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Callable



# =============================================================================
# TQC Training Configuration
# =============================================================================

@dataclass
class TQCTrainingConfig:
    """
    Complete configuration for TQC agent training.

    Inherits base settings and adds TQC-specific parameters.
    """

    # --- Paths ---
    data_path: str = "data/processed_data.parquet"
    eval_data_path: str = "data/processed_data_eval.parquet"
    encoder_path: str = "weights/pretrained_encoder.pth"
    save_path: str = "weights/tqc_agent_final.zip"
    tensorboard_log: str = "logs/tensorboard_tqc/"
    checkpoint_dir: str = "weights/checkpoints/"
    name: Optional[str] = None  # Run name for TensorBoard

    # --- Environment ---
    window_size: int = 64
    commission: float = 0.0015  # 0.15% - Higher cost during training (penalty)
    train_ratio: float = 0.8
    episode_length: int = 2048
    eval_episode_length: int = 720  # 1 month eval (30 days * 24h)

    # Reward function (x100 SCALE applied in env.py)
    reward_scaling: float = 1.0   # Keep at 1.0 (SCALE=100 in env)
    downside_coef: float = 10.0
    upside_coef: float = 0.0
    action_discretization: float = 0.1
    churn_coef: float = 0.5       # Max target après curriculum (réduit)
    smooth_coef: float = 1e-5     # Très bas par défaut (curriculum monte à 0.001 max)

    # Volatility scaling
    target_volatility: float = 0.05  # 5% target vol
    vol_window: int = 24
    max_leverage: float = 2.0

    # --- Foundation Model ---
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    freeze_encoder: bool = True

    # --- TQC Hyperparameters ---
    total_timesteps: int = 300_000
    learning_rate: float = 3e-4
    buffer_size: Optional[int] = None  # Auto-detect from RAM (HardwareManager)
    batch_size: Optional[int] = None   # Auto-detect from VRAM (HardwareManager)
    gamma: float = 0.95  # Horizon ~20h (réduit pour stabilité du Critic)
    tau: float = 0.005
    ent_coef: Union[str, float] = "auto"
    train_freq: int = 1
    gradient_steps: int = 1
    top_quantiles_to_drop: int = 2
    n_critics: int = 2
    n_quantiles: int = 25

    # Policy network
    net_arch: Optional[Dict[str, List[int]]] = None  # Default: dict(pi=[64,64], qf=[64,64])

    # gSDE (State-Dependent Exploration)
    use_sde: bool = True
    sde_sample_freq: int = -1  # -1 = resample once per episode
    use_sde_at_warmup: bool = True

    # --- Callbacks ---
    eval_freq: int = 5_000
    checkpoint_freq: int = 50_000
    log_freq: int = 100

    # --- Resume Training ---
    load_model_path: Optional[str] = None
    load_latest: bool = False
    resume_learning_starts: int = 10_000

    # --- Curriculum Learning ---
    use_curriculum: bool = True
    curriculum_warmup_steps: int = 50_000

    # --- Risk Management ---
    use_risk_management: bool = True
    risk_vol_window: int = 24
    risk_vol_threshold: float = 3.0
    risk_max_drawdown: float = 0.10
    risk_cooldown: int = 12
    risk_augment_obs: bool = False

    # --- Parallelization (P0 Optimization) ---
    n_envs: Optional[int] = None  # Auto-detect from CPU cores (HardwareManager)
                                  # Set to 1 to disable (use DummyVecEnv)


# =============================================================================
# Hyperparameter Tuning Configuration
# =============================================================================

@dataclass
class TuningConfig:
    """Configuration for Optuna hyperparameter tuning."""

    # --- Paths ---
    data_path: str = "data/processed_data.parquet"
    encoder_path: str = "weights/pretrained_encoder.pth"
    output_dir: str = "results/"
    study_db: str = "sqlite:///optuna_study.db"

    # --- Environment ---
    window_size: int = 64
    commission: float = 0.0006
    train_ratio: float = 0.8
    episode_length: int = 2048

    # --- Foundation Model ---
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    freeze_encoder: bool = True

    # --- Fixed TQC Parameters ---
    buffer_size: int = 100_000
    top_quantiles_to_drop: int = 2
    n_critics: int = 2
    n_quantiles: int = 25
    net_arch: List[int] = field(default_factory=lambda: [256, 256])
    use_sde: bool = True
    sde_sample_freq: int = -1
    use_sde_at_warmup: bool = True

    # --- Tuning Settings ---
    trial_timesteps: int = 60_000
    n_trials: int = 50
    eval_freq: int = 5_000
    n_eval_episodes: int = 1


# =============================================================================
# Foundation Model Training Configuration
# =============================================================================

@dataclass
class FoundationTrainingConfig:
    """Configuration for foundation model (MAE) training."""

    # --- Data ---
    seq_len: int = 64
    batch_size: int = 64
    train_ratio: float = 0.8

    # --- Model Architecture ---
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    mask_ratio: float = 0.15

    # --- Training ---
    epochs: int = 70
    lr: float = 1e-4
    weight_decay: float = 1e-5
    patience: int = 7  # Early stopping

    # --- Paths ---
    data_path: str = "data/processed_data.parquet"
    weights_dir: str = "weights"
    checkpoint_path: str = "weights/best_foundation_full.pth"
    encoder_path: str = "weights/pretrained_encoder.pth"
    tensorboard_log: Optional[str] = "logs/tensorboard_mae/"
    run_name: Optional[str] = None


# =============================================================================
# Utility Functions
# =============================================================================

def linear_schedule(initial_value: float, floor_ratio: float = 0.1) -> Callable[[float], float]:
    """
    Linear learning rate schedule with floor.

    Decays linearly from initial_value to floor (10% of initial by default).

    Args:
        initial_value: Initial learning rate.
        floor_ratio: Minimum LR as ratio of initial (default: 0.1 = 10%).

    Returns:
        Callable that maps progress (1.0 -> 0.0) to learning rate.
    """
    floor_value = initial_value * floor_ratio

    def schedule(progress_remaining: float) -> float:
        # progress_remaining goes from 1.0 to 0.0 during training
        return floor_value + (initial_value - floor_value) * progress_remaining

    return schedule

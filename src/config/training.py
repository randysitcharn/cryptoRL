# -*- coding: utf-8 -*-
"""
training.py - Training configurations for TQC agent.

Centralizes all training-related settings for consistency across the codebase.
"""

from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Callable



# =============================================================================
# TQC Training Configuration
# =============================================================================

@dataclass
class TQCTrainingConfig:
    """Complete configuration for TQC agent training."""

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
    eval_episode_length: int = 600  # Safe value < (720 - window_size - 1)

    # Reward function (x100 SCALE applied in env.py)
    reward_scaling: float = 1.0   # Keep at 1.0 (SCALE=100 in env)
    downside_coef: float = 1.0    # Asymmetric penalty for losses (1.0 = symmetric)
    upside_coef: float = 0.0
    action_discretization: float = 0.1

    # Volatility scaling
    target_volatility: float = 0.05  # 5% target vol
    vol_window: int = 24
    max_leverage: float = 2.0

    # Regularization (anti-overfitting)
    observation_noise: float = 0.01  # 1% Gaussian noise on market observations

    # --- Dropout Regularization (DroQ/STAC style) ---
    # See docs/design/DROPOUT_TQC_DESIGN.md for details
    use_dropout_policy: bool = True   # Use TQCDropoutPolicy instead of standard
    critic_dropout: float = 0.01      # DroQ recommends 0.01-0.1 (conservative)
    actor_dropout: float = 0.0        # Phase 1: critics only (0.005 for Phase 2)
    use_layer_norm: bool = True       # CRITICAL for stability with dropout
    
    # --- Spectral Normalization (Stability) ---
    # See audit: Spectral norm crucial for Critic (Lipschitz constraint),
    # more debatable for Actor (may constrain policy too much)
    use_spectral_norm_critic: bool = False  # Default False for reproducibility
    use_spectral_norm_actor: bool = False    # Default False (conservative)

    # --- Foundation Model ---
    d_model: int = 256  # Increased capacity for complex patterns
    n_heads: int = 4
    n_layers: int = 2
    freeze_encoder: bool = True

    # --- TQC Hyperparameters ---
    total_timesteps: int = 90_000_000  # 90M steps
    learning_rate: float = 3e-4      # Standard TQC
    buffer_size: Optional[int] = None  # Auto-detect from RAM (HardwareManager)
    batch_size: Optional[int] = None   # Auto-detect from VRAM (HardwareManager)
    gamma: float = 0.95  # Shorter horizon for faster learning
    tau: float = 0.005
    ent_coef: Union[str, float] = "auto"
    train_freq: int = 1
    gradient_steps: int = 1  # GS=1 with 1024 envs for max diversity
    top_quantiles_to_drop: int = 2
    n_critics: int = 2
    n_quantiles: int = 25

    # Policy network
    net_arch: Optional[Dict[str, List[int]]] = None  # Default: dict(pi=[64,64], qf=[64,64])

    # gSDE (State-Dependent Exploration)
    use_sde: bool = True
    sde_sample_freq: int = -1  # -1 = resample once per episode
    use_sde_at_warmup: bool = True

    # Actor Noise (fallback when use_sde=False, ignored otherwise)
    # OrnsteinUhlenbeck noise for temporally correlated exploration
    use_action_noise: bool = True    # Enable OU noise (only active when use_sde=False)
    action_noise_sigma: float = 0.1  # Noise std dev (0.05-0.3)
    action_noise_theta: float = 0.15 # Mean reversion rate (higher = less correlated)

    # --- Callbacks ---
    eval_freq: int = 5_000
    checkpoint_freq: int = 50_000
    log_freq: int = 100
    verbose: int = 1  # SB3 verbosity level

    # --- Resume Training ---
    load_model_path: Optional[str] = None
    load_latest: bool = False
    resume_learning_starts: int = 10_000

    # --- Curriculum Learning ---
    use_curriculum: bool = True
    curriculum_warmup_steps: int = 50_000

    # --- Overfitting Guard V2 (WFO mode) ---
    # See docs/WFO_OVERFITTING_GUARD.md for details
    use_overfitting_guard: bool = False     # Disabled by default (WFO enables it)
    guard_nav_threshold: float = 5.0        # 5x = +400% (standard mode)
    guard_patience: int = 3                 # Violations before stop
    guard_check_freq: int = 10_000          # Check frequency (steps)
    guard_action_saturation: float = 0.95   # Saturation threshold
    guard_reward_variance: float = 1e-4     # Min variance threshold

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

    # --- Ensemble RL (Design Doc 2026-01-22) ---
    # Reference: docs/design/ENSEMBLE_RL_DESIGN.md
    use_ensemble: bool = False
    ensemble_n_members: int = 3
    ensemble_seeds: List[int] = field(default_factory=lambda: [42, 123, 456])
    ensemble_aggregation: str = 'confidence'  # 'mean', 'median', 'confidence', 'conservative', 'pessimistic_bound'
    ensemble_parallel: bool = True  # Train on multiple GPUs
    ensemble_parallel_gpus: List[int] = field(default_factory=lambda: [0, 1])

    # Seed for single-model training (used by ensemble trainer per member)
    seed: int = 42

    def __post_init__(self):
        """Validate configuration consistency."""
        # Ensemble seeds validation
        if self.use_ensemble and len(self.ensemble_seeds) < self.ensemble_n_members:
            raise ValueError(
                f"ensemble_seeds ({len(self.ensemble_seeds)}) must have at least "
                f"ensemble_n_members ({self.ensemble_n_members}) values"
            )


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
    d_model: int = 256  # Updated to match TQC d_model
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

# =============================================================================
# WFO-Specific Training Configuration
# =============================================================================

@dataclass
class WFOTrainingConfig(TQCTrainingConfig):
    """
    WFO-specific training configuration with optimized hyperparameters.
    
    Inherits from TQCTrainingConfig and overrides values for Walk-Forward Optimization:
    - Slower learning (LR 1e-4) for better generalization OOS
    - Aggressive regularization (dropout 0.1) to prevent overfitting on short windows
    - Boosted exploration (ent_coef auto_0.5) for diverse policy discovery
    - OverfittingGuard enabled with permissive thresholds for long WFO runs
    
    Rationale documented in: docs/design/WFO_CONFIG_RATIONALE.md
    """
    
    # --- TQC Hyperparameters (WFO-optimized) ---
    learning_rate: float = 1e-4           # Slow & stable (vs 3e-4 default)
    buffer_size: int = 2_500_000          # 2.5M replay buffer
    n_envs: int = 1024                    # GPU-optimized (power of 2)
    batch_size: int = 512                 # Smaller batch for more updates
    ent_coef: Union[str, float] = "auto_0.5"  # Boosted exploration
    
    # --- Regularization (aggressive for OOS generalization) ---
    critic_dropout: float = 0.1           # 10% dropout (DroQ max)
    
    # --- Overfitting Guard (enabled for WFO) ---
    use_overfitting_guard: bool = True
    guard_nav_threshold: float = 10.0     # 10x (permissive for long WFO)
    guard_patience: int = 5               # Increased patience
    guard_check_freq: int = 25_000        # ~6 weeks of data
    guard_reward_variance: float = 1e-5   # Lower threshold
    
    # --- WFO-specific timesteps ---
    total_timesteps: int = 30_000_000     # 30M (vs 90M default)
    
    # --- Evaluation commission (lower for realistic backtesting) ---
    eval_commission: float = 0.0004       # 0.04% for test evaluation


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

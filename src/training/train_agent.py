# -*- coding: utf-8 -*-
"""
train_agent.py - TQC Agent Training with Foundation Model.

Trains a TQC agent using the pre-trained CryptoMAE encoder as feature extractor.
The encoder is frozen by default to preserve learned market representations.

Usage:
    python -m src.training.train_agent
"""

import os
import re
import glob
import zipfile
import io
from typing import Optional, Tuple

from sb3_contrib import TQC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from src.training.callbacks import EvalCallbackWithNoiseControl
import torch
import numpy as np

from src.config import DEVICE, SEED
from src.models.tqc_dropout_policy import TQCDropoutPolicy
from src.utils.hardware import HardwareManager
from src.models.rl_adapter import FoundationFeatureExtractor
from src.training.batch_env import BatchCryptoEnv
from src.training.clipped_optimizer import ClippedAdamW
from src.training.callbacks import (
    UnifiedMetricsCallback,
    MORLCurriculumCallback,
    OverfittingGuardCallback,
    OverfittingGuardCallbackV2,
    ModelEMACallback,
)


def clean_compiled_checkpoint(checkpoint_path: str) -> None:
    """
    Clean torch.compile artifacts from SB3 checkpoint.

    torch.compile wraps modules and adds '_orig_mod.' prefix to state_dict keys.
    This function removes that prefix to ensure checkpoint compatibility with
    non-compiled models (e.g., during resume).

    Args:
        checkpoint_path: Path to the .zip checkpoint file
    """
    if not os.path.exists(checkpoint_path):
        return

    try:
        # Read the zip file
        with zipfile.ZipFile(checkpoint_path, 'r') as zf:
            file_contents = {}
            for name in zf.namelist():
                file_contents[name] = zf.read(name)

        # Process .pth files (state dicts)
        modified = False
        for name in list(file_contents.keys()):
            if name.endswith('.pth'):
                # Load state dict from bytes
                buffer = io.BytesIO(file_contents[name])
                state_dict = torch.load(buffer, map_location='cpu', weights_only=False)

                # Check if cleaning is needed
                needs_cleaning = any('_orig_mod.' in k for k in state_dict.keys())
                if needs_cleaning:
                    # Clean keys
                    cleaned_dict = {}
                    for key, value in state_dict.items():
                        new_key = key.replace('_orig_mod.', '')
                        cleaned_dict[new_key] = value

                    # Save cleaned state dict back to bytes
                    buffer = io.BytesIO()
                    torch.save(cleaned_dict, buffer)
                    file_contents[name] = buffer.getvalue()
                    modified = True

        # Write back if modified
        if modified:
            with zipfile.ZipFile(checkpoint_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                for name, content in file_contents.items():
                    zf.writestr(name, content)
            print(f"  [CLEANUP] Removed torch.compile artifacts from {os.path.basename(checkpoint_path)}")

    except Exception as e:
        print(f"  [WARNING] Could not clean checkpoint {checkpoint_path}: {e}")


class RotatingCheckpointCallback(CheckpointCallback):
    """
    Checkpoint callback that keeps only the last saved checkpoint (disk optimization).

    Inherits from CheckpointCallback but deletes the previous checkpoint after
    saving a new one. This prevents disk space from filling up during long
    training runs.

    Args:
        save_freq: Save frequency in timesteps.
        save_path: Directory to save checkpoints.
        name_prefix: Prefix for checkpoint files.
        verbose: Verbosity level.
    """

    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "rl_model", verbose: int = 0):
        super().__init__(
            save_freq=save_freq,
            save_path=save_path,
            name_prefix=name_prefix,
            save_replay_buffer=False,
            verbose=verbose,
        )
        self.last_model_path: Optional[str] = None

    def _on_step(self) -> bool:
        # Call parent to save the model
        result = super()._on_step()

        # Check if a save happened on this step
        if self.n_calls % self.save_freq == 0:
            step = self.num_timesteps
            current_path = os.path.join(self.save_path, f"{self.name_prefix}_{step}_steps.zip")

            # Clean torch.compile artifacts from the checkpoint
            clean_compiled_checkpoint(current_path)

            # Delete old checkpoint if it exists and is different from current
            if self.last_model_path and os.path.exists(self.last_model_path) and self.last_model_path != current_path:
                try:
                    os.remove(self.last_model_path)
                    if self.verbose > 0:
                        print(f"  [Disk Opt] Deleted old checkpoint: {os.path.basename(self.last_model_path)}")
                except OSError as e:
                    print(f"  [Warning] Could not delete {self.last_model_path}: {e}")

            # Update tracker for next iteration
            self.last_model_path = current_path

        return result


class BestModelCleanerCallback(BaseCallback):
    """
    Callback to clean torch.compile artifacts from best_model.zip.

    EvalCallback saves best_model.zip internally, and we can't hook into that.
    This callback periodically checks and cleans best_model.zip.
    """

    def __init__(self, checkpoint_dir: str, check_freq: int = 10000, verbose: int = 0):
        super().__init__(verbose)
        self.best_model_path = os.path.join(checkpoint_dir, "best_model.zip")
        self.check_freq = check_freq
        self.last_mtime: Optional[float] = None

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq != 0:
            return True

        # Check if best_model.zip exists and was modified
        if os.path.exists(self.best_model_path):
            current_mtime = os.path.getmtime(self.best_model_path)
            if self.last_mtime is None or current_mtime > self.last_mtime:
                clean_compiled_checkpoint(self.best_model_path)
                self.last_mtime = current_mtime

        return True


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
# Configuration (imported from centralized config module)
# ============================================================================

from src.config import TQCTrainingConfig as TrainingConfig, linear_schedule


def _validate_spectral_norm_compatibility(model: TQC, config: TrainingConfig) -> None:
    """
    Validate spectral normalization compatibility between checkpoint and config.
    
    Checks if the loaded model's SN configuration matches the current config.
    Raises ValueError if mismatch is detected.
    
    Args:
        model: Loaded TQC model
        config: Current training configuration
        
    Raises:
        ValueError: If SN configuration mismatch detected
    """
    # Only validate if using TQCDropoutPolicy
    if not config.use_dropout_policy:
        return
    
    # Check if policy has SN attributes
    if not hasattr(model.policy, 'use_spectral_norm_critic'):
        return  # Not using TQCDropoutPolicy or old version
    
    # Extract SN config from loaded model
    loaded_sn_critic = getattr(model.policy, 'use_spectral_norm_critic', False)
    loaded_sn_actor = getattr(model.policy, 'use_spectral_norm_actor', False)
    
    # Compare with current config
    config_sn_critic = config.use_spectral_norm_critic
    config_sn_actor = config.use_spectral_norm_actor
    
    if loaded_sn_critic != config_sn_critic or loaded_sn_actor != config_sn_actor:
        raise ValueError(
            f"Checkpoint SN mismatch detected!\n"
            f"  Checkpoint: critic={loaded_sn_critic}, actor={loaded_sn_actor}\n"
            f"  Config:     critic={config_sn_critic}, actor={config_sn_actor}\n"
            f"  Checkpoints are NOT compatible if `use_spectral_norm_*` flags change.\n"
            f"  Solution: Re-train with matching SN configuration or use a compatible checkpoint."
        )


def create_policy_kwargs(config: TrainingConfig) -> dict:
    """
    Create policy kwargs with FoundationFeatureExtractor.

    Args:
        config: Training configuration.

    Returns:
        Dict of policy keyword arguments.
    """
    # Default to tiny architecture if not specified
    net_arch = config.net_arch if config.net_arch else dict(pi=[64, 64], qf=[64, 64])

    policy_kwargs = dict(
        features_extractor_class=FoundationFeatureExtractor,
        features_extractor_kwargs=dict(
            encoder_path=config.encoder_path,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            freeze_encoder=config.freeze_encoder,
        ),
        net_arch=net_arch,
        n_critics=config.n_critics,
        n_quantiles=config.n_quantiles,
        optimizer_class=ClippedAdamW,
        optimizer_kwargs=dict(
            max_grad_norm=0.5,  # Gradient clipping intégré
            weight_decay=1e-5,  # Reduced for tiny arch
            eps=1e-5,
        ),
    )

    # Add dropout parameters if using TQCDropoutPolicy
    if config.use_dropout_policy:
        policy_kwargs.update(
            critic_dropout=config.critic_dropout,
            actor_dropout=config.actor_dropout,
            use_layer_norm=config.use_layer_norm,
            use_spectral_norm_critic=config.use_spectral_norm_critic,
            use_spectral_norm_actor=config.use_spectral_norm_actor,
        )

    # FIX: Increased init exploration (log_std_init=-1 gives std≈0.37 vs default -3 giving std≈0.05)
    policy_kwargs["log_std_init"] = config.log_std_init

    return policy_kwargs


def create_environments(config: TrainingConfig, n_envs: int = 1, use_batch_env: bool = True):
    """
    Create training and evaluation environments using BatchCryptoEnv.

    Args:
        config: Training configuration.
        n_envs: Number of parallel training environments.
        use_batch_env: Deprecated, always uses BatchCryptoEnv (kept for backward compatibility).

    Returns:
        Tuple of (train_vec_env, eval_vec_env, None, None, None).
        Last three values are None (kept for backward compatibility).
    """
    # Curriculum Learning: start with 0 fees if enabled
    if config.use_curriculum:
        initial_commission = 0.0
    else:
        initial_commission = config.commission

    # ==================== Training Environment ====================
    # GPU-Vectorized Batch Environment
    # All envs run in a single process using PyTorch tensors on GPU
    print(f"      [BatchEnv] Creating {n_envs} GPU-vectorized environments (BatchCryptoEnv)...")

    train_vec_env = BatchCryptoEnv(
        parquet_path=config.data_path,
        n_envs=n_envs,
        device=str(DEVICE),
        window_size=config.window_size,
        episode_length=config.episode_length,
        initial_balance=10_000.0,
        commission=initial_commission,
        slippage=0.0001,
        reward_scaling=config.reward_scaling,
        downside_coef=config.downside_coef,
        upside_coef=config.upside_coef,
        action_discretization=config.action_discretization,
        target_volatility=config.target_volatility,
        vol_window=config.vol_window,
        max_leverage=config.max_leverage,
        price_column='BTC_Close',
        observation_noise=config.observation_noise,  # Anti-overfitting
    )

    # ==================== Eval Environment (optional) ====================
    # Note: eval_data_path=None in WFO mode to prevent data leakage
    if config.eval_data_path is not None:
        # Note: Eval env uses target values (no curriculum) for consistent evaluation
        # Use BatchCryptoEnv with small n_envs for evaluation
        eval_vec_env = BatchCryptoEnv(
            parquet_path=config.eval_data_path,
            n_envs=min(n_envs, 32),  # Smaller for evaluation
            device=str(DEVICE),
            window_size=config.window_size,
            episode_length=config.eval_episode_length,  # 720h = 1 month for faster eval
            initial_balance=10_000.0,
            commission=config.commission,
            slippage=0.0001,
            reward_scaling=config.reward_scaling,
            downside_coef=config.downside_coef,
            upside_coef=config.upside_coef,
            action_discretization=config.action_discretization,
            target_volatility=config.target_volatility,
            vol_window=config.vol_window,
            max_leverage=config.max_leverage,
            price_column='BTC_Close',
            random_start=False,  # Sequential for evaluation
            observation_noise=0.0,  # No noise for evaluation
        )
    else:
        # WFO mode: no eval env to prevent data leakage
        eval_vec_env = None
        print("      [WFO Mode] Eval environment disabled (eval_data_path=None)")

    return train_vec_env, eval_vec_env, None, None


def create_callbacks(
    config: TrainingConfig,
    eval_env,
    shared_fee=None,
) -> tuple[list, UnifiedMetricsCallback]:
    """
    Create training callbacks.

    Args:
        config: Training configuration.
        eval_env: Evaluation environment (None in WFO mode).
        shared_fee: Shared memory Value for curriculum fee (SubprocVecEnv).

    Returns:
        Tuple of (callbacks_list, unified_callback).
    """
    callbacks = []

    # Unified metrics callback (replaces StepLoggingCallback and DetailTensorboardCallback)
    unified_callback = UnifiedMetricsCallback(log_freq=config.log_freq, verbose=config.verbose)
    callbacks.append(unified_callback)

    # Note: Gradient clipping is now handled by ClippedAdamW optimizer

    # Evaluation callback (only if eval_env is provided)
    if eval_env is not None:
        # Disable observation noise in eval environment if it's BatchCryptoEnv
        from src.training.callbacks import get_underlying_batch_env
        eval_batch_env = get_underlying_batch_env(eval_env)
        if eval_batch_env is not None and hasattr(eval_batch_env, 'set_training_mode'):
            eval_batch_env.set_training_mode(False)
            if config.verbose > 0:
                print("      [Noise Control] Disabled observation noise in eval environment")

        # Use EvalCallbackWithNoiseControl to manage noise in training env during eval
        eval_callback = EvalCallbackWithNoiseControl(
            eval_env,
            best_model_save_path=config.checkpoint_dir,
            log_path="logs/",
            eval_freq=config.eval_freq,
            deterministic=True,
            render=False,
            verbose=1,
        )
        callbacks.append(eval_callback)

        # Clean torch.compile artifacts from best_model.zip saved by EvalCallback
        best_model_cleaner = BestModelCleanerCallback(
            checkpoint_dir=config.checkpoint_dir,
            check_freq=config.eval_freq,  # Check after each evaluation
            verbose=0,
        )
        callbacks.append(best_model_cleaner)

        # Rotating checkpoint callback (keeps only last checkpoint to save disk space)
        checkpoint_callback = RotatingCheckpointCallback(
            save_freq=config.checkpoint_freq,
            save_path=config.checkpoint_dir,
            name_prefix="tqc_foundation",
            verbose=1,
        )
        callbacks.append(checkpoint_callback)
    else:
        # WFO mode: no EvalCallback, use safety checkpoint with rotation
        print("      [WFO Mode] EvalCallback disabled.")
        print("      [WFO Mode] Portfolio metrics logged via UnifiedMetricsCallback (portfolio/nav, portfolio/position_pct, risk/max_drawdown)")

        # Safety checkpoint with disk rotation (keeps only last checkpoint)
        n_envs = getattr(config, 'n_envs', 1) or 1
        safety_freq = max(1000, 100_000 // n_envs)
        safety_checkpoint = RotatingCheckpointCallback(
            save_freq=safety_freq,
            save_path=config.checkpoint_dir,
            name_prefix="tqc_wfo_safety",
            verbose=1,
        )
        callbacks.append(safety_checkpoint)
        print(f"      [WFO Mode] RotatingCheckpointCallback enabled (freq={safety_freq}, keeps last only)")

    # Note: UnifiedMetricsCallback already added above (replaces DetailTensorboardCallback)

    # Curriculum Learning callback (MORL: Progressive w_cost modulation)
    # Gradually increases w_cost from 0.0 (pure performance) to 0.1 (balanced)
    # over the first 50% of training, then plateaus at 0.1
    if config.use_curriculum:
        curriculum_callback = MORLCurriculumCallback(
            start_cost=0.0,
            end_cost=0.1,
            progress_ratio=0.5,
            total_timesteps=config.total_timesteps,
            verbose=1
        )
        callbacks.append(curriculum_callback)

    # ═══════════════════════════════════════════════════════════════════════
    # Model EMA Callback (Polyak Averaging for Policy Weights)
    # Maintains EMA of policy weights for robust evaluation
    # ═══════════════════════════════════════════════════════════════════════
    ema_callback = ModelEMACallback(
        decay=0.995,  # Corresponds to τ=0.005 (match TQC's tau for target network)
        save_path=config.checkpoint_dir,  # Optional: save EMA model at end
        verbose=config.verbose
    )
    callbacks.append(ema_callback)

    # Overfitting guard - DISABLED (OOS Sharpe 4.75 shows model generalizes well)
    # overfitting_guard = OverfittingGuardCallback(
    #     nav_threshold=5.0,
    #     initial_nav=10_000.0,
    #     check_freq=25_600,  # Check every ~25k steps
    #     verbose=1
    # )
    # callbacks.append(overfitting_guard)

    # ═══════════════════════════════════════════════════════════════════════
    # OverfittingGuard V2 (WFO mode: Signal 3 auto-disabled if no EvalCallback)
    # See docs/WFO_OVERFITTING_GUARD.md for details
    # ═══════════════════════════════════════════════════════════════════════
    if getattr(config, 'use_overfitting_guard', False):
        # En WFO: eval_env=None => pas d'EvalCallback => Signal 3 désactivé
        eval_cb = None
        if eval_env is not None:
            # Mode standard: chercher EvalCallback existant dans la liste
            eval_cb = next(
                (cb for cb in callbacks if isinstance(cb, EvalCallback)),
                None
            )

        guard = OverfittingGuardCallbackV2(
            nav_threshold=getattr(config, 'guard_nav_threshold', 5.0),
            patience=getattr(config, 'guard_patience', 3),
            check_freq=getattr(config, 'guard_check_freq', 10_000),
            action_saturation_threshold=getattr(config, 'guard_action_saturation', 0.95),
            reward_variance_threshold=getattr(config, 'guard_reward_variance', 1e-4),
            eval_callback=eval_cb,
            verbose=1
        )
        callbacks.append(guard)
        signal3_status = 'ON' if eval_cb else 'OFF (WFO mode)'
        print(f"  [Guard] OverfittingGuardCallbackV2 enabled (Signal 3: {signal3_status})")

    return callbacks, unified_callback


def train(
    config: TrainingConfig = None,
    hw_overrides: dict = None,
    use_batch_env: bool = False,
    resume_path: str = None
) -> tuple[TQC, dict]:
    """
    Train TQC agent with Foundation Model feature extractor.

    Args:
        config: Training configuration (uses default if None).
        hw_overrides: Optional dict to override hardware auto-config
                      (e.g., {"n_envs": 4, "batch_size": 512})
        use_batch_env: If True, use BatchCryptoEnv (GPU-vectorized) instead of SubprocVecEnv.
                       This runs all envs in a single process using GPU tensors (10-50x speedup).
        resume_path: Path to checkpoint to resume from (e.g., tqc_last.zip).
                     If provided, loads model and continues TensorBoard steps.

    Returns:
        Tuple of (Trained TQC model, training metrics dict).
    """
    if config is None:
        config = TrainingConfig()

    print("=" * 70)
    print("TQC + Foundation Model Training")
    print("=" * 70)

    # ==================== Hardware Auto-Detection ====================
    print("\n[0/4] Detecting hardware and computing optimal config...")
    import logging
    hw_logger = logging.getLogger("hardware")
    hw_logger.setLevel(logging.INFO)
    if not hw_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("      %(message)s"))
        hw_logger.addHandler(handler)

    hw_manager = HardwareManager(logger=hw_logger)

    # Build overrides from config + explicit hw_overrides
    # Priority: hw_overrides > config attributes > auto-detected
    overrides = {}
    if hasattr(config, 'n_envs') and config.n_envs is not None:
        overrides['n_envs'] = config.n_envs
    if hasattr(config, 'batch_size') and config.batch_size is not None:
        overrides['batch_size'] = config.batch_size
    if hasattr(config, 'buffer_size') and config.buffer_size is not None:
        overrides['buffer_size'] = config.buffer_size
    if hw_overrides:
        overrides.update(hw_overrides)

    adaptive_config = hw_manager.get_adaptive_config(user_overrides=overrides)
    hw_manager.apply_optimizations(adaptive_config)
    hw_manager.log_summary(adaptive_config)

    # Apply adaptive config back to training config
    n_envs = adaptive_config.n_envs
    config.batch_size = adaptive_config.batch_size
    config.buffer_size = adaptive_config.buffer_size

    # Create directories
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.tensorboard_log, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # ==================== Environment Setup ====================
    print("\n[1/4] Creating environments...")
    train_env, eval_env, shared_fee, manager = create_environments(
        config, n_envs=n_envs, use_batch_env=use_batch_env
    )

    obs_space = train_env.observation_space
    # Handle Dict observation space (market + position)
    if hasattr(obs_space, 'spaces'):
        market_shape = obs_space["market"].shape
        pos_shape = obs_space["position"].shape
        print(f"      Observation space: Dict(market={market_shape}, position={pos_shape})")
    else:
        market_shape = obs_space.shape
        print(f"      Observation space: {market_shape}")
    print(f"      Action space: {train_env.action_space.shape}")
    print(f"      Device: {DEVICE}")
    print(f"      Volatility Scaling Active: Target={config.target_volatility}, Window={config.vol_window}")

    # ==================== Policy Setup ====================
    print("\n[2/4] Configuring policy with FoundationFeatureExtractor...")
    policy_kwargs = create_policy_kwargs(config)

    print(f"      Encoder: {config.encoder_path}")
    print(f"      Frozen: {config.freeze_encoder}")
    print(f"      Net arch: {config.net_arch}")
    print(f"      features_dim: {market_shape[0]} * {config.d_model} + 1(pos) = {market_shape[0] * config.d_model + 1}")
    print(f"      gSDE: {config.use_sde} (sample_freq={config.sde_sample_freq})")

    # Create action noise if gSDE is disabled
    action_noise = None
    if not config.use_sde and config.use_action_noise:
        n_actions = train_env.action_space.shape[-1]
        action_noise = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(n_actions),
            sigma=config.action_noise_sigma * np.ones(n_actions),
            theta=config.action_noise_theta,
        )
        print(f"      Action Noise: OrnsteinUhlenbeck (sigma={config.action_noise_sigma}, theta={config.action_noise_theta})")
    elif not config.use_sde:
        print("      Action Noise: None (WARNING: deterministic policy)")
    else:
        print("      Action Noise: N/A (gSDE active)")

    # ==================== Model Creation ====================
    # Use custom name for TensorBoard if provided
    tb_log_name = config.name if config.name else "TQC"

    # Determine resume path: explicit resume_path takes precedence
    effective_resume_path = resume_path if resume_path and os.path.exists(resume_path) else config.load_model_path
    is_resume = effective_resume_path is not None

    if effective_resume_path:
        # ========== RESUME MODE ==========
        print(f"\n[3/4] 🔄 RESUMING training from {effective_resume_path}...")
        print("      Mode: Resume (continuing TensorBoard steps)")

        model = TQC.load(
            effective_resume_path,
            env=train_env,
            device=DEVICE,
            tensorboard_log=config.tensorboard_log,
        )
        
        # Validate spectral normalization compatibility (if model loaded successfully)
        # Note: If SN config mismatch, TQC.load() may fail with state_dict error before this point
        try:
            _validate_spectral_norm_compatibility(model, config)
        except ValueError as e:
            print(f"\n[ERROR] {e}")
            raise

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

        # Select policy class based on config
        if config.use_dropout_policy:
            policy_class = TQCDropoutPolicy
            print(f"      Policy: TQCDropoutPolicy (critic_dropout={config.critic_dropout}, "
                  f"actor_dropout={config.actor_dropout}, use_layer_norm={config.use_layer_norm})")
        else:
            policy_class = "MultiInputPolicy"
            print("      Policy: MultiInputPolicy (standard)")

        model = TQC(
            policy=policy_class,
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
            action_noise=action_noise,
            policy_kwargs=policy_kwargs,
            tensorboard_log=config.tensorboard_log,
            verbose=1,
            seed=SEED,
            device=DEVICE,
        )

    print(f"      Total parameters: {sum(p.numel() for p in model.policy.parameters()):,}")
    print(f"      Trainable parameters: {sum(p.numel() for p in model.policy.parameters() if p.requires_grad):,}")

    # ⚡ OPTIMIZATION: Torch Compile (Surgical + Warmup)
    # Skip torch.compile on resume to avoid state_dict key mismatch
    if torch.cuda.is_available() and os.name != 'nt' and not is_resume:
        print("\n  [ACCELERATOR] Enabling torch.compile(mode='reduce-overhead')...")
        try:
            # 1. Compile Heavy Modules (Surgical - avoids state_dict key issues)
            # Compile the Transformer Encoder (Biggest gain)
            model.policy.features_extractor = torch.compile(
                model.policy.features_extractor, mode="reduce-overhead"
            )

            # Compile MLP Heads
            model.policy.actor = torch.compile(model.policy.actor, mode="reduce-overhead")
            model.policy.critic = torch.compile(model.policy.critic, mode="reduce-overhead")

            # Compile Targets (Faster soft-updates)
            model.policy.actor_target = torch.compile(model.policy.actor_target, mode="reduce-overhead")
            model.policy.critic_target = torch.compile(model.policy.critic_target, mode="reduce-overhead")

            # 2. JIT Warmup (Force compilation NOW, not at step 0)
            print("  [ACCELERATOR] Triggering JIT Warmup (may take ~30s)...")
            with torch.no_grad():
                # Create dummy observation matching the Dict observation space
                dummy_obs = {}
                for key, space in model.observation_space.spaces.items():
                    dummy_obs[key] = torch.randn(1, *space.shape, device=model.device)
                # Run forward pass to build CUDA graphs
                model.policy(dummy_obs)

            print("  [SUCCESS] Policy compiled and warmed up. Ready for high-speed training.")

        except Exception as e:
            print(f"  [WARNING] torch.compile failed: {e}")
            print("  [INFO] Falling back to standard Eager Execution.")
    elif is_resume:
        print("  [INFO] torch.compile skipped (Resume mode - avoiding state_dict key mismatch).")
    else:
        print("  [INFO] torch.compile skipped (Not on Linux/CUDA).")

    # ==================== Training ====================
    print("\n[4/4] Starting training...")
    print(f"      Total timesteps: {config.total_timesteps:,}")
    print(f"      Eval frequency: {config.eval_freq:,}")
    print(f"      Checkpoint frequency: {config.checkpoint_freq:,}")
    print("\n" + "=" * 70)

    callbacks, unified_callback = create_callbacks(
        config, eval_env, shared_fee=shared_fee
    )

    # Note: is_resume was set earlier in Model Creation section
    # reset_num_timesteps=False continues TensorBoard from previous run

    # ==================== Initialize Curriculum State (Gemini safeguard) ====================
    # Prevents "First Step" lag where shared values might be uninitialized
    if shared_fee is not None:
        shared_fee.value = 0.0

    try:
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=callbacks,
            tb_log_name=tb_log_name,
            progress_bar=True,
            reset_num_timesteps=not is_resume,  # False for resume, True for fresh
        )
    except Exception as e:
        print(f"\n[CRITICAL] Exception during training: {e}")
        emergency_path = os.path.join(config.checkpoint_dir, "emergency_save_internal.zip")
        print(f"[CRITICAL] Attempting internal emergency save to {emergency_path}...")
        try:
            model.save(emergency_path)
            clean_compiled_checkpoint(emergency_path)
            print("[CRITICAL] Emergency save SUCCESS.")
        except Exception as save_err:
            print(f"[CRITICAL] Emergency save FAILED: {save_err}")
        raise e  # Re-raise to stop the script properly
    finally:
        # ==================== Cleanup Resources ====================
        print("\n[Cleanup] Closing environments...")
        if train_env is not None:
            train_env.close()

        # Fix for 'NoneType' object has no attribute 'close' on eval_env
        if eval_env is not None:
            eval_env.close()

        if manager is not None:
            try:
                manager.shutdown()
                print("      Manager shutdown complete.")
            except Exception:
                pass

    # ==================== Capture Training Metrics ====================
    training_metrics = unified_callback.get_training_metrics()

    # === Guard Metrics (WFO Fail-over support) ===
    training_metrics['guard_early_stop'] = False
    training_metrics['guard_stop_reason'] = None
    training_metrics['completion_ratio'] = model.num_timesteps / config.total_timesteps

    # Vérifier si OverfittingGuardCallbackV2 a déclenché un arrêt
    for cb in callbacks:
        if isinstance(cb, OverfittingGuardCallbackV2):
            # Le Guard retourne False dans _on_step si arrêt
            # On détecte via violation_counts qui atteint patience
            violated_signals = [
                name for name, count in cb.violation_counts.items()
                if count >= cb.patience
            ]
            if violated_signals:
                training_metrics['guard_early_stop'] = True
                training_metrics['guard_stop_reason'] = f"Signal(s): {', '.join(violated_signals)}"
            # Also check for multi-signal trigger (2+ active)
            active_signals = sum(1 for c in cb.violation_counts.values() if c > 0)
            if active_signals >= 2 and not training_metrics['guard_early_stop']:
                training_metrics['guard_early_stop'] = True
                active_names = [n for n, c in cb.violation_counts.items() if c > 0]
                training_metrics['guard_stop_reason'] = f"Multi-signal: {', '.join(active_names)}"
            break

    print("\n[Training Diagnostics]")
    print(f"  Action Saturation: {training_metrics['action_saturation']:.3f}")
    print(f"  Avg Entropy: {training_metrics['avg_entropy']:.4f}")
    print(f"  Avg Critic Loss: {training_metrics['avg_critic_loss']:.4f}")
    print(f"  Avg Actor Loss: {training_metrics['avg_actor_loss']:.4f}")
    print(f"  Avg Churn Ratio: {training_metrics['avg_churn_ratio']:.3f}")
    print(f"  Avg Actor Grad Norm: {training_metrics['avg_actor_grad_norm']:.4f}")
    print(f"  Avg Critic Grad Norm: {training_metrics['avg_critic_grad_norm']:.4f}")
    print(f"  Completion Ratio: {training_metrics['completion_ratio']:.1%}")
    if training_metrics['guard_early_stop']:
        print(f"  ⚠️ Guard Early Stop: {training_metrics['guard_stop_reason']}")

    # ==================== Save Final Model ====================
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)

    model.save(config.save_path)
    clean_compiled_checkpoint(config.save_path)
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
    parser.add_argument("--ent-coef", type=float, default=None, help="Entropy coefficient (None = auto)")
    parser.add_argument("--name", type=str, default=None, help="Run name (appears in TensorBoard)")
    parser.add_argument("--gradient-steps", type=int, default=None, help="Gradient steps per update (default: 1)")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (default: auto-detect from VRAM)")
    parser.add_argument("--gamma", type=float, default=None, help="Discount factor (default: 0.99)")
    parser.add_argument("--load-model", type=str, default=None,
                        help="Path to .zip model to resume training from (Lightweight mode: no buffer)")
    parser.add_argument("--load-latest", action="store_true",
                        help="Auto-select latest checkpoint from checkpoint_dir")
    # Hardware override arguments
    parser.add_argument("--n-envs", type=int, default=None,
                        help="Number of parallel envs (default: auto-detect from CPU cores)")
    parser.add_argument("--buffer-size", type=int, default=None,
                        help="Replay buffer size (default: auto-detect from RAM)")
    # GPU-Vectorized Environment (BatchCryptoEnv)
    parser.add_argument("--use-batch-env", action="store_true",
                        help="Use GPU-vectorized BatchCryptoEnv (10-50x speedup vs SubprocVecEnv)")
    # Exploration strategy
    parser.add_argument("--no-sde", action="store_true",
                        help="Disable gSDE (use OrnsteinUhlenbeck action noise instead)")
    parser.add_argument("--action-noise-sigma", type=float, default=None,
                        help="OrnsteinUhlenbeck noise sigma (default: 0.1)")
    parser.add_argument("--action-noise-theta", type=float, default=None,
                        help="OrnsteinUhlenbeck noise theta/mean reversion (default: 0.15)")

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
    # Hardware overrides (will be used by HardwareManager)
    if args.n_envs is not None:
        config.n_envs = args.n_envs
    if args.buffer_size is not None:
        config.buffer_size = args.buffer_size
    # Exploration strategy overrides
    if args.no_sde:
        config.use_sde = False
    if args.action_noise_sigma is not None:
        config.action_noise_sigma = args.action_noise_sigma
    if args.action_noise_theta is not None:
        config.action_noise_theta = args.action_noise_theta

    # Resolve --load-latest to actual checkpoint path
    if config.load_latest:
        latest = find_latest_checkpoint(config.checkpoint_dir)
        if latest:
            config.load_model_path = latest
            print(f"[Auto] Latest checkpoint found: {latest}")
        else:
            print(f"[Warning] No checkpoints found in {config.checkpoint_dir}")

    train(config, use_batch_env=args.use_batch_env)

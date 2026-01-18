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
from multiprocessing import Manager
from typing import Optional, Tuple

from sb3_contrib import TQC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from src.training.callbacks import EvalCallbackWithNoiseControl
import torch
import numpy as np
from functools import partial

from src.config import DEVICE, SEED
from src.utils.hardware import HardwareManager
from src.models.rl_adapter import FoundationFeatureExtractor
from src.training.env import CryptoTradingEnv
from src.training.batch_env import BatchCryptoEnv
from src.training.wrappers import RiskManagementWrapper
from src.training.clipped_optimizer import ClippedAdamW
from src.training.callbacks import (
    StepLoggingCallback,
    DetailTensorboardCallback,
    CurriculumFeesCallback,
    ThreePhaseCurriculumCallback,
    OverfittingGuardCallback,
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
        super().__init__(save_freq, save_path, name_prefix, verbose)
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

    return dict(
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


def _make_train_env(
    parquet_path: str,
    window_size: int,
    commission: float,
    episode_length: int,
    reward_scaling: float,
    downside_coef: float,
    upside_coef: float,
    action_discretization: int,
    churn_coef: float,
    smooth_coef: float,
    target_volatility: float,
    vol_window: int,
    max_leverage: float,
    price_column: str,
    seed: int,
    shared_fee=None,
    shared_smooth=None,
):
    """
    Factory function for creating training environments (SubprocVecEnv compatible).

    Must be at module level (not nested) to be picklable for multiprocessing.
    Each subprocess will call this to create its own environment instance.

    Args:
        shared_fee: Shared memory Value for curriculum fee (SubprocVecEnv).
        shared_smooth: Shared memory Value for curriculum smooth_coef (SubprocVecEnv).
    """
    env = CryptoTradingEnv(
        parquet_path=parquet_path,
        window_size=window_size,
        commission=commission,
        episode_length=episode_length,
        reward_scaling=reward_scaling,
        downside_coef=downside_coef,
        upside_coef=upside_coef,
        action_discretization=action_discretization,
        churn_coef=churn_coef,
        smooth_coef=smooth_coef,
        random_start=True,  # Always random for training
        target_volatility=target_volatility,
        vol_window=vol_window,
        max_leverage=max_leverage,
        price_column=price_column,
        shared_fee=shared_fee,
        shared_smooth=shared_smooth,
    )
    # Set seed for reproducibility across parallel envs
    env.reset(seed=seed)
    return Monitor(env)


def create_environments(config: TrainingConfig, n_envs: int = 1, use_batch_env: bool = False):
    """
    Create training and evaluation environments.

    Args:
        config: Training configuration.
        n_envs: Number of parallel training environments (P0 optimization).
                Use n_envs > 1 for SubprocVecEnv parallelization (2-4x speedup).
                Now compatible with curriculum learning via shared memory.
        use_batch_env: If True, use BatchCryptoEnv (GPU-vectorized) instead of SubprocVecEnv.
                       This runs all envs in a single process using GPU tensors (10-50x speedup).

    Returns:
        Tuple of (train_vec_env, eval_vec_env, shared_fee, shared_smooth, manager).
        shared_fee, shared_smooth, and manager are None if not using curriculum with SubprocVecEnv.
    """
    from src.config import SEED

    # Shared memory for curriculum learning with SubprocVecEnv
    shared_fee = None
    shared_smooth = None
    manager = None  # Track Manager for proper cleanup (Gemini recommendation)

    # Curriculum Learning: start with 0 fees/smoothness if enabled
    if config.use_curriculum:
        initial_commission = 0.0
        initial_smooth = 0.0
        # Create shared memory for SubprocVecEnv curriculum communication
        # Use Manager().Value() instead of Value() because it's picklable
        if n_envs > 1:
            manager = Manager()
            shared_fee = manager.Value('d', 0.0)
            shared_smooth = manager.Value('d', 0.0)
            print("      [P0+Curriculum] Using Manager shared memory for curriculum with SubprocVecEnv")
    else:
        initial_commission = config.commission
        initial_smooth = config.smooth_coef

    # ==================== Training Environment(s) ====================
    if use_batch_env:
        # GPU-Vectorized Batch Environment (Gemini recommendation)
        # All envs run in a single process using PyTorch tensors on GPU
        # Eliminates IPC overhead, achieves 10-50x speedup vs SubprocVecEnv
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
            churn_coef=config.churn_coef,
            smooth_coef=initial_smooth,
            target_volatility=config.target_volatility,
            vol_window=config.vol_window,
            max_leverage=config.max_leverage,
            price_column='BTC_Close',
            observation_noise=config.observation_noise,  # Anti-overfitting
        )
        # Note: Curriculum updates use env.set_attr() for BatchCryptoEnv

    elif n_envs > 1:
        # P0 Optimization: SubprocVecEnv for true parallelization
        # Each subprocess creates its own environment independently
        print(f"      [P0] Creating {n_envs} parallel training environments (SubprocVecEnv)...")

        env_fns = [
            partial(
                _make_train_env,
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
                target_volatility=config.target_volatility,
                vol_window=config.vol_window,
                max_leverage=config.max_leverage,
                price_column='BTC_Close',
                seed=SEED + i,  # Different seed per env for diversity
                shared_fee=shared_fee,
                shared_smooth=shared_smooth,
            )
            for i in range(n_envs)
        ]
        train_vec_env = SubprocVecEnv(env_fns, start_method='spawn')
    else:
        # Single environment (DummyVecEnv) - legacy path
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
            target_volatility=config.target_volatility,
            vol_window=config.vol_window,
            max_leverage=config.max_leverage,
            price_column='BTC_Close',
        )
        train_env_monitored = Monitor(train_env)
        train_vec_env = DummyVecEnv([lambda: train_env_monitored])

    # ==================== Eval Environment (optional) ====================
    # Note: eval_data_path=None in WFO mode to prevent data leakage
    if config.eval_data_path is not None:
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
            target_volatility=config.target_volatility,
            vol_window=config.vol_window,
            max_leverage=config.max_leverage,
            price_column='BTC_Close',
        )

        # Wrap with Risk Management (Circuit Breaker) - ASYMMETRIC
        # Train env: NO CB (agent learns from mistakes)
        # Eval env: CB active at configured threshold (monitoring only)
        if config.use_risk_management:
            # Only wrap eval env with circuit breaker
            val_env = RiskManagementWrapper(
                val_env,
                vol_window=config.risk_vol_window,
                vol_threshold=config.risk_vol_threshold,
                max_drawdown=config.risk_max_drawdown,
                cooldown_steps=config.risk_cooldown,
                augment_obs=config.risk_augment_obs,
            )
            # Calibrate baseline volatility
            val_env.calibrate_baseline(n_steps=2000)

        val_env_monitored = Monitor(val_env)
        eval_vec_env = DummyVecEnv([lambda: val_env_monitored])
    else:
        # WFO mode: no eval env to prevent data leakage
        eval_vec_env = None
        print("      [WFO Mode] Eval environment disabled (eval_data_path=None)")

    return train_vec_env, eval_vec_env, shared_fee, shared_smooth, manager


def create_callbacks(
    config: TrainingConfig,
    eval_env,
    shared_fee=None,
    shared_smooth=None,
) -> tuple[list, DetailTensorboardCallback]:
    """
    Create training callbacks.

    Args:
        config: Training configuration.
        eval_env: Evaluation environment (None in WFO mode).
        shared_fee: Shared memory Value for curriculum fee (SubprocVecEnv).
        shared_smooth: Shared memory Value for curriculum smooth_coef (SubprocVecEnv).

    Returns:
        Tuple of (callbacks_list, detail_callback).
    """
    callbacks = []

    # Step logging callback
    step_callback = StepLoggingCallback(log_freq=config.log_freq)
    callbacks.append(step_callback)

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
        print("      [WFO Mode] Portfolio metrics logged via StepLoggingCallback (custom/nav, custom/position, custom/max_drawdown)")

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

    # Detail Tensorboard callback for reward components
    detail_callback = DetailTensorboardCallback(verbose=0)
    callbacks.append(detail_callback)

    # Curriculum Learning callback (3-Phase: Discovery → Discipline → Refinement)
    # Phases calibrées dans callbacks.py pour ratio |PnL|/|Penalties| ≈ 1.0
    if config.use_curriculum:
        curriculum_callback = ThreePhaseCurriculumCallback(
            total_timesteps=config.total_timesteps,
            shared_smooth=shared_smooth,
            verbose=1
        )
        callbacks.append(curriculum_callback)

    # Overfitting guard - DISABLED (OOS Sharpe 4.75 shows model generalizes well)
    # overfitting_guard = OverfittingGuardCallback(
    #     nav_threshold=5.0,
    #     initial_nav=10_000.0,
    #     check_freq=25_600,  # Check every ~25k steps
    #     verbose=1
    # )
    # callbacks.append(overfitting_guard)

    return callbacks, detail_callback


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
    train_env, eval_env, shared_fee, shared_smooth, manager = create_environments(
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
            policy="MultiInputPolicy",  # Dict obs space (market + position)
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

    callbacks, detail_callback = create_callbacks(
        config, eval_env, shared_fee=shared_fee, shared_smooth=shared_smooth
    )

    # Note: is_resume was set earlier in Model Creation section
    # reset_num_timesteps=False continues TensorBoard from previous run

    # ==================== Initialize Curriculum State (Gemini safeguard) ====================
    # Prevents "First Step" lag where shared values might be uninitialized
    if shared_smooth is not None:
        shared_smooth.value = 0.0
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
    parser.add_argument("--churn-coef", type=float, default=None, help="Churn penalty coefficient (0.1 = 10x amplified trading cost)")
    parser.add_argument("--smooth-coef", type=float, default=None, help="Smoothness penalty coefficient (quadratic penalty on position changes)")
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
    # Hardware overrides (will be used by HardwareManager)
    if args.n_envs is not None:
        config.n_envs = args.n_envs
    if args.buffer_size is not None:
        config.buffer_size = args.buffer_size

    # Resolve --load-latest to actual checkpoint path
    if config.load_latest:
        latest = find_latest_checkpoint(config.checkpoint_dir)
        if latest:
            config.load_model_path = latest
            print(f"[Auto] Latest checkpoint found: {latest}")
        else:
            print(f"[Warning] No checkpoints found in {config.checkpoint_dir}")

    train(config, use_batch_env=args.use_batch_env)

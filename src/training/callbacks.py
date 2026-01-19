# -*- coding: utf-8 -*-
"""
callbacks.py - Consolidated callbacks for SB3 training.

Provides all training-related callbacks:
- TensorBoardStepCallback: Detailed step-level TensorBoard logging
- StepLoggingCallback: Console + TensorBoard logging at intervals
- DetailTensorboardCallback: Reward component logging
- CurriculumFeesCallback: Curriculum learning for fees/penalties
"""

import os
import time
import numpy as np
from typing import TYPE_CHECKING, Optional
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import VecEnv

if TYPE_CHECKING:
    from multiprocessing.sharedctypes import Synchronized


# ============================================================================
# Utility Functions
# ============================================================================

def get_next_run_dir(base_dir: str, prefix: str = "run") -> str:
    """
    Find the next available run number.

    Args:
        base_dir: Base directory for logs.
        prefix: Prefix for folders (default: "run").

    Returns:
        Path to the next run folder (e.g., base_dir/run_3).
    """
    os.makedirs(base_dir, exist_ok=True)

    existing = []
    for name in os.listdir(base_dir):
        if name.startswith(f"{prefix}_"):
            try:
                num = int(name.split("_")[1])
                existing.append(num)
            except (IndexError, ValueError):
                pass

    next_num = max(existing, default=0) + 1
    return os.path.join(base_dir, f"{prefix}_{next_num}")


def get_underlying_batch_env(env):
    """
    Unwrap récursif pour trouver BatchCryptoEnv sous les wrappers SB3.

    SB3 peut wrapper les VecEnv dans VecMonitor, VecNormalize, etc.
    Ces wrappers ne forwardent pas correctement set_attr pour les envs GPU.

    Args:
        env: L'environnement (potentiellement wrappé)

    Returns:
        L'instance BatchCryptoEnv sous-jacente, ou l'env original si non trouvé
    """
    depth = 0
    while depth < 20:
        # Cible atteinte (méthode spécifique BatchCryptoEnv)
        if hasattr(env, 'set_smoothness_penalty'):
            return env
        # Wrapper VecEnv (ex: VecMonitor, VecNormalize)
        elif hasattr(env, 'venv'):
            env = env.venv
        # Wrapper Gym standard
        elif hasattr(env, 'env'):
            env = env.env
        else:
            break
        depth += 1
    return env


# ============================================================================
# Checkpoint Callbacks
# ============================================================================

class RotatingCheckpointCallback(CheckpointCallback):
    """
    Checkpoint callback that keeps the last N checkpoints to prevent data loss.
    Refactored 2026-01-16 to fix aggressive deletion bug.
    """
    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "rl_model", verbose: int = 0, keep_last: int = 2):
        super().__init__(save_freq, save_path, name_prefix, verbose)
        self.keep_last = keep_last
        self.saved_checkpoints = []  # List to track paths

    def _on_step(self) -> bool:
        # Call parent to save
        result = super()._on_step()

        # Logic matches CheckpointCallback naming convention
        if self.n_calls % self.save_freq == 0:
            step = self.num_timesteps
            current_path = os.path.join(self.save_path, f"{self.name_prefix}_{step}_steps.zip")

            # Track new file if it exists
            if os.path.exists(current_path):
                self.saved_checkpoints.append(current_path)

                # Prune oldest if we exceed limit
                while len(self.saved_checkpoints) > self.keep_last:
                    to_remove = self.saved_checkpoints.pop(0)
                    try:
                        if os.path.exists(to_remove):
                            os.remove(to_remove)
                            if self.verbose > 0:
                                print(f"  [Disk Opt] Pruned old checkpoint: {os.path.basename(to_remove)}")
                    except OSError as e:
                        print(f"  [Warning] Failed to prune {to_remove}: {e}")
        return result


# ============================================================================
# TensorBoard Callbacks
# ============================================================================

class TensorBoardStepCallback(BaseCallback):
    """
    Callback that logs all relevant metrics at each step.

    Uses SummaryWriter directly to avoid conflicts with SB3's internal logger.
    Runs are automatically numbered (run_1, run_2, etc.).

    Logged metrics:
    - rollout/reward: Instant reward
    - rollout/ep_rew_mean: Episode total reward (at episode end)
    - rollout/ep_len_mean: Episode length (at episode end)
    - env/portfolio_value: Portfolio NAV
    - env/price: Current asset price
    - env/max_drawdown: Max drawdown since start (%)
    - train/actor_loss, critic_loss, ent_coef, ent_coef_loss
    """

    def __init__(self, log_dir: str = None, run_name: str = None, log_freq: int = 1, verbose: int = 0):
        """
        Args:
            log_dir: Base directory for TensorBoard logs.
            run_name: Run name (optional). If None, auto-numbered.
            log_freq: Logging frequency (1 = every step).
            verbose: Verbosity level.
        """
        super().__init__(verbose)
        self.log_dir = log_dir
        self.run_name = run_name
        self.log_freq = log_freq
        self.writer = None
        self.run_path = None

    def _on_training_start(self) -> None:
        """Initialize SummaryWriter at training start."""
        if self.log_dir is None:
            base_dir = self.logger.dir if hasattr(self.logger, 'dir') else "./logs/tensorboard_steps/"
        else:
            base_dir = self.log_dir

        if self.run_name is not None:
            self.run_path = os.path.join(base_dir, self.run_name)
        else:
            self.run_path = get_next_run_dir(base_dir, prefix="run")

        os.makedirs(self.run_path, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.run_path)

        if self.verbose > 0:
            print(f"[TensorBoardStepCallback] Logging to {self.run_path}")

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq != 0:
            return True

        if self.writer is None:
            return True

        step = self.num_timesteps

        try:
            # Log rewards
            if 'rewards' in self.locals:
                rewards = self.locals['rewards']
                if rewards is not None and len(rewards) > 0:
                    self.writer.add_scalar("rollout/reward", float(rewards[0]), step)

            # Log environment metrics (OPTIMIZED: use get_global_metrics for BatchCryptoEnv)
            env = self.training_env
            if hasattr(env, 'get_global_metrics'):
                # BatchCryptoEnv path - direct GPU access
                metrics = env.get_global_metrics()
                self.writer.add_scalar("env/portfolio_value", metrics['portfolio_value'], step)
                self.writer.add_scalar("env/price", metrics['price'], step)
                self.writer.add_scalar("env/max_drawdown", metrics['max_drawdown'] * 100, step)

                # Reward components (for observability)
                if 'avg_rew_pnl' in metrics:
                    self.writer.add_scalar("rewards/pnl", metrics['avg_rew_pnl'], step)
                    self.writer.add_scalar("rewards/churn_penalty", metrics['avg_rew_churn'], step)
                    self.writer.add_scalar("rewards/smooth_penalty", metrics['avg_rew_smooth'], step)
            else:
                # Fallback for SubprocVecEnv/DummyVecEnv - read from infos
                if 'infos' in self.locals:
                    infos = self.locals['infos']
                    if infos is not None and len(infos) > 0:
                        info = infos[0]
                        if info is not None and isinstance(info, dict):
                            if 'portfolio_value' in info:
                                self.writer.add_scalar("env/portfolio_value", info['portfolio_value'], step)
                            if 'price' in info:
                                self.writer.add_scalar("env/price", info['price'], step)
                            if 'max_drawdown' in info:
                                self.writer.add_scalar("env/max_drawdown", info['max_drawdown'] * 100, step)

            # Log episode info (still from infos - only present on done)
            if 'infos' in self.locals:
                infos = self.locals['infos']
                if infos is not None:
                    for info in infos:
                        if info and 'episode' in info:
                            ep_info = info['episode']
                            if 'r' in ep_info:
                                self.writer.add_scalar("rollout/ep_rew_mean", ep_info['r'], step)
                            if 'l' in ep_info:
                                self.writer.add_scalar("rollout/ep_len_mean", ep_info['l'], step)
                            break  # Only log first episode

            # Log training metrics
            if hasattr(self.model, 'logger') and hasattr(self.model.logger, 'name_to_value'):
                logger_values = self.model.logger.name_to_value
                for key in ['train/actor_loss', 'train/critic_loss', 'train/ent_coef', 'train/ent_coef_loss']:
                    if key in logger_values:
                        self.writer.add_scalar(key, logger_values[key], step)

        except Exception as e:
            if self.verbose > 0:
                print(f"[TensorBoardStepCallback] Error logging: {e}")

        return True

    def _on_training_end(self) -> None:
        """Close SummaryWriter at training end."""
        if self.writer is not None:
            self.writer.close()
            if self.verbose > 0:
                print("[TensorBoardStepCallback] Writer closed")


class StepLoggingCallback(BaseCallback):
    """
    Callback for logging at every N steps (console + TensorBoard).
    """

    def __init__(self, log_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_trades = []
        self.last_episode_info = {}
        # FPS tracking
        self.last_time = None
        self.last_step = 0

    def _init_callback(self) -> None:
        """Initialize FPS tracking at callback start."""
        self.last_time = time.time()
        self.last_step = 0

    def _on_step(self) -> bool:
        # Collect episode info if available (only on done)
        if self.locals.get("infos"):
            for info in self.locals["infos"]:
                if info and "episode" in info:
                    ep_reward = info["episode"]["r"]
                    ep_length = info["episode"]["l"]
                    self.episode_rewards.append(ep_reward)
                    self.episode_lengths.append(ep_length)
                    self.last_episode_info = {"reward": ep_reward, "length": ep_length}

        # Log every log_freq steps
        if self.num_timesteps % self.log_freq == 0:
            # Get metrics from env (OPTIMIZED: use get_global_metrics for BatchCryptoEnv)
            env = self.training_env
            if hasattr(env, 'get_global_metrics'):
                metrics = env.get_global_metrics()
                self.last_episode_info["nav"] = metrics["portfolio_value"]
                self.last_episode_info["position"] = metrics["position_pct"]
                self.last_episode_info["max_drawdown"] = metrics["max_drawdown"]

                # Log reward components (for observability)
                # Note: get_global_metrics uses keys with '/' prefix (reward/pnl_component)
                # but DetailTensorboardCallback already logs these via internal/reward/*
                # We skip logging here to avoid duplication

            # Log to TensorBoard
            # Note: mean_reward and mean_length are already logged by SB3 as
            # rollout/ep_rew_mean and rollout/ep_len_mean, so we skip them here

            if self.last_episode_info:
                if "nav" in self.last_episode_info:
                    self.logger.record("custom/nav", self.last_episode_info["nav"])
                if "position" in self.last_episode_info:
                    self.logger.record("custom/position", self.last_episode_info["position"])
                if "max_drawdown" in self.last_episode_info:
                    # Log max_drawdown as percentage
                    self.logger.record("custom/max_drawdown", self.last_episode_info["max_drawdown"] * 100)

            self.logger.dump(self.num_timesteps)

            # Calculate FPS manually (fixes FPS=0 bug with BatchCryptoEnv)
            current_time = time.time()
            if self.last_time is not None:
                dt = current_time - self.last_time
                if dt > 0:
                    fps = (self.num_timesteps - self.last_step) / dt
                    self.logger.record("time/fps_live", fps)
                else:
                    fps = 0
            else:
                fps = 0
            self.last_time = current_time
            self.last_step = self.num_timesteps

            # Console log
            nav = self.last_episode_info.get("nav", 0)
            pos = self.last_episode_info.get("position", 0)
            max_dd = self.last_episode_info.get("max_drawdown", 0) * 100  # Convert to %
            
            # Calculate mean reward for display (from last 10 episodes if available)
            if self.episode_rewards:
                mean_reward_display = np.mean(self.episode_rewards[-10:])
            else:
                mean_reward_display = 0

            print(f"Step {self.num_timesteps:>7} | "
                  f"Reward: {mean_reward_display:>8.2f} | "
                  f"NAV: {nav:>10.2f} | "
                  f"Pos: {pos:>+5.2f} | "
                  f"DD: {max_dd:>5.1f}% | "
                  f"FPS: {fps:>7.0f}")

        return True


class DetailTensorboardCallback(BaseCallback):
    """
    Callback for logging reward components to TensorBoard.

    Logs: log_return, penalty_vol, churn_penalty, total_raw.
    Also captures diagnostic metrics for post-training analysis.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_churn_penalties = []
        self.episode_log_returns = []
        self.episode_position_deltas = []

        # Diagnostic metrics
        self.all_actions = []
        self.entropy_values = []
        self.critic_losses = []
        self.actor_losses = []
        self.churn_ratios = []
        self.actor_grad_norms = []
        self.critic_grad_norms = []

    def _on_step(self) -> bool:
        # Track actions for saturation analysis
        if self.locals.get("actions") is not None:
            self.all_actions.extend(np.abs(self.locals["actions"]).flatten())

        # ═══════════════════════════════════════════════════════════════════
        # DIRECT GPU METRIC POLLING (replaces info-based logging)
        # ═══════════════════════════════════════════════════════════════════
        # Unwrap to find BatchCryptoEnv under SB3 wrappers
        real_env = get_underlying_batch_env(self.model.env)

        if real_env is not None and hasattr(real_env, "get_global_metrics"):
            metrics = real_env.get_global_metrics()
            for key, value in metrics.items():
                # Use record_mean for smoother TensorBoard curves
                self.logger.record_mean(f"internal/{key}", value)

            # Track for episode aggregation
            if "reward/pnl_component" in metrics:
                self.episode_log_returns.append(metrics["reward/pnl_component"])
            if "reward/churn_cost" in metrics:
                self.episode_churn_penalties.append(metrics["reward/churn_cost"])

        # Episode end logging (from info dict - still needed for episode stats)
        if self.locals.get("infos"):
            info = self.locals["infos"][0]
            if "episode" in info:
                if self.episode_churn_penalties:
                    total_churn = sum(self.episode_churn_penalties)
                    total_log_ret = sum(self.episode_log_returns)

                    self.logger.record("churn/episode_total_penalty", total_churn)
                    self.logger.record("churn/episode_total_log_return", total_log_ret)

                    if abs(total_log_ret) > 1e-8:
                        ratio = abs(total_churn / total_log_ret)
                        self.logger.record("churn/penalty_to_return_ratio", ratio)
                        self.churn_ratios.append(ratio)

                # Reset episode accumulators
                self.episode_churn_penalties = []
                self.episode_log_returns = []
                self.episode_position_deltas = []

        # Capture training metrics
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

        # Log gradient norms
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


# ============================================================================
# Curriculum Learning Callback
# ============================================================================

class CurriculumFeesCallback(BaseCallback):
    """
    Curriculum Learning: Start with 0 fees/penalties, linearly increase to targets.

    This callback implements a curriculum learning strategy where transaction costs
    and smoothness penalties start at 0 and gradually increase to their target values.

    Supports both DummyVecEnv (direct env access) and SubprocVecEnv (shared memory).

    Args:
        target_fee_rate: Target commission fee (e.g., 0.0006 = 0.06%).
        target_smooth_coef: Target smoothness penalty coefficient.
        warmup_steps: Number of steps to reach target values.
        shared_fee: Shared memory Value for fee (SubprocVecEnv).
        shared_smooth: Shared memory Value for smooth_coef (SubprocVecEnv).
        verbose: Verbosity level.

    Example:
        >>> callback = CurriculumFeesCallback(
        ...     target_fee_rate=0.0006,
        ...     target_smooth_coef=1.0,
        ...     warmup_steps=50_000
        ... )
    """

    def __init__(
        self,
        target_fee_rate: float = 0.0006,
        target_smooth_coef: float = 1.0,
        warmup_steps: int = 50_000,
        shared_fee: Optional["Synchronized"] = None,
        shared_smooth: Optional["Synchronized"] = None,
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.target_fee_rate = target_fee_rate
        self.target_smooth_coef = target_smooth_coef
        self.warmup_steps = warmup_steps

        # Shared memory for SubprocVecEnv compatibility
        self.shared_fee = shared_fee
        self.shared_smooth = shared_smooth

        self.current_fee = 0.0
        self.current_smooth = 0.0
        self._warmup_complete = False

    def _on_step(self) -> bool:
        progress = min(1.0, self.num_timesteps / self.warmup_steps)

        self.current_fee = progress * self.target_fee_rate
        self.current_smooth = progress * self.target_smooth_coef

        # Update via shared memory (SubprocVecEnv)
        if self.shared_fee is not None:
            self.shared_fee.value = self.current_fee
        if self.shared_smooth is not None:
            self.shared_smooth.value = self.current_smooth

        # ALWAYS update envs (for BatchCryptoEnv direct access)
        self._update_envs()

        self.logger.record("curriculum/fee_rate", self.current_fee)
        self.logger.record("curriculum/smooth_coef", self.current_smooth)
        self.logger.record("curriculum/progress", progress)

        if progress >= 1.0 and not self._warmup_complete:
            self._warmup_complete = True
            if self.verbose > 0:
                print(f"\n[Curriculum] Warmup complete at step {self.num_timesteps}")
                print(f"  fee_rate: {self.current_fee:.6f}")
                print(f"  smooth_coef: {self.current_smooth:.2f}")

        return True

    def _update_envs(self):
        """Update penalties on all environments."""
        env = self.model.env

        if isinstance(env, VecEnv):
            for i in range(env.num_envs):
                base_env = env.envs[i]
                while hasattr(base_env, 'env'):
                    base_env = base_env.env

                if hasattr(base_env, 'update_penalties'):
                    base_env.update_penalties(
                        fee_rate=self.current_fee,
                        smooth_coef=self.current_smooth
                    )
        else:
            base_env = env
            while hasattr(base_env, 'env'):
                base_env = base_env.env

            if hasattr(base_env, 'update_penalties'):
                base_env.update_penalties(
                    fee_rate=self.current_fee,
                    smooth_coef=self.current_smooth
                )

    def _on_training_start(self) -> None:
        if self.verbose > 0:
            print(f"\n[Curriculum] Starting curriculum learning:")
            print(f"  target_fee_rate: {self.target_fee_rate:.6f}")
            print(f"  target_smooth_coef: {self.target_smooth_coef:.2f}")
            print(f"  warmup_steps: {self.warmup_steps:,}")


class ThreePhaseCurriculumCallback(BaseCallback):
    """
    Three-Phase Curriculum Learning with IPC Fault Tolerance.
    Refactored 2026-01-16 to fix multiprocessing crashes.
    """
    # Modified by CryptoRL: reduced smooth_coef 0.02 -> 0.005 to unblock trading
    PHASES = [
        {'end_progress': 0.1, 'churn': (0.0, 0.10), 'smooth': (0.0, 0.0)},
        {'end_progress': 0.3, 'churn': (0.10, 0.50), 'smooth': (0.0, 0.005)},
        {'end_progress': 1.0, 'churn': (0.50, 0.50), 'smooth': (0.005, 0.005)},
    ]

    def __init__(self, total_timesteps: int, shared_smooth: Optional["Synchronized"] = None, verbose: int = 0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.shared_smooth = shared_smooth
        self.current_smooth = 0.0
        self.current_churn = 0.0
        self._phase = 1

    def _on_step(self) -> bool:
        progress = self.num_timesteps / self.total_timesteps

        # Locate Phase
        phase_cfg = self.PHASES[-1]
        for i, p in enumerate(self.PHASES):
            if progress <= p['end_progress']:
                phase_cfg = p
                self._phase = i + 1
                break

        # Interpolate
        prev_end = 0.0 if self._phase == 1 else self.PHASES[self._phase - 2]['end_progress']
        phase_progress = (progress - prev_end) / (phase_cfg['end_progress'] - prev_end)
        phase_progress = max(0.0, min(1.0, phase_progress))

        c_start, c_end = phase_cfg['churn']
        s_start, s_end = phase_cfg['smooth']

        self.current_churn = c_start + (c_end - c_start) * phase_progress
        self.current_smooth = s_start + (s_end - s_start) * phase_progress

        # Apply Logic with IPC Safety Net
        if self.shared_smooth is not None:
            try:
                self.shared_smooth.value = self.current_smooth
            except (ConnectionResetError, BrokenPipeError, FileNotFoundError, OSError):
                # Fail gracefully. Do not kill the training run for a metric update.
                pass

        # ALWAYS update env (for BatchCryptoEnv penalties + curriculum_lambda)
        self._update_envs()

        self.logger.record("curriculum/smooth_coef", self.current_smooth)
        self.logger.record("curriculum/churn_coef", self.current_churn)
        self.logger.record("curriculum/phase", self._phase)
        self.logger.record("curriculum/progress", progress)

        # Log curriculum_lambda from env if available
        real_env = get_underlying_batch_env(self.model.env)
        if hasattr(real_env, 'curriculum_lambda'):
            self.logger.record("curriculum/lambda", real_env.curriculum_lambda)

        return True

    def _update_envs(self):
        """Update penalties on all environments (unwrap for BatchCryptoEnv)."""
        # Unwrap pour atteindre BatchCryptoEnv sous les wrappers SB3
        real_env = get_underlying_batch_env(self.model.env)

        # Calculate progress for curriculum lambda
        progress = self.num_timesteps / self.total_timesteps

        # Appel direct (contourne le Wrapper Hell de SB3)
        if hasattr(real_env, 'set_smoothness_penalty'):
            real_env.set_smoothness_penalty(self.current_smooth)
            real_env.set_churn_penalty(self.current_churn)
            # Sync progress for curriculum_lambda (AAAI 2024 Curriculum Learning)
            if hasattr(real_env, 'set_progress'):
                real_env.set_progress(progress)
            return

        # Fallback: DummyVecEnv path (CPU envs)
        if isinstance(self.model.env, VecEnv) and hasattr(self.model.env, 'envs'):
            for i in range(self.model.env.num_envs):
                base_env = self.model.env.envs[i]
                while hasattr(base_env, 'env'):
                    base_env = base_env.env
                if hasattr(base_env, 'set_smooth_coef'):
                    base_env.set_smooth_coef(self.current_smooth)
                if hasattr(base_env, 'set_churn_coef'):
                    base_env.set_churn_coef(self.current_churn)

    def _on_training_start(self) -> None:
        if self.verbose > 0:
            print(f"\n[3-Phase Curriculum] Configuration (IPC-safe):")
            for i, phase in enumerate(self.PHASES):
                pct = int(phase['end_progress'] * 100)
                churn_str = f"{phase['churn'][0]:.2f}→{phase['churn'][1]:.2f}"
                smooth_str = f"{phase['smooth'][0]:.4f}→{phase['smooth'][1]:.4f}"
                print(f"  Phase {i+1} (0-{pct}%): churn={churn_str}, smooth={smooth_str}")


# ============================================================================
# Overfitting Guard Callback
# ============================================================================

class OverfittingGuardCallback(BaseCallback):
    """
    Early stopping if training shows signs of overfitting.

    Triggers abort if NAV exceeds threshold (e.g., 5x initial = +400%).
    Such returns are unrealistic and indicate memorization of training data.
    """

    def __init__(
        self,
        nav_threshold: float = 5.0,  # 5x = +400%
        initial_nav: float = 10_000.0,
        check_freq: int = 25_600,
        verbose: int = 1
    ):
        """
        Args:
            nav_threshold: Multiplier of initial NAV to trigger stop (5.0 = +400%).
            initial_nav: Starting portfolio value.
            check_freq: How often to check (in timesteps).
            verbose: Verbosity level.
        """
        super().__init__(verbose)
        self.nav_threshold = nav_threshold
        self.initial_nav = initial_nav
        self.check_freq = check_freq
        self.max_nav_seen = initial_nav

    def _on_step(self) -> bool:
        if self.num_timesteps % self.check_freq != 0:
            return True

        # Get current NAV from env
        env = self.training_env
        if hasattr(env, 'get_global_metrics'):
            metrics = env.get_global_metrics()
            current_nav = metrics.get("portfolio_value", self.initial_nav)
            self.max_nav_seen = max(self.max_nav_seen, current_nav)

            # Check threshold
            if self.max_nav_seen > self.initial_nav * self.nav_threshold:
                ratio = self.max_nav_seen / self.initial_nav
                print("\n" + "=" * 60)
                print("  EARLY STOPPING: Potential Overfitting Detected!")
                print("=" * 60)
                print(f"  Max NAV seen: {self.max_nav_seen:,.0f}")
                print(f"  Ratio vs initial: {ratio:.1f}x (+{(ratio-1)*100:.0f}%)")
                print(f"  Threshold: {self.nav_threshold}x")
                print("  Such returns are unrealistic - likely memorization.")
                print("=" * 60 + "\n")
                return False  # Stop training

        return True


# ============================================================================
# Evaluation Callback with Observation Noise Management
# ============================================================================

class EvalCallbackWithNoiseControl(EvalCallback):
    """
    Wrapper around EvalCallback that automatically disables observation noise
    in BatchCryptoEnv during evaluation and re-enables it after.

    This ensures that evaluation metrics are not affected by observation noise,
    which should only be active during training for regularization.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the callback with noise control.

        All arguments are passed to EvalCallback.
        """
        super().__init__(*args, **kwargs)
        self._training_env_has_noise = False
        self._eval_env_has_noise = False
        self._last_eval_step = -1

    def _on_step(self) -> bool:
        """
        Override to manage observation noise before/after evaluation.
        """
        # Check if evaluation will occur (EvalCallback logic)
        will_eval = (self.eval_freq > 0 and 
                     self.n_calls % self.eval_freq == 0 and 
                     self.n_calls != self._last_eval_step)
        
        if will_eval:
            # Before evaluation: disable noise in training env if it's BatchCryptoEnv
            train_env = self._get_batch_env(self.training_env)
            if train_env is not None and hasattr(train_env, 'set_training_mode'):
                self._training_env_has_noise = train_env.training
                train_env.set_training_mode(False)
                if self.verbose > 0:
                    print(f"  [Noise Control] Disabled observation noise in training env for evaluation")

            # Check eval environment (should already be False, but ensure it)
            eval_env = self._get_batch_env(self.eval_env)
            if eval_env is not None and hasattr(eval_env, 'set_training_mode'):
                self._eval_env_has_noise = eval_env.training
                eval_env.set_training_mode(False)
                if self.verbose > 0:
                    print(f"  [Noise Control] Disabled observation noise in eval env")

        # Call parent evaluation (this will trigger evaluation if needed)
        result = super()._on_step()

        # After evaluation: re-enable noise in training env
        if will_eval:
            train_env = self._get_batch_env(self.training_env)
            if train_env is not None and hasattr(train_env, 'set_training_mode'):
                train_env.set_training_mode(self._training_env_has_noise)
                if self.verbose > 0 and self._training_env_has_noise:
                    print(f"  [Noise Control] Re-enabled observation noise in training env")

            # Restore eval env state (should stay False, but restore original)
            eval_env = self._get_batch_env(self.eval_env)
            if eval_env is not None and hasattr(eval_env, 'set_training_mode'):
                eval_env.set_training_mode(self._eval_env_has_noise)
            
            self._last_eval_step = self.n_calls

        return result

    def _get_batch_env(self, env):
        """
        Unwrap environment to find BatchCryptoEnv instance.

        Args:
            env: Environment (potentially wrapped).

        Returns:
            BatchCryptoEnv instance if found, None otherwise.
        """
        return get_underlying_batch_env(env)


# ============================================================================
# PLO (Predictive Lagrangian Optimization) Callbacks
# ============================================================================

class PLOAdaptivePenaltyCallback(BaseCallback):
    """
    Predictive Lagrangian Optimization (PLO) for Drawdown-based Adaptive Penalties.

    This callback implements a PID controller that dynamically adjusts the
    downside risk penalty multiplier based on current and predicted drawdown.

    VERSION: Production - Includes all protections:
    1. Robust prediction via np.polyfit (instead of naive difference)
    2. Adaptive quantile (90% if num_envs >= 16, else LogSumExp)
    3. Lambda smoothing (max ±0.05/step)
    4. Prediction only if slope positive (worsening)
    5. "Wake-up Shock" protection (freeze PID in Phase 1 curriculum)

    Reference: "Predictive Lagrangian Optimization" (2025)
    """

    def __init__(
        self,
        # Drawdown Constraint
        dd_threshold: float = 0.10,
        dd_lambda_min: float = 1.0,
        dd_lambda_max: float = 5.0,
        # PID Gains
        dd_Kp: float = 2.0,
        dd_Ki: float = 0.05,
        dd_Kd: float = 0.3,
        # PLO Prediction
        prediction_horizon: int = 50,
        use_prediction: bool = True,
        # Anti-windup, decay and smoothing
        integral_max: float = 2.0,
        decay_rate: float = 0.995,
        max_lambda_change: float = 0.05,
        # Risk measurement
        dd_quantile: float = 0.9,
        # Logging
        log_freq: int = 100,
        verbose: int = 0,
    ):
        super().__init__(verbose)

        # Parameters
        self.dd_threshold = dd_threshold
        self.dd_lambda_min = dd_lambda_min
        self.dd_lambda_max = dd_lambda_max
        self.dd_Kp = dd_Kp
        self.dd_Ki = dd_Ki
        self.dd_Kd = dd_Kd
        self.prediction_horizon = prediction_horizon
        self.use_prediction = use_prediction
        self.integral_max = integral_max
        self.decay_rate = decay_rate
        self.max_lambda_change = max_lambda_change
        self.dd_quantile = dd_quantile
        self.log_freq = log_freq

        # PID Controller State
        self.dd_integral = 0.0
        self.dd_prev_violation = 0.0
        self.dd_lambda = 1.0

        # Buffer for prediction
        self.dd_history = []

    def _on_step(self) -> bool:
        real_env = get_underlying_batch_env(self.model.env)

        if not hasattr(real_env, 'current_drawdowns'):
            return True

        # ═══════════════════════════════════════════════════════════════════
        # "WAKE-UP SHOCK" PROTECTION
        # Don't accumulate integral if curriculum is not yet active
        # Prevents λ from rising to 5.0 while agent can't react
        # ═══════════════════════════════════════════════════════════════════
        curriculum_active = True
        if hasattr(real_env, 'curriculum_lambda'):
            if real_env.curriculum_lambda < 0.05:  # Phase 1: curriculum ≈ 0
                # Fast decay of integral to avoid saturation
                self.dd_integral *= 0.9
                curriculum_active = False

        # ═══════════════════════════════════════════════════════════════════
        # ADAPTIVE MEASUREMENT (based on num_envs)
        # Quantile unstable if few envs → use LogSumExp
        # ═══════════════════════════════════════════════════════════════════
        import torch
        current_dd = real_env.current_drawdowns

        if real_env.num_envs >= 16:
            # Enough envs for stable quantile
            metric_dd = torch.quantile(current_dd, self.dd_quantile).item()
        else:
            # Few envs: LogSumExp (smooth approximation of max)
            temperature = 10.0
            metric_dd = (torch.logsumexp(current_dd * temperature, dim=0) / temperature).item()

        max_dd = current_dd.max().item()
        violation = max(0.0, metric_dd - self.dd_threshold)

        # Store for prediction
        self.dd_history.append(metric_dd)
        if len(self.dd_history) > self.prediction_horizon:
            self.dd_history.pop(0)

        # ═══════════════════════════════════════════════════════════════════
        # ROBUST PREDICTION (only if curriculum active)
        # Polyfit on 15 points, only if slope positive
        # ═══════════════════════════════════════════════════════════════════
        predicted_violation = 0.0
        slope = 0.0

        if curriculum_active and self.use_prediction and len(self.dd_history) >= 15:
            y = np.array(self.dd_history[-15:])
            x = np.arange(len(y))
            slope, intercept = np.polyfit(x, y, 1)

            # STRICT: Only predict if slope POSITIVE (worsening)
            if slope > 0:
                future_dd = slope * (len(y) + 10) + intercept
                predicted_violation = max(0.0, future_dd - self.dd_threshold)

        # ═══════════════════════════════════════════════════════════════════
        # PID CONTROLLER (conditioned by curriculum)
        # ═══════════════════════════════════════════════════════════════════
        if curriculum_active:
            if violation > 0 or predicted_violation > 0:
                effective_violation = max(violation, 0.7 * predicted_violation)

                P = self.dd_Kp * effective_violation
                self.dd_integral += self.dd_Ki * violation
                self.dd_integral = np.clip(self.dd_integral, 0, self.integral_max)
                I = self.dd_integral
                D = self.dd_Kd * (violation - self.dd_prev_violation)

                target_lambda = self.dd_lambda_min + P + I + D
                target_lambda = np.clip(target_lambda, self.dd_lambda_min, self.dd_lambda_max)
            else:
                # Decay towards λ_min
                target_lambda = max(self.dd_lambda_min, self.dd_lambda * self.decay_rate)
                self.dd_integral *= 0.995

            # Smoothing: limit change per step
            change = np.clip(target_lambda - self.dd_lambda,
                             -self.max_lambda_change, self.max_lambda_change)
            self.dd_lambda = self.dd_lambda + change

        self.dd_prev_violation = violation

        # Apply to environment
        if hasattr(real_env, 'set_downside_multiplier'):
            real_env.set_downside_multiplier(self.dd_lambda)

        # Logging
        if self.num_timesteps % self.log_freq == 0:
            self.logger.record("plo/dd_violation", violation)
            self.logger.record("plo/dd_predicted_violation", predicted_violation)
            self.logger.record("plo/dd_multiplier", self.dd_lambda)
            self.logger.record("plo/dd_integral", self.dd_integral)
            self.logger.record("plo/dd_slope", slope)
            self.logger.record("plo/metric_drawdown", metric_dd)
            self.logger.record("plo/max_drawdown", max_dd)
            self.logger.record("plo/curriculum_active", float(curriculum_active))

        return True

    def _on_training_start(self) -> None:
        if self.verbose > 0:
            print(f"\n[PLO] Predictive Lagrangian Optimization (Drawdown):")
            print(f"  DD threshold: {self.dd_threshold:.1%}")
            print(f"  Lambda range: [{self.dd_lambda_min}, {self.dd_lambda_max}]")
            print(f"  PID gains: Kp={self.dd_Kp}, Ki={self.dd_Ki}, Kd={self.dd_Kd}")
            print(f"  Prediction: {'polyfit (robust)' if self.use_prediction else 'disabled'}")
            print(f"  DD Quantile: {self.dd_quantile:.0%} (adaptive by num_envs)")
            print(f"  Max λ change/step: ±{self.max_lambda_change}")
            print(f"  Wake-up Shock protection: enabled")


class PLOChurnCallback(BaseCallback):
    """
    Predictive Lagrangian Optimization (PLO) for Churn-based Adaptive Penalties.

    This callback implements a PID controller that dynamically adjusts the
    churn penalty multiplier based on current and predicted turnover rate.

    Includes prediction because turnover has inertia (persistent trading patterns).

    VERSION: Production - Includes:
    1. Robust prediction via np.polyfit with minimum slope threshold
    2. "Leak Minimum" fix for "Profit Gate Paradox"
    3. Curriculum protection (churn_coef < 0.05)
    """

    def __init__(
        self,
        # Turnover Constraint
        turnover_threshold: float = 0.08,  # 8% avg change per step (~2 repos/day)
        turnover_lambda_min: float = 1.0,
        turnover_lambda_max: float = 5.0,
        # PID Gains
        turnover_Kp: float = 2.5,
        turnover_Ki: float = 0.08,
        turnover_Kd: float = 0.4,
        # Prediction
        prediction_horizon: int = 50,
        use_prediction: bool = True,
        # Stability
        integral_max: float = 2.0,
        decay_rate: float = 0.995,
        max_lambda_change: float = 0.08,
        # Logging
        log_freq: int = 100,
        verbose: int = 0,
    ):
        super().__init__(verbose)

        self.turnover_threshold = turnover_threshold
        self.turnover_lambda_min = turnover_lambda_min
        self.turnover_lambda_max = turnover_lambda_max
        self.turnover_Kp = turnover_Kp
        self.turnover_Ki = turnover_Ki
        self.turnover_Kd = turnover_Kd
        self.prediction_horizon = prediction_horizon
        self.use_prediction = use_prediction
        self.integral_max = integral_max
        self.decay_rate = decay_rate
        self.max_lambda_change = max_lambda_change
        self.log_freq = log_freq

        # Controller state
        self.turnover_integral = 0.0
        self.turnover_prev_violation = 0.0
        self.turnover_lambda = 1.0

        # Buffer for prediction
        self.turnover_history = []

    def _on_step(self) -> bool:
        real_env = get_underlying_batch_env(self.model.env)

        if not hasattr(real_env, 'current_position_deltas'):
            return True

        # ═══════════════════════════════════════════════════════════════════
        # CURRICULUM PROTECTION
        # Don't activate PLO if churn_coef ≈ 0
        # ═══════════════════════════════════════════════════════════════════
        curriculum_active = True
        if hasattr(real_env, '_current_churn_coef'):
            if real_env._current_churn_coef < 0.05:
                self.turnover_integral *= 0.9  # Fast decay
                curriculum_active = False

        # ═══════════════════════════════════════════════════════════════════
        # TURNOVER MEASUREMENT
        # ═══════════════════════════════════════════════════════════════════
        current_deltas = real_env.current_position_deltas
        avg_turnover = current_deltas.mean().item()

        self.turnover_history.append(avg_turnover)
        if len(self.turnover_history) > self.prediction_horizon:
            self.turnover_history.pop(0)

        # Average turnover over window
        metric_turnover = np.mean(self.turnover_history[-20:]) if len(self.turnover_history) >= 20 else avg_turnover
        max_turnover = current_deltas.max().item()
        violation = max(0.0, metric_turnover - self.turnover_threshold)

        # ═══════════════════════════════════════════════════════════════════
        # PREDICTION (if curriculum active)
        # ═══════════════════════════════════════════════════════════════════
        predicted_violation = 0.0
        slope = 0.0

        if curriculum_active and self.use_prediction and len(self.turnover_history) >= 15:
            y = np.array(self.turnover_history[-15:])
            x = np.arange(len(y))
            slope, intercept = np.polyfit(x, y, 1)

            # AUDIT FIX: Minimum threshold to ignore noise
            # Only predict if turnover rising significantly (> 0.1% per step)
            MIN_SLOPE = 0.001
            if slope > MIN_SLOPE:
                future_turnover = slope * (len(y) + 10) + intercept
                predicted_violation = max(0.0, future_turnover - self.turnover_threshold)

        # ═══════════════════════════════════════════════════════════════════
        # PID CONTROLLER
        # ═══════════════════════════════════════════════════════════════════
        if curriculum_active:
            if violation > 0 or predicted_violation > 0:
                effective_violation = max(violation, 0.6 * predicted_violation)

                P = self.turnover_Kp * effective_violation
                self.turnover_integral += self.turnover_Ki * violation
                self.turnover_integral = np.clip(self.turnover_integral, 0, self.integral_max)
                I = self.turnover_integral
                D = self.turnover_Kd * (violation - self.turnover_prev_violation)

                target_lambda = self.turnover_lambda_min + P + I + D
                target_lambda = np.clip(target_lambda, self.turnover_lambda_min, self.turnover_lambda_max)
            else:
                # Decay towards λ_min
                target_lambda = max(self.turnover_lambda_min, self.turnover_lambda * self.decay_rate)
                self.turnover_integral *= 0.995

            # Smoothing
            change = np.clip(target_lambda - self.turnover_lambda,
                             -self.max_lambda_change, self.max_lambda_change)
            self.turnover_lambda = self.turnover_lambda + change

        self.turnover_prev_violation = violation

        # Apply to environment
        if hasattr(real_env, 'set_churn_multiplier'):
            real_env.set_churn_multiplier(self.turnover_lambda)

        # Logging
        if self.num_timesteps % self.log_freq == 0:
            self.logger.record("plo_churn/turnover_violation", violation)
            self.logger.record("plo_churn/turnover_predicted", predicted_violation)
            self.logger.record("plo_churn/turnover_multiplier", self.turnover_lambda)
            self.logger.record("plo_churn/turnover_integral", self.turnover_integral)
            self.logger.record("plo_churn/turnover_slope", slope)
            self.logger.record("plo_churn/metric_turnover", metric_turnover)
            self.logger.record("plo_churn/max_turnover", max_turnover)
            self.logger.record("plo_churn/curriculum_active", float(curriculum_active))

        return True

    def _on_training_start(self) -> None:
        if self.verbose > 0:
            print(f"\n[PLO Churn] Configuration:")
            print(f"  Turnover threshold: {self.turnover_threshold:.2f}")
            print(f"  Lambda range: [{self.turnover_lambda_min}, {self.turnover_lambda_max}]")
            print(f"  PID gains: Kp={self.turnover_Kp}, Ki={self.turnover_Ki}, Kd={self.turnover_Kd}")
            print(f"  Prediction: {'polyfit (robust)' if self.use_prediction else 'disabled'}")


class PLOSmoothnessCallback(BaseCallback):
    """
    Predictive Lagrangian Optimization (PLO) for Smoothness-based Adaptive Penalties.

    This callback implements a PID controller that dynamically adjusts the
    smoothness penalty multiplier based on current jerk (position change acceleration).

    Differences from PLO Drawdown:
    - NO prediction (jerk is instantaneous)
    - Faster decay (0.99 vs 0.995)
    - More reactive (max_lambda_change = 0.1)

    VERSION: Production - Includes:
    1. Curriculum protection (smooth_coef < 0.001)
    2. Adaptive quantile (90%) or LogSumExp
    3. Off-by-one fix: reads jerk from buffer filled during step
    """

    def __init__(
        self,
        # Jerk Constraint
        jerk_threshold: float = 0.40,  # 40% of position range (tolerates normal adjustments)
        jerk_lambda_min: float = 1.0,
        jerk_lambda_max: float = 5.0,
        # PID Gains
        jerk_Kp: float = 3.0,
        jerk_Ki: float = 0.1,
        jerk_Kd: float = 0.5,
        # Stability
        integral_max: float = 2.0,
        decay_rate: float = 0.99,  # Faster decay than drawdown
        max_lambda_change: float = 0.1,  # More reactive
        # Risk measurement
        jerk_quantile: float = 0.9,
        # Logging
        log_freq: int = 100,
        verbose: int = 0,
    ):
        super().__init__(verbose)

        self.jerk_threshold = jerk_threshold
        self.jerk_lambda_min = jerk_lambda_min
        self.jerk_lambda_max = jerk_lambda_max
        self.jerk_Kp = jerk_Kp
        self.jerk_Ki = jerk_Ki
        self.jerk_Kd = jerk_Kd
        self.integral_max = integral_max
        self.decay_rate = decay_rate
        self.max_lambda_change = max_lambda_change
        self.jerk_quantile = jerk_quantile
        self.log_freq = log_freq

        # Controller state
        self.jerk_integral = 0.0
        self.jerk_prev_violation = 0.0
        self.jerk_lambda = 1.0

    def _on_step(self) -> bool:
        import torch
        real_env = get_underlying_batch_env(self.model.env)

        if not hasattr(real_env, 'current_jerks'):
            return True

        # ═══════════════════════════════════════════════════════════════════
        # CURRICULUM PROTECTION
        # Don't activate PLO if smooth_coef == 0
        # ═══════════════════════════════════════════════════════════════════
        if hasattr(real_env, '_current_smooth_coef'):
            if real_env._current_smooth_coef < 0.001:
                self.jerk_integral *= 0.9  # Fast decay
                return True

        # ═══════════════════════════════════════════════════════════════════
        # JERK MEASUREMENT
        # ═══════════════════════════════════════════════════════════════════
        current_jerks = real_env.current_jerks

        if real_env.num_envs >= 16:
            metric_jerk = torch.quantile(current_jerks, self.jerk_quantile).item()
        else:
            # LogSumExp for small batches
            temperature = 10.0
            metric_jerk = (torch.logsumexp(current_jerks * temperature, dim=0) / temperature).item()

        max_jerk = current_jerks.max().item()
        violation = max(0.0, metric_jerk - self.jerk_threshold)

        # ═══════════════════════════════════════════════════════════════════
        # PID CONTROLLER (no prediction - jerk is instantaneous)
        # ═══════════════════════════════════════════════════════════════════
        if violation > 0:
            P = self.jerk_Kp * violation
            self.jerk_integral += self.jerk_Ki * violation
            self.jerk_integral = np.clip(self.jerk_integral, 0, self.integral_max)
            I = self.jerk_integral
            D = self.jerk_Kd * (violation - self.jerk_prev_violation)

            target_lambda = self.jerk_lambda_min + P + I + D
            target_lambda = np.clip(target_lambda, self.jerk_lambda_min, self.jerk_lambda_max)
        else:
            # Decay towards λ_min
            target_lambda = max(self.jerk_lambda_min, self.jerk_lambda * self.decay_rate)
            self.jerk_integral *= 0.99

        # Smoothing
        change = np.clip(target_lambda - self.jerk_lambda,
                         -self.max_lambda_change, self.max_lambda_change)
        self.jerk_lambda = self.jerk_lambda + change

        self.jerk_prev_violation = violation

        # Apply to environment
        if hasattr(real_env, 'set_smooth_multiplier'):
            real_env.set_smooth_multiplier(self.jerk_lambda)

        # Logging
        if self.num_timesteps % self.log_freq == 0:
            self.logger.record("plo_smooth/jerk_violation", violation)
            self.logger.record("plo_smooth/jerk_multiplier", self.jerk_lambda)
            self.logger.record("plo_smooth/jerk_integral", self.jerk_integral)
            self.logger.record("plo_smooth/metric_jerk", metric_jerk)
            self.logger.record("plo_smooth/max_jerk", max_jerk)

        return True

    def _on_training_start(self) -> None:
        if self.verbose > 0:
            print(f"\n[PLO Smoothness] Configuration:")
            print(f"  Jerk threshold: {self.jerk_threshold:.2f}")
            print(f"  Lambda range: [{self.jerk_lambda_min}, {self.jerk_lambda_max}]")
            print(f"  PID gains: Kp={self.jerk_Kp}, Ki={self.jerk_Ki}, Kd={self.jerk_Kd}")
            print(f"  No prediction (jerk is instantaneous)")

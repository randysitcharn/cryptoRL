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
from stable_baselines3.common.callbacks import BaseCallback
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
                if "avg_rew_pnl" in metrics:
                    self.logger.record("rewards/pnl", metrics["avg_rew_pnl"])
                    self.logger.record("rewards/churn_penalty", metrics["avg_rew_churn"])
                    self.logger.record("rewards/smooth_penalty", metrics["avg_rew_smooth"])

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

            if self.episode_trades:
                self.logger.record("custom/trades_per_episode", self.episode_trades[-1])

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

            print(f"Step {self.num_timesteps:>7} | "
                  f"Reward: {mean_reward:>8.2f} | "
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

        if self.locals.get("infos"):
            info = self.locals["infos"][0]

            # Log reward components
            for key in ["rewards/log_return", "rewards/penalty_vol", "rewards/churn_penalty",
                        "rewards/smoothness_penalty", "rewards/position_delta", "rewards/total_raw",
                        "rewards/scaled"]:
                if key in info:
                    self.logger.record_mean(key, info[key])
                    if key == "rewards/log_return":
                        self.episode_log_returns.append(info[key])
                    elif key == "rewards/churn_penalty":
                        self.episode_churn_penalties.append(info[key])
                    elif key == "rewards/position_delta":
                        self.episode_position_deltas.append(info[key])

            # At episode end, log aggregated stats
            if "episode" in info:
                if self.episode_churn_penalties:
                    total_churn = sum(self.episode_churn_penalties)
                    total_log_ret = sum(self.episode_log_returns)
                    total_delta = sum(self.episode_position_deltas)

                    self.logger.record("churn/episode_total_penalty", total_churn)
                    self.logger.record("churn/episode_total_log_return", total_log_ret)
                    self.logger.record("churn/episode_total_position_delta", total_delta)

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

        # Update via shared memory (SubprocVecEnv) or direct access (DummyVecEnv)
        if self.shared_fee is not None:
            # Write to shared memory - subprocesses read this value
            self.shared_fee.value = self.current_fee
        if self.shared_smooth is not None:
            self.shared_smooth.value = self.current_smooth

        # Fallback for DummyVecEnv (no shared memory)
        if self.shared_fee is None:
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
    Three-Phase Curriculum Learning with Ramp & Plateau (2026-01-16).

    Strategy: Ramp penalties 0-60%, then PLATEAU at max to let agent consolidate.
    - Phase 1 (0-20%): Discovery - churn 0→0.10, smooth=0 (free exploration)
    - Phase 2 (20-60%): Discipline - churn 0.10→0.50, smooth 0→0.02 (ramp up)
    - Phase 3 (60-100%): Consolidation - churn=0.50, smooth=0.02 (PLATEAU)

    The plateau phase (40% of training) allows the agent to stabilize behavior
    at max penalty instead of chasing a moving target until the last step.

    Args:
        total_timesteps: Total training timesteps.
        shared_smooth: Shared memory for SubprocVecEnv.
        verbose: Verbosity level.
    """

    # Phase definitions: (end_progress, churn_range, smooth_range)
    # Ramp ends at 60%, then plateau for consolidation (40% of training)
    PHASES = [
        # Phase 1: Discovery (0% -> 10%) - exploration libre
        {'end_progress': 0.1, 'churn': (0.0, 0.10), 'smooth': (0.0, 0.0)},
        # Phase 2: Discipline (10% -> 30%) - ramp-up rapide vers max
        {'end_progress': 0.3, 'churn': (0.10, 0.50), 'smooth': (0.0, 0.02)},
        # Phase 3: Consolidation (30% -> 100%) - LONG PLATEAU at max
        {'end_progress': 1.0, 'churn': (0.50, 0.50), 'smooth': (0.02, 0.02)},
    ]

    def __init__(
        self,
        total_timesteps: int,
        shared_smooth: Optional["Synchronized"] = None,
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.shared_smooth = shared_smooth

        self.current_smooth = 0.0
        self.current_churn = 0.0
        self._phase = 1

    def _on_step(self) -> bool:
        progress = self.num_timesteps / self.total_timesteps

        # Find current phase and interpolate
        prev_end = 0.0
        for phase_idx, phase in enumerate(self.PHASES):
            if progress <= phase['end_progress']:
                self._phase = phase_idx + 1

                # Linear interpolation within phase
                phase_progress = (progress - prev_end) / (phase['end_progress'] - prev_end)
                phase_progress = max(0.0, min(1.0, phase_progress))

                churn_start, churn_end = phase['churn']
                smooth_start, smooth_end = phase['smooth']

                self.current_churn = churn_start + phase_progress * (churn_end - churn_start)
                self.current_smooth = smooth_start + phase_progress * (smooth_end - smooth_start)
                break
            prev_end = phase['end_progress']

        # Update via shared memory (SubprocVecEnv)
        if self.shared_smooth is not None:
            self.shared_smooth.value = self.current_smooth
        else:
            self._update_envs()

        # Log to TensorBoard
        self.logger.record("curriculum/smooth_coef", self.current_smooth)
        self.logger.record("curriculum/churn_coef", self.current_churn)
        self.logger.record("curriculum/phase", self._phase)

        return True

    def _update_envs(self):
        """Update penalties on all environments (unwrap for BatchCryptoEnv)."""
        # Unwrap pour atteindre BatchCryptoEnv sous les wrappers SB3
        real_env = get_underlying_batch_env(self.model.env)

        # Appel direct (contourne le Wrapper Hell de SB3)
        if hasattr(real_env, 'set_smoothness_penalty'):
            real_env.set_smoothness_penalty(self.current_smooth)
            real_env.set_churn_penalty(self.current_churn)
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
            print(f"\n[3-Phase Curriculum] Configuration (recalibré):")
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

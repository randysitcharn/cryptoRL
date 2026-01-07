# -*- coding: utf-8 -*-
"""
curriculum_callback.py - Curriculum Learning Callback for SB3.

Gradually increases fees and smoothness penalty during training.
This ensures the agent learns to make profits first, then learns to be efficient.
"""

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv


class CurriculumFeesCallback(BaseCallback):
    """
    Curriculum Learning: Start with 0 fees/penalties, linearly increase to targets.

    This callback implements a curriculum learning strategy where transaction costs
    and smoothness penalties start at 0 and gradually increase to their target values.
    This allows the agent to first learn profitable trading patterns, then gradually
    learn to trade efficiently with realistic costs.

    Args:
        target_fee_rate: Target commission fee (e.g., 0.0006 = 0.06%).
        target_smooth_coef: Target smoothness penalty coefficient.
        warmup_steps: Number of steps to reach target values (linear interpolation).
        verbose: Verbosity level (0 = silent, 1 = info).

    Example:
        >>> callback = CurriculumFeesCallback(
        ...     target_fee_rate=0.0006,
        ...     target_smooth_coef=1.0,
        ...     warmup_steps=50_000
        ... )
        >>> model.learn(total_timesteps=150_000, callback=callback)
    """

    def __init__(
        self,
        target_fee_rate: float = 0.0006,
        target_smooth_coef: float = 1.0,
        warmup_steps: int = 50_000,
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.target_fee_rate = target_fee_rate
        self.target_smooth_coef = target_smooth_coef
        self.warmup_steps = warmup_steps

        # Current values (start at 0)
        self.current_fee = 0.0
        self.current_smooth = 0.0

        # Track if warmup is complete (for logging)
        self._warmup_complete = False

    def _on_step(self) -> bool:
        """
        Called at every step. Update penalty values and log to TensorBoard.

        Returns:
            True to continue training, False to stop.
        """
        # Calculate progress ratio (capped at 1.0)
        progress = min(1.0, self.num_timesteps / self.warmup_steps)

        # Linear interpolation from 0 to target
        self.current_fee = progress * self.target_fee_rate
        self.current_smooth = progress * self.target_smooth_coef

        # Update all environments
        self._update_envs()

        # Log to TensorBoard
        self.logger.record("curriculum/fee_rate", self.current_fee)
        self.logger.record("curriculum/smooth_coef", self.current_smooth)
        self.logger.record("curriculum/progress", progress)

        # Log milestone when warmup completes
        if progress >= 1.0 and not self._warmup_complete:
            self._warmup_complete = True
            if self.verbose > 0:
                print(f"\n[Curriculum] Warmup complete at step {self.num_timesteps}")
                print(f"  fee_rate: {self.current_fee:.6f}")
                print(f"  smooth_coef: {self.current_smooth:.2f}")

        return True

    def _update_envs(self):
        """
        Update penalties on all environments.

        Handles VecEnv (DummyVecEnv/SubprocVecEnv) by iterating through
        all sub-environments and unwrapping Monitor wrappers.
        """
        env = self.model.env

        if isinstance(env, VecEnv):
            # DummyVecEnv: iterate through envs
            for i in range(env.num_envs):
                # Unwrap Monitor/other wrappers to get CryptoTradingEnv
                base_env = env.envs[i]
                while hasattr(base_env, 'env'):
                    base_env = base_env.env

                # Call update_penalties if available
                if hasattr(base_env, 'update_penalties'):
                    base_env.update_penalties(
                        fee_rate=self.current_fee,
                        smooth_coef=self.current_smooth
                    )
        else:
            # Single env (non-vectorized)
            base_env = env
            while hasattr(base_env, 'env'):
                base_env = base_env.env

            if hasattr(base_env, 'update_penalties'):
                base_env.update_penalties(
                    fee_rate=self.current_fee,
                    smooth_coef=self.current_smooth
                )

    def _on_training_start(self) -> None:
        """Called when training starts. Log initial curriculum state."""
        if self.verbose > 0:
            print(f"\n[Curriculum] Starting curriculum learning:")
            print(f"  target_fee_rate: {self.target_fee_rate:.6f}")
            print(f"  target_smooth_coef: {self.target_smooth_coef:.2f}")
            print(f"  warmup_steps: {self.warmup_steps:,}")

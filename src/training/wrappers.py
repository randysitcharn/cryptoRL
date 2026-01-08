# -*- coding: utf-8 -*-
"""
wrappers.py - Gymnasium Wrappers for Trading Environments.

Contains action wrappers to filter and modify agent actions before
they are passed to the underlying environment.
"""

import warnings
import gymnasium as gym
import numpy as np
from collections import deque
from typing import Optional


class DeadZoneWrapper(gym.ActionWrapper):
    """
    Filter out small, noisy actions from the Actor.

    If the absolute value of the action is below a threshold,
    force the action to 0.0 (Hold). Otherwise, pass through.

    This helps reduce micro-trading caused by actor noise and
    encourages the agent to only trade when confident.

    Args:
        env: The environment to wrap.
        threshold: Actions with |action| < threshold are set to 0.

    Example:
        >>> env = CryptoTradingEnv(...)
        >>> env = DeadZoneWrapper(env, threshold=0.1)
        >>> # Actions in [-0.1, 0.1] become 0.0
    """

    def __init__(self, env: gym.Env, threshold: float = 0.1):
        super().__init__(env)
        self.threshold = threshold
        self._dead_zone_triggers = 0
        self._total_actions = 0

    def action(self, action: np.ndarray) -> np.ndarray:
        """
        Apply dead zone filter to action.

        Args:
            action: Raw action from the agent.

        Returns:
            Filtered action (0.0 if in dead zone).
        """
        filtered = action.copy()
        mask = np.abs(filtered) < self.threshold
        filtered[mask] = 0.0

        # Track statistics
        self._total_actions += 1
        if mask.any():
            self._dead_zone_triggers += 1

        return filtered

    def get_dead_zone_stats(self) -> dict:
        """
        Get statistics about dead zone filtering.

        Returns:
            Dict with trigger count and ratio.
        """
        ratio = self._dead_zone_triggers / max(self._total_actions, 1)
        return {
            "dead_zone_triggers": self._dead_zone_triggers,
            "total_actions": self._total_actions,
            "dead_zone_ratio": ratio,
        }

    def reset_stats(self):
        """Reset dead zone statistics."""
        self._dead_zone_triggers = 0
        self._total_actions = 0


class ActionSmoothingWrapper(gym.ActionWrapper):
    """
    Smooth actions using exponential moving average.

    Reduces abrupt position changes by blending current action
    with previous actions. Helps reduce churn.

    Args:
        env: The environment to wrap.
        alpha: Smoothing factor in [0, 1]. Higher = less smoothing.
               0.0 = ignore new action, 1.0 = no smoothing.
    """

    def __init__(self, env: gym.Env, alpha: float = 0.5):
        super().__init__(env)
        self.alpha = np.clip(alpha, 0.0, 1.0)
        self._prev_action: Optional[np.ndarray] = None

    def action(self, action: np.ndarray) -> np.ndarray:
        """
        Apply exponential smoothing to action.

        Args:
            action: Raw action from the agent.

        Returns:
            Smoothed action.
        """
        if self._prev_action is None:
            self._prev_action = action.copy()
            return action

        # EMA: smoothed = alpha * new + (1 - alpha) * old
        smoothed = self.alpha * action + (1 - self.alpha) * self._prev_action
        self._prev_action = smoothed.copy()

        return smoothed

    def reset(self, **kwargs):
        """Reset wrapper state on environment reset."""
        self._prev_action = None
        return self.env.reset(**kwargs)


class RiskManagementWrapper(gym.Wrapper):
    """
    Circuit Breaker wrapper for risk management.

    Monitors rolling volatility and drawdown. When thresholds are breached,
    forces HOLD actions for a cooldown period to protect against regime shifts.

    Args:
        env: The environment to wrap.
        vol_window: Rolling window size for volatility calculation (in steps).
        vol_threshold: Trigger if rolling vol > vol_threshold * baseline_vol.
        max_drawdown: Trigger if drawdown exceeds this fraction (e.g., 0.10 = 10%).
        cooldown_steps: Number of steps to force HOLD after trigger.
        augment_obs: If True, add panic_mode flag to observation space.
        baseline_vol: Pre-computed baseline volatility from TRAIN data.
                      If None, defaults to 0.01 with a warning.
    """

    def __init__(
        self,
        env: gym.Env,
        vol_window: int = 24,
        vol_threshold: float = 3.0,
        max_drawdown: float = 0.10,
        cooldown_steps: int = 12,
        augment_obs: bool = False,
        baseline_vol: Optional[float] = None,
    ):
        super().__init__(env)
        self.vol_window = vol_window
        self.vol_threshold = vol_threshold
        self.max_drawdown = max_drawdown
        self.cooldown_steps = cooldown_steps
        self.augment_obs = augment_obs

        # State tracking
        self.nav_history: deque = deque(maxlen=vol_window + 1)
        self.peak_nav: float = 10000.0
        self.current_drawdown: float = 0.0
        self.rolling_vol: float = 0.0
        self.cooldown_remaining: int = 0
        self.panic_triggered: bool = False

        # Baseline volatility (should be computed from TRAIN data)
        if baseline_vol is not None:
            self.baseline_vol = baseline_vol
        else:
            self.baseline_vol = 0.01  # Conservative fallback
            warnings.warn(
                "[RiskManagementWrapper] baseline_vol not provided. "
                "Using default 0.01. For accurate thresholds, compute from TRAIN data.",
                UserWarning
            )

        # Statistics
        self.circuit_breaker_count: int = 0
        self.total_steps: int = 0

    @property
    def observation_space(self):
        """Optionally augment observation space with panic_mode flag."""
        if self.augment_obs:
            base_space = self.env.observation_space
            low = np.concatenate(
                [base_space.low, np.zeros((base_space.shape[0], 1))], axis=1
            )
            high = np.concatenate(
                [base_space.high, np.ones((base_space.shape[0], 1))], axis=1
            )
            return gym.spaces.Box(low=low, high=high, dtype=np.float32)
        return self.env.observation_space

    def reset(self, **kwargs):
        """Reset wrapper state on environment reset."""
        self.nav_history.clear()
        self.peak_nav = 10000.0
        self.current_drawdown = 0.0
        self.rolling_vol = 0.0
        self.cooldown_remaining = 0
        self.panic_triggered = False

        obs, info = self.env.reset(**kwargs)

        if self.augment_obs:
            obs = self._augment_observation(obs)

        return obs, info

    def step(self, action):
        """Execute step with circuit breaker logic."""
        self.total_steps += 1

        # 1. Check if in cooldown - force HOLD
        if self.cooldown_remaining > 0:
            action = np.array([0.0])
            self.cooldown_remaining -= 1

        # 2. Execute action
        obs, reward, terminated, truncated, info = self.env.step(action)

        # 3. Update tracking metrics
        self._update_metrics(info)

        # 4. Check circuit breaker triggers (only if not in cooldown)
        if self.cooldown_remaining == 0 and self._check_circuit_breaker():
            self.cooldown_remaining = self.cooldown_steps
            self.panic_triggered = True
            self.circuit_breaker_count += 1
            info['circuit_breaker'] = True

        # 5. Augment observation (optional)
        if self.augment_obs:
            obs = self._augment_observation(obs)

        # 6. Add risk metrics to info
        info['risk/rolling_vol'] = self.rolling_vol
        info['risk/current_dd'] = self.current_drawdown
        info['risk/panic_mode'] = int(self.cooldown_remaining > 0)

        return obs, reward, terminated, truncated, info

    def _update_metrics(self, info: dict):
        """Track NAV and calculate rolling volatility + drawdown."""
        nav = info.get('nav', 10000.0)
        self.nav_history.append(nav)

        # Peak NAV for drawdown calculation
        self.peak_nav = max(self.peak_nav, nav)
        self.current_drawdown = (self.peak_nav - nav) / self.peak_nav

        # Rolling volatility (std of log returns)
        if len(self.nav_history) > 1:
            navs = np.array(self.nav_history)
            returns = np.diff(np.log(navs))
            self.rolling_vol = np.std(returns) if len(returns) > 1 else 0.0

    def _check_circuit_breaker(self) -> bool:
        """Check if circuit breaker should trigger."""
        # Condition 1: Volatility spike
        vol_trigger = self.rolling_vol > (self.baseline_vol * self.vol_threshold)

        # Condition 2: Drawdown limit breached
        dd_trigger = self.current_drawdown > self.max_drawdown

        return vol_trigger or dd_trigger

    def _augment_observation(self, obs: np.ndarray) -> np.ndarray:
        """Add panic_mode flag to observation."""
        panic_col = np.full((obs.shape[0], 1), float(self.cooldown_remaining > 0))
        return np.concatenate([obs, panic_col], axis=1)

    def calibrate_baseline(self, n_steps: int = 1000):
        """
        DEPRECATED: This method causes data leakage when used on test environments.

        Instead, compute baseline_vol from TRAIN data and pass to __init__:
            baseline_vol = train_df['BTC_Close'].pct_change().std()
            env = RiskManagementWrapper(env, baseline_vol=baseline_vol)

        This method is kept for backward compatibility but will be removed.
        """
        warnings.warn(
            "[RiskManagementWrapper] calibrate_baseline() is DEPRECATED and causes data leakage. "
            "Compute baseline_vol from TRAIN data and pass to __init__ instead.",
            DeprecationWarning,
            stacklevel=2
        )

        # Legacy behavior (kept for backward compatibility)
        self.env.reset()
        navs = []

        for _ in range(n_steps):
            action = self.env.action_space.sample()
            _, _, terminated, truncated, info = self.env.step(action)
            navs.append(info.get('nav', 10000.0))
            if terminated or truncated:
                self.env.reset()

        if len(navs) > 1:
            returns = np.diff(np.log(navs))
            self.baseline_vol = np.std(returns)
        else:
            self.baseline_vol = 0.01  # Fallback

        print(f"[RiskMgmt] DEPRECATED calibration. baseline_vol: {self.baseline_vol:.6f}")

        # Reset env state after calibration
        self.env.reset()

    def get_stats(self) -> dict:
        """Get circuit breaker statistics."""
        return {
            "circuit_breaker_count": self.circuit_breaker_count,
            "total_steps": self.total_steps,
            "trigger_rate": self.circuit_breaker_count / max(self.total_steps, 1),
            "baseline_vol": self.baseline_vol,
        }

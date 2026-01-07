# -*- coding: utf-8 -*-
"""
wrappers.py - Gymnasium Wrappers for Trading Environments.

Contains action wrappers to filter and modify agent actions before
they are passed to the underlying environment.
"""

import gymnasium as gym
import numpy as np
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

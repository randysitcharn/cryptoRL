# -*- coding: utf-8 -*-
"""
trading_env.py - Gymnasium environment for crypto trading.

Implements CryptoTradingEnv compatible with Stable-Baselines3:
- Continuous action [-1, 1] representing position (short/flat/long)
- Normalized technical features observation
- Placeholder reward for future iteration
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class CryptoTradingEnv(gym.Env):
    """Crypto trading environment for RL."""

    metadata = {'render_modes': ['human']}

    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = 10000.0,
        commission: float = 0.001
    ):
        """
        Initialize the trading environment.

        Args:
            df (pd.DataFrame): OHLCV data + normalized features.
            initial_balance (float): Initial capital in USD.
            commission (float): Transaction fee (0.001 = 0.1%).
        """
        super().__init__()

        # Convert DataFrame to float32 arrays
        self.data = df.values.astype(np.float32)
        self.n_steps = len(self.data)

        # Store close prices for P&L calculation
        self.prices = df['close'].values.astype(np.float32)

        # Features for observation (all columns)
        self.n_features = self.data.shape[1]

        # Trading parameters
        self.initial_balance = initial_balance
        self.commission = commission

        # Action and observation spaces
        # Action: continuous position [-1, 1] (short/flat/long)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )

        # Observation: all features from DataFrame
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_features,),
            dtype=np.float32
        )

        # Initial state
        self._reset_state()

    def _reset_state(self):
        """Reset internal environment state."""
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0.0  # Current position [-1, 1]
        self.entry_price = 0.0

    def reset(self, seed=None, options=None):
        """
        Reset environment for a new episode.

        Args:
            seed (int, optional): Seed for reproducibility.
            options (dict, optional): Additional options.

        Returns:
            tuple: (observation, info)
        """
        super().reset(seed=seed)
        self._reset_state()

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action):
        """
        Execute action and advance one step.

        Args:
            action (np.ndarray): Target position [-1, 1].

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Extract scalar action
        target_position = float(action[0])
        target_position = np.clip(target_position, -1.0, 1.0)

        # Current price
        current_price = self.prices[self.current_step]

        # Calculate position change
        position_change = abs(target_position - self.position)

        # Transaction cost (proportional to position change)
        transaction_cost = position_change * self.commission * self.balance

        # Update position
        self.position = target_position

        # Advance one step
        self.current_step += 1

        # Check if episode ended
        terminated = self.current_step >= self.n_steps - 1
        truncated = False

        # Calculate reward (placeholder - simple P&L)
        if not terminated:
            next_price = self.prices[self.current_step]
            price_return = (next_price - current_price) / current_price
            reward = self.position * price_return * self.balance - transaction_cost
        else:
            reward = 0.0

        observation = self._get_observation()
        info = self._get_info()

        return observation, float(reward), terminated, truncated, info

    def _get_observation(self):
        """Return current observation."""
        return self.data[self.current_step].copy()

    def _get_info(self):
        """Return additional information."""
        return {
            'step': self.current_step,
            'balance': self.balance,
            'position': self.position,
            'price': self.prices[self.current_step]
        }

    def render(self):
        """Display current state (human mode)."""
        info = self._get_info()
        print(f"Step: {info['step']} | "
              f"Price: ${info['price']:.2f} | "
              f"Position: {info['position']:.2f} | "
              f"Balance: ${info['balance']:.2f}")
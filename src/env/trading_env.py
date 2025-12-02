# -*- coding: utf-8 -*-
"""
trading_env.py - Gymnasium environment for crypto trading.

Implements CryptoTradingEnv compatible with Stable-Baselines3:
- Continuous action [-1, 1] representing target position (% of portfolio)
- Proper financial execution with slippage and commission
- NAV-based portfolio tracking
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class CryptoTradingEnv(gym.Env):
    """Crypto trading environment for RL with realistic financial logic."""

    metadata = {'render_modes': ['human']}

    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = 10000.0,
        commission: float = 0.001,
        slippage: float = 0.0001
    ):
        """
        Initialize the trading environment.

        Args:
            df (pd.DataFrame): OHLCV data + normalized features.
            initial_balance (float): Initial capital in USD.
            commission (float): Transaction fee (0.001 = 0.1%).
            slippage (float): Slippage cost (0.0001 = 0.01%).
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
        self.slippage = slippage

        # Action and observation spaces
        # Action: target position [-1, 1] as % of portfolio
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
        self.cash = self.initial_balance
        self.asset_holdings = 0.0  # Quantity of crypto held

    def _get_current_price(self) -> float:
        """Return the current close price."""
        return float(self.prices[self.current_step])

    def _get_portfolio_value(self) -> float:
        """Calculate current Net Asset Value (NAV)."""
        current_price = self._get_current_price()
        return self.cash + (self.asset_holdings * current_price)

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
            action (np.ndarray): Target position [-1, 1] as % of portfolio.
                -1 = 100% short (not implemented, treated as 0)
                 0 = 100% cash
                +1 = 100% in asset

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # 1. Get current state
        current_price = self._get_current_price()
        current_portfolio_value = self._get_portfolio_value()

        # 2. Parse action (clip to valid range)
        target_position = float(np.clip(action[0], -1.0, 1.0))

        # For now, treat negative positions as 0 (no shorting)
        target_position = max(0.0, target_position)

        # 3. Calculate target vs current asset value
        target_asset_value = current_portfolio_value * target_position
        current_asset_value = self.asset_holdings * current_price
        trade_amount_usd = target_asset_value - current_asset_value

        # 4. Execute trade with fees
        if abs(trade_amount_usd) > 0.01:  # Minimum trade threshold
            # Calculate transaction cost
            transaction_cost = abs(trade_amount_usd) * (self.commission + self.slippage)

            # Deduct fees from cash
            self.cash -= transaction_cost

            # Execute the trade
            trade_qty = trade_amount_usd / current_price
            self.asset_holdings += trade_qty
            self.cash -= trade_amount_usd

        # 5. Calculate new portfolio value
        new_portfolio_value = self._get_portfolio_value()

        # 6. Check for bankruptcy
        terminated = False
        truncated = False

        if new_portfolio_value <= 0:
            terminated = True
            reward = -100.0  # Death penalty
        else:
            # 7. Advance to next step
            self.current_step += 1

            # Check if episode ended (reached end of data)
            if self.current_step >= self.n_steps - 1:
                terminated = True

            # 8. Calculate reward (simple return)
            if current_portfolio_value > 0:
                reward = (new_portfolio_value - current_portfolio_value) / current_portfolio_value
            else:
                reward = 0.0

        # 9. Get observation and info
        observation = self._get_observation()
        info = self._get_info(action[0])

        return observation, float(reward), terminated, truncated, info

    def _get_observation(self):
        """Return current observation."""
        return self.data[self.current_step].copy()

    def _get_info(self, action=0.0):
        """Return additional information."""
        return {
            'step': self.current_step,
            'portfolio_value': self._get_portfolio_value(),
            'cash': self.cash,
            'asset_holdings': self.asset_holdings,
            'price': self._get_current_price(),
            'action': float(action)
        }

    def render(self):
        """Display current state (human mode)."""
        info = self._get_info()
        print(f"Step: {info['step']} | "
              f"Price: ${info['price']:.2f} | "
              f"Holdings: {info['asset_holdings']:.6f} | "
              f"Cash: ${info['cash']:.2f} | "
              f"NAV: ${info['portfolio_value']:.2f}")

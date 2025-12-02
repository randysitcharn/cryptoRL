# -*- coding: utf-8 -*-
"""
trading_env.py - Gymnasium environment for crypto trading.

Implements CryptoTradingEnv compatible with Stable-Baselines3:
- Continuous action [-1, 1] representing target position (% of portfolio)
- Proper financial execution with slippage and commission
- NAV-based portfolio tracking
- Risk-adjusted reward (Sortino-proxy) with downside deviation
"""

from collections import deque

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
        slippage: float = 0.0001,
        reward_scaling: float = 10.0
    ):
        """
        Initialize the trading environment.

        Args:
            df (pd.DataFrame): OHLCV data + normalized features.
            initial_balance (float): Initial capital in USD.
            commission (float): Transaction fee (0.001 = 0.1%).
            slippage (float): Slippage cost (0.0001 = 0.01%).
            reward_scaling (float): Multiplier for risk-adjusted reward.
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
        self.reward_scaling = reward_scaling

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

        # Risk-adjusted reward tracking
        self.returns_window: deque = deque(maxlen=50)  # Rolling window for Sortino
        self.max_drawdown = 0.0
        self.peak_nav = self.initial_balance

    def _get_current_price(self) -> float:
        """Return the current close price."""
        return float(self.prices[self.current_step])

    def _get_portfolio_value(self) -> float:
        """Calculate current Net Asset Value (NAV)."""
        current_price = self._get_current_price()
        return self.cash + (self.asset_holdings * current_price)

    def _calculate_reward(self, current_nav: float, old_nav: float) -> tuple:
        """
        Calculate risk-adjusted reward using Sortino-proxy.

        Uses log returns and downside deviation to penalize volatility,
        especially negative volatility (Sortino ratio approach).

        Args:
            current_nav (float): Current portfolio value.
            old_nav (float): Previous portfolio value.

        Returns:
            tuple: (reward, log_return) where reward is scaled risk-adjusted return.
        """
        # 1. Log return (more stable than simple return for small changes)
        if old_nav > 0 and current_nav > 0:
            step_log_return = float(np.log(current_nav / old_nav))
        else:
            step_log_return = 0.0

        # 2. Store in rolling window
        self.returns_window.append(step_log_return)

        # 3. Update peak and max drawdown
        if current_nav > self.peak_nav:
            self.peak_nav = current_nav
        current_drawdown = (self.peak_nav - current_nav) / self.peak_nav
        self.max_drawdown = max(self.max_drawdown, current_drawdown)

        # 4. Downside deviation (Sortino denominator)
        negative_returns = [r for r in self.returns_window if r < 0]
        if len(negative_returns) >= 2:
            downside_std = float(np.std(negative_returns))
        else:
            downside_std = 0.0

        # 5. Risk-adjusted return
        if downside_std > 1e-8:
            risk_adjusted = step_log_return / downside_std
        else:
            # No downside deviation yet, use raw log return
            risk_adjusted = step_log_return

        return risk_adjusted * self.reward_scaling, step_log_return

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

        # 5. Check for immediate bankruptcy (after trade, before price change)
        terminated = False
        truncated = False
        log_return = 0.0

        immediate_nav = self._get_portfolio_value()
        if immediate_nav <= 0:
            terminated = True
            reward = -100.0  # Death penalty
        else:
            # 6. Advance to next step (price changes)
            self.current_step += 1

            # Check if episode ended (reached end of data)
            if self.current_step >= self.n_steps - 1:
                terminated = True

            # 7. Calculate new portfolio value at NEW price
            new_portfolio_value = self._get_portfolio_value()

            # 8. Calculate risk-adjusted reward (Sortino-proxy)
            reward, log_return = self._calculate_reward(
                new_portfolio_value, current_portfolio_value
            )

        # 9. Get observation and info
        observation = self._get_observation()
        info = self._get_info(action[0], log_return)

        return observation, float(reward), terminated, truncated, info

    def _get_observation(self):
        """Return current observation."""
        return self.data[self.current_step].copy()

    def _get_info(self, action=0.0, log_return=0.0):
        """Return additional information including monitoring metrics."""
        return {
            'step': self.current_step,
            'portfolio_value': self._get_portfolio_value(),
            'cash': self.cash,
            'asset_holdings': self.asset_holdings,
            'price': self._get_current_price(),
            'action': float(action),
            'log_return': log_return,
            'max_drawdown': self.max_drawdown,
            'peak_nav': self.peak_nav,
        }

    def render(self):
        """Display current state (human mode)."""
        info = self._get_info()
        print(f"Step: {info['step']} | "
              f"Price: ${info['price']:.2f} | "
              f"Holdings: {info['asset_holdings']:.6f} | "
              f"Cash: ${info['cash']:.2f} | "
              f"NAV: ${info['portfolio_value']:.2f}")

# -*- coding: utf-8 -*-
"""
env.py - Trading Environment for RL with Foundation Model.

Gymnasium environment optimized for TQC training with pre-trained encoder.
Uses Hybrid Log-Sortino reward with action discretization to reduce churn.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, TYPE_CHECKING

if TYPE_CHECKING:
    from multiprocessing.sharedctypes import Synchronized

from src.config import EXCLUDE_COLS


class CryptoTradingEnv(gym.Env):
    """
    Crypto Trading Environment for RL with Foundation Model.

    Features:
    - Continuous action space [-1, 1] for position sizing (discretized to 21 levels)
    - Hybrid Log-Sortino reward (log return + downside penalty + upside bonus)
    - Realistic transaction costs (0.06% commission + 0.01% slippage)
    - Random or sequential episode starts
    - Compatible with FoundationFeatureExtractor

    Action Space:
        Box([-1, 1]): Target position as fraction of portfolio
        -1 = 100% short (or 0% long if no shorting)
         0 = Neutral (50% exposure)
        +1 = 100% long

    Observation Space:
        Box([window_size, n_features]): Rolling window of features
    """

    metadata = {'render_modes': ['human']}

    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        parquet_path: Optional[str] = None,
        price_column: str = "close",
        initial_balance: float = 10_000.0,
        commission: float = 0.0006,  # 0.06% per trade
        slippage: float = 0.0001,    # 0.01% slippage
        window_size: int = 64,
        reward_scaling: float = 30.0,  # Amplify signal for tanh (optimal range)
        random_start: bool = True,
        episode_length: Optional[int] = None,
        start_idx: Optional[int] = None,
        end_idx: Optional[int] = None,
        downside_coef: float = 10.0,  # Sortino downside penalty coefficient
        upside_coef: float = 0.0,  # Symmetric upside bonus coefficient
        action_discretization: float = 0.1,  # Discretize actions (0.1 = 21 positions)
        churn_coef: float = 0.0,  # Cognitive tax: amplify trading cost perception (0.1 = 10x)
        smooth_coef: float = 0.0,  # Smoothness penalty: quadratic penalty on position changes
        # Volatility Scaling (Target Volatility)
        target_volatility: float = 0.01,  # 1% target vol (reduces position in high vol)
        vol_window: int = 24,  # Rolling window for vol calculation (24h)
        max_leverage: float = 5.0,  # Max scaling factor (caps position in low vol)
        # Shared memory for curriculum learning (SubprocVecEnv compatible)
        shared_fee: Optional["Synchronized"] = None,
        shared_smooth: Optional["Synchronized"] = None,
    ):
        """
        Initialize the trading environment.

        Args:
            df: DataFrame with OHLCV + features (alternative to parquet_path).
            parquet_path: Path to processed_data.parquet file.
            price_column: Column name for price (for P&L calculation).
                         Use 'close' for legacy CSV data, 'BTC_Close' for parquet.
            initial_balance: Starting capital in USD.
            commission: Transaction fee (0.0006 = 0.06%).
            slippage: Slippage cost (0.0001 = 0.01%).
            window_size: Lookback window size for observations.
            reward_scaling: Multiplier for tanh scaling (default: 100.0).
            random_start: If True, start episodes at random positions.
            episode_length: Max steps per episode (None = full dataset).
            start_idx: Start index for data slice (for train/val split).
            end_idx: End index for data slice (for train/val split).
            downside_coef: Sortino downside penalty coefficient (default: 10.0).
            upside_coef: Symmetric upside bonus coefficient (default: 0.0).
            action_discretization: Discretize actions to reduce churn (0.1 = 21 positions, 0 = disabled).
        """
        super().__init__()

        # Reward parameters
        self.reward_scaling = reward_scaling
        self.downside_coef = downside_coef
        self.upside_coef = upside_coef
        self.action_discretization = action_discretization
        self.churn_coef = churn_coef
        self.smooth_coef = smooth_coef

        # Shared memory for curriculum learning (SubprocVecEnv compatible)
        self.shared_fee = shared_fee
        self.shared_smooth = shared_smooth

        # Volatility Scaling parameters
        self.target_volatility = target_volatility
        self.vol_window = vol_window
        self.max_leverage = max_leverage

        # Load data: either from DataFrame or parquet file
        if df is not None:
            # Use provided DataFrame directly (legacy compatibility)
            pass
        elif parquet_path is not None:
            df = pd.read_parquet(parquet_path)
        else:
            # Default fallback
            df = pd.read_parquet("data/processed_data.parquet")

        # Apply data slice if specified
        if start_idx is not None or end_idx is not None:
            start = start_idx or 0
            end = end_idx or len(df)
            df = df.iloc[start:end].reset_index(drop=True)

        # Extract price column for P&L calculation
        if price_column not in df.columns:
            raise ValueError(f"Price column '{price_column}' not found in data")
        self.prices = df[price_column].values.astype(np.float32)

        # Extract features (exclude raw OHLCV)
        feature_cols = [
            col for col in df.columns
            if col not in EXCLUDE_COLS
            and df[col].dtype in ['float64', 'float32', 'int64']
        ]
        self.feature_names = feature_cols
        self.data = df[feature_cols].values.astype(np.float32)

        # Handle NaN values
        if np.isnan(self.data).any():
            print("[CryptoTradingEnv] Warning: NaN values found, replacing with 0")
            self.data = np.nan_to_num(self.data, nan=0.0)

        # Dimensions
        self.n_steps = len(df)
        self.n_features = len(feature_cols)
        self.window_size = window_size

        # Trading parameters
        self.initial_balance = initial_balance
        self.commission = commission
        self.slippage = slippage
        self.reward_scaling = reward_scaling
        self.random_start = random_start
        self.episode_length = episode_length

        # Action space: continuous [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )

        # Observation space: (window_size, n_features)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size, self.n_features),
            dtype=np.float32
        )

        # Initialize state
        self._reset_state()

        print(f"[CryptoTradingEnv] Initialized: {self.n_steps} steps, "
              f"{self.n_features} features, window={window_size}")

    def _reset_state(self) -> None:
        """Reset internal environment state."""
        # Episode boundaries
        min_start = self.window_size
        max_start = self.n_steps - (self.episode_length or self.n_steps) - 1

        if self.random_start and max_start > min_start:
            self.start_step = np.random.randint(min_start, max_start)
        else:
            self.start_step = min_start

        self.current_step = self.start_step
        self.episode_end = (
            self.start_step + self.episode_length
            if self.episode_length
            else self.n_steps - 1
        )

        # Portfolio state
        self.cash = self.initial_balance
        self.position = 0.0  # Number of units held
        self.current_position_pct = 0.0  # Current position as % [-1, 1]
        self._prev_position_pct = 0.0  # For churn penalty calculation

        # Reward state
        self._prev_valuation = self.initial_balance

        # Tracking
        self.total_trades = 0
        self.total_commission = 0.0
        self.returns_history: List[float] = []

        # Reward metrics for observability
        self.reward_metrics = {
            "rewards/log_return": 0.0,
            "rewards/penalty_vol": 0.0,
            "rewards/bonus_upside": 0.0,
            "rewards/total_raw": 0.0,
        }

        # Volatility scaling state
        self.returns_for_vol: List[float] = []  # Separate buffer for vol calculation
        self.current_volatility: float = self.target_volatility  # Init to target
        self.vol_scalar: float = 1.0
        # EMA variance for O(1) volatility calculation (instead of O(window) np.std)
        # Init to None - will be set from first actual return
        self._ema_var: Optional[float] = None

    def _get_price(self, step: Optional[int] = None) -> float:
        """Get price at given step (default: current step)."""
        idx = step if step is not None else self.current_step
        return float(self.prices[idx])

    def _get_nav(self) -> float:
        """Calculate Net Asset Value."""
        price = self._get_price()
        return self.cash + self.position * price

    def _get_commission(self) -> float:
        """Get current commission (from shared memory or local).

        For curriculum learning with SubprocVecEnv, reads from shared memory.
        Falls back to local value if shared memory is not available.
        """
        if self.shared_fee is not None:
            return self.shared_fee.value
        return self.commission

    def _get_smooth_coef(self) -> float:
        """Get current smooth_coef (from shared memory or local).

        For curriculum learning with SubprocVecEnv, reads from shared memory.
        Falls back to local value if shared memory is not available.
        """
        if self.shared_smooth is not None:
            return self.shared_smooth.value
        return self.smooth_coef

    def _calculate_volatility(self) -> float:
        """
        Calculate rolling volatility using EMA variance (O(1) per step).

        Much faster than np.std() which is O(vol_window) per step.
        EMA formula: var_t = alpha * r_t^2 + (1-alpha) * var_{t-1}
        """
        if len(self.returns_for_vol) < 2:
            return self.target_volatility  # Default when not enough data

        # EMA alpha based on vol_window (equivalent span)
        alpha = 2.0 / (self.vol_window + 1)

        # Get latest return
        ret = self.returns_for_vol[-1]

        # Update EMA variance: var_t = alpha * r^2 + (1-alpha) * var_{t-1}
        if self._ema_var is None:
            self._ema_var = ret ** 2  # Initialize from first return
        else:
            self._ema_var = alpha * (ret ** 2) + (1.0 - alpha) * self._ema_var

        # Return standard deviation (sqrt of variance)
        return max(np.sqrt(self._ema_var), 1e-6)

    def _calculate_reward(self) -> float:
        """
        Hybrid Log-Sortino Reward Function.

        Combines:
        1. Log return for optimal growth (Kelly criterion alignment)
        2. Sortino-style downside penalty (asymmetric risk)
        3. Symmetric upside bonus (reward positive returns)

        Returns:
            Reward value in [-1.0, +1.0] (tanh scaled).
        """
        # 1. PnL proportionnel
        current_value = self._get_nav()
        step_return = (current_value - self._prev_valuation) / self._prev_valuation

        # 2. Log Return (croissance optimale)
        safe_return = max(step_return, -0.99)
        reward_log_return = np.log(1.0 + safe_return)

        # 3. Pénalité Sortino (downside)
        downside_penalty = 0.0
        if step_return < 0:
            downside_penalty = -(step_return ** 2) * self.downside_coef

        # 4. Bonus symétrique (upside)
        upside_bonus = 0.0
        if step_return > 0:
            upside_bonus = (step_return ** 2) * self.upside_coef

        # 5. Pénalité de churn (Taxe Cognitive) - LINÉAIRE
        churn_penalty = 0.0
        position_delta = 0.0
        if hasattr(self, '_prev_position_pct'):
            position_delta = abs(self.current_position_pct - self._prev_position_pct)
            if position_delta > 0 and self.churn_coef > 0:
                # Coût estimé * Multiplicateur d'amplification
                cost_rate = self._get_commission() + self.slippage  # 0.07%
                churn_penalty = -position_delta * cost_rate * self.churn_coef

        # 6. Smoothness penalty - QUADRATIQUE (pénalise les changements brusques)
        smoothness_penalty = 0.0
        current_smooth = self._get_smooth_coef()
        if hasattr(self, '_prev_position_pct') and current_smooth > 0:
            smoothness_penalty = -current_smooth * (position_delta ** 2)

        # 7. Total + Tanh scaling
        total_reward = reward_log_return + downside_penalty + upside_bonus + churn_penalty + smoothness_penalty

        # Compute final scaled reward
        scaled_reward = float(np.tanh(total_reward * self.reward_scaling))

        # Store metrics for observability
        self.reward_metrics = {
            "rewards/log_return": float(reward_log_return),
            "rewards/penalty_vol": float(downside_penalty),
            "rewards/bonus_upside": float(upside_bonus),
            "rewards/churn_penalty": float(churn_penalty),
            "rewards/smoothness_penalty": float(smoothness_penalty),
            "rewards/position_delta": float(position_delta),
            "rewards/total_raw": float(total_reward),
            "rewards/scaled": scaled_reward,
        }

        return scaled_reward

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Reset environment for a new episode.

        Args:
            seed: Random seed for reproducibility.
            options: Additional options (unused).

        Returns:
            Tuple of (observation, info).
        """
        super().reset(seed=seed)
        self._reset_state()

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step in the environment.

        Args:
            action: Target position in [-1, 1].

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        # 1. Parse raw action
        raw_action = float(np.clip(action[0], -1.0, 1.0))

        # 2. Apply volatility scaling (Target Volatility)
        self.current_volatility = self._calculate_volatility()
        self.vol_scalar = self.target_volatility / self.current_volatility
        self.vol_scalar = float(np.clip(self.vol_scalar, 0.1, self.max_leverage))

        # 3. Scale action: reduce position in high vol, increase in low vol
        effective_action = raw_action * self.vol_scalar
        effective_action = float(np.clip(effective_action, -1.0, 1.0))

        # 4. Discretize AFTER scaling to reduce micro-churn
        target_position_pct = effective_action
        if self.action_discretization > 0:
            target_position_pct = round(target_position_pct / self.action_discretization) * self.action_discretization
            target_position_pct = float(np.clip(target_position_pct, -1.0, 1.0))  # Ensure bounds

        # 5. Get current state
        old_nav = self._get_nav()
        old_price = self._get_price()

        # 6. Calculate position change
        # Map [-1, 1] to position: -1 = 0% long, 0 = 50%, +1 = 100%
        target_exposure = (target_position_pct + 1.0) / 2.0  # [0, 1]
        target_position_value = old_nav * target_exposure
        target_position_units = target_position_value / old_price

        # 7. Execute trade only if discretized position changed
        position_changed = (target_position_pct != self.current_position_pct)

        if position_changed:
            position_delta = target_position_units - self.position
            trade_value = abs(position_delta * old_price)

            # Calculate costs
            trade_cost = trade_value * (self._get_commission() + self.slippage)
            self.total_commission += trade_cost
            self.total_trades += 1

            # Update position
            self.cash -= position_delta * old_price + trade_cost
            self.position = target_position_units
            self.current_position_pct = target_position_pct

        # 8. Advance time
        self.current_step += 1

        # 9. Calculate new NAV
        new_nav = self._get_nav()

        # 10. Calculate reward (Hybrid Log-Sortino + Churn Penalty)
        reward = self._calculate_reward()

        # 11. Update state for next step
        self._prev_valuation = new_nav
        self._prev_position_pct = self.current_position_pct

        # Track returns for analysis and volatility calculation
        if old_nav > 0 and new_nav > 0:
            step_return = np.log(new_nav / old_nav)  # Log return for analysis
            simple_return = (new_nav - old_nav) / old_nav  # Simple return for vol calc
        else:
            step_return = 0.0
            simple_return = 0.0
        self.returns_history.append(step_return)
        self.returns_for_vol.append(simple_return)

        # 12. Check termination
        terminated = False
        truncated = False

        # Bankruptcy
        if new_nav <= 0:
            terminated = True
            reward = -1.0  # Max penalty (tanh already bounds to [-1, 1])

        # End of episode
        if self.current_step >= self.episode_end:
            terminated = True

        # 13. Get observation and info
        observation = self._get_observation()
        info = self._get_info(action[0], step_return, new_nav)

        return observation, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Get current observation window."""
        start_idx = self.current_step - self.window_size + 1
        end_idx = self.current_step + 1
        # Must return copy - SB3's replay buffer may modify observations in-place
        return self.data[start_idx:end_idx].copy()

    def _get_info(
        self,
        action: float = 0.0,
        step_return: float = 0.0,
        nav: Optional[float] = None
    ) -> dict:
        """Get info dict with monitoring metrics."""
        if nav is None:
            nav = self._get_nav()

        info = {
            'step': self.current_step,
            'nav': nav,
            'cash': self.cash,
            'position': self.position,
            'position_pct': self.current_position_pct,
            'price': self._get_price(),
            'action': action,
            'return': step_return,
            'total_trades': self.total_trades,
            'total_commission': self.total_commission,
            # Volatility scaling metrics
            'vol/current_volatility': self.current_volatility,
            'vol/vol_scalar': self.vol_scalar,
            'vol/target_volatility': self.target_volatility,
        }
        # Add reward metrics for observability
        info.update(self.reward_metrics)
        return info

    def render(self) -> None:
        """Render current state."""
        info = self._get_info()
        print(
            f"Step {info['step']:5d} | "
            f"Price: ${info['price']:,.2f} | "
            f"NAV: ${info['nav']:,.2f} | "
            f"Position: {info['position_pct']:+.2f} | "
            f"Trades: {info['total_trades']}"
        )

    def update_penalties(
        self,
        fee_rate: Optional[float] = None,
        smooth_coef: Optional[float] = None,
        churn_coef: Optional[float] = None,
    ) -> None:
        """
        Update penalty coefficients for curriculum learning.

        Allows dynamic adjustment of penalty parameters during training
        to implement curriculum strategies (e.g., start easy, increase penalties).

        Args:
            fee_rate: New transaction fee rate (commission).
            smooth_coef: New smoothness penalty coefficient.
            churn_coef: New churn penalty coefficient.
        """
        if fee_rate is not None:
            self.commission = fee_rate
        if smooth_coef is not None:
            self.smooth_coef = smooth_coef
        if churn_coef is not None:
            self.churn_coef = churn_coef

    @classmethod
    def create_train_val_envs(
        cls,
        parquet_path: str = "data/processed_data.parquet",
        train_ratio: float = 0.8,
        **kwargs
    ) -> Tuple['CryptoTradingEnv', 'CryptoTradingEnv']:
        """
        Create train and validation environments with proper splits.

        Args:
            parquet_path: Path to data file.
            train_ratio: Fraction of data for training.
            **kwargs: Additional arguments for environment.

        Returns:
            Tuple of (train_env, val_env).
        """
        df = pd.read_parquet(parquet_path)
        n_total = len(df)
        split_idx = int(n_total * train_ratio)

        train_env = cls(
            parquet_path=parquet_path,
            start_idx=0,
            end_idx=split_idx,
            random_start=True,
            **kwargs
        )

        val_env = cls(
            parquet_path=parquet_path,
            start_idx=split_idx,
            end_idx=n_total,
            random_start=False,  # Sequential for validation
            **kwargs
        )

        return train_env, val_env


if __name__ == "__main__":
    # Test the environment
    print("Testing CryptoTradingEnv...")

    env = CryptoTradingEnv(
        parquet_path="data/processed_data.parquet",
        window_size=64,
        random_start=False
    )

    print(f"\nObservation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.shape}")
    print(f"Features: {env.n_features}")

    # Run a few steps
    obs, info = env.reset()
    print(f"\nInitial observation shape: {obs.shape}")
    print(f"Initial NAV: ${info['nav']:,.2f}")

    total_reward = 0
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if i % 20 == 0:
            env.render()

        if terminated:
            print(f"\nEpisode ended at step {i}")
            break

    print(f"\nTotal reward: {total_reward:.4f}")
    print(f"Total trades: {info['total_trades']}")
    print(f"Total commission: ${info['total_commission']:.2f}")
    print("\n[OK] CryptoTradingEnv test passed!")

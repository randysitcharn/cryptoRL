# -*- coding: utf-8 -*-
"""
env.py - Trading Environment for RL with Foundation Model.

Gymnasium environment optimized for TQC training with pre-trained encoder.
Uses Differential Sharpe Ratio (DSR) for more stable reward signal.

Reference: Moody & Saffell (2001) - Learning to Trade via Direct Reinforcement
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List


# Columns to exclude from features (raw OHLCV and raw volumes)
# Must match src/data/dataset.py EXCLUDE_COLS for dimension consistency with encoder
EXCLUDE_COLS = [
    # Prix OHLC bruts
    'BTC_Close', 'ETH_Close', 'SPX_Close', 'DXY_Close', 'NASDAQ_Close',
    'BTC_Open', 'BTC_High', 'BTC_Low',
    'ETH_Open', 'ETH_High', 'ETH_Low',
    'SPX_Open', 'SPX_High', 'SPX_Low',
    'DXY_Open', 'DXY_High', 'DXY_Low',
    'NASDAQ_Open', 'NASDAQ_High', 'NASDAQ_Low',
    # Volumes bruts (utiliser VolRel à la place)
    'BTC_Volume', 'ETH_Volume', 'SPX_Volume', 'DXY_Volume', 'NASDAQ_Volume',
]


class CryptoTradingEnv(gym.Env):
    """
    Crypto Trading Environment for RL with Foundation Model.

    Features:
    - Continuous action space [-1, 1] for position sizing
    - Differential Sharpe Ratio reward (SOTA for trading)
    - Realistic transaction costs (0.06% commission)
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
        parquet_path: str = "data/processed_data.parquet",
        price_column: str = "BTC_Close",
        initial_balance: float = 10_000.0,
        commission: float = 0.0006,  # 0.06% per trade
        slippage: float = 0.0001,    # 0.01% slippage
        window_size: int = 64,
        reward_scaling: float = 1.0,
        random_start: bool = True,
        episode_length: Optional[int] = None,
        eta: float = 0.01,  # DSR EMA decay factor
        start_idx: Optional[int] = None,
        end_idx: Optional[int] = None,
    ):
        """
        Initialize the trading environment.

        Args:
            parquet_path: Path to processed_data.parquet file.
            price_column: Column name for price (for P&L calculation).
            initial_balance: Starting capital in USD.
            commission: Transaction fee (0.0006 = 0.06%).
            slippage: Slippage cost (0.0001 = 0.01%).
            window_size: Lookback window size for observations.
            reward_scaling: Multiplier for reward signal.
            random_start: If True, start episodes at random positions.
            episode_length: Max steps per episode (None = full dataset).
            eta: EMA decay factor for Differential Sharpe Ratio.
            start_idx: Start index for data slice (for train/val split).
            end_idx: End index for data slice (for train/val split).
        """
        super().__init__()

        # Load data
        df = pd.read_parquet(parquet_path)

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

        # DSR parameters
        self.eta = eta  # EMA decay factor

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

        # DSR state (Differential Sharpe Ratio)
        self.A = 0.0  # EMA of returns
        self.B = 0.0  # EMA of squared returns

        # Tracking
        self.total_trades = 0
        self.total_commission = 0.0
        self.returns_history: List[float] = []

    def _get_price(self, step: Optional[int] = None) -> float:
        """Get price at given step (default: current step)."""
        idx = step if step is not None else self.current_step
        return float(self.prices[idx])

    def _get_nav(self) -> float:
        """Calculate Net Asset Value."""
        price = self._get_price()
        return self.cash + self.position * price

    def _calculate_dsr(self, step_return: float) -> float:
        """
        Calculate Differential Sharpe Ratio.

        DSR provides a gradient-friendly reward that captures both
        return and risk in a single value.

        Reference: Moody & Saffell (2001)

        Args:
            step_return: Log return for this step.

        Returns:
            Differential Sharpe Ratio value.
        """
        # Update EMAs
        delta_A = step_return - self.A
        delta_B = step_return ** 2 - self.B

        self.A += self.eta * delta_A
        self.B += self.eta * delta_B

        # Calculate DSR
        # DSR = (B_{t-1} * δA - 0.5 * A_{t-1} * δB) / (B_{t-1} - A_{t-1}²)^{3/2}
        variance = self.B - self.A ** 2

        if variance > 1e-8:
            dsr = (
                (self.B * delta_A - 0.5 * self.A * delta_B)
                / (variance ** 1.5 + 1e-8)
            )
        else:
            # Fall back to simple return when variance is too small
            dsr = step_return * 100

        return dsr

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
        # 1. Parse action
        target_position_pct = float(np.clip(action[0], -1.0, 1.0))

        # 2. Get current state
        old_nav = self._get_nav()
        old_price = self._get_price()

        # 3. Calculate position change
        # Map [-1, 1] to position: -1 = 0% long, 0 = 50%, +1 = 100%
        target_exposure = (target_position_pct + 1.0) / 2.0  # [0, 1]
        target_position_value = old_nav * target_exposure
        target_position_units = target_position_value / old_price

        # 4. Execute trade
        position_delta = target_position_units - self.position
        trade_value = abs(position_delta * old_price)

        if trade_value > 0.01:  # Minimum trade threshold
            # Calculate costs
            trade_cost = trade_value * (self.commission + self.slippage)
            self.total_commission += trade_cost
            self.total_trades += 1

            # Update position
            self.cash -= position_delta * old_price + trade_cost
            self.position = target_position_units
            self.current_position_pct = target_position_pct

        # 5. Advance time
        self.current_step += 1

        # 6. Calculate new NAV and return
        new_nav = self._get_nav()

        # Log return
        if old_nav > 0 and new_nav > 0:
            step_return = np.log(new_nav / old_nav)
        else:
            step_return = 0.0

        self.returns_history.append(step_return)

        # 7. Calculate reward (Differential Sharpe Ratio)
        reward = self._calculate_dsr(step_return) * self.reward_scaling

        # Clip reward for stability
        reward = float(np.clip(reward, -10.0, 10.0))

        # 8. Check termination
        terminated = False
        truncated = False

        # Bankruptcy
        if new_nav <= 0:
            terminated = True
            reward = -100.0

        # End of episode
        if self.current_step >= self.episode_end:
            terminated = True

        # 9. Get observation and info
        observation = self._get_observation()
        info = self._get_info(action[0], step_return, new_nav)

        return observation, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Get current observation window."""
        start_idx = self.current_step - self.window_size + 1
        end_idx = self.current_step + 1
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

        return {
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
            'sharpe_ema_A': self.A,
            'sharpe_ema_B': self.B,
        }

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

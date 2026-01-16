# -*- coding: utf-8 -*-
"""
batch_env.py - GPU-Vectorized Trading Environment for RL.

Implements a batch environment where N environments run in parallel on GPU
using PyTorch tensors. This eliminates IPC overhead from SubprocVecEnv
and achieves 10-50x speedup.

Architecture:
    SubprocVecEnv: 31 processes × 1 env  → CPU bottleneck (IPC/pickling)
    BatchCryptoEnv: 1 process × 1024 envs → GPU saturated (tensor ops)
"""

import numpy as np
import pandas as pd
import torch
from typing import Optional, Tuple, Dict, Any, List
from stable_baselines3.common.vec_env import VecEnv
from gymnasium import spaces

from src.config import EXCLUDE_COLS


class BatchCryptoEnv(VecEnv):
    """
    GPU-Vectorized Crypto Trading Environment compatible with SB3.

    All N environments are managed by a single process using PyTorch tensors.
    Each step() call processes all environments in parallel on GPU.

    Features:
        - Continuous action space [-1, 1] for position sizing
        - Hybrid Log-Sortino reward (log return + penalties)
        - Volatility scaling with EMA variance
        - Action discretization to reduce churn
        - Compatible with SB3's TQC via VecEnv interface

    Performance:
        - No IPC overhead (single process)
        - Batch operations on GPU (parallel across n_envs)
        - FPS: 2k → 50k steps/s (vs SubprocVecEnv)
    """

    def __init__(
        self,
        parquet_path: str = "data/processed_data.parquet",
        price_column: str = "close",
        n_envs: int = 1024,
        device: str = "cuda",
        # Environment params
        window_size: int = 64,
        episode_length: int = 2048,
        initial_balance: float = 10_000.0,
        commission: float = 0.0006,
        slippage: float = 0.0001,
        # Reward params
        reward_scaling: float = 1.0,
        downside_coef: float = 10.0,
        upside_coef: float = 0.0,
        action_discretization: float = 0.1,
        churn_coef: float = 0.0,
        smooth_coef: float = 0.0,
        # Volatility scaling
        target_volatility: float = 0.01,
        vol_window: int = 24,
        max_leverage: float = 5.0,
        # Data range (for train/val split)
        start_idx: Optional[int] = None,
        end_idx: Optional[int] = None,
        # Regularization (anti-overfitting)
        observation_noise: float = 0.0,  # 0 = disabled by default
    ):
        """
        Initialize the batch environment.

        Args:
            parquet_path: Path to processed data parquet file.
            price_column: Column name for price series.
            n_envs: Number of parallel environments.
            device: PyTorch device ('cuda' or 'cpu').
            window_size: Observation window size.
            episode_length: Max steps per episode.
            initial_balance: Starting capital per env.
            commission: Transaction fee rate.
            slippage: Slippage cost rate.
            reward_scaling: Reward multiplier (keep at 1.0, SCALE=100 internal).
            downside_coef: Sortino downside penalty coefficient.
            upside_coef: Upside bonus coefficient.
            action_discretization: Action discretization step (0.1 = 21 levels).
            churn_coef: Churn penalty coefficient.
            smooth_coef: Smoothness penalty coefficient.
            target_volatility: Target vol for position scaling.
            vol_window: Rolling window for volatility.
            max_leverage: Max vol scaling factor.
            start_idx: Start index for data slice.
            end_idx: End index for data slice.
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.num_envs = n_envs  # Store locally, VecEnv.__init__ will set self.num_envs

        # Store params
        self.window_size = window_size
        self.episode_length = episode_length
        self.initial_balance = initial_balance
        self.commission = commission
        self.slippage = slippage
        self.reward_scaling = reward_scaling
        self.downside_coef = downside_coef
        self.upside_coef = upside_coef
        self.action_discretization = action_discretization
        self.churn_coef = churn_coef
        self.smooth_coef = smooth_coef
        self.target_volatility = target_volatility
        self.vol_window = vol_window
        self.max_leverage = max_leverage
        self.observation_noise = observation_noise
        self.training = True  # Flag for observation noise (disable during eval)

        # Load and preprocess data
        df = pd.read_parquet(parquet_path)

        # Apply data slice
        if start_idx is not None or end_idx is not None:
            start = start_idx or 0
            end = end_idx or len(df)
            df = df.iloc[start:end].reset_index(drop=True)

        # Extract price column
        if price_column not in df.columns:
            raise ValueError(f"Price column '{price_column}' not found")
        prices_np = df[price_column].values.astype(np.float32)

        # Extract features
        feature_cols = [
            col for col in df.columns
            if col not in EXCLUDE_COLS
            and df[col].dtype in ['float64', 'float32', 'int64']
        ]
        self.feature_names = feature_cols
        self.n_features = len(feature_cols)
        data_np = df[feature_cols].values.astype(np.float32)

        # Handle NaN
        data_np = np.nan_to_num(data_np, nan=0.0)

        # Move to GPU as contiguous tensors
        self.prices = torch.tensor(prices_np, device=self.device).contiguous()
        self.data = torch.tensor(data_np, device=self.device).contiguous()
        self.n_steps = len(df)

        # Define spaces (SB3 requirement)
        self.observation_space = spaces.Dict({
            "market": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(window_size, self.n_features),
                dtype=np.float32
            ),
            "position": spaces.Box(
                low=-1.0, high=1.0,
                shape=(1,),
                dtype=np.float32
            )
        })
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(1,),
            dtype=np.float32
        )

        # Initialize VecEnv base class
        super().__init__(n_envs, self.observation_space, self.action_space)

        # Allocate state tensors on GPU
        self._allocate_state_tensors()

        print(f"[BatchCryptoEnv] Initialized: {n_envs} envs on {self.device}, "
              f"{self.n_steps} steps, {self.n_features} features")

    def _allocate_state_tensors(self) -> None:
        """Allocate all state tensors on GPU."""
        n = self.num_envs  # Set by VecEnv.__init__
        device = self.device

        # Episode state
        self.current_steps = torch.zeros(n, dtype=torch.long, device=device)
        self.episode_starts = torch.zeros(n, dtype=torch.long, device=device)
        self.episode_ends = torch.zeros(n, dtype=torch.long, device=device)

        # Portfolio state
        self.cash = torch.full((n,), self.initial_balance, device=device)
        self.positions = torch.zeros(n, device=device)  # Units held
        self.position_pcts = torch.zeros(n, device=device)  # [-1, 1]
        self.prev_position_pcts = torch.zeros(n, device=device)
        self.prev_valuations = torch.full((n,), self.initial_balance, device=device)

        # Volatility state (EMA variance)
        self.ema_vars = torch.full((n,), self.target_volatility ** 2, device=device)
        self.vol_scalars = torch.ones(n, device=device)

        # Tracking
        self.total_trades = torch.zeros(n, dtype=torch.long, device=device)
        self.total_commissions = torch.zeros(n, device=device)

        # Episode tracking for SB3 monitoring (GPU tensors)
        self.episode_rewards = torch.zeros(n, device=device)
        self.episode_lengths = torch.zeros(n, dtype=torch.long, device=device)

        # Drawdown tracking (GPU)
        self.peak_navs = torch.full((n,), self.initial_balance, device=device)
        self.current_drawdowns = torch.zeros(n, device=device)

        # Shared penalty values (for curriculum)
        self._current_smooth_coef = self.smooth_coef
        self._current_churn_coef = self.churn_coef

        # Pre-allocated action buffer (avoids tensor creation each step)
        self._action_buffer = torch.zeros(n, device=device)

        # Done flags buffer (for _build_infos optimization)
        self._dones = torch.zeros(n, dtype=torch.bool, device=device)

        # Reward component buffers (pour observabilité)
        self._rew_pnl = torch.zeros(n, device=device)
        self._rew_churn = torch.zeros(n, device=device)
        self._rew_smooth = torch.zeros(n, device=device)

        # Adaptive Profit-Gated Churn (2026-01-16)
        # EMA tracker for cumulative PnL - gate opens only when profitable
        self.pnl_ema = torch.zeros(n, device=device, dtype=torch.float32)
        self.ema_alpha = 0.02  # Smoothing factor (slow adaptation)

        # OPTIMIZATION: Pre-allocate window offsets to avoid torch.arange in hot path
        self.window_offsets = torch.arange(self.window_size, device=device)

    def _get_prices(self, steps: torch.Tensor) -> torch.Tensor:
        """Get prices at given steps for all envs. Shape: (n_envs,)"""
        return self.prices[steps]

    def _get_navs(self) -> torch.Tensor:
        """Calculate NAV for all envs. Shape: (n_envs,)"""
        prices = self._get_prices(self.current_steps)
        return self.cash + self.positions * prices

    def get_global_metrics(self) -> Dict[str, float]:
        """
        Vectorized GPU metrics for logging (called by callbacks, not every step).

        Returns aggregated metrics computed entirely on GPU with minimal CPU transfer.
        """
        navs = self._get_navs()
        drawdowns = (self.peak_navs - navs) / torch.clamp(self.peak_navs, min=1.0)

        return {
            "portfolio_value": navs.mean().item(),
            "max_drawdown": drawdowns.max().item(),
            "price": self.prices[self.current_steps[0]].item(),
            "position_pct": self.position_pcts.mean().item(),
            "nav_std": navs.std().item(),
            # Reward components (for observability)
            "avg_rew_pnl": self._rew_pnl.mean().item(),
            "avg_rew_churn": self._rew_churn.mean().item(),
            "avg_rew_smooth": self._rew_smooth.mean().item(),
        }

    def _get_batch_windows(self, steps: torch.Tensor) -> torch.Tensor:
        """
        Extract observation windows for all envs efficiently.
        Optimized: Uses pre-allocated offsets to avoid GPU memory allocation per step.

        Shape: (n_envs, window_size, n_features)
        """
        w = self.window_size

        # Calculate start indices: (n_envs,)
        start_indices = steps - w + 1

        # Broadcasting: (n_envs, 1) + (1, window_size) -> (n_envs, window_size)
        # Uses pre-allocated self.window_offsets (zero allocation)
        indices = start_indices.unsqueeze(1) + self.window_offsets.unsqueeze(0)

        # Gather data: (n_envs, window_size, n_features)
        return self.data[indices]

    def _calculate_volatility(self, returns: torch.Tensor) -> torch.Tensor:
        """
        Update EMA variance and return current volatility.

        Args:
            returns: Simple returns for current step. Shape: (n_envs,)

        Returns:
            Current volatility (std dev). Shape: (n_envs,)
        """
        alpha = 2.0 / (self.vol_window + 1)

        # EMA variance update: var_t = alpha * r^2 + (1-alpha) * var_{t-1}
        self.ema_vars = alpha * (returns ** 2) + (1.0 - alpha) * self.ema_vars

        # Return std dev (clamped)
        return torch.clamp(torch.sqrt(self.ema_vars), min=1e-6)

    def _calculate_rewards(
        self,
        step_returns: torch.Tensor,
        position_deltas: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """
        Vectorized Hybrid Log-Sortino reward calculation with Adaptive Profit-Gated Churn.

        Args:
            step_returns: Simple returns (new_nav - old_nav) / old_nav. Shape: (n_envs,)
            position_deltas: |new_pos - old_pos|. Shape: (n_envs,)
            dones: Episode termination flags. Shape: (n_envs,)

        Returns:
            Rewards for all envs. Shape: (n_envs,)
        """
        SCALE = 100.0

        # 1. Log return (clamped to avoid log(0))
        safe_returns = torch.clamp(step_returns, min=-0.99)
        log_returns = torch.log1p(safe_returns) * SCALE

        # 2. Downside penalty (quadratic, only for negative returns)
        downside_mask = step_returns < 0
        downside_penalties = torch.where(
            downside_mask,
            -(step_returns ** 2) * self.downside_coef * SCALE,
            torch.zeros_like(step_returns)
        )

        # 3. Upside bonus (quadratic, only for positive returns)
        upside_mask = step_returns > 0
        upside_bonuses = torch.where(
            upside_mask,
            (step_returns ** 2) * self.upside_coef * SCALE,
            torch.zeros_like(step_returns)
        )

        # 4. Adaptive Profit-Gated Churn (2026-01-16, Fixed 2026-01-16)
        # Update PnL EMA tracker (step_returns = step PnL as fraction)
        self.pnl_ema = (1 - self.ema_alpha) * self.pnl_ema + self.ema_alpha * step_returns

        # Linear Ratio Gate (replaces sigmoid to avoid gradient inversion)
        # Target PnL threshold where full penalty applies (0.5% cumulative return)
        target_pnl = 0.005
        # Linear ramp: 0% penalty at PnL≤0, 100% penalty at PnL≥target_pnl
        # This ensures: Net Reward = PnL × (1 - k) where k < 1, always increasing with PnL
        churn_gate = torch.clamp(self.pnl_ema / target_pnl, min=0.0, max=1.0)

        # Raw churn penalty (linear, aligned with commission)
        cost_rate = self.commission + self.slippage
        raw_churn_penalty = position_deltas * cost_rate * self._current_churn_coef * SCALE

        # Apply gate: No penalty when unprofitable, full penalty when profitable
        churn_penalties = -raw_churn_penalty * churn_gate

        # 5. Smoothness penalty (quadratic, regularization) - NOT gated
        smoothness_penalties = -self._current_smooth_coef * (position_deltas ** 2) * SCALE

        # 6. Store components for observability (BEFORE scaling)
        self._rew_pnl = log_returns + downside_penalties + upside_bonuses
        self._rew_churn = churn_penalties
        self._rew_smooth = smoothness_penalties

        # 7. Total reward
        total = self._rew_pnl + self._rew_churn + self._rew_smooth

        return total * self.reward_scaling

    def reset(self) -> Dict[str, np.ndarray]:
        """
        Reset all environments.

        Returns:
            Observation dict with market and position arrays.
        """
        n = self.num_envs
        min_start = self.window_size
        max_start = self.n_steps - self.episode_length - 1

        # Random starts for each env
        self.episode_starts = torch.randint(
            min_start, max_start, (n,), device=self.device
        )
        self.current_steps = self.episode_starts.clone()
        self.episode_ends = self.episode_starts + self.episode_length

        # Reset portfolio
        self.cash.fill_(self.initial_balance)
        self.positions.zero_()
        self.position_pcts.zero_()
        self.prev_position_pcts.zero_()
        self.prev_valuations.fill_(self.initial_balance)

        # Reset volatility
        self.ema_vars.fill_(self.target_volatility ** 2)
        self.vol_scalars.fill_(1.0)

        # Reset tracking
        self.total_trades.zero_()
        self.total_commissions.zero_()

        # Reset episode tracking for SB3 monitoring
        self.episode_rewards.zero_()
        self.episode_lengths.zero_()

        # Reset drawdown tracking
        self.peak_navs.fill_(self.initial_balance)
        self.current_drawdowns.zero_()

        # Reset PnL EMA tracker (Adaptive Profit-Gated Churn)
        self.pnl_ema.zero_()

        return self._get_observations()

    def _get_observations(self) -> Dict[str, np.ndarray]:
        """Get observations for all envs as numpy (SB3 compatible)."""
        # Market windows: (n_envs, window_size, n_features)
        market = self._get_batch_windows(self.current_steps)

        # Add observation noise for regularization (anti-overfitting)
        if self.observation_noise > 0 and self.training:
            noise = torch.randn_like(market) * self.observation_noise
            market = market + noise

        # Position: (n_envs, 1) - NO noise on position (agent's own state)
        position = self.position_pcts.unsqueeze(1)

        # Transfer to CPU numpy for SB3
        return {
            "market": market.cpu().numpy(),
            "position": position.cpu().numpy()
        }

    def set_training_mode(self, training: bool):
        """Enable/disable observation noise for eval."""
        self.training = training

    def step_async(self, actions) -> None:
        """Store actions for step_wait (VecEnv interface).

        Optimized: uses pre-allocated buffer to avoid tensor creation.
        """
        if isinstance(actions, torch.Tensor):
            self._action_buffer.copy_(actions.squeeze(-1))
        else:
            # Fast numpy → GPU copy via pre-allocated buffer
            self._action_buffer.copy_(
                torch.from_numpy(actions.astype(np.float32)).squeeze(-1).to(self.device)
            )

    def step_wait(self) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, List[Dict]]:
        """
        Execute one step for all environments.

        Returns:
            Tuple of (observations, rewards, dones, infos).
        """
        actions = self._action_buffer  # Shape: (n_envs,) - pre-allocated buffer

        # 1. Clip raw actions
        raw_actions = torch.clamp(actions, -1.0, 1.0)

        # 2. Calculate volatility and vol scalar
        # Use previous step return for vol update (need at least one step)
        old_navs = self._get_navs()

        # 3. Apply volatility scaling
        vol = torch.clamp(torch.sqrt(self.ema_vars), min=1e-6)
        self.vol_scalars = torch.clamp(
            self.target_volatility / vol,
            min=0.1,
            max=self.max_leverage
        )
        effective_actions = torch.clamp(raw_actions * self.vol_scalars, -1.0, 1.0)

        # 4. Discretize actions
        if self.action_discretization > 0:
            target_positions = torch.round(
                effective_actions / self.action_discretization
            ) * self.action_discretization
            target_positions = torch.clamp(target_positions, -1.0, 1.0)
        else:
            target_positions = effective_actions

        # 5. Calculate position deltas
        position_deltas = torch.abs(target_positions - self.position_pcts)

        # 6. Execute trades (only where position changed)
        old_prices = self._get_prices(self.current_steps)
        position_changed = target_positions != self.position_pcts

        # Map [-1, 1] to exposure [0, 1]
        target_exposures = (target_positions + 1.0) / 2.0
        target_values = old_navs * target_exposures
        target_units = target_values / old_prices

        # Calculate trade costs
        units_delta = target_units - self.positions
        trade_values = torch.abs(units_delta * old_prices)
        trade_costs = trade_values * (self.commission + self.slippage)

        # Apply trades where position changed
        self.cash = torch.where(
            position_changed,
            self.cash - units_delta * old_prices - trade_costs,
            self.cash
        )
        self.positions = torch.where(position_changed, target_units, self.positions)
        self.position_pcts = torch.where(position_changed, target_positions, self.position_pcts)
        self.total_trades += position_changed.long()
        self.total_commissions += torch.where(position_changed, trade_costs, torch.zeros_like(trade_costs))

        # 7. Advance time
        self.current_steps += 1

        # 8. Calculate new NAV and returns
        new_navs = self._get_navs()
        step_returns = (new_navs - old_navs) / old_navs

        # 8b. Update peak NAV and calculate drawdown (BEFORE checking dones)
        self.peak_navs = torch.max(self.peak_navs, new_navs)
        self.current_drawdowns = (self.peak_navs - new_navs) / self.peak_navs

        # Update volatility with new returns
        self._calculate_volatility(step_returns)

        # 12. Check termination (needed for reward calculation)
        terminated = self.current_steps >= self.episode_ends
        bankrupt = new_navs <= 0
        dones = terminated | bankrupt

        # 9. Calculate rewards (with Adaptive Profit-Gated Churn)
        rewards = self._calculate_rewards(step_returns, position_deltas, dones)

        # 10. Update episode tracking (BEFORE checking dones)
        self.episode_rewards += rewards
        self.episode_lengths += 1

        # 11. Update state
        self.prev_valuations = new_navs
        self.prev_position_pcts = self.position_pcts.clone()

        # Apply bankruptcy penalty (dones already computed before rewards)
        rewards = torch.where(bankrupt, torch.full_like(rewards, -1.0), rewards)

        # Store dones for _build_infos optimization
        self._dones = dones

        # 13. Capture episode stats BEFORE reset (for SB3 ep_info_buffer)
        final_ep_rewards = None
        final_ep_lengths = None
        n_done = 0
        if dones.any():
            n_done = dones.sum().item()
            final_ep_rewards = self.episode_rewards[dones].cpu().numpy()
            final_ep_lengths = self.episode_lengths[dones].cpu().numpy()

        # 14. Auto-reset terminated environments (resets episode tracking)
        if n_done > 0:
            self._auto_reset(dones)

        # 15. Build info dicts (OPTIMIZED: only for done envs)
        infos = self._build_infos(dones, final_ep_rewards, final_ep_lengths)

        # 14. Get observations and transfer to CPU
        obs = self._get_observations()
        rewards_np = rewards.cpu().numpy()
        dones_np = dones.cpu().numpy()

        return obs, rewards_np, dones_np, infos

    def _auto_reset(self, dones: torch.Tensor) -> None:
        """Reset environments that are done."""
        n_done = dones.sum().item()
        if n_done == 0:
            return

        min_start = self.window_size
        max_start = self.n_steps - self.episode_length - 1

        # Generate new random starts for done envs
        new_starts = torch.randint(
            min_start, max_start, (n_done,), device=self.device
        )

        # Reset done envs
        self.episode_starts[dones] = new_starts
        self.current_steps[dones] = new_starts
        self.episode_ends[dones] = new_starts + self.episode_length

        self.cash[dones] = self.initial_balance
        self.positions[dones] = 0.0
        self.position_pcts[dones] = 0.0
        self.prev_position_pcts[dones] = 0.0
        self.prev_valuations[dones] = self.initial_balance

        self.ema_vars[dones] = self.target_volatility ** 2
        self.vol_scalars[dones] = 1.0

        self.total_trades[dones] = 0
        self.total_commissions[dones] = 0.0

        # Reset episode tracking for done envs
        self.episode_rewards[dones] = 0.0
        self.episode_lengths[dones] = 0

        # Reset drawdown tracking for done envs
        self.peak_navs[dones] = self.initial_balance
        self.current_drawdowns[dones] = 0.0

        # Reset PnL EMA tracker for done envs (Adaptive Profit-Gated Churn)
        self.pnl_ema[dones] = 0.0

    def _build_infos(
        self,
        dones: torch.Tensor,
        final_ep_rewards: Optional[np.ndarray] = None,
        final_ep_lengths: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """Build info dicts - OPTIMIZED: only processes done envs.

        Args:
            dones: Done flags for all envs (GPU tensor).
            final_ep_rewards: Episode rewards for done envs (captured before reset).
            final_ep_lengths: Episode lengths for done envs (captured before reset).

        Note:
            For per-step metrics (NAV, price, etc.), use get_global_metrics()
            in callbacks instead of reading from info dict.
        """
        # Fast path: pre-allocate empty dicts (no loop, very fast)
        infos = [{} for _ in range(self.num_envs)]

        # Only process done environments (typically 0-2 per step)
        if not dones.any():
            return infos

        # Get done indices (small tensor, fast transfer)
        done_indices = torch.nonzero(dones, as_tuple=True)[0].cpu().numpy()

        for done_idx, env_idx in enumerate(done_indices):
            env_idx = int(env_idx)

            # SB3 expects terminal_observation for done envs
            infos[env_idx]["terminal_observation"] = self._get_single_obs(env_idx)

            # Inject episode info for SB3's ep_info_buffer
            if final_ep_rewards is not None and final_ep_lengths is not None:
                infos[env_idx]["episode"] = {
                    "r": float(final_ep_rewards[done_idx]),
                    "l": int(final_ep_lengths[done_idx]),
                    "t": 0.0
                }

        return infos

    def _get_single_obs(self, idx: int) -> Dict[str, np.ndarray]:
        """Get observation for a single env (for terminal_observation)."""
        step = self.current_steps[idx].item()
        start = step - self.window_size + 1
        market = self.data[start:step+1].cpu().numpy()
        position = np.array([self.position_pcts[idx].item()], dtype=np.float32)
        return {"market": market, "position": position}

    def close(self) -> None:
        """Clean up resources."""
        pass

    def env_is_wrapped(self, wrapper_class, indices=None) -> List[bool]:
        """Check if envs are wrapped (always False for batch env)."""
        return [False] * self.num_envs

    def env_method(self, method_name: str, *args, indices=None, **kwargs):
        """Call method on envs (limited support for batch env)."""
        if method_name == "update_penalties":
            if "smooth_coef" in kwargs:
                self._current_smooth_coef = kwargs["smooth_coef"]
            if "churn_coef" in kwargs:
                self._current_churn_coef = kwargs["churn_coef"]
            return [None] * self.num_envs
        raise NotImplementedError(f"env_method '{method_name}' not supported")

    def set_attr(self, attr_name: str, value, indices=None) -> None:
        """Set attribute on envs."""
        if attr_name == "smooth_coef":
            self._current_smooth_coef = value
        elif attr_name == "churn_coef":
            self._current_churn_coef = value
        else:
            raise NotImplementedError(f"set_attr '{attr_name}' not supported")

    def set_churn_penalty(self, value: float) -> None:
        """Direct setter for curriculum learning (bypasses wrapper issues)."""
        self._current_churn_coef = value

    def set_smoothness_penalty(self, value: float) -> None:
        """Direct setter for curriculum learning (bypasses wrapper issues)."""
        self._current_smooth_coef = value

    def get_attr(self, attr_name: str, indices=None):
        """Get attribute from envs."""
        if attr_name == "smooth_coef":
            return [self._current_smooth_coef] * self.num_envs
        elif attr_name == "churn_coef":
            return [self._current_churn_coef] * self.num_envs
        elif attr_name == "render_mode":
            return [None] * self.num_envs  # No rendering
        elif attr_name == "spec":
            return [None] * self.num_envs  # No gym spec
        # Return None for unknown attributes rather than raising
        return [None] * self.num_envs

    def seed(self, seed: Optional[int] = None) -> List[Optional[int]]:
        """Set random seed."""
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        return [seed] * self.num_envs

    def get_images(self) -> List:
        """Get rendered images (not implemented)."""
        raise NotImplementedError("Rendering not supported in BatchCryptoEnv")

    @classmethod
    def create_train_val_envs(
        cls,
        parquet_path: str = "data/processed_data.parquet",
        train_ratio: float = 0.8,
        n_envs: int = 1024,
        **kwargs
    ) -> Tuple["BatchCryptoEnv", "BatchCryptoEnv"]:
        """
        Create train and validation environments.

        Args:
            parquet_path: Path to data file.
            train_ratio: Fraction of data for training.
            n_envs: Number of environments per batch.
            **kwargs: Additional environment arguments.

        Returns:
            Tuple of (train_env, val_env).
        """
        df = pd.read_parquet(parquet_path)
        n_total = len(df)
        split_idx = int(n_total * train_ratio)

        train_env = cls(
            parquet_path=parquet_path,
            n_envs=n_envs,
            start_idx=0,
            end_idx=split_idx,
            **kwargs
        )

        val_env = cls(
            parquet_path=parquet_path,
            n_envs=min(n_envs, 32),  # Smaller for validation
            start_idx=split_idx,
            end_idx=n_total,
            **kwargs
        )

        return train_env, val_env


if __name__ == "__main__":
    # Test the batch environment
    print("Testing BatchCryptoEnv...")

    # Create environment
    env = BatchCryptoEnv(
        parquet_path="data/processed_data.parquet",
        n_envs=4,  # Small for testing
        device="cuda" if torch.cuda.is_available() else "cpu",
        window_size=64,
        episode_length=100,
    )

    print(f"\nNum envs: {env.num_envs}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Device: {env.device}")

    # Reset and run a few steps
    obs = env.reset()
    print(f"\nObservation shapes: market={obs['market'].shape}, position={obs['position'].shape}")

    total_rewards = np.zeros(env.num_envs)
    for i in range(50):
        actions = np.random.uniform(-1, 1, (env.num_envs, 1)).astype(np.float32)
        env.step_async(actions)
        obs, rewards, dones, infos = env.step_wait()
        total_rewards += rewards

        if i % 10 == 0:
            print(f"Step {i}: mean_reward={rewards.mean():.4f}, dones={dones.sum()}")

    print(f"\nTotal rewards: {total_rewards}")
    print(f"Final NAVs: {[info['nav'] for info in infos]}")
    print("\n[OK] BatchCryptoEnv test passed!")

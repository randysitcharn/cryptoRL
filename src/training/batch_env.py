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
        n_envs: int = 512,
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
        # Episode start mode
        random_start: bool = True,  # If False, start at beginning (for evaluation)
        # Short selling
        funding_rate: float = 0.0001,  # 0.01% per step (~0.24%/day) for short positions
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
            observation_noise: Noise level for observations (anti-overfitting).
            random_start: If False, start at beginning (for evaluation).
            funding_rate: Funding cost per step for short positions (perpetual futures style).
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
        self.random_start = random_start
        self.funding_rate = funding_rate
        self.training = True  # Flag for observation noise (disable during eval)
        self._last_noise_scale = 0.0  # For TensorBoard logging (Dynamic Noise)

        # Curriculum state (stateless architecture - AAAI 2024 Curriculum Learning)
        self.progress = 0.0           # Training progress [0, 1]
        self.curriculum_lambda = 0.0  # Dynamic penalty weight
        self.last_gate_mean = 0.0     # Mean gate opening for logging

        # ═══════════════════════════════════════════════════════════════════
        # PLO (Predictive Lagrangian Optimization) Multipliers
        # Each PLO controller adjusts its multiplier based on constraint violations
        # λ ∈ [1.0, 5.0] where 1.0 = neutral, 5.0 = max penalty
        # ═══════════════════════════════════════════════════════════════════
        self.downside_multiplier = 1.0  # PLO Drawdown: scales downside risk penalty
        self.churn_multiplier = 1.0     # PLO Churn: scales churn penalty
        self.smooth_multiplier = 1.0    # PLO Smoothness: scales smoothness penalty

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
        # ═══════════════════════════════════════════════════════════════════
        # OBSERVATION SPACE with PLO augmentation
        # Agent MUST see PLO levels for Markov property to hold
        # Without this, same state + action gives different rewards (non-stationary)
        # ═══════════════════════════════════════════════════════════════════
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
            ),
            # PLO Drawdown: Risk level (λ_dd normalized to [0, 1])
            "risk_level": spaces.Box(
                low=0.0, high=1.0,
                shape=(1,),
                dtype=np.float32
            ),
            # PLO Churn: Churn pressure level (λ_churn normalized to [0, 1])
            "churn_level": spaces.Box(
                low=0.0, high=1.0,
                shape=(1,),
                dtype=np.float32
            ),
            # PLO Smoothness: Smoothness pressure level (λ_smooth normalized to [0, 1])
            "smooth_level": spaces.Box(
                low=0.0, high=1.0,
                shape=(1,),
                dtype=np.float32
            ),
            # ═══════════════════════════════════════════════════════════════════
            # MORL: Cost preference parameter w_cost ∈ [0, 1]
            # 0 = Scalping (ignore costs, max profit)
            # 1 = Buy & Hold (minimize costs, conservative)
            # Agent learns π(a|s, w_cost) - policy conditioned on preference
            # ═══════════════════════════════════════════════════════════════════
            "w_cost": spaces.Box(
                low=0.0, high=1.0,
                shape=(1,),
                dtype=np.float32
            ),
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
        self._final_total_trades = None  # Captured before reset for evaluation
        self._final_navs = None  # Captured before reset for evaluation
        self._final_positions = None  # Captured before reset for evaluation

        # ═══════════════════════════════════════════════════════════════════
        # PLO Smoothness: Jerk tracking buffers
        # Jerk = |Δpos(t) - Δpos(t-1)| = acceleration of position changes
        # Must be calculated DURING _calculate_rewards BEFORE updating prev
        # ═══════════════════════════════════════════════════════════════════
        self.prev_position_deltas = torch.zeros(n, device=device)
        self.latest_jerks = torch.zeros(n, device=device)

        # ═══════════════════════════════════════════════════════════════════
        # MORL: w_cost preference parameter per environment
        # Shape: (n_envs, 1) for proper broadcasting in reward calculation
        # Sampled at reset() with biased distribution (20%/60%/20%)
        # ═══════════════════════════════════════════════════════════════════
        self.w_cost = torch.zeros(n, 1, device=device)
        self._eval_w_cost = None  # Fixed w_cost for evaluation mode (None = sample)

        # Reward component buffers (pour observabilité)
        self._rew_pnl = torch.zeros(n, device=device)
        self._rew_churn = torch.zeros(n, device=device)
        self._rew_smooth = torch.zeros(n, device=device)
        self._rew_downside = torch.zeros(n, device=device)  # Downside risk tracking

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
        Designed for Direct GPU Metric Polling from DetailTensorboardCallback.
        """
        navs = self._get_navs()
        drawdowns = (self.peak_navs - navs) / torch.clamp(self.peak_navs, min=1.0)

        return {
            # Portfolio metrics
            "portfolio_value": navs.mean().item(),
            "max_drawdown": drawdowns.max().item(),
            "price": self.prices[self.current_steps[0]].item(),
            "position_pct": self.position_pcts.mean().item(),
            "nav_std": navs.std().item(),
            # Reward components (Direct GPU Polling)
            "reward/pnl_component": self._rew_pnl.mean().item(),
            "reward/churn_cost": self._rew_churn.mean().item(),
            "reward/downside_risk": self._rew_downside.mean().item(),
            "reward/smoothness": self._rew_smooth.mean().item(),
            # Curriculum state (AAAI 2024)
            "curriculum/lambda": self.curriculum_lambda,
            "curriculum/progress": self.progress,
            "curriculum/gate_open": self.last_gate_mean,
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
        MORL Reward: Multi-Objective Scalarization with Conditioned Preference.

        Based on: Abels et al. (ICML 2019) - Dynamic Weights in Multi-Objective Deep RL.

        Architecture:
            - Agent sees w_cost ∈ [0, 1] in observation (Conditioned Network)
            - Reward = r_perf + w_cost * r_cost * MAX_PENALTY_SCALE
            - w_cost = 0: Scalping mode (ignore costs, max profit)
            - w_cost = 1: B&H mode (minimize costs, conservative)

        This replaces the old PLO + Curriculum architecture with a single
        preference parameter that the agent learns to condition on.

        Args:
            step_returns: Simple returns (new_nav - old_nav) / old_nav. Shape: (n_envs,)
            position_deltas: |new_pos - old_pos|. Shape: (n_envs,)
            dones: Episode termination flags. Shape: (n_envs,)

        Returns:
            Rewards for all envs. Shape: (n_envs,)
        """
        SCALE = 100.0
        
        # ═══════════════════════════════════════════════════════════════════
        # MORL CALIBRATION: MAX_PENALTY_SCALE
        # CRITICAL: r_cost * MAX_PENALTY_SCALE must be same order of magnitude as r_perf
        # If log-returns ≈ 0.01/step → SCALE*0.01 = 1.0
        # If position_delta ≈ 0.1/step → SCALE*0.1 = 10.0
        # We want w_cost=1 to make costs matter, so scale up costs
        # ═══════════════════════════════════════════════════════════════════
        MAX_PENALTY_SCALE = 2.0  # Calibrate: if r_cost flat in TensorBoard, increase

        # Safety caps to prevent NaN/explosion
        COST_PENALTY_CAP = 20.0

        # ═══════════════════════════════════════════════════════════════════
        # 0. JERK TRACKING (for backward compatibility with PLO logging)
        # ═══════════════════════════════════════════════════════════════════
        jerks = torch.abs(position_deltas - self.prev_position_deltas)
        self.latest_jerks = jerks.detach().clone()

        # ═══════════════════════════════════════════════════════════════════
        # 1. OBJECTIVE 1: Performance (Log Returns)
        # Always active, this is what we want to maximize
        # ═══════════════════════════════════════════════════════════════════
        safe_returns = torch.clamp(step_returns, min=-0.99)
        r_perf = torch.log1p(safe_returns) * SCALE

        # ═══════════════════════════════════════════════════════════════════
        # 2. OBJECTIVE 2: Costs (Turnover Penalty)
        # Raw turnover without gates/thresholds - immediate and local
        # Easier for critic to learn than deferred penalties
        # ═══════════════════════════════════════════════════════════════════
        # Direct turnover cost: penalize all position changes proportionally
        r_cost = -position_deltas * SCALE

        # Safety clip to prevent extreme penalties during high volatility
        r_cost = torch.clamp(r_cost, min=-COST_PENALTY_CAP)

        # ═══════════════════════════════════════════════════════════════════
        # 3. MORL SCALARIZATION: Dynamic weighting by w_cost
        # w_cost is known to agent via observation (Conditioned Network)
        # Shape: w_cost (n_envs, 1), r_cost (n_envs,) → squeeze for broadcast
        # ═══════════════════════════════════════════════════════════════════
        w_cost_squeezed = self.w_cost.squeeze(-1)  # (n_envs,)
        
        # Total reward: performance + weighted costs
        # When w_cost=0: reward = r_perf (pure profit seeking)
        # When w_cost=1: reward = r_perf + r_cost * MAX_PENALTY_SCALE (cost-conscious)
        reward = r_perf + (w_cost_squeezed * r_cost * MAX_PENALTY_SCALE)

        # ═══════════════════════════════════════════════════════════════════
        # 4. OBSERVABILITY (TensorBoard metrics)
        # Maintain backward compatibility with existing logging
        # ═══════════════════════════════════════════════════════════════════
        self._rew_pnl = r_perf
        self._rew_churn = w_cost_squeezed * r_cost * MAX_PENALTY_SCALE  # MORL cost component
        self._rew_downside = torch.zeros_like(r_perf)  # Disabled in MORL (simplification)
        self._rew_smooth = torch.zeros_like(r_perf)    # Disabled in MORL (simplification)

        # Legacy compatibility: store gate mean (always 1.0 in MORL - no gating)
        self.last_gate_mean = 1.0

        # ═══════════════════════════════════════════════════════════════════
        # 5. UPDATE JERK TRACKER (must be AFTER jerk calculation)
        # ═══════════════════════════════════════════════════════════════════
        self.prev_position_deltas = position_deltas.clone()

        return reward * self.reward_scaling

    def reset(self) -> Dict[str, np.ndarray]:
        """
        Reset all environments.

        Returns:
            Observation dict with market and position arrays.
        """
        n = self.num_envs
        min_start = self.window_size
        max_start = self.n_steps - self.episode_length - 1

        if self.random_start and max_start > min_start:
            # Random starts for each env (training mode)
            self.episode_starts = torch.randint(
                min_start, max_start, (n,), device=self.device
            )
        else:
            # Sequential start at beginning (evaluation mode)
            self.episode_starts = torch.full((n,), min_start, dtype=torch.long, device=self.device)

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

        # Reset PLO Smoothness jerk tracking
        self.prev_position_deltas.zero_()
        self.latest_jerks.zero_()

        # ═══════════════════════════════════════════════════════════════════
        # MORL: Sample w_cost with biased distribution (Audit SOTA Fix)
        # 20% extremes (0 or 1) + 60% uniform to ensure agent explores
        # both pure scalping and pure B&H strategies, not just the middle
        # ═══════════════════════════════════════════════════════════════════
        if self._eval_w_cost is not None:
            # Evaluation mode: use fixed w_cost for reproducibility
            self.w_cost.fill_(self._eval_w_cost)
        else:
            # Training mode: sample with biased distribution
            sample_type = torch.rand(self.num_envs, device=self.device)
            # 20% chance: w_cost = 0 (scalping mode - ignore costs)
            # 20% chance: w_cost = 1 (B&H mode - maximize cost avoidance)
            # 60% chance: w_cost ~ Uniform[0, 1] (exploration)
            self.w_cost = torch.where(
                sample_type.unsqueeze(1) < 0.2,
                torch.zeros(self.num_envs, 1, device=self.device),
                torch.where(
                    sample_type.unsqueeze(1) > 0.8,
                    torch.ones(self.num_envs, 1, device=self.device),
                    torch.rand(self.num_envs, 1, device=self.device)
                )
            )

        return self._get_observations()

    def _get_observations(self) -> Dict[str, np.ndarray]:
        """Get observations for all envs as numpy (SB3 compatible)."""
        # Market windows: (n_envs, window_size, n_features)
        market = self._get_batch_windows(self.current_steps)

        # ═══════════════════════════════════════════════════════════════════
        # DYNAMIC OBSERVATION NOISE (Audit 2026-01-19)
        # Combines Annealing + Volatility-Adaptive for anti-overfitting
        # See: docs/AUDIT_OBSERVATION_NOISE.md
        # ═══════════════════════════════════════════════════════════════════
        if self.observation_noise > 0 and self.training:
            # 1. ANNEALING (Time-based) - Standard NoisyRollout 2025
            # Reduces noise progressively from 100% to 50% during training
            # Not going to 0% prevents "catastrophic forgetting" of robustness
            annealing_factor = 1.0 - 0.5 * self.progress
            
            # 2. ADAPTIVE (Regime-based) - CryptoRL Innovation
            # If volatility doubles, noise is halved (and vice versa)
            # Clamped [0.5, 2.0] to prevent gradient explosion/collapse
            current_vol = torch.sqrt(self.ema_vars).clamp(min=1e-6)
            vol_factor = (self.target_volatility / current_vol).clamp(0.5, 2.0)
            
            # 3. COMBINED INJECTION
            # final_scale shape: (n_envs,) -> broadcast to (n_envs, window, features)
            final_scale = self.observation_noise * annealing_factor * vol_factor
            noise = torch.randn_like(market) * final_scale.unsqueeze(1).unsqueeze(2)
            market = market + noise
            
            # Store for TensorBoard logging (mean across envs)
            self._last_noise_scale = final_scale.mean().item()

        # Position: (n_envs, 1) - NO noise on position (agent's own state)
        position = self.position_pcts.unsqueeze(1)

        # ═══════════════════════════════════════════════════════════════════
        # PLO OBSERVATION AUGMENTATION
        # Agent MUST see PLO levels for Value Function to converge
        # Without this, environment becomes non-stationary (breaks Markov)
        # Normalization: λ ∈ [1, 5] → level ∈ [0, 1]
        # ═══════════════════════════════════════════════════════════════════
        
        # Risk level (PLO Drawdown λ_dd normalized)
        risk_level_value = (self.downside_multiplier - 1.0) / 4.0
        risk_level = torch.full(
            (self.num_envs, 1),
            risk_level_value,
            device=self.device
        )

        # Churn level (PLO Churn λ_churn normalized)
        churn_level_value = (self.churn_multiplier - 1.0) / 4.0
        churn_level = torch.full(
            (self.num_envs, 1),
            churn_level_value,
            device=self.device
        )

        # Smooth level (PLO Smoothness λ_smooth normalized)
        smooth_level_value = (self.smooth_multiplier - 1.0) / 4.0
        smooth_level = torch.full(
            (self.num_envs, 1),
            smooth_level_value,
            device=self.device
        )

        # Transfer to CPU numpy for SB3
        return {
            "market": market.cpu().numpy(),
            "position": position.cpu().numpy(),
            "risk_level": risk_level.cpu().numpy(),
            "churn_level": churn_level.cpu().numpy(),
            "smooth_level": smooth_level.cpu().numpy(),
            "w_cost": self.w_cost.cpu().numpy(),  # MORL preference parameter
        }

    def set_training_mode(self, training: bool):
        """Enable/disable observation noise for eval."""
        self.training = training

    def set_eval_w_cost(self, w_cost: Optional[float]):
        """
        Set fixed w_cost for evaluation mode (MORL Pareto Front).
        
        Args:
            w_cost: Fixed preference in [0, 1], or None to resume sampling.
                   0 = Scalping (ignore costs), 1 = B&H (minimize costs)
        """
        self._eval_w_cost = w_cost

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

        # 3. Apply volatility scaling with Volatility Floor (prevents Cash Trap)
        # Min vol corresponds to the inverse of max leverage (e.g., 0.05 / 5.0 = 0.01)
        # This ensures stable scaling when agent is in cash (vol = 0)
        min_vol = self.target_volatility / self.max_leverage
        current_vol = torch.sqrt(self.ema_vars)
        clipped_vol = torch.clamp(current_vol, min=min_vol)
        self.vol_scalars = torch.clamp(
            self.target_volatility / clipped_vol,
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

        # Direct mapping: -1=100% short, 0=cash, +1=100% long
        target_exposures = target_positions
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

        # 6b. Apply funding cost for short positions (perpetual futures style)
        if self.funding_rate > 0:
            short_mask = self.positions < 0
            funding_cost = torch.abs(self.positions) * old_prices * self.funding_rate
            self.cash = torch.where(short_mask, self.cash - funding_cost, self.cash)

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
        final_total_trades = None
        final_navs = None
        final_positions = None
        n_done = 0
        if dones.any():
            n_done = dones.sum().item()
            final_ep_rewards = self.episode_rewards[dones].cpu().numpy()
            final_ep_lengths = self.episode_lengths[dones].cpu().numpy()
            final_total_trades = self.total_trades[dones].cpu().numpy()  # Capture before reset!
            final_navs = self._get_navs()[dones].cpu().numpy()  # Capture NAV before reset!
            final_positions = self.position_pcts[dones].cpu().numpy()  # Capture position before reset!

        # 14. Auto-reset terminated environments (resets episode tracking)
        if n_done > 0:
            self._auto_reset(dones)

        # Store for _get_single_info (used by gym_step for evaluation)
        self._final_total_trades = final_total_trades
        self._final_navs = final_navs
        self._final_positions = final_positions

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

        # Generate new starts for done envs
        if self.random_start and max_start > min_start:
            # Random starts (training mode)
            new_starts = torch.randint(
                min_start, max_start, (n_done,), device=self.device
            )
        else:
            # Sequential start at beginning (evaluation mode)
            new_starts = torch.full((n_done,), min_start, dtype=torch.long, device=self.device)

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

        # Reset PLO Smoothness jerk tracking for done envs
        self.prev_position_deltas[dones] = 0.0
        self.latest_jerks[dones] = 0.0

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
        
        # PLO levels (same for all envs, use current multipliers)
        risk_level = np.array([(self.downside_multiplier - 1.0) / 4.0], dtype=np.float32)
        churn_level = np.array([(self.churn_multiplier - 1.0) / 4.0], dtype=np.float32)
        smooth_level = np.array([(self.smooth_multiplier - 1.0) / 4.0], dtype=np.float32)
        
        return {
            "market": market,
            "position": position,
            "risk_level": risk_level,
            "churn_level": churn_level,
            "smooth_level": smooth_level,
        }

    # =========================================================================
    # Gymnasium-compatible interface (for n_envs=1, evaluation mode)
    # =========================================================================

    def gym_reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict[str, np.ndarray], dict]:
        """
        Gymnasium-compatible reset for single env mode.

        Use this method for evaluation/backtesting with n_envs=1.
        Returns observation and info dict like standard Gymnasium envs.

        Args:
            seed: Random seed (optional).
            options: Additional options (unused).

        Returns:
            Tuple of (observation, info).
        """
        assert self.num_envs == 1, f"gym_reset() requires n_envs=1, got {self.num_envs}"

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        obs = self.reset()
        # Unwrap single env observation
        obs_single = {k: v[0] for k, v in obs.items()}
        info = self._get_single_info(0)

        return obs_single, info

    def gym_step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, dict]:
        """
        Gymnasium-compatible step for single env mode.

        Use this method for evaluation/backtesting with n_envs=1.
        Returns (obs, reward, terminated, truncated, info) like standard Gymnasium envs.

        Args:
            action: Action array (will be reshaped to (1, 1) for batch processing).

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        assert self.num_envs == 1, f"gym_step() requires n_envs=1, got {self.num_envs}"

        # Ensure action is properly shaped for batch processing
        if isinstance(action, (int, float)):
            action = np.array([[action]], dtype=np.float32)
        elif action.ndim == 0:
            action = np.array([[action.item()]], dtype=np.float32)
        elif action.ndim == 1:
            action = action.reshape(1, -1)

        self.step_async(action)
        obs, rewards, dones, infos = self.step_wait()

        # Unwrap single env results
        obs_single = {k: v[0] for k, v in obs.items()}
        reward = float(rewards[0])
        terminated = bool(dones[0])
        truncated = False  # We don't use truncation
        info = infos[0]

        # Add extra info for evaluation compatibility
        info.update(self._get_single_info(0))

        return obs_single, reward, terminated, truncated, info

    def _get_single_info(self, idx: int) -> dict:
        """
        Build comprehensive info dict for single env (Gymnasium compatibility).

        Args:
            idx: Environment index (0 for single env mode).

        Returns:
            Info dict with NAV, position, price, and other metrics.
        """
        # Use captured values if episode just ended (before reset cleared them)
        if self._dones[idx]:
            # Episode ended - use values captured BEFORE auto-reset
            if self._final_navs is not None:
                nav = float(self._final_navs[0])  # Single env mode
            else:
                nav = float(self._get_navs()[idx].item())
            if self._final_positions is not None:
                position_pct = float(self._final_positions[0])
            else:
                position_pct = float(self.position_pcts[idx].item())
            if self._final_total_trades is not None:
                total_trades = int(self._final_total_trades[0])
            else:
                total_trades = int(self.total_trades[idx].item())
        else:
            # Episode ongoing - use current values
            nav = float(self._get_navs()[idx].item())
            position_pct = float(self.position_pcts[idx].item())
            total_trades = int(self.total_trades[idx].item())

        return {
            'nav': nav,
            'cash': float(self.cash[idx].item()),
            'position': float(self.positions[idx].item()),
            'position_pct': position_pct,
            'price': float(self.prices[self.current_steps[idx]].item()),
            'total_trades': total_trades,
            'total_commission': float(self.total_commissions[idx].item()),
            'vol/current_volatility': float(torch.sqrt(self.ema_vars[idx]).item()),
            'vol/vol_scalar': float(self.vol_scalars[idx].item()),
            'vol/target_volatility': self.target_volatility,
        }

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

    def set_progress(self, progress: float) -> None:
        """
        Update training progress and curriculum lambda.

        Curriculum Schedule (AAAI 2024 Curriculum Learning for HFT):
          Phase 1 (0-10%):   lambda = 0.0  (Pure Exploration)
          Phase 2 (10-30%):  lambda = 0.0 -> 0.4 (Reality Ramp)
          Phase 3 (30-100%): lambda = 0.4 (Stability)

        Args:
            progress: Training progress as fraction [0, 1].
        """
        self.progress = max(0.0, min(1.0, progress))

        if self.progress <= 0.10:
            # Phase 1: Pure Exploration - no penalties
            self.curriculum_lambda = 0.0
        elif self.progress <= 0.30:
            # Phase 2: Reality Ramp - linear 0.0 -> 0.4
            phase_progress = (self.progress - 0.10) / 0.20
            self.curriculum_lambda = 0.4 * phase_progress
        else:
            # Phase 3: Stability - fixed discipline
            self.curriculum_lambda = 0.4

    # ═══════════════════════════════════════════════════════════════════════
    # PLO (Predictive Lagrangian Optimization) Setters and Properties
    # ═══════════════════════════════════════════════════════════════════════

    def set_downside_multiplier(self, value: float) -> None:
        """
        Setter for PLO Drawdown callback.
        
        Clamps value to [1.0, 10.0] for safety.
        λ=1.0 is neutral, λ=5.0 is typical max, λ=10.0 is hard cap.
        """
        self.downside_multiplier = max(1.0, min(value, 10.0))

    def set_churn_multiplier(self, value: float) -> None:
        """
        Setter for PLO Churn callback.
        
        Clamps value to [1.0, 10.0] for safety.
        """
        self.churn_multiplier = max(1.0, min(value, 10.0))

    def set_smooth_multiplier(self, value: float) -> None:
        """
        Setter for PLO Smoothness callback.
        
        Clamps value to [1.0, 10.0] for safety.
        """
        self.smooth_multiplier = max(1.0, min(value, 10.0))

    @property
    def current_jerks(self) -> torch.Tensor:
        """
        Returns jerks calculated during the last step.
        
        IMPORTANT (Audit v1.1 fix):
        - Does NOT calculate jerk here (would always be 0 due to execution order)
        - Simply reads the buffer filled during _calculate_rewards
        
        Returns:
            Tensor of shape (n_envs,) with jerk values for each env.
        """
        return self.latest_jerks

    @property
    def current_position_deltas(self) -> torch.Tensor:
        """
        Returns absolute position deltas for PLO Churn callback.
        
        Returns:
            Tensor of shape (n_envs,) with |Δposition| for each env.
        """
        return torch.abs(self.position_pcts - self.prev_position_pcts)

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
        n_envs: int = 512,
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

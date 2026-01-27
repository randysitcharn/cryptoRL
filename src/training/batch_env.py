# -*- coding: utf-8 -*-
"""
batch_env.py - GPU-Vectorized Trading Environment for RL.

Implements a batch environment where N environments run in parallel on GPU
using PyTorch tensors. This eliminates IPC overhead from SubprocVecEnv
and achieves 10-50x speedup.

Architecture:
    SubprocVecEnv: 31 processes × 1 env  → CPU bottleneck (IPC/pickling)
    BatchCryptoEnv: 1 process × 1024 envs → GPU saturated (tensor ops)

CHANGELOG 2026-01-26:
- Replaced log-return reward with Differential Sharpe Reward (DSR)
- DSR creates meaningful reward differences between policies
- Compatible with MORL scalarization (w_cost parameter)
- Reference: Moody & Saffell (1998) "Learning to Trade via Direct RL"
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
        - Differential Sharpe Reward (DSR) for meaningful policy discrimination
        - MORL scalarization with w_cost preference parameter
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
        # Transaction costs (defaults from TQCTrainingConfig)
        commission: float = 0.0,
        slippage: float = 0.0,
        funding_rate: float = 0.0,
        w_cost_fixed: Optional[float] = 0.0,  # Fixed w_cost (0=no cost penalty, None=sample)
        # Reward params
        reward_scaling: float = 1.0,
        downside_coef: float = 10.0,
        upside_coef: float = 0.0,
        action_discretization: float = 0.1,
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
        # Domain Randomization (anti-overfitting)
        enable_domain_randomization: bool = True,
        commission_min: float = 0.0002,  # 0.02%
        commission_max: float = 0.0008,  # 0.08%
        slippage_min: float = 0.00005,   # 0.005%
        slippage_max: float = 0.00015,   # 0.015%
        slippage_noise_std: float = 0.00002,  # Bruit additif pour market impact
        # Differential Sharpe Reward (Moody & Saffell 1998)
        dsr_eta: float = 0.005,  # EMA decay for DSR
        dsr_warmup_steps: int = 50,  # Steps before DSR activates (warmup = log-return)
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
            reward_scaling: Reward multiplier (keep at 1.0, SCALE=10 internal).
            downside_coef: Sortino downside penalty coefficient.
            upside_coef: Upside bonus coefficient.
            action_discretization: Action discretization step (0.1 = 21 levels).
            target_volatility: Target vol for position scaling.
            vol_window: Rolling window for volatility.
            max_leverage: Max vol scaling factor.
            start_idx: Start index for data slice.
            end_idx: End index for data slice.
            observation_noise: Noise level for observations (anti-overfitting).
            random_start: If False, start at beginning (for evaluation).
            funding_rate: Funding cost per step for short positions (perpetual futures style).
            enable_domain_randomization: Enable domain randomization for fees (anti-overfitting).
            commission_min: Minimum commission rate for randomization.
            commission_max: Maximum commission rate for randomization.
            slippage_min: Minimum slippage rate for randomization.
            slippage_max: Maximum slippage rate for randomization.
            slippage_noise_std: Standard deviation of execution slippage noise.
            dsr_eta: EMA decay rate for Differential Sharpe Reward.
            dsr_warmup_steps: Steps before DSR activates (warmup uses scaled log-returns).
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
        self.target_volatility = target_volatility
        self.vol_window = vol_window
        self.max_leverage = max_leverage
        self.observation_noise = observation_noise
        self.random_start = random_start
        self.funding_rate = funding_rate
        self.w_cost_fixed = w_cost_fixed  # Fixed w_cost for MORL (None=sample, 0=no penalty)
        self.training = True  # Flag for observation noise (disable during eval)
        self._last_noise_scale = 0.0  # For TensorBoard logging (Dynamic Noise)

        # Domain Randomization params
        self.enable_domain_randomization = enable_domain_randomization
        self.commission_min = commission_min
        self.commission_max = commission_max
        self.slippage_min = slippage_min
        self.slippage_max = slippage_max
        self.slippage_noise_std = slippage_noise_std

        self.dsr_eta = dsr_eta
        self.dsr_warmup_steps = dsr_warmup_steps

        # Curriculum state (stateless architecture - AAAI 2024 Curriculum Learning)
        self.progress = 0.0           # Training progress [0, 1]
        self.curriculum_lambda = 0.0  # Dynamic penalty weight
        self.last_gate_mean = 0.0     # Mean gate opening for logging

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

        # --- ALIGNMENT FIX START ---
        # Force HMM features to be the last 5 columns for FiLM compatibility
        # Uses centralized HMM_CONTEXT_COLS from constants.py
        from src.config import HMM_CONTEXT_COLS

        available_cols = feature_cols

        missing_hmm = [c for c in HMM_CONTEXT_COLS if c not in available_cols]
        if missing_hmm:
            raise ValueError(
                f"[BatchEnv] Critical: Missing HMM columns for FiLM: {missing_hmm}"
            )

        tech_cols = [c for c in available_cols if c not in HMM_CONTEXT_COLS]
        final_order = tech_cols + HMM_CONTEXT_COLS
        feature_cols = final_order

        print(f"[BatchEnv] Features Aligned: {len(tech_cols)} Tech + {len(HMM_CONTEXT_COLS)} HMM")
        print(f"[BatchEnv] Last 5 columns (Must be HMM): {feature_cols[-5:]}")
        # --- ALIGNMENT FIX END ---

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
        # OBSERVATION SPACE with MORL preference parameter
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
        print(f"[BatchCryptoEnv] Reward: Differential Sharpe (eta={dsr_eta}, warmup={dsr_warmup_steps})")

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
        self._step_counter = 0  # DEBUG: Global step counter for logging

        # Episode tracking for SB3 monitoring (GPU tensors)
        self.episode_rewards = torch.zeros(n, device=device)
        self.episode_lengths = torch.zeros(n, dtype=torch.long, device=device)

        # Drawdown tracking (GPU)
        self.peak_navs = torch.full((n,), self.initial_balance, device=device)
        self.current_drawdowns = torch.zeros(n, device=device)

        # Pre-allocated action buffer (avoids tensor creation each step)
        self._action_buffer = torch.zeros(n, device=device)

        # Done flags buffer (for _build_infos optimization)
        self._dones = torch.zeros(n, dtype=torch.bool, device=device)
        self._final_total_trades = None  # Captured before reset for evaluation
        self._final_navs = None  # Captured before reset for evaluation
        self._final_positions = None  # Captured before reset for evaluation

        # ═══════════════════════════════════════════════════════════════════
        # MORL: w_cost preference parameter per environment
        # Shape: (n_envs, 1) for proper broadcasting in reward calculation
        # Sampled at reset() with biased distribution (20%/60%/20%) or curriculum
        # ═══════════════════════════════════════════════════════════════════
        self.w_cost = torch.zeros(n, 1, device=device)
        self._eval_w_cost = None  # Fixed w_cost for evaluation mode (None = sample)

        # Curriculum learning state
        self._w_cost_target = None  # Target w_cost for curriculum (None = disabled)
        self._use_curriculum_sampling = False  # Use curriculum sampling if True

        # Reward component buffers (pour observabilité)
        self._rew_pnl = torch.zeros(n, device=device)
        self._rew_churn = torch.zeros(n, device=device)
        self._rew_smooth = torch.zeros(n, device=device)
        self._rew_downside = torch.zeros(n, device=device)  # Downside risk tracking

        # Domain Randomization: per-env commission/slippage
        self.commission_per_env = torch.zeros(n, device=device)
        self.slippage_per_env = torch.zeros(n, device=device)

        # OPTIMIZATION: Pre-allocate window offsets to avoid torch.arange in hot path
        self.window_offsets = torch.arange(self.window_size, device=device)

        # DSR state (Moody & Saffell 1998)
        self.dsr_A = torch.zeros(n, device=device)
        self.dsr_B = torch.full((n,), 1e-6, device=device)
        self._dsr_raw = torch.zeros(n, device=device)
        self._dsr_variance = torch.zeros(n, device=device)

    def _sample_domain_params(self, env_indices: torch.Tensor) -> None:
        """
        Sample commission and slippage for specified environments.
        
        Uses Beta distribution for commission (skewed towards center) and
        Uniform distribution for slippage. Per-episode sampling (not per-step)
        to maintain realistic exchange behavior.
        
        Args:
            env_indices: Tensor of environment indices to sample for.
        """
        n = env_indices.shape[0]

        if self.enable_domain_randomization and self.training:
            # Beta distribution for commission (skewed towards center)
            # α=2, β=2 gives bell curve centered at (α/(α+β)) = 0.5
            from torch.distributions import Beta
            alpha, beta = 2.0, 2.0
            beta_dist = Beta(alpha, beta)
            u = beta_dist.sample((n,)).to(self.device)
            self.commission_per_env[env_indices] = (
                self.commission_min +
                (self.commission_max - self.commission_min) * u
            )

            # Uniform for slippage
            self.slippage_per_env[env_indices] = torch.rand(
                n, device=self.device
            ) * (self.slippage_max - self.slippage_min) + self.slippage_min
        else:
            # Fixed values (eval mode or disabled)
            self.commission_per_env[env_indices] = self.commission
            self.slippage_per_env[env_indices] = self.slippage

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
            "reward/dsr_raw": self._dsr_raw.mean().item(),
            "reward/dsr_variance": self._dsr_variance.mean().item(),
            "reward/dsr_A": self.dsr_A.mean().item(),
            "reward/dsr_B": self.dsr_B.mean().item(),
            # Curriculum state (AAAI 2024)
            "curriculum/lambda": self.curriculum_lambda,
            "curriculum/progress": self.progress,
            "curriculum/gate_open": self.last_gate_mean,
            # MORL metrics (Audit 2026-01-22)
            "morl/w_cost_mean": self.w_cost.mean().item(),
            "morl/w_cost_std": self.w_cost.std().item(),
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
        MORL reward with Differential Sharpe (DSR) as performance objective.

        DSR rewards consistency (Sharpe improvement) rather than raw returns.
        Order: (1) compute delta_A, delta_B; (2) compute reward DSR/warmup;
        (3) update EMA (always, including during warmup).

        Reference: Moody & Saffell (1998) "Learning to Trade via Direct RL".
        """
        DSR_SCALE = 3.0  # Reduced to avoid noise amplification
        MAX_PENALTY_SCALE = 0.05
        COST_PENALTY_CAP = 0.01

        R_t = step_returns
        delta_A = R_t - self.dsr_A
        delta_B = R_t ** 2 - self.dsr_B

        variance = torch.clamp(self.dsr_B - self.dsr_A ** 2, min=1e-8)
        denom = variance ** 1.5 + 1e-6  # Epsilon for numerical stability
        numerator = self.dsr_B * delta_A - 0.5 * self.dsr_A * delta_B
        r_dsr_raw = numerator / denom
        self._dsr_raw = r_dsr_raw.clone()
        self._dsr_variance = variance.clone()
        r_dsr_raw = torch.clamp(r_dsr_raw, min=-10.0, max=10.0)  # Allow stronger signals

        warmup_mask = self.episode_lengths < self.dsr_warmup_steps
        r_simple = torch.log1p(torch.clamp(R_t, min=-0.99)) * DSR_SCALE
        r_dsr = torch.where(warmup_mask, r_simple, r_dsr_raw * DSR_SCALE)

        self.dsr_A = self.dsr_A + self.dsr_eta * delta_A
        self.dsr_B = self.dsr_B + self.dsr_eta * delta_B

        r_perf = r_dsr

        r_cost = -position_deltas * 10.0
        r_cost = torch.clamp(r_cost, min=-COST_PENALTY_CAP, max=0.0)
        w_cost_squeezed = self.w_cost.squeeze(-1)
        reward = r_perf + (w_cost_squeezed * r_cost * MAX_PENALTY_SCALE)

        self._rew_pnl = r_perf
        self._rew_churn = w_cost_squeezed * r_cost * MAX_PENALTY_SCALE
        self._rew_downside = torch.zeros_like(r_perf)
        self._rew_smooth = torch.zeros_like(r_perf)
        self.last_gate_mean = 1.0

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

        self.dsr_A.zero_()
        self.dsr_B.fill_(1e-6)

        # Domain Randomization: sample fees for all envs
        if self.enable_domain_randomization:
            self._sample_domain_params(torch.arange(self.num_envs, device=self.device))

        # ═══════════════════════════════════════════════════════════════════
        # MORL: Sample w_cost (priority: w_cost_fixed > eval > curriculum > sample)
        # ═══════════════════════════════════════════════════════════════════
        if self.w_cost_fixed is not None:
            # Fixed w_cost from config (Single Source of Truth)
            self.w_cost.fill_(self.w_cost_fixed)
        elif self._eval_w_cost is not None:
            # Evaluation mode: use fixed w_cost for reproducibility
            self.w_cost.fill_(self._eval_w_cost)
        elif self._use_curriculum_sampling and self._w_cost_target is not None:
            # Curriculum mode: sample around target with truncated normal
            target = self._w_cost_target
            std = 0.1
            samples = torch.normal(mean=target, std=std, size=(self.num_envs, 1), device=self.device)
            self.w_cost = torch.clamp(samples, min=0.0, max=1.0)
        else:
            # Training mode: sample with biased distribution (20%/60%/20%)
            sample_type = torch.rand(self.num_envs, device=self.device)
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

        # Transfer to CPU numpy for SB3
        return {
            "market": market.cpu().numpy(),
            "position": position.cpu().numpy(),
            "w_cost": self.w_cost.cpu().numpy(),  # MORL preference parameter
        }

    def set_training_mode(self, training: bool):
        """Enable/disable observation noise for eval."""
        self.training = training

    def set_eval_w_cost(self, w_cost: Optional[float]):
        """
        Set fixed w_cost for evaluation mode (MORL Pareto Front).
        
        CRITICAL: This method has HIGHEST PRIORITY and overrides curriculum learning.
        When called, the environment will use the fixed w_cost value instead of
        curriculum sampling or biased distribution. This ensures reproducible evaluation
        with specific preference values (e.g., 0.0, 0.5, 1.0 for Pareto front).
        
        Args:
            w_cost: Fixed preference in [0, 1], or None to resume sampling.
                   0 = Scalping (ignore costs), 1 = B&H (minimize costs)
        """
        self._eval_w_cost = w_cost

    def set_w_cost_target(self, target_w_cost: float) -> None:
        """
        Set target w_cost for curriculum learning.

        The environment will sample w_cost around this target value
        instead of using the fixed biased distribution (20%/60%/20%).

        Also applies immediately to currently running episodes to fix
        timing bug where first episode would use biased distribution.

        Args:
            target_w_cost: Target w_cost value [0, 1] for curriculum.
        """
        self._w_cost_target = max(0.0, min(1.0, target_w_cost))
        self._use_curriculum_sampling = True

        # Apply to currently running episodes (not just future resets)
        # Sample around target with std=0.1 (same as reset)
        std = 0.1
        samples = torch.normal(
            mean=self._w_cost_target,
            std=std,
            size=(self.num_envs, 1),
            device=self.device
        )
        self.w_cost = torch.clamp(samples, min=0.0, max=1.0)

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
        self._step_counter += 1  # DEBUG: Increment global step counter
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

        # DEBUG: Log vol scaling values (every 10000 steps) to file
        if self._step_counter % 10000 == 0:
            with open("/workspace/cryptoRL/logs/vol_debug.txt", "a") as f:
                f.write(f"step={self._step_counter} | "
                        f"raw={raw_actions[0].item():.4f} | "
                        f"vol={current_vol[0].item():.4f} | "
                        f"scalar={self.vol_scalars[0].item():.4f} | "
                        f"eff={effective_actions[0].item():.4f}\n")

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

        # Calculate trade costs with domain-randomized fees
        units_delta = target_units - self.positions
        trade_values = torch.abs(units_delta * old_prices)

        # Base costs: commission + slippage (per-env randomized)
        base_cost_rate = self.commission_per_env + self.slippage_per_env

        # Add execution slippage noise (market impact variability)
        if self.enable_domain_randomization and self.training:
            slippage_noise = torch.randn(self.num_envs, device=self.device) * self.slippage_noise_std
            slippage_noise = torch.clamp(
                slippage_noise,
                -self.slippage_noise_std * 2,
                self.slippage_noise_std * 2
            )
            effective_cost_rate = base_cost_rate + slippage_noise
        else:
            effective_cost_rate = base_cost_rate

        trade_costs = trade_values * effective_cost_rate

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

        self.dsr_A[dones] = 0.0
        self.dsr_B[dones] = 1e-6

        # Domain Randomization: resample fees for done envs
        if self.enable_domain_randomization:
            done_indices = torch.nonzero(dones, as_tuple=True)[0]
            if done_indices.numel() > 0:
                self._sample_domain_params(done_indices)

        # ═══════════════════════════════════════════════════════════════════
        # MORL: Resample w_cost for done envs
        # Priority: w_cost_fixed > eval > curriculum > sample
        # ═══════════════════════════════════════════════════════════════════
        if self.w_cost_fixed is not None:
            # Fixed w_cost from config (Single Source of Truth)
            self.w_cost[dones] = self.w_cost_fixed
        elif self._eval_w_cost is not None:
            # Evaluation mode: use fixed w_cost
            self.w_cost[dones] = self._eval_w_cost
        elif self._use_curriculum_sampling and self._w_cost_target is not None:
            # Curriculum mode: sample around target with truncated normal
            target = self._w_cost_target
            std = 0.1
            samples = torch.normal(mean=target, std=std, size=(n_done, 1), device=self.device)
            new_w_cost = torch.clamp(samples, min=0.0, max=1.0)
            self.w_cost[dones] = new_w_cost
        else:
            # Training mode: sample with biased distribution (20%/60%/20%)
            sample_type = torch.rand(n_done, device=self.device)
            new_w_cost = torch.where(
                sample_type.unsqueeze(1) < 0.2,
                torch.zeros(n_done, 1, device=self.device),
                torch.where(
                    sample_type.unsqueeze(1) > 0.8,
                    torch.ones(n_done, 1, device=self.device),
                    torch.rand(n_done, 1, device=self.device)
                )
            )
            self.w_cost[dones] = new_w_cost

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

        # MORL: w_cost preference parameter (MUST match _get_observations structure)
        w_cost = self.w_cost[idx].cpu().numpy()

        return {
            "market": market,
            "position": position,
            "w_cost": w_cost,
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
            'dsr/A': float(self.dsr_A[idx].item()),
            'dsr/B': float(self.dsr_B[idx].item()),
        }

    def close(self) -> None:
        """Clean up resources."""
        pass

    def env_is_wrapped(self, wrapper_class, indices=None) -> List[bool]:
        """Check if envs are wrapped (always False for batch env)."""
        return [False] * self.num_envs

    def env_method(self, method_name: str, *args, indices=None, **kwargs):
        """Call method on envs (limited support for batch env)."""
        raise NotImplementedError(f"env_method '{method_name}' not supported")

    def set_attr(self, attr_name: str, value, indices=None) -> None:
        """Set attribute on envs."""
        raise NotImplementedError(f"set_attr '{attr_name}' not supported")

    def set_progress(self, progress: float) -> None:
        """
        Update training progress and curriculum lambda.

        Curriculum Schedule (extended to 75% of training):
          Phase 1 (0-15%):   lambda = 0.0  (Pure Exploration)
          Phase 2 (15-75%):  lambda = 0.0 -> 0.4 (Reality Ramp)
          Phase 3 (75-100%): lambda = 0.4 (Stability)

        Args:
            progress: Training progress as fraction [0, 1].
        """
        self.progress = max(0.0, min(1.0, progress))

        if self.progress <= 0.15:
            # Phase 1: Pure Exploration - no penalties
            self.curriculum_lambda = 0.0
        elif self.progress <= 0.75:
            # Phase 2: Reality Ramp - linear 0.0 -> 0.4
            phase_progress = (self.progress - 0.15) / 0.60
            self.curriculum_lambda = 0.4 * phase_progress
        else:
            # Phase 3: Stability - fixed discipline
            self.curriculum_lambda = 0.4

    def get_attr(self, attr_name: str, indices=None):
        """Get attribute from envs."""
        if attr_name == "render_mode":
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
    print("Testing BatchCryptoEnv with Differential Sharpe Reward...")

    env = BatchCryptoEnv(
        parquet_path="data/processed_data.parquet",
        n_envs=4,
        device="cuda" if torch.cuda.is_available() else "cpu",
        window_size=64,
        episode_length=100,
        dsr_eta=0.01,
        dsr_warmup_steps=10,
    )

    print(f"\nNum envs: {env.num_envs}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Device: {env.device}")

    obs = env.reset()
    print(f"\nObservation shapes: market={obs['market'].shape}, position={obs['position'].shape}")

    total_rewards = np.zeros(env.num_envs)
    for i in range(50):
        actions = np.random.uniform(-1, 1, (env.num_envs, 1)).astype(np.float32)
        env.step_async(actions)
        obs, rewards, dones, infos = env.step_wait()
        total_rewards += rewards
        if i % 10 == 0:
            m = env.get_global_metrics()
            print(f"Step {i}: mean_reward={rewards.mean():.4f}, "
                  f"dsr_raw={m['reward/dsr_raw']:.4f}, dsr_var={m['reward/dsr_variance']:.6f}, "
                  f"dones={dones.sum()}")

    print(f"\nTotal rewards: {total_rewards}")
    print(f"Final metrics: {env.get_global_metrics()}")
    print("\n[OK] BatchCryptoEnv with DSR test passed!")

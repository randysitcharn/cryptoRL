# -*- coding: utf-8 -*-
"""
rewards.py - SOTA Reward Wrappers for Trading Environments.

Implements "Thresholded Lexicographic" reward logic from SOTA literature (2024-2025)
to prevent Policy Collapse in TQC agents.

Based on:
- Dynamic scalarization where penalties increase drastically only when safety limits are breached
- Multi-objective reward with regime-dependent weights
"""

import numpy as np
import gymnasium as gym
from collections import deque
from typing import Dict, Optional, Any


class SotaRewardWrapper(gym.Wrapper):
    """
    SOTA Reward Wrapper implementing Thresholded Lexicographic logic.
    
    Prevents Policy Collapse by using dynamic scalarization where transaction cost
    penalties increase drastically only when turnover exceeds a target threshold.
    
    Mathematical Formula:
        R_t = w_ret * r_log + w_risk * r_sortino + w_inv * r_inv + w_cost * r_cost
    
    Where:
        - r_log: Log-returns of the portfolio
        - r_sortino: Differential Downside Ratio (returns / downside_deviation)
        - r_inv: Quadratic inventory penalty (mean-reversion)
        - r_cost: Transaction costs with soft thresholding based on rolling turnover
    
    Args:
        env: The environment to wrap (must be a single environment, not VecEnv).
        target_turnover: Target turnover threshold (default: 5.0).
                        If rolling turnover < target: w_cost = cost_low (low friction)
                        If rolling turnover >= target: w_cost = cost_high (heavy penalty)
        window_size: Rolling window size for downside deviation and turnover (default: 24).
        weights: Dictionary of weights for reward components:
            - 'ret': Weight for returns component (default: 1.0)
            - 'risk': Weight for Sortino risk component (default: 0.5)
            - 'inv': Weight for inventory penalty (default: -0.1)
            - 'cost_low': Weight for costs when turnover < target (default: -0.05)
            - 'cost_high': Weight for costs when turnover >= target (default: -0.5)
        eps: Small epsilon for numerical stability (default: 1e-6).
    
    Example:
        >>> env = BatchCryptoEnv(n_envs=1, ...)
        >>> env = SotaRewardWrapper(env, target_turnover=5.0, window_size=24)
        >>> obs, info = env.reset()
        >>> obs, reward, terminated, truncated, info = env.step(action)
        >>> # info contains: 'reward/risk_component', 'reward/cost_penalty', 'metric/rolling_turnover'
    """
    
    def __init__(
        self,
        env: gym.Env,
        target_turnover: float = 5.0,
        window_size: int = 24,
        weights: Optional[Dict[str, float]] = None,
        eps: float = 1e-6,
    ):
        super().__init__(env)
        
        self.target_turnover = target_turnover
        self.window_size = window_size
        self.eps = eps
        
        # Default weights
        default_weights = {
            'ret': 1.0,
            'risk': 0.5,
            'inv': -0.1,
            'cost_low': -0.05,
            'cost_high': -0.5,
        }
        
        if weights is None:
            weights = default_weights
        else:
            # Merge with defaults
            for key in default_weights:
                if key not in weights:
                    weights[key] = default_weights[key]
        
        self.weights = weights
        
        # Buffers for rolling calculations (per-environment state)
        # Using deque for efficient rolling window operations
        self.returns_buffer = deque(maxlen=window_size)
        self.turnover_buffer = deque(maxlen=window_size)
        
        # Current position for inventory penalty
        self.current_position = 0.0
        
        # Track previous NAV for return calculation
        self.prev_nav = None
    
    def reset(self, **kwargs):
        """Reset wrapper state on environment reset."""
        # Clear buffers
        self.returns_buffer.clear()
        self.turnover_buffer.clear()
        self.current_position = 0.0
        self.prev_nav = None
        
        # Reset underlying environment
        obs, info = self.env.reset(**kwargs)
        
        # Initialize prev_nav from info if available
        if info is not None and 'nav' in info:
            self.prev_nav = info['nav']
        elif hasattr(self.env, 'initial_balance'):
            self.prev_nav = self.env.initial_balance
        else:
            self.prev_nav = 10000.0  # Default fallback
        
        return obs, info
    
    def step(self, action):
        """
        Execute step with SOTA reward transformation.
        
        Modifies the reward while preserving observation space.
        Adds metrics to info dict for TensorBoard tracking.
        """
        # Execute underlying environment step
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Extract current state from info
        current_nav = info.get('nav', self.prev_nav if self.prev_nav is not None else 10000.0)
        current_position = info.get('position_pct', 0.0)
        
        # Calculate step return (log-return)
        if self.prev_nav is not None and self.prev_nav > 0:
            step_return = (current_nav - self.prev_nav) / self.prev_nav
            # Clamp to prevent extreme values
            step_return = np.clip(step_return, -0.99, np.inf)
            log_return = np.log1p(step_return)
        else:
            log_return = 0.0
            step_return = 0.0
        
        # Calculate position delta (turnover)
        position_delta = abs(current_position - self.current_position)
        
        # Update buffers
        self.returns_buffer.append(step_return)
        self.turnover_buffer.append(position_delta)
        
        # Calculate rolling turnover (average over window)
        if len(self.turnover_buffer) > 0:
            rolling_turnover = np.mean(list(self.turnover_buffer)) * self.window_size
        else:
            rolling_turnover = 0.0
        
        # ═══════════════════════════════════════════════════════════════════
        # 1. RETURNS COMPONENT (r_log)
        # ═══════════════════════════════════════════════════════════════════
        r_log = log_return
        
        # ═══════════════════════════════════════════════════════════════════
        # 2. RISK COMPONENT (r_sortino) - Differential Downside Ratio
        # ═══════════════════════════════════════════════════════════════════
        # Calculate downside deviation (only negative returns)
        if len(self.returns_buffer) >= 2:
            returns_array = np.array(list(self.returns_buffer))
            downside_returns = returns_array[returns_array < 0]
            
            if len(downside_returns) > 0:
                downside_deviation = np.std(downside_returns)
            else:
                downside_deviation = self.eps
            
            # Sortino ratio: log_return / downside_deviation
            # Formula from spec: r_sortino = (r_log) / (σ_d + ε)
            r_sortino = log_return / (downside_deviation + self.eps)
        else:
            # Not enough data for Sortino calculation
            r_sortino = 0.0
            downside_deviation = self.eps
        
        # ═══════════════════════════════════════════════════════════════════
        # 3. INVENTORY COMPONENT (r_inv) - Quadratic penalty for mean-reversion
        # ═══════════════════════════════════════════════════════════════════
        r_inv = -(current_position ** 2)
        
        # ═══════════════════════════════════════════════════════════════════
        # 4. TRANSACTION COSTS COMPONENT (r_cost) - Soft Thresholding
        # ═══════════════════════════════════════════════════════════════════
        # Base cost is proportional to position delta
        r_cost_base = -position_delta
        
        # Dynamic weight based on rolling turnover threshold
        if rolling_turnover < self.target_turnover:
            # Low friction sandbox mode
            w_cost = self.weights['cost_low']
        else:
            # Heavy penalty mode (safety limit breached)
            w_cost = self.weights['cost_high']
        
        r_cost = w_cost * r_cost_base
        
        # ═══════════════════════════════════════════════════════════════════
        # 5. TOTAL REWARD - Dynamic Scalarization
        # ═══════════════════════════════════════════════════════════════════
        total_reward = (
            self.weights['ret'] * r_log +
            self.weights['risk'] * r_sortino +
            self.weights['inv'] * r_inv +
            r_cost  # Already weighted
        )
        
        # ═══════════════════════════════════════════════════════════════════
        # 6. UPDATE STATE
        # ═══════════════════════════════════════════════════════════════════
        self.current_position = current_position
        self.prev_nav = current_nav
        
        # ═══════════════════════════════════════════════════════════════════
        # 7. ADD METRICS TO INFO DICT (TensorBoard tracking)
        # ═══════════════════════════════════════════════════════════════════
        if info is None:
            info = {}
        
        info['reward/risk_component'] = float(self.weights['risk'] * r_sortino)
        info['reward/cost_penalty'] = float(r_cost)
        info['reward/inventory_penalty'] = float(self.weights['inv'] * r_inv)
        info['reward/returns_component'] = float(self.weights['ret'] * r_log)
        info['metric/rolling_turnover'] = float(rolling_turnover)
        info['metric/target_turnover'] = float(self.target_turnover)
        info['metric/cost_weight_active'] = float(w_cost)
        info['metric/downside_deviation'] = float(downside_deviation)
        
        return obs, total_reward, terminated, truncated, info

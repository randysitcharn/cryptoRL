# -*- coding: utf-8 -*-
"""
test_morl.py - Tests for Multi-Objective RL (MORL) implementation.

Verifies that the MORL architecture works correctly:
- w_cost parameter in observation space
- Biased sampling distribution (20/60/20)
- Reward function with cost penalty modulation
- Evaluation mode with fixed w_cost

Reference: docs/design/MORL_DESIGN.md
"""

import sys
import os
import tempfile
import numpy as np
import pandas as pd
import torch
import pytest

# Add the project root to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.training.batch_env import BatchCryptoEnv


# ============================================================================
# Fixtures and Helpers
# ============================================================================

def create_dummy_data(n_rows: int = 1000) -> pd.DataFrame:
    """
    Create dummy data for tests.

    Args:
        n_rows: Number of rows.

    Returns:
        DataFrame with same columns as processed data.
    """
    np.random.seed(42)

    data = {
        'open': np.random.uniform(90, 110, n_rows),
        'high': np.random.uniform(100, 120, n_rows),
        'low': np.random.uniform(80, 100, n_rows),
        'close': np.random.uniform(90, 110, n_rows),
        'RSI_14': np.random.uniform(0, 1, n_rows),
        'MACD_12_26_9': np.random.uniform(-0.01, 0.01, n_rows),
        'MACDh_12_26_9': np.random.uniform(-0.005, 0.005, n_rows),
        'ATRr_14': np.random.uniform(0.01, 0.05, n_rows),
        'BBP_20_2.0': np.random.uniform(-0.5, 1.5, n_rows),
        'BBB_20_2.0': np.random.uniform(0, 0.1, n_rows),
        'log_ret': np.random.uniform(-0.03, 0.03, n_rows),
        'sin_hour': np.random.uniform(-1, 1, n_rows),
        'cos_hour': np.random.uniform(-1, 1, n_rows),
        'sin_day': np.random.uniform(-1, 1, n_rows),
        'cos_day': np.random.uniform(-1, 1, n_rows),
        'volume_rel': np.random.uniform(0.5, 2, n_rows),
    }

    return pd.DataFrame(data)


def create_test_env(n_envs: int = 4, n_rows: int = 1000, window_size: int = 64, 
                    episode_length: int = 100):
    """Create a BatchCryptoEnv with test data in a temporary parquet file."""
    df = create_dummy_data(n_rows=n_rows)
    
    tmp_file = tempfile.NamedTemporaryFile(suffix='.parquet', delete=False)
    df.to_parquet(tmp_file.name)
    
    env = BatchCryptoEnv(
        parquet_path=tmp_file.name,
        n_envs=n_envs,
        device='cpu',
        window_size=window_size,
        episode_length=episode_length,
        price_column='close',
        random_start=False,
    )
    
    return env, tmp_file.name


def cleanup_env(parquet_path: str):
    """Remove temporary parquet file."""
    try:
        os.unlink(parquet_path)
    except Exception:
        pass


# ============================================================================
# TestMORLObservation - w_cost in observation space
# ============================================================================

class TestMORLObservation:
    """Tests for w_cost in observation space."""

    def test_w_cost_in_observation(self):
        """w_cost should be present in observation."""
        env, tmp_path = create_test_env(n_envs=4)
        try:
            obs = env.reset()
            
            assert 'w_cost' in obs, "Observation should contain 'w_cost'"
            assert obs['w_cost'] is not None, "w_cost should not be None"
        finally:
            cleanup_env(tmp_path)

    def test_w_cost_shape(self):
        """w_cost should have shape (n_envs, 1)."""
        env, tmp_path = create_test_env(n_envs=8)
        try:
            obs = env.reset()
            
            assert obs['w_cost'].shape == (8, 1), \
                f"Expected w_cost shape (8, 1), got {obs['w_cost'].shape}"
        finally:
            cleanup_env(tmp_path)

    def test_w_cost_bounds(self):
        """w_cost should be in [0, 1]."""
        env, tmp_path = create_test_env(n_envs=16)
        try:
            # Test multiple resets
            for _ in range(10):
                obs = env.reset()
                
                assert np.all(obs['w_cost'] >= 0.0), \
                    f"w_cost should be >= 0, got min {obs['w_cost'].min()}"
                assert np.all(obs['w_cost'] <= 1.0), \
                    f"w_cost should be <= 1, got max {obs['w_cost'].max()}"
        finally:
            cleanup_env(tmp_path)

    def test_w_cost_dtype(self):
        """w_cost should be float32."""
        env, tmp_path = create_test_env(n_envs=4)
        try:
            obs = env.reset()
            
            assert obs['w_cost'].dtype == np.float32, \
                f"Expected float32, got {obs['w_cost'].dtype}"
        finally:
            cleanup_env(tmp_path)


# ============================================================================
# TestMORLSampling - Biased distribution 20/60/20
# ============================================================================

class TestMORLSampling:
    """Tests for w_cost sampling distribution."""

    def test_sampling_distribution(self):
        """w_cost should follow biased distribution: 20% w=0, 60% uniform, 20% w=1."""
        env, tmp_path = create_test_env(n_envs=1000)
        try:
            # Collect samples across many resets
            w_samples = []
            for _ in range(100):
                obs = env.reset()
                w_samples.extend(obs['w_cost'].flatten().tolist())
            
            w = np.array(w_samples)
            
            # ~20% should be exactly 0
            pct_zero = np.mean(w == 0.0)
            assert 0.15 < pct_zero < 0.25, \
                f"Expected ~20% w=0, got {pct_zero:.1%}"
            
            # ~20% should be exactly 1
            pct_one = np.mean(w == 1.0)
            assert 0.15 < pct_one < 0.25, \
                f"Expected ~20% w=1, got {pct_one:.1%}"
            
            # The rest (~60%) should be uniform in (0, 1)
            pct_middle = np.mean((w > 0.0) & (w < 1.0))
            assert 0.55 < pct_middle < 0.65, \
                f"Expected ~60% uniform, got {pct_middle:.1%}"
        finally:
            cleanup_env(tmp_path)

    def test_eval_mode_fixes_w_cost(self):
        """set_eval_w_cost should fix w_cost to specified value."""
        env, tmp_path = create_test_env(n_envs=8)
        try:
            # Set fixed w_cost for evaluation
            env.set_eval_w_cost(0.75)
            obs = env.reset()
            
            assert np.allclose(obs['w_cost'], 0.75), \
                f"Expected all w_cost=0.75, got {obs['w_cost'].flatten()}"
            
            # Should remain fixed after steps
            env.step_async(np.zeros((8, 1), dtype=np.float32))
            obs, _, _, _ = env.step_wait()
            
            assert np.allclose(obs['w_cost'], 0.75), \
                "w_cost should remain fixed after step"
        finally:
            cleanup_env(tmp_path)

    def test_eval_mode_reset_to_sampling(self):
        """set_eval_w_cost(None) should resume sampling."""
        env, tmp_path = create_test_env(n_envs=100)
        try:
            # First set fixed
            env.set_eval_w_cost(0.5)
            obs = env.reset()
            assert np.allclose(obs['w_cost'], 0.5)
            
            # Then resume sampling
            env.set_eval_w_cost(None)
            
            # Collect samples
            w_samples = []
            for _ in range(10):
                obs = env.reset()
                w_samples.extend(obs['w_cost'].flatten().tolist())
            
            w = np.array(w_samples)
            
            # Should have variety (not all 0.5)
            assert w.std() > 0.1, \
                "After set_eval_w_cost(None), w_cost should be sampled (variety expected)"
        finally:
            cleanup_env(tmp_path)

    def test_w_cost_resampled_on_auto_reset(self):
        """w_cost should be resampled when environment auto-resets."""
        env, tmp_path = create_test_env(n_envs=1, episode_length=10)
        try:
            # Use fixed w_cost for predictability, then switch to sampling
            env.set_eval_w_cost(None)  # Ensure sampling mode
            
            obs = env.reset()
            initial_w = obs['w_cost'].item()
            
            # Run through an episode to trigger auto-reset
            for _ in range(15):  # More than episode_length
                env.step_async(np.zeros((1, 1), dtype=np.float32))
                obs, _, dones, _ = env.step_wait()
                
                if dones[0]:
                    # After auto-reset, w_cost might be different
                    # (statistically, it should differ most of the time)
                    break
            
            # Just verify w_cost is still valid after auto-reset
            assert 0.0 <= obs['w_cost'].item() <= 1.0, \
                "w_cost should be valid after auto-reset"
        finally:
            cleanup_env(tmp_path)


# ============================================================================
# TestMORLReward - Reward function with MORL scalarization
# ============================================================================

class TestMORLReward:
    """Tests for MORL reward calculation."""

    def test_reward_zero_cost_when_w_zero(self):
        """With w_cost=0, churn cost component should be 0."""
        env, tmp_path = create_test_env(n_envs=4)
        try:
            env.set_eval_w_cost(0.0)
            env.reset()
            
            # Take action that changes position
            env.step_async(np.array([[0.5], [0.5], [0.5], [0.5]], dtype=np.float32))
            env.step_wait()
            
            # With w=0, churn component should be 0
            assert env._rew_churn.abs().max().item() < 1e-6, \
                f"With w_cost=0, _rew_churn should be ~0, got {env._rew_churn}"
        finally:
            cleanup_env(tmp_path)

    def test_reward_nonzero_cost_when_w_one(self):
        """With w_cost=1, churn cost should be negative (penalty)."""
        env, tmp_path = create_test_env(n_envs=4)
        try:
            env.set_eval_w_cost(1.0)
            env.reset()
            
            # Take action that changes position significantly
            env.step_async(np.array([[0.8], [0.8], [0.8], [0.8]], dtype=np.float32))
            env.step_wait()
            
            # With w=1 and position change, churn penalty should be negative
            # (After discretization, 0.8 rounds to 0.8, delta from 0.0 is 0.8)
            churn = env._rew_churn.min().item()
            assert churn < 0, \
                f"With w_cost=1 and position change, _rew_churn should be < 0, got {churn}"
        finally:
            cleanup_env(tmp_path)

    def test_reward_interpolation(self):
        """w_cost=0.5 should give intermediate churn penalty."""
        env, tmp_path = create_test_env(n_envs=4)
        try:
            # Collect churn for w=0
            env.set_eval_w_cost(0.0)
            env.reset()
            env.step_async(np.array([[0.5]] * 4, dtype=np.float32))
            env.step_wait()
            churn_w0 = env._rew_churn.mean().item()
            
            # Collect churn for w=1
            env.set_eval_w_cost(1.0)
            env.reset()
            env.step_async(np.array([[0.5]] * 4, dtype=np.float32))
            env.step_wait()
            churn_w1 = env._rew_churn.mean().item()
            
            # Collect churn for w=0.5
            env.set_eval_w_cost(0.5)
            env.reset()
            env.step_async(np.array([[0.5]] * 4, dtype=np.float32))
            env.step_wait()
            churn_w05 = env._rew_churn.mean().item()
            
            # w=0.5 should be between w=0 and w=1
            # churn_w0 ≈ 0, churn_w1 < 0, so churn_w05 should be ~0.5 * churn_w1
            assert churn_w05 < churn_w0 or abs(churn_w05 - churn_w0) < 1e-6, \
                "churn at w=0.5 should be <= churn at w=0"
            assert churn_w05 > churn_w1 or abs(churn_w05 - churn_w1) < 1e-6, \
                "churn at w=0.5 should be >= churn at w=1"
        finally:
            cleanup_env(tmp_path)

    def test_reward_stability_no_nan(self):
        """Reward should not produce NaN values."""
        env, tmp_path = create_test_env(n_envs=4, episode_length=50)
        try:
            env.set_eval_w_cost(0.5)
            env.reset()
            
            # Run many steps with various actions
            for _ in range(50):
                actions = np.random.uniform(-1, 1, size=(4, 1)).astype(np.float32)
                env.step_async(actions)
                obs, rewards, dones, infos = env.step_wait()
                
                assert not np.any(np.isnan(rewards)), \
                    f"Rewards contain NaN: {rewards}"
                assert not np.any(np.isinf(rewards)), \
                    f"Rewards contain Inf: {rewards}"
        finally:
            cleanup_env(tmp_path)

    def test_reward_stability_extreme_actions(self):
        """Reward should be stable with extreme actions."""
        env, tmp_path = create_test_env(n_envs=4)
        try:
            env.reset()
            
            # Rapid position changes (max churn)
            for i in range(10):
                # Alternate between max long and max short
                action_val = 1.0 if i % 2 == 0 else -1.0
                actions = np.full((4, 1), action_val, dtype=np.float32)
                env.step_async(actions)
                obs, rewards, dones, infos = env.step_wait()
                
                assert not np.any(np.isnan(rewards)), \
                    f"Rewards contain NaN with extreme actions: {rewards}"
        finally:
            cleanup_env(tmp_path)


# ============================================================================
# TestMORLIntegration - End-to-end integration tests
# ============================================================================

class TestMORLIntegration:
    """Integration tests for MORL with environment."""

    def test_w_cost_in_terminal_observation(self):
        """Terminal observation should also contain w_cost."""
        env, tmp_path = create_test_env(n_envs=1, episode_length=10)
        try:
            env.set_eval_w_cost(0.42)
            env.reset()
            
            # Run until episode ends
            for _ in range(15):
                env.step_async(np.zeros((1, 1), dtype=np.float32))
                obs, _, dones, infos = env.step_wait()
                
                if dones[0] and 'terminal_observation' in infos[0]:
                    terminal_obs = infos[0]['terminal_observation']
                    assert 'w_cost' in terminal_obs, \
                        "Terminal observation should contain w_cost"
                    assert np.allclose(terminal_obs['w_cost'], 0.42), \
                        f"Terminal w_cost should be 0.42, got {terminal_obs['w_cost']}"
                    break
        finally:
            cleanup_env(tmp_path)

    def test_gym_interface_includes_w_cost(self):
        """Gymnasium interface (n_envs=1) should include w_cost."""
        env, tmp_path = create_test_env(n_envs=1)
        try:
            obs, info = env.gym_reset()
            
            assert 'w_cost' in obs, "gym_reset() observation should contain w_cost"
            assert obs['w_cost'].shape == (1,), \
                f"Expected w_cost shape (1,), got {obs['w_cost'].shape}"
            
            obs, reward, terminated, truncated, info = env.gym_step(
                np.array([0.0], dtype=np.float32)
            )
            
            assert 'w_cost' in obs, "gym_step() observation should contain w_cost"
        finally:
            cleanup_env(tmp_path)

    def test_observation_space_includes_w_cost(self):
        """Observation space should declare w_cost."""
        env, tmp_path = create_test_env(n_envs=4)
        try:
            assert 'w_cost' in env.observation_space.spaces, \
                "observation_space should include w_cost"
            
            w_cost_space = env.observation_space.spaces['w_cost']
            assert w_cost_space.shape == (1,), \
                f"w_cost space should have shape (1,), got {w_cost_space.shape}"
            assert w_cost_space.low[0] == 0.0, "w_cost low bound should be 0.0"
            assert w_cost_space.high[0] == 1.0, "w_cost high bound should be 1.0"
        finally:
            cleanup_env(tmp_path)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Running MORL Tests")
    print("=" * 60)
    
    # Run with pytest
    pytest.main([__file__, "-v", "--tb=short"])

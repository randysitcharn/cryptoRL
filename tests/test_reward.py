# -*- coding: utf-8 -*-
"""
test_reward.py - Tests for BatchCryptoEnv reward function.

Verifies basic reward behavior: positive returns give positive rewards,
negative returns give negative rewards, NAV tracking works correctly.
"""

import sys
import os
import tempfile
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.training.batch_env import BatchCryptoEnv


def create_price_series(prices: list) -> pd.DataFrame:
    """Create test data with specified price series."""
    n_rows = len(prices)
    data = {
        'open': np.array(prices),
        'high': np.array(prices),
        'low': np.array(prices),
        'close': np.array(prices),
        'RSI_14': np.full(n_rows, 0.5),
        'MACD_12_26_9': np.zeros(n_rows),
        'MACDh_12_26_9': np.zeros(n_rows),
        'ATRr_14': np.full(n_rows, 0.02),
        'BBP_20_2.0': np.full(n_rows, 0.5),
        'BBB_20_2.0': np.full(n_rows, 0.05),
        'log_ret': np.zeros(n_rows),
        'sin_hour': np.zeros(n_rows),
        'cos_hour': np.ones(n_rows),
        'sin_day': np.zeros(n_rows),
        'cos_day': np.ones(n_rows),
        'volume_rel': np.ones(n_rows),
        # HMM features (required for FiLM alignment in BatchCryptoEnv)
        'HMM_Prob_0': np.full(n_rows, 0.25),
        'HMM_Prob_1': np.full(n_rows, 0.25),
        'HMM_Prob_2': np.full(n_rows, 0.25),
        'HMM_Prob_3': np.full(n_rows, 0.25),
        'HMM_Entropy': np.full(n_rows, 0.5),
    }
    return pd.DataFrame(data)


def create_test_env_with_prices(prices: list, window_size: int = 10, episode_length=None, **kwargs):
    """Create BatchCryptoEnv with specific price series."""
    df = create_price_series(prices)
    tmp_file = tempfile.NamedTemporaryFile(suffix='.parquet', delete=False)
    df.to_parquet(tmp_file.name)
    ep_len = episode_length if episode_length is not None else (len(prices) - window_size - 1)
    env = BatchCryptoEnv(
        parquet_path=tmp_file.name,
        n_envs=1,
        device='cpu',
        window_size=window_size,
        episode_length=ep_len,
        price_column='close',
        random_start=False,
        **kwargs,
    )
    return env, tmp_file.name


def run_policy(env, position: float, steps: int) -> np.ndarray:
    """Run env for `steps` steps with constant action [position], return rewards array."""
    env.gym_reset()
    rewards = []
    for _ in range(steps):
        obs, reward, term, trunc, _ = env.gym_step(np.array([position], dtype=np.float32))
        rewards.append(reward)
        if term or trunc:
            break
    return np.array(rewards)


def cleanup_env(parquet_path: str):
    """Remove temporary parquet file."""
    try:
        os.unlink(parquet_path)
    except Exception:
        pass


def test_reward_positive_return():
    """Test: Price increase should give positive reward."""
    print("Test 1: Positive Return Reward...")

    # Price goes up: 100 -> 110 (10% increase)
    prices = [100.0] * 15 + [110.0] * 10
    env, tmp_path = create_test_env_with_prices(prices)

    try:
        obs, info = env.gym_reset()

        # Buy at step 0 (price=100)
        action = np.array([1.0], dtype=np.float32)
        obs, reward1, _, _, info = env.gym_step(action)

        # Hold until price jumps to 110
        reward_gain = 0.0
        for _ in range(10):
            obs, reward, _, _, info = env.gym_step(action)
            if reward > reward_gain:
                reward_gain = reward

        # Reward should be positive when price increases
        assert reward_gain > 0, f"Expected positive reward for price increase, got {reward_gain}"

        print(f"  Reward on price increase: {reward_gain:.6f}")
        print("  PASSED!")
    finally:
        cleanup_env(tmp_path)


def test_reward_negative_return():
    """Test: Price decrease should give negative reward."""
    print("\nTest 2: Negative Return Reward...")

    # Price goes down: 100 -> 90 (10% decrease)
    prices = [100.0] * 15 + [90.0] * 10
    env, tmp_path = create_test_env_with_prices(prices)

    try:
        obs, info = env.gym_reset()

        # Buy at step 0
        action = np.array([1.0], dtype=np.float32)
        obs, reward1, _, _, info = env.gym_step(action)

        # Hold until price drops to 90 (capture minimum reward)
        reward_loss = 0.0
        for _ in range(10):
            obs, reward, _, _, info = env.gym_step(action)
            if reward < reward_loss:
                reward_loss = reward

        # Reward should be negative when price decreases
        assert reward_loss < 0, f"Expected negative reward for price decrease, got {reward_loss}"

        print(f"  Reward on price decrease: {reward_loss:.6f}")
        print("  PASSED!")
    finally:
        cleanup_env(tmp_path)


def test_nav_tracking():
    """Test: NAV should track portfolio value correctly through price changes."""
    print("\nTest 3: NAV Tracking...")

    # Price: 100 -> 120 (increase) -> 100 (decrease back)
    prices = [100.0] * 12 + [120.0] * 8 + [100.0] * 8
    env, tmp_path = create_test_env_with_prices(prices)

    try:
        obs, info = env.gym_reset()
        initial_nav = info['nav']

        # Buy and hold
        action = np.array([1.0], dtype=np.float32)

        # Step through to price=120
        for _ in range(8):
            obs, _, _, _, info = env.gym_step(action)

        peak_nav = info['nav']

        # Step through to price=100 (drop)
        for _ in range(6):
            obs, _, _, _, info = env.gym_step(action)

        final_nav = info['nav']

        # NAV should have increased at peak (120 price)
        assert peak_nav > initial_nav, \
            f"Expected NAV to increase at peak, got {initial_nav:.2f} -> {peak_nav:.2f}"

        # NAV should have decreased from peak when price drops
        assert final_nav < peak_nav, \
            f"Expected NAV to decrease from peak, got {peak_nav:.2f} -> {final_nav:.2f}"

        print(f"  Initial NAV: ${initial_nav:.2f}")
        print(f"  Peak NAV: ${peak_nav:.2f}")
        print(f"  Final NAV: ${final_nav:.2f}")
        print("  PASSED!")
    finally:
        cleanup_env(tmp_path)


def test_dsr_policy_discrimination():
    """DSR must discriminate between policies (fixed +0.05 vs B&H +1.0); difference > 0.1.
    TDD: run before DSR impl → fail with log-return; after DSR → pass."""
    print("\nTest: DSR Policy Discrimination...")
    # Mild upward drift + noise so both policies get returns but B&H gets more
    np.random.seed(42)
    n = 200
    prices = 100.0 + np.cumsum(np.random.randn(n) * 0.5 + 0.05).astype(float)
    prices = np.clip(prices, 80.0, 150.0)
    prices = prices.tolist()
    window_size = 10
    episode_length = 150
    steps = 100
    env, tmp_path = create_test_env_with_prices(
        prices,
        window_size=window_size,
        episode_length=episode_length,
        dsr_eta=0.005,
        dsr_warmup_steps=10,
    )
    try:
        env.set_eval_w_cost(0.0)
        rewards_fixed = run_policy(env, 0.05, steps)
        env.gym_reset()
        rewards_bh = run_policy(env, 1.0, steps)
        diff = abs(rewards_fixed.mean() - rewards_bh.mean())
        assert diff > 0.1, (
            f"DSR should discriminate policies: |mean(+0.05)-mean(+1.0)|={diff:.4f} <= 0.1"
        )
        print(f"  |mean(+0.05)-mean(+1.0)|={diff:.4f} > 0.1  PASSED")
    finally:
        cleanup_env(tmp_path)


def test_dsr_global_metrics():
    """Test: get_global_metrics() contains DSR keys after reset and steps."""
    print("\nTest: DSR get_global_metrics...")
    prices = [100.0] * 80
    env, tmp_path = create_test_env_with_prices(
        prices,
        window_size=10,
        episode_length=60,
        dsr_eta=0.01,
        dsr_warmup_steps=5,
    )
    try:
        env.gym_reset()
        for _ in range(10):
            env.gym_step(np.array([0.5], dtype=np.float32))
        m = env.get_global_metrics()
        for key in ("reward/dsr_raw", "reward/dsr_A", "reward/dsr_B"):
            assert key in m, f"Missing DSR key {key} in get_global_metrics"
        print(f"  {list(m.keys())} includes DSR keys  PASSED")
    finally:
        cleanup_env(tmp_path)


def test_info_contains_metrics():
    """Test: Info dict should contain monitoring metrics."""
    print("\nTest 4: Info Contains Metrics...")

    prices = [100.0] * 30
    env, tmp_path = create_test_env_with_prices(prices)

    try:
        obs, info_reset = env.gym_reset()

        # Check reset info
        required_keys = ['nav', 'cash', 'position', 'position_pct', 'price',
                        'total_trades', 'total_commission']
        for key in required_keys:
            assert key in info_reset, f"Missing key '{key}' in reset info"

        # Do a step and check step info
        action = np.array([0.5], dtype=np.float32)
        obs, reward, _, _, info_step = env.gym_step(action)

        # After step, should have same metrics
        for key in required_keys:
            assert key in info_step, f"Missing key '{key}' in step info"

        print(f"  Reset info keys: {list(info_reset.keys())}")
        print(f"  NAV: ${info_step['nav']:.2f}")
        print("  PASSED!")
    finally:
        cleanup_env(tmp_path)


if __name__ == "__main__":
    print("=" * 50)
    print("Testing BatchCryptoEnv Reward Function")
    print("=" * 50)

    test_reward_positive_return()
    test_reward_negative_return()
    test_nav_tracking()
    test_dsr_policy_discrimination()
    test_dsr_global_metrics()
    test_info_contains_metrics()

    print("\n" + "=" * 50)
    print("[OK] All reward function tests passed!")
    print("=" * 50)

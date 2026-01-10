"""
test_reward.py - Tests for risk-adjusted reward function.

Verifies that CryptoTradingEnv correctly calculates Sortino-proxy
rewards with log returns, downside deviation, and drawdown tracking.
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.training.env import CryptoTradingEnv


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
    }
    return pd.DataFrame(data)


def test_reward_positive_return():
    """Test: Price increase should give positive reward."""
    print("Test 1: Positive Return Reward...")

    # Price goes up: 100 -> 110 (10% increase = ~9.5% log return)
    # With window_size=2, reset starts at step 2, so price changes at step 10 (index 10)
    prices = [100.0] * 10 + [110.0] * 10
    df = create_price_series(prices)
    env = CryptoTradingEnv(df, initial_balance=10000.0, reward_scaling=10.0, window_size=2)

    obs, info = env.reset()

    # Buy at step 0 (price=100)
    action = np.array([1.0], dtype=np.float32)
    obs, reward1, _, _, info = env.step(action)

    # Hold until price jumps to 110 (happens when current_step reaches index 10)
    # With window_size=2, we start at step 2. After step 1: current_step=3
    # We need current_step to reach 10, so 10-3=7 more steps
    reward_gain = 0.0
    for _ in range(7):
        obs, reward, _, _, info = env.step(action)
        if reward > reward_gain:
            reward_gain = reward  # Capture max reward (price increase step)

    # Reward should be positive when price increases
    assert reward_gain > 0, f"Expected positive reward for price increase, got {reward_gain}"

    print(f"  Reward on price increase: {reward_gain:.6f}")
    print("  PASSED!")


def test_reward_negative_return():
    """Test: Price decrease should give negative reward."""
    print("\nTest 2: Negative Return Reward...")

    # Price goes down: 100 -> 90 (10% decrease = ~-10.5% log return)
    prices = [100.0] * 10 + [90.0] * 10
    df = create_price_series(prices)
    env = CryptoTradingEnv(df, initial_balance=10000.0, reward_scaling=10.0, window_size=2)

    obs, info = env.reset()

    # Buy at step 0
    action = np.array([1.0], dtype=np.float32)
    obs, reward1, _, _, info = env.step(action)

    # Hold until price drops to 90 (capture minimum reward)
    reward_loss = 0.0
    for _ in range(7):
        obs, reward, _, _, info = env.step(action)
        if reward < reward_loss:
            reward_loss = reward  # Capture min reward (price decrease step)

    # Reward should be negative when price decreases
    assert reward_loss < 0, f"Expected negative reward for price decrease, got {reward_loss}"

    print(f"  Reward on price decrease: {reward_loss:.6f}")
    print("  PASSED!")


def test_nav_tracking():
    """Test: NAV should track portfolio value correctly through price changes."""
    print("\nTest 3: NAV Tracking...")

    # Price: 100 -> 120 (increase) -> 100 (decrease back)
    prices = [100.0] * 5 + [120.0] * 5 + [100.0] * 5
    df = create_price_series(prices)
    env = CryptoTradingEnv(df, initial_balance=10000.0, window_size=2)

    obs, info = env.reset()
    initial_nav = info['nav']

    # Buy and hold
    action = np.array([1.0], dtype=np.float32)

    # Step through to price=120
    for _ in range(5):
        obs, _, _, _, info = env.step(action)

    peak_nav = info['nav']

    # Step through to price=100 (drop)
    for _ in range(4):
        obs, _, _, _, info = env.step(action)

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


def test_reward_scaling():
    """Test: Reward should be scaled by reward_scaling parameter."""
    print("\nTest 4: Reward Scaling...")

    # Need enough data: window_size=2 starts at step 2, price changes at index 5
    # So we need 3 steps to reach the price jump (step 2->3->4->5)
    prices = [100.0] * 5 + [110.0] * 10  # 10% price increase at index 5
    df = create_price_series(prices)

    # Test with scaling=1
    env1 = CryptoTradingEnv(df.copy(), initial_balance=10000.0, reward_scaling=1.0, window_size=2)
    env1.reset()
    action = np.array([1.0], dtype=np.float32)
    # Buy and capture max reward (price jump step)
    reward_scale1 = 0.0
    for _ in range(5):
        _, reward, _, _, _ = env1.step(action)
        if reward > reward_scale1:
            reward_scale1 = reward

    # Test with scaling=10
    env10 = CryptoTradingEnv(df.copy(), initial_balance=10000.0, reward_scaling=10.0, window_size=2)
    env10.reset()
    reward_scale10 = 0.0
    for _ in range(5):
        _, reward, _, _, _ = env10.step(action)
        if reward > reward_scale10:
            reward_scale10 = reward

    # Reward with scale=10 should be roughly 10x reward with scale=1
    # Note: Due to non-linear components (downside penalty, etc.), ratio may not be exactly 10
    ratio = reward_scale10 / reward_scale1 if reward_scale1 != 0 else 0

    assert 5.0 < ratio < 15.0, \
        f"Expected ratio roughly proportional to scaling, got {ratio:.2f} (rewards: {reward_scale1:.6f}, {reward_scale10:.6f})"

    print(f"  Reward (scale=1): {reward_scale1:.6f}")
    print(f"  Reward (scale=10): {reward_scale10:.6f}")
    print(f"  Ratio: {ratio:.2f}")
    print("  PASSED!")


def test_info_contains_metrics():
    """Test: Info dict should contain all monitoring metrics."""
    print("\nTest 5: Info Contains Metrics...")

    prices = [100.0] * 10
    df = create_price_series(prices)
    env = CryptoTradingEnv(df, initial_balance=10000.0, window_size=2)

    obs, info_reset = env.reset()

    # Check reset info (using current env keys)
    required_keys = ['step', 'nav', 'cash', 'position', 'position_pct', 'price',
                     'total_trades', 'total_commission']
    for key in required_keys:
        assert key in info_reset, f"Missing key '{key}' in reset info"

    # Do a step and check step info
    action = np.array([0.5], dtype=np.float32)
    obs, reward, _, _, info_step = env.step(action)

    # After step, should also have reward metrics (keys use forward slash /)
    required_keys_step = required_keys + ['action', 'return', 'rewards/log_return']
    for key in required_keys_step:
        assert key in info_step, f"Missing key '{key}' in step info"

    print(f"  Reset info keys: {list(info_reset.keys())}")
    print(f"  Step info keys: {list(info_step.keys())}")
    print(f"  Log return: {info_step['rewards/log_return']:.6f}")
    print(f"  NAV: ${info_step['nav']:.2f}")
    print("  PASSED!")


if __name__ == "__main__":
    print("=" * 50)
    print("Testing Risk-Adjusted Reward Function")
    print("=" * 50)

    test_reward_positive_return()
    test_reward_negative_return()
    test_nav_tracking()
    test_reward_scaling()
    test_info_contains_metrics()

    print("\n" + "=" * 50)
    print("[OK] All reward function tests passed!")
    print("=" * 50)

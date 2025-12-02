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

from src.env.trading_env import CryptoTradingEnv


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

    # Price goes up: 100 -> 110
    prices = [100.0] * 5 + [110.0] * 5
    df = create_price_series(prices)
    env = CryptoTradingEnv(df, initial_balance=10000.0, reward_scaling=10.0)

    obs, info = env.reset()

    # Buy at step 0 (price=100)
    action = np.array([1.0], dtype=np.float32)
    obs, reward1, _, _, info = env.step(action)

    # Hold through price increase (steps 1-4, price still 100)
    for _ in range(3):
        action = np.array([1.0], dtype=np.float32)
        obs, reward, _, _, info = env.step(action)

    # Step to price=110
    action = np.array([1.0], dtype=np.float32)
    obs, reward_gain, _, _, info = env.step(action)

    # Reward should be positive when price increases
    assert reward_gain > 0, f"Expected positive reward for price increase, got {reward_gain}"

    print(f"  Reward on price increase: {reward_gain:.6f}")
    print(f"  Log return: {info['log_return']:.6f}")
    print("  PASSED!")


def test_reward_negative_return():
    """Test: Price decrease should give negative reward."""
    print("\nTest 2: Negative Return Reward...")

    # Price goes down: 100 -> 90
    prices = [100.0] * 5 + [90.0] * 5
    df = create_price_series(prices)
    env = CryptoTradingEnv(df, initial_balance=10000.0, reward_scaling=10.0)

    obs, info = env.reset()

    # Buy at step 0
    action = np.array([1.0], dtype=np.float32)
    obs, reward1, _, _, info = env.step(action)

    # Hold through steps
    for _ in range(3):
        action = np.array([1.0], dtype=np.float32)
        obs, reward, _, _, info = env.step(action)

    # Step to price=90
    action = np.array([1.0], dtype=np.float32)
    obs, reward_loss, _, _, info = env.step(action)

    # Reward should be negative when price decreases
    assert reward_loss < 0, f"Expected negative reward for price decrease, got {reward_loss}"

    print(f"  Reward on price decrease: {reward_loss:.6f}")
    print(f"  Log return: {info['log_return']:.6f}")
    print("  PASSED!")


def test_drawdown_tracking():
    """Test: Max drawdown should increase after price drop from peak."""
    print("\nTest 3: Drawdown Tracking...")

    # Price: 100 -> 120 (new peak) -> 100 (drawdown)
    prices = [100.0, 100.0, 120.0, 120.0, 100.0, 100.0]
    df = create_price_series(prices)
    env = CryptoTradingEnv(df, initial_balance=10000.0)

    obs, info = env.reset()
    initial_drawdown = info['max_drawdown']

    # Buy and hold
    action = np.array([1.0], dtype=np.float32)

    # Step 0->1 (price 100->100)
    obs, _, _, _, info = env.step(action)
    dd_1 = info['max_drawdown']

    # Step 1->2 (price 100->120, new peak)
    obs, _, _, _, info = env.step(action)
    dd_2 = info['max_drawdown']
    peak_2 = info['peak_nav']

    # Step 2->3 (price 120->120)
    obs, _, _, _, info = env.step(action)

    # Step 3->4 (price 120->100, drawdown!)
    obs, _, _, _, info = env.step(action)
    dd_final = info['max_drawdown']

    assert dd_final > dd_2, \
        f"Expected max_drawdown to increase after drop, got {dd_2} -> {dd_final}"

    # Expected drawdown ~16.7% (from peak with ~120*99.89 holdings to 100*99.89)
    assert dd_final > 0.10, f"Expected significant drawdown, got {dd_final:.4f}"

    print(f"  Initial drawdown: {initial_drawdown:.4f}")
    print(f"  Drawdown at peak: {dd_2:.4f}")
    print(f"  Final drawdown: {dd_final:.4f}")
    print(f"  Peak NAV: ${peak_2:.2f}")
    print("  PASSED!")


def test_reward_scaling():
    """Test: Reward should be scaled by reward_scaling parameter."""
    print("\nTest 4: Reward Scaling...")

    prices = [100.0, 110.0, 110.0]  # 10% price increase
    df = create_price_series(prices)

    # Test with scaling=1
    env1 = CryptoTradingEnv(df.copy(), initial_balance=10000.0, reward_scaling=1.0)
    env1.reset()
    action = np.array([1.0], dtype=np.float32)
    env1.step(action)  # Buy
    _, reward_scale1, _, _, _ = env1.step(action)  # Price jump

    # Test with scaling=10
    env10 = CryptoTradingEnv(df.copy(), initial_balance=10000.0, reward_scaling=10.0)
    env10.reset()
    env10.step(action)  # Buy
    _, reward_scale10, _, _, _ = env10.step(action)  # Price jump

    # Reward with scale=10 should be ~10x reward with scale=1
    ratio = reward_scale10 / reward_scale1 if reward_scale1 != 0 else 0

    assert 9.0 < ratio < 11.0, \
        f"Expected ratio ~10, got {ratio:.2f} (rewards: {reward_scale1:.6f}, {reward_scale10:.6f})"

    print(f"  Reward (scale=1): {reward_scale1:.6f}")
    print(f"  Reward (scale=10): {reward_scale10:.6f}")
    print(f"  Ratio: {ratio:.2f}")
    print("  PASSED!")


def test_info_contains_metrics():
    """Test: Info dict should contain all monitoring metrics."""
    print("\nTest 5: Info Contains Metrics...")

    prices = [100.0] * 10
    df = create_price_series(prices)
    env = CryptoTradingEnv(df, initial_balance=10000.0)

    obs, info_reset = env.reset()

    # Check reset info
    required_keys = ['step', 'portfolio_value', 'cash', 'asset_holdings',
                     'price', 'max_drawdown', 'peak_nav']
    for key in required_keys:
        assert key in info_reset, f"Missing key '{key}' in reset info"

    # Do a step and check step info
    action = np.array([0.5], dtype=np.float32)
    obs, reward, _, _, info_step = env.step(action)

    required_keys_step = required_keys + ['action', 'log_return']
    for key in required_keys_step:
        assert key in info_step, f"Missing key '{key}' in step info"

    print(f"  Reset info keys: {list(info_reset.keys())}")
    print(f"  Step info keys: {list(info_step.keys())}")
    print(f"  Log return: {info_step['log_return']:.6f}")
    print(f"  Max drawdown: {info_step['max_drawdown']:.4f}")
    print(f"  Peak NAV: ${info_step['peak_nav']:.2f}")
    print("  PASSED!")


if __name__ == "__main__":
    print("=" * 50)
    print("Testing Risk-Adjusted Reward Function")
    print("=" * 50)

    test_reward_positive_return()
    test_reward_negative_return()
    test_drawdown_tracking()
    test_reward_scaling()
    test_info_contains_metrics()

    print("\n" + "=" * 50)
    print("[OK] All reward function tests passed!")
    print("=" * 50)

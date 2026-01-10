"""
test_env_logic.py - Tests de la logique financiere de l'environnement.

Verifie que CryptoTradingEnv execute correctement les trades
avec les frais de commission et slippage.
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.training.env import CryptoTradingEnv


def create_stable_price_data(n_rows: int = 100, price: float = 100.0) -> pd.DataFrame:
    """Create test data with stable price for predictable calculations."""
    data = {
        'open': np.full(n_rows, price),
        'high': np.full(n_rows, price),
        'low': np.full(n_rows, price),
        'close': np.full(n_rows, price),
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


def test_buy_max_position():
    """Test: Action=1 should buy max position (holdings > 0, cash ~ 0)."""
    print("Test 1: Buy Max Position (action=1)...")

    df = create_stable_price_data(price=100.0)
    env = CryptoTradingEnv(df, initial_balance=10000.0, commission=0.001, slippage=0.0001, window_size=10)

    obs, info = env.reset()
    initial_nav = info['nav']

    # Action = 1 (100% in asset)
    action = np.array([1.0], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)

    # Verify holdings > 0
    assert info['position'] > 0, f"Expected holdings > 0, got {info['position']}"

    # Verify cash is close to 0 (only fees taken)
    assert info['cash'] < 100, f"Expected cash < 100 (near 0), got {info['cash']}"

    # Verify NAV decreased slightly due to fees
    assert info['nav'] < initial_nav, \
        f"Expected NAV < {initial_nav} due to fees, got {info['nav']}"

    print(f"  Holdings: {info['position']:.6f}")
    print(f"  Cash: ${info['cash']:.2f}")
    print(f"  NAV: ${info['nav']:.2f} (initial: ${initial_nav:.2f})")
    print("  PASSED!")


def test_sell_all_position():
    """Test: Action=-1 after buying should sell all (holdings ~ 0)."""
    print("\nTest 2: Sell All Position (action=-1 after buy)...")

    df = create_stable_price_data(price=100.0)
    env = CryptoTradingEnv(df, initial_balance=10000.0, commission=0.001, slippage=0.0001, window_size=10)

    obs, info = env.reset()

    # First buy (action=1 -> 100% position)
    action_buy = np.array([1.0], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action_buy)
    nav_after_buy = info['nav']

    # Then sell all (action=-1 -> 0% position)
    action_sell = np.array([-1.0], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action_sell)

    # Verify holdings are back to 0 (or very close)
    assert abs(info['position']) < 0.0001, \
        f"Expected holdings ~ 0, got {info['position']}"

    # Verify cash is positive
    assert info['cash'] > 0, f"Expected cash > 0, got {info['cash']}"

    print(f"  Holdings after sell: {info['position']:.6f}")
    print(f"  Cash after sell: ${info['cash']:.2f}")
    print(f"  NAV after sell: ${info['nav']:.2f}")
    print("  PASSED!")


def test_fee_impact():
    """Test: Buy+Sell should reduce portfolio value due to fees."""
    print("\nTest 3: Fee Impact (buy then sell)...")

    df = create_stable_price_data(price=100.0)
    env = CryptoTradingEnv(df, initial_balance=10000.0, commission=0.001, slippage=0.0001, window_size=10)

    obs, info = env.reset()
    initial_nav = info['nav']

    # Buy max (action=1 -> 100%)
    action_buy = np.array([1.0], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action_buy)

    # Sell all (action=-1 -> 0%)
    action_sell = np.array([-1.0], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action_sell)

    final_nav = info['nav']

    # NAV should be less than initial due to commission + slippage on both trades
    # Expected loss: ~0.11% per trade * 2 trades = ~0.22% total
    expected_min_loss = initial_nav * 0.001  # At least 0.1% loss
    actual_loss = initial_nav - final_nav

    assert actual_loss > expected_min_loss, \
        f"Expected loss > ${expected_min_loss:.2f}, got ${actual_loss:.2f}"

    print(f"  Initial NAV: ${initial_nav:.2f}")
    print(f"  Final NAV: ${final_nav:.2f}")
    print(f"  Loss due to fees: ${actual_loss:.2f} ({100*actual_loss/initial_nav:.3f}%)")
    print("  PASSED!")


def test_partial_position():
    """Test: Action=0 should put 50% in asset (remapped from [-1,1] to [0,1])."""
    print("\nTest 4: Partial Position (action=0 -> 50%)...")

    df = create_stable_price_data(price=100.0)
    # Disable action_discretization to get exact 50% position
    env = CryptoTradingEnv(df, initial_balance=10000.0, commission=0.001, slippage=0.0001, window_size=10, action_discretization=0.0)

    obs, info = env.reset()

    # First go to 100% position (action=1) to establish a different position_pct
    # Initial position_pct=0 but actual holdings=0, so action=0 won't trigger trade
    action_full = np.array([1.0], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action_full)

    # Now 50% position: action=0 maps to (0+1)/2 = 0.5 = 50%
    action = np.array([0.0], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)

    # Expected: ~50 units at $100 = $5000 in asset
    expected_holdings = 50.0  # Approximately
    expected_cash = 5000.0  # Approximately (minus fees)

    assert 45 < info['position'] < 55, \
        f"Expected holdings ~50, got {info['position']}"
    assert 4900 < info['cash'] < 5100, \
        f"Expected cash ~5000, got {info['cash']}"

    print(f"  Holdings: {info['position']:.4f} (expected ~50)")
    print(f"  Cash: ${info['cash']:.2f} (expected ~$5000)")
    print("  PASSED!")


def test_full_cash_position():
    """Test: Action=-1 should result in 0% asset (100% cash)."""
    print("\nTest 5: Full Cash Position (action=-1 -> 0%)...")

    df = create_stable_price_data(price=100.0)
    env = CryptoTradingEnv(df, initial_balance=10000.0, window_size=10)

    obs, info = env.reset()

    # Action=-1 maps to (−1+1)/2 = 0 = 0% in asset
    action = np.array([-1.0], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)

    # Should stay in cash (holdings = 0)
    assert abs(info['position']) < 0.0001, \
        f"Expected holdings = 0, got {info['position']}"

    print(f"  Holdings: {info['position']:.6f}")
    print(f"  Cash: ${info['cash']:.2f}")
    print("  PASSED!")


if __name__ == "__main__":
    print("=" * 50)
    print("Testing CryptoTradingEnv Financial Logic")
    print("=" * 50)

    test_buy_max_position()
    test_sell_all_position()
    test_fee_impact()
    test_partial_position()
    test_full_cash_position()

    print("\n" + "=" * 50)
    print("[OK] All financial logic tests passed!")
    print("=" * 50)

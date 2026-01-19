# -*- coding: utf-8 -*-
"""
test_env_logic.py - Tests de la logique financiere de l'environnement.

Verifie que BatchCryptoEnv execute correctement les trades
avec les frais de commission et slippage.
"""

import sys
import os
import tempfile
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.training.batch_env import BatchCryptoEnv


def create_stable_price_data(n_rows: int = 200, price: float = 100.0) -> pd.DataFrame:
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


def create_variable_price_data(n_rows: int = 200, prices: list = None) -> pd.DataFrame:
    """Create test data with variable prices for testing short profit/loss."""
    if prices is None:
        prices = [100.0] * n_rows
    
    # Extend prices if needed
    if len(prices) < n_rows:
        prices = prices + [prices[-1]] * (n_rows - len(prices))
    
    prices = np.array(prices[:n_rows])
    
    data = {
        'open': prices,
        'high': prices,
        'low': prices,
        'close': prices,
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


def create_test_env(commission: float = 0.001, slippage: float = 0.0001, 
                    action_discretization: float = 0.1, window_size: int = 10,
                    price: float = 100.0, prices: list = None,
                    funding_rate: float = 0.0001):
    """Create a BatchCryptoEnv with test data in a temporary parquet file."""
    if prices is not None:
        df = create_variable_price_data(n_rows=200, prices=prices)
    else:
        df = create_stable_price_data(n_rows=200, price=price)
    
    # Create temporary parquet file
    tmp_file = tempfile.NamedTemporaryFile(suffix='.parquet', delete=False)
    df.to_parquet(tmp_file.name)
    
    env = BatchCryptoEnv(
        parquet_path=tmp_file.name,
        n_envs=1,
        device='cpu',  # Use CPU for tests
        window_size=window_size,
        episode_length=100,
        initial_balance=10_000.0,
        commission=commission,
        slippage=slippage,
        action_discretization=action_discretization,
        price_column='close',
        random_start=False,
        funding_rate=funding_rate,
    )
    
    return env, tmp_file.name


def cleanup_env(parquet_path: str):
    """Remove temporary parquet file."""
    try:
        os.unlink(parquet_path)
    except Exception:
        pass


def test_buy_max_position():
    """Test: Action=1 should buy max position (holdings > 0, cash ~ 0)."""
    print("Test 1: Buy Max Position (action=1)...")

    env, tmp_path = create_test_env()
    try:
        obs, info = env.gym_reset()
        initial_nav = info['nav']

        # Action = 1 (100% in asset)
        action = np.array([1.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.gym_step(action)

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
    finally:
        cleanup_env(tmp_path)


def test_sell_all_position():
    """Test: Action=0 after buying should sell all (holdings ~ 0)."""
    print("\nTest 2: Sell All Position (action=0 after buy)...")

    env, tmp_path = create_test_env()
    try:
        obs, info = env.gym_reset()

        # First buy (action=1 -> 100% position)
        action_buy = np.array([1.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.gym_step(action_buy)

        # Then sell all (action=0 -> 0% position = cash)
        action_sell = np.array([0.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.gym_step(action_sell)

        # Verify holdings are back to 0 (or very close)
        assert abs(info['position']) < 0.0001, \
            f"Expected holdings ~ 0, got {info['position']}"

        # Verify cash is positive
        assert info['cash'] > 0, f"Expected cash > 0, got {info['cash']}"

        print(f"  Holdings after sell: {info['position']:.6f}")
        print(f"  Cash after sell: ${info['cash']:.2f}")
        print(f"  NAV after sell: ${info['nav']:.2f}")
        print("  PASSED!")
    finally:
        cleanup_env(tmp_path)


def test_fee_impact():
    """Test: Buy+Sell should reduce portfolio value due to fees."""
    print("\nTest 3: Fee Impact (buy then sell)...")

    env, tmp_path = create_test_env()
    try:
        obs, info = env.gym_reset()
        initial_nav = info['nav']

        # Buy max (action=1 -> 100%)
        action_buy = np.array([1.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.gym_step(action_buy)

        # Sell all (action=0 -> 0% = cash)
        action_sell = np.array([0.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.gym_step(action_sell)

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
    finally:
        cleanup_env(tmp_path)


def test_partial_position():
    """Test: Action=0.5 should put 50% in asset (direct mapping [-1,1])."""
    print("\nTest 4: Partial Position (action=0.5 -> 50%)...")

    # Disable action_discretization to get exact 50% position
    env, tmp_path = create_test_env(action_discretization=0.0)
    try:
        obs, info = env.gym_reset()

        # 50% position: action=0.5 maps directly to 50% exposure
        action = np.array([0.5], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.gym_step(action)

        # Expected: ~50 units at $100 = $5000 in asset
        assert 45 < info['position'] < 55, \
            f"Expected holdings ~50, got {info['position']}"
        assert 4900 < info['cash'] < 5100, \
            f"Expected cash ~5000, got {info['cash']}"

        print(f"  Holdings: {info['position']:.4f} (expected ~50)")
        print(f"  Cash: ${info['cash']:.2f} (expected ~$5000)")
        print("  PASSED!")
    finally:
        cleanup_env(tmp_path)


def test_full_cash_position():
    """Test: Action=0 should result in 0% asset (100% cash)."""
    print("\nTest 5: Full Cash Position (action=0 -> 0%)...")

    env, tmp_path = create_test_env()
    try:
        obs, info = env.gym_reset()

        # Action=0 maps directly to 0% exposure (cash)
        action = np.array([0.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.gym_step(action)

        # Should stay in cash (holdings = 0)
        assert abs(info['position']) < 0.0001, \
            f"Expected holdings = 0, got {info['position']}"

        print(f"  Holdings: {info['position']:.6f}")
        print(f"  Cash: ${info['cash']:.2f}")
        print("  PASSED!")
    finally:
        cleanup_env(tmp_path)


def test_short_position():
    """Test: Action=-1 should create a negative position (100% short)."""
    print("\nTest 6: Short Position (action=-1 -> -100%)...")

    env, tmp_path = create_test_env(funding_rate=0.0)  # Disable funding for this test
    try:
        obs, info = env.gym_reset()
        initial_nav = info['nav']

        # Action=-1 = 100% short
        action = np.array([-1.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.gym_step(action)

        # Verify position is negative (short)
        assert info['position'] < 0, f"Expected negative position, got {info['position']}"

        # Verify cash increased (received from short sale)
        # When shorting: we sell units we don't own, receiving cash
        # NAV = cash + position * price should still equal ~initial_nav (minus fees)
        assert info['cash'] > initial_nav, f"Expected cash > {initial_nav}, got {info['cash']}"

        # NAV should be close to initial (slightly less due to fees)
        assert info['nav'] < initial_nav, \
            f"Expected NAV < {initial_nav} due to fees, got {info['nav']}"
        assert info['nav'] > initial_nav * 0.99, \
            f"Expected NAV > {initial_nav * 0.99}, got {info['nav']}"

        print(f"  Position: {info['position']:.6f} (negative = short)")
        print(f"  Cash: ${info['cash']:.2f} (> initial due to short sale proceeds)")
        print(f"  NAV: ${info['nav']:.2f} (initial: ${initial_nav:.2f})")
        print("  PASSED!")
    finally:
        cleanup_env(tmp_path)


def test_short_profit_on_price_drop():
    """Test: Short position should profit when price drops."""
    print("\nTest 7: Short Profit on Price Drop...")

    # Create prices: stable at 100 for window, then drops to 90
    window_size = 10
    prices = [100.0] * (window_size + 5) + [90.0] * 50
    
    env, tmp_path = create_test_env(
        prices=prices, 
        window_size=window_size,
        funding_rate=0.0,  # Disable funding to isolate price effect
        commission=0.0,    # Disable fees to isolate price effect
        slippage=0.0
    )
    try:
        obs, info = env.gym_reset()
        initial_nav = info['nav']  # 10,000

        # Open short position at price 100
        action_short = np.array([-1.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.gym_step(action_short)
        
        nav_after_short = info['nav']
        position_units = info['position']  # Should be -100 units
        
        # Advance one more step (price drops to 90)
        action_hold = np.array([-1.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.gym_step(action_hold)
        
        nav_after_drop = info['nav']

        # Profit calculation: short 100 units at 100, now worth 90
        # Profit = 100 units * (100 - 90) = $1000
        expected_profit = abs(position_units) * (100 - 90)
        actual_profit = nav_after_drop - nav_after_short

        assert actual_profit > 0, f"Expected profit > 0, got {actual_profit}"
        assert abs(actual_profit - expected_profit) < 10, \
            f"Expected profit ~${expected_profit:.2f}, got ${actual_profit:.2f}"

        print(f"  Initial NAV: ${initial_nav:.2f}")
        print(f"  NAV after short: ${nav_after_short:.2f}")
        print(f"  NAV after price drop: ${nav_after_drop:.2f}")
        print(f"  Profit: ${actual_profit:.2f} (expected ~${expected_profit:.2f})")
        print("  PASSED!")
    finally:
        cleanup_env(tmp_path)


def test_short_loss_on_price_rise():
    """Test: Short position should lose when price rises."""
    print("\nTest 8: Short Loss on Price Rise...")

    # Create prices: stable at 100 for window, then rises to 110
    window_size = 10
    prices = [100.0] * (window_size + 5) + [110.0] * 50
    
    env, tmp_path = create_test_env(
        prices=prices, 
        window_size=window_size,
        funding_rate=0.0,  # Disable funding to isolate price effect
        commission=0.0,    # Disable fees to isolate price effect
        slippage=0.0
    )
    try:
        obs, info = env.gym_reset()
        initial_nav = info['nav']  # 10,000

        # Open short position at price 100
        action_short = np.array([-1.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.gym_step(action_short)
        
        nav_after_short = info['nav']
        position_units = info['position']  # Should be -100 units
        
        # Advance one more step (price rises to 110)
        action_hold = np.array([-1.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.gym_step(action_hold)
        
        nav_after_rise = info['nav']

        # Loss calculation: short 100 units at 100, now worth 110
        # Loss = 100 units * (110 - 100) = $1000
        expected_loss = abs(position_units) * (110 - 100)
        actual_loss = nav_after_short - nav_after_rise

        assert actual_loss > 0, f"Expected loss > 0, got {actual_loss}"
        assert abs(actual_loss - expected_loss) < 10, \
            f"Expected loss ~${expected_loss:.2f}, got ${actual_loss:.2f}"

        print(f"  Initial NAV: ${initial_nav:.2f}")
        print(f"  NAV after short: ${nav_after_short:.2f}")
        print(f"  NAV after price rise: ${nav_after_rise:.2f}")
        print(f"  Loss: ${actual_loss:.2f} (expected ~${expected_loss:.2f})")
        print("  PASSED!")
    finally:
        cleanup_env(tmp_path)


def test_funding_cost():
    """Test: Funding rate should reduce cash for short positions."""
    print("\nTest 9: Funding Cost for Shorts...")

    # High funding rate for visible effect
    funding_rate = 0.01  # 1% per step
    
    env, tmp_path = create_test_env(
        funding_rate=funding_rate,
        commission=0.0,  # Disable fees to isolate funding effect
        slippage=0.0
    )
    try:
        obs, info = env.gym_reset()
        initial_nav = info['nav']  # 10,000

        # Open short position
        action_short = np.array([-1.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.gym_step(action_short)
        
        cash_after_short = info['cash']
        position_units = info['position']  # -100 units at $100 = -$10,000 exposure
        
        # Hold short for one more step (funding applies)
        action_hold = np.array([-1.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.gym_step(action_hold)
        
        cash_after_funding = info['cash']

        # Funding cost = |position| * price * funding_rate
        # = 100 units * $100 * 0.01 = $100
        expected_funding_cost = abs(position_units) * 100 * funding_rate
        actual_funding_cost = cash_after_short - cash_after_funding

        assert actual_funding_cost > 0, f"Expected funding cost > 0, got {actual_funding_cost}"
        assert abs(actual_funding_cost - expected_funding_cost) < 1, \
            f"Expected funding ~${expected_funding_cost:.2f}, got ${actual_funding_cost:.2f}"

        print(f"  Position: {position_units:.2f} units (short)")
        print(f"  Cash after short: ${cash_after_short:.2f}")
        print(f"  Cash after 1 step: ${cash_after_funding:.2f}")
        print(f"  Funding cost: ${actual_funding_cost:.2f} (expected ~${expected_funding_cost:.2f})")
        print("  PASSED!")
    finally:
        cleanup_env(tmp_path)


def test_no_funding_for_long():
    """Test: Funding rate should NOT apply to long positions."""
    print("\nTest 10: No Funding for Long Positions...")

    # High funding rate
    funding_rate = 0.01  # 1% per step
    
    env, tmp_path = create_test_env(
        funding_rate=funding_rate,
        commission=0.0,
        slippage=0.0
    )
    try:
        obs, info = env.gym_reset()

        # Open long position
        action_long = np.array([1.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.gym_step(action_long)
        
        cash_after_long = info['cash']  # Should be ~0
        
        # Hold long for one more step
        action_hold = np.array([1.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.gym_step(action_hold)
        
        cash_after_hold = info['cash']

        # Cash should not change (no funding for longs)
        assert abs(cash_after_long - cash_after_hold) < 0.01, \
            f"Expected no funding for long, cash changed from {cash_after_long} to {cash_after_hold}"

        print(f"  Cash after long: ${cash_after_long:.2f}")
        print(f"  Cash after 1 step: ${cash_after_hold:.2f}")
        print(f"  No funding applied (as expected)")
        print("  PASSED!")
    finally:
        cleanup_env(tmp_path)


if __name__ == "__main__":
    print("=" * 50)
    print("Testing BatchCryptoEnv Financial Logic")
    print("=" * 50)

    test_buy_max_position()
    test_sell_all_position()
    test_fee_impact()
    test_partial_position()
    test_full_cash_position()
    
    print("\n" + "-" * 50)
    print("Short Selling Tests")
    print("-" * 50)
    
    test_short_position()
    test_short_profit_on_price_drop()
    test_short_loss_on_price_rise()
    test_funding_cost()
    test_no_funding_for_long()

    print("\n" + "=" * 50)
    print("[OK] All financial logic tests passed!")
    print("=" * 50)

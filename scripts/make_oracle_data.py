# -*- coding: utf-8 -*-
"""
make_oracle_data.py - Generate synthetic "Oracle" dataset for architecture validation.

The Oracle Test: If the model can't profit when the signal is OBVIOUS,
the architecture is broken. If it can, the problem is in our real features.

Logic:
- ORACLE_SIGNAL: +1 (buy) or -1 (sell), randomly generated
- Price movement at t+1 is GUARANTEED to follow the signal (with noise)
- Correlation(ORACLE_SIGNAL[t], Return[t+1]) > 0.9

Usage:
    python scripts/make_oracle_data.py
    python scripts/make_oracle_data.py --rows 10000 --signal-strength 0.95
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def generate_oracle_data(
    n_rows: int = 20000,
    signal_strength: float = 0.95,  # Correlation between signal and next return
    base_price: float = 50000.0,
    volatility: float = 0.02,  # 2% base volatility
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic data with a perfect oracle signal.

    Args:
        n_rows: Number of rows to generate
        signal_strength: How strongly the signal predicts the next return (0.0-1.0)
        base_price: Starting price
        volatility: Base price volatility
        seed: Random seed for reproducibility

    Returns:
        DataFrame compatible with BatchCryptoEnv
    """
    np.random.seed(seed)

    print(f"Generating Oracle dataset:")
    print(f"  Rows: {n_rows}")
    print(f"  Signal strength: {signal_strength}")
    print(f"  Base volatility: {volatility}")

    # 1. Generate ORACLE_SIGNAL: +1 or -1 (random)
    oracle_signal = np.random.choice([-1.0, 1.0], size=n_rows)

    # 2. Generate returns that FOLLOW the signal
    # Return = signal_strength * signal * volatility + noise
    noise = np.random.randn(n_rows) * volatility * (1 - signal_strength)
    returns = signal_strength * oracle_signal * volatility + noise

    # 3. Build price series from returns
    # Price[t+1] = Price[t] * (1 + return[t])
    prices = np.zeros(n_rows)
    prices[0] = base_price
    for i in range(1, n_rows):
        prices[i] = prices[i-1] * (1 + returns[i-1])

    # 4. Generate OHLCV data (realistic but simple)
    high = prices * (1 + np.abs(np.random.randn(n_rows) * 0.005))
    low = prices * (1 - np.abs(np.random.randn(n_rows) * 0.005))
    open_price = prices * (1 + np.random.randn(n_rows) * 0.002)
    volume = np.random.uniform(1000, 10000, n_rows)

    # Ensure OHLC consistency
    high = np.maximum(high, np.maximum(prices, open_price))
    low = np.minimum(low, np.minimum(prices, open_price))

    # 5. Create DataFrame with required columns
    # BatchCryptoEnv expects: BTC_Open, BTC_High, BTC_Low, BTC_Close, BTC_Volume
    # Plus any feature columns

    df = pd.DataFrame({
        # OHLCV (required)
        'BTC_Open': open_price,
        'BTC_High': high,
        'BTC_Low': low,
        'BTC_Close': prices,
        'BTC_Volume': volume,

        # Oracle signal (the "cheat" feature)
        'ORACLE_SIGNAL': oracle_signal,

        # Additional features to match expected format
        # (BatchCryptoEnv may expect certain columns)
        'BTC_LogRet': np.concatenate([[0], returns[:-1]]),
        'BTC_Vol_ZScore': np.random.randn(n_rows) * 0.5,  # Dummy
        'BTC_ZScore': np.random.randn(n_rows) * 0.5,  # Dummy
        'BTC_Parkinson_Vol': np.full(n_rows, volatility),  # Constant
        'BTC_GK_Vol': np.full(n_rows, volatility),  # Constant
        'BTC_FFD': np.random.randn(n_rows) * 0.1,  # Dummy
    })

    # Add dummy columns for other assets (ETH, SPX, etc.) if needed
    # BatchCryptoEnv might expect multi-asset data
    for asset in ['ETH', 'SPX', 'DXY', 'NASDAQ']:
        df[f'{asset}_Open'] = df['BTC_Open'] * np.random.uniform(0.01, 0.1)
        df[f'{asset}_High'] = df['BTC_High'] * np.random.uniform(0.01, 0.1)
        df[f'{asset}_Low'] = df['BTC_Low'] * np.random.uniform(0.01, 0.1)
        df[f'{asset}_Close'] = df['BTC_Close'] * np.random.uniform(0.01, 0.1)
        df[f'{asset}_Volume'] = volume * np.random.uniform(0.5, 2.0)
        df[f'{asset}_LogRet'] = np.random.randn(n_rows) * 0.01
        df[f'{asset}_Vol_ZScore'] = np.random.randn(n_rows) * 0.5
        df[f'{asset}_ZScore'] = np.random.randn(n_rows) * 0.5
        df[f'{asset}_Parkinson_Vol'] = np.full(n_rows, 0.01)
        df[f'{asset}_GK_Vol'] = np.full(n_rows, 0.01)
        df[f'{asset}_FFD'] = np.random.randn(n_rows) * 0.1

    # 6. Validate signal correlation
    actual_returns = np.diff(prices) / prices[:-1]
    correlation = np.corrcoef(oracle_signal[:-1], actual_returns)[0, 1]
    print(f"\n  Signal-Return Correlation: {correlation:.4f}")

    if correlation < 0.8:
        print(f"  WARNING: Correlation below 0.8!")
    else:
        print(f"  SUCCESS: Strong predictive signal")

    # 7. Print statistics
    print(f"\n  Price range: {prices.min():.2f} - {prices.max():.2f}")
    print(f"  Mean return: {np.mean(returns)*100:.4f}%")
    print(f"  Return std: {np.std(returns)*100:.4f}%")

    return df


def main():
    parser = argparse.ArgumentParser(description="Generate Oracle dataset")
    parser.add_argument("--rows", type=int, default=20000, help="Number of rows")
    parser.add_argument("--signal-strength", type=float, default=0.95, help="Signal strength (0-1)")
    parser.add_argument("--output", type=str, default="data/oracle_dataset.parquet", help="Output path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Generate data
    df = generate_oracle_data(
        n_rows=args.rows,
        signal_strength=args.signal_strength,
        seed=args.seed,
    )

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    print(f"\n  Saved to: {output_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)[:10]}...")


if __name__ == "__main__":
    main()

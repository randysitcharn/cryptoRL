"""
test_splitting.py - Tests for TimeSeriesSplitter.

Verifies chronological splitting with purge windows.
"""

import sys
import os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_engineering.splitter import TimeSeriesSplitter


def test_split_sizes():
    """Test: Split sizes are correct accounting for purge."""
    print("Test 1: Split Sizes...")

    df = pd.read_csv("data/processed/BTC-USD_processed.csv",
                     index_col=0, parse_dates=True)
    splitter = TimeSeriesSplitter(df)

    train_df, val_df, test_df = splitter.split_data(
        train_ratio=0.7, val_ratio=0.15, purge_window=50
    )

    # Total should be original minus 2*purge
    total_used = len(train_df) + len(val_df) + len(test_df)
    expected_loss = 2 * 50  # 2 purge windows

    assert total_used == len(df) - expected_loss, \
        f"Expected {len(df) - expected_loss} rows, got {total_used}"

    print(f"  Total: {len(df)}, Used: {total_used}, Purged: {expected_loss}")
    print("  PASSED!")


def test_chronological_order():
    """Test: Sets are in chronological order."""
    print("\nTest 2: Chronological Order...")

    df = pd.read_csv("data/processed/BTC-USD_processed.csv",
                     index_col=0, parse_dates=True)
    splitter = TimeSeriesSplitter(df)

    train_df, val_df, test_df = splitter.split_data()

    # Train end < Val start
    assert train_df.index[-1] < val_df.index[0], \
        "Train end should be before Val start"

    # Val end < Test start
    assert val_df.index[-1] < test_df.index[0], \
        "Val end should be before Test start"

    print("  Train end < Val start: OK")
    print("  Val end < Test start: OK")
    print("  PASSED!")


def test_no_empty_sets():
    """Test: All sets have data."""
    print("\nTest 3: No Empty Sets...")

    df = pd.read_csv("data/processed/BTC-USD_processed.csv",
                     index_col=0, parse_dates=True)
    splitter = TimeSeriesSplitter(df)

    train_df, val_df, test_df = splitter.split_data()

    assert len(train_df) > 0, "Train set is empty"
    assert len(val_df) > 0, "Val set is empty"
    assert len(test_df) > 0, "Test set is empty"

    print(f"  Train: {len(train_df)} rows")
    print(f"  Val: {len(val_df)} rows")
    print(f"  Test: {len(test_df)} rows")
    print("  PASSED!")


if __name__ == "__main__":
    print("=" * 50)
    print("Testing TimeSeriesSplitter")
    print("=" * 50)

    test_split_sizes()
    test_chronological_order()
    test_no_empty_sets()

    print("\n" + "=" * 50)
    print("[OK] All splitting tests passed!")
    print("=" * 50)

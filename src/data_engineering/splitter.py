"""
splitter.py - Time series data splitting for RL training.

Splits processed data chronologically into train/val/test sets
with purge windows to prevent lookahead bias from technical indicators.

IMPORTANT: Purge window must be >= max indicator window (720h for Z-Score)
to prevent data leakage. See audit DATA_PIPELINE_AUDIT_REPORT.md P0.2
and AUDIT_MODELES_RL_RESULTATS.md Contre-Audit section.
"""

import pandas as pd

from src.config.constants import (
    DEFAULT_PURGE_WINDOW,
    DEFAULT_EMBARGO_WINDOW,
    MAX_LOOKBACK_WINDOW,
)


def validate_purge_window(purge_window: int, raise_error: bool = True) -> bool:
    """
    Validate that purge window is >= max lookback window.
    
    This validation is CRITICAL to prevent data leakage in WFO.
    The purge window must be at least as large as the longest
    feature lookback window to ensure no overlap between train/test.
    
    Args:
        purge_window: The purge window in hours/bars.
        raise_error: If True, raises ValueError on invalid. If False, returns False.
        
    Returns:
        True if valid, False if invalid (when raise_error=False).
        
    Raises:
        ValueError: If purge_window < MAX_LOOKBACK_WINDOW and raise_error=True.
        
    Example:
        >>> validate_purge_window(720)  # OK
        True
        >>> validate_purge_window(48, raise_error=False)  # Too small
        False
    """
    if purge_window < MAX_LOOKBACK_WINDOW:
        msg = (
            f"PURGE WINDOW TOO SMALL: {purge_window}h < {MAX_LOOKBACK_WINDOW}h (MAX_LOOKBACK_WINDOW). "
            f"This can cause data leakage! Features with long lookback windows "
            f"(e.g., Z-Score=720h) will see data from the test set. "
            f"Set purge_window >= {MAX_LOOKBACK_WINDOW}h."
        )
        if raise_error:
            raise ValueError(msg)
        else:
            print(f"[WARNING] {msg}")
            return False
    return True


class TimeSeriesSplitter:
    """Chronological data splitter with purge and embargo windows."""

    def __init__(self, df: pd.DataFrame):
        """
        Initialize splitter with processed DataFrame.

        Args:
            df (pd.DataFrame): Processed data with datetime index.
        """
        self.df = df
        self.n_samples = len(df)

    def split_data(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        purge_window: int = DEFAULT_PURGE_WINDOW,
        embargo_window: int = DEFAULT_EMBARGO_WINDOW
    ) -> tuple:
        """
        Split data chronologically with purge and embargo windows.

        Purge: Gap BEFORE val/test sets to prevent indicator look-ahead bias.
               Must be >= max indicator window (720h for Z-Score).

        Embargo: Gap AFTER test set before next train (for WFO).
                 Allows label correlations to decay.

        Args:
            train_ratio (float): Fraction for training (default 0.7).
            val_ratio (float): Fraction for validation (default 0.15).
            purge_window (int): Bars to skip between sets (default 720).
            embargo_window (int): Bars after test before next train (default 24).

        Returns:
            tuple: (train_df, val_df, test_df)
            
        Raises:
            ValueError: If purge_window < MAX_LOOKBACK_WINDOW (data leakage risk).
        """
        # Validate purge window (CRITICAL for data integrity)
        validate_purge_window(purge_window, raise_error=True)
        
        # Calculate split indices
        idx_train_end = int(self.n_samples * train_ratio)
        idx_val_end = int(self.n_samples * (train_ratio + val_ratio))

        # Split with purge windows (gap BEFORE val/test)
        train_df = self.df.iloc[:idx_train_end]
        val_df = self.df.iloc[idx_train_end + purge_window:idx_val_end]
        test_df = self.df.iloc[idx_val_end + purge_window:]

        # Print stats
        self._print_split_stats(train_df, val_df, test_df, purge_window, embargo_window)

        return train_df, val_df, test_df

    def _print_split_stats(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        purge_window: int,
        embargo_window: int
    ) -> None:
        """Print split statistics for validation."""
        print("=" * 60)
        print("Time Series Split Summary (Leak-Free)")
        print("=" * 60)

        print(f"\nTrain Set:")
        print(f"  Rows: {len(train_df)}")
        print(f"  Start: {train_df.index[0]}")
        print(f"  End:   {train_df.index[-1]}")

        print(f"\nValidation Set:")
        print(f"  Rows: {len(val_df)}")
        print(f"  Start: {val_df.index[0]}")
        print(f"  End:   {val_df.index[-1]}")

        print(f"\nTest Set:")
        print(f"  Rows: {len(test_df)}")
        print(f"  Start: {test_df.index[0]}")
        print(f"  End:   {test_df.index[-1]}")

        purge_lost = 2 * purge_window
        total_used = len(train_df) + len(val_df) + len(test_df)
        print(f"\nPurge & Embargo Statistics:")
        print(f"  Purge window: {purge_window} bars (gap before val/test)")
        print(f"  Embargo window: {embargo_window} bars (gap after test for WFO)")
        print(f"  Rows lost to purge: {purge_lost}")
        print(f"  Total rows used: {total_used} / {self.n_samples}")
        print("=" * 60)

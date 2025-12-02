"""
splitter.py - Time series data splitting for RL training.

Splits processed data chronologically into train/val/test sets
with purge windows to prevent lookahead bias from technical indicators.
"""

import pandas as pd


class TimeSeriesSplitter:
    """Chronological data splitter with purge windows."""

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
        purge_window: int = 50
    ) -> tuple:
        """
        Split data chronologically with purge windows.

        Args:
            train_ratio (float): Fraction for training (default 0.7).
            val_ratio (float): Fraction for validation (default 0.15).
            purge_window (int): Bars to skip between sets (default 50).

        Returns:
            tuple: (train_df, val_df, test_df)
        """
        # Calculate split indices
        idx_train_end = int(self.n_samples * train_ratio)
        idx_val_end = int(self.n_samples * (train_ratio + val_ratio))

        # Split with purge windows
        train_df = self.df.iloc[:idx_train_end]
        val_df = self.df.iloc[idx_train_end + purge_window:idx_val_end]
        test_df = self.df.iloc[idx_val_end + purge_window:]

        # Print stats
        self._print_split_stats(train_df, val_df, test_df, purge_window)

        return train_df, val_df, test_df

    def _print_split_stats(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        purge_window: int
    ) -> None:
        """Print split statistics for validation."""
        print("=" * 60)
        print("Time Series Split Summary")
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
        print(f"\nPurge Statistics:")
        print(f"  Purge window: {purge_window} bars")
        print(f"  Rows lost to purge: {purge_lost}")
        print(f"  Total rows used: {total_used} / {self.n_samples}")
        print("=" * 60)

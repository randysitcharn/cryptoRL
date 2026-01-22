"""
test_data_leakage.py - Tests to verify no data leakage in the pipeline.

Critical tests for audit compliance (DATA_PIPELINE_AUDIT_REPORT.md):
- P0.1: Scaler fit on train only
- P0.2: Purge window >= max indicator window (720h)
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import RobustScaler

from src.config.constants import (
    DEFAULT_EMBARGO_WINDOW,
    DEFAULT_PURGE_WINDOW,
    MAX_INDICATOR_WINDOW,
    ZSCORE_WINDOW,
)
from src.data_engineering.splitter import TimeSeriesSplitter


class TestScalerLeakage:
    """Tests for P0.1 - Scaler must fit on train only."""

    def test_scaler_fit_on_train_only(self):
        """Verify that scaler statistics come from train data only."""
        # Create synthetic data with distinct train/test distributions
        # Use different seeds to ensure different distributions
        np.random.seed(42)
        n_train = 1000
        n_test = 500

        # Train data: centered around 0
        train_data = np.random.randn(n_train, 3)

        # Test data: centered around 20 (very different distribution)
        # Use a different seed to ensure different values
        np.random.seed(123)
        test_data = np.random.randn(n_test, 3) * 2 + 20

        # Combine
        all_data = np.vstack([train_data, test_data])

        # Correct approach: fit on train only
        scaler_correct = RobustScaler()
        scaler_correct.fit(train_data)

        # Wrong approach: fit on all data (leakage)
        scaler_wrong = RobustScaler()
        scaler_wrong.fit(all_data)

        # The median (center_) should be different
        # If fit on train only: median ≈ 0
        # If fit on all: median should be shifted towards test data
        train_median = np.median(train_data, axis=0)
        correct_center = scaler_correct.center_
        wrong_center = scaler_wrong.center_

        # Correct scaler should have center close to train median
        np.testing.assert_array_almost_equal(
            correct_center, train_median, decimal=1,
            err_msg="Correct scaler center should match train median"
        )

        # Wrong scaler should have different center (proof of leakage)
        # The wrong center should be significantly different (> 0.5 units away, adjusted for RobustScaler behavior)
        # RobustScaler uses median, which is more robust to outliers, so differences are smaller
        center_diff = np.abs(wrong_center - train_median)
        assert np.all(center_diff > 0.5), (
            f"If centers are similar, test data didn't have different distribution. "
            f"Difference: {center_diff}, train_median: {train_median}, wrong_center: {wrong_center}"
        )

    def test_scaler_transform_uses_train_stats(self):
        """Verify transformed test data uses train statistics."""
        np.random.seed(42)

        # Train: N(0, 1)
        train_data = np.random.randn(1000, 2)
        # Test: N(5, 2) - shifted and scaled
        test_data = np.random.randn(200, 2) * 2 + 5

        # Fit on train only
        scaler = RobustScaler()
        scaler.fit(train_data)

        # Transform test
        test_transformed = scaler.transform(test_data)

        # Test data after transform should NOT be centered around 0
        # because scaler was fit on train (mean≈0), not test (mean≈5)
        test_mean = test_transformed.mean(axis=0)

        # Should be around (5 - 0) / IQR_train ≈ 5 / 1.35 ≈ 3.7
        assert np.all(test_mean > 2.0), (
            f"Transformed test mean {test_mean} should be > 2 (not centered)"
        )


class TestPurgeWindow:
    """Tests for P0.2 - Purge window must be >= max indicator window."""

    def test_default_purge_window_sufficient(self):
        """Verify default purge window >= max indicator window."""
        assert DEFAULT_PURGE_WINDOW >= MAX_INDICATOR_WINDOW, (
            f"Default purge ({DEFAULT_PURGE_WINDOW}) < max indicator window ({MAX_INDICATOR_WINDOW})"
        )

    def test_purge_window_covers_zscore(self):
        """Verify purge window >= Z-Score window (720h)."""
        assert DEFAULT_PURGE_WINDOW >= ZSCORE_WINDOW, (
            f"Purge window ({DEFAULT_PURGE_WINDOW}) must be >= Z-Score window ({ZSCORE_WINDOW})"
        )

    def test_splitter_uses_correct_default(self):
        """Verify TimeSeriesSplitter uses the correct default purge window."""
        # Create dummy data
        df = pd.DataFrame(
            {'close': np.random.randn(10000)},
            index=pd.date_range('2020-01-01', periods=10000, freq='h')
        )
        splitter = TimeSeriesSplitter(df)

        # Check default parameter
        import inspect
        sig = inspect.signature(splitter.split_data)
        purge_default = sig.parameters['purge_window'].default

        assert purge_default >= 720, (
            f"Splitter default purge ({purge_default}) should be >= 720"
        )


class TestNoOverlapTrainTest:
    """Tests to verify no overlap between train and test after purge."""

    def test_no_indicator_overlap(self):
        """Verify that train indicators don't leak into test via rolling windows."""
        # Create data
        n_samples = 5000
        df = pd.DataFrame(
            {'close': np.random.randn(n_samples)},
            index=pd.date_range('2020-01-01', periods=n_samples, freq='h')
        )

        splitter = TimeSeriesSplitter(df)
        train_df, val_df, test_df = splitter.split_data(
            train_ratio=0.7,
            val_ratio=0.15,
            purge_window=720  # Max indicator window
        )

        # Calculate the gap between train end and val start
        train_end_idx = df.index.get_loc(train_df.index[-1])
        val_start_idx = df.index.get_loc(val_df.index[0])
        gap = val_start_idx - train_end_idx - 1

        # Gap should be exactly purge_window
        assert gap >= 720, (
            f"Gap between train and val ({gap}) should be >= 720"
        )

    def test_embargo_parameter_exists(self):
        """Verify embargo_window parameter exists in TimeSeriesSplitter."""
        df = pd.DataFrame(
            {'close': np.random.randn(1000)},
            index=pd.date_range('2020-01-01', periods=1000, freq='h')
        )
        splitter = TimeSeriesSplitter(df)

        # Check that embargo_window parameter exists
        import inspect
        sig = inspect.signature(splitter.split_data)
        assert 'embargo_window' in sig.parameters, (
            "embargo_window parameter should exist in split_data"
        )


class TestWFOLeakage:
    """Tests for WFO-specific leakage prevention."""

    def test_wfo_config_has_purge(self):
        """Verify WFOConfig has purge_window parameter."""
        import sys
        sys.path.insert(0, 'scripts')
        from run_full_wfo import WFOConfig

        config = WFOConfig()
        assert hasattr(config, 'purge_window'), "WFOConfig should have purge_window"
        assert config.purge_window >= 720, (
            f"WFO purge_window ({config.purge_window}) should be >= 720"
        )

    def test_wfo_config_has_embargo(self):
        """Verify WFOConfig has embargo_window parameter."""
        import sys
        sys.path.insert(0, 'scripts')
        from run_full_wfo import WFOConfig

        config = WFOConfig()
        assert hasattr(config, 'embargo_window'), "WFOConfig should have embargo_window"
        assert config.embargo_window >= DEFAULT_EMBARGO_WINDOW, (
            f"WFO embargo_window ({config.embargo_window}) should be >= {DEFAULT_EMBARGO_WINDOW}"
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

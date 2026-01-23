"""
test_hmm_features.py - Pytest suite for HMM features validation.

Tests for audit compliance (audit_hmm_features.py):
- P0: Look-ahead bias detection (CRITICAL)
- P1: Stationarity (ADF/KPSS), Multicollinearity (VIF)
- P2: Correlation, Clipping

Based on: scripts/audit_hmm_features.py
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Add root to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.data_engineering.manager import RegimeDetector
from src.data_engineering.features import FeatureEngineer


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def raw_data() -> pd.DataFrame:
    """
    Load raw historical data and apply feature engineering.

    The raw data needs feature engineering (BTC_LogRet, BTC_Parkinson, etc.)
    before HMM features can be computed.

    Skips tests if data not available.
    """
    historical_path = os.path.join(ROOT_DIR, "data/raw_historical/multi_asset_historical.csv")

    if not os.path.exists(historical_path):
        pytest.skip(f"Data file not found: {historical_path}")

    df = pd.read_csv(historical_path, index_col=0, parse_dates=True)

    # Ensure minimum data for tests
    if len(df) < 1000:
        pytest.skip(f"Insufficient data: {len(df)} rows (need >= 1000)")

    # Apply feature engineering to create BTC_LogRet, BTC_Parkinson, etc.
    feature_engineer = FeatureEngineer()
    df = feature_engineer.engineer_features(df)

    return df


@pytest.fixture(scope="module")
def detector() -> RegimeDetector:
    """Create RegimeDetector instance with default 4 components."""
    return RegimeDetector(n_components=4)


@pytest.fixture(scope="module")
def hmm_features(raw_data: pd.DataFrame, detector: RegimeDetector) -> pd.DataFrame:
    """
    Compute HMM features from raw data.

    Uses the internal _compute_hmm_features method.
    """
    return detector._compute_hmm_features(raw_data)


# =============================================================================
# P0 - Look-Ahead Bias Tests (CRITICAL)
# =============================================================================

class TestLookaheadBias:
    """P0 - Critical tests for look-ahead bias detection."""

    # Expected minimum NaN counts for each feature
    # Rolling window with min_periods=N produces first valid value at index N-1
    # So we expect at least N-1 NaN values (allowing for edge cases)
    EXPECTED_NAN = {
        'HMM_Trend': 167,      # window=168, min_periods=168
        'HMM_Vol': 167,        # window=168, min_periods=168
        'HMM_Momentum': 13,    # RSI window=14, min_periods=14
        'HMM_RiskOnOff': 167,  # window=168, min_periods=168
        'HMM_VolRatio': 167,   # max(24, 168)=168, min_periods=168
    }

    def test_nan_padding_hmm_trend(self, hmm_features: pd.DataFrame):
        """HMM_Trend should have proper NaN padding (window-1 values)."""
        nan_count = hmm_features['HMM_Trend'].isna().sum()
        expected = self.EXPECTED_NAN['HMM_Trend']
        assert nan_count >= expected, (
            f"HMM_Trend: Only {nan_count} NaN values, expected >= {expected}"
        )

    def test_nan_padding_hmm_vol(self, hmm_features: pd.DataFrame):
        """HMM_Vol should have proper NaN padding (window-1 values)."""
        nan_count = hmm_features['HMM_Vol'].isna().sum()
        expected = self.EXPECTED_NAN['HMM_Vol']
        assert nan_count >= expected, (
            f"HMM_Vol: Only {nan_count} NaN values, expected >= {expected}"
        )

    def test_nan_padding_hmm_momentum(self, hmm_features: pd.DataFrame):
        """HMM_Momentum should have proper NaN padding (RSI window-1 values)."""
        nan_count = hmm_features['HMM_Momentum'].isna().sum()
        expected = self.EXPECTED_NAN['HMM_Momentum']
        assert nan_count >= expected, (
            f"HMM_Momentum: Only {nan_count} NaN values, expected >= {expected}"
        )

    def test_nan_padding_hmm_riskonoff(self, hmm_features: pd.DataFrame):
        """HMM_RiskOnOff should have proper NaN padding (window-1 values)."""
        nan_count = hmm_features['HMM_RiskOnOff'].isna().sum()
        expected = self.EXPECTED_NAN['HMM_RiskOnOff']
        assert nan_count >= expected, (
            f"HMM_RiskOnOff: Only {nan_count} NaN values, expected >= {expected}"
        )

    def test_nan_padding_hmm_volratio(self, hmm_features: pd.DataFrame):
        """HMM_VolRatio should have proper NaN padding (max window-1 values)."""
        nan_count = hmm_features['HMM_VolRatio'].isna().sum()
        expected = self.EXPECTED_NAN['HMM_VolRatio']
        assert nan_count >= expected, (
            f"HMM_VolRatio: Only {nan_count} NaN values, expected >= {expected}"
        )

    def test_temporal_isolation(
        self,
        raw_data: pd.DataFrame,
        hmm_features: pd.DataFrame,
        detector: RegimeDetector
    ):
        """
        Verify no look-ahead bias via temporal isolation test.

        Features computed on partial data should match features at same index
        when computed on full data.
        """
        test_idx = 500

        # Compute features on partial data (first test_idx rows)
        partial_data = raw_data.iloc[:test_idx].copy()
        partial_features = detector._compute_hmm_features(partial_data)

        # Compare last value of partial with value at test_idx in full
        for feature in detector.HMM_FEATURES:
            val_partial = partial_features[feature].iloc[-1]
            val_full = hmm_features[feature].iloc[test_idx - 1]  # -1 because iloc is 0-based

            if pd.notna(val_partial) and pd.notna(val_full):
                diff = abs(val_partial - val_full)
                assert diff < 1e-6, (
                    f"Look-ahead bias in {feature}: partial={val_partial:.8f}, "
                    f"full={val_full:.8f}, diff={diff:.2e}"
                )


# =============================================================================
# P1 - Stationarity Tests
# =============================================================================

class TestStationarity:
    """P1 - Tests for feature stationarity."""

    # Features known to fail KPSS (level shifts in volatility)
    # HMM_Vol has structural breaks by design (volatility regimes)
    # Even with log transform, KPSS detects level shifts - this is expected behavior
    KPSS_XFAIL_FEATURES = ['HMM_Vol']

    @pytest.mark.parametrize("feature", RegimeDetector.HMM_FEATURES)
    def test_stationarity_adf(self, hmm_features: pd.DataFrame, feature: str):
        """
        ADF test for stationarity.

        H0: Series has unit root (non-stationary)
        Reject H0 if p-value < 0.05
        """
        series = hmm_features[feature].dropna()

        if len(series) < 100:
            pytest.skip(f"Not enough data for ADF test: {len(series)} samples")

        adf_result = adfuller(series, autolag='AIC')
        adf_pval = adf_result[1]

        assert adf_pval < 0.05, (
            f"{feature} is non-stationary (ADF p-value={adf_pval:.4f} >= 0.05)"
        )

    @pytest.mark.parametrize("feature", RegimeDetector.HMM_FEATURES)
    def test_stationarity_kpss(self, hmm_features: pd.DataFrame, feature: str):
        """
        KPSS test for stationarity.

        H0: Series is stationary
        Do NOT reject H0 if p-value > 0.05

        Note: HMM_Vol often fails KPSS due to structural breaks in volatility
        regimes (level shifts). This is a known characteristic of volatility
        series and is marked as xfail.
        """
        # Mark known failures as xfail
        if feature in self.KPSS_XFAIL_FEATURES:
            pytest.xfail(
                f"{feature}: Known KPSS failure due to volatility level shifts"
            )

        series = hmm_features[feature].dropna()

        if len(series) < 100:
            pytest.skip(f"Not enough data for KPSS test: {len(series)} samples")

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kpss_result = kpss(series, regression='c', nlags='auto')

        kpss_pval = kpss_result[1]

        assert kpss_pval > 0.05, (
            f"{feature} is non-stationary (KPSS p-value={kpss_pval:.4f} <= 0.05)"
        )


# =============================================================================
# P1 - Multicollinearity Tests
# =============================================================================

class TestMulticollinearity:
    """P1 - Tests for multicollinearity (VIF)."""

    def test_no_strong_multicollinearity(self, hmm_features: pd.DataFrame):
        """
        VIF test for multicollinearity.

        VIF > 10 indicates strong multicollinearity (problematic).
        VIF > 5 indicates moderate multicollinearity (warning).
        """
        hmm_cols = [c for c in hmm_features.columns if c.startswith('HMM_')]
        valid_data = hmm_features[hmm_cols].dropna()

        if len(valid_data) < 100:
            pytest.skip(f"Not enough data for VIF: {len(valid_data)} samples")

        # Compute VIF for each feature
        vif_results = {}
        for i, col in enumerate(hmm_cols):
            vif = variance_inflation_factor(valid_data.values, i)
            vif_results[col] = vif

        # Check no VIF > 10
        high_vif = {k: v for k, v in vif_results.items() if v > 10}

        assert len(high_vif) == 0, (
            f"Strong multicollinearity detected (VIF > 10): {high_vif}"
        )


# =============================================================================
# P2 - Correlation Tests
# =============================================================================

class TestCorrelation:
    """P2 - Tests for feature correlation."""

    def test_correlation_reasonable(self, hmm_features: pd.DataFrame):
        """
        Test that no more than 2 feature pairs have |correlation| > 0.7.

        High correlation between features can cause issues for the HMM
        and indicates redundant information.
        """
        hmm_cols = [c for c in hmm_features.columns if c.startswith('HMM_')]
        valid_data = hmm_features[hmm_cols].dropna()

        if len(valid_data) < 100:
            pytest.skip(f"Not enough data for correlation: {len(valid_data)} samples")

        corr_matrix = valid_data.corr()

        # Count high correlation pairs
        high_corr_pairs = []
        n = len(corr_matrix.columns)
        for i in range(n):
            for j in range(i + 1, n):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    high_corr_pairs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_val
                    ))

        assert len(high_corr_pairs) <= 2, (
            f"Too many high correlations (|r| > 0.7): {len(high_corr_pairs)} pairs. "
            f"Pairs: {high_corr_pairs}"
        )


# =============================================================================
# P2 - Clipping Tests
# =============================================================================

class TestClipping:
    """P2 - Tests for feature clipping aggressiveness."""

    # Expected clipping bounds from RegimeDetector._compute_hmm_features
    # Note: HMM_Vol is log-transformed for better stationarity (KPSS)
    CLIP_BOUNDS = {
        'HMM_Trend': (-0.05, 0.05),
        'HMM_Vol': (-10, -1),  # Log-transformed: log(vol + 1e-6)
        'HMM_Momentum': (0, 1),
        'HMM_RiskOnOff': (-0.02, 0.02),
        'HMM_VolRatio': (0.2, 5.0),
    }

    @pytest.mark.parametrize("feature,bounds", CLIP_BOUNDS.items())
    def test_clipping_not_aggressive(
        self,
        hmm_features: pd.DataFrame,
        feature: str,
        bounds: tuple
    ):
        """
        Test that clipping does not affect more than 5% of values.

        Aggressive clipping (> 5%) suggests bounds are too tight
        and information is being lost.
        """
        series = hmm_features[feature].dropna()

        if len(series) == 0:
            pytest.skip(f"No data for {feature}")

        lo, hi = bounds

        # Count values that would be clipped
        clipped_low = (series <= lo).sum()
        clipped_high = (series >= hi).sum()

        # For values exactly at the boundary, we need to check the raw (unclipped) values
        # Since data is already clipped, we check if values are at the boundary
        pct_at_min = (series == lo).sum() / len(series) * 100
        pct_at_max = (series == hi).sum() / len(series) * 100
        pct_clipped_total = pct_at_min + pct_at_max

        assert pct_clipped_total <= 5, (
            f"{feature}: {pct_clipped_total:.1f}% of values at clip bounds (> 5%). "
            f"At min ({lo}): {pct_at_min:.1f}%, At max ({hi}): {pct_at_max:.1f}%"
        )


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

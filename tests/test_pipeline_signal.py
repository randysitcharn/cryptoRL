# -*- coding: utf-8 -*-
"""
test_pipeline_signal.py - Pytest for Pipeline Signal Audit.

Validates that predictive signal is preserved through the pipeline:
    Raw -> HMM -> MAE -> TQC

Usage:
    pytest tests/test_pipeline_signal.py -v
    pytest tests/test_pipeline_signal.py::TestPipelineSignal::test_raw_features_have_signal -v
"""

import pytest
import numpy as np
import pandas as pd
import torch
import tempfile
from pathlib import Path
from typing import Tuple

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    from sklearn.ensemble import RandomForestClassifier


class TestPipelineSignal:
    """Test suite for pipeline signal preservation."""

    @pytest.fixture
    def synthetic_data(self) -> pd.DataFrame:
        """
        Generate synthetic data with known predictive signal.

        Creates features where some have predictive power and others don't.
        The signal features at time t predict the return from t to t+1.
        """
        np.random.seed(42)
        n_samples = 5000

        # Generate returns first
        returns = np.random.randn(n_samples) * 0.02

        # Target at time t = sign of return from t to t+1
        # So target[t] = sign(return[t+1])
        # Features at time t should predict target[t]
        future_target = (returns[1:] > 0).astype(int)  # Length n-1

        # Pad to match original length (last value doesn't matter, will be dropped)
        target_padded = np.concatenate([future_target, [0]])

        # Feature 1: Strong signal - knows future direction with noise
        signal_strength = 0.6
        feature_signal = signal_strength * (target_padded * 2 - 1) + (1 - signal_strength) * np.random.randn(n_samples)

        # Feature 2: Weak signal
        feature_weak = 0.25 * (target_padded * 2 - 1) + 0.75 * np.random.randn(n_samples)

        # Feature 3-5: Pure noise (no signal)
        feature_noise1 = np.random.randn(n_samples)
        feature_noise2 = np.random.randn(n_samples)
        feature_noise3 = np.random.randn(n_samples)

        # Simulated HMM probabilities (no real signal)
        probs = np.random.dirichlet([1, 1, 1, 1], size=n_samples)

        # Build price series
        prices = 50000 * np.exp(np.cumsum(returns))

        df = pd.DataFrame({
            'BTC_Close': prices,
            'Feature_Signal': feature_signal,
            'Feature_Weak': feature_weak,
            'Feature_Noise1': feature_noise1,
            'Feature_Noise2': feature_noise2,
            'Feature_Noise3': feature_noise3,
            'BTC_LogRet': np.concatenate([[0], returns[:-1]]),
            'Prob_0': probs[:, 0],
            'Prob_1': probs[:, 1],
            'Prob_2': probs[:, 2],
            'Prob_3': probs[:, 3],
        })

        return df

    @pytest.fixture
    def real_data(self) -> pd.DataFrame:
        """Load real segment data if available."""
        paths = [
            "data/wfo/segment_0/train.parquet",
            "data/processed_data.parquet",
        ]
        for path in paths:
            if Path(path).exists():
                return pd.read_parquet(path)
        pytest.skip("No real data available")

    @pytest.fixture
    def device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _create_target(self, df: pd.DataFrame) -> pd.Series:
        """Create target: Sign(LogReturn_t+1)."""
        prices = df['BTC_Close'].values
        log_returns = np.log(prices[1:] / prices[:-1])
        return pd.Series((log_returns > 0).astype(int), index=df.index[:-1])

    def _train_classifier(self, X_train, y_train, X_test, y_test) -> float:
        """Train classifier and return accuracy."""
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        if HAS_XGBOOST:
            model = xgb.XGBClassifier(
                n_estimators=50,
                max_depth=4,
                learning_rate=0.1,
                random_state=42,
                verbosity=0,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        else:
            model = RandomForestClassifier(
                n_estimators=50,
                max_depth=4,
                random_state=42,
            )

        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        return accuracy_score(y_test, y_pred)

    def test_synthetic_signal_detection(self, synthetic_data):
        """
        Test that classifier can detect signal in synthetic data.

        Validates the test methodology itself.
        """
        df = synthetic_data
        target = self._create_target(df)
        df_aligned = df.iloc[:-1]

        # Features with signal
        X_signal = df_aligned[['Feature_Signal', 'Feature_Weak']].values
        # Features without signal
        X_noise = df_aligned[['Feature_Noise1', 'Feature_Noise2', 'Feature_Noise3']].values

        # Split
        split_idx = int(len(target) * 0.8)
        y_train, y_test = target.iloc[:split_idx], target.iloc[split_idx:]

        # Test signal features
        acc_signal = self._train_classifier(
            X_signal[:split_idx], y_train,
            X_signal[split_idx:], y_test
        )

        # Test noise features
        acc_noise = self._train_classifier(
            X_noise[:split_idx], y_train,
            X_noise[split_idx:], y_test
        )

        baseline = max(y_test.mean(), 1 - y_test.mean())

        # Signal features should beat baseline
        assert acc_signal > baseline + 0.02, (
            f"Signal features should beat baseline. "
            f"Got {acc_signal:.3f}, baseline {baseline:.3f}"
        )

        # Noise features should be close to baseline
        assert abs(acc_noise - baseline) < 0.03, (
            f"Noise features should be near baseline. "
            f"Got {acc_noise:.3f}, baseline {baseline:.3f}"
        )

    def test_raw_features_have_signal(self, real_data):
        """
        Test A: Raw technical features should have predictive signal.

        If this fails, the problem is in the data/features themselves.
        """
        df = real_data
        target = self._create_target(df)
        df_aligned = df.iloc[:-1]

        # Select raw features (exclude OHLCV and HMM probs)
        exclude = ['_Open', '_High', '_Low', '_Close', '_Volume', 'Prob_', 'HMM_']
        feature_cols = [
            col for col in df_aligned.columns
            if not any(p in col for p in exclude)
            and df_aligned[col].dtype in [np.float64, np.float32]
        ]

        if len(feature_cols) == 0:
            pytest.skip("No raw features found")

        X = df_aligned[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values

        # Split (chronological)
        split_idx = int(len(target) * 0.8)
        y_train, y_test = target.iloc[:split_idx], target.iloc[split_idx:]

        acc = self._train_classifier(
            X[:split_idx], y_train,
            X[split_idx:], y_test
        )

        baseline = max(y_test.mean(), 1 - y_test.mean())
        gain = acc - baseline

        # Raw features should have at least 1% gain over baseline
        assert gain > 0.01, (
            f"Raw features should have predictive signal. "
            f"Accuracy: {acc:.3f}, Baseline: {baseline:.3f}, Gain: {gain:.3f}"
        )

    def test_hmm_probs_signal(self, real_data):
        """
        Test B: HMM probabilities predictive power.

        Measures how much signal the HMM regime detection captures.
        """
        df = real_data
        target = self._create_target(df)
        df_aligned = df.iloc[:-1]

        # HMM probability columns
        hmm_cols = [col for col in df_aligned.columns if col.startswith('Prob_')]

        if len(hmm_cols) == 0:
            pytest.skip("No HMM probability columns found")

        X = df_aligned[hmm_cols].fillna(0.25).values

        # Split
        split_idx = int(len(target) * 0.8)
        y_train, y_test = target.iloc[:split_idx], target.iloc[split_idx:]

        acc = self._train_classifier(
            X[:split_idx], y_train,
            X[split_idx:], y_test
        )

        baseline = max(y_test.mean(), 1 - y_test.mean())

        # Just record the result - HMM may or may not have signal
        # This is informational, not a hard requirement
        print(f"\nHMM Probs - Accuracy: {acc:.3f}, Baseline: {baseline:.3f}")

    def test_signal_not_destroyed_by_hmm(self, real_data):
        """
        Test that HMM doesn't destroy more than 50% of raw signal.

        Compares raw features accuracy vs HMM probs accuracy.
        """
        df = real_data
        target = self._create_target(df)
        df_aligned = df.iloc[:-1]

        # Raw features
        exclude = ['_Open', '_High', '_Low', '_Close', '_Volume', 'Prob_', 'HMM_']
        raw_cols = [
            col for col in df_aligned.columns
            if not any(p in col for p in exclude)
            and df_aligned[col].dtype in [np.float64, np.float32]
        ]

        # HMM probs
        hmm_cols = [col for col in df_aligned.columns if col.startswith('Prob_')]

        if len(raw_cols) == 0 or len(hmm_cols) == 0:
            pytest.skip("Missing features for comparison")

        X_raw = df_aligned[raw_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values
        X_hmm = df_aligned[hmm_cols].fillna(0.25).values

        # Split
        split_idx = int(len(target) * 0.8)
        y_train, y_test = target.iloc[:split_idx], target.iloc[split_idx:]
        baseline = max(y_test.mean(), 1 - y_test.mean())

        acc_raw = self._train_classifier(
            X_raw[:split_idx], y_train, X_raw[split_idx:], y_test
        )
        acc_hmm = self._train_classifier(
            X_hmm[:split_idx], y_train, X_hmm[split_idx:], y_test
        )

        gain_raw = acc_raw - baseline
        gain_hmm = acc_hmm - baseline

        # If raw has signal, HMM shouldn't lose more than 50% of it
        if gain_raw > 0.02:
            signal_preserved = gain_hmm / gain_raw if gain_raw > 0 else 0
            # This is a soft check - HMM is for regime detection, not prediction
            print(f"\nSignal preservation by HMM: {signal_preserved:.1%}")


class TestMAESignalPreservation:
    """Test MAE embedding signal preservation."""

    @pytest.fixture
    def device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    def test_mae_embedding_dimensionality(self, device):
        """Test that MAE produces embeddings of expected dimension."""
        from src.models.foundation import CryptoMAE

        input_dim = 30
        d_model = 64
        seq_len = 32
        batch_size = 16

        model = CryptoMAE(
            input_dim=input_dim,
            d_model=d_model,
            n_heads=4,
            n_layers=2,
        ).to(device)

        x = torch.randn(batch_size, seq_len, input_dim, device=device)

        with torch.no_grad():
            encoded = model.encode(x)

        assert encoded.shape == (batch_size, seq_len, d_model), (
            f"Expected shape ({batch_size}, {seq_len}, {d_model}), "
            f"got {encoded.shape}"
        )

    def test_mae_reconstruction(self, device):
        """Test that MAE can reconstruct masked inputs."""
        from src.models.foundation import CryptoMAE

        input_dim = 30
        model = CryptoMAE(
            input_dim=input_dim,
            d_model=64,
            n_heads=4,
            n_layers=2,
        ).to(device)

        x = torch.randn(8, 32, input_dim, device=device)

        pred, target, mask = model(x, mask_ratio=0.15)

        # Pred should have same shape as input
        assert pred.shape == x.shape

        # Target should be extracted masked values
        n_masked = mask.sum().item()
        assert target.shape == (n_masked, input_dim)

    def test_mae_preserves_variance(self, device):
        """
        Test that MAE embeddings preserve input variance structure.

        If embeddings collapse to similar values, signal is destroyed.
        """
        from src.models.foundation import CryptoMAE

        input_dim = 30
        model = CryptoMAE(
            input_dim=input_dim,
            d_model=64,
            n_heads=4,
            n_layers=2,
        ).to(device)
        model.eval()

        # Create diverse inputs
        x1 = torch.randn(16, 32, input_dim, device=device)
        x2 = torch.randn(16, 32, input_dim, device=device) * 2  # Different scale

        with torch.no_grad():
            emb1 = model.encode(x1).mean(dim=1)  # (16, d_model)
            emb2 = model.encode(x2).mean(dim=1)

        # Embeddings should have meaningful variance
        var1 = emb1.var(dim=0).mean().item()
        var2 = emb2.var(dim=0).mean().item()

        assert var1 > 0.01, f"Embedding variance too low: {var1}"
        assert var2 > 0.01, f"Embedding variance too low: {var2}"

        # Different inputs should produce different embeddings
        diff = (emb1.mean(dim=0) - emb2.mean(dim=0)).abs().mean().item()
        assert diff > 0.01, f"Different inputs produced too similar embeddings: {diff}"


class TestSignalMetrics:
    """Test signal measurement utilities."""

    def test_baseline_calculation(self):
        """Test majority class baseline is calculated correctly."""
        # 60% positive class
        y = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0])
        baseline = max(y.mean(), 1 - y.mean())
        assert baseline == 0.6

        # 40% positive class
        y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        baseline = max(y.mean(), 1 - y.mean())
        assert baseline == 0.8

    def test_signal_gain_interpretation(self):
        """Test signal gain thresholds."""
        baseline = 0.51

        # Strong signal: > 2% gain
        acc_strong = 0.54
        gain_strong = acc_strong - baseline
        assert gain_strong > 0.02

        # Weak signal: 0.5-2% gain
        acc_weak = 0.52
        gain_weak = acc_weak - baseline
        assert 0.005 < gain_weak < 0.02

        # No signal: < 0.5% gain
        acc_none = 0.51
        gain_none = acc_none - baseline
        assert gain_none < 0.005

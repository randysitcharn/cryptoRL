#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_feature_consistency.py - Tests for feature consistency between model and environment.

Detects mismatches between:
- Model observation space (trained with N features)
- Environment observation space (current data has M features)

This prevents silent evaluation failures where the model can't process observations.
"""

import pytest
import os
import numpy as np
from pathlib import Path


class TestFeatureConsistency:
    """Tests for feature count consistency."""

    def test_env_feature_count_matches_config(self):
        """Verify environment feature count matches expected constants."""
        from src.training.batch_env import BatchCryptoEnv
        from src.config.constants import EXCLUDE_COLS

        data_path = "data/processed_data.parquet"
        if not os.path.exists(data_path):
            pytest.skip(f"Data file not found: {data_path}")

        env = BatchCryptoEnv(
            parquet_path=data_path,
            n_envs=1,
            device='cpu',
            window_size=64,
            episode_length=100,
            price_column='BTC_Close',
        )

        obs, _ = env.gym_reset()
        n_features = obs['market'].shape[-1]
        env.close()

        # Log feature count for debugging
        print(f"Environment n_features: {n_features}")
        print(f"Feature names: {env.feature_names[:5]}...") if hasattr(env, 'feature_names') else None

        # Feature count should be stable across runs
        # Update this value when features change intentionally
        EXPECTED_MIN_FEATURES = 30
        EXPECTED_MAX_FEATURES = 50

        assert n_features >= EXPECTED_MIN_FEATURES, \
            f"Too few features: {n_features} < {EXPECTED_MIN_FEATURES}"
        assert n_features <= EXPECTED_MAX_FEATURES, \
            f"Too many features: {n_features} > {EXPECTED_MAX_FEATURES}"

    def test_model_env_feature_match(self):
        """
        CRITICAL: Verify saved model observation space matches current environment.

        This test prevents the silent failure where:
        - Model trained with N features
        - Evaluation runs with M features (M != N)
        - Model.predict() fails or produces garbage
        """
        from src.training.batch_env import BatchCryptoEnv

        data_path = "data/processed_data.parquet"
        model_path = "weights/wfo/segment_0/tqc.zip"

        if not os.path.exists(data_path):
            pytest.skip(f"Data file not found: {data_path}")
        if not os.path.exists(model_path):
            pytest.skip(f"Model file not found: {model_path}")

        # Get model observation space
        from sb3_contrib import TQC
        model = TQC.load(model_path, device='cpu')
        model_market_shape = model.observation_space['market'].shape
        model_n_features = model_market_shape[-1]

        # Get environment observation space
        env = BatchCryptoEnv(
            parquet_path=data_path,
            n_envs=1,
            device='cpu',
            window_size=model_market_shape[0],  # Use model's window size
            episode_length=100,
            price_column='BTC_Close',
        )

        obs, _ = env.gym_reset()
        env_n_features = obs['market'].shape[-1]
        env.close()

        print(f"Model expects: {model_n_features} features")
        print(f"Environment provides: {env_n_features} features")

        assert model_n_features == env_n_features, \
            f"FEATURE MISMATCH! Model expects {model_n_features} features, " \
            f"but environment provides {env_n_features}. " \
            f"This will cause evaluation to fail silently."

    def test_model_can_predict_on_current_env(self):
        """
        Integration test: Verify model can actually make predictions on current env.

        This catches issues that simple shape checks might miss.
        """
        from src.training.batch_env import BatchCryptoEnv

        data_path = "data/processed_data.parquet"
        model_path = "weights/wfo/segment_0/tqc.zip"

        if not os.path.exists(data_path):
            pytest.skip(f"Data file not found: {data_path}")
        if not os.path.exists(model_path):
            pytest.skip(f"Model file not found: {model_path}")

        from sb3_contrib import TQC
        model = TQC.load(model_path, device='cpu')
        model_window_size = model.observation_space['market'].shape[0]
        model_n_features = model.observation_space['market'].shape[-1]

        # Create env matching model's expected observation space
        env = BatchCryptoEnv(
            parquet_path=data_path,
            n_envs=1,
            device='cpu',
            window_size=model_window_size,
            episode_length=100,
            price_column='BTC_Close',
        )

        obs, _ = env.gym_reset()
        env_n_features = obs['market'].shape[-1]

        # Skip if feature mismatch (covered by other test)
        if model_n_features != env_n_features:
            env.close()
            pytest.skip(f"Feature mismatch: model={model_n_features}, env={env_n_features}")

        # Try to predict
        try:
            for i in range(5):
                action, _ = model.predict(obs, deterministic=True)
                assert action.shape == (1,), f"Unexpected action shape: {action.shape}"
                assert -1.0 <= action[0] <= 1.0, f"Action out of range: {action[0]}"
                obs, _, _, _, _ = env.gym_step(action)
        except Exception as e:
            env.close()
            pytest.fail(f"Model failed to predict on current env: {e}")

        env.close()


class TestFeatureEngineering:
    """Tests for feature engineering consistency."""

    def test_exclude_cols_applied(self):
        """Verify EXCLUDE_COLS are not in final features."""
        from src.training.batch_env import BatchCryptoEnv
        from src.config.constants import EXCLUDE_COLS

        data_path = "data/processed_data.parquet"
        if not os.path.exists(data_path):
            pytest.skip(f"Data file not found: {data_path}")

        env = BatchCryptoEnv(
            parquet_path=data_path,
            n_envs=1,
            device='cpu',
            window_size=64,
            episode_length=100,
            price_column='BTC_Close',
        )

        feature_names = env.feature_names if hasattr(env, 'feature_names') else []
        env.close()

        if not feature_names:
            pytest.skip("Environment doesn't expose feature_names")

        # Check no excluded columns are in features
        for col in EXCLUDE_COLS:
            assert col not in feature_names, \
                f"Excluded column '{col}' found in features. " \
                f"This may cause data leakage or inconsistency."

    def test_no_nan_in_features(self):
        """Verify no NaN values in feature observations."""
        from src.training.batch_env import BatchCryptoEnv

        data_path = "data/processed_data.parquet"
        if not os.path.exists(data_path):
            pytest.skip(f"Data file not found: {data_path}")

        env = BatchCryptoEnv(
            parquet_path=data_path,
            n_envs=1,
            device='cpu',
            window_size=64,
            episode_length=100,
            price_column='BTC_Close',
        )

        obs, _ = env.gym_reset()

        # Check for NaN in market features
        market_obs = obs['market']
        nan_count = np.isnan(market_obs).sum()

        env.close()

        assert nan_count == 0, \
            f"Found {nan_count} NaN values in market observations. " \
            f"This will cause model training/inference to fail."


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

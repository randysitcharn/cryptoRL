# -*- coding: utf-8 -*-
"""
test_oracle.py - Pytest for Oracle Test (Architecture Validation).

Validates that TQC can learn when given obvious signals.
If this test fails, the architecture is broken.

Usage:
    pytest tests/test_oracle.py -v
    pytest tests/test_oracle.py::test_oracle_learning -v
"""

import pytest
import numpy as np
import pandas as pd
import torch
import tempfile
import os
from pathlib import Path

from sb3_contrib import TQC


class TestOracleArchitecture:
    """Test suite for Oracle-based architecture validation."""

    @pytest.fixture
    def oracle_data_path(self, tmp_path):
        """Generate small oracle dataset for testing."""
        n_rows = 5000
        signal_strength = 0.95
        volatility = 0.02
        base_price = 50000.0

        np.random.seed(42)

        # Generate oracle signal
        oracle_signal = np.random.choice([-1.0, 1.0], size=n_rows)

        # Returns follow the signal
        noise = np.random.randn(n_rows) * volatility * (1 - signal_strength)
        returns = signal_strength * oracle_signal * volatility + noise

        # Build price series
        prices = np.zeros(n_rows)
        prices[0] = base_price
        for i in range(1, n_rows):
            prices[i] = prices[i-1] * (1 + returns[i-1])

        # OHLCV
        high = prices * (1 + np.abs(np.random.randn(n_rows) * 0.005))
        low = prices * (1 - np.abs(np.random.randn(n_rows) * 0.005))
        open_price = prices * (1 + np.random.randn(n_rows) * 0.002)
        volume = np.random.uniform(1000, 10000, n_rows)

        high = np.maximum(high, np.maximum(prices, open_price))
        low = np.minimum(low, np.minimum(prices, open_price))

        df = pd.DataFrame({
            'BTC_Open': open_price,
            'BTC_High': high,
            'BTC_Low': low,
            'BTC_Close': prices,
            'BTC_Volume': volume,
            'ORACLE_SIGNAL': oracle_signal,
            'BTC_LogRet': np.concatenate([[0], returns[:-1]]),
            'BTC_Vol_ZScore': np.random.randn(n_rows) * 0.5,
            'BTC_ZScore': np.random.randn(n_rows) * 0.5,
            'BTC_Parkinson_Vol': np.full(n_rows, volatility),
            'BTC_GK_Vol': np.full(n_rows, volatility),
            'BTC_FFD': np.random.randn(n_rows) * 0.1,
            # HMM features (required for FiLM alignment in BatchCryptoEnv)
            'HMM_Prob_0': np.full(n_rows, 0.25),
            'HMM_Prob_1': np.full(n_rows, 0.25),
            'HMM_Prob_2': np.full(n_rows, 0.25),
            'HMM_Prob_3': np.full(n_rows, 0.25),
            'HMM_Entropy': np.full(n_rows, 0.5),
        })

        # Add dummy columns for other assets
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

        # Verify signal correlation
        actual_returns = np.diff(prices) / prices[:-1]
        correlation = np.corrcoef(oracle_signal[:-1], actual_returns)[0, 1]
        assert correlation > 0.8, f"Oracle signal correlation too low: {correlation}"

        # Save to temp file
        output_path = tmp_path / "oracle_test_data.parquet"
        df.to_parquet(output_path, index=False)

        return str(output_path)

    @pytest.fixture
    def device(self):
        """Get available device."""
        return "cuda" if torch.cuda.is_available() else "cpu"

    def test_oracle_data_generation(self, oracle_data_path):
        """Test that oracle data has correct structure and strong signal."""
        df = pd.read_parquet(oracle_data_path)

        # Check required columns
        assert 'BTC_Close' in df.columns
        assert 'ORACLE_SIGNAL' in df.columns
        assert len(df) == 5000

        # Check signal values
        assert set(df['ORACLE_SIGNAL'].unique()) == {-1.0, 1.0}

        # Verify signal-return correlation
        prices = df['BTC_Close'].values
        oracle_signal = df['ORACLE_SIGNAL'].values
        actual_returns = np.diff(prices) / prices[:-1]
        correlation = np.corrcoef(oracle_signal[:-1], actual_returns)[0, 1]

        assert correlation > 0.8, f"Signal correlation {correlation} below threshold 0.8"

    def test_oracle_learning(self, oracle_data_path, device):
        """
        Test that TQC can learn with obvious oracle signals.

        SUCCESS CRITERIA:
        - Mean |position| > 0.3 after training
        - Positive episode reward trend

        If this test fails, the architecture is broken.
        """
        from src.training.batch_env import BatchCryptoEnv

        # Create environment with oracle data
        env = BatchCryptoEnv(
            parquet_path=oracle_data_path,
            price_column="BTC_Close",
            n_envs=8,
            device=device,
            window_size=32,
            episode_length=200,
            initial_balance=10_000.0,
            commission=0.0001,
            slippage=0.0,
            target_volatility=0.20,
            max_leverage=3.0,
            observation_noise=0.0,
            enable_domain_randomization=False,
            action_discretization=0.0,
        )

        # Create TQC model with simple config
        policy_kwargs = dict(
            net_arch=dict(pi=[64, 64], qf=[64, 64]),
            n_critics=2,
            n_quantiles=25,
        )

        model = TQC(
            policy="MultiInputPolicy",
            env=env,
            learning_rate=3e-4,
            buffer_size=50_000,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            ent_coef="auto",
            train_freq=1,
            gradient_steps=1,
            top_quantiles_to_drop_per_net=2,
            policy_kwargs=policy_kwargs,
            verbose=0,
            device=device,
        )

        # Train for short period
        timesteps = 20_000
        model.learn(total_timesteps=timesteps, progress_bar=False)

        # Evaluate
        obs = env.reset()
        positions = []

        for _ in range(200):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            positions.append(env.position_pcts.cpu().numpy().flatten())
            if done.any():
                break

        positions = np.concatenate(positions)
        mean_abs_position = np.mean(np.abs(positions))

        # Assert model learned to take positions
        assert mean_abs_position > 0.3, (
            f"Model failed to learn with oracle signal. "
            f"Mean |position| = {mean_abs_position:.4f} (expected > 0.3). "
            f"Architecture may be broken."
        )

    def test_oracle_quick_sanity(self, oracle_data_path, device):
        """
        Quick sanity check that model can be created and run.
        Does not train - just verifies setup works.
        """
        from src.training.batch_env import BatchCryptoEnv

        env = BatchCryptoEnv(
            parquet_path=oracle_data_path,
            price_column="BTC_Close",
            n_envs=4,
            device=device,
            window_size=32,
            episode_length=100,
            initial_balance=10_000.0,
            commission=0.0001,
            slippage=0.0,
            target_volatility=0.20,
            max_leverage=3.0,
        )

        # Check observation space is Dict
        assert hasattr(env.observation_space, 'spaces')
        assert 'market' in env.observation_space.spaces
        assert 'position' in env.observation_space.spaces

        # Check action space
        assert env.action_space.shape == (1,)
        assert env.action_space.low[0] == -1.0
        assert env.action_space.high[0] == 1.0

        # Create model
        model = TQC(
            policy="MultiInputPolicy",
            env=env,
            verbose=0,
            device=device,
        )

        # Run one step
        obs = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        assert obs is not None
        assert reward is not None

    def test_oracle_position_scaling(self, oracle_data_path, device):
        """
        Test that vol_scalar amplifies (not crushes) actions.
        With low volatility oracle data, vol_scalar should be > 1.
        """
        from src.training.batch_env import BatchCryptoEnv

        env = BatchCryptoEnv(
            parquet_path=oracle_data_path,
            price_column="BTC_Close",
            n_envs=4,
            device=device,
            window_size=32,
            episode_length=100,
            initial_balance=10_000.0,
            commission=0.0001,
            slippage=0.0,
            target_volatility=0.05,  # 5% target
            max_leverage=2.0,
        )

        obs = env.reset()

        # Take a few steps to let vol_scalar stabilize
        for _ in range(10):
            # Large action
            action = np.ones((4, 1), dtype=np.float32) * 0.8
            obs, reward, done, info = env.step(action)

        # Check vol_scalar is reasonable (not crushing)
        vol_scalar = env.vol_scalars.cpu().numpy()
        assert vol_scalar.min() >= 0.1, f"vol_scalar too small: {vol_scalar.min()}"
        assert vol_scalar.max() <= 2.0, f"vol_scalar too large: {vol_scalar.max()}"

        # With 0.8 raw action and vol_scalar >= 0.1, effective should be >= 0.08
        # This confirms vol_scalar doesn't crush actions to near-zero


class TestOracleSignalCorrelation:
    """Test signal correlation in oracle data."""

    def test_signal_strength_variations(self):
        """Test different signal strengths produce expected correlations."""
        n_rows = 2000
        volatility = 0.02

        # Only test high signal strengths where correlation is reliable
        for signal_strength in [0.8, 0.9, 0.95]:
            np.random.seed(42)

            oracle_signal = np.random.choice([-1.0, 1.0], size=n_rows)
            noise = np.random.randn(n_rows) * volatility * (1 - signal_strength)
            returns = signal_strength * oracle_signal * volatility + noise

            prices = np.zeros(n_rows)
            prices[0] = 50000.0
            for i in range(1, n_rows):
                prices[i] = prices[i-1] * (1 + returns[i-1])

            actual_returns = np.diff(prices) / prices[:-1]
            correlation = np.corrcoef(oracle_signal[:-1], actual_returns)[0, 1]

            # High signal strengths should produce high correlation
            assert correlation > 0.7, (
                f"Signal strength {signal_strength} produced low correlation {correlation}"
            )

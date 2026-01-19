# -*- coding: utf-8 -*-
"""
test_plo.py - Unit tests for PLO (Predictive Lagrangian Optimization) components.

Tests cover:
1. PLO Drawdown (PLOAdaptivePenaltyCallback)
2. PLO Churn (PLOChurnCallback)
3. PLO Smoothness (PLOSmoothnessCallback)
4. BatchCryptoEnv PLO integration
"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, MagicMock, patch

# Import PLO callbacks
from src.training.callbacks import (
    PLOAdaptivePenaltyCallback,
    PLOChurnCallback,
    PLOSmoothnessCallback,
    get_underlying_batch_env,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def mock_env():
    """Create a mock BatchCryptoEnv for testing."""
    env = Mock()
    env.num_envs = 64
    env.curriculum_lambda = 0.4  # Phase 3 (active)
    env._current_churn_coef = 0.5
    env._current_smooth_coef = 0.005
    env.current_drawdowns = torch.tensor([0.05] * 64)
    env.current_position_deltas = torch.tensor([0.05] * 64)
    env.current_jerks = torch.tensor([0.1] * 64)
    env.downside_multiplier = 1.0
    env.churn_multiplier = 1.0
    env.smooth_multiplier = 1.0
    env.set_downside_multiplier = Mock()
    env.set_churn_multiplier = Mock()
    env.set_smooth_multiplier = Mock()
    return env


@pytest.fixture
def mock_model(mock_env):
    """Create a mock SB3 model."""
    model = Mock()
    model.env = mock_env
    model.logger = Mock()
    model.logger.record = Mock()
    return model


# ============================================================================
# PLO Drawdown Tests
# ============================================================================

class TestPLODrawdown:
    """Tests for PLOAdaptivePenaltyCallback."""

    def test_lambda_increases_on_violation(self, mock_model, mock_env):
        """λ should increase when drawdown > threshold."""
        callback = PLOAdaptivePenaltyCallback(dd_threshold=0.10, verbose=0)
        callback.model = mock_model
        callback.logger = mock_model.logger
        callback.num_timesteps = 100

        # Simulate DD = 15% (5% above threshold)
        mock_env.current_drawdowns = torch.tensor([0.15] * 64)

        # Run multiple steps to accumulate
        for _ in range(10):
            callback._on_step()

        # Lambda should have increased
        assert callback.dd_lambda > 1.0, f"λ should increase on violation, got {callback.dd_lambda}"

    def test_lambda_decreases_when_ok(self, mock_model, mock_env):
        """λ should decay when drawdown < threshold."""
        callback = PLOAdaptivePenaltyCallback(dd_threshold=0.10, decay_rate=0.99, verbose=0)
        callback.model = mock_model
        callback.logger = mock_model.logger
        callback.num_timesteps = 100
        callback.dd_lambda = 3.0  # Start high

        # Simulate DD = 5% (below threshold)
        mock_env.current_drawdowns = torch.tensor([0.05] * 64)

        initial_lambda = callback.dd_lambda
        for _ in range(50):
            callback._on_step()

        # Lambda should have decreased
        assert callback.dd_lambda < initial_lambda, f"λ should decay, was {initial_lambda}, got {callback.dd_lambda}"

    def test_lambda_bounds(self, mock_model, mock_env):
        """λ should stay within [λ_min, λ_max]."""
        callback = PLOAdaptivePenaltyCallback(
            dd_threshold=0.10,
            dd_lambda_min=1.0,
            dd_lambda_max=5.0,
            dd_Kp=100.0,  # Very aggressive to test bounds
            verbose=0
        )
        callback.model = mock_model
        callback.logger = mock_model.logger
        callback.num_timesteps = 100

        # Simulate extreme violation
        mock_env.current_drawdowns = torch.tensor([0.50] * 64)  # 50% DD

        for _ in range(100):
            callback._on_step()

        assert callback.dd_lambda >= 1.0, f"λ below min: {callback.dd_lambda}"
        assert callback.dd_lambda <= 5.0, f"λ above max: {callback.dd_lambda}"

    def test_smoothing_respected(self, mock_model, mock_env):
        """λ should not change more than max_lambda_change per step."""
        callback = PLOAdaptivePenaltyCallback(
            dd_threshold=0.10,
            max_lambda_change=0.05,
            dd_Kp=10.0,
            verbose=0
        )
        callback.model = mock_model
        callback.logger = mock_model.logger
        callback.num_timesteps = 100

        # Simulate sudden large violation
        mock_env.current_drawdowns = torch.tensor([0.30] * 64)

        initial_lambda = callback.dd_lambda
        callback._on_step()

        change = abs(callback.dd_lambda - initial_lambda)
        assert change <= 0.05 + 1e-6, f"Change {change} exceeds max 0.05"

    def test_prediction_only_on_positive_slope(self, mock_model, mock_env):
        """Prediction should only activate if slope is positive (worsening)."""
        callback = PLOAdaptivePenaltyCallback(
            dd_threshold=0.10,
            use_prediction=True,
            verbose=0
        )
        callback.model = mock_model
        callback.logger = mock_model.logger
        callback.num_timesteps = 100

        # Fill history with DECREASING drawdown (recovery)
        callback.dd_history = [0.20 - i * 0.01 for i in range(20)]

        # Simulate current DD just below threshold (recovering)
        mock_env.current_drawdowns = torch.tensor([0.08] * 64)

        callback._on_step()

        # With decreasing trend, prediction should not trigger
        # Lambda should be near minimum or decaying
        assert callback.dd_lambda < 2.0, f"Should not predict on recovery, λ={callback.dd_lambda}"

    def test_wakeup_shock_protection(self, mock_model, mock_env):
        """PLO should not accumulate integral in Phase 1 (curriculum ≈ 0)."""
        callback = PLOAdaptivePenaltyCallback(dd_threshold=0.10, verbose=0)
        callback.model = mock_model
        callback.logger = mock_model.logger
        callback.num_timesteps = 100

        # Phase 1: curriculum_lambda ≈ 0
        mock_env.curriculum_lambda = 0.01

        # Simulate violation
        mock_env.current_drawdowns = torch.tensor([0.20] * 64)

        # Prime the integral
        callback.dd_integral = 1.0

        for _ in range(10):
            callback._on_step()

        # Integral should have decayed, not accumulated
        assert callback.dd_integral < 1.0, f"Integral should decay in Phase 1, got {callback.dd_integral}"


# ============================================================================
# PLO Churn Tests
# ============================================================================

class TestPLOChurn:
    """Tests for PLOChurnCallback."""

    def test_lambda_increases_on_high_turnover(self, mock_model, mock_env):
        """λ should increase when turnover > threshold."""
        callback = PLOChurnCallback(turnover_threshold=0.08, verbose=0)
        callback.model = mock_model
        callback.logger = mock_model.logger
        callback.num_timesteps = 100

        # Simulate high turnover (30%)
        mock_env.current_position_deltas = torch.tensor([0.30] * 64)

        for _ in range(20):
            callback._on_step()

        assert callback.turnover_lambda > 1.0, f"λ should increase on high turnover, got {callback.turnover_lambda}"

    def test_lambda_bounds(self, mock_model, mock_env):
        """λ should stay within bounds."""
        callback = PLOChurnCallback(
            turnover_threshold=0.08,
            turnover_lambda_min=1.0,
            turnover_lambda_max=5.0,
            turnover_Kp=50.0,
            verbose=0
        )
        callback.model = mock_model
        callback.logger = mock_model.logger
        callback.num_timesteps = 100

        # Extreme turnover
        mock_env.current_position_deltas = torch.tensor([1.0] * 64)

        for _ in range(100):
            callback._on_step()

        assert 1.0 <= callback.turnover_lambda <= 5.0

    def test_curriculum_protection(self, mock_model, mock_env):
        """PLO should decay integral if churn_coef ≈ 0."""
        callback = PLOChurnCallback(turnover_threshold=0.08, verbose=0)
        callback.model = mock_model
        callback.logger = mock_model.logger
        callback.num_timesteps = 100

        # Curriculum not active
        mock_env._current_churn_coef = 0.01

        callback.turnover_integral = 1.0

        for _ in range(10):
            callback._on_step()

        assert callback.turnover_integral < 1.0, "Integral should decay when curriculum inactive"

    def test_prediction_with_min_slope(self, mock_model, mock_env):
        """Prediction should ignore negligible slopes."""
        callback = PLOChurnCallback(
            turnover_threshold=0.08,
            use_prediction=True,
            verbose=0
        )
        callback.model = mock_model
        callback.logger = mock_model.logger
        callback.num_timesteps = 100

        # Fill history with nearly flat turnover (slope < 0.001)
        callback.turnover_history = [0.10 + 0.0001 * i for i in range(20)]

        mock_env.current_position_deltas = torch.tensor([0.10] * 64)

        callback._on_step()

        # Should not panic on tiny slope
        assert callback.turnover_lambda < 3.0, "Should not over-react to tiny slope"


# ============================================================================
# PLO Smoothness Tests
# ============================================================================

class TestPLOSmoothness:
    """Tests for PLOSmoothnessCallback."""

    def test_lambda_increases_on_high_jerk(self, mock_model, mock_env):
        """λ should increase when jerk > threshold."""
        callback = PLOSmoothnessCallback(jerk_threshold=0.40, verbose=0)
        callback.model = mock_model
        callback.logger = mock_model.logger
        callback.num_timesteps = 100

        # Simulate high jerk (60%)
        mock_env.current_jerks = torch.tensor([0.60] * 64)

        for _ in range(10):
            callback._on_step()

        assert callback.jerk_lambda > 1.0, f"λ should increase on high jerk, got {callback.jerk_lambda}"

    def test_lambda_stays_neutral_on_low_jerk(self, mock_model, mock_env):
        """λ should stay near 1.0 if jerk < threshold."""
        callback = PLOSmoothnessCallback(jerk_threshold=0.40, verbose=0)
        callback.model = mock_model
        callback.logger = mock_model.logger
        callback.num_timesteps = 100

        # Simulate low jerk (10%)
        mock_env.current_jerks = torch.tensor([0.10] * 64)

        for _ in range(50):
            callback._on_step()

        # Should be at or near minimum
        assert callback.jerk_lambda <= 1.1, f"λ should stay near 1.0, got {callback.jerk_lambda}"

    def test_lambda_bounds(self, mock_model, mock_env):
        """λ should stay within bounds."""
        callback = PLOSmoothnessCallback(
            jerk_threshold=0.40,
            jerk_lambda_min=1.0,
            jerk_lambda_max=5.0,
            jerk_Kp=50.0,
            verbose=0
        )
        callback.model = mock_model
        callback.logger = mock_model.logger
        callback.num_timesteps = 100

        # Extreme jerk
        mock_env.current_jerks = torch.tensor([2.0] * 64)

        for _ in range(100):
            callback._on_step()

        assert 1.0 <= callback.jerk_lambda <= 5.0

    def test_faster_decay_than_drawdown(self, mock_model, mock_env):
        """Smoothness should have faster decay (0.99 vs 0.995)."""
        callback = PLOSmoothnessCallback(
            jerk_threshold=0.40,
            decay_rate=0.99,
            verbose=0
        )
        callback.model = mock_model
        callback.logger = mock_model.logger
        callback.num_timesteps = 100
        callback.jerk_lambda = 3.0

        # Low jerk → should decay
        mock_env.current_jerks = torch.tensor([0.10] * 64)

        # Run 100 steps
        for _ in range(100):
            callback._on_step()

        # Should decay faster than drawdown
        assert callback.jerk_lambda < 1.5, f"Should decay quickly, got {callback.jerk_lambda}"

    def test_curriculum_protection(self, mock_model, mock_env):
        """PLO should not activate if smooth_coef == 0."""
        callback = PLOSmoothnessCallback(jerk_threshold=0.40, verbose=0)
        callback.model = mock_model
        callback.logger = mock_model.logger
        callback.num_timesteps = 100

        # Curriculum not active for smoothness
        mock_env._current_smooth_coef = 0.0

        callback.jerk_integral = 1.0
        mock_env.current_jerks = torch.tensor([0.80] * 64)  # High jerk

        for _ in range(10):
            callback._on_step()

        # Should have decayed, not accumulated
        assert callback.jerk_integral < 1.0, "Integral should decay when curriculum inactive"


# ============================================================================
# BatchCryptoEnv PLO Integration Tests
# ============================================================================

class TestBatchEnvPLOIntegration:
    """Tests for PLO integration in BatchCryptoEnv."""

    @pytest.fixture
    def simple_env(self, tmp_path):
        """Create a minimal BatchCryptoEnv for integration testing."""
        # Skip if data file doesn't exist
        pytest.importorskip("src.training.batch_env")

        # Create a minimal parquet file for testing
        import pandas as pd
        data = pd.DataFrame({
            'BTC_Close': np.linspace(10000, 11000, 1000),
            'feature_1': np.random.randn(1000),
            'feature_2': np.random.randn(1000),
        })
        data_path = tmp_path / "test_data.parquet"
        data.to_parquet(data_path)

        from src.training.batch_env import BatchCryptoEnv
        env = BatchCryptoEnv(
            parquet_path=str(data_path),
            price_column='BTC_Close',
            n_envs=4,
            device='cpu',
            window_size=10,
            episode_length=100,
        )
        return env

    def test_observation_space_has_plo_keys(self, simple_env):
        """Observation space should include PLO level keys."""
        obs_space = simple_env.observation_space
        assert "risk_level" in obs_space.spaces, "Missing risk_level in obs space"
        assert "churn_level" in obs_space.spaces, "Missing churn_level in obs space"
        assert "smooth_level" in obs_space.spaces, "Missing smooth_level in obs space"

    def test_observations_include_plo_levels(self, simple_env):
        """Reset should return observations with PLO levels."""
        obs = simple_env.reset()
        assert "risk_level" in obs, "Missing risk_level in observations"
        assert "churn_level" in obs, "Missing churn_level in observations"
        assert "smooth_level" in obs, "Missing smooth_level in observations"

        # Initial values should be 0 (λ=1.0 → normalized=0)
        assert np.allclose(obs["risk_level"], 0.0), f"Initial risk_level should be 0, got {obs['risk_level']}"
        assert np.allclose(obs["churn_level"], 0.0), f"Initial churn_level should be 0, got {obs['churn_level']}"
        assert np.allclose(obs["smooth_level"], 0.0), f"Initial smooth_level should be 0, got {obs['smooth_level']}"

    def test_plo_setters_work(self, simple_env):
        """PLO setters should update multipliers."""
        simple_env.set_downside_multiplier(3.0)
        assert simple_env.downside_multiplier == 3.0

        simple_env.set_churn_multiplier(2.5)
        assert simple_env.churn_multiplier == 2.5

        simple_env.set_smooth_multiplier(4.0)
        assert simple_env.smooth_multiplier == 4.0

    def test_plo_setters_clamp_values(self, simple_env):
        """PLO setters should clamp to valid range."""
        simple_env.set_downside_multiplier(0.5)  # Below min
        assert simple_env.downside_multiplier == 1.0

        simple_env.set_downside_multiplier(15.0)  # Above max
        assert simple_env.downside_multiplier == 10.0

    def test_plo_levels_reflect_multipliers(self, simple_env):
        """Observations should reflect current PLO multipliers."""
        simple_env.reset()
        simple_env.set_downside_multiplier(3.0)  # normalized: (3-1)/4 = 0.5
        simple_env.set_churn_multiplier(5.0)     # normalized: (5-1)/4 = 1.0
        simple_env.set_smooth_multiplier(2.0)    # normalized: (2-1)/4 = 0.25

        # Step to get new observations
        actions = np.zeros((simple_env.num_envs, 1), dtype=np.float32)
        simple_env.step_async(actions)
        obs, _, _, _ = simple_env.step_wait()

        expected_risk = (3.0 - 1.0) / 4.0
        expected_churn = (5.0 - 1.0) / 4.0
        expected_smooth = (2.0 - 1.0) / 4.0

        assert np.allclose(obs["risk_level"], expected_risk, atol=1e-5)
        assert np.allclose(obs["churn_level"], expected_churn, atol=1e-5)
        assert np.allclose(obs["smooth_level"], expected_smooth, atol=1e-5)

    def test_jerk_buffer_not_always_zero(self, simple_env):
        """AUDIT v1.1: latest_jerks buffer should not always be 0."""
        simple_env.reset()

        # Step 1: Position change
        actions1 = np.full((simple_env.num_envs, 1), 0.5, dtype=np.float32)
        simple_env.step_async(actions1)
        simple_env.step_wait()

        # Step 2: Opposite position change (creates jerk)
        actions2 = np.full((simple_env.num_envs, 1), -0.5, dtype=np.float32)
        simple_env.step_async(actions2)
        simple_env.step_wait()

        # Jerk should be non-zero for at least some envs
        max_jerk = simple_env.current_jerks.max().item()
        assert max_jerk > 0, "Off-by-one bug: jerk buffer always 0!"

    def test_jerk_reset_on_episode_end(self, simple_env):
        """Jerk buffers should reset when episode ends."""
        simple_env.reset()

        # Create some jerk history
        simple_env.prev_position_deltas = torch.ones(simple_env.num_envs)
        simple_env.latest_jerks = torch.ones(simple_env.num_envs)

        # Reset
        simple_env.reset()

        assert torch.allclose(simple_env.prev_position_deltas, torch.zeros(simple_env.num_envs))
        assert torch.allclose(simple_env.latest_jerks, torch.zeros(simple_env.num_envs))


# ============================================================================
# Utility Function Tests
# ============================================================================

class TestUtilityFunctions:
    """Tests for PLO utility functions."""

    def test_get_underlying_batch_env_direct(self):
        """Should return env directly if it has PLO setters."""
        env = Mock()
        env.set_smoothness_penalty = Mock()

        result = get_underlying_batch_env(env)
        assert result is env

    def test_get_underlying_batch_env_wrapped(self):
        """Should unwrap to find BatchCryptoEnv."""
        inner_env = Mock()
        inner_env.set_smoothness_penalty = Mock()

        wrapper1 = Mock()
        wrapper1.venv = inner_env
        del wrapper1.set_smoothness_penalty  # Ensure it doesn't match

        wrapper2 = Mock()
        wrapper2.venv = wrapper1
        del wrapper2.set_smoothness_penalty

        result = get_underlying_batch_env(wrapper2)
        assert result is inner_env


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

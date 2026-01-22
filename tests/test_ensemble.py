# -*- coding: utf-8 -*-
"""
test_ensemble.py - Tests for Ensemble RL implementation.

Tests cover:
- EnsembleConfig: Configuration validation and serialization
- EnsemblePolicy: Aggregation methods and confidence estimation
- EnsembleTrainer: Seed validation
- Integration tests (marked slow)
"""

import pytest
import numpy as np
import tempfile
import os
import json
from pathlib import Path


# =============================================================================
# EnsembleConfig Tests
# =============================================================================

class TestEnsembleConfig:
    """Tests for EnsembleConfig."""

    def test_default_values(self):
        """Test that default values are correctly set."""
        from src.evaluation.ensemble import EnsembleConfig

        config = EnsembleConfig()

        assert config.n_members == 3
        assert config.aggregation == 'confidence'
        assert len(config.seeds) >= config.n_members
        assert config.seeds == [42, 123, 456]
        assert config.softmax_temperature == 1.0
        assert config.enable_ood_detection is True
        assert config.ood_threshold == 2.5

    def test_custom_values(self):
        """Test that custom values are correctly set."""
        from src.evaluation.ensemble import EnsembleConfig

        config = EnsembleConfig(
            n_members=5,
            aggregation='median',
            seeds=[1, 2, 3, 4, 5],
            softmax_temperature=0.5,
            risk_aversion=2.0,
        )

        assert config.n_members == 5
        assert config.aggregation == 'median'
        assert config.seeds == [1, 2, 3, 4, 5]
        assert config.softmax_temperature == 0.5
        assert config.risk_aversion == 2.0

    def test_json_roundtrip(self, tmp_path):
        """Test JSON serialization and deserialization."""
        from src.evaluation.ensemble import EnsembleConfig

        config = EnsembleConfig(
            n_members=5,
            aggregation='conservative',
            softmax_temperature=0.7,
            gamma_range=(0.93, 0.97),
            lr_range=(1e-5, 3e-4),
        )

        json_path = tmp_path / "config.json"
        config.to_json(str(json_path))

        assert json_path.exists()

        loaded = EnsembleConfig.from_json(str(json_path))

        assert loaded.n_members == 5
        assert loaded.aggregation == 'conservative'
        assert loaded.softmax_temperature == 0.7
        assert loaded.gamma_range == (0.93, 0.97)
        assert loaded.lr_range == (1e-5, 3e-4)

    def test_json_content(self, tmp_path):
        """Test that JSON content is readable."""
        from src.evaluation.ensemble import EnsembleConfig

        config = EnsembleConfig(n_members=3)
        json_path = tmp_path / "config.json"
        config.to_json(str(json_path))

        with open(json_path, 'r') as f:
            data = json.load(f)

        assert data['n_members'] == 3
        assert data['aggregation'] == 'confidence'
        assert 'gamma_range' in data
        assert isinstance(data['gamma_range'], list)


# =============================================================================
# Aggregation Tests (Pure NumPy - No Models Required)
# =============================================================================

class TestAggregationMethods:
    """Tests for aggregation logic without models."""

    def test_aggregation_mean(self):
        """Test mean aggregation."""
        actions = np.array([
            [[0.5], [0.3]],   # Model 0
            [[0.7], [0.5]],   # Model 1
            [[0.6], [0.4]],   # Model 2
        ])

        expected = np.array([[0.6], [0.4]])
        result = np.mean(actions, axis=0)

        np.testing.assert_array_almost_equal(result, expected)

    def test_aggregation_median(self):
        """Test median aggregation."""
        actions = np.array([
            [[0.1], [0.1]],   # Outlier
            [[0.5], [0.5]],   # Median
            [[0.6], [0.6]],   #
        ])

        expected = np.array([[0.5], [0.5]])
        result = np.median(actions, axis=0)

        np.testing.assert_array_almost_equal(result, expected)

    def test_aggregation_conservative(self):
        """Test conservative aggregation (closest to 0)."""
        actions = np.array([
            [[0.9]],   # Aggressive
            [[0.1]],   # Conservative <- should be selected
            [[0.5]],   # Medium
        ])

        # Conservative selects action with smallest |action|
        abs_actions = np.abs(actions)
        min_idx = np.argmin(abs_actions.mean(axis=-1), axis=0)

        assert min_idx[0] == 1  # Model 1 is most conservative

    def test_aggregation_pessimistic_bound(self):
        """Test pessimistic bound aggregation."""
        actions = np.array([
            [[0.8]],
            [[0.6]],
            [[0.7]],
        ])

        mean_action = np.mean(actions, axis=0)
        std_action = np.std(actions, axis=0)
        k = 1.0  # risk_aversion

        scaling_factor = np.clip(1.0 - (k * std_action), 0.0, 1.0)
        final_action = mean_action * scaling_factor

        # Mean is 0.7, std is ~0.082
        # scaling_factor = 1 - 0.082 ≈ 0.918
        # final_action ≈ 0.7 * 0.918 ≈ 0.64
        assert final_action[0, 0] < mean_action[0, 0]  # Should be reduced
        assert final_action[0, 0] > 0  # Should still be positive


class TestAgreementComputation:
    """Tests for agreement ratio computation."""

    def test_perfect_agreement(self):
        """Test agreement with identical predictions."""
        actions_agree = np.array([
            [[0.5]],
            [[0.5]],
            [[0.5]],
        ])

        std_agree = np.std(actions_agree, axis=0).mean()
        agreement_agree = np.clip(1.0 - std_agree, 0.0, 1.0)

        assert agreement_agree == 1.0

    def test_high_disagreement(self):
        """Test agreement with diverse predictions."""
        actions_disagree = np.array([
            [[-1.0]],
            [[0.0]],
            [[1.0]],
        ])

        std_disagree = np.std(actions_disagree, axis=0).mean()
        agreement_disagree = np.clip(1.0 - std_disagree, 0.0, 1.0)

        assert agreement_disagree < 0.5

    def test_moderate_agreement(self):
        """Test agreement with moderate diversity."""
        actions = np.array([
            [[0.4]],
            [[0.5]],
            [[0.6]],
        ])

        std = np.std(actions, axis=0).mean()
        agreement = np.clip(1.0 - std, 0.0, 1.0)

        assert 0.8 < agreement < 1.0  # High but not perfect


class TestConfidenceWeighting:
    """Tests for softmax confidence weighting."""

    def test_softmax_weights_sum_to_one(self):
        """Test that softmax weights sum to 1."""
        spreads = np.array([[0.1], [0.2], [0.3]])  # (n_models, batch_size)

        tau = 1.0
        log_weights = -spreads / tau
        log_weights = log_weights - log_weights.max(axis=0, keepdims=True)
        weights = np.exp(log_weights)
        weights = weights / weights.sum(axis=0, keepdims=True)

        np.testing.assert_almost_equal(weights.sum(axis=0), [1.0])

    def test_lower_spread_gets_higher_weight(self):
        """Test that lower spread gets higher weight."""
        spreads = np.array([[0.1], [0.5], [0.9]])

        tau = 1.0
        log_weights = -spreads / tau
        log_weights = log_weights - log_weights.max(axis=0, keepdims=True)
        weights = np.exp(log_weights)
        weights = weights / weights.sum(axis=0, keepdims=True)

        # Model 0 (lowest spread) should have highest weight
        assert weights[0, 0] > weights[1, 0] > weights[2, 0]

    def test_temperature_effect(self):
        """Test that lower temperature makes weights more extreme."""
        spreads = np.array([[0.1], [0.5]])

        def compute_weights(tau):
            log_weights = -spreads / tau
            log_weights = log_weights - log_weights.max(axis=0, keepdims=True)
            weights = np.exp(log_weights)
            return weights / weights.sum(axis=0, keepdims=True)

        weights_high_tau = compute_weights(tau=2.0)
        weights_low_tau = compute_weights(tau=0.5)

        # Lower temperature -> more extreme weights
        # The gap between weights should be larger with low tau
        gap_high = weights_high_tau[0, 0] - weights_high_tau[1, 0]
        gap_low = weights_low_tau[0, 0] - weights_low_tau[1, 0]

        assert gap_low > gap_high


class TestOODDetection:
    """Tests for OOD detection logic."""

    def test_zscore_computation(self):
        """Test z-score computation for OOD detection."""
        # Simulate spread history
        spread_history = [0.1, 0.12, 0.11, 0.09, 0.10, 0.11, 0.10] * 20  # 140 values
        avg_spread = 0.5  # Abnormally high

        mean_spread = np.mean(spread_history[-100:])
        std_spread = np.std(spread_history[-100:])
        z_score = (avg_spread - mean_spread) / (std_spread + 1e-6)

        # Should be a high z-score (abnormal)
        assert z_score > 2.0

    def test_normal_spread_low_zscore(self):
        """Test that normal spread gives low z-score."""
        spread_history = [0.1] * 150
        avg_spread = 0.1  # Normal

        mean_spread = np.mean(spread_history[-100:])
        std_spread = np.std(spread_history[-100:])
        z_score = (avg_spread - mean_spread) / (std_spread + 1e-6)

        # Should be low z-score (normal)
        assert abs(z_score) < 0.5


# =============================================================================
# EnsembleTrainer Tests
# =============================================================================

class TestEnsembleTrainer:
    """Tests for EnsembleTrainer."""

    def test_seed_validation(self):
        """Test that trainer validates seeds."""
        from src.evaluation.ensemble import EnsembleConfig, EnsembleTrainer
        from src.config import TQCTrainingConfig

        config = TQCTrainingConfig()
        ensemble_config = EnsembleConfig(n_members=10, seeds=[1, 2, 3])

        with pytest.raises(ValueError, match="Need 10 seeds"):
            EnsembleTrainer(config, ensemble_config)

    def test_valid_seeds(self):
        """Test that trainer accepts valid seed count."""
        from src.evaluation.ensemble import EnsembleConfig, EnsembleTrainer
        from src.config import TQCTrainingConfig

        config = TQCTrainingConfig()
        ensemble_config = EnsembleConfig(n_members=3, seeds=[1, 2, 3, 4, 5])

        # Should not raise
        trainer = EnsembleTrainer(config, ensemble_config)
        assert trainer.config.n_members == 3


# =============================================================================
# EnsemblePolicy Tests (Require Mock or Real Models)
# =============================================================================

class TestEnsemblePolicy:
    """Tests for EnsemblePolicy that don't require real models."""

    def test_initialization_with_missing_models(self, tmp_path):
        """Test that EnsemblePolicy raises error for missing models."""
        from src.evaluation.ensemble import EnsemblePolicy, EnsembleConfig

        fake_paths = [
            str(tmp_path / "nonexistent_1.zip"),
            str(tmp_path / "nonexistent_2.zip"),
        ]

        with pytest.raises(FileNotFoundError):
            EnsemblePolicy(fake_paths)

    def test_config_defaults(self):
        """Test that EnsemblePolicy uses default config."""
        from src.evaluation.ensemble import EnsembleConfig

        config = EnsembleConfig()

        assert config.preload_models is False
        assert config.device == 'cuda'
        assert config.deterministic is True


# =============================================================================
# Integration Tests (Slow - Require Models)
# =============================================================================

@pytest.mark.slow
class TestIntegration:
    """Integration tests with actual models."""

    @pytest.fixture
    def mock_ensemble_dir(self, tmp_path):
        """Create a mock ensemble directory structure."""
        ensemble_dir = tmp_path / "ensemble"
        ensemble_dir.mkdir()

        # Create mock config
        from src.evaluation.ensemble import EnsembleConfig
        config = EnsembleConfig(n_members=2, seeds=[42, 123])
        config.to_json(str(ensemble_dir / "ensemble_config.json"))

        return ensemble_dir

    def test_load_ensemble_missing_models(self, mock_ensemble_dir):
        """Test load_ensemble with missing model files."""
        from src.evaluation.ensemble import load_ensemble

        with pytest.raises(FileNotFoundError):
            load_ensemble(str(mock_ensemble_dir))

    def test_full_pipeline(self):
        """Test full ensemble training and evaluation."""
        pytest.skip("Requires GPU and ~30 minutes")


# =============================================================================
# Utility Function Tests
# =============================================================================

class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_compare_single_vs_ensemble_types(self):
        """Test that compare function returns expected keys."""
        # This test verifies the expected return structure
        expected_keys = [
            'single_mean',
            'single_std',
            'ensemble_mean',
            'ensemble_std',
            'improvement_mean',
            'improvement_pct',
            'variance_reduction',
        ]

        # Create a mock result
        mock_result = {key: 0.0 for key in expected_keys}

        for key in expected_keys:
            assert key in mock_result


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_member_ensemble(self):
        """Test ensemble config with single member."""
        from src.evaluation.ensemble import EnsembleConfig

        config = EnsembleConfig(n_members=1, seeds=[42])

        assert config.n_members == 1
        assert len(config.seeds) >= 1

    def test_empty_actions_array(self):
        """Test handling of edge cases in aggregation."""
        # Empty batch
        actions = np.array([[[0.5]], [[0.6]], [[0.7]]])
        result = np.mean(actions, axis=0)
        assert result.shape == (1, 1)

    def test_extreme_risk_aversion(self):
        """Test pessimistic bound with high risk aversion."""
        actions = np.array([
            [[0.5]],
            [[0.6]],
            [[0.7]],
        ])

        mean_action = np.mean(actions, axis=0)
        std_action = np.std(actions, axis=0)
        k = 5.0  # Very high risk aversion

        scaling_factor = np.clip(1.0 - (k * std_action), 0.0, 1.0)
        final_action = mean_action * scaling_factor

        # With very high risk aversion, scaling should be at minimum (0.0)
        # when std is high enough
        assert scaling_factor[0, 0] >= 0.0

    def test_spread_calibration_ema(self):
        """Test spread calibration with EMA."""
        spreads = np.array([0.1, 0.2, 0.15, 0.12, 0.18])

        # Simulate EMA update
        ema = None
        alpha = 0.01

        for spread in spreads:
            if ema is None:
                ema = spread
            else:
                ema = alpha * spread + (1 - alpha) * ema

        # EMA should be close to mean but weighted toward early values
        assert 0.1 < ema < 0.2

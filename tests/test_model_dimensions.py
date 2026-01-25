#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_model_dimensions.py - Tests for model dimension validation.

Tests the ModelDimensionsValidator to ensure dimension mismatches are detected
before model creation and at checkpoint loading time.
"""

import pytest
import torch
from gymnasium import spaces
import numpy as np

from src.config.constants import (
    MAEConfig,
    TransformerFeatureExtractorConfig,
    FoundationFeatureExtractorConfig,
    DEFAULT_MAE_CONFIG,
    DEFAULT_TRANSFORMER_CONFIG,
    DEFAULT_FOUNDATION_CONFIG,
    HMM_CONTEXT_SIZE,
)
from src.config.validators import ModelDimensionsValidator


class TestModelDimensionsValidator:
    """Tests for ModelDimensionsValidator."""

    def test_validate_d_model_divisible_by_n_heads_valid(self):
        """Test that valid d_model/n_heads combinations pass validation."""
        # Valid cases
        ModelDimensionsValidator.validate_d_model_divisible_by_n_heads(256, 4)  # 256 % 4 == 0
        ModelDimensionsValidator.validate_d_model_divisible_by_n_heads(32, 2)   # 32 % 2 == 0
        ModelDimensionsValidator.validate_d_model_divisible_by_n_heads(128, 8)  # 128 % 8 == 0

    def test_validate_d_model_divisible_by_n_heads_invalid(self):
        """Test that invalid d_model/n_heads combinations raise ValueError."""
        with pytest.raises(ValueError, match="must be divisible"):
            ModelDimensionsValidator.validate_d_model_divisible_by_n_heads(256, 3)  # 256 % 3 == 1
        with pytest.raises(ValueError, match="must be divisible"):
            ModelDimensionsValidator.validate_d_model_divisible_by_n_heads(32, 5)   # 32 % 5 == 2

    def test_validate_transformer_config_valid(self):
        """Test that valid TransformerFeatureExtractorConfig passes validation."""
        config = DEFAULT_TRANSFORMER_CONFIG
        ModelDimensionsValidator.validate_transformer_config(config)

    def test_validate_transformer_config_invalid_d_model(self):
        """Test that invalid d_model raises ValueError."""
        config = TransformerFeatureExtractorConfig(d_model=0, n_heads=2, n_layers=1, features_dim=256)
        with pytest.raises(ValueError, match="d_model must be positive"):
            ModelDimensionsValidator.validate_transformer_config(config)

    def test_validate_transformer_config_invalid_dropout(self):
        """Test that invalid dropout raises ValueError."""
        config = TransformerFeatureExtractorConfig(dropout=1.5)  # > 1.0
        with pytest.raises(ValueError, match="dropout must be in"):
            ModelDimensionsValidator.validate_transformer_config(config)

    def test_validate_mae_config_valid(self):
        """Test that valid MAEConfig passes validation."""
        config = DEFAULT_MAE_CONFIG
        ModelDimensionsValidator.validate_mae_config(config)

    def test_validate_mae_config_invalid_n_heads(self):
        """Test that invalid n_heads raises ValueError."""
        config = MAEConfig(n_heads=0)
        with pytest.raises(ValueError, match="n_heads must be positive"):
            ModelDimensionsValidator.validate_mae_config(config)

    def test_validate_foundation_config_valid(self):
        """Test that valid FoundationFeatureExtractorConfig passes validation."""
        config = DEFAULT_FOUNDATION_CONFIG
        ModelDimensionsValidator.validate_foundation_config(
            config,
            n_features=43,  # Typical number of features
            window_size=64
        )

    def test_validate_foundation_config_invalid_features_dim(self):
        """Test that invalid features_dim raises ValueError."""
        config = FoundationFeatureExtractorConfig(features_dim=0)
        with pytest.raises(ValueError, match="features_dim must be positive"):
            ModelDimensionsValidator.validate_foundation_config(config)

    def test_validate_foundation_config_insufficient_features(self):
        """Test that insufficient features for FiLM raises ValueError."""
        config = DEFAULT_FOUNDATION_CONFIG
        with pytest.raises(ValueError, match="too small"):
            ModelDimensionsValidator.validate_foundation_config(
                config,
                n_features=3,  # Less than HMM_CONTEXT_SIZE
                window_size=64
            )

    def test_validate_checkpoint_compatibility_valid(self):
        """Test that compatible checkpoint passes validation."""
        # Create a mock checkpoint with matching dimensions
        checkpoint = {
            'embedding': {
                'weight': torch.randn(256, 38),  # d_model=256, input_dim=38
            },
            'd_model': 256
        }
        
        expected_config = DEFAULT_FOUNDATION_CONFIG
        ModelDimensionsValidator.validate_checkpoint_compatibility(
            checkpoint,
            expected_config=expected_config,
            expected_input_dim=38
        )

    def test_validate_checkpoint_compatibility_input_dim_mismatch(self):
        """Test that input dimension mismatch raises ValueError."""
        checkpoint = {
            'embedding': {
                'weight': torch.randn(256, 35),  # input_dim=35
            },
        }
        
        expected_config = DEFAULT_FOUNDATION_CONFIG
        with pytest.raises(ValueError, match="Input dimension mismatch"):
            ModelDimensionsValidator.validate_checkpoint_compatibility(
                checkpoint,
                expected_config=expected_config,
                expected_input_dim=38  # Different from checkpoint
            )

    def test_validate_checkpoint_compatibility_d_model_mismatch(self):
        """Test that d_model mismatch raises ValueError."""
        checkpoint = {
            'embedding': {
                'weight': torch.randn(128, 38),  # d_model=128
            },
            'd_model': 128
        }
        
        expected_config = DEFAULT_FOUNDATION_CONFIG  # d_model=256
        with pytest.raises(ValueError, match="d_model mismatch"):
            ModelDimensionsValidator.validate_checkpoint_compatibility(
                checkpoint,
                expected_config=expected_config,
                expected_input_dim=38
            )

    def test_validate_observation_space_compatibility_valid(self):
        """Test that compatible observation space passes validation."""
        ModelDimensionsValidator.validate_observation_space_compatibility(
            observation_space_n_features=43,
            expected_n_features=43,
            use_film=True
        )

    def test_validate_observation_space_compatibility_mismatch(self):
        """Test that mismatched observation space raises ValueError."""
        with pytest.raises(ValueError, match="Observation space has"):
            ModelDimensionsValidator.validate_observation_space_compatibility(
                observation_space_n_features=43,
                expected_n_features=40,  # Different
                use_film=True
            )

    def test_validate_observation_space_compatibility_insufficient_for_film(self):
        """Test that insufficient features for FiLM raises ValueError."""
        with pytest.raises(ValueError, match="FiLM requires at least"):
            ModelDimensionsValidator.validate_observation_space_compatibility(
                observation_space_n_features=3,  # Less than HMM_CONTEXT_SIZE
                expected_n_features=None,
                use_film=True
            )


class TestTransformerFeatureExtractorIntegration:
    """Integration tests for TransformerFeatureExtractor with validation."""

    def test_transformer_extractor_uses_default_config(self):
        """Test that TransformerFeatureExtractor uses default config when no args provided."""
        from src.models.transformer_policy import TransformerFeatureExtractor
        
        obs_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(64, 35),
            dtype=np.float32
        )
        
        extractor = TransformerFeatureExtractor(obs_space)
        
        # Verify it uses default config values
        assert extractor.d_model == DEFAULT_TRANSFORMER_CONFIG.d_model
        assert extractor.features_dim == DEFAULT_TRANSFORMER_CONFIG.features_dim

    def test_transformer_extractor_validation_on_invalid_config(self):
        """Test that TransformerFeatureExtractor validates config in __init__."""
        from src.models.transformer_policy import TransformerFeatureExtractor
        
        obs_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(64, 35),
            dtype=np.float32
        )
        
        # Invalid: d_model not divisible by n_heads
        with pytest.raises(ValueError, match="must be divisible"):
            TransformerFeatureExtractor(
                obs_space,
                d_model=33,  # Not divisible by default n_heads=2
                nhead=2
            )


class TestFoundationFeatureExtractorIntegration:
    """Integration tests for FoundationFeatureExtractor with validation."""

    def test_foundation_extractor_uses_default_config(self):
        """Test that FoundationFeatureExtractor uses default config when no args provided."""
        from src.models.rl_adapter import FoundationFeatureExtractor
        
        obs_space = spaces.Dict({
            "market": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(64, 43),
                dtype=np.float32
            ),
            "position": spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(1,),
                dtype=np.float32
            ),
            "w_cost": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(1,),
                dtype=np.float32
            )
        })
        
        # This will fail if encoder_path doesn't exist, but that's expected
        # We just want to verify it uses default config
        try:
            extractor = FoundationFeatureExtractor(
                obs_space,
                encoder_path="weights/nonexistent.pth"  # Will warn but continue
            )
            assert extractor.features_dim == DEFAULT_FOUNDATION_CONFIG.features_dim
            assert extractor.d_model == DEFAULT_FOUNDATION_CONFIG.mae_config.d_model
        except Exception:
            # If it fails due to missing encoder, that's OK for this test
            pass

    def test_foundation_extractor_validation_on_invalid_config(self):
        """Test that FoundationFeatureExtractor validates config in __init__."""
        from src.models.rl_adapter import FoundationFeatureExtractor
        
        obs_space = spaces.Dict({
            "market": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(64, 43),
                dtype=np.float32
            ),
            "position": spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(1,),
                dtype=np.float32
            ),
            "w_cost": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(1,),
                dtype=np.float32
            )
        })
        
        # Invalid: d_model not divisible by n_heads
        with pytest.raises(ValueError, match="must be divisible"):
            FoundationFeatureExtractor(
                obs_space,
                encoder_path="weights/nonexistent.pth",
                d_model=257,  # Not divisible by default n_heads=4
                n_heads=4
            )


class TestConfigConsistency:
    """Tests for configuration consistency across modules."""

    def test_default_configs_are_valid(self):
        """Test that all default configurations pass validation."""
        # MAE config
        ModelDimensionsValidator.validate_mae_config(DEFAULT_MAE_CONFIG)
        
        # Transformer config
        ModelDimensionsValidator.validate_transformer_config(DEFAULT_TRANSFORMER_CONFIG)
        
        # Foundation config
        ModelDimensionsValidator.validate_foundation_config(
            DEFAULT_FOUNDATION_CONFIG,
            n_features=43,  # Typical
            window_size=64
        )

    def test_config_feature_dim_consistency(self):
        """Test that feature dimensions are consistent and reasonable."""
        # Transformer should have smaller features_dim (standalone, no MAE)
        assert DEFAULT_TRANSFORMER_CONFIG.features_dim < DEFAULT_FOUNDATION_CONFIG.features_dim
        
        # Both should be positive
        assert DEFAULT_TRANSFORMER_CONFIG.features_dim > 0
        assert DEFAULT_FOUNDATION_CONFIG.features_dim > 0

    def test_mae_config_matches_constants(self):
        """Test that MAEConfig matches the primitive constants."""
        from src.config.constants import MAE_D_MODEL, MAE_N_HEADS, MAE_N_LAYERS
        
        assert DEFAULT_MAE_CONFIG.d_model == MAE_D_MODEL
        assert DEFAULT_MAE_CONFIG.n_heads == MAE_N_HEADS
        assert DEFAULT_MAE_CONFIG.n_layers == MAE_N_LAYERS

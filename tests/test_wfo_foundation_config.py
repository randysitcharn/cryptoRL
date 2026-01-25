#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_wfo_foundation_config.py - Tests for WFO foundation_config integration.

Tests the integration of foundation_config in the WFO pipeline to ensure:
- WFOConfig properly includes foundation_config
- CLI overrides work correctly
- Validation encoder compatibility works
- Edge cases are handled properly
"""

import os
import sys
import pytest
from dataclasses import replace

# Add root to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

# Import torch only when needed (for encoder compatibility tests)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from scripts.run_full_wfo import WFOConfig
from src.config.constants import (
    DEFAULT_FOUNDATION_CONFIG,
    FoundationFeatureExtractorConfig,
    MAEConfig,
)
from src.config.validators import ModelDimensionsValidator
from src.training.train_agent import create_policy_kwargs
from src.config import WFOTrainingConfig


class TestWFOConfigFoundationConfig:
    """Tests for WFOConfig foundation_config integration."""

    def test_wfo_config_has_foundation_config(self):
        """Test that WFOConfig has foundation_config attribute."""
        config = WFOConfig()
        assert hasattr(config, 'foundation_config')
        assert isinstance(config.foundation_config, FoundationFeatureExtractorConfig)

    def test_wfo_config_uses_default_foundation_config(self):
        """Test that WFOConfig uses DEFAULT_FOUNDATION_CONFIG by default."""
        config = WFOConfig()
        assert config.foundation_config == DEFAULT_FOUNDATION_CONFIG
        assert config.foundation_config.features_dim == DEFAULT_FOUNDATION_CONFIG.features_dim
        assert config.foundation_config.mae_config.d_model == DEFAULT_FOUNDATION_CONFIG.mae_config.d_model

    def test_foundation_config_override_features_dim(self):
        """Test that foundation_config can be overridden with custom features_dim."""
        config = WFOConfig()
        custom_config = replace(
            config.foundation_config,
            features_dim=1024
        )
        config.foundation_config = custom_config
        assert config.foundation_config.features_dim == 1024
        # Other values should remain unchanged
        assert config.foundation_config.mae_config.d_model == DEFAULT_FOUNDATION_CONFIG.mae_config.d_model

    def test_foundation_config_override_mae_config(self):
        """Test that foundation_config can be overridden with custom MAE config."""
        config = WFOConfig()
        custom_mae_config = replace(
            config.foundation_config.mae_config,
            d_model=512,
            n_heads=8,
            n_layers=4
        )
        custom_config = replace(
            config.foundation_config,
            mae_config=custom_mae_config
        )
        config.foundation_config = custom_config
        assert config.foundation_config.mae_config.d_model == 512
        assert config.foundation_config.mae_config.n_heads == 8
        assert config.foundation_config.mae_config.n_layers == 4

    def test_foundation_config_validation(self):
        """Test that foundation_config passes validation."""
        config = WFOConfig()
        ModelDimensionsValidator.validate_foundation_config(
            config.foundation_config,
            n_features=43,  # Typical number of features
            window_size=64
        )


class TestCreatePolicyKwargs:
    """Tests for create_policy_kwargs with foundation_config."""

    def test_create_policy_kwargs_with_default_foundation_config(self):
        """Test that create_policy_kwargs uses DEFAULT_FOUNDATION_CONFIG when None."""
        config = WFOTrainingConfig()
        policy_kwargs = create_policy_kwargs(config, foundation_config=None)
        
        assert 'features_extractor_kwargs' in policy_kwargs
        assert policy_kwargs['features_extractor_kwargs']['features_dim'] == DEFAULT_FOUNDATION_CONFIG.features_dim

    def test_create_policy_kwargs_with_custom_foundation_config(self):
        """Test that create_policy_kwargs uses provided foundation_config."""
        config = WFOTrainingConfig()
        custom_config = replace(
            DEFAULT_FOUNDATION_CONFIG,
            features_dim=1024
        )
        policy_kwargs = create_policy_kwargs(config, foundation_config=custom_config)
        
        assert policy_kwargs['features_extractor_kwargs']['features_dim'] == 1024

    def test_create_policy_kwargs_mae_config_priority(self):
        """Test that config.d_model takes priority over foundation_config.mae_config.d_model."""
        config = WFOTrainingConfig()
        config.d_model = 512  # Override
        
        custom_config = replace(
            DEFAULT_FOUNDATION_CONFIG,
            mae_config=replace(DEFAULT_FOUNDATION_CONFIG.mae_config, d_model=256)
        )
        policy_kwargs = create_policy_kwargs(config, foundation_config=custom_config)
        
        # config.d_model should take priority
        assert policy_kwargs['features_extractor_kwargs']['d_model'] == 512


class TestEncoderCompatibility:
    """Tests for encoder compatibility validation."""

    def test_validate_encoder_compatibility_valid(self, tmp_path):
        """Test that validate_encoder_compatibility passes with matching config."""
        if not TORCH_AVAILABLE:
            pytest.skip("torch not available")
        
        # Create a dummy encoder checkpoint with correct d_model
        encoder_path = tmp_path / "encoder.pth"
        checkpoint = {
            'encoder.proj.weight': torch.randn(DEFAULT_FOUNDATION_CONFIG.mae_config.d_model, 128),
            'd_model': DEFAULT_FOUNDATION_CONFIG.mae_config.d_model,
            'n_heads': DEFAULT_FOUNDATION_CONFIG.mae_config.n_heads,
            'n_layers': DEFAULT_FOUNDATION_CONFIG.mae_config.n_layers,
        }
        torch.save(checkpoint, encoder_path)
        
        # Should not raise
        ModelDimensionsValidator.validate_encoder_compatibility(
            str(encoder_path),
            DEFAULT_FOUNDATION_CONFIG
        )

    def test_validate_encoder_compatibility_d_model_mismatch(self, tmp_path):
        """Test that validate_encoder_compatibility detects d_model mismatch."""
        if not TORCH_AVAILABLE:
            pytest.skip("torch not available")
        
        # Create a dummy encoder checkpoint with wrong d_model
        encoder_path = tmp_path / "encoder.pth"
        checkpoint = {
            'encoder.proj.weight': torch.randn(512, 128),  # d_model=512 instead of 256
            'd_model': 512,
        }
        torch.save(checkpoint, encoder_path)
        
        # Should raise ValueError
        with pytest.raises(ValueError, match="d_model mismatch"):
            ModelDimensionsValidator.validate_encoder_compatibility(
                str(encoder_path),
                DEFAULT_FOUNDATION_CONFIG
            )

    def test_validate_encoder_compatibility_n_heads_mismatch(self, tmp_path):
        """Test that validate_encoder_compatibility detects n_heads mismatch."""
        if not TORCH_AVAILABLE:
            pytest.skip("torch not available")
        
        # Create a dummy encoder checkpoint with wrong n_heads
        encoder_path = tmp_path / "encoder.pth"
        checkpoint = {
            'encoder.proj.weight': torch.randn(DEFAULT_FOUNDATION_CONFIG.mae_config.d_model, 128),
            'd_model': DEFAULT_FOUNDATION_CONFIG.mae_config.d_model,
            'n_heads': 8,  # Wrong: should be 4
            'n_layers': DEFAULT_FOUNDATION_CONFIG.mae_config.n_layers,
        }
        torch.save(checkpoint, encoder_path)
        
        # Should raise ValueError
        with pytest.raises(ValueError, match="n_heads mismatch"):
            ModelDimensionsValidator.validate_encoder_compatibility(
                str(encoder_path),
                DEFAULT_FOUNDATION_CONFIG
            )

    def test_validate_encoder_compatibility_file_not_found(self):
        """Test that validate_encoder_compatibility raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            ModelDimensionsValidator.validate_encoder_compatibility(
                "nonexistent_encoder.pth",
                DEFAULT_FOUNDATION_CONFIG
            )

    def test_validate_encoder_compatibility_proj_weight_format(self, tmp_path):
        """Test that validate_encoder_compatibility works with different checkpoint formats."""
        if not TORCH_AVAILABLE:
            pytest.skip("torch not available")
        
        # Format 1: encoder.proj.weight
        encoder_path = tmp_path / "encoder1.pth"
        checkpoint1 = {
            'encoder.proj.weight': torch.randn(DEFAULT_FOUNDATION_CONFIG.mae_config.d_model, 128),
        }
        torch.save(checkpoint1, encoder_path)
        ModelDimensionsValidator.validate_encoder_compatibility(
            str(encoder_path),
            DEFAULT_FOUNDATION_CONFIG
        )
        
        # Format 2: proj.weight (without encoder prefix)
        encoder_path2 = tmp_path / "encoder2.pth"
        checkpoint2 = {
            'proj.weight': torch.randn(DEFAULT_FOUNDATION_CONFIG.mae_config.d_model, 128),
        }
        torch.save(checkpoint2, encoder_path2)
        ModelDimensionsValidator.validate_encoder_compatibility(
            str(encoder_path2),
            DEFAULT_FOUNDATION_CONFIG
        )


class TestEdgeCases:
    """Tests for edge cases in foundation_config handling."""

    def test_foundation_config_none_fallback(self):
        """Test that foundation_config=None uses DEFAULT_FOUNDATION_CONFIG."""
        config = WFOTrainingConfig()
        policy_kwargs = create_policy_kwargs(config, foundation_config=None)
        
        # Should use default
        assert policy_kwargs['features_extractor_kwargs']['features_dim'] == DEFAULT_FOUNDATION_CONFIG.features_dim

    def test_partial_override_features_dim_only(self):
        """Test that partial override (only features_dim) works."""
        config = WFOConfig()
        # Override only features_dim, keep MAE config unchanged
        config.foundation_config = replace(
            config.foundation_config,
            features_dim=1024
        )
        
        assert config.foundation_config.features_dim == 1024
        assert config.foundation_config.mae_config.d_model == DEFAULT_FOUNDATION_CONFIG.mae_config.d_model
        assert config.foundation_config.mae_config.n_heads == DEFAULT_FOUNDATION_CONFIG.mae_config.n_heads

    def test_partial_override_mae_config_only(self):
        """Test that partial override (only MAE config) works."""
        config = WFOConfig()
        # Override only MAE config, keep features_dim unchanged
        custom_mae = replace(
            config.foundation_config.mae_config,
            d_model=512
        )
        config.foundation_config = replace(
            config.foundation_config,
            mae_config=custom_mae
        )
        
        assert config.foundation_config.mae_config.d_model == 512
        assert config.foundation_config.features_dim == DEFAULT_FOUNDATION_CONFIG.features_dim

    def test_full_override(self):
        """Test that full override (both features_dim and MAE config) works."""
        config = WFOConfig()
        custom_mae = replace(
            config.foundation_config.mae_config,
            d_model=512,
            n_heads=8,
            n_layers=4
        )
        config.foundation_config = replace(
            config.foundation_config,
            features_dim=1024,
            mae_config=custom_mae
        )
        
        assert config.foundation_config.features_dim == 1024
        assert config.foundation_config.mae_config.d_model == 512
        assert config.foundation_config.mae_config.n_heads == 8
        assert config.foundation_config.mae_config.n_layers == 4

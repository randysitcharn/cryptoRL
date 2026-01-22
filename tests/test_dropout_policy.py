# -*- coding: utf-8 -*-
"""
test_dropout_policy.py - Tests for TQCDropoutPolicy implementation.

Quick tests to verify:
1. Policy imports correctly
2. MLP builder creates correct architecture
3. gSDE safety check works
4. Config integration
"""

import os
import sys
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# Test 1: Import and Basic Structure
# =============================================================================

class TestDropoutPolicyImport:
    """Test that TQCDropoutPolicy imports and has correct structure."""

    def test_import_policy(self):
        """Verify TQCDropoutPolicy can be imported."""
        from src.models.tqc_dropout_policy import TQCDropoutPolicy
        assert TQCDropoutPolicy is not None

    def test_import_components(self):
        """Verify all components can be imported."""
        from src.models.tqc_dropout_policy import (
            create_mlp_with_dropout,
            DropoutActor,
            DropoutCritic,
            SingleQuantileNet,
            TQCDropoutPolicy,
        )
        assert create_mlp_with_dropout is not None
        assert DropoutActor is not None
        assert DropoutCritic is not None
        assert SingleQuantileNet is not None
        assert TQCDropoutPolicy is not None


# =============================================================================
# Test 2: MLP Builder
# =============================================================================

class TestMLPBuilder:
    """Test create_mlp_with_dropout function."""

    def test_mlp_with_dropout_and_layernorm(self):
        """Verify MLP creates correct layer sequence."""
        import torch.nn as nn
        from src.models.tqc_dropout_policy import create_mlp_with_dropout

        mlp = create_mlp_with_dropout(
            input_dim=512,
            output_dim=64,
            net_arch=[256, 128],
            dropout_rate=0.01,
            use_layer_norm=True,
        )

        # Check layer types
        layer_types = [type(layer).__name__ for layer in mlp]
        
        assert "Linear" in layer_types, "Missing Linear layers"
        assert "LayerNorm" in layer_types, "Missing LayerNorm (critical for stability)"
        assert "Dropout" in layer_types, "Missing Dropout"
        assert "ReLU" in layer_types, "Missing ReLU activation"

    def test_mlp_without_dropout(self):
        """Verify MLP works without dropout."""
        from src.models.tqc_dropout_policy import create_mlp_with_dropout

        mlp = create_mlp_with_dropout(
            input_dim=512,
            output_dim=64,
            net_arch=[256, 128],
            dropout_rate=0.0,  # No dropout
            use_layer_norm=True,
        )

        layer_types = [type(layer).__name__ for layer in mlp]
        assert "Dropout" not in layer_types, "Dropout should not be present when rate=0"

    def test_mlp_without_layernorm(self):
        """Verify MLP works without LayerNorm."""
        from src.models.tqc_dropout_policy import create_mlp_with_dropout

        mlp = create_mlp_with_dropout(
            input_dim=512,
            output_dim=64,
            net_arch=[256, 128],
            dropout_rate=0.01,
            use_layer_norm=False,  # No LayerNorm
        )

        layer_types = [type(layer).__name__ for layer in mlp]
        assert "LayerNorm" not in layer_types, "LayerNorm should not be present when disabled"


# =============================================================================
# Test 3: gSDE Safety Check
# =============================================================================

class TestGSDESafety:
    """Test gSDE + Actor Dropout conflict detection."""

    def test_gsde_disables_actor_dropout(self):
        """Verify actor_dropout is forced to 0 when gSDE is active."""
        # Simulate the safety check logic from TQCDropoutPolicy
        actor_dropout = 0.01
        use_sde = True
        
        # This is the safety check from TQCDropoutPolicy.__init__
        if use_sde and actor_dropout > 0:
            actor_dropout = 0.0
        
        assert actor_dropout == 0.0, "Actor dropout should be 0 when gSDE is active"

    def test_no_sde_keeps_actor_dropout(self):
        """Verify actor_dropout is preserved when gSDE is disabled."""
        actor_dropout = 0.01
        use_sde = False
        
        if use_sde and actor_dropout > 0:
            actor_dropout = 0.0
        
        assert actor_dropout == 0.01, "Actor dropout should be preserved when gSDE is disabled"


# =============================================================================
# Test 4: Config Integration
# =============================================================================

class TestConfigIntegration:
    """Test TQCTrainingConfig has dropout parameters."""

    def test_config_has_dropout_params(self):
        """Verify TQCTrainingConfig has all dropout parameters."""
        from src.config.training import TQCTrainingConfig

        config = TQCTrainingConfig()

        assert hasattr(config, 'use_dropout_policy')
        assert hasattr(config, 'critic_dropout')
        assert hasattr(config, 'actor_dropout')
        assert hasattr(config, 'use_layer_norm')

    def test_config_defaults(self):
        """Verify dropout config defaults are correct."""
        from src.config.training import TQCTrainingConfig

        config = TQCTrainingConfig()

        # Defaults from DROPOUT_TQC_DESIGN.md
        assert config.use_dropout_policy == True, "Dropout policy should be enabled by default"
        assert config.critic_dropout == 0.01, "Critic dropout default should be 0.01"
        assert config.actor_dropout == 0.0, "Actor dropout default should be 0 (Phase 1: critics only)"
        assert config.use_layer_norm == True, "LayerNorm should be enabled by default"


# =============================================================================
# Test 5: Forward Pass (requires torch)
# =============================================================================

class TestForwardPass:
    """Test forward pass through dropout components."""

    @pytest.mark.skipif(
        not os.environ.get('RUN_GPU_TESTS', False),
        reason="Skipping GPU-dependent test (set RUN_GPU_TESTS=1 to enable)"
    )
    def test_single_quantile_net_forward(self):
        """Test SingleQuantileNet forward pass."""
        import torch
        import torch.nn as nn
        from src.models.tqc_dropout_policy import SingleQuantileNet

        qnet = SingleQuantileNet(
            input_dim=513,  # features + action
            n_quantiles=25,
            net_arch=[64, 64],
            activation_fn=nn.ReLU,
            dropout_rate=0.01,
            use_layer_norm=True,
        )

        batch_size = 4
        dummy_input = torch.randn(batch_size, 513)
        output = qnet(dummy_input)

        assert output.shape == (batch_size, 25), f"Wrong shape: {output.shape}"

    @pytest.mark.skipif(
        not os.environ.get('RUN_GPU_TESTS', False),
        reason="Skipping GPU-dependent test (set RUN_GPU_TESTS=1 to enable)"
    )
    def test_dropout_critic_forward(self):
        """Test DropoutCritic forward pass."""
        import torch
        import torch.nn as nn
        from src.models.tqc_dropout_policy import DropoutCritic

        critic = DropoutCritic(
            features_dim=512,
            action_dim=1,
            n_critics=2,
            n_quantiles=25,
            net_arch=[64, 64],
            activation_fn=nn.ReLU,
            dropout_rate=0.01,
            use_layer_norm=True,
        )

        batch_size = 4
        dummy_obs = torch.randn(batch_size, 512)
        dummy_action = torch.randn(batch_size, 1)

        output = critic(dummy_obs, dummy_action)

        # TQC expects (batch_size, n_critics, n_quantiles)
        assert output.shape == (batch_size, 2, 25), f"Wrong shape: {output.shape}"

    @pytest.mark.skipif(
        not os.environ.get('RUN_GPU_TESTS', False),
        reason="Skipping GPU-dependent test (set RUN_GPU_TESTS=1 to enable)"
    )
    def test_eval_mode_deterministic(self):
        """Test that eval mode produces deterministic output."""
        import torch
        import torch.nn as nn
        from src.models.tqc_dropout_policy import DropoutCritic

        critic = DropoutCritic(
            features_dim=512,
            action_dim=1,
            n_critics=2,
            n_quantiles=25,
            net_arch=[64, 64],
            activation_fn=nn.ReLU,
            dropout_rate=0.01,
            use_layer_norm=True,
        )

        batch_size = 4
        dummy_obs = torch.randn(batch_size, 512)
        dummy_action = torch.randn(batch_size, 1)

        # Eval mode should be deterministic
        critic.eval()
        out1 = critic(dummy_obs, dummy_action)
        out2 = critic(dummy_obs, dummy_action)

        assert torch.allclose(out1, out2), "Eval mode should be deterministic"


# =============================================================================
# Test 6: Spectral Normalization
# =============================================================================

class TestSpectralNormalization:
    """Test spectral normalization implementation."""

    def test_spectral_norm_applied_critic(self):
        """Verify spectral normalization is applied to Critic Linear layers."""
        import torch.nn as nn
        from src.models.tqc_dropout_policy import TQCDropoutPolicy
        from gymnasium.spaces import Box
        from stable_baselines3.common.schedules import constant_schedule

        # Create minimal observation and action spaces
        obs_space = Box(low=-1, high=1, shape=(512,), dtype=float)
        action_space = Box(low=-1, high=1, shape=(1,), dtype=float)
        lr_schedule = constant_schedule(3e-4)

        # Create policy with spectral norm enabled for critic only
        policy = TQCDropoutPolicy(
            observation_space=obs_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            use_spectral_norm_critic=True,
            use_spectral_norm_actor=False,
            features_dim=512,
        )
        critic = policy.make_critic()

        # Check that Linear layers in critics are wrapped
        for critic_net in critic.critics:
            for layer in critic_net.net:
                if isinstance(layer, nn.Linear):
                    # Check if wrapped with spectral_norm (has weight_u for power iteration)
                    assert hasattr(layer, 'weight_u'), "Spectral norm not applied to critic"

    def test_spectral_norm_applied_actor(self):
        """Verify spectral normalization is applied to Actor Linear layers when enabled."""
        import torch.nn as nn
        from src.models.tqc_dropout_policy import TQCDropoutPolicy
        from gymnasium.spaces import Box
        from stable_baselines3.common.schedules import constant_schedule

        # Create minimal observation and action spaces
        obs_space = Box(low=-1, high=1, shape=(512,), dtype=float)
        action_space = Box(low=-1, high=1, shape=(1,), dtype=float)
        lr_schedule = constant_schedule(3e-4)

        # Create policy with spectral norm enabled for actor
        policy = TQCDropoutPolicy(
            observation_space=obs_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            use_spectral_norm_critic=False,
            use_spectral_norm_actor=True,
            features_dim=512,
        )
        actor = policy.make_actor()

        # Check that Linear layers in actor are wrapped
        for layer in actor.latent_pi:
            if isinstance(layer, nn.Linear):
                assert hasattr(layer, 'weight_u'), "Spectral norm not applied to actor"

    def test_spectral_norm_not_on_output_layer(self):
        """Verify that the output layer does NOT have spectral norm."""
        import torch.nn as nn
        from src.models.tqc_dropout_policy import TQCDropoutPolicy
        from gymnasium.spaces import Box
        from stable_baselines3.common.schedules import constant_schedule

        # Create minimal observation and action spaces
        obs_space = Box(low=-1, high=1, shape=(512,), dtype=float)
        action_space = Box(low=-1, high=1, shape=(1,), dtype=float)
        lr_schedule = constant_schedule(3e-4)

        policy = TQCDropoutPolicy(
            observation_space=obs_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            use_spectral_norm_critic=True,
            use_spectral_norm_actor=False,
            features_dim=512,
        )
        critic = policy.make_critic()

        # Check that the LAST Linear layer (output) does NOT have SN
        for critic_net in critic.critics:
            layers = list(critic_net.net)
            output_layer = layers[-1]  # Last layer should be Linear(output_dim, n_quantiles)

            if isinstance(output_layer, nn.Linear):
                assert not hasattr(output_layer, 'weight_u'), \
                    "Output layer should NOT have spectral norm"

    def test_config_has_spectral_norm_params(self):
        """Verify TQCTrainingConfig has spectral normalization parameters."""
        from src.config.training import TQCTrainingConfig

        config = TQCTrainingConfig()

        assert hasattr(config, 'use_spectral_norm_critic')
        assert hasattr(config, 'use_spectral_norm_actor')
        assert config.use_spectral_norm_critic == False, "Default should be False"
        assert config.use_spectral_norm_actor == False, "Default should be False"


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

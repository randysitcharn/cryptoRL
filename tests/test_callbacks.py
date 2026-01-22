# -*- coding: utf-8 -*-
"""
test_callbacks.py - Tests for RL Callbacks.

Verifies that the critical callbacks work correctly:
- ThreePhaseCurriculumCallback: phase transitions and curriculum_lambda
- OverfittingGuardCallbackV2: 5 signal detection and multi-signal decision logic
- ModelEMACallback: Polyak averaging and weight management

Reference: docs/audit/AUDIT_MODELES_RL_RESULTATS.md - Tests Callbacks section
"""

import sys
import os
from unittest.mock import Mock, MagicMock, patch
from collections import deque
import numpy as np
import torch
import pytest

# Add the project root to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.training.callbacks import (
    ThreePhaseCurriculumCallback,
    OverfittingGuardCallbackV2,
    ModelEMACallback,
    get_underlying_batch_env,
)


# ============================================================================
# Fixtures and Helpers
# ============================================================================

def create_mock_model(n_params: int = 10, param_size: int = 100):
    """
    Create a mock SB3 model with policy parameters.
    
    Args:
        n_params: Number of parameter tensors.
        param_size: Size of each parameter tensor.
        
    Returns:
        Mock model object.
    """
    model = Mock()
    model.policy = Mock()
    
    # Create trainable parameters
    params = [
        torch.nn.Parameter(torch.randn(param_size, requires_grad=True))
        for _ in range(n_params)
    ]
    model.policy.parameters = Mock(return_value=iter(params))
    
    # Store params as list for repeated iteration
    model._params_list = params
    model.policy.parameters = Mock(side_effect=lambda: iter(model._params_list))
    
    # Mock logger
    model.logger = Mock()
    model.logger.record = Mock()
    
    # Mock env
    model.env = Mock()
    
    return model


def create_mock_env_with_curriculum():
    """Create a mock environment that supports curriculum learning."""
    env = Mock()
    env.curriculum_lambda = 0.0
    env.set_progress = Mock()
    return env


# ============================================================================
# TestThreePhaseCurriculumCallback
# ============================================================================

class TestThreePhaseCurriculumCallback:
    """Tests for ThreePhaseCurriculumCallback."""
    
    def test_initialization(self):
        """Test callback initialization."""
        callback = ThreePhaseCurriculumCallback(total_timesteps=100_000)
        
        assert callback.total_timesteps == 100_000
        assert callback._phase == 1
        assert len(callback.PHASES) == 3
    
    def test_phase_boundaries(self):
        """Test phase transitions at 15% and 75% boundaries."""
        callback = ThreePhaseCurriculumCallback(total_timesteps=100_000)
        callback.model = create_mock_model()
        callback.logger = callback.model.logger
        
        # Mock environment
        mock_env = create_mock_env_with_curriculum()
        with patch('src.training.callbacks.get_underlying_batch_env', return_value=mock_env):
            # Phase 1: 0% - 15%
            callback.num_timesteps = 0
            callback._on_step()
            assert callback._phase == 1
            
            callback.num_timesteps = 14_999  # Just before 15%
            callback._on_step()
            assert callback._phase == 1
            
            # Phase 2: 15% - 75%
            callback.num_timesteps = 15_000  # Exactly at 15%
            callback._on_step()
            assert callback._phase == 2
            
            callback.num_timesteps = 50_000  # Middle of phase 2
            callback._on_step()
            assert callback._phase == 2
            
            callback.num_timesteps = 74_999  # Just before 75%
            callback._on_step()
            assert callback._phase == 2
            
            # Phase 3: 75% - 100%
            callback.num_timesteps = 75_000  # Exactly at 75%
            callback._on_step()
            assert callback._phase == 3
            
            callback.num_timesteps = 100_000  # End
            callback._on_step()
            assert callback._phase == 3
    
    def test_env_update_called(self):
        """Test that set_progress is called on environment."""
        callback = ThreePhaseCurriculumCallback(total_timesteps=100_000)
        callback.model = create_mock_model()
        callback.logger = callback.model.logger
        
        mock_env = create_mock_env_with_curriculum()
        with patch('src.training.callbacks.get_underlying_batch_env', return_value=mock_env):
            callback.num_timesteps = 50_000  # 50% progress
            callback._on_step()
            
            # Verify set_progress was called with correct value
            mock_env.set_progress.assert_called()
            call_args = mock_env.set_progress.call_args[0][0]
            assert abs(call_args - 0.5) < 0.01  # Should be 0.5
    
    def test_logging_records(self):
        """Test that metrics are logged to TensorBoard."""
        callback = ThreePhaseCurriculumCallback(total_timesteps=100_000)
        callback.model = create_mock_model()
        callback.logger = callback.model.logger
        
        mock_env = create_mock_env_with_curriculum()
        mock_env.curriculum_lambda = 0.2
        
        with patch('src.training.callbacks.get_underlying_batch_env', return_value=mock_env):
            callback.num_timesteps = 50_000
            callback._on_step()
            
            # Verify logger.record was called
            record_calls = callback.logger.record.call_args_list
            keys_logged = [call[0][0] for call in record_calls]
            
            assert "curriculum/phase" in keys_logged
            assert "curriculum/progress" in keys_logged
            assert "curriculum/lambda" in keys_logged


# ============================================================================
# TestOverfittingGuardCallbackV2
# ============================================================================

class TestOverfittingGuardCallbackV2:
    """Tests for OverfittingGuardCallbackV2."""
    
    def test_initialization(self):
        """Test callback initialization with default and custom values."""
        # Default values
        guard = OverfittingGuardCallbackV2()
        assert guard.nav_threshold == 5.0
        assert guard.patience == 3
        assert guard.check_freq == 10_000
        
        # Custom values
        guard = OverfittingGuardCallbackV2(
            nav_threshold=10.0,
            patience=5,
            check_freq=25_000
        )
        assert guard.nav_threshold == 10.0
        assert guard.patience == 5
        assert guard.check_freq == 25_000
    
    def test_violation_counts_initialized(self):
        """Test that violation counts are initialized to zero."""
        guard = OverfittingGuardCallbackV2()
        
        for signal, count in guard.violation_counts.items():
            assert count == 0, f"Signal {signal} should start at 0"
    
    def test_signal_nav_threshold(self):
        """Test Signal 1: NAV threshold detection."""
        guard = OverfittingGuardCallbackV2(
            nav_threshold=5.0,
            initial_nav=10_000.0
        )
        
        # Setup mock
        guard.training_env = Mock()
        guard.training_env.get_global_metrics = Mock(
            return_value={"portfolio_value": 60_000.0}  # 6x > 5x threshold
        )
        
        # Should detect violation
        result = guard._check_nav_threshold()
        assert result is not None
        assert "NAV" in result
        assert "6.0x" in result
    
    def test_signal_nav_threshold_no_violation(self):
        """Test NAV threshold with acceptable returns."""
        guard = OverfittingGuardCallbackV2(
            nav_threshold=5.0,
            initial_nav=10_000.0
        )
        
        guard.training_env = Mock()
        guard.training_env.get_global_metrics = Mock(
            return_value={"portfolio_value": 20_000.0}  # 2x < 5x threshold
        )
        
        result = guard._check_nav_threshold()
        assert result is None
    
    def test_signal_action_saturation(self):
        """Test Signal 4: Action saturation detection."""
        guard = OverfittingGuardCallbackV2(
            action_saturation_threshold=0.95,
            saturation_ratio_limit=0.8,
            reward_window=100
        )
        
        # Fill with saturated actions (all near 1.0)
        saturated_actions = np.full(100, 0.99)
        guard.actions_history = deque(saturated_actions, maxlen=100)
        
        result = guard._check_action_saturation()
        assert result is not None
        assert "saturation" in result.lower()
    
    def test_signal_action_saturation_no_violation(self):
        """Test action saturation with diverse actions."""
        guard = OverfittingGuardCallbackV2(
            action_saturation_threshold=0.95,
            saturation_ratio_limit=0.8,
            reward_window=100
        )
        
        # Fill with diverse actions (mostly below threshold)
        diverse_actions = np.random.uniform(0.0, 0.5, 100)
        guard.actions_history = deque(diverse_actions, maxlen=100)
        
        result = guard._check_action_saturation()
        assert result is None
    
    def test_signal_reward_variance(self):
        """Test Signal 5: Reward variance detection."""
        guard = OverfittingGuardCallbackV2(
            reward_variance_threshold=1e-4,
            reward_window=100
        )
        
        # Fill with constant rewards (variance ~0)
        constant_rewards = np.full(100, 0.01)
        guard.rewards_history = deque(constant_rewards, maxlen=100)
        
        result = guard._check_reward_variance()
        assert result is not None
        assert "variance" in result.lower() or "CV" in result
    
    def test_signal_reward_variance_no_violation(self):
        """Test reward variance with varying rewards."""
        guard = OverfittingGuardCallbackV2(
            reward_variance_threshold=1e-4,
            reward_window=100
        )
        
        # Fill with varying rewards
        varying_rewards = np.random.randn(100) * 0.1  # High variance
        guard.rewards_history = deque(varying_rewards, maxlen=100)
        
        result = guard._check_reward_variance()
        assert result is None
    
    def test_multi_signal_decision_two_signals(self):
        """Test decision logic: 2+ signals = stop."""
        guard = OverfittingGuardCallbackV2(patience=3)
        
        # Two signals active (should trigger stop)
        violations = ["Signal A", "Signal B"]
        should_stop = guard._decide_stop(violations)
        
        assert should_stop is True
    
    def test_multi_signal_decision_one_signal_no_patience(self):
        """Test decision logic: 1 signal without patience = continue."""
        guard = OverfittingGuardCallbackV2(patience=3)
        guard.violation_counts = {
            'nav': 1,  # Only 1 violation, patience is 3
            'weight': 0,
            'divergence': 0,
            'saturation': 0,
            'variance': 0
        }
        
        violations = ["NAV violation"]
        should_stop = guard._decide_stop(violations)
        
        assert should_stop is False
    
    def test_multi_signal_decision_patience_reached(self):
        """Test decision logic: patience reached = stop."""
        guard = OverfittingGuardCallbackV2(patience=3)
        guard.violation_counts = {
            'nav': 3,  # Patience reached
            'weight': 0,
            'divergence': 0,
            'saturation': 0,
            'variance': 0
        }
        
        violations = ["NAV violation"]
        should_stop = guard._decide_stop(violations)
        
        assert should_stop is True
    
    def test_violation_count_reset(self):
        """Test that violation count resets when signal clears."""
        guard = OverfittingGuardCallbackV2(
            check_freq=1,
            patience=3
        )
        guard.model = create_mock_model()
        guard.logger = guard.model.logger
        guard.training_env = Mock()
        guard.training_env.get_global_metrics = Mock(return_value={})
        guard.num_timesteps = 1
        guard.locals = {}
        
        # Set initial violation count
        guard.violation_counts['nav'] = 2
        
        # Run step - NAV check returns None (no violation)
        guard._check_nav_threshold = Mock(return_value=None)
        guard._check_weight_stagnation = Mock(return_value=None)
        guard._check_train_eval_divergence = Mock(return_value=None)
        guard._check_action_saturation = Mock(return_value=None)
        guard._check_reward_variance = Mock(return_value=None)
        
        guard._on_step()
        
        # Count should be reset to 0
        assert guard.violation_counts['nav'] == 0
    
    def test_eval_callback_none_handling(self):
        """Test graceful handling when eval_callback is None."""
        guard = OverfittingGuardCallbackV2(eval_callback=None)
        
        # Should not crash and return None (no violation)
        result = guard._check_train_eval_divergence()
        assert result is None
    
    def test_data_collection_with_deque(self):
        """Test that data collection respects deque maxlen."""
        guard = OverfittingGuardCallbackV2(reward_window=10)
        
        # Verify deque has correct maxlen
        assert guard.actions_history.maxlen == 10
        assert guard.rewards_history.maxlen == 10
        
        # Add more than maxlen items
        for i in range(20):
            guard.actions_history.append(i)
        
        # Should only keep last 10
        assert len(guard.actions_history) == 10
        assert list(guard.actions_history) == list(range(10, 20))


# ============================================================================
# TestModelEMACallback
# ============================================================================

class TestModelEMACallback:
    """Tests for ModelEMACallback."""
    
    def test_initialization(self):
        """Test callback initialization."""
        callback = ModelEMACallback(decay=0.995)
        
        assert callback.decay == 0.995
        assert callback.tau == 0.005  # 1 - decay
        assert callback.ema_params is None  # Not initialized until training starts
    
    def test_tau_calculation(self):
        """Test tau calculation for different decay values."""
        # Slow EMA
        callback = ModelEMACallback(decay=0.999)
        assert abs(callback.tau - 0.001) < 1e-6
        
        # Medium EMA
        callback = ModelEMACallback(decay=0.99)
        assert abs(callback.tau - 0.01) < 1e-6
        
        # Fast EMA
        callback = ModelEMACallback(decay=0.95)
        assert abs(callback.tau - 0.05) < 1e-6
    
    def test_ema_initialization_on_training_start(self):
        """Test that EMA params are initialized on training start."""
        callback = ModelEMACallback(decay=0.995, verbose=1)
        callback.model = create_mock_model(n_params=5, param_size=10)
        
        callback._on_training_start()
        
        assert callback.ema_params is not None
        assert len(callback.ema_params) == 5
        assert callback.param_shapes is not None
    
    def test_ema_params_cloned_and_detached(self):
        """Test that EMA params are properly cloned and detached."""
        callback = ModelEMACallback(decay=0.995)
        callback.model = create_mock_model(n_params=3, param_size=10)
        
        callback._on_training_start()
        
        # Get original params
        original_params = list(callback.model.policy.parameters())
        
        # Verify EMA params are detached (no grad)
        for ema_param in callback.ema_params:
            assert not ema_param.requires_grad
        
        # Verify modification to original doesn't affect EMA
        with torch.no_grad():
            original_params[0].data.fill_(999.0)
        
        assert callback.ema_params[0].data[0].item() != 999.0
    
    def test_polyak_update_convergence(self):
        """Test that EMA converges towards current weights over time."""
        callback = ModelEMACallback(decay=0.9, verbose=0)  # Fast decay for test
        callback.model = create_mock_model(n_params=1, param_size=10)
        callback.logger = callback.model.logger
        callback.num_timesteps = 0
        
        callback._on_training_start()
        
        # Set target weight
        target_value = 100.0
        with torch.no_grad():
            callback.model._params_list[0].data.fill_(target_value)
        
        # Record initial EMA value
        initial_ema = callback.ema_params[0].data.mean().item()
        
        # Run multiple updates
        for i in range(100):
            callback.num_timesteps = i
            callback._on_step()
        
        # EMA should have moved towards target
        final_ema = callback.ema_params[0].data.mean().item()
        
        # Final EMA should be closer to target than initial
        assert abs(final_ema - target_value) < abs(initial_ema - target_value)
    
    def test_shape_mismatch_reinitialization(self):
        """Test EMA reinitialization when parameter shapes change."""
        callback = ModelEMACallback(decay=0.995, verbose=1)
        callback.model = create_mock_model(n_params=3, param_size=10)
        
        callback._on_training_start()
        
        initial_ema_count = len(callback.ema_params)
        
        # Change number of parameters
        callback.model._params_list = [
            torch.nn.Parameter(torch.randn(10, requires_grad=True))
            for _ in range(5)  # Different count
        ]
        callback.model.policy.parameters = Mock(
            side_effect=lambda: iter(callback.model._params_list)
        )
        
        callback.num_timesteps = 1
        callback._on_step()
        
        # Should have reinitialized with new parameter count
        assert len(callback.ema_params) == 5
    
    def test_load_ema_weights(self):
        """Test loading EMA weights into policy."""
        callback = ModelEMACallback(decay=0.995, verbose=1)
        callback.model = create_mock_model(n_params=2, param_size=10)
        
        callback._on_training_start()
        
        # Set distinct EMA values
        with torch.no_grad():
            callback.ema_params[0].fill_(42.0)
            callback.ema_params[1].fill_(84.0)
        
        # Load EMA into policy
        callback.load_ema_weights()
        
        # Verify policy params match EMA
        policy_params = list(callback.model.policy.parameters())
        assert policy_params[0].data.mean().item() == pytest.approx(42.0, abs=1e-5)
        assert policy_params[1].data.mean().item() == pytest.approx(84.0, abs=1e-5)
    
    def test_load_ema_weights_before_init(self):
        """Test load_ema_weights gracefully handles uninitialized state."""
        callback = ModelEMACallback(decay=0.995, verbose=1)
        callback.model = create_mock_model()
        
        # Should not crash when ema_params is None
        callback.load_ema_weights()  # No exception
    
    def test_save_path_none_skips_save(self):
        """Test that save is skipped when save_path is None."""
        callback = ModelEMACallback(decay=0.995, save_path=None, verbose=1)
        callback.model = create_mock_model()
        
        callback._on_training_start()
        
        # Should not crash and not attempt to save
        callback._on_training_end()


# ============================================================================
# Test get_underlying_batch_env helper
# ============================================================================

class TestGetUnderlyingBatchEnv:
    """Tests for get_underlying_batch_env helper function."""
    
    def test_unwraps_vec_env(self):
        """Test unwrapping VecEnv to find BatchCryptoEnv."""
        # Create mock chain: VecNormalize -> VecMonitor -> BatchCryptoEnv
        batch_env = Mock()
        batch_env.__class__.__name__ = 'BatchCryptoEnv'
        
        monitor = Mock()
        monitor.__class__.__name__ = 'VecMonitor'
        monitor.venv = batch_env
        
        normalize = Mock()
        normalize.__class__.__name__ = 'VecNormalize'
        normalize.venv = monitor
        
        result = get_underlying_batch_env(normalize)
        
        assert result is batch_env
    
    def test_returns_none_for_non_batch_env(self):
        """Test returns None when BatchCryptoEnv not found."""
        env = Mock()
        env.__class__.__name__ = 'DummyVecEnv'
        # No venv attribute
        
        result = get_underlying_batch_env(env)
        
        assert result is None


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

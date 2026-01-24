# -*- coding: utf-8 -*-
"""
test_callbacks.py - Tests for RL Callbacks.

Verifies that the critical callbacks work correctly:
- UnifiedMetricsCallback: TensorBoard logging with standardized namespaces
- ThreePhaseCurriculumCallback: phase transitions and curriculum_lambda
- OverfittingGuardCallbackV2: 5 signal detection and multi-signal decision logic
- ModelEMACallback: Polyak averaging and weight management

Reference: docs/audit/AUDIT_MODELES_RL_RESULTATS.md - Tests Callbacks section
"""

import sys
import os
from unittest.mock import Mock, MagicMock, patch, PropertyMock, PropertyMock
from collections import deque
import numpy as np
import torch
import pytest

# Add the project root to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.training.callbacks import (
    UnifiedMetricsCallback,
    ThreePhaseCurriculumCallback,
    MORLCurriculumCallback,
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
# TestUnifiedMetricsCallback
# ============================================================================

class TestUnifiedMetricsCallback:
    """Tests for UnifiedMetricsCallback TensorBoard logging."""
    
    def test_initialization(self):
        """Test callback initialization with default and custom values."""
        # Default values
        callback = UnifiedMetricsCallback()
        assert callback.log_freq == 100
        assert callback.verbose == 0
        assert len(callback.metrics_buffer) == 0
        
        # Custom values
        callback = UnifiedMetricsCallback(log_freq=50, verbose=1)
        assert callback.log_freq == 50
        assert callback.verbose == 1
    
    def test_collect_light_metrics_from_infos(self):
        """Test collection of light metrics from infos dict."""
        callback = UnifiedMetricsCallback(log_freq=10)
        callback.model = create_mock_model()
        callback.locals = {
            "infos": [{
                "portfolio_value": 10_500.0,
                "position_pct": 0.75,
                "max_drawdown": 0.05
            }]
        }
        
        callback._collect_light_metrics()
        
        assert len(callback.metrics_buffer["portfolio/nav"]) == 1
        assert callback.metrics_buffer["portfolio/nav"][0] == 10_500.0
        assert len(callback.metrics_buffer["portfolio/position_pct"]) == 1
        assert callback.metrics_buffer["portfolio/position_pct"][0] == 0.75
        assert len(callback.metrics_buffer["risk/max_drawdown"]) == 1
        assert callback.metrics_buffer["risk/max_drawdown"][0] == 5.0  # Converted to %
    
    def test_log_buffered_metrics(self):
        """Test logging of buffered metrics using record_mean."""
        callback = UnifiedMetricsCallback(log_freq=10)
        callback.model = create_mock_model()
        
        # Fill buffer
        callback.metrics_buffer["portfolio/nav"] = [10_000.0, 10_100.0, 10_200.0]
        callback.metrics_buffer["portfolio/position_pct"] = [0.5, 0.6, 0.7]
        
        callback._log_buffered_metrics()
        
        # Verify record_mean was called
        record_mean_calls = [call for call in callback.logger.record_mean.call_args_list]
        assert len(record_mean_calls) == 2
        
        # Verify buffer was cleared
        assert len(callback.metrics_buffer) == 0
    
    def test_log_global_metrics(self):
        """Test logging of global metrics from get_global_metrics()."""
        callback = UnifiedMetricsCallback(log_freq=10)
        callback.model = create_mock_model()
        
        # Create mock environment with get_global_metrics
        mock_env = Mock()
        mock_env.get_global_metrics = Mock(return_value={
            "portfolio_value": 10_500.0,
            "position_pct": 0.75,
            "max_drawdown": 0.05,
            "reward/pnl_component": 0.1,
            "reward/churn_cost": 0.01,
            "reward/smoothness": 0.02,
            "reward/downside_risk": 0.005
        })
        callback.model.env = mock_env
        
        with patch('src.training.callbacks.get_underlying_batch_env', return_value=mock_env):
            callback._log_global_metrics()
        
        # Verify logger.record was called for all metrics
        record_calls = callback.logger.record.call_args_list
        keys_logged = [call[0][0] for call in record_calls]
        
        assert "portfolio/nav" in keys_logged
        assert "portfolio/position_pct" in keys_logged
        assert "risk/max_drawdown" in keys_logged
        assert "rewards/pnl_component" in keys_logged
        assert "rewards/total_penalties" in keys_logged
        
        # Verify total_penalties aggregation
        total_penalties_call = [call for call in record_calls if call[0][0] == "rewards/total_penalties"][0]
        assert abs(total_penalties_call[0][1] - 0.035) < 1e-6  # 0.01 + 0.02 + 0.005
    
    def test_log_gradients(self):
        """Test logging of gradient norms."""
        callback = UnifiedMetricsCallback(log_freq=10)
        callback.model = create_mock_model()
        
        # Create mock actor and critic with gradients
        actor = Mock()
        actor_param1 = torch.nn.Parameter(torch.randn(10, requires_grad=True))
        actor_param2 = torch.nn.Parameter(torch.randn(10, requires_grad=True))
        actor_param1.grad = torch.randn(10) * 0.1
        actor_param2.grad = torch.randn(10) * 0.1
        actor.parameters = Mock(return_value=iter([actor_param1, actor_param2]))
        
        critic = Mock()
        critic_param = torch.nn.Parameter(torch.randn(10, requires_grad=True))
        critic_param.grad = torch.randn(10) * 0.05
        critic.parameters = Mock(return_value=iter([critic_param]))
        
        callback.model.policy.actor = actor
        callback.model.policy.critic = critic
        
        callback._log_gradients()
        
        # Verify gradient norms were logged
        record_calls = callback.logger.record.call_args_list
        keys_logged = [call[0][0] for call in record_calls]
        
        assert "debug/grad_actor_norm" in keys_logged
        assert "debug/grad_critic_norm" in keys_logged
    
    def test_log_gradients_no_grads(self):
        """Test gradient logging when no gradients are available."""
        callback = UnifiedMetricsCallback(log_freq=10)
        callback.model = create_mock_model()
        
        # Create mock actor without gradients
        actor = Mock()
        actor_param = torch.nn.Parameter(torch.randn(10, requires_grad=True))
        actor_param.grad = None
        actor.parameters = Mock(return_value=iter([actor_param]))
        
        callback.model.policy.actor = actor
        
        # Should not crash
        callback._log_gradients()
        
        # Should not log anything if no gradients
        record_calls = callback.logger.record.call_args_list
        keys_logged = [call[0][0] for call in record_calls]
        assert "debug/grad_actor_norm" not in keys_logged
    
    def test_handle_episode_end_churn_ratio(self):
        """Test episode end handling and churn ratio calculation."""
        callback = UnifiedMetricsCallback(log_freq=10)
        callback.model = create_mock_model()
        
        # Set up episode data
        callback.episode_churn_penalties = [0.01, 0.02, 0.01]
        callback.episode_log_returns = [0.1, 0.2, 0.1]
        
        callback.locals = {
            "infos": [{
                "episode": {"r": 0.4}
            }]
        }
        
        callback._handle_episode_end()
        
        # Verify churn ratio was logged
        record_calls = callback.logger.record.call_args_list
        keys_logged = [call[0][0] for call in record_calls]
        
        assert "strategy/churn_ratio" in keys_logged
        
        # Verify accumulators were reset
        assert len(callback.episode_churn_penalties) == 0
        assert len(callback.episode_log_returns) == 0
    
    def test_handle_episode_end_no_data(self):
        """Test episode end handling when no data available."""
        callback = UnifiedMetricsCallback(log_freq=10)
        callback.model = create_mock_model()
        
        callback.locals = {
            "infos": [{
                "episode": {"r": 0.4}
            }]
        }
        
        # Should not crash
        callback._handle_episode_end()
        
        # Should not log churn ratio if no data
        record_calls = callback.logger.record.call_args_list
        keys_logged = [call[0][0] for call in record_calls]
        assert "strategy/churn_ratio" not in keys_logged
    
    def test_logging_frequency(self):
        """Test that heavy metrics are only logged at log_freq."""
        callback = UnifiedMetricsCallback(log_freq=10)
        callback.model = create_mock_model()
        callback.logger.dump = Mock()
        
        # Mock environment
        mock_env = Mock()
        mock_env.get_global_metrics = Mock(return_value={})
        callback.model.env = mock_env
        
        with patch('src.training.callbacks.get_underlying_batch_env', return_value=mock_env):
            # Step 1: Should not log (not at log_freq)
            callback.n_calls = 1
            callback.num_timesteps = 1
            callback.locals = {"infos": [{}]}
            callback._on_step()
            
            # dump should not be called
            assert callback.logger.dump.call_count == 0
            
            # Step 10: Should log (at log_freq)
            callback.n_calls = 10
            callback.num_timesteps = 10
            callback._on_step()
            
            # dump should be called once
            assert callback.logger.dump.call_count == 1
    
    def test_console_logging_verbose(self):
        """Test console logging when verbose > 0."""
        callback = UnifiedMetricsCallback(log_freq=10, verbose=1)
        callback.model = create_mock_model()
        callback.logger.dump = Mock()
        
        # Mock environment with metrics
        mock_env = Mock()
        mock_env.get_global_metrics = Mock(return_value={
            "portfolio_value": 10_500.0,
            "position_pct": 0.75,
            "max_drawdown": 0.05
        })
        callback.model.env = mock_env
        
        callback.locals = {
            "infos": [{
                "episode": {"r": 0.4}
            }]
        }
        
        callback.last_time = None
        callback.last_step = 0
        callback.num_timesteps = 10
        callback.n_calls = 10
        
        with patch('src.training.callbacks.get_underlying_batch_env', return_value=mock_env):
            with patch('builtins.print') as mock_print:
                callback._on_step()
                
                # Verify print was called (console logging)
                assert mock_print.called
    
    def test_console_logging_not_verbose(self):
        """Test that console logging is skipped when verbose == 0."""
        callback = UnifiedMetricsCallback(log_freq=10, verbose=0)
        callback.model = create_mock_model()
        callback.logger.dump = Mock()
        
        callback.locals = {"infos": [{}]}
        callback.num_timesteps = 10
        callback.n_calls = 10
        
        with patch('builtins.print') as mock_print:
            callback._on_step()
            
            # Verify print was not called
            assert not mock_print.called
    
    def test_get_training_metrics(self):
        """Test get_training_metrics returns correct diagnostic metrics."""
        callback = UnifiedMetricsCallback()
        
        # Fill diagnostic deques
        callback.all_actions.extend([0.5, 0.6, 0.7])
        callback.entropy_values.extend([1.0, 1.1, 1.2])
        callback.critic_losses.extend([0.1, 0.2, 0.3])
        callback.actor_losses.extend([0.05, 0.1, 0.15])
        callback.churn_ratios.extend([0.1, 0.2, 0.3])
        callback.actor_grad_norms.extend([0.5, 0.6, 0.7])
        callback.critic_grad_norms.extend([0.3, 0.4, 0.5])
        
        metrics = callback.get_training_metrics()
        
        assert "action_saturation" in metrics
        assert "avg_entropy" in metrics
        assert "avg_critic_loss" in metrics
        assert "avg_actor_loss" in metrics
        assert "avg_churn_ratio" in metrics
        assert "avg_actor_grad_norm" in metrics
        assert "avg_critic_grad_norm" in metrics
        
        # Verify values are means
        assert abs(metrics["action_saturation"] - 0.6) < 1e-6
        assert abs(metrics["avg_entropy"] - 1.1) < 1e-6
    
    def test_get_training_metrics_empty(self):
        """Test get_training_metrics with empty deques."""
        callback = UnifiedMetricsCallback()
        
        metrics = callback.get_training_metrics()
        
        # All metrics should be 0.0 when deques are empty
        assert metrics["action_saturation"] == 0.0
        assert metrics["avg_entropy"] == 0.0
        assert metrics["avg_critic_loss"] == 0.0
        assert metrics["avg_actor_loss"] == 0.0
        assert metrics["avg_churn_ratio"] == 0.0
        assert metrics["avg_actor_grad_norm"] == 0.0
        assert metrics["avg_critic_grad_norm"] == 0.0
    
    def test_init_callback_fps_tracking(self):
        """Test that FPS tracking is initialized."""
        callback = UnifiedMetricsCallback()
        callback._init_callback()
        
        assert callback.last_time is not None
        assert callback.last_step == 0
    
    def test_log_tqc_stats_no_critic(self):
        """Test TQC stats logging when critic is not available."""
        callback = UnifiedMetricsCallback(log_freq=10)
        callback.model = create_mock_model()
        
        # No critic in policy
        callback.model.policy.critic = None
        
        # Should not crash
        callback.locals = {"new_obs": np.array([[1.0, 2.0]]), "actions": np.array([[0.5]])}
        callback._log_tqc_stats()
        
        # Should not log anything
        record_calls = callback.logger.record.call_args_list
        keys_logged = [call[0][0] for call in record_calls]
        assert "debug/q_values_mean" not in keys_logged
    
    def test_log_tqc_stats_no_obs_actions(self):
        """Test TQC stats logging when obs/actions are not available."""
        callback = UnifiedMetricsCallback(log_freq=10)
        callback.model = create_mock_model()
        
        # Mock critic
        critic = Mock()
        callback.model.policy.critic = critic
        
        # No obs/actions in locals
        callback.locals = {}
        
        # Should not crash
        callback._log_tqc_stats()
        
        # Should not log anything
        record_calls = callback.logger.record.call_args_list
        keys_logged = [call[0][0] for call in record_calls]
        assert "debug/q_values_mean" not in keys_logged


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
        # logger property reads from model.logger, which is already mocked
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
            # At exactly 15%, phase is still 1 (boundary is inclusive)
            callback.num_timesteps = 15_000  # Exactly at 15%
            callback._on_step()
            assert callback._phase == 1  # Still phase 1 at boundary
            
            # Phase 2 starts after 15%
            callback.num_timesteps = 15_001  # Just after 15%
            callback._on_step()
            assert callback._phase == 2
            
            callback.num_timesteps = 50_000  # Middle of phase 2
            callback._on_step()
            assert callback._phase == 2
            
            callback.num_timesteps = 74_999  # Just before 75%
            callback._on_step()
            assert callback._phase == 2
            
            # Phase 3: 75% - 100%
            # At exactly 75%, phase is still 2 (boundary is inclusive)
            callback.num_timesteps = 75_000  # Exactly at 75%
            callback._on_step()
            assert callback._phase == 2  # Still phase 2 at boundary
            
            # Phase 3 starts after 75%
            callback.num_timesteps = 75_001  # Just after 75%
            callback._on_step()
            assert callback._phase == 3
            
            callback.num_timesteps = 100_000  # End
            callback._on_step()
            assert callback._phase == 3
    
    def test_env_update_called(self):
        """Test that set_progress is called on environment."""
        callback = ThreePhaseCurriculumCallback(total_timesteps=100_000)
        callback.model = create_mock_model()
        # logger property reads from model.logger, which is already mocked
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
        # logger property reads from model.logger, which is already mocked
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
# TestMORLCurriculumCallback
# ============================================================================

class TestMORLCurriculumCallback:
    """Tests for MORLCurriculumCallback (replaces ThreePhaseCurriculumCallback)."""
    
    def test_initialization_default(self):
        """Test callback initialization with default values."""
        callback = MORLCurriculumCallback(total_timesteps=100_000)
        
        assert callback.start_cost == 0.0
        assert callback.end_cost == 0.1
        assert callback.progress_ratio == 0.5
        assert callback.total_timesteps == 100_000
        assert callback.verbose == 0
    
    def test_initialization_custom(self):
        """Test callback initialization with custom values."""
        callback = MORLCurriculumCallback(
            start_cost=0.0,
            end_cost=0.2,
            progress_ratio=0.75,
            total_timesteps=50_000,
            verbose=1
        )
        
        assert callback.start_cost == 0.0
        assert callback.end_cost == 0.2
        assert callback.progress_ratio == 0.75
        assert callback.total_timesteps == 50_000
        assert callback.verbose == 1
    
    def test_initialization_validation(self):
        """Test that start_cost <= end_cost validation works."""
        # Valid: start_cost < end_cost
        callback = MORLCurriculumCallback(
            start_cost=0.0,
            end_cost=0.1,
            total_timesteps=100_000
        )
        assert callback.start_cost == 0.0
        assert callback.end_cost == 0.1
        
        # Valid: start_cost == end_cost
        callback = MORLCurriculumCallback(
            start_cost=0.1,
            end_cost=0.1,
            total_timesteps=100_000
        )
        assert callback.start_cost == 0.1
        assert callback.end_cost == 0.1
        
        # Invalid: start_cost > end_cost
        with pytest.raises(ValueError, match="start_cost.*must be <= end_cost"):
            MORLCurriculumCallback(
                start_cost=0.2,
                end_cost=0.1,
                total_timesteps=100_000
            )
    
    def test_clipping_values(self):
        """Test that cost values are clipped to [0, 1]."""
        # Values > 1.0 should be clipped
        callback = MORLCurriculumCallback(
            start_cost=0.0,
            end_cost=1.5,  # > 1.0
            total_timesteps=100_000
        )
        assert callback.end_cost == 1.0
        
        # Values < 0.0 should be clipped
        callback = MORLCurriculumCallback(
            start_cost=-0.5,  # < 0.0
            end_cost=0.1,
            total_timesteps=100_000
        )
        assert callback.start_cost == 0.0
    
    def test_linear_ramp_phase(self):
        """Test linear ramp calculation during ramp phase (progress < progress_ratio)."""
        callback = MORLCurriculumCallback(
            start_cost=0.0,
            end_cost=0.1,
            progress_ratio=0.5,
            total_timesteps=100_000
        )
        callback.model = create_mock_model()
        mock_env = Mock()
        mock_env.set_w_cost_target = Mock()
        
        with patch('src.training.callbacks.get_underlying_batch_env', return_value=mock_env):
            # At 0% progress: should be start_cost
            callback.num_timesteps = 0
            callback._on_step()
            call_args = mock_env.set_w_cost_target.call_args[0][0]
            assert abs(call_args - 0.0) < 1e-6
            
            # At 25% progress (half of ramp): should be midpoint
            callback.num_timesteps = 25_000  # 25% of total, 50% of ramp
            callback._on_step()
            call_args = mock_env.set_w_cost_target.call_args[0][0]
            assert abs(call_args - 0.05) < 1e-6  # Midpoint: (0.0 + 0.1) / 2
            
            # At 50% progress (end of ramp): should be end_cost
            callback.num_timesteps = 50_000  # Exactly at progress_ratio
            callback._on_step()
            call_args = mock_env.set_w_cost_target.call_args[0][0]
            assert abs(call_args - 0.1) < 1e-6
    
    def test_plateau_phase(self):
        """Test plateau phase (progress >= progress_ratio)."""
        callback = MORLCurriculumCallback(
            start_cost=0.0,
            end_cost=0.1,
            progress_ratio=0.5,
            total_timesteps=100_000
        )
        callback.model = create_mock_model()
        mock_env = Mock()
        mock_env.set_w_cost_target = Mock()
        
        with patch('src.training.callbacks.get_underlying_batch_env', return_value=mock_env):
            # At 50% progress: should be end_cost (boundary)
            callback.num_timesteps = 50_000
            callback._on_step()
            call_args = mock_env.set_w_cost_target.call_args[0][0]
            assert abs(call_args - 0.1) < 1e-6
            
            # At 75% progress: should still be end_cost (plateau)
            callback.num_timesteps = 75_000
            callback._on_step()
            call_args = mock_env.set_w_cost_target.call_args[0][0]
            assert abs(call_args - 0.1) < 1e-6
            
            # At 100% progress: should still be end_cost (plateau)
            callback.num_timesteps = 100_000
            callback._on_step()
            call_args = mock_env.set_w_cost_target.call_args[0][0]
            assert abs(call_args - 0.1) < 1e-6
    
    def test_env_update_called(self):
        """Test that set_w_cost_target is called on environment."""
        callback = MORLCurriculumCallback(
            start_cost=0.0,
            end_cost=0.1,
            progress_ratio=0.5,
            total_timesteps=100_000
        )
        callback.model = create_mock_model()
        mock_env = Mock()
        mock_env.set_w_cost_target = Mock()
        
        with patch('src.training.callbacks.get_underlying_batch_env', return_value=mock_env):
            callback.num_timesteps = 25_000  # 25% progress
            callback._on_step()
            
            # Verify set_w_cost_target was called
            assert mock_env.set_w_cost_target.called
            call_args = mock_env.set_w_cost_target.call_args[0][0]
            assert 0.0 <= call_args <= 0.1  # Should be in valid range
    
    def test_logging_records(self):
        """Test that metrics are logged to TensorBoard."""
        callback = MORLCurriculumCallback(
            start_cost=0.0,
            end_cost=0.1,
            progress_ratio=0.5,
            total_timesteps=100_000
        )
        callback.model = create_mock_model()
        mock_env = Mock()
        mock_env.set_w_cost_target = Mock()
        
        with patch('src.training.callbacks.get_underlying_batch_env', return_value=mock_env):
            callback.num_timesteps = 25_000  # 25% progress
            callback._on_step()
            
            # Verify logger.record was called
            record_calls = callback.logger.record.call_args_list
            keys_logged = [call[0][0] for call in record_calls]
            
            assert "curriculum/w_cost_target" in keys_logged
            assert "curriculum/w_cost_progress" in keys_logged
            
            # Verify values are correct
            w_cost_target_call = next(call for call in record_calls if call[0][0] == "curriculum/w_cost_target")
            w_cost_progress_call = next(call for call in record_calls if call[0][0] == "curriculum/w_cost_progress")
            
            assert abs(w_cost_target_call[0][1] - 0.05) < 1e-6  # 25% of ramp = 0.05
            assert abs(w_cost_progress_call[0][1] - 0.25) < 1e-6  # 25% progress
    
    def test_no_env_warning(self):
        """Test that warning is printed if BatchCryptoEnv not found."""
        callback = MORLCurriculumCallback(
            start_cost=0.0,
            end_cost=0.1,
            total_timesteps=100_000,
            verbose=1
        )
        callback.model = create_mock_model()
        callback.n_calls = 1000  # Multiple of 1000 to trigger warning
        
        # Mock env without set_w_cost_target
        mock_env = Mock()
        del mock_env.set_w_cost_target  # Remove the method
        
        with patch('src.training.callbacks.get_underlying_batch_env', return_value=mock_env):
            with patch('builtins.print') as mock_print:
                callback._on_step()
                # Warning should be printed (verbose=1 and n_calls % 1000 == 0)
                assert any("Warning" in str(call) for call in mock_print.call_args_list)
    
    def test_custom_progress_ratio(self):
        """Test with custom progress_ratio (ramp on 75% of training)."""
        callback = MORLCurriculumCallback(
            start_cost=0.0,
            end_cost=0.2,
            progress_ratio=0.75,  # Ramp on first 75%
            total_timesteps=100_000
        )
        callback.model = create_mock_model()
        mock_env = Mock()
        mock_env.set_w_cost_target = Mock()
        
        with patch('src.training.callbacks.get_underlying_batch_env', return_value=mock_env):
            # At 37.5% progress (half of ramp): should be midpoint
            callback.num_timesteps = 37_500  # 37.5% of total, 50% of ramp
            callback._on_step()
            call_args = mock_env.set_w_cost_target.call_args[0][0]
            assert abs(call_args - 0.1) < 1e-6  # Midpoint: (0.0 + 0.2) / 2
            
            # At 75% progress (end of ramp): should be end_cost
            callback.num_timesteps = 75_000
            callback._on_step()
            call_args = mock_env.set_w_cost_target.call_args[0][0]
            assert abs(call_args - 0.2) < 1e-6
            
            # At 90% progress (plateau): should still be end_cost
            callback.num_timesteps = 90_000
            callback._on_step()
            call_args = mock_env.set_w_cost_target.call_args[0][0]
            assert abs(call_args - 0.2) < 1e-6


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

        # Setup mock env
        mock_env = Mock()
        mock_env.get_global_metrics = Mock(return_value={"portfolio_value": 60_000.0})

        # Patch training_env property (inherited from SB3 BaseCallback)
        with patch.object(type(guard), 'training_env', new_callable=lambda: property(lambda self: mock_env)):
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

        # Setup mock env
        mock_env = Mock()
        mock_env.get_global_metrics = Mock(return_value={"portfolio_value": 20_000.0})

        # Patch training_env property (inherited from SB3 BaseCallback)
        with patch.object(type(guard), 'training_env', new_callable=lambda: property(lambda self: mock_env)):
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
        # logger and training_env are read-only properties, patch via __dict__
        mock_env = Mock()
        mock_env.get_global_metrics = Mock(return_value={})
        guard.__dict__['training_env'] = mock_env
        try:
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
        finally:
            if 'training_env' in guard.__dict__:
                del guard.__dict__['training_env']
    
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
        assert abs(callback.tau - 0.005) < 1e-10  # 1 - decay (with floating point tolerance)
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
        # logger property reads from model.logger, which is already mocked
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
        # Create a simple class that has the identifying method
        class MockBatchEnv:
            def set_smoothness_penalty(self, value):
                pass
        
        batch_env = MockBatchEnv()
        
        # Create monitor with venv attribute
        monitor = Mock(spec=['venv'])
        monitor.__class__.__name__ = 'VecMonitor'
        monitor.venv = batch_env
        # Ensure hasattr works correctly
        type(monitor).venv = property(lambda self: batch_env)
        
        # Create normalize with venv attribute pointing to monitor
        normalize = Mock(spec=['venv'])
        normalize.__class__.__name__ = 'VecNormalize'
        normalize.venv = monitor
        # Ensure hasattr works correctly and venv returns monitor
        type(normalize).venv = property(lambda self: monitor)
        
        result = get_underlying_batch_env(normalize)
        
        # The function should find batch_env by checking hasattr(env, 'set_smoothness_penalty')
        assert result is batch_env
    
    def test_returns_none_for_non_batch_env(self):
        """Test returns original env when BatchCryptoEnv not found."""
        env = Mock()
        env.__class__.__name__ = 'DummyVecEnv'
        # No venv attribute, no set_smoothness_penalty method
        
        result = get_underlying_batch_env(env)
        
        # Function returns the original env when BatchCryptoEnv is not found
        assert result is env


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

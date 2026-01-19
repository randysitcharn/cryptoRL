# -*- coding: utf-8 -*-
"""
test_overfitting_guard_wfo.py - Tests for OverfittingGuardCallbackV2 in WFO mode.

Tests cover:
1. Guard initialization without EvalCallback (Signal 3 disabled)
2. Guard violation detection (all signals)
3. Fail-over logic (RECOVERED vs FAILED)
4. Chain of Inheritance (rollback on failure)
5. Fallback strategies (flat vs buy_and_hold)
"""

import os
import sys
import pytest
import numpy as np
import tempfile
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.callbacks import OverfittingGuardCallbackV2


# =============================================================================
# Test 1: Guard Without EvalCallback (WFO Mode)
# =============================================================================

class TestGuardWithoutEvalCallback:
    """Test that Signal 3 is gracefully disabled when no EvalCallback."""

    def test_guard_init_without_eval_callback(self):
        """Verify Guard initializes correctly without EvalCallback."""
        guard = OverfittingGuardCallbackV2(
            eval_callback=None,  # WFO mode
            verbose=0
        )
        
        assert guard.eval_callback is None
        assert guard.patience == 3  # Default
        assert guard.nav_threshold == 5.0  # Default

    def test_signal3_returns_none_without_eval_callback(self):
        """Verify Signal 3 returns None (no violation) without EvalCallback."""
        guard = OverfittingGuardCallbackV2(
            eval_callback=None,
            verbose=0
        )
        
        # Signal 3 should return None (disabled, not a violation)
        result = guard._check_train_eval_divergence()
        assert result is None, "Signal 3 should be disabled without EvalCallback"

    def test_guard_with_wfo_thresholds(self):
        """Verify Guard accepts WFO-specific thresholds."""
        guard = OverfittingGuardCallbackV2(
            nav_threshold=10.0,        # WFO threshold
            patience=5,                # WFO patience
            check_freq=25_000,         # WFO check freq
            action_saturation_threshold=0.95,
            reward_variance_threshold=1e-5,
            eval_callback=None,
            verbose=0
        )
        
        assert guard.nav_threshold == 10.0
        assert guard.patience == 5
        assert guard.check_freq == 25_000
        assert guard.action_saturation_threshold == 0.95
        assert guard.reward_variance_threshold == 1e-5


# =============================================================================
# Test 2: Signal Detection
# =============================================================================

class TestSignalDetection:
    """Test individual signal detection logic."""

    @pytest.fixture
    def guard(self):
        """Create a Guard instance for testing."""
        return OverfittingGuardCallbackV2(
            nav_threshold=5.0,
            action_saturation_threshold=0.95,
            reward_variance_threshold=1e-4,
            eval_callback=None,
            verbose=0
        )

    def test_action_saturation_detection(self, guard):
        """Test Signal 4: Action saturation detection."""
        # Fill history with saturated actions
        guard.actions_history.extend([0.99] * 1000)
        
        result = guard._check_action_saturation()
        assert result is not None, "Should detect action saturation"
        assert "saturation" in result.lower()

    def test_action_saturation_no_violation(self, guard):
        """Test Signal 4: No violation with varied actions."""
        # Fill history with varied actions
        guard.actions_history.extend(np.random.uniform(0, 0.5, 1000))
        
        result = guard._check_action_saturation()
        assert result is None, "Should not detect saturation with varied actions"

    def test_reward_variance_detection(self, guard):
        """Test Signal 5: Reward variance collapse detection."""
        # Fill history with constant rewards (variance = 0)
        guard.rewards_history.extend([0.01] * 1000)
        
        result = guard._check_reward_variance()
        assert result is not None, "Should detect reward variance collapse"
        assert "variance" in result.lower()

    def test_reward_variance_no_violation(self, guard):
        """Test Signal 5: No violation with varied rewards."""
        # Fill history with varied rewards
        guard.rewards_history.extend(np.random.normal(0, 1, 1000))
        
        result = guard._check_reward_variance()
        assert result is None, "Should not detect collapse with varied rewards"


# =============================================================================
# Test 3: Violation Counting and Decision Logic
# =============================================================================

class TestDecisionLogic:
    """Test the multi-signal decision logic."""

    def test_patience_trigger(self):
        """Test that reaching patience on any signal triggers stop."""
        guard = OverfittingGuardCallbackV2(
            patience=3,
            eval_callback=None,
            verbose=0
        )
        
        # Simulate 3 NAV violations
        guard.violation_counts['nav'] = 3
        
        violations = []
        should_stop = guard._decide_stop(violations)
        
        assert should_stop, "Should stop when patience exhausted"

    def test_multi_signal_trigger(self):
        """Test that 2+ active signals triggers stop."""
        guard = OverfittingGuardCallbackV2(
            patience=3,
            eval_callback=None,
            verbose=0
        )
        
        # Simulate 2 active violations (below patience each)
        guard.violation_counts['nav'] = 1
        guard.violation_counts['saturation'] = 1
        
        violations = ['nav', 'saturation']
        should_stop = guard._decide_stop(violations)
        
        assert should_stop, "Should stop with 2+ active signals"

    def test_single_signal_no_stop(self):
        """Test that single signal below patience doesn't stop."""
        guard = OverfittingGuardCallbackV2(
            patience=3,
            eval_callback=None,
            verbose=0
        )
        
        guard.violation_counts['nav'] = 1  # Below patience
        
        violations = ['nav']
        should_stop = guard._decide_stop(violations)
        
        assert not should_stop, "Should not stop with single signal below patience"


# =============================================================================
# Test 4: WFO Config Integration
# =============================================================================

class TestWFOConfigIntegration:
    """Test WFOConfig parameters for Guard."""

    def test_wfo_config_defaults(self):
        """Verify WFOConfig has correct Guard defaults."""
        # Import dynamically to avoid circular imports
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
        
        from run_full_wfo import WFOConfig
        
        config = WFOConfig()
        
        assert config.use_overfitting_guard == True
        assert config.guard_nav_threshold == 10.0
        assert config.guard_patience == 5
        assert config.guard_check_freq == 25_000
        assert config.guard_action_saturation == 0.95
        assert config.guard_reward_variance == 1e-5

    def test_wfo_config_failover_defaults(self):
        """Verify WFOConfig has correct Fail-over defaults."""
        from run_full_wfo import WFOConfig
        
        config = WFOConfig()
        
        assert config.use_checkpoint_on_failure == True
        assert config.min_completion_ratio == 0.30
        assert config.fallback_strategy == 'flat'

    def test_wfo_config_chain_defaults(self):
        """Verify WFOConfig has correct Chain of Inheritance defaults."""
        from run_full_wfo import WFOConfig
        
        config = WFOConfig()
        
        assert config.use_warm_start == True
        assert config.pretrained_model_path is None
        assert config.cleanup_failed_checkpoints == True


# =============================================================================
# Test 5: Fallback Strategy
# =============================================================================

class TestFallbackStrategy:
    """Test fallback strategy implementations."""

    def test_flat_strategy(self):
        """Test that flat strategy returns zero returns."""
        from run_full_wfo import WFOPipeline, WFOConfig
        
        config = WFOConfig()
        config.fallback_strategy = 'flat'
        pipeline = WFOPipeline(config)
        
        # Create dummy segment and test path
        segment = {'id': 0, 'test_start': 0, 'test_end': 100}
        
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            import pandas as pd
            df = pd.DataFrame({
                'BTC_Close': np.linspace(100, 150, 100)  # 50% increase
            })
            df.to_parquet(f.name)
            
            metrics = pipeline._run_fallback_strategy(segment, f.name)
            
            os.unlink(f.name)
        
        assert metrics['sharpe'] == 0.0
        assert metrics['total_return'] == 0.0
        assert metrics['is_fallback'] == True
        assert metrics['strategy'] == 'FLAT (fallback)'

    def test_buy_and_hold_strategy(self):
        """Test that B&H strategy calculates correct returns."""
        from run_full_wfo import WFOPipeline, WFOConfig
        
        config = WFOConfig()
        config.fallback_strategy = 'buy_and_hold'
        pipeline = WFOPipeline(config)
        
        segment = {'id': 0, 'test_start': 0, 'test_end': 100}
        
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            import pandas as pd
            # 50% price increase (100 -> 150)
            df = pd.DataFrame({
                'BTC_Close': np.linspace(100, 150, 100)
            })
            df.to_parquet(f.name)
            
            metrics = pipeline._run_fallback_strategy(segment, f.name)
            
            os.unlink(f.name)
        
        assert metrics['is_fallback'] == True
        assert metrics['strategy'] == 'BUY_AND_HOLD (fallback)'
        # B&H return should be ~50%
        assert abs(metrics['total_return'] - 0.5) < 0.01


# =============================================================================
# Test 6: TrainingConfig Integration
# =============================================================================

class TestTrainingConfigIntegration:
    """Test TrainingConfig parameters for Guard."""

    def test_training_config_guard_defaults(self):
        """Verify TrainingConfig has Guard parameters."""
        from src.config.training import TQCTrainingConfig
        
        config = TQCTrainingConfig()
        
        assert hasattr(config, 'use_overfitting_guard')
        assert hasattr(config, 'guard_nav_threshold')
        assert hasattr(config, 'guard_patience')
        assert hasattr(config, 'guard_check_freq')
        assert hasattr(config, 'guard_action_saturation')
        assert hasattr(config, 'guard_reward_variance')
        
        # Default should be disabled for standard training
        assert config.use_overfitting_guard == False


# =============================================================================
# Test 7: Checkpoint Recovery Logic
# =============================================================================

class TestCheckpointRecovery:
    """Test checkpoint finding and recovery logic."""

    def test_find_checkpoint_empty_dir(self):
        """Test finding checkpoint in empty directory."""
        from run_full_wfo import WFOPipeline, WFOConfig
        
        config = WFOConfig()
        pipeline = WFOPipeline(config)
        
        # Non-existent segment
        result = pipeline._find_last_valid_checkpoint(999)
        assert result is None

    def test_find_checkpoint_with_files(self):
        """Test finding checkpoint with multiple files."""
        from run_full_wfo import WFOPipeline, WFOConfig
        
        config = WFOConfig()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config.weights_dir = tmpdir
            pipeline = WFOPipeline(config)
            
            # Create checkpoint directory and files
            ckpt_dir = os.path.join(tmpdir, "segment_0", "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            
            # Create fake checkpoints with different mtimes
            import time
            for i, name in enumerate(["ckpt_1000.zip", "ckpt_2000.zip", "ckpt_3000.zip"]):
                path = os.path.join(ckpt_dir, name)
                with open(path, 'w') as f:
                    f.write("fake")
                # Ensure different mtimes
                time.sleep(0.01)
            
            result = pipeline._find_last_valid_checkpoint(0)
            
            # Should return the last created (ckpt_3000.zip)
            assert result is not None
            assert "ckpt_3000.zip" in result


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

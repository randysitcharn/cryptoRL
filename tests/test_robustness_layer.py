# -*- coding: utf-8 -*-
"""
test_robustness_layer.py - Tests for Robustness & Generalization Layer.

Tests:
1. Domain Randomization: Verify fees are sampled per-episode
2. ModelEMACallback: Verify EMA weights are updated correctly
3. Integration: Verify compatibility with MORL/PLO
"""

import sys
import os
import tempfile
import pytest
import numpy as np
import pandas as pd
import torch
from unittest.mock import Mock, MagicMock

# Add the project root to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.training.batch_env import BatchCryptoEnv
from src.training.callbacks import ModelEMACallback


def create_dummy_data(n_rows: int = 1000) -> pd.DataFrame:
    """
    Crée des données factices pour les tests.
    
    Args:
        n_rows (int): Nombre de lignes.
    
    Returns:
        pd.DataFrame: DataFrame avec les mêmes colonnes que les données traitées.
    """
    np.random.seed(42)
    
    # Colonnes du processor (16 colonnes)
    data = {
        'open': np.random.uniform(90, 110, n_rows),
        'high': np.random.uniform(100, 120, n_rows),
        'low': np.random.uniform(80, 100, n_rows),
        'close': np.random.uniform(90, 110, n_rows),
        'RSI_14': np.random.uniform(0, 1, n_rows),
        'MACD_12_26_9': np.random.uniform(-0.01, 0.01, n_rows),
        'MACDh_12_26_9': np.random.uniform(-0.005, 0.005, n_rows),
        'ATRr_14': np.random.uniform(0.01, 0.05, n_rows),
        'BBP_20_2.0': np.random.uniform(-0.5, 1.5, n_rows),
        'BBB_20_2.0': np.random.uniform(0, 0.1, n_rows),
        'log_ret': np.random.uniform(-0.03, 0.03, n_rows),
        'sin_hour': np.random.uniform(-1, 1, n_rows),
        'cos_hour': np.random.uniform(-1, 1, n_rows),
        'sin_day': np.random.uniform(-1, 1, n_rows),
        'cos_day': np.random.uniform(-1, 1, n_rows),
        'volume_rel': np.random.uniform(0.5, 2, n_rows),
    }
    
    return pd.DataFrame(data)


def test_domain_randomization_sampling():
    """Test that domain randomization samples fees per-episode."""
    # Create temporary data file
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp_file:
        df = create_dummy_data(n_rows=500)
        df.to_parquet(tmp_file.name, index=False)
        tmp_path = tmp_file.name
    
    try:
        env = BatchCryptoEnv(
            parquet_path=tmp_path,
            n_envs=4,
            device="cpu",
            enable_domain_randomization=True,
            commission_min=0.0002,
            commission_max=0.0008,
            slippage_min=0.00005,
            slippage_max=0.00015,
        )
        env.training = True  # Set training mode after init
    
    # Reset and capture initial fees
    env.reset()
    initial_commissions = env.commission_per_env.clone()
    initial_slippages = env.slippage_per_env.clone()
    
    # Run a few steps (fees should NOT change)
    for _ in range(10):
        actions = np.random.uniform(-1, 1, (env.num_envs, 1)).astype(np.float32)
        env.step_async(actions)
        env.step_wait()
        
        # Fees should remain constant during episode
        assert torch.allclose(env.commission_per_env, initial_commissions), \
            "Fees should not change during episode"
        assert torch.allclose(env.slippage_per_env, initial_slippages), \
            "Slippages should not change during episode"
    
    # Reset again (fees should change)
    env.reset()
    new_commissions = env.commission_per_env.clone()
    new_slippages = env.slippage_per_env.clone()
    
    # Fees should be different (with high probability)
    # Note: Very unlikely that all fees are identical after resampling
    assert not torch.allclose(new_commissions, initial_commissions, atol=1e-6), \
        "Fees should change after reset (new episode)"
    assert not torch.allclose(new_slippages, initial_slippages, atol=1e-6), \
        "Slippages should change after reset (new episode)"
    
    # Verify fees are within bounds
    assert torch.all(env.commission_per_env >= env.commission_min), \
        "Commissions should be >= commission_min"
    assert torch.all(env.commission_per_env <= env.commission_max), \
        "Commissions should be <= commission_max"
    assert torch.all(env.slippage_per_env >= env.slippage_min), \
        "Slippages should be >= slippage_min"
    assert torch.all(env.slippage_per_env <= env.slippage_max), \
        "Slippages should be <= slippage_max"
    
        print("✓ Domain Randomization: Fees sampled per-episode correctly")
    finally:
        # Cleanup
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_domain_randomization_eval_mode():
    """Test that domain randomization is disabled in eval mode."""
    # Create temporary data file
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp_file:
        df = create_dummy_data(n_rows=500)
        df.to_parquet(tmp_file.name, index=False)
        tmp_path = tmp_file.name
    
    try:
        env = BatchCryptoEnv(
            parquet_path=tmp_path,
            n_envs=4,
            device="cpu",
            enable_domain_randomization=True,
        )
        env.training = False  # Eval mode
    
    env.reset()
    
    # In eval mode, fees should be fixed (not randomized)
    assert torch.allclose(env.commission_per_env, torch.full((env.num_envs,), env.commission)), \
        "In eval mode, commissions should be fixed"
    assert torch.allclose(env.slippage_per_env, torch.full((env.num_envs,), env.slippage)), \
        "In eval mode, slippages should be fixed"
    
        print("✓ Domain Randomization: Disabled in eval mode correctly")
    finally:
        # Cleanup
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_ema_callback_initialization():
    """Test that ModelEMACallback initializes correctly."""
    # Create mock model
    mock_model = Mock()
    mock_policy = Mock()
    mock_policy.parameters.return_value = [
        torch.randn(10, 10, requires_grad=True),
        torch.randn(5, 5, requires_grad=True),
    ]
    mock_model.policy = mock_policy
    
    callback = ModelEMACallback(decay=0.995, verbose=1)
    callback.model = mock_model
    
    # Initialize
    callback._on_training_start()
    
    # Verify EMA params are initialized
    assert callback.ema_params is not None, "EMA params should be initialized"
    assert len(callback.ema_params) == 2, "Should have 2 parameter tensors"
    
    # Verify they are detached clones
    for ema_param, orig_param in zip(callback.ema_params, mock_policy.parameters()):
        assert ema_param.requires_grad == False, "EMA params should not require grad"
        assert ema_param.shape == orig_param.shape, "EMA params should have same shape"
        # Note: At initialization, EMA params equal original params (they are clones)
        # They will diverge after updates
    
    print("✓ ModelEMACallback: Initialization correct")


def test_ema_callback_update():
    """Test that ModelEMACallback updates EMA weights correctly."""
    # Create mock model with actual parameters
    param1 = torch.randn(10, 10, requires_grad=True)
    param2 = torch.randn(5, 5, requires_grad=True)
    
    mock_model = Mock()
    mock_policy = Mock()
    mock_policy.parameters.return_value = [param1, param2]
    mock_model.policy = mock_policy
    mock_model.logger = Mock()
    mock_model.logger.record = Mock()
    
    callback = ModelEMACallback(decay=0.995, verbose=0)
    callback.model = mock_model
    callback.num_timesteps = 0
    
    # Initialize
    callback._on_training_start()
    initial_ema = [p.clone() for p in callback.ema_params]
    
    # Update a few times
    for i in range(5):
        # Modify original parameters (simulate training)
        with torch.no_grad():
            param1.add_(0.1)
            param2.add_(0.1)
        
        callback.num_timesteps = (i + 1) * 1000
        callback._on_step()
    
    # EMA should have moved towards current params (but not fully)
    for ema_param, initial_ema_param, current_param in zip(
        callback.ema_params, initial_ema, [param1, param2]
    ):
        # EMA should be between initial and current (tau=0.005 is small, so closer to initial)
        assert not torch.allclose(ema_param, initial_ema_param, atol=1e-3), \
            "EMA should have moved from initial"
        assert not torch.allclose(ema_param, current_param, atol=1e-3), \
            "EMA should not equal current (tau is small)"
    
    print("✓ ModelEMACallback: EMA weights updated correctly")


def test_ema_callback_load_weights():
    """Test that ModelEMACallback can load EMA weights."""
    # Create mock model
    param1 = torch.randn(10, 10, requires_grad=True)
    param2 = torch.randn(5, 5, requires_grad=True)
    
    mock_model = Mock()
    mock_policy = Mock()
    mock_policy.parameters.return_value = [param1, param2]
    mock_model.policy = mock_policy
    
    callback = ModelEMACallback(decay=0.995, verbose=0)
    callback.model = mock_model
    
    # Initialize
    callback._on_training_start()
    ema_params_snapshot = [p.clone() for p in callback.ema_params]
    
    # Modify current params
    with torch.no_grad():
        param1.add_(1.0)
        param2.add_(1.0)
    
    current_params_snapshot = [p.clone() for p in [param1, param2]]
    
    # Load EMA weights
    callback.load_ema_weights()
    
    # Current params should now equal EMA params
    for current_param, ema_param in zip([param1, param2], callback.ema_params):
        assert torch.allclose(current_param, ema_param), \
            "Current params should equal EMA params after load_ema_weights"
    
    print("✓ ModelEMACallback: load_ema_weights works correctly")


def test_integration_morl_compatibility():
    """Test that Domain Randomization works with MORL (w_cost)."""
    # Create temporary data file
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp_file:
        df = create_dummy_data(n_rows=500)
        df.to_parquet(tmp_file.name, index=False)
        tmp_path = tmp_file.name
    
    try:
        env = BatchCryptoEnv(
            parquet_path=tmp_path,
            n_envs=4,
            device="cpu",
            enable_domain_randomization=True,
        )
        env.training = True  # Set training mode
    
    env.reset()
    
    # Verify w_cost is still sampled (MORL functionality)
    assert env.w_cost.shape == (env.num_envs, 1), "w_cost should have correct shape"
    assert torch.all(env.w_cost >= 0.0) and torch.all(env.w_cost <= 1.0), \
        "w_cost should be in [0, 1]"
    
    # Verify domain randomization is active
    assert torch.any(env.commission_per_env != env.commission), \
        "Commissions should be randomized"
    assert torch.any(env.slippage_per_env != env.slippage), \
        "Slippages should be randomized"
    
    # Run a step and verify reward calculation still works
    actions = np.random.uniform(-1, 1, (env.num_envs, 1)).astype(np.float32)
    env.step_async(actions)
    obs, rewards, dones, infos = env.step_wait()
    
    assert rewards.shape == (env.num_envs,), "Rewards should have correct shape"
    assert not np.isnan(rewards).any(), "Rewards should not contain NaN"
    
        print("✓ Integration: Domain Randomization compatible with MORL")
    finally:
        # Cleanup
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    print("Testing Robustness & Generalization Layer...")
    print("=" * 60)
    
    try:
        test_domain_randomization_sampling()
        test_domain_randomization_eval_mode()
        test_ema_callback_initialization()
        test_ema_callback_update()
        test_ema_callback_load_weights()
        test_integration_morl_compatibility()
        
        print("=" * 60)
        print("✓ All tests passed!")
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

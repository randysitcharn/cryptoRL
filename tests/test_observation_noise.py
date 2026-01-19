# -*- coding: utf-8 -*-
"""
test_observation_noise.py - Tests du Dynamic Observation Noise.

Vérifie que le mécanisme de bruit d'observation fonctionne correctement:
1. Noise Annealing (décroissance avec progress)
2. Volatility-Adaptive (inverse de la volatilité)
3. Désactivation en mode eval
4. Logging pour TensorBoard

Ref: docs/AUDIT_OBSERVATION_NOISE.md
"""

import sys
import os
import tempfile
import numpy as np
import pandas as pd
import torch

# Add the project root to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.training.batch_env import BatchCryptoEnv


def create_dummy_data(n_rows: int = 1000) -> pd.DataFrame:
    """Crée des données factices pour les tests."""
    np.random.seed(42)
    
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


def create_test_env(
    n_rows: int = 1000,
    window_size: int = 64,
    episode_length: int = 100,
    observation_noise: float = 0.01,
    n_envs: int = 4,
):
    """Create a BatchCryptoEnv with test data."""
    df = create_dummy_data(n_rows=n_rows)
    
    tmp_file = tempfile.NamedTemporaryFile(suffix='.parquet', delete=False)
    df.to_parquet(tmp_file.name)
    
    env = BatchCryptoEnv(
        parquet_path=tmp_file.name,
        n_envs=n_envs,
        device='cpu',
        window_size=window_size,
        episode_length=episode_length,
        price_column='close',
        random_start=False,
        observation_noise=observation_noise,
        target_volatility=0.015,  # 1.5% target vol
    )
    
    return env, tmp_file.name


def cleanup_env(parquet_path: str):
    """Remove temporary parquet file."""
    try:
        os.unlink(parquet_path)
    except Exception:
        pass


# =============================================================================
# TEST 1: Noise Disabled in Eval Mode
# =============================================================================
def test_noise_disabled_in_eval():
    """Vérifie que le bruit est désactivé en mode évaluation."""
    print("\n" + "="*60)
    print("TEST 1: Noise Disabled in Eval Mode")
    print("="*60)
    
    env, tmp_path = create_test_env(observation_noise=0.01, n_envs=1)
    try:
        # Reset and get initial observation
        env.reset()
        
        # Set to eval mode (no noise)
        env.set_training_mode(False)
        assert env.training == False, "Training mode should be False"
        
        # Get two observations - they should be IDENTICAL without noise
        # (same state, same position, no randomness)
        obs1 = env._get_observations()
        obs2 = env._get_observations()
        
        market1 = obs1['market']
        market2 = obs2['market']
        
        # In eval mode, observations should be deterministic
        assert np.allclose(market1, market2), \
            f"Eval mode: observations should be identical! Max diff: {np.max(np.abs(market1 - market2))}"
        
        print("✅ PASSED: Noise correctly disabled in eval mode")
        print(f"   Max difference between observations: {np.max(np.abs(market1 - market2)):.2e}")
        
    finally:
        cleanup_env(tmp_path)


# =============================================================================
# TEST 2: Noise Active in Training Mode
# =============================================================================
def test_noise_active_in_training():
    """Vérifie que le bruit est actif en mode training."""
    print("\n" + "="*60)
    print("TEST 2: Noise Active in Training Mode")
    print("="*60)
    
    env, tmp_path = create_test_env(observation_noise=0.01, n_envs=1)
    try:
        env.reset()
        
        # Set to training mode (with noise)
        env.set_training_mode(True)
        assert env.training == True, "Training mode should be True"
        
        # Get two observations - they should be DIFFERENT due to noise
        obs1 = env._get_observations()
        obs2 = env._get_observations()
        
        market1 = obs1['market']
        market2 = obs2['market']
        
        # In training mode, observations should differ due to noise
        max_diff = np.max(np.abs(market1 - market2))
        assert max_diff > 0, \
            "Training mode: observations should differ due to noise!"
        
        print("✅ PASSED: Noise correctly active in training mode")
        print(f"   Max difference between observations: {max_diff:.4f}")
        
    finally:
        cleanup_env(tmp_path)


# =============================================================================
# TEST 3: Noise Annealing (decreases with progress)
# =============================================================================
def test_noise_annealing():
    """Vérifie que le bruit décroît avec le progress."""
    print("\n" + "="*60)
    print("TEST 3: Noise Annealing (decreases with progress)")
    print("="*60)
    
    env, tmp_path = create_test_env(observation_noise=0.01, n_envs=16)
    try:
        env.reset()
        env.set_training_mode(True)
        
        # Measure noise at progress=0.0
        env.progress = 0.0
        
        # Get multiple observations and measure variance
        variances_start = []
        for _ in range(10):
            obs = env._get_observations()
            variances_start.append(np.var(obs['market']))
        noise_start = env._last_noise_scale
        
        # Measure noise at progress=1.0
        env.progress = 1.0
        
        variances_end = []
        for _ in range(10):
            obs = env._get_observations()
            variances_end.append(np.var(obs['market']))
        noise_end = env._last_noise_scale
        
        # The effective noise scale should be lower at progress=1.0
        # Annealing factor: 1.0 - 0.5 * progress
        # At progress=0: factor=1.0, at progress=1: factor=0.5
        
        print(f"   Noise scale at progress=0.0: {noise_start:.6f}")
        print(f"   Noise scale at progress=1.0: {noise_end:.6f}")
        print(f"   Expected ratio (end/start): ~0.5")
        print(f"   Actual ratio: {noise_end/noise_start:.3f}")
        
        # Allow some tolerance due to volatility factor
        assert noise_end < noise_start, \
            f"Noise should decrease with progress! Start: {noise_start}, End: {noise_end}"
        
        # The ratio should be around 0.5 (annealing factor)
        ratio = noise_end / noise_start
        assert 0.3 < ratio < 0.7, \
            f"Annealing ratio should be ~0.5, got {ratio:.3f}"
        
        print("✅ PASSED: Noise correctly decreases with progress (annealing)")
        
    finally:
        cleanup_env(tmp_path)


# =============================================================================
# TEST 4: Volatility-Adaptive Noise
# =============================================================================
def test_volatility_adaptive():
    """Vérifie que le bruit s'adapte à la volatilité."""
    print("\n" + "="*60)
    print("TEST 4: Volatility-Adaptive Noise")
    print("="*60)
    
    env, tmp_path = create_test_env(observation_noise=0.01, n_envs=4)
    try:
        env.reset()
        env.set_training_mode(True)
        env.progress = 0.5  # Fixed progress for this test
        
        # Simulate LOW volatility (calm market)
        # ema_vars should be low -> vol_factor should be HIGH (clamped at 2.0)
        env.ema_vars = torch.full((4,), 0.0001, device=env.device)  # Very low vol
        _ = env._get_observations()
        noise_low_vol = env._last_noise_scale
        
        # Simulate HIGH volatility (volatile market)
        # ema_vars should be high -> vol_factor should be LOW (clamped at 0.5)
        env.ema_vars = torch.full((4,), 0.01, device=env.device)  # High vol
        _ = env._get_observations()
        noise_high_vol = env._last_noise_scale
        
        print(f"   Target volatility: {env.target_volatility}")
        print(f"   Low vol scenario (ema_vars=0.0001): noise_scale = {noise_low_vol:.6f}")
        print(f"   High vol scenario (ema_vars=0.01): noise_scale = {noise_high_vol:.6f}")
        print(f"   Ratio (low_vol / high_vol): {noise_low_vol / noise_high_vol:.2f}")
        
        # In low vol market, noise should be HIGHER (more regularization needed)
        # In high vol market, noise should be LOWER (market already noisy)
        assert noise_low_vol > noise_high_vol, \
            f"Noise should be higher in low-vol market! Low: {noise_low_vol}, High: {noise_high_vol}"
        
        # The ratio should be up to 4x (2.0/0.5 = 4 due to clamping)
        ratio = noise_low_vol / noise_high_vol
        assert ratio > 1.5, \
            f"Volatility adaptation ratio should be > 1.5, got {ratio:.2f}"
        
        print("✅ PASSED: Noise correctly adapts to volatility (inverse relationship)")
        
    finally:
        cleanup_env(tmp_path)


# =============================================================================
# TEST 5: Clamping Bounds [0.5, 2.0]
# =============================================================================
def test_clamping_bounds():
    """Vérifie que le vol_factor est bien borné entre 0.5 et 2.0."""
    print("\n" + "="*60)
    print("TEST 5: Clamping Bounds [0.5, 2.0]")
    print("="*60)
    
    env, tmp_path = create_test_env(observation_noise=0.01, n_envs=4)
    try:
        env.reset()
        env.set_training_mode(True)
        env.progress = 0.0  # No annealing effect
        
        # Test EXTREME low volatility (should clamp vol_factor at 2.0)
        env.ema_vars = torch.full((4,), 1e-10, device=env.device)  # Near-zero vol
        _ = env._get_observations()
        noise_extreme_low = env._last_noise_scale
        
        # Test EXTREME high volatility (should clamp vol_factor at 0.5)
        env.ema_vars = torch.full((4,), 100.0, device=env.device)  # Extreme vol
        _ = env._get_observations()
        noise_extreme_high = env._last_noise_scale
        
        # Expected: with annealing_factor=1.0 and observation_noise=0.01
        # Max noise: 0.01 * 1.0 * 2.0 = 0.02
        # Min noise: 0.01 * 1.0 * 0.5 = 0.005
        
        print(f"   Extreme low vol -> noise_scale: {noise_extreme_low:.6f} (expected ~0.02)")
        print(f"   Extreme high vol -> noise_scale: {noise_extreme_high:.6f} (expected ~0.005)")
        
        # Check bounds (with small tolerance for floating point)
        assert 0.018 < noise_extreme_low < 0.022, \
            f"Max noise should be ~0.02, got {noise_extreme_low}"
        assert 0.004 < noise_extreme_high < 0.006, \
            f"Min noise should be ~0.005, got {noise_extreme_high}"
        
        # Ratio should be exactly 4.0 (2.0 / 0.5)
        ratio = noise_extreme_low / noise_extreme_high
        assert 3.8 < ratio < 4.2, \
            f"Clamping ratio should be ~4.0, got {ratio:.2f}"
        
        print("✅ PASSED: Vol_factor correctly clamped to [0.5, 2.0]")
        
    finally:
        cleanup_env(tmp_path)


# =============================================================================
# TEST 6: TensorBoard Logging Attribute
# =============================================================================
def test_tensorboard_logging_attribute():
    """Vérifie que _last_noise_scale est correctement mis à jour."""
    print("\n" + "="*60)
    print("TEST 6: TensorBoard Logging Attribute")
    print("="*60)
    
    env, tmp_path = create_test_env(observation_noise=0.01, n_envs=4)
    try:
        env.reset()
        env.set_training_mode(True)
        
        # Initial value should be 0.0
        assert hasattr(env, '_last_noise_scale'), \
            "Environment should have _last_noise_scale attribute"
        
        # After getting observations, it should be updated
        env.progress = 0.5
        _ = env._get_observations()
        
        noise_scale = env._last_noise_scale
        assert noise_scale > 0, \
            f"_last_noise_scale should be > 0 after observation, got {noise_scale}"
        assert isinstance(noise_scale, float), \
            f"_last_noise_scale should be float, got {type(noise_scale)}"
        
        print(f"   _last_noise_scale value: {noise_scale:.6f}")
        print(f"   Type: {type(noise_scale)}")
        print("✅ PASSED: TensorBoard logging attribute correctly updated")
        
    finally:
        cleanup_env(tmp_path)


# =============================================================================
# TEST 7: No Noise When observation_noise=0
# =============================================================================
def test_no_noise_when_disabled():
    """Vérifie qu'aucun bruit n'est ajouté quand observation_noise=0."""
    print("\n" + "="*60)
    print("TEST 7: No Noise When observation_noise=0")
    print("="*60)
    
    env, tmp_path = create_test_env(observation_noise=0.0, n_envs=1)
    try:
        env.reset()
        env.set_training_mode(True)
        
        # Even in training mode, no noise should be added
        obs1 = env._get_observations()
        obs2 = env._get_observations()
        
        market1 = obs1['market']
        market2 = obs2['market']
        
        assert np.allclose(market1, market2), \
            f"With noise=0, observations should be identical! Max diff: {np.max(np.abs(market1 - market2))}"
        
        print("✅ PASSED: No noise added when observation_noise=0")
        
    finally:
        cleanup_env(tmp_path)


# =============================================================================
# TEST 8: Combined Annealing + Volatility
# =============================================================================
def test_combined_factors():
    """Vérifie que annealing et volatility se combinent correctement."""
    print("\n" + "="*60)
    print("TEST 8: Combined Annealing + Volatility Factors")
    print("="*60)
    
    env, tmp_path = create_test_env(observation_noise=0.01, n_envs=4)
    try:
        env.reset()
        env.set_training_mode(True)
        
        # Set normal volatility (vol_factor ~ 1.0)
        target_vol = env.target_volatility
        env.ema_vars = torch.full((4,), target_vol**2, device=env.device)
        
        # Test at different progress values
        results = []
        for progress in [0.0, 0.25, 0.5, 0.75, 1.0]:
            env.progress = progress
            _ = env._get_observations()
            results.append((progress, env._last_noise_scale))
        
        print("   Progress -> Noise Scale (with normal volatility):")
        for progress, scale in results:
            expected = 0.01 * (1.0 - 0.5 * progress)  # Approx, vol_factor ~1
            print(f"   {progress:.2f} -> {scale:.6f} (expected ~{expected:.6f})")
        
        # Verify monotonic decrease
        scales = [s for _, s in results]
        for i in range(len(scales) - 1):
            assert scales[i] >= scales[i+1], \
                f"Noise should decrease monotonically! {scales[i]} -> {scales[i+1]}"
        
        # Verify approximate values (allowing for vol_factor variation)
        assert 0.008 < results[0][1] < 0.012, f"At progress=0, noise should be ~0.01"
        assert 0.004 < results[-1][1] < 0.006, f"At progress=1, noise should be ~0.005"
        
        print("✅ PASSED: Annealing and volatility factors combine correctly")
        
    finally:
        cleanup_env(tmp_path)


# =============================================================================
# MAIN
# =============================================================================
def run_all_tests():
    """Execute all observation noise tests."""
    print("\n" + "="*60)
    print("DYNAMIC OBSERVATION NOISE - TEST SUITE")
    print("Ref: docs/AUDIT_OBSERVATION_NOISE.md")
    print("="*60)
    
    tests = [
        test_noise_disabled_in_eval,
        test_noise_active_in_training,
        test_noise_annealing,
        test_volatility_adaptive,
        test_clamping_bounds,
        test_tensorboard_logging_attribute,
        test_no_noise_when_disabled,
        test_combined_factors,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"❌ FAILED: {test.__name__}")
            print(f"   Error: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ ERROR: {test.__name__}")
            print(f"   Exception: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"RESULTS: {passed}/{passed+failed} tests passed")
    print("="*60)
    
    if failed > 0:
        print("❌ Some tests FAILED!")
        return 1
    else:
        print("✅ All tests PASSED!")
        return 0


if __name__ == "__main__":
    exit_code = run_all_tests()
    exit(exit_code)

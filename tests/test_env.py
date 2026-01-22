# -*- coding: utf-8 -*-
"""
test_env.py - Tests de l'environnement de trading.

Vérifie que BatchCryptoEnv fonctionne correctement avec l'interface
Gymnasium via gym_reset() et gym_step().
"""

import sys
import os
import tempfile
import numpy as np
import pandas as pd

# Add the project root to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.training.batch_env import BatchCryptoEnv


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


def create_test_env(n_rows: int = 1000, window_size: int = 64, episode_length: int = 100):
    """Create a BatchCryptoEnv with test data in a temporary parquet file."""
    df = create_dummy_data(n_rows=n_rows)
    
    # Create temporary parquet file
    tmp_file = tempfile.NamedTemporaryFile(suffix='.parquet', delete=False)
    df.to_parquet(tmp_file.name)
    
    env = BatchCryptoEnv(
        parquet_path=tmp_file.name,
        n_envs=1,
        device='cpu',
        window_size=window_size,
        episode_length=episode_length,
        price_column='close',
        random_start=False,
    )
    
    return env, tmp_file.name


def cleanup_env(parquet_path: str):
    """Remove temporary parquet file."""
    try:
        os.unlink(parquet_path)
    except Exception:
        pass


def test_reset():
    """Test de la méthode gym_reset."""
    print("Testing gym_reset()...")

    env, tmp_path = create_test_env()
    try:
        obs, info = env.gym_reset()

        # BatchCryptoEnv returns dict observation
        assert isinstance(obs, dict), f"Expected dict, got {type(obs)}"
        assert 'market' in obs, "Observation should contain 'market'"
        assert 'position' in obs, "Observation should contain 'position'"
        
        # Check shapes
        assert obs['market'].shape == (64, 12), f"Expected market shape (64, 12), got {obs['market'].shape}"
        assert obs['position'].shape == (1,), f"Expected position shape (1,), got {obs['position'].shape}"
        
        # Check info
        assert 'nav' in info, "Info should contain 'nav'"
        assert info['nav'] == 10000.0, f"Expected initial NAV 10000, got {info['nav']}"

        print("SUCCESS: gym_reset() works correctly!")
    finally:
        cleanup_env(tmp_path)


def test_step():
    """Test de la méthode gym_step."""
    print("Testing gym_step()...")

    env, tmp_path = create_test_env()
    try:
        obs, info = env.gym_reset()

        # Test action neutre
        action = np.array([0.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.gym_step(action)

        assert isinstance(obs, dict), f"Expected dict, got {type(obs)}"
        assert isinstance(reward, float), f"Reward should be float, got {type(reward)}"
        assert isinstance(terminated, bool), "Terminated should be bool"
        assert isinstance(truncated, bool), "Truncated should be bool"

        # Test action long
        action = np.array([1.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.gym_step(action)

        # Test action short
        action = np.array([-1.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.gym_step(action)

        print("SUCCESS: gym_step() works correctly!")
    finally:
        cleanup_env(tmp_path)


def test_episode_completion():
    """Test qu'un épisode se termine correctement."""
    print("Testing episode completion...")

    env, tmp_path = create_test_env(n_rows=200, episode_length=50)
    try:
        obs, info = env.gym_reset()

        done = False
        steps = 0
        while not done:
            action = np.random.uniform(-1, 1, size=(1,)).astype(np.float32)
            obs, reward, terminated, truncated, info = env.gym_step(action)
            done = terminated or truncated
            steps += 1
            if steps > 100:  # Safety limit
                break

        assert steps == 50, f"Expected 50 steps, got {steps}"
        print(f"SUCCESS: Episode completed in {steps} steps!")
    finally:
        cleanup_env(tmp_path)


def test_batch_mode():
    """Test du mode batch (n_envs > 1)."""
    print("Testing batch mode (n_envs=4)...")

    df = create_dummy_data(n_rows=500)
    tmp_file = tempfile.NamedTemporaryFile(suffix='.parquet', delete=False)
    df.to_parquet(tmp_file.name)
    
    try:
        env = BatchCryptoEnv(
            parquet_path=tmp_file.name,
            n_envs=4,
            device='cpu',
            window_size=64,
            episode_length=50,
            price_column='close',
        )
        
        obs = env.reset()
        
        # Check batch shapes
        assert obs['market'].shape == (4, 64, 12), f"Expected (4, 64, 12), got {obs['market'].shape}"
        assert obs['position'].shape == (4, 1), f"Expected (4, 1), got {obs['position'].shape}"
        
        # Run a few steps
        for _ in range(10):
            actions = np.random.uniform(-1, 1, size=(4, 1)).astype(np.float32)
            env.step_async(actions)
            obs, rewards, dones, infos = env.step_wait()
            
            assert rewards.shape == (4,), f"Expected rewards shape (4,), got {rewards.shape}"
            assert dones.shape == (4,), f"Expected dones shape (4,), got {dones.shape}"
        
        print("SUCCESS: Batch mode works correctly!")
    finally:
        # Close environment to release file handle on Windows
        if 'env' in locals():
            env.close()
        # Small delay to allow file handle to be released on Windows
        import time
        time.sleep(0.1)
        try:
            os.unlink(tmp_file.name)
        except PermissionError:
            # File might still be locked, skip deletion (will be cleaned up by tempfile)
            pass


if __name__ == "__main__":
    test_reset()
    test_step()
    test_episode_completion()
    test_batch_mode()

    print("\n[OK] All tests passed!")

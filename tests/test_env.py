"""
test_env.py - Tests de l'environnement de trading.

Vérifie que CryptoTradingEnv est compatible avec Stable-Baselines3
en utilisant check_env et des tests de base.
"""

import sys
import os
import numpy as np
import pandas as pd

# Add the project root to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.env.trading_env import CryptoTradingEnv
from stable_baselines3.common.env_checker import check_env


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


def test_check_env():
    """Test de compatibilité Stable-Baselines3."""
    print("Testing SB3 compatibility with check_env...")

    df = create_dummy_data()
    env = CryptoTradingEnv(df)

    # check_env raises an exception if something is wrong
    check_env(env, warn=True)

    print("SUCCESS: Environment passed check_env!")


def test_reset():
    """Test de la méthode reset."""
    print("Testing reset()...")

    df = create_dummy_data()
    env = CryptoTradingEnv(df)

    obs, info = env.reset()

    assert obs.shape == (16,), f"Expected shape (16,), got {obs.shape}"
    assert obs.dtype == np.float32, f"Expected float32, got {obs.dtype}"
    assert 'step' in info, "Info should contain 'step'"
    assert 'balance' in info, "Info should contain 'balance'"

    print("SUCCESS: reset() works correctly!")


def test_step():
    """Test de la méthode step."""
    print("Testing step()...")

    df = create_dummy_data()
    env = CryptoTradingEnv(df)

    obs, info = env.reset()

    # Test action neutre
    action = np.array([0.0], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)

    assert obs.shape == (16,), f"Expected shape (16,), got {obs.shape}"
    assert isinstance(reward, float), f"Reward should be float, got {type(reward)}"
    assert isinstance(terminated, bool), "Terminated should be bool"
    assert isinstance(truncated, bool), "Truncated should be bool"

    # Test action long
    action = np.array([1.0], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)

    # Test action short
    action = np.array([-1.0], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)

    print("SUCCESS: step() works correctly!")


def test_episode_completion():
    """Test qu'un épisode se termine correctement."""
    print("Testing episode completion...")

    df = create_dummy_data(n_rows=100)
    env = CryptoTradingEnv(df)

    obs, info = env.reset()

    done = False
    steps = 0
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1

    assert steps == 99, f"Expected 99 steps, got {steps}"
    print(f"SUCCESS: Episode completed in {steps} steps!")


if __name__ == "__main__":
    test_check_env()
    test_reset()
    test_step()
    test_episode_completion()

    print("\n[OK] All tests passed!")
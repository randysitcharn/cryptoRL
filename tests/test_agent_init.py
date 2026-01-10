"""
test_agent_init.py - Test d'initialisation de l'agent SAC.

Verifie que l'agent SAC peut etre cree et predire une action.
"""

import sys
import os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import DEVICE
from src.training.env import CryptoTradingEnv
from src.models.agent import create_sac_agent


def test_agent_initialization():
    """Test: Agent SAC initializes correctly."""
    print("Test: SAC Agent Initialization...")

    # Load processed data
    df = pd.read_csv("data/processed/BTC-USD_processed.csv", index_col=0, parse_dates=True)
    print(f"  Loaded {len(df)} rows of processed data")

    # Create environment
    env = CryptoTradingEnv(df)

    # Create agent
    agent = create_sac_agent(env)
    print(f"  Agent created on device: {DEVICE}")

    # Sanity check: predict action
    obs, _ = env.reset()
    action, _ = agent.predict(obs, deterministic=True)

    # Verify action shape and bounds
    assert action.shape == (1,), f"Expected shape (1,), got {action.shape}"
    assert -1.0 <= action[0] <= 1.0, f"Action {action[0]} out of bounds [-1, 1]"

    print(f"  Action output shape: {action.shape}")
    print(f"  Action value: {action[0]:.4f}")
    print("  PASSED!")


if __name__ == "__main__":
    print("=" * 50)
    print("Testing SAC Agent Initialization")
    print("=" * 50)

    test_agent_initialization()

    print("\n" + "=" * 50)
    print("[OK] Agent initialization test passed!")
    print("=" * 50)

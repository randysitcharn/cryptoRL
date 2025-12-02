"""
agent.py - Factory pour l'agent SAC.

Cree et configure un agent SAC (Soft Actor-Critic) de Stable-Baselines3
avec des hyperparametres SOTA pour le trading crypto.
"""

from stable_baselines3 import SAC
import torch

from src.config import DEVICE, SEED


def create_sac_agent(env, hyperparams=None):
    """
    Cree un agent SAC configure pour le trading.

    Args:
        env: Gymnasium environment (CryptoTradingEnv).
        hyperparams (dict, optional): Override des hyperparametres.

    Returns:
        SAC: Agent SAC instancie.
    """
    # Hyperparametres SOTA par defaut
    default_params = {
        "policy": "MlpPolicy",
        "learning_rate": 3e-4,
        "buffer_size": 100_000,
        "batch_size": 256,
        "ent_coef": "auto",
        "gamma": 0.99,
        "tau": 0.005,
        "train_freq": 1,
        "gradient_steps": 1,
        "policy_kwargs": dict(net_arch=[256, 256]),
    }

    # Merge avec hyperparams custom
    if hyperparams is not None:
        default_params.update(hyperparams)

    # Extraire policy separement (pas un kwarg de SAC)
    policy = default_params.pop("policy")

    # Creer l'agent
    agent = SAC(
        policy,
        env,
        verbose=1,
        seed=SEED,
        device=DEVICE,
        **default_params
    )

    return agent

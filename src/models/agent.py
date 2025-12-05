# -*- coding: utf-8 -*-
"""
agent.py - Factory pour l'agent TQC avec Transformer.

Cree et configure un agent TQC (Truncated Quantile Critics) de sb3-contrib
avec une architecture Transformer personnalisee optimisee pour les petits
datasets financiers (Low Data Regime).

TQC ameliorations vs SAC:
- Distributional RL avec quantiles
- Truncation des estimations optimistes (top_quantiles_to_drop)
- Plus robuste a l'overestimation bias
"""

from sb3_contrib import TQC

from src.config import DEVICE, SEED
from src.models.transformer_policy import TransformerFeatureExtractor


def create_tqc_agent(env, hyperparams=None, tensorboard_log=None):
    """
    Cree un agent TQC avec Transformer pour le trading.

    Args:
        env: Gymnasium environment (CryptoTradingEnv).
        hyperparams (dict, optional): Override des hyperparametres.
        tensorboard_log (str, optional): Path pour les logs TensorBoard.

    Returns:
        TQC: Agent TQC instancie avec Transformer feature extractor.
    """
    # Policy kwargs avec Transformer extractor
    policy_kwargs = dict(
        features_extractor_class=TransformerFeatureExtractor,
        features_extractor_kwargs=dict(
            features_dim=256,
            d_model=64,
            nhead=4,
            num_layers=2,
            dim_feedforward=128,
            dropout=0.2
        ),
        net_arch=[256, 256],
        n_critics=2,
        n_quantiles=25
    )

    # Hyperparametres TQC SOTA par defaut
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
        "top_quantiles_to_drop_per_net": 2,  # Reduit l'optimisme (drop 2/25)
        "policy_kwargs": policy_kwargs,
    }

    # Merge avec hyperparams custom
    if hyperparams is not None:
        default_params.update(hyperparams)

    # Extraire policy separement (pas un kwarg de TQC)
    policy = default_params.pop("policy")

    # Creer l'agent TQC
    agent = TQC(
        policy,
        env,
        verbose=1,
        seed=SEED,
        device=DEVICE,
        tensorboard_log=tensorboard_log,
        **default_params
    )

    return agent


# Alias pour retrocompatibilite
def create_agent(env, hyperparams=None, tensorboard_log=None):
    """Alias pour create_tqc_agent (retrocompatibilite)."""
    return create_tqc_agent(env, hyperparams, tensorboard_log)

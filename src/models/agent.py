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
    # Policy kwargs avec Transformer extractor (Architecture Nano pour stabilité)
    policy_kwargs = dict(
        features_extractor_class=TransformerFeatureExtractor,
        features_extractor_kwargs=dict(
            features_dim=256,
            d_model=32,       # Réduit de 64 (moins de variance)
            nhead=2,          # Réduit de 4
            num_layers=1,     # Réduit de 2
            dim_feedforward=64,  # Réduit de 128
            dropout=0.1       # Réduit de 0.2
        ),
        net_arch=[256, 256],
        n_critics=2,
        n_quantiles=25,
        optimizer_kwargs=dict(weight_decay=1e-3),  # Régularisation L2 forte
    )

    # Hyperparametres TQC ULTRA-CONSERVATEURS (anti-explosion)
    default_params = {
        "policy": "MlpPolicy",
        "learning_rate": 5e-5,    # Ultra-lent (0.00005) pour stabilité
        "buffer_size": 50_000,    # Plus petit pour recycler plus vite
        "batch_size": 128,        # Plus petit = plus de bruit régularisateur
        "ent_coef": "auto",
        "gamma": 0.99,
        "tau": 0.005,
        "train_freq": 1,
        "gradient_steps": 1,
        "top_quantiles_to_drop_per_net": 2,  # Reduit l'optimisme (drop 2/25)
        "use_sde": True,          # State Dependent Exploration (stabilité)
        "use_sde_at_warmup": True,
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

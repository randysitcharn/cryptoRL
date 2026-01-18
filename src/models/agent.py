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

from typing import Callable

from sb3_contrib import TQC
from torch.optim import AdamW

from src.config import DEVICE, SEED
from src.models.transformer_policy import TransformerFeatureExtractor


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    Decroit lineairement le LR de initial_value a 0 au cours du training.

    Args:
        initial_value: Valeur initiale du learning rate.

    Returns:
        Callable qui prend progress_remaining (1.0 -> 0.0) et retourne le LR.
    """
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


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
        # AdamW avec régularisation propre (SOTA)
        optimizer_class=AdamW,
        optimizer_kwargs=dict(
            weight_decay=1e-4,  # Régularisation L2 propre
            eps=1e-5            # Stabilité numérique
        ),
        log_std_init=-2,      # gSDE: exploration modérée au départ
    )

    # Hyperparametres TQC STABILISATION ACTOR (anti-explosion entropie)
    default_params = {
        "policy": "MlpPolicy",
        "learning_rate": linear_schedule(5e-5),  # Scheduler linéaire décroissant
        "buffer_size": 50_000,    # Plus petit pour recycler plus vite
        "batch_size": 512,        # Increased for better GPU utilization
        "ent_coef": 0.05,         # FIXE (auto fait exploser le Transformer)
        "gamma": 0.99,
        "tau": 0.005,
        "train_freq": 4,          # 1 update tous les 4 steps (stabilise gradients)
        "gradient_steps": 4,      # 4 updates d'un coup
        "top_quantiles_to_drop_per_net": 2,  # Reduit l'optimisme (drop 2/25)
        "use_sde": True,          # gSDE: State Dependent Exploration (SOTA)
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

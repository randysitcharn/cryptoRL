# -*- coding: utf-8 -*-
"""
regret_dsr.py - Regret-based Differential Sharpe Reward (DSR sur Alpha).

Calcule le DSR (Moody & Saffell) sur l'Alpha (agent return - benchmark return).
Rend l'inaction coûteuse en marché directionnel en comparant à un benchmark actif
(Buy & Hold) au lieu d'un état fixe.
"""

import torch
from torch import Tensor
from typing import Tuple


class RegretDSR:
    """
    Calcule le Differential Sharpe Reward (DSR) sur l'Alpha (Agent - Benchmark).
    Rend l'inaction coûteuse en marché directionnel.
    """

    def __init__(
        self,
        n_envs: int,
        device: torch.device,
        eta: float = 0.02,
        scale: float = 10.0,
        warmup_steps: int = 200,
        clip_val: float = 5.0,
    ):
        self.device = device
        self.eta = eta
        self.scale = scale
        self.warmup_steps = warmup_steps
        self.clip_val = clip_val

        # Moments EMA sur l'Alpha
        self.A = torch.zeros(n_envs, device=device)
        self.B = torch.full((n_envs,), 1e-6, device=device)

    def update_and_reward(
        self,
        step_returns: Tensor,
        benchmark_returns: Tensor,
        episode_lengths: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Calcule le reward DSR sur l'Alpha et met à jour les moments EMA.

        Args:
            step_returns: Rendements de l'agent (new_nav - old_nav) / old_nav. Shape: (n_envs,)
            benchmark_returns: Rendements du benchmark B&H (prix). Shape: (n_envs,)
            episode_lengths: Longueur courante par épisode. Shape: (n_envs,)

        Returns:
            (r_perf, A_new, B_new) pour logging / persistance.
        """
        # 1. Alpha = surperformance relative
        alpha = step_returns - benchmark_returns

        # 2. Deltas par rapport aux moments historiques (avant update)
        delta_A = alpha - self.A
        delta_B = (alpha ** 2) - self.B

        # 3. DSR différentiel (Moody & Saffell)
        variance = torch.clamp(self.B - (self.A ** 2), min=1e-8)
        dsr_val = (self.B * delta_A - 0.5 * self.A * delta_B) / (variance ** 1.5)
        r_dsr = torch.clamp(dsr_val * self.scale, -self.clip_val, self.clip_val)

        # 4. Warmup : log-alpha pour guider l'exploration initiale
        warmup_mask = episode_lengths < self.warmup_steps
        r_simple = torch.log1p(torch.clamp(alpha, min=-0.9)) * self.scale
        r_perf = torch.where(warmup_mask, r_simple, r_dsr)

        # 5. Update EMA (même sémantique que batch_env : add_(delta, alpha=eta))
        self.A.add_(delta_A, alpha=self.eta)
        self.B.add_(delta_B, alpha=self.eta)

        return r_perf, self.A.clone(), self.B.clone()

    def reset(self, env_mask: Tensor) -> None:
        """
        Réinitialise les moments pour les envs dont l'épisode vient de terminer.

        Args:
            env_mask: Bool tensor True pour les envs à reset. Shape: (n_envs,)
        """
        if env_mask.any():
            self.A[env_mask] = 0.0
            self.B[env_mask] = 1e-6

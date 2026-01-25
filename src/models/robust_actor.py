# -*- coding: utf-8 -*-
"""
robust_actor.py - RobustDropoutActor fixant le Policy Collapse (gradients à 0).

Hérite de BasePolicy uniquement (pas d'Actor). Ordre d'init : BasePolicy → trunk
latent_pi (create_mlp_with_dropout, output_dim=-1) → distribution (gSDE ou
SquashedGaussian). Net_arch complet, pas de slice. Compatible sauvegarde/chargement.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

import torch
import torch.nn as nn
from gymnasium import spaces

from stable_baselines3.common.distributions import (
    SquashedDiagGaussianDistribution,
    StateDependentNoiseDistribution,
)
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.type_aliases import PyTorchObs

from src.models.tqc_dropout_policy import create_mlp_with_dropout


class RobustDropoutActor(BasePolicy):
    """
    Actor standalone (BasePolicy uniquement) avec Dropout/LayerNorm.

    Corrige le Policy Collapse en construisant le trunk **avant** la distribution
    gSDE, sans appeler Actor.__init__. Net_arch complet, pas de slice.
    """

    LOG_STD_MIN: float = -20.0
    LOG_STD_MAX: float = 2.0
    action_space: spaces.Box

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3.0,
        full_std: bool = True,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        normalize_images: bool = True,
        dropout_rate: float = 0.0,
        use_layer_norm: bool = False,
        use_spectral_norm: bool = False,
    ) -> None:
        BasePolicy.__init__(
            self,
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=True,
        )

        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn
        self.log_std_init = log_std_init
        self.use_sde = use_sde
        self.full_std = full_std
        self.use_expln = use_expln
        self.clip_mean = clip_mean
        self._dropout_rate = dropout_rate
        self._use_layer_norm = use_layer_norm
        self._use_spectral_norm = use_spectral_norm

        action_dim = get_action_dim(self.action_space)
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else features_dim

        self.latent_pi = create_mlp_with_dropout(
            input_dim=features_dim,
            output_dim=-1,
            net_arch=net_arch,
            activation_fn=activation_fn,
            dropout_rate=dropout_rate,
            use_layer_norm=use_layer_norm,
            use_spectral_norm=use_spectral_norm,
            squash_output=False,
        )

        if self.use_sde:
            action_dist = StateDependentNoiseDistribution(
                action_dim,
                full_std=full_std,
                use_expln=use_expln,
                learn_features=True,
                squash_output=True,
            )
            self.mu, self.log_std = action_dist.proba_distribution_net(
                latent_dim=last_layer_dim,
                latent_sde_dim=last_layer_dim,
                log_std_init=log_std_init,
            )
            if clip_mean > 0.0:
                self.mu = nn.Sequential(
                    self.mu,
                    nn.Hardtanh(min_val=-clip_mean, max_val=clip_mean),
                )
            self.action_dist = action_dist
        else:
            action_dist = SquashedDiagGaussianDistribution(action_dim)
            self.mu = nn.Linear(last_layer_dim, action_dim)
            self.log_std = nn.Linear(last_layer_dim, action_dim)
            self.action_dist = action_dist

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                use_sde=self.use_sde,
                log_std_init=self.log_std_init,
                full_std=self.full_std,
                use_expln=self.use_expln,
                clip_mean=self.clip_mean,
                features_extractor=self.features_extractor,
                dropout_rate=self._dropout_rate,
                use_layer_norm=self._use_layer_norm,
                use_spectral_norm=self._use_spectral_norm,
            )
        )
        return data

    def get_action_dist_params(
        self, obs: PyTorchObs
    ) -> tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        features = self.extract_features(obs, self.features_extractor)
        latent_pi = self.latent_pi(features)
        mean_actions = self.mu(latent_pi)

        if self.use_sde:
            return mean_actions, self.log_std, dict(latent_sde=latent_pi)

        log_std = self.log_std(latent_pi)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean_actions, log_std, {}

    def forward(self, obs: PyTorchObs, deterministic: bool = False) -> torch.Tensor:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        return self.action_dist.actions_from_params(
            mean_actions, log_std, deterministic=deterministic, **kwargs
        )

    def get_std(self) -> torch.Tensor:
        msg = "get_std() is only available when using gSDE"
        assert isinstance(
            self.action_dist, StateDependentNoiseDistribution
        ), msg
        return self.action_dist.get_std(self.log_std)

    def reset_noise(self, batch_size: int = 1) -> None:
        msg = "reset_noise() is only available when using gSDE"
        assert isinstance(
            self.action_dist, StateDependentNoiseDistribution
        ), msg
        self.action_dist.sample_weights(self.log_std, batch_size=batch_size)

    def action_log_prob(self, obs: PyTorchObs) -> tuple[torch.Tensor, torch.Tensor]:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        return self.action_dist.log_prob_from_params(
            mean_actions, log_std, **kwargs
        )

    def _predict(
        self, observation: PyTorchObs, deterministic: bool = False
    ) -> torch.Tensor:
        return self(observation, deterministic)

    def set_training_mode(self, mode: bool) -> None:
        """Set training mode (affects dropout)."""
        self.train(mode)

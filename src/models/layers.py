# -*- coding: utf-8 -*-
"""
layers.py - Custom layers for RL models (FiLM, etc.).

FiLM (Feature-wise Linear Modulation) forces the network to use context
(e.g. HMM regime) to scale and shift features, addressing feature collapse
where the model ignores regime information in concatenation-based fusion.
"""

import torch
import torch.nn as nn


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation: out = (features * gamma) + beta.

    Context (e.g. HMM probs + entropy) is mapped via a light MLP to
    scale (gamma) and shift (beta), then applied to features. This
    forces the model to use the regime to modulate MAE embeddings.

    Refs: Perez et al. (2018) - FiLM: Visual Reasoning with Compositionality.
    """

    def __init__(self, feature_dim: int, context_dim: int, hidden_dim: int = 64):
        """
        Args:
            feature_dim: Size of each feature vector (e.g. d_model = 128).
            context_dim: Size of context vector (e.g. 5 for HMM_Prob_* + HMM_Entropy).
            hidden_dim: Hidden size of the MLP (default 64).
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.context_dim = context_dim
        self.mlp = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2 * feature_dim),
        )
        # Initialize last linear so that initially gamma ≈ 1, beta ≈ 0 (identity)
        nn.init.zeros_(self.mlp[-1].bias)
        nn.init.zeros_(self.mlp[-1].weight)
        with torch.no_grad():
            self.mlp[-1].bias[:feature_dim].fill_(1.0)  # gamma init 1

    def forward(
        self,
        features: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply FiLM modulation.

        Args:
            features: (B, ..., feature_dim) e.g. (B, 64, 128) MAE embeddings.
            context: (B, context_dim) e.g. (B, 5) HMM context.

        Returns:
            (B, ..., feature_dim) modulated features.
        """
        params = self.mlp(context)  # (B, 2 * feature_dim)
        gamma, beta = params.chunk(2, dim=-1)  # each (B, feature_dim)
        # Broadcast over sequence/time: (B, 1, feature_dim)
        for _ in range(features.dim() - 2):
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
        out = (features * gamma) + beta
        return out

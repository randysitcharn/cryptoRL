# -*- coding: utf-8 -*-
"""
clipped_optimizer.py - Optimizers with built-in gradient clipping.

Provides ClippedAdam which clips gradients BEFORE applying updates.
This is the correct way to prevent gradient explosion in TQC/SAC.

Usage:
    policy_kwargs = dict(
        optimizer_class=ClippedAdam,
        optimizer_kwargs=dict(max_grad_norm=0.5),
    )
"""

import torch
from torch.optim import Adam, AdamW


class ClippedAdam(Adam):
    """
    Adam optimizer with gradient clipping built-in.

    Clips gradients BEFORE optimizer.step() to prevent explosion.
    Uses global norm across all parameters (not per-parameter).

    Args:
        params: Model parameters to optimize.
        max_grad_norm: Maximum gradient norm (default: 0.5).
        **kwargs: Additional Adam arguments (lr, betas, eps, weight_decay).
    """

    def __init__(self, params, max_grad_norm: float = 0.5, **kwargs):
        self.max_grad_norm = max_grad_norm
        super().__init__(params, **kwargs)

    def step(self, closure=None):
        """
        Performs a single optimization step with gradient clipping.

        1. Collect all parameters from all param_groups
        2. Clip gradients using global norm
        3. Apply Adam update
        """
        # Collect all parameters with gradients
        all_params = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    all_params.append(p)

        # Clip gradients (global norm across all params)
        if all_params:
            torch.nn.utils.clip_grad_norm_(all_params, self.max_grad_norm)

        # Apply Adam update
        return super().step(closure)


class ClippedAdamW(AdamW):
    """
    AdamW optimizer with gradient clipping built-in.

    Same as ClippedAdam but uses decoupled weight decay (AdamW).

    Args:
        params: Model parameters to optimize.
        max_grad_norm: Maximum gradient norm (default: 0.5).
        **kwargs: Additional AdamW arguments (lr, betas, eps, weight_decay).
    """

    def __init__(self, params, max_grad_norm: float = 0.5, **kwargs):
        self.max_grad_norm = max_grad_norm
        super().__init__(params, **kwargs)

    def step(self, closure=None):
        """
        Performs a single optimization step with gradient clipping.
        """
        # Collect all parameters with gradients
        all_params = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    all_params.append(p)

        # Clip gradients (global norm across all params)
        if all_params:
            torch.nn.utils.clip_grad_norm_(all_params, self.max_grad_norm)

        # Apply AdamW update
        return super().step(closure)

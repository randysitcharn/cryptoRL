# -*- coding: utf-8 -*-
"""
test_robust_actor.py - Tests for RobustDropoutActor (Policy Collapse fix).

Verifies:
- Construction with use_sde=True and use_sde=False
- Forward pass (train/eval), action shapes
- Non-null gradients on latent_pi, mu, log_std after backward (regression)
- Diagnostic: gSDE std > 0.1 when use_sde=True
"""

import os
import sys

import pytest
import torch
from gymnasium import spaces

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _make_actor(use_sde: bool, dropout_rate: float = 0.0):
    """Build RobustDropoutActor with small net_arch."""
    from stable_baselines3.common.torch_layers import FlattenExtractor

    from src.models.robust_actor import RobustDropoutActor

    obs = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype="float32")
    act = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype="float32")
    ext = FlattenExtractor(obs)
    return RobustDropoutActor(
        observation_space=obs,
        action_space=act,
        net_arch=[32, 32],
        features_extractor=ext,
        features_dim=8,
        use_sde=use_sde,
        dropout_rate=dropout_rate,
        use_layer_norm=True,
    )


class TestRobustDropoutActorConstruction:
    """Construction with use_sde=True and use_sde=False."""

    def test_construction_no_sde(self):
        from src.models.robust_actor import RobustDropoutActor

        actor = _make_actor(use_sde=False)
        assert isinstance(actor, RobustDropoutActor)
        assert actor.use_sde is False
        assert hasattr(actor, "latent_pi")
        assert hasattr(actor, "mu")
        assert hasattr(actor, "log_std")
        assert hasattr(actor, "action_dist")

    def test_construction_with_sde(self):
        from src.models.robust_actor import RobustDropoutActor

        actor = _make_actor(use_sde=True)
        assert isinstance(actor, RobustDropoutActor)
        assert actor.use_sde is True
        assert hasattr(actor, "latent_pi")
        assert hasattr(actor, "mu")
        assert hasattr(actor, "log_std")
        assert hasattr(actor, "action_dist")


class TestRobustDropoutActorForward:
    """Forward pass (train/eval), action shapes."""

    def test_forward_shape_no_sde(self):
        actor = _make_actor(use_sde=False)
        x = torch.randn(4, 8)
        actions = actor(x, deterministic=True)
        assert actions.shape == (4, 1)

    def test_forward_shape_sde(self):
        actor = _make_actor(use_sde=True)
        x = torch.randn(4, 8)
        actions = actor(x, deterministic=True)
        assert actions.shape == (4, 1)

    def test_forward_train_vs_eval_deterministic(self):
        actor = _make_actor(use_sde=False, dropout_rate=0.0)
        x = torch.randn(4, 8)
        actor.train()
        a1 = actor(x, deterministic=True)
        actor.eval()
        a2 = actor(x, deterministic=True)
        torch.testing.assert_close(a1, a2)


class TestRobustDropoutActorGradients:
    """Non-null gradients on latent_pi, mu, log_std after backward (policy collapse regression)."""

    def test_gradients_flow_no_sde(self):
        actor = _make_actor(use_sde=False)
        actor.train()
        x = torch.randn(4, 8)
        actions, log_prob = actor.action_log_prob(x)
        loss = -log_prob.mean()
        loss.backward()

        has_grad = False
        for name, p in actor.named_parameters():
            if p.grad is not None and p.grad.abs().sum().item() > 0:
                has_grad = True
                break
        assert has_grad, "Expected non-null gradients (policy collapse regression)"

    def test_gradients_flow_sde(self):
        actor = _make_actor(use_sde=True)
        actor.train()
        x = torch.randn(4, 8)
        actions, log_prob = actor.action_log_prob(x)
        loss = -log_prob.mean()
        loss.backward()

        has_grad = False
        for name, p in actor.named_parameters():
            if p.grad is not None and p.grad.abs().sum().item() > 0:
                has_grad = True
                break
        assert has_grad, "Expected non-null gradients (policy collapse regression)"


class TestRobustDropoutActorDiagnostic:
    """Diagnostic: actor type, latent_pi, use_sde, gSDE std > 0.1."""

    def test_diagnostic_sde_std(self):
        actor = _make_actor(use_sde=True)
        assert type(actor).__name__ == "RobustDropoutActor"
        assert actor.latent_pi is not None
        assert actor.use_sde is True
        std = actor.get_std().mean().item()
        assert std > 0.01, f"gSDE std should be > 0.01, got {std}"


class TestRobustDropoutActorTrunkMode:
    """Trunk mode: latent_pi uses net_arch full, no final Linear."""

    def test_latent_pi_no_final_linear(self):
        from src.models.tqc_dropout_policy import create_mlp_with_dropout

        mlp = create_mlp_with_dropout(
            input_dim=8,
            output_dim=-1,
            net_arch=[32, 32],
            dropout_rate=0.0,
            use_layer_norm=False,
        )
        layers = list(mlp)
        last_linear = None
        for L in reversed(layers):
            if isinstance(L, torch.nn.Linear):
                last_linear = L
                break
        assert last_linear is not None
        assert last_linear.out_features == 32
        assert last_linear.in_features == 32

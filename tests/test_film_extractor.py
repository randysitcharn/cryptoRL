# -*- coding: utf-8 -*-
"""
test_film_extractor.py - Dry-run test for FoundationFeatureExtractor + FiLM.

Verifies:
- FiLMLayer and FoundationFeatureExtractor instantiate and forward correctly.
- Output shape (B, features_dim), no NaNs/Infs.
- FiLM modulates output when HMM context changes.
"""

import numpy as np
import torch
from gymnasium import spaces

from src.models.layers import FiLMLayer
from src.models.rl_adapter import FoundationFeatureExtractor, HMM_CONTEXT_SIZE


def test_film_layer_shapes() -> None:
    """FiLMLayer forward: correct shapes, no NaNs."""
    B, T, D = 4, 64, 128
    C = 5
    film = FiLMLayer(feature_dim=D, context_dim=C, hidden_dim=64)
    features = torch.randn(B, T, D)
    context = torch.randn(B, C)
    out = film(features, context)
    assert out.shape == (B, T, D)
    assert not torch.isnan(out).any() and not torch.isinf(out).any()


def test_film_layer_identity_init() -> None:
    """FiLM with identity-like init: output close to input when context zero."""
    B, T, D = 2, 8, 16
    film = FiLMLayer(feature_dim=D, context_dim=5, hidden_dim=32)
    # Zero context -> gamma≈1, beta≈0 (from our init)
    features = torch.randn(B, T, D)
    context = torch.zeros(B, 5)
    out = film(features, context)
    # Allow small deviation (init sets last-layer bias for gamma=1, beta=0)
    diff = (out - features).abs().max().item()
    assert diff < 2.0, f"Identity init expect small diff, got {diff}"


def test_foundation_feature_extractor_film_dry_run() -> None:
    """FoundationFeatureExtractor + FiLM: forward (B, 64, 43), output (B, 512)."""
    seq_len = 64
    n_features = 43
    batch_size = 8
    features_dim = 512

    obs_space = spaces.Dict({
        "market": spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(seq_len, n_features),
            dtype=np.float32,
        ),
        "position": spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        "w_cost": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
    })

    extractor = FoundationFeatureExtractor(
        observation_space=obs_space,
        encoder_path="weights/pretrained_encoder.pth",
        d_model=128,
        freeze_encoder=True,
        features_dim=features_dim,
    )

    assert getattr(extractor, "use_film", False)
    assert extractor.film_layer is not None

    dummy_obs = {
        "market": torch.randn(batch_size, seq_len, n_features, dtype=torch.float32),
        "position": torch.randn(batch_size, 1, dtype=torch.float32).clamp(-1.0, 1.0),
        "w_cost": torch.rand(batch_size, 1, dtype=torch.float32),
    }
    out = extractor(dummy_obs)

    assert out.shape == (batch_size, features_dim)
    assert not torch.isnan(out).any() and not torch.isinf(out).any()


def test_foundation_feature_extractor_film_sensitivity() -> None:
    """Changing HMM context (last 5 cols) changes FiLM-modulated output."""
    seq_len = 64
    n_features = 43
    batch_size = 4
    features_dim = 512

    obs_space = spaces.Dict({
        "market": spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(seq_len, n_features),
            dtype=np.float32,
        ),
        "position": spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        "w_cost": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
    })

    extractor = FoundationFeatureExtractor(
        observation_space=obs_space,
        encoder_path="weights/pretrained_encoder.pth",
        d_model=128,
        freeze_encoder=True,
        features_dim=features_dim,
    )

    base_obs = {
        "market": torch.randn(batch_size, seq_len, n_features, dtype=torch.float32),
        "position": torch.randn(batch_size, 1, dtype=torch.float32).clamp(-1.0, 1.0),
        "w_cost": torch.rand(batch_size, 1, dtype=torch.float32),
    }
    out1 = extractor(base_obs)

    pert_obs = {
        "market": base_obs["market"].clone(),
        "position": base_obs["position"].clone(),
        "w_cost": base_obs["w_cost"].clone(),
    }
    pert_obs["market"][:, -1, -HMM_CONTEXT_SIZE:] += 1.0
    out2 = extractor(pert_obs)

    diff = (out2 - out1).abs().mean().item()
    assert diff > 1e-5, f"FiLM should change output when HMM context changes (diff={diff})"


def test_foundation_feature_extractor_no_film_when_few_features() -> None:
    """When n_features < 5, FiLM is disabled (no HMM context)."""
    seq_len = 64
    n_features = 4  # < HMM_CONTEXT_SIZE
    batch_size = 4
    features_dim = 512

    obs_space = spaces.Dict({
        "market": spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(seq_len, n_features),
            dtype=np.float32,
        ),
        "position": spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        "w_cost": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
    })

    extractor = FoundationFeatureExtractor(
        observation_space=obs_space,
        encoder_path="weights/pretrained_encoder.pth",
        d_model=128,
        freeze_encoder=True,
        features_dim=features_dim,
    )

    assert not getattr(extractor, "use_film", True)
    assert extractor.film_layer is None

    dummy_obs = {
        "market": torch.randn(batch_size, seq_len, n_features, dtype=torch.float32),
        "position": torch.randn(batch_size, 1, dtype=torch.float32).clamp(-1.0, 1.0),
        "w_cost": torch.rand(batch_size, 1, dtype=torch.float32),
    }
    out = extractor(dummy_obs)
    assert out.shape == (batch_size, features_dim)

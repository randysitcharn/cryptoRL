# -*- coding: utf-8 -*-
"""
test_film_extractor.py - Dry-run test for FoundationFeatureExtractor + FiLM.

Verifies:
1. FoundationFeatureExtractor instantiates with FiLM.
2. Forward pass with (B, Seq, Features) market + (B, 1) position.
3. Output shape (B, features_dim) and no dimension errors in FiLM.

Run: python -m scripts.test_film_extractor
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from gymnasium import spaces

# Ensure project root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.rl_adapter import FoundationFeatureExtractor, HMM_CONTEXT_SIZE


def main() -> None:
    seq_len = 64
    n_features = 43  # Match WFO pipeline (includes HMM_Prob_* + HMM_Entropy)
    batch_size = 8
    features_dim = 512

    obs_space = spaces.Dict({
        "market": spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(seq_len, n_features),
            dtype=np.float32,
        ),
        "position": spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        ),
        "w_cost": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
    })

    print("[1/3] Instantiating FoundationFeatureExtractor (with FiLM)...")
    extractor = FoundationFeatureExtractor(
        observation_space=obs_space,
        encoder_path="weights/pretrained_encoder.pth",
        d_model=128,
        freeze_encoder=True,
        features_dim=features_dim,
    )

    print(f"      use_film={getattr(extractor, 'use_film', False)}")
    print(f"      film_layer is not None: {extractor.film_layer is not None}")

    print("\n[2/3] Forward pass with random (B, Seq, Features)...")
    dummy_obs = {
        "market": torch.randn(batch_size, seq_len, n_features, dtype=torch.float32),
        "position": torch.randn(batch_size, 1, dtype=torch.float32).clamp(-1.0, 1.0),
        "w_cost": torch.rand(batch_size, 1, dtype=torch.float32),
    }
    out = extractor(dummy_obs)

    print(f"      market shape:  {dummy_obs['market'].shape}")
    print(f"      position shape: {dummy_obs['position'].shape}")
    print(f"      output shape:  {out.shape}")

    assert out.shape == (batch_size, features_dim), (
        f"Expected output {(batch_size, features_dim)}, got {out.shape}"
    )
    assert not torch.isnan(out).any() and not torch.isinf(out).any(), (
        "Output contains NaN or Inf"
    )

    print("\n[3/3] FiLM sensitivity check (different HMM context -> different output)...")
    obs2 = {
        "market": dummy_obs["market"].clone(),
        "position": dummy_obs["position"].clone(),
        "w_cost": dummy_obs["w_cost"].clone(),
    }
    # Perturb last 5 columns (HMM context) at last timestep
    obs2["market"][:, -1, -HMM_CONTEXT_SIZE:] += 1.0
    out2 = extractor(obs2)
    diff = (out2 - out).abs().mean().item()
    assert diff > 1e-5, f"FiLM should change output when HMM context changes (diff={diff})"
    print(f"      mean |out2 - out| = {diff:.6f} (FiLM reactive to HMM)")

    print("\n[OK] FoundationFeatureExtractor + FiLM dry-run passed.")
    print("     - Output shape correct, no NaNs/Infs")
    print("     - FiLM modulates output wrt HMM context")


if __name__ == "__main__":
    main()

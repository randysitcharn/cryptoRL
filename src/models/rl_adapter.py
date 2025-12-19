# -*- coding: utf-8 -*-
"""
rl_adapter.py - Foundation Model Feature Extractor for SB3.

Integrates the pre-trained CryptoMAE encoder as a feature extractor
for TQC (Truncated Quantile Critics) agents in Stable Baselines3.

The encoder weights are frozen by default to preserve the learned
market representations during initial RL training.
"""

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Optional

from src.models.foundation import CryptoMAE


class FoundationFeatureExtractor(BaseFeaturesExtractor):
    """
    Feature extractor using a pre-trained CryptoMAE encoder.

    Loads a pre-trained Masked Auto-Encoder and uses its encoder
    to transform observations into rich feature representations
    for the RL policy network.

    Architecture:
    - Input: (batch, seq_len, n_features) from environment
    - Encoder: Pre-trained CryptoMAE encoder
    - Output: Flattened (batch, seq_len * d_model) for policy

    The encoder is frozen by default to prevent "breaking" the
    pre-trained representations during early RL training.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        encoder_path: str = "weights/pretrained_encoder.pth",
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.1,
        freeze_encoder: bool = True,
        features_dim: Optional[int] = None
    ):
        """
        Initialize the Foundation Feature Extractor.

        Args:
            observation_space: Gym observation space (seq_len, n_features).
            encoder_path: Path to pretrained encoder weights.
            d_model: Transformer model dimension (must match pretrained).
            n_heads: Number of attention heads (must match pretrained).
            n_layers: Number of encoder layers (must match pretrained).
            dim_feedforward: FFN dimension (default: 4 * d_model).
            dropout: Dropout rate.
            freeze_encoder: If True, freeze encoder weights.
            features_dim: Output dimension. If None, uses seq_len * d_model.
        """
        # Extract dimensions from observation space
        # Shape: (window_size, n_features)
        self.window_size = observation_space.shape[0]
        self.n_features = observation_space.shape[1]
        self.d_model = d_model

        # Compute features_dim dynamically if not specified
        computed_features_dim = self.window_size * d_model
        if features_dim is None:
            features_dim = computed_features_dim

        # Initialize parent class with features_dim
        super().__init__(observation_space, features_dim)

        # Store config
        self.encoder_path = encoder_path
        self._is_frozen = freeze_encoder

        # Create CryptoMAE encoder with matching architecture
        self.mae = CryptoMAE(
            input_dim=self.n_features,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_len=max(512, self.window_size)
        )

        # Load pretrained weights (encoder only, strict=False)
        self._load_pretrained_weights(encoder_path)

        # Freeze encoder if requested
        if freeze_encoder:
            self._freeze_encoder()

        # Output projection: flatten + optional linear if features_dim differs
        if features_dim == computed_features_dim:
            # Simple flatten, no projection needed
            self.output_projection = nn.Sequential(
                nn.Flatten(),
                nn.LayerNorm(features_dim)
            )
        else:
            # Flatten + linear projection to target features_dim
            self.output_projection = nn.Sequential(
                nn.Flatten(),
                nn.LayerNorm(computed_features_dim),
                nn.Linear(computed_features_dim, features_dim),
                nn.ReLU(),
                nn.LayerNorm(features_dim)
            )

    def _load_pretrained_weights(self, encoder_path: str) -> None:
        """
        Load pretrained encoder weights.

        Supports two formats:
        1. Nested dict: {'embedding': {...}, 'pos_encoder': {...}, 'encoder': {...}, 'd_model': int}
        2. Standard state_dict: Direct model state dict (uses strict=False)

        Args:
            encoder_path: Path to the pretrained weights file.
        """
        try:
            checkpoint = torch.load(encoder_path, map_location="cpu", weights_only=True)

            # Check if checkpoint is in nested format (our custom encoder-only format)
            if isinstance(checkpoint, dict) and 'embedding' in checkpoint and 'encoder' in checkpoint:
                # Nested format: load each component separately
                self.mae.embedding.load_state_dict(checkpoint['embedding'])
                self.mae.encoder.load_state_dict(checkpoint['encoder'])

                # pos_encoder contains buffer 'pe', need to handle specially
                if 'pos_encoder' in checkpoint:
                    pe_data = checkpoint['pos_encoder'].get('pe')
                    if pe_data is not None:
                        self.mae.pos_encoder.pe.copy_(pe_data)

                # Verify d_model matches
                saved_d_model = checkpoint.get('d_model')
                if saved_d_model is not None and saved_d_model != self.d_model:
                    print(f"[FoundationFeatureExtractor] WARNING: d_model mismatch! "
                          f"Saved: {saved_d_model}, Current: {self.d_model}")

                print(f"[FoundationFeatureExtractor] Loaded pretrained encoder (nested format) from {encoder_path}")

            else:
                # Standard state_dict format (full model or partial)
                missing, unexpected = self.mae.load_state_dict(checkpoint, strict=False)

                if missing:
                    print(f"[FoundationFeatureExtractor] Missing keys (expected for encoder-only): {len(missing)}")
                if unexpected:
                    print(f"[FoundationFeatureExtractor] Unexpected keys: {len(unexpected)}")

                print(f"[FoundationFeatureExtractor] Loaded pretrained weights from {encoder_path}")

        except FileNotFoundError:
            print(f"[FoundationFeatureExtractor] WARNING: Pretrained weights not found at {encoder_path}")
            print("[FoundationFeatureExtractor] Using randomly initialized encoder")

    def _freeze_encoder(self) -> None:
        """Freeze all encoder parameters (embedding, pos_encoder, encoder)."""
        # Freeze embedding layer
        for param in self.mae.embedding.parameters():
            param.requires_grad = False

        # Freeze positional encoding (already a buffer, but just in case)
        for param in self.mae.pos_encoder.parameters():
            param.requires_grad = False

        # Freeze transformer encoder
        for param in self.mae.encoder.parameters():
            param.requires_grad = False

        # Freeze mask token (not used in encode, but for consistency)
        self.mae.mask_token.requires_grad = False

        self._is_frozen = True
        print("[FoundationFeatureExtractor] Encoder weights frozen")

    def unfreeze_encoder(self) -> None:
        """
        Unfreeze encoder weights for fine-tuning.

        Call this after initial RL training to enable end-to-end
        fine-tuning of the entire model.
        """
        for param in self.mae.embedding.parameters():
            param.requires_grad = True

        for param in self.mae.pos_encoder.parameters():
            param.requires_grad = True

        for param in self.mae.encoder.parameters():
            param.requires_grad = True

        self.mae.mask_token.requires_grad = True

        self._is_frozen = False
        print("[FoundationFeatureExtractor] Encoder weights unfrozen for fine-tuning")

    @property
    def is_frozen(self) -> bool:
        """Return whether the encoder is frozen."""
        return self._is_frozen

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the pretrained encoder.

        Args:
            observations: Input tensor of shape (batch, seq_len, n_features).

        Returns:
            Feature tensor of shape (batch, features_dim).
        """
        # 1. Encode observations using pretrained MAE encoder
        # Output: (batch, seq_len, d_model)
        encoded = self.mae.encode(observations)

        # 2. Output projection: flatten (+ optional linear)
        # Output: (batch, features_dim)
        features = self.output_projection(encoded)

        return features


if __name__ == "__main__":
    # Test the feature extractor
    from gymnasium import spaces
    import numpy as np

    # Simulate observation space: (64 timesteps, 35 features)
    seq_len = 64
    n_features = 35

    obs_space = spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(seq_len, n_features),
        dtype=np.float32
    )

    # Create feature extractor (will warn if weights not found)
    extractor = FoundationFeatureExtractor(
        observation_space=obs_space,
        encoder_path="weights/pretrained_encoder.pth",
        freeze_encoder=True
    )

    # Test forward pass
    batch_size = 4
    dummy_obs = torch.randn(batch_size, seq_len, n_features)

    features = extractor(dummy_obs)

    print(f"\nInput shape:  {dummy_obs.shape}")
    print(f"Output shape: {features.shape}")
    print(f"features_dim: {extractor.features_dim}")
    print(f"Encoder frozen: {extractor.is_frozen}")

    # Verify output dimension
    expected_dim = seq_len * 128  # 64 * 128 = 8192
    assert features.shape == (batch_size, expected_dim), f"Shape mismatch! Expected {(batch_size, expected_dim)}"
    print("\n[OK] FoundationFeatureExtractor test passed!")

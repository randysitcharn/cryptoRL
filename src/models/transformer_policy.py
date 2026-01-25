# -*- coding: utf-8 -*-
"""
transformer_policy.py - Transformer Feature Extractor for SB3.

Implements a custom Transformer-based feature extractor optimized for
low data regime in financial time series (small datasets).

Architecture:
- Positional Encoding (Sinusoidal)
- Input Projection: Linear(n_features -> d_model)
- Transformer Encoder: 2 layers, 4 heads, d_model=64
- Output: Flatten -> Linear -> features_dim
"""

import math
from typing import Optional
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from src.config.constants import DEFAULT_TRANSFORMER_CONFIG, TransformerFeatureExtractorConfig
from src.config.validators import ModelDimensionsValidator


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.

    Adds positional information to the input embeddings since
    Transformers have no inherent notion of sequence order.

    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        """
        Args:
            d_model (int): Embedding dimension.
            max_len (int): Maximum sequence length.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Shape: (1, max_len, d_model) for broadcasting
        pe = pe.unsqueeze(0)

        # Register as buffer (not a parameter, but saved with model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, d_model).

        Returns:
            torch.Tensor: Input with positional encoding added.
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerFeatureExtractor(BaseFeaturesExtractor):
    """
    Transformer-based feature extractor for SB3 policies.

    Optimized for low data regime in financial time series:
    - Small d_model (64) to prevent overfitting
    - Only 2 encoder layers (keep it simple)
    - High dropout (0.2) for regularization
    - 4 attention heads

    Input: (Batch, Window, Features) from SB3
    Output: (Batch, features_dim) for policy network
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: Optional[int] = None,
        d_model: Optional[int] = None,
        nhead: Optional[int] = None,
        num_layers: Optional[int] = None,
        dim_feedforward: Optional[int] = None,
        dropout: Optional[float] = None
    ):
        """
        Args:
            observation_space (spaces.Box): Observation space from env.
            features_dim (int, optional): Output dimension for policy network.
                Defaults to DEFAULT_TRANSFORMER_CONFIG.features_dim.
            d_model (int, optional): Transformer model dimension.
                Defaults to DEFAULT_TRANSFORMER_CONFIG.d_model.
            nhead (int, optional): Number of attention heads.
                Defaults to DEFAULT_TRANSFORMER_CONFIG.n_heads.
            num_layers (int, optional): Number of encoder layers.
                Defaults to DEFAULT_TRANSFORMER_CONFIG.n_layers.
            dim_feedforward (int, optional): Feedforward network dimension.
                Defaults to DEFAULT_TRANSFORMER_CONFIG.dim_feedforward.
            dropout (float, optional): Dropout rate (regularization).
                Defaults to DEFAULT_TRANSFORMER_CONFIG.dropout.
        """
        # Use default config values if not provided
        config = DEFAULT_TRANSFORMER_CONFIG
        features_dim = features_dim if features_dim is not None else config.features_dim
        d_model = d_model if d_model is not None else config.d_model
        nhead = nhead if nhead is not None else config.n_heads
        num_layers = num_layers if num_layers is not None else config.n_layers
        dim_feedforward = dim_feedforward if dim_feedforward is not None else config.dim_feedforward
        dropout = dropout if dropout is not None else config.dropout
        
        # Create config object for validation
        actual_config = TransformerFeatureExtractorConfig(
            d_model=d_model,
            n_heads=nhead,
            n_layers=num_layers,
            dim_feedforward=dim_feedforward,
            features_dim=features_dim,
            dropout=dropout
        )
        
        # Validate configuration
        ModelDimensionsValidator.validate_transformer_config(
            actual_config,
            n_features=observation_space.shape[1] if len(observation_space.shape) > 1 else None
        )
        
        super().__init__(observation_space, features_dim)

        # Extract dimensions from observation space
        # Shape: (window_size, n_features)
        self.window_size = observation_space.shape[0]
        self.n_features = observation_space.shape[1]
        self.d_model = d_model

        # Input LayerNorm for stability
        self.input_norm = nn.LayerNorm(self.n_features)

        # Input projection: n_features -> d_model
        self.input_projection = nn.Linear(self.n_features, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            d_model=d_model,
            max_len=self.window_size,
            dropout=dropout
        )

        # Transformer encoder (batch_first=True for better performance)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # Native batch-first for efficiency
            norm_first=True    # Pre-LayerNorm: SOTA pour stabiliser gradients RL
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=True  # Optimized for batch_first=True
        )

        # Output LayerNorm for stability before projection
        self.output_norm = nn.LayerNorm(d_model)

        # Output projection: flatten -> linear -> features_dim
        self.output_projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.window_size * d_model, features_dim),
            nn.ReLU(),
            nn.LayerNorm(features_dim)  # Final LayerNorm for stability
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Transformer.

        Args:
            observations (torch.Tensor): Input of shape (Batch, Window, Features).

        Returns:
            torch.Tensor: Output of shape (Batch, features_dim).
        """
        # Input: (Batch, Window, Features)

        # 1. Input LayerNorm for stability
        x = self.input_norm(observations)

        # 2. Project input features to d_model
        # (Batch, Window, Features) -> (Batch, Window, d_model)
        x = self.input_projection(x)

        # 3. Add positional encoding
        x = self.pos_encoder(x)

        # 4. Pass through Transformer encoder (batch_first=True, no permute needed)
        x = self.transformer_encoder(x)

        # 5. Output LayerNorm for stability
        x = self.output_norm(x)

        # 6. Output projection: flatten and project to features_dim
        x = self.output_projection(x)

        return x

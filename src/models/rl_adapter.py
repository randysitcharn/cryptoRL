# -*- coding: utf-8 -*-
"""
rl_adapter.py - Foundation Model Feature Extractor for SB3.

Integrates the pre-trained CryptoMAE encoder as a feature extractor
for TQC (Truncated Quantile Critics) agents in Stable Baselines3.

The encoder weights are frozen by default to preserve the learned
market representations during initial RL training.

Architecture (FiLM + Dict observation space):
- Input: Dict {"market": (B, 64, 43), "position": (B, 1), "w_cost": (B, 1)}
- Market → MAE Encoder (frozen) → (B, 64, 128)
- HMM context = last 5 cols of market (4 probs + entropy) → FiLM modulation
- Modulated embed → Flatten (8192) → Concat(position, w_cost) → Linear → LeakyReLU
- Output: (B, features_dim)

FIX 2026-01-25: Added w_cost to forward() to enable MORL conditioning.
"""

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Optional, Dict

from src.config.constants import (
    DEFAULT_FOUNDATION_CONFIG,
    FoundationFeatureExtractorConfig,
    HMM_CONTEXT_SIZE,
)
from src.config.validators import ModelDimensionsValidator
from src.models.foundation import CryptoMAE
from src.models.layers import FiLMLayer


class FoundationFeatureExtractor(BaseFeaturesExtractor):
    """
    Feature extractor using a pre-trained CryptoMAE encoder with FiLM + position + w_cost.

    Accepts Dict observation space with:
    - "market": (seq_len, n_features) market data (last 5 cols = HMM context)
    - "position": (1,) current position
    - "w_cost": (1,) MORL cost preference parameter

    Architecture (Split Input):
    - Technical features (cols 0 to N-5) → MAE Encoder (frozen) → embeddings (B, 64, 128)
    - HMM context = market[:, -1, -5:] (4 probs + entropy) → FiLM modulation
    - FiLM(embeddings, HMM context) → modulated embeddings
    - Flatten → LayerNorm → Concat(position, w_cost) → Linear → LeakyReLU

    Critical: The MAE receives ONLY technical features (excludes HMM columns) to match
    the pre-trained weights. HMM features are injected via FiLM modulation only.
    The encoder is frozen; only FiLM and fusion layers are trainable.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        encoder_path: str = "weights/pretrained_encoder.pth",
        d_model: Optional[int] = None,
        n_heads: Optional[int] = None,
        n_layers: Optional[int] = None,
        dim_feedforward: Optional[int] = None,
        dropout: Optional[float] = None,
        freeze_encoder: bool = True,
        features_dim: Optional[int] = None,
        config: Optional[FoundationFeatureExtractorConfig] = None
    ):
        """
        Initialize the Foundation Feature Extractor.

        Args:
            observation_space: Dict observation space with "market", "position", and "w_cost".
            encoder_path: Path to pretrained encoder weights.
            d_model: Transformer model dimension (must match pretrained).
                Defaults to config.mae_config.d_model.
            n_heads: Number of attention heads (must match pretrained).
                Defaults to config.mae_config.n_heads.
            n_layers: Number of encoder layers (must match pretrained).
                Defaults to config.mae_config.n_layers.
            dim_feedforward: FFN dimension (default: 4 * d_model).
            dropout: Dropout rate. Defaults to config.mae_config.dropout.
            freeze_encoder: If True, freeze encoder weights.
            features_dim: Output dimension. Defaults to config.features_dim.
            config: FoundationFeatureExtractorConfig to use (uses DEFAULT if None).
        """
        # Use default config if not provided
        if config is None:
            config = DEFAULT_FOUNDATION_CONFIG
        
        # Use config values if not explicitly provided
        d_model = d_model if d_model is not None else config.mae_config.d_model
        n_heads = n_heads if n_heads is not None else config.mae_config.n_heads
        n_layers = n_layers if n_layers is not None else config.mae_config.n_layers
        dropout = dropout if dropout is not None else config.mae_config.dropout
        features_dim = features_dim if features_dim is not None else config.features_dim
        
        # Validate observation space type
        assert isinstance(observation_space, spaces.Dict), \
            f"Expected spaces.Dict, got {type(observation_space)}"
        assert "market" in observation_space.spaces, "Missing 'market' in observation space"
        assert "position" in observation_space.spaces, "Missing 'position' in observation space"
        assert "w_cost" in observation_space.spaces, "Missing 'w_cost' in observation space"

        # Extract dimensions from market observation
        market_space = observation_space["market"]
        self.window_size = market_space.shape[0]  # 64
        self.n_features = market_space.shape[1]   # 43
        self.d_model = d_model
        
        # Create actual config for validation
        from src.config.constants import MAEConfig
        actual_mae_config = MAEConfig(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dim_feedforward=dim_feedforward if dim_feedforward is not None else d_model * 4,
            dropout=dropout
        )
        actual_config = FoundationFeatureExtractorConfig(
            mae_config=actual_mae_config,
            features_dim=features_dim
        )
        
        # Validate configuration
        ModelDimensionsValidator.validate_foundation_config(
            actual_config,
            n_features=self.n_features,
            window_size=self.window_size
        )
        
        # Validate observation space compatibility
        ModelDimensionsValidator.validate_observation_space_compatibility(
            observation_space_n_features=self.n_features,
            expected_n_features=None,  # We don't enforce a specific count, just validate structure
            use_film=self.n_features >= HMM_CONTEXT_SIZE
        )

        # Position dimension
        position_space = observation_space["position"]
        self.position_dim = position_space.shape[0]  # 1

        # w_cost dimension (MORL preference parameter)
        w_cost_space = observation_space["w_cost"]
        self.w_cost_dim = w_cost_space.shape[0]  # 1

        # Flatten dimension from encoder output + position + w_cost
        self.market_flatten_dim = self.window_size * d_model
        self.total_input_dim = self.market_flatten_dim + self.position_dim + self.w_cost_dim

        # FiLM: use HMM context only if we have enough feature columns
        self.use_film = self.n_features >= HMM_CONTEXT_SIZE

        # Adjust MAE input dimension: exclude HMM columns if FiLM is used
        # This ensures the MAE receives only the features it was pre-trained on
        if self.use_film:
            self.mae_input_dim = self.n_features - HMM_CONTEXT_SIZE
            print(f"[FoundationFeatureExtractor] Splitting Input: {self.mae_input_dim} Tech Features -> MAE | {HMM_CONTEXT_SIZE} HMM -> FiLM")
        else:
            self.mae_input_dim = self.n_features

        # Initialize parent class with features_dim
        super().__init__(observation_space, features_dim)

        # Store config
        self.encoder_path = encoder_path
        self._is_frozen = freeze_encoder

        # Create CryptoMAE encoder with CORRECT input dimension (excludes HMM if FiLM is used)
        self.mae = CryptoMAE(
            input_dim=self.mae_input_dim,  # Critical fix: matches pre-trained dimension
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_len=max(512, self.window_size)
        )

        # Load pretrained weights (encoder only, strict=False)
        # Validation happens inside _load_pretrained_weights
        self._load_pretrained_weights(encoder_path, expected_config=actual_config, expected_input_dim=self.mae_input_dim)

        # Freeze encoder if requested
        if freeze_encoder:
            self._freeze_encoder()

        # Convert frozen encoder to float16 for faster inference (CUDA optimization)
        self._use_amp = freeze_encoder and torch.cuda.is_available()
        if self._use_amp:
            self.mae.embedding = self.mae.embedding.half()
            self.mae.encoder = self.mae.encoder.half()
            print("[FoundationFeatureExtractor] Encoder converted to float16 for faster inference")

        # FiLM: context (HMM) modulates MAE embeddings before flatten
        # Only instantiate when we have enough market columns (HMM_Prob_* + HMM_Entropy)
        self.film_layer: Optional[nn.Module] = None
        if self.use_film:
            self.film_layer = FiLMLayer(
                feature_dim=d_model,
                context_dim=HMM_CONTEXT_SIZE,
                hidden_dim=64,
            )
            print(f"[FoundationFeatureExtractor] FiLM: context_dim={HMM_CONTEXT_SIZE}, feature_dim={d_model} (HMM modulates MAE)")

        # Fusion Projector: market features + position + w_cost → features_dim
        # Architecture: Concat(market_flat, position, w_cost) → Linear → LayerNorm → LeakyReLU
        # NO TANH to avoid vanishing gradient with TQC's internal Tanh
        self.market_flatten = nn.Flatten()  # (B, 64, 128) → (B, 8192)
        self.market_layernorm = nn.LayerNorm(self.market_flatten_dim)

        self.fusion_projection = nn.Sequential(
            nn.Linear(self.total_input_dim, features_dim),
            nn.LayerNorm(features_dim),
            nn.LeakyReLU(negative_slope=0.01)  # LeakyReLU instead of Tanh
        )
        print(f"[FoundationFeatureExtractor] Fusion Projector: {self.total_input_dim} → {features_dim} (LeakyReLU, position+w_cost aware)")

    def _load_pretrained_weights(
        self,
        encoder_path: str,
        expected_config: Optional[FoundationFeatureExtractorConfig] = None,
        expected_input_dim: Optional[int] = None
    ) -> None:
        """
        Load pretrained encoder weights.

        Supports two formats:
        1. Nested dict: {'embedding': {...}, 'pos_encoder': {...}, 'encoder': {...}, 'd_model': int}
        2. Standard state_dict: Direct model state dict (uses strict=False)

        Args:
            encoder_path: Path to the pretrained weights file.
            expected_config: Expected FoundationFeatureExtractorConfig for validation.
            expected_input_dim: Expected input dimension for MAE (excludes HMM if FiLM used).
        """
        try:
            checkpoint = torch.load(encoder_path, map_location="cpu", weights_only=True)
            
            # Validate checkpoint compatibility using centralized validator
            if expected_config is not None:
                try:
                    ModelDimensionsValidator.validate_checkpoint_compatibility(
                        checkpoint,
                        expected_config=expected_config,
                        expected_input_dim=expected_input_dim
                    )
                except ValueError as e:
                    # Re-raise with clear context
                    raise ValueError(
                        f"[FoundationFeatureExtractor] Checkpoint validation failed:\n{e}"
                    ) from e

            # Check if checkpoint is in nested format (our custom encoder-only format)
            if isinstance(checkpoint, dict) and 'embedding' in checkpoint and 'encoder' in checkpoint:
                # CRITICAL: Check input_dim dimension mismatch (embedding layer)
                # This detects if checkpoint was trained with different number of features
                checkpoint_embedding = checkpoint['embedding']
                model_input_dim = self.mae.embedding.weight.shape[1]
                
                if 'weight' in checkpoint_embedding:
                    checkpoint_input_dim = checkpoint_embedding['weight'].shape[1]
                    
                    if checkpoint_input_dim != model_input_dim:
                        raise ValueError(
                            f"[FoundationFeatureExtractor] CRITICAL: Input dimension mismatch!\n"
                            f"  Checkpoint was trained with {checkpoint_input_dim} features\n"
                            f"  Current model expects {model_input_dim} features\n"
                            f"  This usually means:\n"
                            f"    - Old checkpoint (pre-trained with HMM features included)\n"
                            f"    - OR new checkpoint (pre-trained without HMM features)\n"
                            f"  Solution: Re-pre-train the MAE with the current feature set\n"
                            f"  (dataset.py now auto-excludes HMM features for MAE pre-training)\n"
                            f"  Expected config: d_model={expected_config.mae_config.d_model if expected_config else 'N/A'}, "
                            f"n_heads={expected_config.mae_config.n_heads if expected_config else 'N/A'}, "
                            f"n_layers={expected_config.mae_config.n_layers if expected_config else 'N/A'}"
                        )
                
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
                if 'weight' in checkpoint_embedding:
                    print(f"  ✓ Input dimension verified: {model_input_dim} features (matches checkpoint)")

            else:
                # Standard state_dict format (full model or partial)
                # Check input_dim dimension mismatch before loading
                model_input_dim = self.mae.embedding.weight.shape[1]
                
                if 'embedding.weight' in checkpoint:
                    checkpoint_input_dim = checkpoint['embedding.weight'].shape[1]
                    
                    if checkpoint_input_dim != model_input_dim:
                        raise ValueError(
                            f"[FoundationFeatureExtractor] CRITICAL: Input dimension mismatch!\n"
                            f"  Checkpoint was trained with {checkpoint_input_dim} features\n"
                            f"  Current model expects {model_input_dim} features\n"
                            f"  This usually means:\n"
                            f"    - Old checkpoint (pre-trained with HMM features included)\n"
                            f"    - OR new checkpoint (pre-trained without HMM features)\n"
                            f"  Solution: Re-pre-train the MAE with the current feature set\n"
                            f"  (dataset.py now auto-excludes HMM features for MAE pre-training)"
                        )
                
                missing, unexpected = self.mae.load_state_dict(checkpoint, strict=False)

                if missing:
                    print(f"[FoundationFeatureExtractor] Missing keys (expected for encoder-only): {len(missing)}")
                if unexpected:
                    print(f"[FoundationFeatureExtractor] Unexpected keys: {len(unexpected)}")

                print(f"[FoundationFeatureExtractor] Loaded pretrained weights from {encoder_path}")
                if 'embedding.weight' in checkpoint:
                    print(f"  ✓ Input dimension verified: {model_input_dim} features (matches checkpoint)")

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

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the pretrained encoder with position and w_cost fusion.

        Args:
            observations: Dict with:
                - "market": (batch, seq_len, n_features) market data
                  - If use_film=True: cols 0 to N-5 are technical features (→ MAE),
                    cols N-5 to N are HMM context (→ FiLM)
                  - If use_film=False: all cols go to MAE
                - "position": (batch, 1) current position
                - "w_cost": (batch, 1) MORL cost preference parameter

        Returns:
            Feature tensor of shape (batch, features_dim).
        """
        # Extract market data, position, and w_cost from dict
        market_obs = observations["market"]    # (B, 64, n_features)
        position = observations["position"]    # (B, 1)
        w_cost = observations["w_cost"]        # (B, 1) - MORL preference parameter

        # 1. SPLIT INPUT: Tech features -> MAE, HMM features -> FiLM
        if self.use_film:
            # Split: Technical features (0 to N-5) go to MAE, HMM context (N-5 to N) goes to FiLM
            mae_obs = market_obs[:, :, :-HMM_CONTEXT_SIZE]  # (B, 64, mae_input_dim)
            hmm_context = market_obs[:, -1, -HMM_CONTEXT_SIZE:].float()  # (B, 5)
        else:
            # No FiLM, use full observation for MAE
            mae_obs = market_obs
            hmm_context = None

        # 2. Encode technical features using pretrained MAE encoder (frozen)
        # Output: (batch, seq_len, d_model) = (B, 64, 128)
        encoder_dtype = next(self.mae.embedding.parameters()).dtype
        if encoder_dtype == torch.float16:
            if mae_obs.is_cuda:
                with torch.amp.autocast('cuda'):
                    encoded = self.mae.encode(mae_obs.half())
                encoded = encoded.float()
            else:
                self.mae.embedding = self.mae.embedding.float()
                self.mae.encoder = self.mae.encoder.float()
                encoded = self.mae.encode(mae_obs.float())
        else:
            encoded = self.mae.encode(mae_obs.float())

        # 3. FiLM modulation: HMM context (last 5 cols at last timestep) scales/shifts MAE embeddings
        if self.film_layer is not None:
            modulated = self.film_layer(encoded, hmm_context)  # (B, 64, 128)
        else:
            modulated = encoded

        # 4. Flatten and normalize
        market_flat = self.market_flatten(modulated)  # (B, 8192)
        market_flat = self.market_layernorm(market_flat)

        # 5. Concat with position AND w_cost, then project
        combined = torch.cat([market_flat, position, w_cost], dim=1)  # (B, 8194)
        features = self.fusion_projection(combined)  # (B, 512)

        return features


if __name__ == "__main__":
    # Test the feature extractor with Dict observation space
    from gymnasium import spaces
    import numpy as np

    # Simulate Dict observation space
    seq_len = 64
    n_features = 35

    obs_space = spaces.Dict({
        "market": spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(seq_len, n_features),
            dtype=np.float32
        ),
        "position": spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        ),
        "w_cost": spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )
    })

    # Create feature extractor (will warn if weights not found)
    extractor = FoundationFeatureExtractor(
        observation_space=obs_space,
        encoder_path="weights/pretrained_encoder.pth",
        freeze_encoder=True
    )

    # Test forward pass with dict observation
    batch_size = 4
    dummy_obs = {
        "market": torch.randn(batch_size, seq_len, n_features),
        "position": torch.randn(batch_size, 1).clamp(-1, 1),
        "w_cost": torch.rand(batch_size, 1)  # w_cost in [0, 1]
    }

    features = extractor(dummy_obs)

    print(f"\nMarket input shape:  {dummy_obs['market'].shape}")
    print(f"Position input shape: {dummy_obs['position'].shape}")
    print(f"w_cost input shape: {dummy_obs['w_cost'].shape}")
    print(f"Output shape: {features.shape}")
    print(f"features_dim: {extractor.features_dim}")
    print(f"Encoder frozen: {extractor.is_frozen}")
    print(f"Total input dim (market+position+w_cost): {extractor.total_input_dim}")

    # Verify output dimension
    expected_dim = 512
    assert features.shape == (batch_size, expected_dim), f"Shape mismatch! Expected {(batch_size, expected_dim)}"

    # Verify position is being used (sanity check)
    dummy_obs2 = {
        "market": dummy_obs["market"].clone(),
        "position": torch.ones(batch_size, 1),  # Different position
        "w_cost": dummy_obs["w_cost"].clone()
    }
    features2 = extractor(dummy_obs2)
    assert not torch.allclose(features, features2), "Position should affect output!"

    # Verify w_cost is being used (sanity check)
    dummy_obs3 = {
        "market": dummy_obs["market"].clone(),
        "position": dummy_obs["position"].clone(),
        "w_cost": torch.ones(batch_size, 1)  # Different w_cost
    }
    features3 = extractor(dummy_obs3)
    assert not torch.allclose(features, features3), "w_cost should affect output!"

    print("\n[OK] FoundationFeatureExtractor test passed!")
    print("[OK] Position injection verified (different position = different output)")
    print("[OK] w_cost injection verified (different w_cost = different output)")

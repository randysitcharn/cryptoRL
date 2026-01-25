# -*- coding: utf-8 -*-
"""
validators.py - Model dimension validation utilities.

Centralizes validation logic to detect dimension mismatches before model creation
and at checkpoint loading time. Prevents cryptic runtime errors like "mat1 and mat2
shapes cannot be multiplied".

Architecture: This module imports dataclasses from constants.py (one-way dependency).
Dataclasses do NOT have validate() methods to avoid circular imports.
"""

from typing import Optional, Dict, Any
import torch

from src.config.constants import (
    MAEConfig,
    TransformerFeatureExtractorConfig,
    FoundationFeatureExtractorConfig,
    DEFAULT_MAE_CONFIG,
    DEFAULT_TRANSFORMER_CONFIG,
    DEFAULT_FOUNDATION_CONFIG,
    HMM_CONTEXT_SIZE,
)


class ModelDimensionsValidator:
    """
    Validator for model dimension consistency.
    
    All methods are static to facilitate usage without instantiation.
    Raises ValueError with clear error messages when mismatches are detected.
    """

    @staticmethod
    def validate_d_model_divisible_by_n_heads(d_model: int, n_heads: int) -> None:
        """
        Validate that d_model is divisible by n_heads (required for multi-head attention).
        
        Args:
            d_model: Model dimension.
            n_heads: Number of attention heads.
            
        Raises:
            ValueError: If d_model is not divisible by n_heads.
        """
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads}). "
                f"Current remainder: {d_model % n_heads}"
            )

    @staticmethod
    def validate_transformer_config(
        config: TransformerFeatureExtractorConfig,
        n_features: Optional[int] = None,
    ) -> None:
        """
        Validate TransformerFeatureExtractor configuration.
        
        Args:
            config: TransformerFeatureExtractorConfig to validate.
            n_features: Optional number of input features (for input_dim validation).
            
        Raises:
            ValueError: If configuration is invalid.
        """
        # Validate d_model divisible by n_heads
        ModelDimensionsValidator.validate_d_model_divisible_by_n_heads(
            config.d_model, config.n_heads
        )
        
        # Validate positive dimensions
        if config.d_model <= 0:
            raise ValueError(f"d_model must be positive, got {config.d_model}")
        if config.n_heads <= 0:
            raise ValueError(f"n_heads must be positive, got {config.n_heads}")
        if config.n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {config.n_layers}")
        if config.features_dim <= 0:
            raise ValueError(f"features_dim must be positive, got {config.features_dim}")
        
        # Validate dropout in valid range
        if not 0.0 <= config.dropout < 1.0:
            raise ValueError(f"dropout must be in [0.0, 1.0), got {config.dropout}")

    @staticmethod
    def validate_mae_config(config: MAEConfig) -> None:
        """
        Validate MAE configuration.
        
        Args:
            config: MAEConfig to validate.
            
        Raises:
            ValueError: If configuration is invalid.
        """
        # Validate d_model divisible by n_heads
        ModelDimensionsValidator.validate_d_model_divisible_by_n_heads(
            config.d_model, config.n_heads
        )
        
        # Validate positive dimensions
        if config.d_model <= 0:
            raise ValueError(f"d_model must be positive, got {config.d_model}")
        if config.n_heads <= 0:
            raise ValueError(f"n_heads must be positive, got {config.n_heads}")
        if config.n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {config.n_layers}")
        if config.dim_feedforward <= 0:
            raise ValueError(f"dim_feedforward must be positive, got {config.dim_feedforward}")
        
        # Validate dropout in valid range
        if not 0.0 <= config.dropout < 1.0:
            raise ValueError(f"dropout must be in [0.0, 1.0), got {config.dropout}")

    @staticmethod
    def validate_foundation_config(
        config: FoundationFeatureExtractorConfig,
        n_features: Optional[int] = None,
        window_size: Optional[int] = None,
    ) -> None:
        """
        Validate FoundationFeatureExtractor configuration.
        
        Args:
            config: FoundationFeatureExtractorConfig to validate.
            n_features: Optional number of input features (for input_dim validation).
            window_size: Optional window size (for flatten dimension validation).
            
        Raises:
            ValueError: If configuration is invalid.
        """
        # Validate MAE config
        ModelDimensionsValidator.validate_mae_config(config.mae_config)
        
        # Validate features_dim
        if config.features_dim <= 0:
            raise ValueError(f"features_dim must be positive, got {config.features_dim}")
        
        # Validate input dimension consistency if n_features provided
        if n_features is not None:
            # MAE input should exclude HMM context if FiLM is used
            if n_features >= HMM_CONTEXT_SIZE:
                expected_mae_input = n_features - HMM_CONTEXT_SIZE
            else:
                expected_mae_input = n_features
            
            # Note: We can't validate the actual MAE input_dim here without loading the checkpoint,
            # but we can validate that n_features is reasonable
            if n_features < HMM_CONTEXT_SIZE and n_features < 5:
                raise ValueError(
                    f"n_features ({n_features}) is too small. "
                    f"Expected at least {HMM_CONTEXT_SIZE} features for HMM context."
                )
        
        # Validate flatten dimension if window_size provided
        if window_size is not None:
            market_flatten_dim = window_size * config.mae_config.d_model
            total_input_dim = market_flatten_dim + 1 + 1  # +1 for position, +1 for w_cost
            
            if total_input_dim <= 0:
                raise ValueError(
                    f"Total input dimension ({total_input_dim}) must be positive. "
                    f"window_size={window_size}, d_model={config.mae_config.d_model}"
                )

    @staticmethod
    def validate_checkpoint_compatibility(
        checkpoint: Dict[str, Any],
        expected_config: Optional[FoundationFeatureExtractorConfig] = None,
        expected_input_dim: Optional[int] = None,
    ) -> None:
        """
        Validate checkpoint compatibility with expected configuration.
        
        Checks:
        - Input dimension (embedding layer) matches expected
        - d_model matches expected (if available in checkpoint)
        - n_heads matches expected (if available in checkpoint)
        
        Args:
            checkpoint: Checkpoint state dict or nested dict.
            expected_config: Expected FoundationFeatureExtractorConfig (uses DEFAULT if None).
            expected_input_dim: Expected input dimension for MAE (excludes HMM if FiLM used).
            
        Raises:
            ValueError: If checkpoint is incompatible with expected configuration.
        """
        if expected_config is None:
            expected_config = DEFAULT_FOUNDATION_CONFIG
        
        # Extract checkpoint dimensions
        checkpoint_input_dim = None
        checkpoint_d_model = None
        checkpoint_n_heads = None
        
        # Try nested format first (our custom encoder-only format)
        if isinstance(checkpoint, dict) and 'embedding' in checkpoint:
            embedding_dict = checkpoint['embedding']
            if 'weight' in embedding_dict:
                checkpoint_input_dim = embedding_dict['weight'].shape[1]
            
            # Check for d_model in checkpoint metadata
            checkpoint_d_model = checkpoint.get('d_model')
            
        # Try standard state_dict format
        elif isinstance(checkpoint, dict):
            if 'embedding.weight' in checkpoint:
                checkpoint_input_dim = checkpoint['embedding.weight'].shape[1]
            elif 'mae.embedding.weight' in checkpoint:
                checkpoint_input_dim = checkpoint['mae.embedding.weight'].shape[1]
        
        # Validate input dimension if both are available
        if expected_input_dim is not None and checkpoint_input_dim is not None:
            if checkpoint_input_dim != expected_input_dim:
                raise ValueError(
                    f"[ModelDimensionsValidator] CRITICAL: Input dimension mismatch!\n"
                    f"  Checkpoint was trained with {checkpoint_input_dim} features\n"
                    f"  Current model expects {expected_input_dim} features\n"
                    f"  This usually means:\n"
                    f"    - Old checkpoint (pre-trained with HMM features included)\n"
                    f"    - OR new checkpoint (pre-trained without HMM features)\n"
                    f"  Solution: Re-pre-train the MAE with the current feature set\n"
                    f"  Expected config: d_model={expected_config.mae_config.d_model}, "
                    f"n_heads={expected_config.mae_config.n_heads}, "
                    f"n_layers={expected_config.mae_config.n_layers}"
                )
        
        # Validate d_model if available in checkpoint
        if checkpoint_d_model is not None:
            if checkpoint_d_model != expected_config.mae_config.d_model:
                raise ValueError(
                    f"[ModelDimensionsValidator] CRITICAL: d_model mismatch!\n"
                    f"  Checkpoint d_model: {checkpoint_d_model}\n"
                    f"  Expected d_model: {expected_config.mae_config.d_model}\n"
                    f"  Solution: Use a checkpoint trained with d_model={expected_config.mae_config.d_model}"
                )

    @staticmethod
    def validate_observation_space_compatibility(
        observation_space_n_features: int,
        expected_n_features: Optional[int] = None,
        use_film: bool = True,
    ) -> None:
        """
        Validate that observation space feature count is compatible with model expectations.
        
        Args:
            observation_space_n_features: Number of features in observation space.
            expected_n_features: Expected number of features (optional).
            use_film: Whether FiLM is used (affects expected MAE input dimension).
            
        Raises:
            ValueError: If observation space is incompatible.
        """
        if expected_n_features is not None:
            if observation_space_n_features != expected_n_features:
                raise ValueError(
                    f"Observation space has {observation_space_n_features} features, "
                    f"but model expects {expected_n_features} features."
                )
        
        # Validate minimum features for FiLM
        if use_film and observation_space_n_features < HMM_CONTEXT_SIZE:
            raise ValueError(
                f"Observation space has {observation_space_n_features} features, "
                f"but FiLM requires at least {HMM_CONTEXT_SIZE} features "
                f"(for HMM context: {HMM_CONTEXT_SIZE} columns)."
            )

    @staticmethod
    def validate_encoder_compatibility(
        encoder_path: str,
        foundation_config: FoundationFeatureExtractorConfig,
    ) -> None:
        """
        Validate that the encoder checkpoint matches the foundation_config.
        
        Checks:
        - d_model matches expected (from encoder.proj.weight or metadata)
        - n_heads matches expected (if available in checkpoint)
        - n_layers matches expected (if available in checkpoint)
        
        Args:
            encoder_path: Path to encoder checkpoint (.pth file).
            foundation_config: Expected FoundationFeatureExtractorConfig.
            
        Raises:
            ValueError: If encoder is incompatible with foundation_config.
            FileNotFoundError: If encoder_path does not exist.
        """
        import os
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"Encoder checkpoint not found: {encoder_path}")
        
        try:
            checkpoint = torch.load(encoder_path, map_location='cpu', weights_only=False)
        except Exception as e:
            raise ValueError(f"Failed to load encoder checkpoint: {e}")
        
        # Extract d_model from checkpoint
        # Try different possible key formats
        encoder_d_model = None
        encoder_n_heads = None
        encoder_n_layers = None
        
        # Format 1: encoder.proj.weight (projection layer output dimension)
        if 'encoder.proj.weight' in checkpoint:
            encoder_d_model = checkpoint['encoder.proj.weight'].shape[0]
        # Format 2: proj.weight (without encoder prefix)
        elif 'proj.weight' in checkpoint:
            encoder_d_model = checkpoint['proj.weight'].shape[0]
        # Format 3: Nested dict with 'encoder' key
        elif isinstance(checkpoint, dict) and 'encoder' in checkpoint:
            encoder_dict = checkpoint['encoder']
            if isinstance(encoder_dict, dict) and 'proj.weight' in encoder_dict:
                encoder_d_model = encoder_dict['proj.weight'].shape[0]
        
        # Try to extract from metadata if available
        # d_model: use metadata if not found via proj.weight
        if encoder_d_model is None:
            encoder_d_model = checkpoint.get('d_model')
        
        # n_heads and n_layers: always try metadata (may be present even if d_model found via proj.weight)
        if encoder_n_heads is None:
            encoder_n_heads = checkpoint.get('n_heads')
        if encoder_n_layers is None:
            encoder_n_layers = checkpoint.get('n_layers')
        
        # Validate d_model if we found it
        expected_d_model = foundation_config.mae_config.d_model
        if encoder_d_model is not None:
            if encoder_d_model != expected_d_model:
                raise ValueError(
                    f"Encoder d_model mismatch!\n"
                    f"  Checkpoint: d_model={encoder_d_model}\n"
                    f"  Config:     d_model={expected_d_model}\n"
                    f"  Solution: Use an encoder trained with d_model={expected_d_model} "
                    f"or update foundation_config.mae_config.d_model={encoder_d_model}"
                )
        
        # Validate n_heads if available
        expected_n_heads = foundation_config.mae_config.n_heads
        if encoder_n_heads is not None:
            if encoder_n_heads != expected_n_heads:
                raise ValueError(
                    f"Encoder n_heads mismatch!\n"
                    f"  Checkpoint: n_heads={encoder_n_heads}\n"
                    f"  Config:     n_heads={expected_n_heads}\n"
                    f"  Solution: Use an encoder trained with n_heads={expected_n_heads} "
                    f"or update foundation_config.mae_config.n_heads={encoder_n_heads}"
                )
        
        # Validate n_layers if available
        expected_n_layers = foundation_config.mae_config.n_layers
        if encoder_n_layers is not None:
            if encoder_n_layers != expected_n_layers:
                raise ValueError(
                    f"Encoder n_layers mismatch!\n"
                    f"  Checkpoint: n_layers={encoder_n_layers}\n"
                    f"  Config:     n_layers={expected_n_layers}\n"
                    f"  Solution: Use an encoder trained with n_layers={expected_n_layers} "
                    f"or update foundation_config.mae_config.n_layers={encoder_n_layers}"
                )

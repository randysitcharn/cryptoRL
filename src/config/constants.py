# -*- coding: utf-8 -*-
"""
constants.py - Shared constants for cryptoRL.

Centralizes column definitions to ensure consistency across modules.
"""

from dataclasses import dataclass, field
from typing import List

# =============================================================================
# Window Sizes (for features and purge calculation)
# =============================================================================
ZSCORE_WINDOW: int = 720       # 30 jours (prix Z-Score)
VOL_ZSCORE_WINDOW: int = 336   # 14 jours (volume Z-Score)
HMM_SMOOTHING_WINDOW: int = 168  # 7 jours (HMM feature smoothing)
FFD_WINDOW: int = 100          # ~4 jours (Fractional Differentiation)
VOL_WINDOW: int = 24           # 1 jour (Parkinson/GK volatility)

# =============================================================================
# Purge & Embargo Configuration
# =============================================================================
# Purge window must be >= max indicator window to prevent data leakage
# See audit AUDIT_MODELES_RL_RESULTATS.md - Contre-Audit section for rationale

# Compute MAX_LOOKBACK_WINDOW as the maximum of all feature windows
# This ensures purge window covers all lookback dependencies
MAX_LOOKBACK_WINDOW: int = max(
    ZSCORE_WINDOW,           # 720h - price Z-Score
    VOL_ZSCORE_WINDOW,       # 336h - volume Z-Score
    HMM_SMOOTHING_WINDOW,    # 168h - HMM feature smoothing
    FFD_WINDOW,              # 100h - Fractional Differentiation
    VOL_WINDOW,              # 24h  - Parkinson/GK volatility
)

# Backward compatibility alias
MAX_INDICATOR_WINDOW: int = MAX_LOOKBACK_WINDOW

# Default purge must be >= max lookback to prevent data leakage
DEFAULT_PURGE_WINDOW: int = MAX_LOOKBACK_WINDOW  # 720h
DEFAULT_EMBARGO_WINDOW: int = 24  # Gap after test before next train (for label correlation decay)

# Standard OHLCV column names
OHLCV_COLS: List[str] = ['Open', 'High', 'Low', 'Close', 'Volume']

# Columns to exclude from ML features
# These are raw price/volume columns that should not be used directly
# as features (use normalized/derived features instead)
EXCLUDE_COLS: List[str] = [
    # Legacy format (single asset, lowercase)
    'open', 'high', 'low', 'close',
    # Multi-asset format: Prix OHLC bruts
    'BTC_Close', 'ETH_Close', 'SPX_Close', 'DXY_Close', 'NASDAQ_Close',
    'BTC_Open', 'BTC_High', 'BTC_Low',
    'ETH_Open', 'ETH_High', 'ETH_Low',
    'SPX_Open', 'SPX_High', 'SPX_Low',
    'DXY_Open', 'DXY_High', 'DXY_Low',
    'NASDAQ_Open', 'NASDAQ_High', 'NASDAQ_Low',
    # Volumes bruts (utiliser VolRel à la place)
    'BTC_Volume', 'ETH_Volume', 'SPX_Volume', 'DXY_Volume', 'NASDAQ_Volume',
    # HMM intermediate features (only use Prob_0/1/2/3 or HMM_Prob_0/1/2/3 outputs)
    'HMM_Trend', 'HMM_Vol',
    'HMM_RSI_14', 'HMM_MACD_Hist', 'HMM_ADX_14',  # Momentum features depuis FeatureEngineer
    'HMM_Momentum',  # Legacy: remplacé par HMM_RSI_14, HMM_MACD_Hist, HMM_ADX_14
    'HMM_Funding',  # Removed from features (audit P1.2), kept here for backward compatibility
    'HMM_RiskOnOff', 'HMM_VolRatio',
    # Synthetic/problematic features (audit P1.2)
    'Funding_Rate',  # Synthetic funding rate - removed for data integrity
    # Legacy HMM belief states (use HMM_Prob_* instead to avoid duplicates)
    'Prob_0', 'Prob_1', 'Prob_2', 'Prob_3',
]

# Ticker configurations
CRYPTO_TICKERS: List[str] = ['BTC-USD', 'ETH-USD']
MACRO_TICKERS: List[str] = ['^GSPC', 'DX-Y.NYB', '^IXIC']

TICKER_MAPPING = {
    'BTC-USD': 'BTC',
    'ETH-USD': 'ETH',
    '^GSPC': 'SPX',
    'DX-Y.NYB': 'DXY',
    '^IXIC': 'NASDAQ'
}

# Unified asset list (short names used in feature columns)
# Centralized here to ensure consistency across all modules
ASSETS: List[str] = ['BTC', 'ETH', 'SPX', 'DXY', 'NASDAQ']

# Assets for which to compute Fractional Differentiation
FRACDIFF_ASSETS: List[str] = ASSETS

# Assets for which to compute Volume features
VOLUME_ASSETS: List[str] = ASSETS

# =============================================================================
# Feature Filtering (Centralized Logic)
# =============================================================================
# HMM features that should be excluded from MAE pre-training
# These are injected via FiLM modulation in RL training instead
HMM_FEATURE_PREFIXES: List[str] = ['HMM_', 'Prob_']

# Expected HMM context columns for FiLM (must be last 5 columns in RL observations)
HMM_CONTEXT_COLS: List[str] = [
    'HMM_Prob_0', 'HMM_Prob_1', 'HMM_Prob_2',
    'HMM_Prob_3', 'HMM_Entropy',
]
HMM_CONTEXT_SIZE: int = len(HMM_CONTEXT_COLS)  # 5 (4 probabilities + 1 entropy)

# =============================================================================
# MAE Architecture (Single Source of Truth)
# =============================================================================
# Ces valeurs DOIVENT correspondre au checkpoint pré-entraîné.
# Modifier ici = re-pré-entraîner le MAE.
MAE_D_MODEL: int = 256
MAE_N_HEADS: int = 4
MAE_N_LAYERS: int = 2
MAE_DIM_FEEDFORWARD: int = MAE_D_MODEL * 4  # Standard transformer (1024)
MAE_DROPOUT: float = 0.1

# =============================================================================
# RL Training Constants (Single Source of Truth)
# =============================================================================
# Default log_std_init for gSDE exploration
# Formula: std = exp(log_std_init)
# 0.0 gives std=1.0, -3.0 (SB3 default) gives std≈0.05
# Using 3.0 for maximum exploration (std≈20)
DEFAULT_LOG_STD_INIT: float = 3.0

# Default minimum entropy coefficient for EntropyFloorCallback
# Prevents entropy collapse in SAC/TQC auto-tuning
DEFAULT_MIN_ENT_COEF: float = 0.01

# =============================================================================
# Volatility Scaling Constants (Single Source of Truth)
# =============================================================================
# These control position sizing based on market volatility (risk parity)
# Formula: effective_action = raw_action * (target_vol / current_vol)
# Set target_volatility = 1.0 to effectively disable vol scaling

DEFAULT_TARGET_VOLATILITY: float = 1.0  # 100% = disabled (was 0.05 = 5%)
DEFAULT_VOL_WINDOW: int = 24            # EMA window for vol estimation (hours)
DEFAULT_MAX_LEVERAGE: float = 2.0       # Max vol scaling multiplier

# =============================================================================
# Model Architecture Configurations (Dataclasses)
# =============================================================================
# These dataclasses centralize all model dimensions to prevent mismatches.
# All model components should use these configurations instead of hardcoded values.


@dataclass(frozen=True)
class MAEConfig:
    """
    Configuration for the MAE (Masked Auto-Encoder) model.
    
    These values MUST match the pre-trained checkpoint.
    Changing these = re-pre-train the MAE.
    """
    d_model: int = MAE_D_MODEL  # 256
    n_heads: int = MAE_N_HEADS  # 4
    n_layers: int = MAE_N_LAYERS  # 2
    dim_feedforward: int = MAE_DIM_FEEDFORWARD  # 1024 (4 * d_model)
    dropout: float = MAE_DROPOUT  # 0.1


@dataclass(frozen=True)
class TransformerFeatureExtractorConfig:
    """
    Configuration for TransformerFeatureExtractor (standalone, without MAE).
    
    Used when training RL agents without a pre-trained foundation model.
    Smaller dimensions (d_model=32) to prevent overfitting in low data regime.
    """
    d_model: int = 32
    n_heads: int = 2
    n_layers: int = 1
    dim_feedforward: int = 64
    features_dim: int = 256
    dropout: float = 0.1


@dataclass(frozen=True)
class FoundationFeatureExtractorConfig:
    """
    Configuration for FoundationFeatureExtractor (with pre-trained MAE).
    
    Uses MAE encoder as feature extractor, with FiLM modulation for HMM context.
    The MAE dimensions are explicitly composed via mae_config (no implicit inheritance).
    """
    mae_config: MAEConfig = field(default_factory=MAEConfig)  # Explicit composition
    features_dim: int = 512


# Default configuration instances (single source of truth)
DEFAULT_MAE_CONFIG: MAEConfig = MAEConfig()
DEFAULT_TRANSFORMER_CONFIG: TransformerFeatureExtractorConfig = TransformerFeatureExtractorConfig()
DEFAULT_FOUNDATION_CONFIG: FoundationFeatureExtractorConfig = FoundationFeatureExtractorConfig()

# -*- coding: utf-8 -*-
"""
constants.py - Shared constants for cryptoRL.

Centralizes column definitions to ensure consistency across modules.
"""

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

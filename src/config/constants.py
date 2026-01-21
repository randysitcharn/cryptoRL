# -*- coding: utf-8 -*-
"""
constants.py - Shared constants for cryptoRL.

Centralizes column definitions to ensure consistency across modules.
"""

from typing import List

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
    # HMM intermediate features (only use Prob_0/1/2/3 outputs)
    'HMM_Trend', 'HMM_Vol', 'HMM_Momentum',
    'HMM_Funding', 'HMM_RiskOnOff', 'HMM_VolRatio',
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

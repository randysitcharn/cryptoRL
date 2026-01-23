"""
data_engineering - Data processing and splitting module.

Contains:
- DataProcessor: Unified preprocessing (scaling, clipping, safety patches)
- OHLCVProcessor: OHLCV data cleaning and technical indicators
- TimeSeriesSplitter: Chronological train/val/test splitting
- MultiAssetDownloader: Multi-asset data ingestion
- FeatureEngineer: Advanced feature engineering (FFD, volatility, Z-Score)
- RegimeDetector: GMM-HMM market regime detection
- DataManager: Orchestrating the complete data pipeline
"""

from src.data_engineering.processor import DataProcessor, OHLCVProcessor
from src.data_engineering.splitter import TimeSeriesSplitter, validate_purge_window
from src.data_engineering.loader import MultiAssetDownloader
from src.data_engineering.features import FeatureEngineer
from src.data_engineering.manager import RegimeDetector, DataManager

__all__ = [
    'DataProcessor',  # Unified preprocessing (scaling, clipping)
    'OHLCVProcessor',  # OHLCV data cleaning (legacy, renamed)
    'TimeSeriesSplitter',
    'validate_purge_window',
    'MultiAssetDownloader',
    'FeatureEngineer',
    'RegimeDetector',
    'DataManager'
]

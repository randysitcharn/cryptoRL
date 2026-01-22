"""
data_engineering - Data processing and splitting module.

Contains DataProcessor for feature engineering, TimeSeriesSplitter
for chronological train/val/test splitting, MultiAssetDownloader
for multi-asset data ingestion, FeatureEngineer for advanced
feature engineering (FFD, volatility, Z-Score), RegimeDetector
for GMM-HMM market regime detection, and DataManager for
orchestrating the complete data pipeline.
"""

from src.data_engineering.processor import DataProcessor
from src.data_engineering.splitter import TimeSeriesSplitter, validate_purge_window
from src.data_engineering.loader import MultiAssetDownloader
from src.data_engineering.features import FeatureEngineer
from src.data_engineering.manager import RegimeDetector, DataManager

__all__ = [
    'DataProcessor',
    'TimeSeriesSplitter',
    'validate_purge_window',
    'MultiAssetDownloader',
    'FeatureEngineer',
    'RegimeDetector',
    'DataManager'
]

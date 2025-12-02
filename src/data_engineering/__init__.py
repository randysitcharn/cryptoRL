"""
data_engineering - Data processing and splitting module.

Contains DataProcessor for feature engineering and TimeSeriesSplitter
for chronological train/val/test splitting.
"""

from src.data_engineering.processor import DataProcessor
from src.data_engineering.splitter import TimeSeriesSplitter

__all__ = ['DataProcessor', 'TimeSeriesSplitter']

"""
processor.py - Traitement et enrichissement des données.

Classe DataProcessor pour nettoyer les données OHLCV et ajouter :
- Indicateurs techniques (RSI, MACD, ATR, Bollinger Bands)
- Log-returns
- Encodage temporel cyclique (heure, jour de semaine)
- Normalisation des features pour RL (scaling, clipping)
"""

import os
import numpy as np
import pandas as pd
import pandas_ta as ta


class DataProcessor:
    """Nettoie et enrichit les données de marché."""

    def __init__(self, processed_data_dir: str = "data/processed"):
        """
        Initializes the DataProcessor.

        Args:
            processed_data_dir (str): Directory where processed data will be saved.
        """
        self.processed_data_dir = processed_data_dir
        os.makedirs(self.processed_data_dir, exist_ok=True)

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reindex sur plage horaire complète, gère les NaN.

        Args:
            df (pd.DataFrame): Raw OHLCV data.

        Returns:
            pd.DataFrame: Cleaned data with complete hourly index.
        """
        # Vérifier/convertir index datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Créer index complet
        full_idx = pd.date_range(df.index.min(), df.index.max(), freq='1h')
        df = df.reindex(full_idx)

        # ffill puis dropna (NaN au début)
        df = df.ffill().dropna()

        return df

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ajoute RSI, MACD, ATR normalisé, Bollinger %B et Bandwidth.

        Args:
            df (pd.DataFrame): Cleaned OHLCV data.

        Returns:
            pd.DataFrame: Data with technical indicators.
        """
        # RSI (14)
        df['RSI_14'] = df.ta.rsi(length=14)

        # MACD (12, 26, 9)
        macd = df.ta.macd(fast=12, slow=26, signal=9)
        df['MACD_12_26_9'] = macd['MACD_12_26_9']
        df['MACDh_12_26_9'] = macd['MACDh_12_26_9']

        # ATR normalisé (ATR / Close)
        df['ATRr_14'] = df.ta.atr(length=14) / df['close']

        # Bollinger Bands (20, 2)
        bb = df.ta.bbands(length=20, std=2)
        df['BBP_20_2.0'] = bb['BBP_20_2.0_2.0']  # %B
        df['BBB_20_2.0'] = bb['BBB_20_2.0_2.0']  # Bandwidth

        return df

    def add_log_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule les log-returns, remplace inf par 0.

        Args:
            df (pd.DataFrame): Data with OHLCV.

        Returns:
            pd.DataFrame: Data with log returns.
        """
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        df['log_ret'] = df['log_ret'].replace([np.inf, -np.inf], 0)

        return df

    def add_time_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode cyclique de l'heure (24h) et jour de semaine (7j).

        Args:
            df (pd.DataFrame): Data with datetime index.

        Returns:
            pd.DataFrame: Data with time encoding features.
        """
        hour = df.index.hour
        day = df.index.dayofweek

        df['sin_hour'] = np.sin(2 * np.pi * hour / 24)
        df['cos_hour'] = np.cos(2 * np.pi * hour / 24)
        df['sin_day'] = np.sin(2 * np.pi * day / 7)
        df['cos_day'] = np.cos(2 * np.pi * day / 7)

        return df

    def normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalise les features pour le RL (scaling, clipping).

        Args:
            df (pd.DataFrame): Data with all features.

        Returns:
            pd.DataFrame: Normalized data ready for RL.
        """
        # RSI: Scale [0, 100] -> [0, 1]
        df['RSI_14'] = df['RSI_14'] / 100.0

        # MACD: Normaliser par le prix (rendre stationnaire)
        df['MACD_12_26_9'] = df['MACD_12_26_9'] / df['close']
        df['MACDh_12_26_9'] = df['MACDh_12_26_9'] / df['close']

        # Volume: Relatif logarithmique (gère les zéros et réduit l'impact des whales)
        vol_ma = df['volume'].rolling(window=24).mean()
        df['volume_rel'] = np.log1p(df['volume']) / np.log1p(vol_ma)
        df = df.drop(columns=['volume'])

        # Log Returns: Clipping at 5-sigma to handle fat tails while preserving
        # market volatility structure (SOTA winsorization)
        sigma = df['log_ret'].std()
        limit = 5 * sigma
        print(f"[DEBUG] Dynamic Clipping Limit calculated: +/- {limit:.6f}")
        df['log_ret'] = df['log_ret'].clip(lower=-limit, upper=limit)

        return df

    def process_data(self, filepath: str) -> pd.DataFrame:
        """
        Pipeline complet: Clean -> Indicators -> Returns -> TimeEncoding -> Normalize -> DropNA -> Save.

        Args:
            filepath (str): Path to raw CSV file.

        Returns:
            pd.DataFrame: Processed and normalized data.
        """
        print(f"[INFO] Loading {filepath}...")
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)

        print("[INFO] Cleaning data...")
        df = self.clean_data(df)

        print("[INFO] Adding technical indicators...")
        df = self.add_technical_indicators(df)

        print("[INFO] Adding log returns...")
        df = self.add_log_returns(df)

        print("[INFO] Adding time encoding...")
        df = self.add_time_encoding(df)

        print("[INFO] Normalizing features...")
        df = self.normalize_features(df)

        # Supprime NaN créés par indicateurs et rolling
        df = df.dropna()

        # Extraire ticker du nom de fichier
        ticker = os.path.basename(filepath).replace('_1h.csv', '')
        output_path = os.path.join(self.processed_data_dir, f"{ticker}_processed.csv")
        df.to_csv(output_path)
        print(f"[SUCCESS] Saved to {output_path}")

        return df

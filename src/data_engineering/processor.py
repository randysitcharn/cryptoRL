"""
processor.py - Traitement et enrichissement des données.

Contient deux classes :
- OHLCVProcessor : Nettoie les données OHLCV et ajoute des indicateurs techniques
- DataProcessor : Classe unifiée pour le scaling et clipping des features (RobustScaler + Safety Patches)
"""

import os
import numpy as np
import pandas as pd
from typing import Optional, Dict, List
from sklearn.preprocessing import RobustScaler
from typing import Optional, Dict, List, Tuple
from sklearn.preprocessing import RobustScaler


class OHLCVProcessor:
    """Nettoie et enrichit les données de marché."""

    def __init__(self, processed_data_dir: str = "data/processed"):
        """
        Initializes the OHLCVProcessor.

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


class DataProcessor:
    """
    Classe unifiée pour le preprocessing des features (Scaling, Clipping, Safety Patches).
    
    Centralise la logique de transformation des données pour garantir la cohérence
    entre DataManager (Global), WFOPipeline (Segmented) et RegimeDetector.
    
    Permet une exécution "Leak-Free" pour le WFO en contrôlant sur quelles données
    le .fit() est appelé.
    
    Usage:
        # Mode WFO (leak-free)
        processor = DataProcessor(config={'min_iqr': 1e-2, 'clip_range': (-5, 5)})
        processor.fit(train_df[cols])
        train_scaled = processor.transform(train_df[cols])
        eval_scaled = processor.transform(eval_df[cols])
        
        # Mode Global (audit)
        processor = DataProcessor()
        processor.fit(full_data[cols])
        full_scaled = processor.transform(full_data[cols])
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialise le DataProcessor avec configuration centralisée.
        
        Args:
            config: Dictionnaire de configuration optionnel avec:
                - min_iqr: Minimum IQR pour Safety Patch (défaut: 1e-2)
                - clip_range: Tuple (min, max) pour clipping (défaut: (-5, 5))
        """
        if config is None:
            config = {}
        
        # Configuration centralisée (une seule vérité)
        self.min_iqr = config.get('min_iqr', 1e-2)
        self.clip_range = config.get('clip_range', (-5, 5))
        
        # Scaler (initialisé lors du fit)
        self.scaler: Optional[RobustScaler] = None
        
        # Colonnes utilisées lors du fit (pour validation dans transform)
        self.fitted_columns: Optional[List[str]] = None
        
    def fit(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> 'DataProcessor':
        """
        Apprend les statistiques (Mean, Median, IQR) sur les données d'entraînement.
        
        - En mode Audit : appelé sur tout le dataset.
        - En mode WFO : appelé UNIQUEMENT sur le train_set (leak-free).
        
        Args:
            df: DataFrame avec les données d'entraînement.
            columns: Liste des colonnes à scaler. Si None, utilise toutes les colonnes numériques.
            
        Returns:
            self pour permettre le chaining (processor.fit().transform())
            
        Raises:
            ValueError: Si le DataFrame contient des NaNs ou des Inf.
        """
        # Sélectionner les colonnes à scaler
        if columns is None:
            target_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            target_cols = columns
        
        if len(target_cols) == 0:
            raise ValueError("No numeric columns found to scale")
        
        # Vérifier les NaNs (RobustScaler n'aime pas les NaNs)
        if df[target_cols].isna().any().any():
            raise ValueError(
                "DataFrame contains NaN values. Please handle NaNs before calling fit(). "
                "Use dropna(), ffill(), or bfill() as appropriate."
            )
        
        # Vérifier les Inf
        if np.isinf(df[target_cols].values).any():
            raise ValueError(
                "DataFrame contains Inf values. Please handle Inf before calling fit()."
            )
        
        # 1. Initialiser le RobustScaler
        self.scaler = RobustScaler()
        self.scaler.fit(df[target_cols])
        
        # 2. Appliquer le Safety Patch (Centralisé ici !)
        # Empêche l'explosion des valeurs si l'IQR est proche de 0 (ex: SPX_MACD_Hist)
        self.scaler.scale_ = np.maximum(self.scaler.scale_, self.min_iqr)
        
        # 3. Stocker les colonnes utilisées (pour validation dans transform)
        self.fitted_columns = target_cols.copy()
        
        return self
    
    def transform(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Applique la transformation (scaling + clipping) sans apprendre.
        
        IMPORTANT: Retourne un DataFrame avec l'index et les colonnes préservés,
        contrairement à RobustScaler.transform() qui retourne un numpy array.
        
        Args:
            df: DataFrame à transformer.
            columns: Liste des colonnes à transformer. Si None, utilise les colonnes
                    utilisées lors du fit().
            
        Returns:
            DataFrame avec les colonnes transformées (index et colonnes préservés).
            
        Raises:
            RuntimeError: Si fit() n'a pas été appelé avant.
            ValueError: Si le DataFrame contient des NaNs ou des Inf.
        """
        if self.scaler is None:
            raise RuntimeError("fit() must be called before transform()")
        
        # Sélectionner les colonnes à transformer
        if columns is None:
            # Utiliser les colonnes qui ont été fittées
            if self.fitted_columns is None:
                raise RuntimeError("Cannot determine columns to transform. Either call fit() with explicit columns, or pass columns parameter to transform().")
            target_cols = self.fitted_columns
        else:
            target_cols = columns
        
        if len(target_cols) == 0:
            raise ValueError("No numeric columns found to transform")
        
        # Vérifier les NaNs
        if df[target_cols].isna().any().any():
            raise ValueError(
                "DataFrame contains NaN values. Please handle NaNs before calling transform()."
            )
        
        # Vérifier les Inf
        if np.isinf(df[target_cols].values).any():
            raise ValueError(
                "DataFrame contains Inf values. Please handle Inf before calling transform()."
            )
        
        # Créer une copie pour ne pas modifier l'original
        df_out = df.copy()
        
        # 1. Scaling
        data_scaled = self.scaler.transform(df[target_cols])
        
        # 2. Clipping (Centralisé ici !)
        if self.clip_range is not None:
            clip_min, clip_max = self.clip_range
            data_scaled = np.clip(data_scaled, clip_min, clip_max)
        
        # 3. Retourner le DataFrame avec les colonnes modifiées (index préservé)
        df_out[target_cols] = data_scaled
        
        return df_out
    
    def fit_transform(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Fit puis transform (méthode de convenance).
        
        Args:
            df: DataFrame avec les données d'entraînement.
            columns: Liste des colonnes à scaler. Si None, utilise toutes les colonnes numériques.
            
        Returns:
            DataFrame avec les colonnes transformées.
        """
        return self.fit(df, columns).transform(df, columns)
    
    def get_scaler(self) -> Optional[RobustScaler]:
        """
        Retourne le RobustScaler interne (pour sauvegarde/compatibilité).
        
        Returns:
            RobustScaler fitté, ou None si fit() n'a pas été appelé.
        """
        return self.scaler

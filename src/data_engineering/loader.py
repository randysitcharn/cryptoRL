"""
loader.py - Module d'ingestion multi-actifs pour Trading RL.

Classe MultiAssetDownloader pour récupérer et synchroniser les données
horaires de marchés Crypto (24/7) et Traditionnels (5/7) depuis Yahoo Finance.

L'index temporel de BTC-USD sert de Master Timeframe. Les actifs Macro
sont alignés via reindex + forward-fill pour gérer les fermetures week-end/nuit.
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, Tuple

from src.config import OHLCV_COLS, CRYPTO_TICKERS, MACRO_TICKERS, TICKER_MAPPING


class MultiAssetDownloader:
    """
    Télécharge et synchronise les données multi-actifs pour un agent RL.

    Crypto (Master Timeframe): BTC-USD, ETH-USD
    Macro (Slave Timeframe): ^GSPC (S&P 500), DX-Y.NYB (Dollar Index), ^IXIC (Nasdaq)

    La synchronisation utilise BTC-USD comme Index Maître. Les actifs Macro
    sont forward-filled pour combler les périodes de fermeture (week-end, nuit).
    """

    def __init__(self, processed_data_dir: str = "data/processed"):
        """
        Initialise le MultiAssetDownloader.

        Args:
            processed_data_dir: Répertoire de sauvegarde du DataFrame synchronisé.
        """
        self.processed_data_dir = processed_data_dir
        os.makedirs(self.processed_data_dir, exist_ok=True)

    def _download_asset(self, ticker: str, days: int = 730) -> pd.DataFrame:
        """
        Télécharge les données horaires d'un actif via yfinance.

        Args:
            ticker: Symbole de l'actif (ex: 'BTC-USD', '^GSPC').
            days: Nombre de jours d'historique (max 730 pour interval='1h').

        Returns:
            DataFrame avec colonnes OHLCV et index DatetimeIndex.
        """
        # Calculer date de début (729 jours pour rester dans la limite yfinance)
        start_date = datetime.now() - timedelta(days=min(days, 729))

        print(f"[INFO] Downloading {ticker}...")

        try:
            df = yf.download(
                ticker,
                start=start_date,
                interval='1h',
                progress=False,
                auto_adjust=True
            )
        except Exception as e:
            print(f"[ERROR] Failed to download {ticker}: {e}")
            return pd.DataFrame()

        if df.empty:
            print(f"[WARNING] No data returned for {ticker}")
            return df

        # Gérer les colonnes MultiIndex (versions récentes de yfinance)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Standardiser les noms de colonnes (Title Case)
        df.columns = [c.title() for c in df.columns]

        # Garder uniquement les colonnes OHLCV existantes
        available_cols = [c for c in OHLCV_COLS if c in df.columns]
        df = df[available_cols]

        # S'assurer que l'index est DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        print(f"[SUCCESS] {ticker}: {len(df)} rows, {df.index.min()} -> {df.index.max()}")

        return df

    def _generate_synthetic_funding(self, price_series: pd.Series) -> pd.Series:
        """
        Génère un Funding Rate synthétique réaliste via processus Ornstein-Uhlenbeck.

        Le Funding Rate est mean-reverting autour de 0.01% (0.0001) et corrélé
        positivement à la volatilité du prix (quand ça pump, le funding monte).

        Processus O-U: dX = theta * (mu - X) * dt + sigma * dW

        Args:
            price_series: Série des prix Close pour calculer la volatilité.

        Returns:
            Série des Funding Rates synthétiques.
        """
        n = len(price_series)

        # Paramètres Ornstein-Uhlenbeck
        mu = 0.0001  # Mean (0.01% - funding rate moyen)
        theta = 0.1  # Vitesse de mean reversion (plus élevé = retour plus rapide)
        sigma_base = 0.0002  # Volatilité de base du funding

        # Calculer la volatilité rolling du prix (corrélation avec le funding)
        log_returns = np.log(price_series / price_series.shift(1)).fillna(0)
        rolling_vol = log_returns.rolling(window=24, min_periods=1).std().fillna(0)

        # Normaliser la volatilité pour moduler le sigma
        vol_normalized = (rolling_vol - rolling_vol.mean()) / (rolling_vol.std() + 1e-8)
        vol_normalized = vol_normalized.clip(-3, 3)  # Clip à 3 sigma

        # Générer le processus O-U
        np.random.seed(42)  # Reproductibilité
        funding = np.zeros(n)
        funding[0] = mu

        dt = 1.0 / 24  # 1 heure en fraction de jour

        for t in range(1, n):
            # Sigma modulé par la volatilité (quand ça pump, plus de variance)
            sigma_t = sigma_base * (1 + 0.5 * vol_normalized.iloc[t])

            # Drift vers la moyenne + biais haussier si volatilité haute
            vol_bias = 0.5 * mu * vol_normalized.iloc[t]  # Biais corrélé à la vol

            # Euler-Maruyama discretization
            dW = np.random.normal(0, np.sqrt(dt))
            funding[t] = (
                funding[t-1]
                + theta * (mu + vol_bias - funding[t-1]) * dt
                + sigma_t * dW
            )

        # Clip pour rester réaliste (-0.1% à 0.3%)
        funding = np.clip(funding, -0.001, 0.003)

        return pd.Series(funding, index=price_series.index, name='Funding_Rate')

    def _synchronize_dataframes(
        self,
        crypto_dfs: Dict[str, pd.DataFrame],
        macro_dfs: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Synchronise tous les DataFrames sur l'Index Maître (BTC-USD).

        RÈGLE CRITIQUE: Ne jamais supprimer de lignes parce que le marché US est fermé.
        Les actifs Macro sont forward-filled pour combler les week-ends et nuits.

        Args:
            crypto_dfs: Dict des DataFrames crypto {ticker: df}.
            macro_dfs: Dict des DataFrames macro {ticker: df}.

        Returns:
            DataFrame unique avec toutes les colonnes synchronisées.
        """
        # 1. Créer l'Index Maître depuis BTC-USD
        if 'BTC-USD' not in crypto_dfs or crypto_dfs['BTC-USD'].empty:
            raise ValueError("BTC-USD data is required as Master Timeframe")

        master_index = crypto_dfs['BTC-USD'].index
        print(f"[INFO] Master Index (BTC-USD): {len(master_index)} timestamps")

        # 2. Synchroniser tous les actifs
        all_dfs = []

        # Crypto assets (reindex + ffill pour combler les éventuels trous)
        for ticker, df in crypto_dfs.items():
            if df.empty:
                continue

            prefix = TICKER_MAPPING.get(ticker, ticker.replace('-', '_'))

            # Reindex sur Master Index
            df_synced = df.reindex(master_index)

            # Forward-fill uniquement (pas de bfill pour éviter look-ahead bias)
            df_synced = df_synced.ffill()

            # Renommer colonnes: {TICKER}_{FEATURE}
            df_synced.columns = [f"{prefix}_{col}" for col in df_synced.columns]

            all_dfs.append(df_synced)

        # Macro assets (reindex + ffill CRITIQUE pour week-ends/nuits)
        for ticker, df in macro_dfs.items():
            if df.empty:
                print(f"[WARNING] Skipping empty {ticker}")
                continue

            prefix = TICKER_MAPPING.get(ticker, ticker.replace('^', '').replace('-', '_'))

            # IMPORTANT: Arrondir les timestamps Macro à l'heure (floor)
            # Car SPX/NASDAQ ont des timestamps non-ronds (ex: 18:30)
            # qui ne matchent pas l'index BTC (heures rondes: 14:00, 15:00, etc.)
            df_rounded = df.copy()
            df_rounded.index = df_rounded.index.floor('h')

            # Supprimer les duplicats créés par l'arrondi (garder le dernier)
            df_rounded = df_rounded[~df_rounded.index.duplicated(keep='last')]

            # Reindex sur Master Index
            df_synced = df_rounded.reindex(master_index)

            # Forward-fill: La valeur du S&P500 le samedi = celle du vendredi close
            # Pas de bfill pour éviter look-ahead bias (NaN initiaux seront droppés)
            df_synced = df_synced.ffill()

            # Renommer colonnes: {TICKER}_{FEATURE}
            df_synced.columns = [f"{prefix}_{col}" for col in df_synced.columns]

            all_dfs.append(df_synced)

        # 3. Concaténer tous les DataFrames
        if not all_dfs:
            raise ValueError("No valid data to concatenate")

        result = pd.concat(all_dfs, axis=1)

        print(f"[INFO] Synchronized DataFrame: {result.shape[0]} rows, {result.shape[1]} columns")

        return result

    def download_multi_asset(self) -> pd.DataFrame:
        """
        Pipeline complet: Download -> Synchronize -> Add Funding -> Clean -> Save.

        Returns:
            DataFrame multi-actifs synchronisé et nettoyé.
        """
        print("=" * 60)
        print("MULTI-ASSET DOWNLOADER - Starting...")
        print("=" * 60)

        # 1. Télécharger les actifs Crypto
        print("\n[PHASE 1] Downloading Crypto assets (Master Timeframe)...")
        crypto_dfs = {}
        for ticker in CRYPTO_TICKERS:
            crypto_dfs[ticker] = self._download_asset(ticker)

        # 2. Télécharger les actifs Macro
        print("\n[PHASE 2] Downloading Macro assets (Slave Timeframe)...")
        macro_dfs = {}
        for ticker in MACRO_TICKERS:
            macro_dfs[ticker] = self._download_asset(ticker)

        # 3. Synchroniser sur l'Index Maître (BTC-USD)
        print("\n[PHASE 3] Synchronizing on Master Index (BTC-USD)...")
        df = self._synchronize_dataframes(crypto_dfs, macro_dfs)

        # 4. Générer le Funding Rate synthétique
        print("\n[PHASE 4] Generating synthetic Funding Rate (Ornstein-Uhlenbeck)...")
        btc_close_col = 'BTC_Close'
        if btc_close_col in df.columns:
            funding_rate = self._generate_synthetic_funding(df[btc_close_col])
            df['Funding_Rate'] = funding_rate
            print(f"[SUCCESS] Funding Rate generated: mean={funding_rate.mean():.6f}, std={funding_rate.std():.6f}")
        else:
            print("[WARNING] BTC_Close not found, skipping Funding Rate generation")

        # 5. Nettoyer les NaN (uniquement au début à cause du lag ffill)
        initial_rows = len(df)
        df = df.dropna()
        dropped_rows = initial_rows - len(df)
        if dropped_rows > 0:
            print(f"[INFO] Dropped {dropped_rows} initial rows with NaN (ffill lag)")

        # 6. Vérification finale: aucun NaN au milieu
        nan_count = df.isna().sum().sum()
        if nan_count > 0:
            print(f"[ERROR] {nan_count} NaN values remaining in DataFrame!")
        else:
            print("[SUCCESS] No NaN values in final DataFrame")

        # 7. Sauvegarder
        output_path = os.path.join(self.processed_data_dir, "multi_asset.csv")
        df.to_csv(output_path)
        print(f"\n[SUCCESS] Saved to {output_path}")

        print("=" * 60)
        print("MULTI-ASSET DOWNLOADER - Complete!")
        print("=" * 60)

        return df


if __name__ == "__main__":
    # Test du téléchargement multi-actifs
    downloader = MultiAssetDownloader()
    df = downloader.download_multi_asset()

    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    print("\n--- df.head() ---")
    print(df.head())

    print("\n--- df.tail() ---")
    print(df.tail())

    print(f"\n--- Total Rows: {len(df)} ---")

    print("\n--- NaN Check (df.isna().sum()) ---")
    nan_summary = df.isna().sum()
    print(nan_summary)

    print("\n--- Column Names ---")
    print(df.columns.tolist())

    print("\n--- Data Types ---")
    print(df.dtypes)

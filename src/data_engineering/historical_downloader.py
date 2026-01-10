# -*- coding: utf-8 -*-
"""
historical_downloader.py - Téléchargement de données historiques horaires.

Combine:
- Polygon.io pour les stocks (SPY, QQQ, UUP)
- Binance API pour les cryptos (BTC, ETH)

Permet de télécharger des données depuis 2017 (au lieu de 730 jours avec Yahoo).
"""

import os
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

from src.config import OHLCV_COLS


class HistoricalDownloader:
    """
    Télécharge les données horaires historiques depuis Polygon.io et Binance.
    """

    # Polygon.io tickers pour les stocks
    POLYGON_TICKERS = {
        'SPY': 'SPX',      # S&P 500 ETF -> SPX
        'QQQ': 'NASDAQ',   # NASDAQ ETF -> NASDAQ
        'UUP': 'DXY',      # Dollar ETF -> DXY
    }

    # Binance symbols pour les cryptos
    BINANCE_SYMBOLS = {
        'BTCUSDT': 'BTC',
        'ETHUSDT': 'ETH',
    }

    def __init__(
        self,
        polygon_api_key: str,
        output_dir: str = "data/raw_historical"
    ):
        """
        Initialise le HistoricalDownloader.

        Args:
            polygon_api_key: Clé API Polygon.io.
            output_dir: Répertoire de sortie pour les fichiers CSV.
        """
        self.polygon_api_key = polygon_api_key
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # =========================================================================
    # BINANCE API (Crypto)
    # =========================================================================

    def _download_binance_chunk(
        self,
        symbol: str,
        start_ts: int,
        end_ts: int
    ) -> pd.DataFrame:
        """
        Télécharge un chunk de données Binance (max 1000 barres).

        Args:
            symbol: Symbole Binance (ex: 'BTCUSDT').
            start_ts: Timestamp de début en millisecondes.
            end_ts: Timestamp de fin en millisecondes.

        Returns:
            DataFrame avec colonnes OHLCV.
        """
        url = "https://api.binance.com/api/v3/klines"
        params = {
            'symbol': symbol,
            'interval': '1h',
            'startTime': start_ts,
            'endTime': end_ts,
            'limit': 1000
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"[ERROR] Binance API error: {e}")
            return pd.DataFrame()

        if not data:
            return pd.DataFrame()

        # Parser les données Binance
        # Format: [open_time, open, high, low, close, volume, close_time, ...]
        rows = []
        for kline in data:
            rows.append({
                'timestamp': pd.to_datetime(kline[0], unit='ms', utc=True),
                'Open': float(kline[1]),
                'High': float(kline[2]),
                'Low': float(kline[3]),
                'Close': float(kline[4]),
                'Volume': float(kline[5]),
            })

        df = pd.DataFrame(rows)
        df.set_index('timestamp', inplace=True)
        return df

    def download_binance(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Télécharge les données horaires Binance sur une période.

        Args:
            symbol: Symbole Binance (ex: 'BTCUSDT').
            start_date: Date de début 'YYYY-MM-DD'.
            end_date: Date de fin 'YYYY-MM-DD'.

        Returns:
            DataFrame complet avec colonnes OHLCV.
        """
        prefix = self.BINANCE_SYMBOLS.get(symbol, symbol)
        print(f"[BINANCE] Downloading {symbol} ({prefix}) from {start_date} to {end_date}...")

        start_dt = pd.to_datetime(start_date, utc=True)
        end_dt = pd.to_datetime(end_date, utc=True)

        start_ts = int(start_dt.timestamp() * 1000)
        end_ts = int(end_dt.timestamp() * 1000)

        all_data = []
        current_ts = start_ts

        while current_ts < end_ts:
            df_chunk = self._download_binance_chunk(symbol, current_ts, end_ts)

            if df_chunk.empty:
                break

            all_data.append(df_chunk)

            # Avancer au timestamp suivant
            last_ts = int(df_chunk.index[-1].timestamp() * 1000)
            current_ts = last_ts + 3600000  # +1 heure en ms

            # Rate limit (1200 requêtes/minute)
            time.sleep(0.05)

            # Progress
            progress_dt = pd.to_datetime(current_ts, unit='ms', utc=True)
            print(f"  Progress: {progress_dt.strftime('%Y-%m-%d')}...", end='\r')

        if not all_data:
            print(f"[WARNING] No data for {symbol}")
            return pd.DataFrame()

        df = pd.concat(all_data)
        df = df[~df.index.duplicated(keep='first')]
        df.sort_index(inplace=True)

        print(f"[SUCCESS] {symbol}: {len(df)} rows, {df.index[0]} -> {df.index[-1]}")

        return df

    # =========================================================================
    # POLYGON.IO API (Stocks)
    # =========================================================================

    def _download_polygon_chunk(
        self,
        ticker: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Télécharge un chunk de données Polygon.io (max ~50k barres).

        Args:
            ticker: Symbole Polygon (ex: 'SPY').
            start_date: Date de début 'YYYY-MM-DD'.
            end_date: Date de fin 'YYYY-MM-DD'.

        Returns:
            DataFrame avec colonnes OHLCV.
        """
        url = f"https://api.massive.com/v2/aggs/ticker/{ticker}/range/1/hour/{start_date}/{end_date}"
        params = {
            'apiKey': self.polygon_api_key,
            'limit': 50000,
            'sort': 'asc'
        }

        try:
            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"[ERROR] Polygon API error: {e}")
            return pd.DataFrame()

        if data.get('status') != 'OK' or not data.get('results'):
            print(f"[WARNING] No data from Polygon for {ticker}: {data.get('status')}")
            return pd.DataFrame()

        # Parser les données Polygon
        rows = []
        for bar in data['results']:
            rows.append({
                'timestamp': pd.to_datetime(bar['t'], unit='ms', utc=True),
                'Open': bar['o'],
                'High': bar['h'],
                'Low': bar['l'],
                'Close': bar['c'],
                'Volume': bar['v'],
            })

        df = pd.DataFrame(rows)
        df.set_index('timestamp', inplace=True)
        return df

    def download_polygon(
        self,
        ticker: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Télécharge les données horaires Polygon.io sur une période.

        Args:
            ticker: Symbole Polygon (ex: 'SPY').
            start_date: Date de début 'YYYY-MM-DD'.
            end_date: Date de fin 'YYYY-MM-DD'.

        Returns:
            DataFrame complet avec colonnes OHLCV.
        """
        prefix = self.POLYGON_TICKERS.get(ticker, ticker)
        print(f"[POLYGON] Downloading {ticker} ({prefix}) from {start_date} to {end_date}...")

        # Polygon permet de télécharger de grandes périodes d'un coup
        # Mais on découpe par année pour éviter les timeouts
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        all_data = []
        current_start = start_dt

        while current_start < end_dt:
            # Chunk de 6 mois
            current_end = min(current_start + timedelta(days=180), end_dt)

            chunk_start = current_start.strftime('%Y-%m-%d')
            chunk_end = current_end.strftime('%Y-%m-%d')

            df_chunk = self._download_polygon_chunk(ticker, chunk_start, chunk_end)

            if not df_chunk.empty:
                all_data.append(df_chunk)
                print(f"  Chunk {chunk_start} to {chunk_end}: {len(df_chunk)} rows")

            current_start = current_end + timedelta(days=1)

            # Rate limit (5 requêtes/minute pour free tier)
            time.sleep(12)

        if not all_data:
            print(f"[WARNING] No data for {ticker}")
            return pd.DataFrame()

        df = pd.concat(all_data)
        df = df[~df.index.duplicated(keep='first')]
        df.sort_index(inplace=True)

        print(f"[SUCCESS] {ticker}: {len(df)} rows, {df.index[0]} -> {df.index[-1]}")

        return df

    # =========================================================================
    # SYNTHETIC FUNDING RATE
    # =========================================================================

    def _generate_synthetic_funding(self, price_series: pd.Series) -> pd.Series:
        """
        Génère un Funding Rate synthétique via processus Ornstein-Uhlenbeck.
        """
        n = len(price_series)

        mu = 0.0001  # Mean (0.01%)
        theta = 0.1
        sigma_base = 0.0002

        log_returns = np.log(price_series / price_series.shift(1)).fillna(0)
        rolling_vol = log_returns.rolling(window=24, min_periods=1).std().fillna(0)

        vol_normalized = (rolling_vol - rolling_vol.mean()) / (rolling_vol.std() + 1e-8)
        vol_normalized = vol_normalized.clip(-3, 3)

        np.random.seed(42)
        funding = np.zeros(n)
        funding[0] = mu

        dt = 1.0 / 24

        for t in range(1, n):
            sigma_t = sigma_base * (1 + 0.5 * vol_normalized.iloc[t])
            vol_bias = 0.5 * mu * vol_normalized.iloc[t]

            dW = np.random.normal(0, np.sqrt(dt))
            funding[t] = (
                funding[t-1]
                + theta * (mu + vol_bias - funding[t-1]) * dt
                + sigma_t * dW
            )

        funding = np.clip(funding, -0.001, 0.003)

        return pd.Series(funding, index=price_series.index, name='Funding_Rate')

    # =========================================================================
    # MAIN PIPELINE
    # =========================================================================

    def download_all(
        self,
        start_date: str = "2017-01-01",
        end_date: str = None
    ) -> pd.DataFrame:
        """
        Télécharge toutes les données et les synchronise.

        Args:
            start_date: Date de début 'YYYY-MM-DD'.
            end_date: Date de fin (défaut: aujourd'hui).

        Returns:
            DataFrame multi-actifs synchronisé.
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        print("=" * 70)
        print("HISTORICAL DOWNLOADER - Polygon.io + Binance")
        print("=" * 70)
        print(f"Period: {start_date} to {end_date}")

        # 1. Télécharger les cryptos depuis Binance
        print("\n[PHASE 1] Downloading Crypto from Binance...")
        crypto_dfs = {}
        for symbol, prefix in self.BINANCE_SYMBOLS.items():
            df = self.download_binance(symbol, start_date, end_date)
            if not df.empty:
                df.columns = [f"{prefix}_{col}" for col in df.columns]
                crypto_dfs[prefix] = df
                # Save raw
                df.to_csv(os.path.join(self.output_dir, f"{prefix}_1h.csv"))

        # 2. Télécharger les stocks depuis Polygon
        print("\n[PHASE 2] Downloading Stocks from Polygon.io...")
        stock_dfs = {}
        for ticker, prefix in self.POLYGON_TICKERS.items():
            df = self.download_polygon(ticker, start_date, end_date)
            if not df.empty:
                df.columns = [f"{prefix}_{col}" for col in df.columns]
                stock_dfs[prefix] = df
                # Save raw
                df.to_csv(os.path.join(self.output_dir, f"{prefix}_1h.csv"))

        # 3. Synchroniser sur l'index BTC (Master)
        print("\n[PHASE 3] Synchronizing on BTC Master Index...")

        if 'BTC' not in crypto_dfs:
            raise ValueError("BTC data is required as Master Index")

        master_index = crypto_dfs['BTC'].index
        print(f"Master Index: {len(master_index)} timestamps")

        all_dfs = []

        # Ajouter les cryptos
        for prefix, df in crypto_dfs.items():
            df_synced = df.reindex(master_index).ffill()
            all_dfs.append(df_synced)

        # Ajouter les stocks (avec ffill pour week-ends)
        for prefix, df in stock_dfs.items():
            # Arrondir à l'heure
            df_rounded = df.copy()
            df_rounded.index = df_rounded.index.floor('h')
            df_rounded = df_rounded[~df_rounded.index.duplicated(keep='last')]

            df_synced = df_rounded.reindex(master_index).ffill()
            all_dfs.append(df_synced)

        # Concaténer
        result = pd.concat(all_dfs, axis=1)

        # 4. Générer Funding Rate synthétique
        print("\n[PHASE 4] Generating synthetic Funding Rate...")
        if 'BTC_Close' in result.columns:
            funding = self._generate_synthetic_funding(result['BTC_Close'])
            result['Funding_Rate'] = funding

        # 5. Nettoyer les NaN initiaux
        initial_rows = len(result)
        result = result.dropna()
        dropped = initial_rows - len(result)
        print(f"Dropped {dropped} initial rows with NaN")

        # 6. Sauvegarder
        output_path = os.path.join(self.output_dir, "multi_asset_historical.csv")
        result.to_csv(output_path)
        print(f"\n[SUCCESS] Saved to {output_path}")

        print("\n" + "=" * 70)
        print(f"COMPLETE: {len(result)} rows, {result.shape[1]} columns")
        print(f"Date range: {result.index[0]} to {result.index[-1]}")
        print("=" * 70)

        return result


if __name__ == "__main__":
    # API Key Polygon.io
    POLYGON_API_KEY = "tsHVgcCE6TNepRWK5c3x_dY50gIr9woc"

    downloader = HistoricalDownloader(polygon_api_key=POLYGON_API_KEY)

    # Télécharger depuis 2017 (ETH existe, Binance a des données)
    df = downloader.download_all(start_date="2017-08-01")

    print("\n--- Sample Data ---")
    print(df.head())
    print(df.tail())

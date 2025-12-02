import os
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

class YFinanceDownloader:
    def __init__(self, raw_data_dir: str = "data/raw"):
        """
        Initializes the YFinanceDownloader.

        Args:
            raw_data_dir (str): Directory where raw data will be saved.
        """
        self.raw_data_dir = raw_data_dir
        os.makedirs(self.raw_data_dir, exist_ok=True)

    def download_recent_data(self, ticker: str, interval: str = "1h") -> pd.DataFrame:
        """
        Downloads the last 729 days of hourly data from yfinance (limit for 1h interval).

        Args:
            ticker (str): The symbol to download (e.g., 'BTC-USDT').
            interval (str): Data interval (default: '1h').

        Returns:
            pd.DataFrame: The processed DataFrame.
        """
        # 1. Calculate safe start date (729 days ago to adhere to yfinance 730d limit)
        start_date = datetime.now() - timedelta(days=729)
        start_date_str = start_date.strftime('%Y-%m-%d')

        print(f"\n[INFO] Downloading {ticker} from {start_date_str} (Last ~729 days)...")

        # 2. API Call
        try:
            df = yf.download(ticker, start=start_date, interval=interval, progress=False)
        except Exception as e:
            print(f"[ERROR] Failed download for {ticker}: {e}")
            return pd.DataFrame()

        if df.empty:
            print(f"[WARNING] DataFrame is empty for {ticker}.")
            return df

        # 3. Clean DataFrame
        # Handle MultiIndex columns if present (common in recent yfinance versions)
        if isinstance(df.columns, pd.MultiIndex):
            # We assume level 0 is the Price Type (Close, Open, etc.) and level 1 might be Ticker
            # We want to keep just the Price Type
            try:
                df.columns = df.columns.get_level_values(0)
            except IndexError:
                pass

        # Rename to lowercase
        df.columns = [c.lower() for c in df.columns]

        # Keep only standard OHLCV columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        # Filter existing columns
        existing_cols = [c for c in required_cols if c in df.columns]
        df = df[existing_cols]

        # Drop NaNs
        df = df.dropna()

        # Check if empty after cleaning
        if df.empty:
            print(f"[WARNING] DataFrame is empty after cleaning for {ticker}.")
            return df

        # 4. Save
        filename = f"{ticker}_{interval}.csv"
        filepath = os.path.join(self.raw_data_dir, filename)
        df.to_csv(filepath)
        print(f"[SUCCESS] Saved to {filepath}")

        return df

if __name__ == "__main__":
    downloader = YFinanceDownloader()
    tickers = ['BTC-USD', 'ETH-USD']

    for ticker in tickers:
        df = downloader.download_recent_data(ticker)

        if not df.empty:
            print(f"Ticker: {ticker}")
            print(f"Rows: {len(df)}")
            print(f"Time range: {df.index.min()} -> {df.index.max()}")
        else:
            print(f"Ticker: {ticker} - NO DATA")

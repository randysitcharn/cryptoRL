"""
train.py - Script d'entrainement principal pour l'agent SAC.

Lance l'entrainement de l'agent sur les donnees de trading crypto
avec evaluation periodique sur les donnees de validation.
"""

import os
import pandas as pd

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback

from src.env.trading_env import CryptoTradingEnv
from src.models.agent import create_sac_agent
from src.data_engineering.splitter import TimeSeriesSplitter


def main():
    """Main training function."""
    print("=" * 60)
    print("CryptoRL - SAC Training Script")
    print("=" * 60)

    # ==================== Configuration ====================
    DATA_PATH = "data/processed/BTC-USD_processed.csv"
    TOTAL_TIMESTEPS = 50_000
    EVAL_FREQ = 2000

    # Create directories if they don't exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    print("\n[OK] Directories created: models/, logs/")

    # ==================== Data Loading ====================
    print(f"\n[INFO] Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    print(f"[OK] Loaded {len(df)} rows")

    # ==================== Data Splitting ====================
    print("\n[INFO] Splitting data...")
    splitter = TimeSeriesSplitter(df)
    train_df, val_df, test_df = splitter.split_data(
        train_ratio=0.7,
        val_ratio=0.15,
        purge_window=50
    )

    # test_df is reserved for final evaluation (not used during training)
    print(f"\n[INFO] Test set ({len(test_df)} rows) reserved for final evaluation")

    # ==================== Environment Setup ====================
    print("\n[INFO] Creating environments...")

    # DummyVecEnv expects a list of callables (lambda functions)
    train_env = DummyVecEnv([lambda df=train_df: CryptoTradingEnv(df)])
    eval_env = DummyVecEnv([lambda df=val_df: CryptoTradingEnv(df)])

    print("[OK] Train environment created")
    print("[OK] Eval environment created")

    # ==================== Callback Setup ====================
    print("\n[INFO] Setting up evaluation callback...")
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./models/',
        log_path='./logs/',
        eval_freq=EVAL_FREQ,
        deterministic=True,
        render=False
    )
    print(f"[OK] EvalCallback: eval every {EVAL_FREQ} steps")

    # ==================== Agent Creation ====================
    print("\n[INFO] Creating SAC agent...")
    model = create_sac_agent(train_env)
    print("[OK] Agent created")

    # ==================== Training ====================
    print("\n" + "=" * 60)
    print(f"Starting training for {TOTAL_TIMESTEPS:,} timesteps...")
    print("=" * 60 + "\n")

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=eval_callback,
        progress_bar=True
    )

    # ==================== Save Final Model ====================
    print("\n[INFO] Saving final model...")
    model.save("models/final_model")
    print("[OK] Model saved to models/final_model.zip")

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print("\nArtifacts:")
    print("  - Best model: models/best_model.zip")
    print("  - Final model: models/final_model.zip")
    print("  - Logs: logs/evaluations.npz")


if __name__ == "__main__":
    main()

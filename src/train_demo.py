# -*- coding: utf-8 -*-
"""
train_demo.py - Script de smoke test rapide pour TQC + Transformer.

Version allegee de train.py pour verifier le bon fonctionnement
du pipeline sans attendre un entrainement complet.

Parametres reduits:
- TOTAL_TIMESTEPS: 1,000 (vs 50,000)
- EVAL_FREQ: 500 (vs 2,000)
- buffer_size: 1,000 (vs 100,000)
- batch_size: 64 (vs 256)
"""

import os
import pandas as pd

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from src.training.env import CryptoTradingEnv
from src.models.agent import create_tqc_agent
from src.training.callbacks import TensorBoardStepCallback
from src.data_engineering.splitter import TimeSeriesSplitter


def main():
    """Main demo training function."""
    print("=" * 60)
    print("CryptoRL - TQC + Transformer [DEMO MODE]")
    print("=" * 60)
    print("\n*** DEMO MODE: Parametres alleges pour test rapide ***\n")

    # ==================== Configuration (DEMO) ====================
    DATA_PATH = "data/processed/BTC-USD_processed.csv"
    WINDOW_SIZE = 64
    TOTAL_TIMESTEPS = 1_000      # Reduit pour demo
    EVAL_FREQ = 500              # Reduit pour demo

    # Hyperparametres alleges pour demo
    DEMO_HYPERPARAMS = {
        "buffer_size": 1_000,    # Reduit (vs 100,000)
        "batch_size": 64,        # Reduit (vs 256)
    }

    # Create directories if they don't exist
    os.makedirs("models/demo", exist_ok=True)
    os.makedirs("logs/demo", exist_ok=True)
    os.makedirs("logs/demo/tensorboard", exist_ok=True)
    print("[OK] Directories created: models/demo/, logs/demo/, logs/demo/tensorboard/")

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

    # Validate minimum data requirements for Transformer
    assert len(train_df) >= WINDOW_SIZE, f"Train set too small: {len(train_df)} < {WINDOW_SIZE}"
    assert len(val_df) >= WINDOW_SIZE, f"Val set too small: {len(val_df)} < {WINDOW_SIZE}"
    print(f"[OK] Data validation passed (min window_size={WINDOW_SIZE})")

    # ==================== Environment Setup ====================
    print("\n[INFO] Creating environments...")

    # Monitor wrapper required for TensorBoard episode rewards logging
    train_env = DummyVecEnv([lambda df=train_df: Monitor(CryptoTradingEnv(df, window_size=WINDOW_SIZE))])
    # Wrap eval env with Monitor for proper episode tracking
    eval_env = DummyVecEnv([lambda df=val_df: Monitor(CryptoTradingEnv(df, window_size=WINDOW_SIZE))])

    print(f"[OK] Train environment created (window_size={WINDOW_SIZE})")
    print(f"[OK] Eval environment created (window_size={WINDOW_SIZE})")

    # ==================== Callback Setup ====================
    print("\n[INFO] Setting up evaluation callback...")
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./models/demo/',
        log_path='./logs/demo/',
        eval_freq=EVAL_FREQ,
        deterministic=True,
        render=False
    )
    print(f"[OK] EvalCallback: eval every {EVAL_FREQ} steps")

    # ==================== Agent Creation ====================
    print("\n[INFO] Creating TQC agent with Transformer (DEMO params)...")
    model = create_tqc_agent(
        train_env,
        hyperparams=DEMO_HYPERPARAMS,
        tensorboard_log="./logs/demo/tensorboard/"
    )
    print("[OK] Agent created (TensorBoard: logs/demo/tensorboard/)")

    # ==================== Training ====================
    print("\n" + "=" * 60)
    print(f"[DEMO] Starting training for {TOTAL_TIMESTEPS:,} timesteps...")
    print("=" * 60 + "\n")

    # Callbacks: EvalCallback + TensorBoard step logging
    tb_callback = TensorBoardStepCallback(
        log_dir="./logs/demo/tensorboard_steps/",
        log_freq=1,
        verbose=1
    )
    callbacks = [eval_callback, tb_callback]

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callbacks,
        progress_bar=True
    )

    # ==================== Save Final Model ====================
    print("\n[INFO] Saving demo model...")
    model.save("models/demo/final_model_demo")
    print("[OK] Model saved to models/demo/final_model_demo.zip")

    print("\n" + "=" * 60)
    print("[DEMO] Training Complete!")
    print("=" * 60)
    print("\nDemo Artifacts:")
    print("  - Best model: models/demo/best_model.zip")
    print("  - Final model: models/demo/final_model_demo.zip")
    print("  - Logs: logs/demo/evaluations.npz")
    print("  - TensorBoard: logs/demo/tensorboard/")
    print("\nTo view TensorBoard:")
    print("  tensorboard --logdir=logs/demo/tensorboard/")
    print("\n*** Si ce test passe, le pipeline TQC/Transformer fonctionne! ***")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
run_full_wfo.py - Walk-Forward Optimization with Full Retrain.

Implements a strict WFO pipeline that retrains the entire stack
(Preprocessing + HMM + MAE + TQC) for each temporal window to avoid data leakage.

Pipeline per Segment:
1. Preprocessing "Leak-Free": RobustScaler fit on TRAIN only
2. HMM: fit on TRAIN, predict on TRAIN+TEST
3. MAE: train on TRAIN
4. TQC: train on TRAIN
5. Evaluation: backtest on TEST

Usage:
    python scripts/run_full_wfo.py --segments 10 --timesteps 300000
    python scripts/run_full_wfo.py --segment 0 --timesteps 300000  # Single segment
"""

import os
import sys
import gc
import json
import pickle
import shutil
import socket
import subprocess
import argparse
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

# Add project root to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.data_engineering.features import FeatureEngineer
from src.data_engineering.manager import RegimeDetector


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class WFOConfig:
    """Walk-Forward Optimization configuration."""

    # Data
    raw_data_path: str = "data/raw_training_data.parquet"

    # GPU Acceleration
    use_batch_env: bool = True  # Use BatchCryptoEnv for GPU-accelerated training (default ON)
    resume: bool = False  # Resume TQC training from tqc_last.zip
    output_dir: str = "data/wfo"
    models_dir: str = "models/wfo"
    weights_dir: str = "weights/wfo"
    results_path: str = "results/wfo_results.csv"

    # WFO Parameters
    train_months: int = 12
    test_months: int = 3
    step_months: int = 3  # Rolling step
    hours_per_month: int = 720  # 30 days * 24 hours
    window_size: int = 64  # Observation window for transformer encoder

    # Training Parameters
    mae_epochs: int = 90
    tqc_timesteps: int = 30_000_000  # 30M steps

    # TQC Hyperparameters (Gemini collab 2026-01-13)
    learning_rate: float = 1e-4      # Conservative for stability
    buffer_size: int = 2_500_000  # 2.5M replay buffer
    n_envs: int = 512   # Optimized for dual-GPU parallel run (CPU bottleneck relief)
    batch_size: int = 2048  # Large batch for GPU efficiency
    gamma: float = 0.99    # Horizon ~100h (increased for long-term strategy)
    ent_coef: Union[str, float] = "auto"  # Auto entropy tuning
    churn_coef: float = 0.5    # Max target après curriculum (réduit)
    smooth_coef: float = 1e-5  # Très bas (curriculum monte à 0.00005 max)

    # Regularization (anti-overfitting)
    observation_noise: float = 0.01  # 1% Gaussian noise on market observations

    # Volatility Scaling (Target Volatility)
    target_volatility: float = 0.05  # 5% target vol
    vol_window: int = 24  # 24h rolling window
    max_leverage: float = 2.0  # Conservative max scaling

    # Columns to exclude from scaling
    exclude_from_scaling: List[str] = field(default_factory=lambda: [
        # Prix OHLC bruts
        'BTC_Close', 'ETH_Close', 'SPX_Close', 'DXY_Close', 'NASDAQ_Close',
        'BTC_Open', 'BTC_High', 'BTC_Low',
        'ETH_Open', 'ETH_High', 'ETH_Low',
        'SPX_Open', 'SPX_High', 'SPX_Low',
        'DXY_Open', 'DXY_High', 'DXY_Low',
        'NASDAQ_Open', 'NASDAQ_High', 'NASDAQ_Low',
        # Volumes bruts
        'BTC_Volume', 'ETH_Volume', 'SPX_Volume', 'DXY_Volume', 'NASDAQ_Volume',
        # Log-returns (déjà clippés)
        'BTC_LogRet', 'ETH_LogRet', 'SPX_LogRet', 'DXY_LogRet', 'NASDAQ_LogRet',
        # Z-Scores (déjà normalisés mean≈0, std≈1)
        'BTC_ZScore', 'ETH_ZScore', 'SPX_ZScore', 'DXY_ZScore', 'NASDAQ_ZScore',
        'BTC_Vol_ZScore', 'ETH_Vol_ZScore', 'SPX_Vol_ZScore', 'DXY_Vol_ZScore', 'NASDAQ_Vol_ZScore',
        # Probabilités HMM (déjà dans [0, 1])
        'Prob_0', 'Prob_1', 'Prob_2', 'Prob_3',
    ])

    @property
    def train_rows(self) -> int:
        return self.train_months * self.hours_per_month

    @property
    def test_rows(self) -> int:
        return self.test_months * self.hours_per_month

    @property
    def step_rows(self) -> int:
        return self.step_months * self.hours_per_month


# =============================================================================
# WFO Pipeline
# =============================================================================

class WFOPipeline:
    """
    Walk-Forward Optimization Pipeline.

    Handles the full retrain pipeline for each segment:
    1. Feature Engineering (from raw OHLCV)
    2. Leak-Free Scaling (fit on train only)
    3. HMM Training (fit on train, predict on both)
    4. MAE Training
    5. TQC Training
    6. Evaluation
    """

    def __init__(self, config: WFOConfig):
        self.config = config
        self.feature_engineer = FeatureEngineer()

        # Create directories
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.models_dir, exist_ok=True)
        os.makedirs(config.weights_dir, exist_ok=True)
        os.makedirs(os.path.dirname(config.results_path), exist_ok=True)

    def calculate_segments(self, total_rows: int) -> List[Dict[str, int]]:
        """
        Calculate segment boundaries for rolling WFO.

        Returns:
            List of dicts with train_start, train_end, test_start, test_end
        """
        segments = []
        segment_size = self.config.train_rows + self.config.test_rows

        start = 0
        segment_id = 0

        while start + segment_size <= total_rows:
            train_start = start
            train_end = start + self.config.train_rows
            test_start = train_end
            test_end = train_end + self.config.test_rows

            segments.append({
                'id': segment_id,
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
            })

            start += self.config.step_rows
            segment_id += 1

        return segments

    def preprocess_segment(
        self,
        df_raw: pd.DataFrame,
        segment: Dict[str, int]
    ) -> tuple[pd.DataFrame, pd.DataFrame, RobustScaler]:
        """
        Preprocess a segment with leak-free scaling.

        1. Extract train/test slices from raw data
        2. Apply feature engineering
        3. Fit scaler on TRAIN only
        4. Transform both TRAIN and TEST

        Returns:
            (train_df, test_df, scaler)
        """
        segment_id = segment['id']
        print(f"\n[Segment {segment_id}] Preprocessing...")

        # 1. Extract full segment from pre-computed features (P1 optimization)
        full_start = segment['train_start']
        full_end = segment['test_end']

        # Use global pre-computed features if available
        if hasattr(self, '_df_features_global') and self._df_features_global is not None:
            # Convert integer positions to datetime using raw data's index
            # This preserves alignment because features keep the original DatetimeIndex
            start_time = df_raw.index[full_start]
            end_time = df_raw.index[full_end - 1]  # -1 because .loc[] is inclusive
            df_features = self._df_features_global.loc[start_time:end_time].copy()
            print(f"  Using pre-computed features: {start_time} to {end_time} ({len(df_features)} rows)")
        else:
            # Fallback: compute features on-the-fly (legacy path)
            buffer = 720
            safe_start = max(0, full_start - buffer)
            df_segment = df_raw.iloc[safe_start:full_end].copy()
            print(f"  Raw segment: rows {safe_start} to {full_end} ({len(df_segment)} rows)")
            print("  Applying feature engineering...")
            df_features = self.feature_engineer.engineer_features(df_segment)

        # 2. Split train/test (take from END to get most recent valid data)
        train_len = segment['train_end'] - segment['train_start']
        test_len = segment['test_end'] - segment['test_start']
        total_needed = train_len + test_len

        if len(df_features) < total_needed:
            print(f"  [WARNING] Not enough data: {len(df_features)} < {total_needed}")
            train_df = df_features.iloc[:train_len].copy()
            test_df = df_features.iloc[train_len:].copy()
        else:
            # CRITICAL: Take from the END to match fallback behavior
            train_df = df_features.iloc[-total_needed:-test_len].copy()
            test_df = df_features.iloc[-test_len:].copy()

        print(f"  Train: {len(train_df)} rows, Test: {len(test_df)} rows")

        # 4. Leak-Free Scaling: fit on TRAIN only
        print("  Applying RobustScaler (fit on train only)...")
        scaler = RobustScaler()

        # Identify columns to scale
        cols_to_scale = [
            col for col in train_df.columns
            if col not in self.config.exclude_from_scaling
            and train_df[col].dtype in ['float64', 'float32']
        ]

        print(f"  Scaling {len(cols_to_scale)} columns")

        # Fit on train
        scaler.fit(train_df[cols_to_scale])

        # Transform both
        train_df[cols_to_scale] = scaler.transform(train_df[cols_to_scale])
        test_df[cols_to_scale] = scaler.transform(test_df[cols_to_scale])

        # Clip scaled features to [-5, 5]
        train_df[cols_to_scale] = train_df[cols_to_scale].clip(-5, 5)
        test_df[cols_to_scale] = test_df[cols_to_scale].clip(-5, 5)
        print(f"  Clipped {len(cols_to_scale)} scaled columns to [-5, 5]")

        # Clip ZScores to [-5, 5] (already normalized, just safety clip)
        zscore_cols = [c for c in train_df.columns if 'ZScore' in c]
        train_df[zscore_cols] = train_df[zscore_cols].clip(-5, 5)
        test_df[zscore_cols] = test_df[zscore_cols].clip(-5, 5)
        print(f"  Clipped {len(zscore_cols)} ZScore columns to [-5, 5]")

        return train_df, test_df, scaler

    def train_hmm(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        segment_id: int
    ) -> tuple[pd.DataFrame, pd.DataFrame, RegimeDetector, int]:
        """
        Train HMM on train data and predict on both.

        Returns:
            (train_with_hmm, test_with_hmm, detector, context_rows)
            context_rows: Number of buffer rows prepended to test for env warmup
        """
        print(f"\n[Segment {segment_id}] Training HMM...")

        # Initialize detector
        detector = RegimeDetector(n_components=4)

        # Fit on train with TensorBoard logging
        train_with_hmm = detector.fit_predict(
            train_df,
            tensorboard_log="logs/wfo/hmm/",
            run_name=f"segment_{segment_id}",
            segment_id=segment_id
        )

        # Predict on test WITH CONTEXT BUFFER (for HMM lookback requirements)
        # HMM needs 168h window, use 336h to be safe
        max_lookback = 336
        context_rows = min(max_lookback, len(train_df))

        # Include last context_rows from train as buffer for test prediction
        test_with_context = pd.concat([
            train_df.tail(context_rows),
            test_df
        ], ignore_index=True)

        test_with_hmm_full = detector.predict(test_with_context)

        # KEEP buffer rows for env warmup (window_size=64 needs enough rows)
        # context_rows will be skipped during metrics calculation
        test_with_hmm = test_with_hmm_full.reset_index(drop=True)
        actual_test_rows = len(test_df)

        # Safety assertion: ensure we have context + actual test rows
        assert len(test_with_hmm) == context_rows + actual_test_rows, \
            f"Context buffer error: got {len(test_with_hmm)} rows, expected {context_rows + actual_test_rows}"

        # Save HMM
        hmm_dir = os.path.join(self.config.models_dir, f"segment_{segment_id}")
        os.makedirs(hmm_dir, exist_ok=True)
        hmm_path = os.path.join(hmm_dir, "hmm.pkl")
        detector.save(hmm_path)

        # Remove HMM intermediate features (keep only Prob_*)
        hmm_features = ['HMM_Trend', 'HMM_Vol', 'HMM_Momentum']
        train_final = train_with_hmm.drop(columns=hmm_features, errors='ignore')
        test_final = test_with_hmm.drop(columns=hmm_features, errors='ignore')

        print(f"  Train final: {train_final.shape}, Test final: {test_final.shape} (context_rows={context_rows})")

        return train_final, test_final, detector, context_rows

    def save_segment_data(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        scaler: RobustScaler,
        segment_id: int,
        context_rows: int = 0
    ):
        """Save preprocessed data, scaler, and metadata for a segment."""

        # Data directory
        data_dir = os.path.join(self.config.output_dir, f"segment_{segment_id}")
        os.makedirs(data_dir, exist_ok=True)

        # Save parquet files
        train_path = os.path.join(data_dir, "train.parquet")
        test_path = os.path.join(data_dir, "test.parquet")
        train_df.to_parquet(train_path, index=False)
        test_df.to_parquet(test_path, index=False)

        # Save metadata (for eval-only mode)
        metadata = {
            'context_rows': context_rows,
            'total_test_rows': len(test_df),
            'actual_test_rows': len(test_df) - context_rows
        }
        metadata_path = os.path.join(data_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save scaler
        models_dir = os.path.join(self.config.models_dir, f"segment_{segment_id}")
        os.makedirs(models_dir, exist_ok=True)
        scaler_path = os.path.join(models_dir, "scaler.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump({'scaler': scaler, 'columns': list(train_df.columns)}, f)

        print(f"  Saved: {train_path}, {test_path}, {metadata_path}")

        return train_path, test_path

    def train_mae(self, train_path: str, segment_id: int) -> str:
        """
        Train MAE on segment train data.

        Returns:
            Path to saved encoder weights.
        """
        print(f"\n[Segment {segment_id}] Training MAE...")

        # Import here to avoid circular imports
        from src.training.train_foundation import train, TrainingConfig

        # Configure MAE training
        config = TrainingConfig()
        config.data_path = train_path
        config.epochs = self.config.mae_epochs
        config.patience = 7

        # Set segment-specific paths
        weights_dir = os.path.join(self.config.weights_dir, f"segment_{segment_id}")
        os.makedirs(weights_dir, exist_ok=True)

        config.weights_dir = weights_dir
        config.checkpoint_path = os.path.join(weights_dir, "mae_full.pth")
        config.encoder_path = os.path.join(weights_dir, "encoder.pth")

        # TensorBoard logging
        config.tensorboard_log = "logs/wfo/mae/"
        config.run_name = f"segment_{segment_id}"

        # DEBUG: Verify d_model alignment
        print(f"  DEBUG: Starting MAE Training with d_model={config.d_model}")

        # Train
        model, best_loss = train(config, from_scratch=True)

        print(f"  MAE trained. Best loss: {best_loss:.4f}")
        print(f"  Encoder saved: {config.encoder_path}")

        # Cleanup
        del model
        gc.collect()

        return config.encoder_path

    def train_tqc(
        self,
        train_path: str,
        encoder_path: str,
        segment_id: int,
        use_batch_env: bool = False,
        resume: bool = False
    ) -> tuple[str, Dict[str, Any]]:
        """
        Train TQC agent on segment train data.

        Args:
            train_path: Path to train data parquet.
            encoder_path: Path to encoder weights.
            segment_id: Segment identifier.
            use_batch_env: If True, use GPU-accelerated BatchCryptoEnv.
            resume: If True, resume from tqc_last.zip (continues TensorBoard steps).

        Returns:
            Tuple of (path to saved agent, training metrics dict).
        """
        env_type = "BatchCryptoEnv (GPU)" if use_batch_env else "SubprocVecEnv (CPU)"
        print(f"\n[Segment {segment_id}] Training TQC with {env_type}...")

        # Import here to avoid circular imports
        import torch
        from src.training.train_agent import train, TrainingConfig

        # Configure TQC training
        config = TrainingConfig()
        config.data_path = train_path
        config.encoder_path = encoder_path
        config.total_timesteps = self.config.tqc_timesteps
        config.learning_rate = self.config.learning_rate
        config.buffer_size = self.config.buffer_size  # 2.5M replay buffer
        # batch_size, n_envs: Auto-detected by HardwareManager
        config.gamma = self.config.gamma
        config.ent_coef = self.config.ent_coef
        config.churn_coef = self.config.churn_coef
        config.smooth_coef = self.config.smooth_coef
        config.observation_noise = self.config.observation_noise  # Anti-overfitting
        config.target_volatility = self.config.target_volatility
        config.vol_window = self.config.vol_window
        config.max_leverage = self.config.max_leverage

        # 3-Phase Curriculum: Discovery → Discipline → Refinement (Gemini 2026-01-13)
        config.use_curriculum = True

        # Set segment-specific paths
        weights_dir = os.path.join(self.config.weights_dir, f"segment_{segment_id}")
        config.save_path = os.path.join(weights_dir, "tqc.zip")
        config.checkpoint_dir = os.path.join(weights_dir, "checkpoints/")
        config.tensorboard_log = f"logs/wfo/segment_{segment_id}/"
        config.name = f"WFO_seg{segment_id}"

        # NOTE: eval_data_path set to None to disable EvalCallback (WFO mode).
        # This prevents data leakage and crashes from mismatched eval environments.
        # Safety CheckpointCallback saves every 100k steps instead.
        config.eval_data_path = None  # WFO mode: disable EvalCallback

        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.tensorboard_log, exist_ok=True)

        # Train - now returns (model, train_metrics)
        # Pass n_envs and batch_size overrides for GPU-optimized parallelism
        hw_overrides = {
            'n_envs': self.config.n_envs,
            'batch_size': self.config.batch_size
        } if use_batch_env else None

        # Resume logic: use tqc_last.zip if --resume flag is set
        resume_path = None
        if resume:
            tqc_last_path = os.path.join(weights_dir, "tqc_last.zip")
            if os.path.exists(tqc_last_path):
                resume_path = tqc_last_path
                print(f"  🔄 RESUME MODE: Loading from {tqc_last_path}")
            else:
                print(f"  ⚠️ Resume requested but {tqc_last_path} not found. Starting fresh.")

        model, train_metrics = train(
            config,
            hw_overrides=hw_overrides,
            use_batch_env=use_batch_env,
            resume_path=resume_path
        )

        print(f"  TQC trained. Saved: {config.save_path}")

        # Cleanup
        del model
        torch.cuda.empty_cache()
        gc.collect()

        return config.save_path, train_metrics

    def evaluate_segment(
        self,
        test_path: str,
        encoder_path: str,
        tqc_path: str,
        segment_id: int,
        context_rows: int = 0,
        train_metrics: Optional[Dict[str, Any]] = None,
        train_path: Optional[str] = None
    ) -> tuple[Dict[str, Any], np.ndarray]:
        """
        Evaluate trained agent on test data.

        Args:
            test_path: Path to test data parquet.
            encoder_path: Path to encoder weights.
            tqc_path: Path to TQC agent.
            segment_id: Segment identifier.
            context_rows: Number of warmup rows to skip in metrics (env warmup period).
            train_metrics: Optional training metrics from train_tqc.
            train_path: Path to train data parquet (for baseline_vol calculation).

        Returns:
            Tuple of (metrics dict, navs array for plotting).
        """
        print(f"\n[Segment {segment_id}] Evaluating (skipping {context_rows} warmup rows)...")

        import torch
        from sb3_contrib import TQC
        from src.training.env import CryptoTradingEnv
        from src.training.wrappers import RiskManagementWrapper

        # Calculate baseline_vol from TRAIN data (avoids data leakage)
        if train_path and os.path.exists(train_path):
            train_df = pd.read_parquet(train_path)
            baseline_vol = train_df['BTC_Close'].pct_change().std()
            print(f"  Circuit Breaker calibrated on TRAIN data. Baseline Vol: {baseline_vol:.5f}")
        else:
            baseline_vol = 0.01  # Conservative fallback
            print(f"  [WARNING] train_path not provided. Using default baseline_vol: {baseline_vol:.5f}")

        # Create test environment
        env = CryptoTradingEnv(
            parquet_path=test_path,
            window_size=self.config.window_size,
            commission=0.0004,  # churn_analysis.yaml
            episode_length=None,  # Full episode
            random_start=False,
            target_volatility=self.config.target_volatility,
            vol_window=self.config.vol_window,
            max_leverage=self.config.max_leverage,
            price_column='BTC_Close',
        )

        # Wrap with Risk Management (Circuit Breaker)
        # baseline_vol computed from TRAIN data to avoid data leakage
        env = RiskManagementWrapper(
            env,
            vol_window=24,
            vol_threshold=3.0,
            max_drawdown=0.10,  # 10% drawdown trigger
            cooldown_steps=12,
            augment_obs=False,
            baseline_vol=baseline_vol,
        )

        # Load agent
        model = TQC.load(tqc_path)

        # Run evaluation
        obs, _ = env.reset()
        done = False
        total_reward = 0
        rewards = []
        navs = []
        circuit_breaker_count = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += reward
            rewards.append(reward)
            if 'nav' in info:
                navs.append(info['nav'])
            if info.get('circuit_breaker'):
                circuit_breaker_count += 1

        # Calculate metrics - SKIP context_rows (warmup period)
        all_rewards = np.array(rewards)
        all_navs = np.array(navs) if navs else np.array([10000])

        # Skip warmup rows for metrics calculation (with safety check)
        if len(all_rewards) <= context_rows:
            print(f"  [WARNING] Test data ({len(all_rewards)}) shorter than warmup ({context_rows}). Using full data.")
            rewards = all_rewards
            navs = all_navs
        else:
            rewards = all_rewards[context_rows:]
            navs = all_navs[context_rows:]

        print(f"  Total steps: {len(all_rewards)}, Metrics computed on: {len(rewards)} steps")

        # Sharpe ratio (annualized)
        if len(rewards) > 1 and rewards.std() > 0:
            sharpe = (rewards.mean() / rewards.std()) * np.sqrt(8760)  # Hourly to annual
        else:
            sharpe = 0.0

        # PnL - compute from actual test period (after warmup)
        pnl = navs[-1] - navs[0] if len(navs) > 1 else 0
        pnl_pct = (pnl / navs[0]) * 100 if navs[0] > 0 else 0

        # Max Drawdown - on actual test period only
        if len(navs) > 1:
            peak = np.maximum.accumulate(navs)
            drawdown = (peak - navs) / peak
            max_drawdown = drawdown.max() * 100
        else:
            max_drawdown = 0.0

        # Number of trades (total from env, not split by warmup)
        total_trades = info.get('total_trades', 0)

        # ==================== BENCHMARK: Buy & Hold ====================
        # Load price series for same period
        test_df = pd.read_parquet(test_path)
        prices = test_df['BTC_Close'].values

        # Align with agent evaluation period (skip warmup, match length)
        # Agent starts at window_size, then we skip context_rows
        price_start_idx = self.config.window_size + context_rows
        price_end_idx = price_start_idx + len(navs)
        if price_end_idx > len(prices):
            price_end_idx = len(prices)

        bnh_prices = prices[price_start_idx:price_end_idx]

        if len(bnh_prices) > 1:
            # B&H Return
            bnh_return = (bnh_prices[-1] - bnh_prices[0]) / bnh_prices[0]
            bnh_pct = bnh_return * 100

            # B&H NAV (normalized to 10,000 like agent)
            bnh_navs = (bnh_prices / bnh_prices[0]) * 10000

            # Alpha = Agent PnL - B&H PnL
            alpha = pnl_pct - bnh_pct

            # Market Regime
            if bnh_return > 0.15:
                market_regime = "BULLISH 🟢"
            elif bnh_return < -0.15:
                market_regime = "BEARISH 🔴"
            else:
                market_regime = "RANGING ⚪"

            # Correlation (agent returns vs market returns)
            if len(navs) > 10:
                agent_returns = np.diff(navs) / navs[:-1]
                market_returns = np.diff(bnh_prices) / bnh_prices[:-1]
                min_len = min(len(agent_returns), len(market_returns))
                if min_len > 10:
                    correlation = np.corrcoef(agent_returns[:min_len], market_returns[:min_len])[0, 1]
                else:
                    correlation = 0.0
            else:
                correlation = 0.0
        else:
            bnh_pct = 0.0
            bnh_navs = np.array([10000])
            alpha = pnl_pct
            market_regime = "UNKNOWN"
            correlation = 0.0

        print(f"    B&H Return: {bnh_pct:+.2f}%")
        print(f"    Alpha: {alpha:+.2f}%")
        print(f"    Market: {market_regime}")

        metrics = {
            'segment_id': segment_id,
            'total_reward': float(rewards.sum()),  # Sum of actual test period
            'sharpe': sharpe,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'circuit_breakers': circuit_breaker_count,
            'final_nav': navs[-1] if len(navs) > 0 else 10000,
            'test_rows': len(rewards),
            'context_rows': context_rows,
            # Benchmark metrics
            'bnh_pct': bnh_pct,
            'bnh_navs': bnh_navs,  # For plotting overlay
            'alpha': alpha,
            'market_regime': market_regime,
            'correlation': correlation,
        }

        # Merge training diagnostics if provided
        if train_metrics:
            metrics['train_action_sat'] = train_metrics.get('action_saturation', 0.0)
            metrics['train_entropy'] = train_metrics.get('avg_entropy', 0.0)
            metrics['train_critic_loss'] = train_metrics.get('avg_critic_loss', 0.0)
            metrics['train_actor_loss'] = train_metrics.get('avg_actor_loss', 0.0)
            metrics['churn_ratio'] = train_metrics.get('avg_churn_ratio', 0.0)

        print(f"  Results:")
        print(f"    Sharpe: {sharpe:.2f}")
        print(f"    PnL: {pnl_pct:+.2f}%")
        print(f"    Max DD: {max_drawdown:.2f}%")
        print(f"    Trades: {total_trades}")
        print(f"    Circuit Breakers: {circuit_breaker_count}")

        # TensorBoard logging for evaluation
        from torch.utils.tensorboard import SummaryWriter
        eval_log_dir = f"logs/wfo/eval/segment_{segment_id}"
        os.makedirs(eval_log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=eval_log_dir)

        # Log scalar metrics
        writer.add_scalar("eval/sharpe", sharpe, segment_id)
        writer.add_scalar("eval/pnl_pct", pnl_pct, segment_id)
        writer.add_scalar("eval/max_drawdown", max_drawdown, segment_id)
        writer.add_scalar("eval/total_trades", total_trades, segment_id)
        writer.add_scalar("eval/circuit_breakers", circuit_breaker_count, segment_id)
        writer.add_scalar("eval/final_nav", navs[-1] if len(navs) > 0 else 10000, segment_id)

        # Log NAV curve as figure
        if len(navs) > 10:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(navs, color='blue', linewidth=1)
            ax.axhline(y=10000, color='gray', linestyle='--', alpha=0.5)
            ax.set_xlabel('Step')
            ax.set_ylabel('NAV')
            ax.set_title(f'Segment {segment_id} - NAV Curve (PnL: {pnl_pct:+.2f}%)')
            ax.grid(True, alpha=0.3)
            writer.add_figure("eval/nav_curve", fig, segment_id)
            plt.close(fig)

        writer.close()
        print(f"  TensorBoard logs: {eval_log_dir}")

        # Cleanup
        del model, env
        torch.cuda.empty_cache()
        gc.collect()

        return metrics, navs

    def run_segment(
        self,
        df_raw: pd.DataFrame,
        segment: Dict[str, int],
        use_batch_env: bool = False,
        resume: bool = False
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Run full WFO pipeline for a single segment.

        Args:
            df_raw: Raw DataFrame with OHLCV data.
            segment: Segment boundaries dict.
            use_batch_env: If True, use GPU-accelerated BatchCryptoEnv for TQC.
            resume: If True, resume TQC training from tqc_last.zip.

        Returns:
            Tuple of (metrics dict, train_metrics dict) for this segment.
        """
        segment_id = segment['id']
        print("\n" + "=" * 70)
        print(f"SEGMENT {segment_id}")
        print(f"Train: rows {segment['train_start']} - {segment['train_end']}")
        print(f"Test:  rows {segment['test_start']} - {segment['test_end']}")
        print("=" * 70)

        # 1. Preprocessing (Feature Engineering + Leak-Free Scaling)
        train_df, test_df, scaler = self.preprocess_segment(df_raw, segment)

        # 2. HMM Training (returns context_rows for env warmup)
        train_df, test_df, hmm, context_rows = self.train_hmm(train_df, test_df, segment_id)

        # 3. Save preprocessed data (with context buffer metadata)
        train_path, test_path = self.save_segment_data(
            train_df, test_df, scaler, segment_id, context_rows
        )

        # Cleanup dataframes
        del train_df, test_df, hmm
        gc.collect()

        # 4. MAE Training
        encoder_path = self.train_mae(train_path, segment_id)

        # 5. TQC Training
        tqc_path, train_metrics = self.train_tqc(
            train_path, encoder_path, segment_id,
            use_batch_env=use_batch_env,
            resume=resume
        )

        # 6. Organize Artifacts (Swap & Archive) - BEFORE evaluation
        # This ensures tqc.zip=Best and tqc_last.zip=Last exist on disk
        self._organize_artifacts(segment_id)

        # 7. Double Evaluation: Best Model + Last Model
        weights_dir = os.path.join(self.config.weights_dir, f"segment_{segment_id}")
        models_to_eval = [
            ('best', os.path.join(weights_dir, "tqc.zip")),
            ('last', os.path.join(weights_dir, "tqc_last.zip"))
        ]

        best_metrics = None
        for model_type, model_path in models_to_eval:
            if not os.path.exists(model_path):
                print(f"  [SKIP] {model_type} model not found: {model_path}")
                continue

            print(f"\n  [EVAL] Evaluating {model_type.upper()} model: {model_path}")

            # Evaluate
            metrics, navs = self.evaluate_segment(
                test_path, encoder_path, model_path, segment_id,
                context_rows=context_rows,
                train_metrics=train_metrics,
                train_path=train_path
            )

            # Tag results
            metrics['model_type'] = model_type

            # Generate diagnostic plots with prefix
            plot_path = self.generate_segment_plots(
                segment_id, navs, metrics, train_metrics, prefix=f"{model_type}_"
            )
            metrics['plot_path'] = plot_path

            # Save results
            self.save_results(metrics)

            # Keep best_metrics for return value and teacher report
            if model_type == 'best':
                best_metrics = metrics

        # 8. Teacher Report - Hyperparameter Hints (based on best model)
        if best_metrics:
            self._print_teacher_report(train_metrics, segment_id)

        return best_metrics if best_metrics else metrics, train_metrics

    def _organize_artifacts(self, segment_id: int) -> int:
        """
        Organize model artifacts with Swap & Archive logic.

        Called AFTER training, BEFORE evaluation to ensure both models exist.

        Artifact Strategy:
        - tqc.zip: BEST model (highest eval reward) - used for evaluation
        - tqc_last.zip: LAST model (end of training) - used for resume/extend

        Args:
            segment_id: ID of the segment to organize

        Returns:
            Total bytes freed from cleanup
        """
        import shutil

        freed_bytes = 0
        segment_weights_dir = Path(self.config.models_dir) / f"segment_{segment_id}"

        # === SWAP & ARCHIVE LOGIC ===
        tqc_path = segment_weights_dir / "tqc.zip"
        tqc_last_path = segment_weights_dir / "tqc_last.zip"
        best_model_path = segment_weights_dir / "best_model.zip"

        # Step A: Archive Last (tqc.zip -> tqc_last.zip)
        if tqc_path.exists():
            tqc_path.rename(tqc_last_path)
            print(f"  [ARCHIVE] tqc.zip -> tqc_last.zip (Resume point)")

        # Step B: Promote Best or Fallback
        if best_model_path.exists():
            # Promote best_model.zip to tqc.zip
            best_model_path.rename(tqc_path)
            print(f"  🏆 Optimization: 'tqc.zip' is now the BEST model (Eval). "
                  f"Last state archived as 'tqc_last.zip' (Resume).")
        elif tqc_last_path.exists():
            # Fallback: copy tqc_last.zip back to tqc.zip
            shutil.copy2(tqc_last_path, tqc_path)
            print(f"  ⚠️ No best_model found. 'tqc.zip' remains the LAST model.")

        # === CLEANUP INTERMEDIATE FILES ===

        # 1. Remove checkpoints/ directory (intermediate periodic saves)
        checkpoints_dir = segment_weights_dir / "checkpoints"
        if checkpoints_dir.exists():
            size = sum(f.stat().st_size for f in checkpoints_dir.rglob('*') if f.is_file())
            shutil.rmtree(checkpoints_dir)
            freed_bytes += size
            print(f"  [CLEANUP] Removed checkpoints/: {size / 1024 / 1024:.1f} MB")

        # 2. Remove mae_full.pth (encoder.pth is sufficient for inference)
        mae_full = segment_weights_dir / "mae_full.pth"
        if mae_full.exists():
            size = mae_full.stat().st_size
            mae_full.unlink()
            freed_bytes += size
            print(f"  [CLEANUP] Removed mae_full.pth: {size / 1024 / 1024:.1f} MB")

        if freed_bytes > 0:
            print(f"  [CLEANUP] Total freed segment {segment_id}: {freed_bytes / 1024 / 1024:.1f} MB")

        return freed_bytes

    def _print_teacher_report(self, train_metrics: Dict[str, Any], segment_id: int):
        """Print hyperparameter diagnostic hints based on training metrics."""
        print("\n" + "-" * 50)
        print(f"HYPERPARAMETER HINTS (Segment {segment_id})")
        print("-" * 50)

        issues_found = False

        # Check entropy (SAC/TQC uses log-entropy, negative values are normal)
        entropy = train_metrics.get('avg_entropy', 0.0)
        if entropy < -20:
            print(f"  [OVERFITTING] Entropy too low ({entropy:.2f})")
            print(f"    -> Increase ent_coef (current behavior is deterministic)")
            issues_found = True

        # Check action saturation
        action_sat = train_metrics.get('action_saturation', 0.0)
        if action_sat > 0.98:
            print(f"  [SATURATION] Agent stuck at extremes ({action_sat:.3f})")
            print(f"    -> Increase downside_coef or reduce reward_scaling")
            issues_found = True

        # Check churn ratio
        churn_ratio = train_metrics.get('avg_churn_ratio', 0.0)
        if churn_ratio > 0.3:
            print(f"  [HIGH CHURN] Trading eating profits ({churn_ratio:.3f})")
            print(f"    -> Increase churn_coef to penalize position changes")
            issues_found = True

        # Check critic loss
        critic_loss = train_metrics.get('avg_critic_loss', 0.0)
        if critic_loss > 1.0:
            print(f"  [INSTABILITY] High critic loss ({critic_loss:.4f})")
            print(f"    -> Reduce learning_rate or increase batch_size")
            issues_found = True

        if not issues_found:
            print("  [HEALTHY] All training metrics within normal range")

        print("-" * 50)

    def generate_segment_plots(
        self,
        segment_id: int,
        navs: np.ndarray,
        metrics: Dict[str, Any],
        train_metrics: Dict[str, Any],
        prefix: str = ""
    ) -> str:
        """
        Generate diagnostic plots for a segment and save as PNG.

        Args:
            segment_id: Segment identifier.
            navs: NAV history array.
            metrics: Evaluation metrics dict.
            train_metrics: Training metrics dict.
            prefix: Filename prefix (e.g., "best_" or "last_").

        Returns:
            Path to the generated PNG file.
        """
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Segment {segment_id} - Diagnostic Report", fontsize=14)

        # 1. NAV Curve with B&H Benchmark Overlay
        ax1 = axes[0, 0]
        ax1.plot(navs, color='blue', linewidth=1.5, label='Agent')

        # Overlay B&H benchmark if available
        bnh_navs = metrics.get('bnh_navs')
        if bnh_navs is not None and len(bnh_navs) > 1:
            # Align lengths (in case of mismatch)
            plot_len = min(len(navs), len(bnh_navs))
            ax1.plot(bnh_navs[:plot_len], color='gray', linestyle='--', linewidth=1, label='B&H', alpha=0.7)
            ax1.legend(loc='upper left')

        ax1.axhline(y=10000, color='black', linestyle=':', alpha=0.3)

        # Title with Alpha
        alpha = metrics.get('alpha', 0)
        alpha_color = 'green' if alpha > 0 else 'red'
        ax1.set_title(f"NAV vs B&H (Alpha: {alpha:+.2f}%)", color=alpha_color)
        ax1.set_xlabel("Step")
        ax1.set_ylabel("NAV")
        ax1.grid(True, alpha=0.3)

        # 2. Training Metrics Bar Chart
        ax2 = axes[0, 1]
        metric_names = ['action_sat', 'entropy', 'churn_ratio']
        metric_values = [
            train_metrics.get('action_saturation', 0),
            min(abs(train_metrics.get('avg_entropy', 0)) / 20, 1.5),  # Normalized, capped
            train_metrics.get('avg_churn_ratio', 0),
        ]
        thresholds = [0.98, 1.0, 0.3]
        colors = ['red' if v > t else 'green' for v, t in zip(metric_values, thresholds)]
        ax2.bar(metric_names, metric_values, color=colors)
        ax2.axhline(y=0.3, color='orange', linestyle='--', label='Churn threshold')
        ax2.axhline(y=0.98, color='red', linestyle='--', label='Saturation threshold')
        ax2.set_title("Training Health Metrics")
        ax2.set_ylim(0, 1.5)
        ax2.legend()

        # 3. Key Metrics Text
        ax3 = axes[1, 0]
        ax3.axis('off')

        # Extract benchmark metrics
        bnh_pct = metrics.get('bnh_pct', 0)
        alpha = metrics.get('alpha', 0)
        market_regime = metrics.get('market_regime', 'UNKNOWN')
        correlation = metrics.get('correlation', 0)

        text = f"""
    MARKET CONTEXT
    ==============
    Market: {market_regime}
    B&H Return: {bnh_pct:+.2f}%
    Correlation: {correlation:.2f}

    AGENT PERFORMANCE
    =================
    PnL: {metrics['pnl_pct']:+.2f}%
    Alpha: {alpha:+.2f}%
    Sharpe: {metrics['sharpe']:.2f}
    Max DD: {metrics['max_drawdown']:.2f}%
    Trades: {metrics['total_trades']}

    TRAINING HEALTH
    ===============
    Entropy: {train_metrics.get('avg_entropy', 0):.2f}
    Churn: {train_metrics.get('avg_churn_ratio', 0):.3f}
        """
        ax3.text(0.05, 0.5, text, fontsize=9, family='monospace', va='center')

        # 4. Pass/Fail Summary with Alpha Context
        ax4 = axes[1, 1]
        ax4.axis('off')

        issues = []
        positives = []

        # Training health checks
        if train_metrics.get('avg_entropy', 0) < -20:
            issues.append("OVERFITTING (entropy < -20)")
        if train_metrics.get('action_saturation', 0) > 0.98:
            issues.append("SATURATION (action_sat > 0.98)")
        if train_metrics.get('avg_churn_ratio', 0) > 0.3:
            issues.append("HIGH CHURN (ratio > 0.3)")

        # Performance checks (relative to benchmark)
        if alpha > 5:
            positives.append(f"STRONG ALPHA (+{alpha:.1f}%)")
        elif alpha > 0:
            positives.append(f"POSITIVE ALPHA (+{alpha:.1f}%)")
        elif alpha < -10:
            issues.append(f"UNDERPERFORM B&H ({alpha:.1f}%)")

        if metrics['sharpe'] > 1:
            positives.append(f"GOOD SHARPE ({metrics['sharpe']:.2f})")
        elif metrics['sharpe'] < -1:
            issues.append(f"BAD SHARPE ({metrics['sharpe']:.2f})")

        if metrics['pnl_pct'] < -20:
            issues.append("SEVERE LOSS (> 20%)")

        # Determine overall status
        if len(issues) >= 2 or (len(issues) == 1 and alpha < -5):
            status = "FAIL"
            color = 'red'
        elif len(positives) >= 2 or alpha > 5:
            status = "PASS"
            color = 'green'
        else:
            status = "MIXED"
            color = 'orange'

        # Build detail text
        detail_lines = []
        if positives:
            detail_lines.extend([f"  ✓ {p}" for p in positives[:2]])
        if issues:
            detail_lines.extend([f"  ✗ {i}" for i in issues[:2]])
        if not detail_lines:
            detail_lines.append("  Neutral performance")
        detail = "\n".join(detail_lines)

        ax4.text(0.5, 0.65, status, fontsize=36, ha='center', va='center',
                 color=color, fontweight='bold')
        ax4.text(0.5, 0.3, detail, fontsize=9, ha='center', va='center',
                 family='monospace')

        plt.tight_layout()

        # Save
        plot_dir = "results/plots"
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, f"segment_{segment_id}_{prefix}report.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"  [Plot] Saved: {plot_path}")
        return plot_path

    def _check_segment0_gate(self, metrics: Dict[str, Any], train_metrics: Dict[str, Any]) -> bool:
        """
        Check if segment 0 results are satisfactory to continue WFO.

        Returns:
            True if we should continue, False to stop.
        """
        print("\n" + "=" * 50)
        print("SEGMENT 0 GATE - Quality Check")
        print("=" * 50)

        issues = []

        # Training health checks
        if train_metrics.get('avg_entropy', 0) < -20:
            issues.append(f"Entropy too low: {train_metrics['avg_entropy']:.2f}")
        if train_metrics.get('action_saturation', 0) > 0.98:
            issues.append(f"Action saturation: {train_metrics['action_saturation']:.3f}")
        if train_metrics.get('avg_churn_ratio', 0) > 0.3:
            issues.append(f"Churn ratio: {train_metrics['avg_churn_ratio']:.3f}")

        # Performance checks
        if metrics['sharpe'] < -1:
            issues.append(f"Sharpe too negative: {metrics['sharpe']:.2f}")
        if metrics['pnl_pct'] < -30:
            issues.append(f"Severe loss: {metrics['pnl_pct']:.2f}%")

        if issues:
            print("RESULT: BLOCKED")
            print("\nIssues detected:")
            for issue in issues:
                print(f"  - {issue}")
            print("\nWFO stopped. Fix hyperparameters before continuing.")
            print("=" * 50)
            return False
        else:
            print("RESULT: PASSED")
            print("Segment 0 metrics are acceptable. Continuing WFO...")
            print("=" * 50)
            return True

    def save_results(self, metrics: Dict[str, Any]):
        """Append metrics to results CSV."""
        df = pd.DataFrame([metrics])

        # Add timestamp
        df['timestamp'] = datetime.now().isoformat()

        # Append or create
        if os.path.exists(self.config.results_path):
            df_existing = pd.read_csv(self.config.results_path)
            df = pd.concat([df_existing, df], ignore_index=True)

        df.to_csv(self.config.results_path, index=False)
        print(f"\nResults saved to: {self.config.results_path}")

    def _setup_logging(self):
        """Setup logging directory. Does NOT delete existing logs (parallel execution support)."""
        # NOTE: We intentionally do NOT delete logs here.
        # Parallel GPU runs would delete each other's logs.
        # User should manually clean logs/wfo if needed.

        logs_dir = "logs/wfo"
        # if os.path.exists(logs_dir):
        #     shutil.rmtree(logs_dir)  # DISABLED: breaks parallel execution
        os.makedirs(logs_dir, exist_ok=True)
        print(f"  [INFO] Appending to logs dir: {logs_dir}")

        print("  TensorBoard: start manually with 'tensorboard --logdir logs/wfo --port 8081'")

    def run(
        self,
        segment_ids: Optional[List[int]] = None,
        max_segments: Optional[int] = None
    ):
        """
        Run WFO pipeline.

        Args:
            segment_ids: Specific segments to run (if None, run all)
            max_segments: Maximum number of segments to run
        """
        print("\n" + "=" * 70)
        print("WALK-FORWARD OPTIMIZATION - Full Retrain Pipeline")
        print("=" * 70)

        # Setup logging and TensorBoard
        self._setup_logging()

        print(f"\nStart time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nConfiguration:")
        print(f"  Train: {self.config.train_months} months ({self.config.train_rows} rows)")
        print(f"  Test: {self.config.test_months} months ({self.config.test_rows} rows)")
        print(f"  Step: {self.config.step_months} months ({self.config.step_rows} rows)")
        print(f"  Volatility Scaling: Target={self.config.target_volatility}, Window={self.config.vol_window}")
        print(f"  GPU Acceleration: {'BatchCryptoEnv' if self.config.use_batch_env else 'SubprocVecEnv (CPU)'}")

        # Load raw data
        print(f"\nLoading raw data: {self.config.raw_data_path}")
        df_raw = pd.read_parquet(self.config.raw_data_path)
        print(f"  Shape: {df_raw.shape}")

        # Pre-calculate features globally (P1 optimization: ~20-30% speedup)
        print("\n[OPTIMIZATION] Pre-calculating features globally (once)...")
        self._df_features_global = self.feature_engineer.engineer_features(df_raw)
        print(f"  Features shape: {self._df_features_global.shape}")

        # Calculate segments
        segments = self.calculate_segments(len(df_raw))
        print(f"\nTotal segments: {len(segments)}")

        # Filter segments if specified
        if segment_ids is not None:
            segments = [s for s in segments if s['id'] in segment_ids]
        if max_segments is not None:
            segments = segments[:max_segments]

        print(f"Running segments: {[s['id'] for s in segments]}")

        # Run each segment
        all_metrics = []
        for segment in segments:
            try:
                metrics, train_metrics = self.run_segment(
                    df_raw, segment,
                    use_batch_env=self.config.use_batch_env,
                    resume=self.config.resume
                )
                all_metrics.append(metrics)
                # Note: save_results is now called inside run_segment for each model type

                # GATE: Disabled - let WFO run all segments regardless of segment 0 performance
                # if segment['id'] == 0:
                #     if not self._check_segment0_gate(metrics, train_metrics):
                #         print("\n[WFO] Stopped at segment 0 gate.")
                #         break

            except Exception as e:
                print(f"\n[ERROR] Segment {segment['id']} failed: {e}")
                import traceback
                traceback.print_exc()
                continue

            # Force garbage collection between segments
            gc.collect()

        # Summary
        print("\n" + "=" * 70)
        print("WFO COMPLETE - Summary")
        print("=" * 70)

        if all_metrics:
            df_results = pd.DataFrame(all_metrics)
            print(f"\nSegments completed: {len(all_metrics)}")
            print(f"Average Sharpe: {df_results['sharpe'].mean():.2f}")
            print(f"Average PnL: {df_results['pnl_pct'].mean():+.2f}%")
            print(f"Average Max DD: {df_results['max_drawdown'].mean():.2f}%")
            print(f"\nResults saved to: {self.config.results_path}")

        print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    def run_eval_only(
        self,
        segment_ids: List[int]
    ):
        """
        Re-evaluate segments using existing TQC models.
        Regenerates test data with context buffer and runs evaluation.

        Args:
            segment_ids: List of segment IDs to re-evaluate.
        """
        print("\n" + "=" * 70)
        print("WALK-FORWARD OPTIMIZATION - Eval-Only Mode")
        print("=" * 70)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Segments to evaluate: {segment_ids}")

        # Load raw data
        print(f"\nLoading raw data: {self.config.raw_data_path}")
        df_raw = pd.read_parquet(self.config.raw_data_path)
        print(f"  Shape: {df_raw.shape}")

        # Pre-calculate features globally (P1 optimization)
        print("\n[OPTIMIZATION] Pre-calculating features globally (once)...")
        self._df_features_global = self.feature_engineer.engineer_features(df_raw)
        print(f"  Features shape: {self._df_features_global.shape}")

        # Calculate all segments to get their boundaries
        all_segments = self.calculate_segments(len(df_raw))
        segments_by_id = {s['id']: s for s in all_segments}

        all_metrics = []
        for seg_id in segment_ids:
            if seg_id not in segments_by_id:
                print(f"\n[WARNING] Segment {seg_id} not found, skipping...")
                continue

            segment = segments_by_id[seg_id]
            print("\n" + "=" * 70)
            print(f"SEGMENT {seg_id} - Eval Only")
            print(f"Test: rows {segment['test_start']} - {segment['test_end']}")
            print("=" * 70)

            try:
                # 1. Preprocessing
                train_df, test_df, scaler = self.preprocess_segment(df_raw, segment)

                # 2. HMM (with context buffer)
                train_df, test_df, hmm, context_rows = self.train_hmm(train_df, test_df, seg_id)

                # 3. Save data (with metadata)
                train_path, test_path = self.save_segment_data(
                    train_df, test_df, scaler, seg_id, context_rows
                )

                # Cleanup
                del train_df, test_df, hmm
                gc.collect()

                # 4. Check encoder exists
                weights_dir = os.path.join(self.config.weights_dir, f"segment_{seg_id}")
                encoder_path = os.path.join(weights_dir, "encoder.pth")

                if not os.path.exists(encoder_path):
                    print(f"  [ERROR] Encoder not found: {encoder_path}")
                    continue

                # 5. Evaluate both Best and Last models (like run_segment)
                models_to_eval = [
                    ('best', os.path.join(weights_dir, "tqc.zip")),
                    ('last', os.path.join(weights_dir, "tqc_last.zip"))
                ]

                for model_type, tqc_path in models_to_eval:
                    if not os.path.exists(tqc_path):
                        print(f"  [SKIP] {model_type.upper()} model not found: {tqc_path}")
                        continue

                    print(f"\n  --- Evaluating {model_type.upper()} model ---")
                    print(f"  Using: {tqc_path}")

                    metrics, navs = self.evaluate_segment(
                        test_path, encoder_path, tqc_path, seg_id,
                        context_rows=context_rows,
                        train_metrics={},  # No training metrics in eval-only mode
                        train_path=train_path
                    )

                    # Tag metrics with model type
                    metrics['model_type'] = model_type

                    all_metrics.append(metrics)
                    self.save_results(metrics)

                    # Generate plots (with minimal train_metrics)
                    prefix = f"{model_type}_" if model_type != 'best' else ""
                    self.generate_segment_plots(
                        seg_id, navs, metrics,
                        {'action_saturation': 0, 'avg_entropy': 0, 'avg_critic_loss': 0, 'avg_churn_ratio': 0},
                        prefix=prefix
                    )

            except Exception as e:
                print(f"\n[ERROR] Segment {seg_id} failed: {e}")
                import traceback
                traceback.print_exc()
                continue

            gc.collect()

        # Summary
        print("\n" + "=" * 70)
        print("EVAL-ONLY COMPLETE")
        print("=" * 70)

        if all_metrics:
            df_results = pd.DataFrame(all_metrics)
            print(f"\nSegments evaluated: {len(all_metrics)}")
            print(f"Average Sharpe: {df_results['sharpe'].mean():.2f}")
            print(f"Average PnL: {df_results['pnl_pct'].mean():+.2f}%")

        print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# =============================================================================
# Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Walk-Forward Optimization")
    parser.add_argument("--raw-data", type=str, default="data/raw_training_data.parquet",
                        help="Path to raw OHLCV data")
    parser.add_argument("--segments", type=int, default=None,
                        help="Max number of segments to run")
    parser.add_argument("--segment", type=int, default=None,
                        help="Run specific segment only")
    parser.add_argument("--timesteps", type=int, default=WFOConfig.tqc_timesteps,
                        help="TQC training timesteps per segment")
    parser.add_argument("--mae-epochs", type=int, default=WFOConfig.mae_epochs,
                        help="MAE training epochs per segment")
    parser.add_argument("--train-months", type=int, default=WFOConfig.train_months,
                        help="Training window in months")
    parser.add_argument("--test-months", type=int, default=WFOConfig.test_months,
                        help="Test window in months")
    parser.add_argument("--step-months", type=int, default=WFOConfig.step_months,
                        help="Rolling step in months")
    parser.add_argument("--eval-only", action="store_true",
                        help="Re-run evaluation only (skip MAE/TQC training)")
    parser.add_argument("--eval-segments", type=str, default=None,
                        help="Comma-separated segment IDs to evaluate (e.g., '0,1,2')")
    parser.add_argument("--no-batch-env", action="store_true",
                        help="Disable GPU-accelerated BatchCryptoEnv (Force CPU mode)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from tqc_last.zip (continues TensorBoard steps)")

    args = parser.parse_args()

    # Create config
    config = WFOConfig()
    config.raw_data_path = args.raw_data
    config.tqc_timesteps = args.timesteps
    config.mae_epochs = args.mae_epochs
    config.train_months = args.train_months
    config.test_months = args.test_months
    config.step_months = args.step_months
    # Default to GPU (True) unless explicitly disabled via CLI
    # This prevents accidental CPU runs which are 100x slower
    if args.no_batch_env:
        print("[WARNING] GPU Acceleration DISABLED via --no-batch-env flag")
        config.use_batch_env = False
    else:
        config.use_batch_env = True

    # Resume mode: continue from tqc_last.zip
    if args.resume:
        print("[INFO] RESUME MODE: Will continue from tqc_last.zip (TensorBoard continues)")
        config.resume = True

    # Create pipeline
    pipeline = WFOPipeline(config)

    # Run (normal or eval-only mode)
    if args.eval_only:
        if args.eval_segments is None:
            print("[ERROR] --eval-only requires --eval-segments (e.g., --eval-segments 0,1,2)")
            sys.exit(1)
        segment_ids = [int(x.strip()) for x in args.eval_segments.split(',')]
        pipeline.run_eval_only(segment_ids)
    else:
        segment_ids = [args.segment] if args.segment is not None else None
        pipeline.run(segment_ids=segment_ids, max_segments=args.segments)


if __name__ == "__main__":
    main()

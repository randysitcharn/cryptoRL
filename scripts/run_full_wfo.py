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
    output_dir: str = "data/wfo"
    models_dir: str = "models/wfo"
    weights_dir: str = "weights/wfo"
    results_path: str = "results/wfo_results.csv"

    # WFO Parameters
    train_months: int = 12
    test_months: int = 3
    step_months: int = 3  # Rolling step
    hours_per_month: int = 720  # 30 days * 24 hours

    # Training Parameters
    mae_epochs: int = 90
    tqc_timesteps: int = 250_000

    # TQC Hyperparameters (aggressive regularization)
    learning_rate: float = 9e-5
    batch_size: int = 512  # Balanced batch size
    gamma: float = 0.95
    ent_coef: Union[str, float] = "auto"  # Auto entropy tuning
    churn_coef: float = 0.0  # Disabled: smooth_coef handles position smoothing
    smooth_coef: float = 0.1  # Reduced smoothness penalty

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

        # 1. Extract full segment (train + test) from raw data
        full_start = segment['train_start']
        full_end = segment['test_end']

        # Add buffer for rolling calculations (720h = 30 days max window)
        buffer = 720
        safe_start = max(0, full_start - buffer)

        df_segment = df_raw.iloc[safe_start:full_end].copy()
        print(f"  Raw segment: rows {safe_start} to {full_end} ({len(df_segment)} rows)")

        # 2. Feature Engineering on full segment
        print("  Applying feature engineering...")
        df_features = self.feature_engineer.engineer_features(df_segment)

        # 3. Adjust indices after feature engineering (NaN dropped)
        # Find the offset caused by dropna
        offset = full_start - safe_start
        actual_train_start = max(0, offset - (len(df_segment) - len(df_features)))

        # Calculate relative indices
        train_len = segment['train_end'] - segment['train_start']
        test_len = segment['test_end'] - segment['test_start']

        # Split into train and test
        # After feature engineering, we need to find where train/test boundaries are
        total_needed = train_len + test_len
        if len(df_features) < total_needed:
            print(f"  [WARNING] Not enough data after feature engineering: {len(df_features)} < {total_needed}")
            # Use what we have
            train_df = df_features.iloc[:train_len].copy()
            test_df = df_features.iloc[train_len:].copy()
        else:
            # Take from the end to ensure we have the most recent data
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
        segment_id: int
    ) -> tuple[str, Dict[str, Any]]:
        """
        Train TQC agent on segment train data.

        Returns:
            Tuple of (path to saved agent, training metrics dict).
        """
        print(f"\n[Segment {segment_id}] Training TQC...")

        # Import here to avoid circular imports
        import torch
        from src.training.train_agent import train, TrainingConfig

        # Configure TQC training
        config = TrainingConfig()
        config.data_path = train_path
        config.encoder_path = encoder_path
        config.total_timesteps = self.config.tqc_timesteps
        config.learning_rate = self.config.learning_rate
        config.batch_size = self.config.batch_size
        config.gamma = self.config.gamma
        config.ent_coef = self.config.ent_coef
        config.churn_coef = self.config.churn_coef
        config.smooth_coef = self.config.smooth_coef
        config.target_volatility = self.config.target_volatility
        config.vol_window = self.config.vol_window
        config.max_leverage = self.config.max_leverage

        # Set segment-specific paths
        weights_dir = os.path.join(self.config.weights_dir, f"segment_{segment_id}")
        config.save_path = os.path.join(weights_dir, "tqc.zip")
        config.checkpoint_dir = os.path.join(weights_dir, "checkpoints/")
        config.tensorboard_log = f"logs/wfo/segment_{segment_id}/"
        config.name = f"WFO_seg{segment_id}"

        # We don't have separate eval data in WFO mode
        # Use train data for eval (not ideal but necessary)
        config.eval_data_path = train_path

        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.tensorboard_log, exist_ok=True)

        # Train - now returns (model, train_metrics)
        model, train_metrics = train(config)

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
            window_size=64,
            commission=0.0004,  # churn_analysis.yaml
            episode_length=None,  # Full episode
            random_start=False,
            target_volatility=self.config.target_volatility,
            vol_window=self.config.vol_window,
            max_leverage=self.config.max_leverage,
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

        # Skip warmup rows for metrics calculation
        rewards = all_rewards[context_rows:] if len(all_rewards) > context_rows else all_rewards
        navs = all_navs[context_rows:] if len(all_navs) > context_rows else all_navs

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
        segment: Dict[str, int]
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Run full WFO pipeline for a single segment.

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
        tqc_path, train_metrics = self.train_tqc(train_path, encoder_path, segment_id)

        # 6. Evaluation (skip context_rows in metrics)
        metrics, navs = self.evaluate_segment(
            test_path, encoder_path, tqc_path, segment_id,
            context_rows=context_rows,
            train_metrics=train_metrics,
            train_path=train_path
        )

        # 7. Teacher Report - Hyperparameter Hints
        self._print_teacher_report(train_metrics, segment_id)

        # 8. Generate diagnostic plots (PNG for visual analysis)
        plot_path = self.generate_segment_plots(segment_id, navs, metrics, train_metrics)
        metrics['plot_path'] = plot_path

        return metrics, train_metrics

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
        train_metrics: Dict[str, Any]
    ) -> str:
        """
        Generate diagnostic plots for a segment and save as PNG.

        Returns:
            Path to the generated PNG file.
        """
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Segment {segment_id} - Diagnostic Report", fontsize=14)

        # 1. NAV Curve
        ax1 = axes[0, 0]
        ax1.plot(navs, color='blue', linewidth=1)
        ax1.axhline(y=10000, color='gray', linestyle='--', alpha=0.5)
        ax1.set_title(f"NAV Curve (Final: {navs[-1]:.0f}, PnL: {metrics['pnl_pct']:+.2f}%)")
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
        text = f"""
    EVALUATION METRICS
    ==================
    Sharpe Ratio: {metrics['sharpe']:.2f}
    PnL: {metrics['pnl_pct']:+.2f}%
    Max Drawdown: {metrics['max_drawdown']:.2f}%
    Total Trades: {metrics['total_trades']}

    TRAINING DIAGNOSTICS
    ====================
    Action Saturation: {train_metrics.get('action_saturation', 0):.3f}
    Avg Entropy: {train_metrics.get('avg_entropy', 0):.4f}
    Avg Critic Loss: {train_metrics.get('avg_critic_loss', 0):.4f}
    Churn Ratio: {train_metrics.get('avg_churn_ratio', 0):.3f}
        """
        ax3.text(0.1, 0.5, text, fontsize=10, family='monospace', va='center')

        # 4. Pass/Fail Summary
        ax4 = axes[1, 1]
        ax4.axis('off')

        issues = []
        if train_metrics.get('avg_entropy', 0) < -20:
            issues.append("OVERFITTING (entropy < -20)")
        if train_metrics.get('action_saturation', 0) > 0.98:
            issues.append("SATURATION (action_sat > 0.98)")
        if train_metrics.get('avg_churn_ratio', 0) > 0.3:
            issues.append("HIGH CHURN (ratio > 0.3)")
        if metrics['sharpe'] < 0:
            issues.append("NEGATIVE SHARPE")
        if metrics['pnl_pct'] < -20:
            issues.append("SEVERE LOSS (> 20%)")

        if issues:
            status = "FAIL"
            color = 'red'
            detail = "\n".join(f"  - {issue}" for issue in issues)
        else:
            status = "PASS"
            color = 'green'
            detail = "  All metrics within acceptable range"

        ax4.text(0.5, 0.6, status, fontsize=40, ha='center', va='center',
                 color=color, fontweight='bold')
        ax4.text(0.5, 0.3, detail, fontsize=10, ha='center', va='center',
                 family='monospace')

        plt.tight_layout()

        # Save
        plot_dir = "results/plots"
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, f"segment_{segment_id}_report.png")
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
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nConfiguration:")
        print(f"  Train: {self.config.train_months} months ({self.config.train_rows} rows)")
        print(f"  Test: {self.config.test_months} months ({self.config.test_rows} rows)")
        print(f"  Step: {self.config.step_months} months ({self.config.step_rows} rows)")
        print(f"  Volatility Scaling: Target={self.config.target_volatility}, Window={self.config.vol_window}")

        # Load raw data
        print(f"\nLoading raw data: {self.config.raw_data_path}")
        df_raw = pd.read_parquet(self.config.raw_data_path)
        print(f"  Shape: {df_raw.shape}")

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
                metrics, train_metrics = self.run_segment(df_raw, segment)
                all_metrics.append(metrics)
                self.save_results(metrics)

                # GATE: Check segment 0 before continuing
                if segment['id'] == 0:
                    if not self._check_segment0_gate(metrics, train_metrics):
                        print("\n[WFO] Stopped at segment 0 gate.")
                        break

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

                # 4. Check if TQC model exists
                tqc_path = os.path.join(self.config.weights_dir, f"segment_{seg_id}", "tqc.zip")
                encoder_path = os.path.join(self.config.weights_dir, f"segment_{seg_id}", "encoder.pth")

                if not os.path.exists(tqc_path):
                    print(f"  [ERROR] TQC model not found: {tqc_path}")
                    continue
                if not os.path.exists(encoder_path):
                    print(f"  [ERROR] Encoder not found: {encoder_path}")
                    continue

                print(f"  Using existing models: {tqc_path}")

                # 5. Evaluate
                metrics, navs = self.evaluate_segment(
                    test_path, encoder_path, tqc_path, seg_id,
                    context_rows=context_rows,
                    train_metrics={},  # No training metrics in eval-only mode
                    train_path=train_path
                )

                all_metrics.append(metrics)
                self.save_results(metrics)

                # Generate plots (with minimal train_metrics)
                self.generate_segment_plots(seg_id, navs, metrics, {
                    'action_saturation': 0, 'avg_entropy': 0,
                    'avg_critic_loss': 0, 'avg_churn_ratio': 0
                })

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

    args = parser.parse_args()

    # Create config
    config = WFOConfig()
    config.raw_data_path = args.raw_data
    config.tqc_timesteps = args.timesteps
    config.mae_epochs = args.mae_epochs
    config.train_months = args.train_months
    config.test_months = args.test_months
    config.step_months = args.step_months

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

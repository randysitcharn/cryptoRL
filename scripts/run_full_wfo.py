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
import pickle
import argparse
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

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
    mae_epochs: int = 70
    tqc_timesteps: int = 150_000  # Reduced to prevent overfitting

    # TQC Hyperparameters (aggressive regularization)
    learning_rate: float = 6e-5
    batch_size: int = 1024  # Larger batch for gradient smoothing
    gamma: float = 0.95

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
    ) -> tuple[pd.DataFrame, pd.DataFrame, RegimeDetector]:
        """
        Train HMM on train data and predict on both.

        Returns:
            (train_with_hmm, test_with_hmm, detector)
        """
        print(f"\n[Segment {segment_id}] Training HMM...")

        # Initialize detector
        detector = RegimeDetector(n_components=4)

        # Fit on train with TensorBoard logging
        train_with_hmm = detector.fit_predict(
            train_df,
            tensorboard_log="logs/wfo/hmm/",
            run_name=f"segment_{segment_id}"
        )

        # Predict on test (no retraining)
        test_with_hmm = detector.predict(test_df)

        # Save HMM
        hmm_dir = os.path.join(self.config.models_dir, f"segment_{segment_id}")
        os.makedirs(hmm_dir, exist_ok=True)
        hmm_path = os.path.join(hmm_dir, "hmm.pkl")
        detector.save(hmm_path)

        # Remove HMM intermediate features (keep only Prob_*)
        hmm_features = ['HMM_Trend', 'HMM_Vol', 'HMM_Momentum']
        train_final = train_with_hmm.drop(columns=hmm_features, errors='ignore')
        test_final = test_with_hmm.drop(columns=hmm_features, errors='ignore')

        print(f"  Train final: {train_final.shape}, Test final: {test_final.shape}")

        return train_final, test_final, detector

    def save_segment_data(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        scaler: RobustScaler,
        segment_id: int
    ):
        """Save preprocessed data and scaler for a segment."""

        # Data directory
        data_dir = os.path.join(self.config.output_dir, f"segment_{segment_id}")
        os.makedirs(data_dir, exist_ok=True)

        # Save parquet files
        train_path = os.path.join(data_dir, "train.parquet")
        test_path = os.path.join(data_dir, "test.parquet")
        train_df.to_parquet(train_path, index=False)
        test_df.to_parquet(test_path, index=False)

        # Save scaler
        models_dir = os.path.join(self.config.models_dir, f"segment_{segment_id}")
        os.makedirs(models_dir, exist_ok=True)
        scaler_path = os.path.join(models_dir, "scaler.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump({'scaler': scaler, 'columns': list(train_df.columns)}, f)

        print(f"  Saved: {train_path}, {test_path}, {scaler_path}")

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
    ) -> str:
        """
        Train TQC agent on segment train data.

        Returns:
            Path to saved agent.
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

        # Train
        model = train(config)

        print(f"  TQC trained. Saved: {config.save_path}")

        # Cleanup
        del model
        torch.cuda.empty_cache()
        gc.collect()

        return config.save_path

    def evaluate_segment(
        self,
        test_path: str,
        encoder_path: str,
        tqc_path: str,
        segment_id: int
    ) -> Dict[str, Any]:
        """
        Evaluate trained agent on test data.

        Returns:
            Dict with metrics (sharpe, pnl, drawdown, etc.)
        """
        print(f"\n[Segment {segment_id}] Evaluating...")

        import torch
        from sb3_contrib import TQC
        from src.training.env import CryptoTradingEnv

        # Create test environment
        env = CryptoTradingEnv(
            parquet_path=test_path,
            window_size=64,
            commission=0.0004,  # churn_analysis.yaml
            episode_length=None,  # Full episode
            random_start=False,
        )

        # Load agent
        model = TQC.load(tqc_path)

        # Run evaluation
        obs, _ = env.reset()
        done = False
        total_reward = 0
        rewards = []
        navs = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += reward
            rewards.append(reward)
            if 'nav' in info:
                navs.append(info['nav'])

        # Calculate metrics
        rewards = np.array(rewards)
        navs = np.array(navs) if navs else np.array([10000])

        # Sharpe ratio (annualized)
        if len(rewards) > 1 and rewards.std() > 0:
            sharpe = (rewards.mean() / rewards.std()) * np.sqrt(8760)  # Hourly to annual
        else:
            sharpe = 0.0

        # PnL
        pnl = navs[-1] - navs[0] if len(navs) > 1 else 0
        pnl_pct = (pnl / navs[0]) * 100 if navs[0] > 0 else 0

        # Max Drawdown
        if len(navs) > 1:
            peak = np.maximum.accumulate(navs)
            drawdown = (peak - navs) / peak
            max_drawdown = drawdown.max() * 100
        else:
            max_drawdown = 0.0

        # Number of trades
        total_trades = info.get('total_trades', 0)

        metrics = {
            'segment_id': segment_id,
            'total_reward': total_reward,
            'sharpe': sharpe,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'final_nav': navs[-1] if len(navs) > 0 else 10000,
            'test_rows': len(rewards),
        }

        print(f"  Results:")
        print(f"    Sharpe: {sharpe:.2f}")
        print(f"    PnL: {pnl_pct:+.2f}%")
        print(f"    Max DD: {max_drawdown:.2f}%")
        print(f"    Trades: {total_trades}")

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

        return metrics

    def run_segment(
        self,
        df_raw: pd.DataFrame,
        segment: Dict[str, int]
    ) -> Dict[str, Any]:
        """
        Run full WFO pipeline for a single segment.

        Returns:
            Metrics dict for this segment.
        """
        segment_id = segment['id']
        print("\n" + "=" * 70)
        print(f"SEGMENT {segment_id}")
        print(f"Train: rows {segment['train_start']} - {segment['train_end']}")
        print(f"Test:  rows {segment['test_start']} - {segment['test_end']}")
        print("=" * 70)

        # 1. Preprocessing (Feature Engineering + Leak-Free Scaling)
        train_df, test_df, scaler = self.preprocess_segment(df_raw, segment)

        # 2. HMM Training
        train_df, test_df, hmm = self.train_hmm(train_df, test_df, segment_id)

        # 3. Save preprocessed data
        train_path, test_path = self.save_segment_data(train_df, test_df, scaler, segment_id)

        # Cleanup dataframes
        del train_df, test_df, hmm
        gc.collect()

        # 4. MAE Training
        encoder_path = self.train_mae(train_path, segment_id)

        # 5. TQC Training
        tqc_path = self.train_tqc(train_path, encoder_path, segment_id)

        # 6. Evaluation
        metrics = self.evaluate_segment(test_path, encoder_path, tqc_path, segment_id)

        return metrics

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
                metrics = self.run_segment(df_raw, segment)
                all_metrics.append(metrics)
                self.save_results(metrics)
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
    parser.add_argument("--timesteps", type=int, default=300_000,
                        help="TQC training timesteps per segment")
    parser.add_argument("--mae-epochs", type=int, default=70,
                        help="MAE training epochs per segment")
    parser.add_argument("--train-months", type=int, default=12,
                        help="Training window in months")
    parser.add_argument("--test-months", type=int, default=3,
                        help="Test window in months")
    parser.add_argument("--step-months", type=int, default=3,
                        help="Rolling step in months")

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

    # Run
    segment_ids = [args.segment] if args.segment is not None else None
    pipeline.run(segment_ids=segment_ids, max_segments=args.segments)


if __name__ == "__main__":
    main()

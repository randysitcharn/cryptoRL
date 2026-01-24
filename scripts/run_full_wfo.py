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
    python scripts/run_full_wfo.py --clean              # Nettoyer artefacts WFO puis quitter
    python scripts/run_full_wfo.py --clean --clean-dry-run  # Simuler le nettoyage
"""

import os
import subprocess
import sys
import gc
import json
import pickle
import shutil
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

# Add project root to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.data_engineering.features import FeatureEngineer
from src.data_engineering.manager import RegimeDetector
from src.data_engineering.splitter import validate_purge_window
from src.data_engineering.processor import DataProcessor
from src.config import WFOTrainingConfig


# =============================================================================
# Configuration
# =============================================================================

# MORL: w_cost values for Pareto front evaluation (Audit 2026-01-22)
# 5 points provide good resolution: scalping -> B&H continuum
EVAL_W_COST_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]
DEFAULT_EVAL_W_COST = 0.5  # Balanced mode for standard evaluation


@dataclass
class WFOConfig:
    """Walk-Forward Optimization configuration."""

    # Data
    raw_data_path: str = "data/raw_historical/multi_asset_historical.csv"

    # GPU Acceleration
    use_batch_env: bool = True  # Use BatchCryptoEnv for GPU-accelerated training (default ON)
    resume: bool = False  # Resume TQC training from tqc_last.zip
    output_dir: str = "data/wfo"
    models_dir: str = "models/wfo"
    weights_dir: str = "weights/wfo"
    results_path: str = "results/wfo_results.csv"

    # WFO Parameters
    train_months: int = 13       # Training data (excluding eval)
    eval_months: int = 1         # In-train evaluation (last month of train window)
    test_months: int = 3         # Out-of-sample test
    step_months: int = 3         # Rolling step
    hours_per_month: int = 720   # 30 days * 24 hours
    window_size: int = 64        # Observation window for transformer encoder

    # Purge & Embargo (Leak Prevention) - See audit P0.2
    purge_window: int = 720      # Gap between train and eval/test (>= max indicator window)
    embargo_window: int = 24     # Gap after test before next train (label correlation decay)

    # Training Parameters
    mae_epochs: int = 90
    
    # === TQC Training Configuration (centralized in WFOTrainingConfig) ===
    # All TQC hyperparameters are now in src/config/training.py:WFOTrainingConfig
    # This ensures single source of truth and eliminates config divergence.
    # See docs/design/WFO_CONFIG_RATIONALE.md for parameter rationale.
    training_config: WFOTrainingConfig = field(default_factory=WFOTrainingConfig)

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
        # Belief States HMM (probabilités filtrées Forward-Only, déjà dans [0, 1])
        'HMM_Prob_0', 'HMM_Prob_1', 'HMM_Prob_2', 'HMM_Prob_3',
        # Entropie HMM (incertitude du régime, déjà normalisée)
        'HMM_Entropy',
    ])

    # === Overfitting Guard parameters are now in WFOTrainingConfig ===
    # Access via: self.training_config.use_overfitting_guard, etc.

    # === Fail-over Strategy (Gestion des échecs) ===
    use_checkpoint_on_failure: bool = True  # Tenter recovery via checkpoint
    min_completion_ratio: float = 0.30      # Min 30% du training pour recovery
    fallback_strategy: str = 'flat'         # 'flat' ou 'buy_and_hold'

    # === Chain of Inheritance (Continuité WFO) ===
    use_warm_start: bool = False            # MORL: Cold start par défaut (architecture change)
    pretrained_model_path: Optional[str] = None  # Modèle de départ pour Seg 0
    cleanup_failed_checkpoints: bool = True # Supprimer checkpoints des segments FAILED

    # === Ensemble RL (Design Doc 2026-01-22) ===
    # Reference: docs/design/ENSEMBLE_RL_DESIGN.md
    use_ensemble: bool = False
    ensemble_n_members: int = 3
    ensemble_seeds: List[int] = field(default_factory=lambda: [42, 123, 456])
    ensemble_aggregation: str = 'confidence'  # 'mean', 'median', 'confidence', 'conservative', 'pessimistic_bound'
    ensemble_parallel: bool = True  # Train on multiple GPUs

    @property
    def train_rows(self) -> int:
        """Training rows (excluding eval)."""
        return self.train_months * self.hours_per_month

    @property
    def eval_rows(self) -> int:
        """In-train evaluation rows."""
        return self.eval_months * self.hours_per_month

    @property
    def test_rows(self) -> int:
        """Out-of-sample test rows."""
        return self.test_months * self.hours_per_month

    @property
    def step_rows(self) -> int:
        """Rolling step size."""
        return self.step_months * self.hours_per_month

    @property
    def full_train_rows(self) -> int:
        """Full train window (train + eval) for scaler fitting."""
        return self.train_rows + self.eval_rows


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

    def _load_raw_data(self, path: str) -> pd.DataFrame:
        """
        Smart loader for raw data files (CSV or Parquet).

        Handles:
        - CSV: Auto-parses dates, sets DatetimeIndex
        - Parquet: Direct load

        Args:
            path: Path to data file (.csv or .parquet)

        Returns:
            DataFrame with sorted DatetimeIndex
        """
        if path.endswith('.csv'):
            print(f"  [INFO] Detected CSV input. Auto-parsing dates...")
            df = pd.read_csv(path, index_col=0, parse_dates=True)

            # Safety check: ensure DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                # Try to find a date/timestamp column
                date_cols = [c for c in df.columns if c.lower() in ('date', 'timestamp', 'datetime', 'time')]
                if date_cols:
                    df[date_cols[0]] = pd.to_datetime(df[date_cols[0]])
                    df.set_index(date_cols[0], inplace=True)
                    print(f"  [INFO] Set '{date_cols[0]}' as DatetimeIndex")
                else:
                    raise ValueError(f"Could not find DatetimeIndex. Index type: {type(df.index)}")
        else:
            df = pd.read_parquet(path)

        # Ensure sorted index (required for WFO slicing)
        df.sort_index(inplace=True)

        # Final validation
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(f"Data must have DatetimeIndex, got {type(df.index)}")

        return df

    def calculate_segments(self, total_rows: int, warmup_offset: int = 0) -> List[Dict[str, int]]:
        """
        Calculate segment boundaries for rolling WFO with purge and embargo windows.

        Segment structure (Leak-Free):
            [train] [PURGE] [eval] [PURGE] [test] [EMBARGO]
            |<-train_months->|      |<-eval->|      |<-test->|

        - PURGE (720h): Gap between train→eval and eval→test to prevent indicator leakage
        - EMBARGO (24h): Gap after test before next segment's train (label correlation decay)

        Dynamic WFO Offset:
            If warmup_offset > 0, the first segment is shifted to accommodate feature
            engineering warmup (rolling windows create NaN at the start).
            This allows --train-months 36 without crashing on Segment 0.

        See audit DATA_PIPELINE_AUDIT_REPORT.md P0.2 for details.

        Args:
            total_rows: Total rows in raw dataset.
            warmup_offset: Rows lost at the start due to feature engineering (NaN dropped).
                          If provided, first segment starts at this offset.

        Returns:
            List of dicts with train_start, train_end, eval_start, eval_end, test_start, test_end
        """
        validate_purge_window(self.config.purge_window, raise_error=True)
        segments = []
        purge = self.config.purge_window
        embargo = self.config.embargo_window

        # Total segment size: train + purge + eval + purge + test + embargo
        segment_size = (
            self.config.train_rows +
            purge +  # Purge before eval
            self.config.eval_rows +
            purge +  # Purge before test
            self.config.test_rows +
            embargo  # Embargo after test
        )

        # Dynamic WFO Offset: Start at warmup_offset to ensure first segment has enough data
        start = warmup_offset
        segment_id = 0

        if warmup_offset > 0:
            print(f"[INFO] Dynamic WFO Offset: First segment shifted by {warmup_offset} rows "
                  f"to accommodate feature engineering warmup.")

        while start + segment_size <= total_rows:
            train_start = start
            train_end = start + self.config.train_rows
            # Purge gap after train
            eval_start = train_end + purge
            eval_end = eval_start + self.config.eval_rows
            # Purge gap after eval
            test_start = eval_end + purge
            test_end = test_start + self.config.test_rows
            # Embargo gap after test (accounted for in step)

            segments.append({
                'id': segment_id,
                'train_start': train_start,
                'train_end': train_end,
                'eval_start': eval_start,
                'eval_end': eval_end,
                'test_start': test_start,
                'test_end': test_end,
                'purge_window': purge,
                'embargo_window': embargo,
            })

            # Step includes embargo from previous segment
            start += self.config.step_rows
            segment_id += 1

        print(f"[WFO] Calculated {len(segments)} segments with purge={purge}h, embargo={embargo}h")
        return segments

    def preprocess_segment(
        self,
        df_raw: pd.DataFrame,
        segment: Dict[str, int]
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, RobustScaler]:
        """
        Preprocess a segment with leak-free scaling.

        1. Extract train/eval/test slices from raw data
        2. Apply feature engineering
        3. Fit scaler on TRAIN only (NOT eval, to prevent leakage)
        4. Transform TRAIN, EVAL, and TEST

        Segment structure:
            [train] [eval] [test]
            - train: Used for TQC learning
            - eval: Used for in-train EvalCallback (early stopping, best model)
            - test: Out-of-sample final evaluation

        Returns:
            (train_df, eval_df, test_df, scaler)
        """
        segment_id = segment['id']
        print(f"\n[Segment {segment_id}] Preprocessing...")

        # 1. Extract full segment from pre-computed features (P1 optimization)
        full_start = segment['train_start']
        full_end = segment['test_end']

        # Use global pre-computed features (always available - computed in run() or run_eval_only() methods)
        assert hasattr(self, '_df_features_global') and self._df_features_global is not None, \
            "Global features must be pre-computed. Features are computed in run() or run_eval_only() methods before processing segments."
        
        # Convert integer positions to datetime using raw data's index
        # This preserves alignment because features keep the original DatetimeIndex
        start_time = df_raw.index[full_start]
        end_time = df_raw.index[full_end - 1]  # -1 because .loc[] is inclusive
        df_features = self._df_features_global.loc[start_time:end_time].copy()
        print(f"  Using pre-computed features: {start_time} to {end_time} ({len(df_features)} rows)")

        # 2. Split train/eval/test by exact segment boundaries (exclude purge rows)
        # Structure: [train][PURGE][eval][PURGE][test]; purge must not be in any block.
        # See docs/audit/PURGE_EMBARGO_WFO_TRAIN_EVAL_TEST.md
        train_len = segment['train_end'] - segment['train_start']
        eval_len = segment['eval_end'] - segment['eval_start']
        test_len = segment['test_end'] - segment['test_start']
        purge = segment['purge_window']
        total_needed = train_len + 2 * purge + eval_len + test_len

        if len(df_features) < total_needed:
            print(f"  [WARNING] Not enough data: {len(df_features)} < {total_needed}")
            raise ValueError(
                f"Segment {segment_id}: df_features has {len(df_features)} rows, "
                f"need at least {total_needed} (train+2*purge+eval+test). "
                "Check segment boundaries or feature computation."
            )

        train_df = df_features.iloc[0:train_len].copy()
        eval_df = df_features.iloc[train_len + purge : train_len + purge + eval_len].copy()
        test_df = df_features.iloc[train_len + 2 * purge + eval_len :].copy()

        print(f"  Train: {len(train_df)} rows, Eval: {len(eval_df)} rows, Test: {len(test_df)} rows")

        # 3. Leak-Free Scaling: fit on TRAIN only (NOT eval!) - Utilise DataProcessor unifié
        print("  Applying RobustScaler (fit on train only, NOT eval) via DataProcessor...")
        
        # Identify columns to scale
        cols_to_scale = [
            col for col in train_df.columns
            if col not in self.config.exclude_from_scaling
            and train_df[col].dtype in ['float64', 'float32']
        ]

        print(f"  Scaling {len(cols_to_scale)} columns")

        # Utiliser DataProcessor unifié (configuration standardisée pour WFO)
        processor = DataProcessor(config={'min_iqr': 1e-2, 'clip_range': (-5, 5)})
        
        # Fit on train ONLY (leak-free)
        processor.fit(train_df[cols_to_scale], columns=cols_to_scale)
        
        # Transform all three splits (scaling + clipping gérés par DataProcessor)
        train_df[cols_to_scale] = processor.transform(train_df[cols_to_scale], columns=cols_to_scale)
        eval_df[cols_to_scale] = processor.transform(eval_df[cols_to_scale], columns=cols_to_scale)
        test_df[cols_to_scale] = processor.transform(test_df[cols_to_scale], columns=cols_to_scale)
        
        # Récupérer le scaler pour retour (compatibilité)
        scaler = processor.get_scaler()

        # Clip ZScores to [-5, 5] (already normalized, just safety clip)
        zscore_cols = [c for c in train_df.columns if 'ZScore' in c]
        train_df[zscore_cols] = train_df[zscore_cols].clip(-5, 5)
        eval_df[zscore_cols] = eval_df[zscore_cols].clip(-5, 5)
        test_df[zscore_cols] = test_df[zscore_cols].clip(-5, 5)
        print(f"  Clipped {len(zscore_cols)} ZScore columns to [-5, 5]")

        return train_df, eval_df, test_df, scaler

    def train_hmm(
        self,
        train_df: pd.DataFrame,
        eval_df: pd.DataFrame,
        test_df: pd.DataFrame,
        segment_id: int
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, RegimeDetector, int]:
        """
        Train HMM on train data and predict on train, eval, and test.

        Returns:
            (train_with_hmm, eval_with_hmm, test_with_hmm, detector, context_rows)
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

        # Predict on eval WITH CONTEXT BUFFER (for HMM lookback requirements)
        # HMM needs 168h window, use 336h to be safe
        max_lookback = 336
        eval_context_rows = min(max_lookback, len(train_df))

        # Eval: use last rows from train as context
        eval_with_context = pd.concat([
            train_df.tail(eval_context_rows),
            eval_df
        ], ignore_index=True)
        eval_with_hmm_full = detector.predict(eval_with_context)
        # Remove context rows from eval (we want clean eval data)
        eval_with_hmm = eval_with_hmm_full.iloc[eval_context_rows:].reset_index(drop=True)

        # Predict on test WITH CONTEXT BUFFER
        # For test, use last rows from (train + eval) as context
        combined_train_eval = pd.concat([train_df, eval_df], ignore_index=True)
        context_rows = min(max_lookback, len(combined_train_eval))

        # Include last context_rows from combined as buffer for test prediction
        test_with_context = pd.concat([
            combined_train_eval.tail(context_rows),
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
        eval_final = eval_with_hmm.drop(columns=hmm_features, errors='ignore')
        test_final = test_with_hmm.drop(columns=hmm_features, errors='ignore')

        print(f"  Train final: {train_final.shape}, Eval final: {eval_final.shape}, Test final: {test_final.shape} (context_rows={context_rows})")

        return train_final, eval_final, test_final, detector, context_rows

    def save_segment_data(
        self,
        train_df: pd.DataFrame,
        eval_df: pd.DataFrame,
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
        eval_path = os.path.join(data_dir, "eval.parquet")
        test_path = os.path.join(data_dir, "test.parquet")
        train_df.to_parquet(train_path, index=False)
        eval_df.to_parquet(eval_path, index=False)
        test_df.to_parquet(test_path, index=False)

        # Save metadata (for eval-only mode)
        metadata = {
            'context_rows': context_rows,
            'total_test_rows': len(test_df),
            'actual_test_rows': len(test_df) - context_rows,
            'eval_rows': len(eval_df),
            'train_rows': len(train_df)
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

        print(f"  Saved: {train_path}, {eval_path}, {test_path}, {metadata_path}")

        return train_path, eval_path, test_path

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

        # Train with supervised auxiliary loss (direction prediction)
        # aux_loss_weight=1.0 ensures signal preservation (see analyze_pipeline_hmm_mae.py)
        model, best_loss = train(
            config,
            from_scratch=True,
            supervised=True,
            aux_loss_weight=1.0
        )

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
        resume: bool = False,
        init_model_path: Optional[str] = None,
        eval_path: Optional[str] = None
    ) -> tuple[str, Dict[str, Any]]:
        """
        Train TQC agent on segment train data with optional in-train evaluation.

        Args:
            train_path: Path to train data parquet.
            encoder_path: Path to encoder weights.
            segment_id: Segment identifier.
            use_batch_env: If True, use GPU-accelerated BatchCryptoEnv.
            resume: If True, resume from tqc_last.zip (continues TensorBoard steps).
            init_model_path: Path to model for warm start (Chain of Inheritance).
            eval_path: Path to eval data parquet for in-train EvalCallback.
                       If provided, enables early stopping and best model selection.

        Returns:
            Tuple of (path to saved agent, training metrics dict).
        """
        env_type = "BatchCryptoEnv (GPU)" if use_batch_env else "SubprocVecEnv (CPU)"
        print(f"\n[Segment {segment_id}] Training TQC with {env_type}...")

        # Import here to avoid circular imports
        import torch
        from src.training.train_agent import train
        from dataclasses import replace

        # Use WFOTrainingConfig from centralized config (no hardcodes!)
        # This ensures single source of truth for all TQC hyperparameters.
        tc = self.config.training_config  # WFOTrainingConfig instance
        
        # Create a copy with segment-specific paths
        config = replace(
            tc,
            data_path=train_path,
            encoder_path=encoder_path,
            use_curriculum=True,  # 3-Phase Curriculum enabled
        )
        
        # Log overfitting guard status
        if config.use_overfitting_guard:
            print(f"  [Guard] OverfittingGuardV2 enabled (NAV>{config.guard_nav_threshold}x, patience={config.guard_patience})")

        # Set segment-specific paths
        weights_dir = os.path.join(self.config.weights_dir, f"segment_{segment_id}")
        config.save_path = os.path.join(weights_dir, "tqc.zip")
        config.checkpoint_dir = os.path.join(weights_dir, "checkpoints/")
        config.tensorboard_log = f"logs/wfo/segment_{segment_id}/"
        config.name = f"WFO_seg{segment_id}"

        # In-train evaluation: use eval_path for EvalCallback if provided
        # This enables early stopping and best model selection during WFO
        if eval_path and os.path.exists(eval_path):
            config.eval_data_path = eval_path
            print(f"  [Eval In-Train] Enabled with {eval_path}")
            print(f"  [Eval In-Train] EvalCallback will run every {config.eval_freq} steps")
        else:
            config.eval_data_path = None
            print("  [Eval In-Train] Disabled (no eval_path provided)")

        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.tensorboard_log, exist_ok=True)

        # Train - now returns (model, train_metrics)
        # Pass n_envs and batch_size overrides for GPU-optimized parallelism
        hw_overrides = {
            'n_envs': config.n_envs,
            'batch_size': config.batch_size
        } if use_batch_env else None

        # Resume logic: Priority order:
        # 1. --resume flag: use tqc_last.zip (continue training)
        # 2. init_model_path: warm start from previous segment (Chain of Inheritance)
        # 3. Fresh start (no model loaded)
        resume_path = None
        if resume:
            tqc_last_path = os.path.join(weights_dir, "tqc_last.zip")
            if os.path.exists(tqc_last_path):
                resume_path = tqc_last_path
                print(f"  🔄 RESUME MODE: Loading from {tqc_last_path}")
            else:
                print(f"  ⚠️ Resume requested but {tqc_last_path} not found. Starting fresh.")
        elif init_model_path and os.path.exists(init_model_path):
            # Chain of Inheritance: warm start from previous segment's model
            resume_path = init_model_path
            print(f"  🔗 WARM START: Inheriting from {os.path.basename(init_model_path)}")

        model = None
        try:
            model, train_metrics = train(
                config,
                hw_overrides=hw_overrides,
                use_batch_env=use_batch_env,
                resume_path=resume_path
            )
            print(f"  TQC trained. Saved: {config.save_path}")

        except KeyboardInterrupt:
            print("\n[WFO] User interrupted training.")
            sys.exit(0)

        except Exception as e:
            print(f"\n[CRITICAL WFO FAILURE] Segment training crashed: {e}")
            print(f"[INFO] Check 'emergency_save_internal.zip' in {weights_dir} (saved by train_agent.py).")
            # Do NOT attempt to save model here (it is likely None).
            # We re-raise to stop the WFO pipeline immediately.
            raise e

        finally:
            # Cleanup regardless of success/failure
            if 'model' in locals() and model is not None:
                del model
            torch.cuda.empty_cache()
            gc.collect()

        return config.save_path, train_metrics

    def train_tqc_ensemble(
        self,
        train_path: str,
        encoder_path: str,
        segment_id: int,
        eval_path: Optional[str] = None,
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        Train TQC ensemble for a segment.

        Trains multiple TQC models with different seeds for ensemble inference.
        Supports both sequential (single GPU) and parallel (multi-GPU) training.

        Args:
            train_path: Path to training data.
            encoder_path: Path to encoder weights.
            segment_id: Segment identifier.
            eval_path: Path to eval data (optional).

        Returns:
            Tuple of (model_paths, aggregated_metrics).
        """
        import copy
        import torch
        from src.training.train_agent import train
        from src.evaluation.ensemble import EnsembleConfig, EnsembleTrainer
        from dataclasses import replace

        print(f"\n[Segment {segment_id}] Training TQC Ensemble...")
        print(f"  Members: {self.config.ensemble_n_members}")
        print(f"  Seeds: {self.config.ensemble_seeds[:self.config.ensemble_n_members]}")
        print(f"  Parallel: {self.config.ensemble_parallel}")
        print(f"  Aggregation: {self.config.ensemble_aggregation}")

        # Use WFOTrainingConfig from centralized config (no hardcodes!)
        tc = self.config.training_config
        base_config = replace(
            tc,
            data_path=train_path,
            encoder_path=encoder_path,
            use_curriculum=True,
            eval_data_path=eval_path if eval_path and os.path.exists(eval_path) else None,
        )

        # Ensemble config
        ensemble_config = EnsembleConfig(
            n_members=self.config.ensemble_n_members,
            seeds=self.config.ensemble_seeds[:self.config.ensemble_n_members],
            aggregation=self.config.ensemble_aggregation,
            # Détection automatique des GPUs disponibles
            parallel_gpus=list(range(torch.cuda.device_count())) if torch.cuda.is_available() else [0],
            use_diverse_hyperparams=True,
        )

        # Output directory
        weights_dir = os.path.join(self.config.weights_dir, f"segment_{segment_id}")
        ensemble_dir = os.path.join(weights_dir, "ensemble")
        os.makedirs(ensemble_dir, exist_ok=True)

        # Set base paths for trainer
        base_config.tensorboard_log = f"logs/wfo/segment_{segment_id}/"
        os.makedirs(base_config.tensorboard_log, exist_ok=True)

        # Create trainer
        trainer = EnsembleTrainer(
            base_config=base_config,
            ensemble_config=ensemble_config,
            verbose=1,
        )

        # Train
        try:
            if self.config.ensemble_parallel:
                model_paths = trainer.train_parallel(ensemble_dir)
            else:
                model_paths = trainer.train_sequential(ensemble_dir)
        except Exception as e:
            print(f"\n[CRITICAL] Ensemble training failed: {e}")
            raise e

        # Load aggregated metrics
        metrics_path = os.path.join(ensemble_dir, "ensemble_summary.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                aggregated_metrics = json.load(f)
        else:
            aggregated_metrics = {'n_members': len(model_paths)}

        # Add ensemble-specific info
        aggregated_metrics['segment_status'] = 'SUCCESS'
        aggregated_metrics['guard_early_stop'] = False
        aggregated_metrics['completion_ratio'] = 1.0

        # Create symlink for backward compatibility (first member as "best")
        if model_paths:
            best_model = model_paths[0]
            tqc_symlink = os.path.join(weights_dir, "tqc.zip")
            if os.path.exists(tqc_symlink):
                os.remove(tqc_symlink)
            # On Windows, use copy instead of symlink
            if os.name == 'nt':
                import shutil
                shutil.copy2(best_model, tqc_symlink)
            else:
                os.symlink(best_model, tqc_symlink)

        print(f"  Ensemble trained: {len(model_paths)} models")
        print(f"  Models saved to: {ensemble_dir}")

        # Cleanup
        torch.cuda.empty_cache()
        gc.collect()

        return model_paths, aggregated_metrics

    def evaluate_ensemble_segment(
        self,
        test_path: str,
        encoder_path: str,
        ensemble_dir: str,
        segment_id: int,
        context_rows: int = 0,
        train_metrics: Optional[Dict] = None,
        train_path: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], np.ndarray]:
        """
        Evaluate ensemble on test data.

        Compares:
        1. Ensemble with configured aggregation

        Args:
            test_path: Path to test data parquet.
            encoder_path: Path to encoder weights.
            ensemble_dir: Directory containing ensemble models.
            segment_id: Segment identifier.
            context_rows: Number of warmup rows to skip.
            train_metrics: Optional training metrics.
            train_path: Path to train data for baseline_vol.

        Returns:
            Tuple of (metrics dict, navs array).
        """
        print(f"\n[Segment {segment_id}] Evaluating Ensemble (skipping {context_rows} warmup rows)...")

        import torch
        from src.training.batch_env import BatchCryptoEnv
        from src.evaluation.ensemble import load_ensemble

        # Calculate baseline_vol from TRAIN data
        if train_path and os.path.exists(train_path):
            train_df = pd.read_parquet(train_path)
            baseline_vol = train_df['BTC_Close'].pct_change().std()
        else:
            baseline_vol = 0.01

        # Load ensemble
        ensemble = load_ensemble(ensemble_dir, device='cuda', verbose=1)

        # Calculate episode length from test data
        test_df = pd.read_parquet(test_path)
        test_episode_length = len(test_df) - self.config.window_size - 1

        # Create test env
        env = BatchCryptoEnv(
            parquet_path=test_path,
            n_envs=1,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            window_size=self.config.window_size,
            episode_length=test_episode_length,
            commission=self.config.training_config.eval_commission,
            slippage=0.0001,
            target_volatility=self.config.training_config.target_volatility,
            vol_window=self.config.training_config.vol_window,
            max_leverage=self.config.training_config.max_leverage,
            price_column='BTC_Close',
            random_start=False,
        )
        
        # MORL: Fix w_cost for evaluation (Audit 2026-01-22)
        env.set_eval_w_cost(DEFAULT_EVAL_W_COST)

        # Run evaluation
        obs, info = env.gym_reset()
        done = False
        rewards = []
        navs = []
        ensemble_metrics_history = []

        while not done:
            # Use predict_with_safety for OOD detection
            action, ensemble_info = ensemble.predict_with_safety(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.gym_step(action)
            done = terminated or truncated

            rewards.append(reward)
            navs.append(info.get('nav', 10000.0))
            ensemble_metrics_history.append({
                'std': ensemble_info.get('ensemble_std', 0.0),
                'agreement': ensemble_info.get('ensemble_agreement', 1.0),
                'mode': ensemble_info.get('mode', 'NORMAL'),
                'ood_score': ensemble_info.get('ood_score', 0.0),
            })

        ensemble.close()
        env.close()

        # Calculate metrics - SKIP context_rows
        all_rewards = np.array(rewards)
        all_navs = np.array(navs) if navs else np.array([10000])

        if len(all_rewards) > context_rows:
            rewards_arr = all_rewards[context_rows:]
            navs_arr = all_navs[context_rows:]
        else:
            rewards_arr = all_rewards
            navs_arr = all_navs

        # Sharpe
        if len(rewards_arr) > 1 and rewards_arr.std() > 0:
            sharpe = (rewards_arr.mean() / rewards_arr.std()) * np.sqrt(8760)
        else:
            sharpe = 0.0

        # PnL
        pnl = navs_arr[-1] - navs_arr[0] if len(navs_arr) > 1 else 0
        pnl_pct = (pnl / navs_arr[0]) * 100 if navs_arr[0] > 0 else 0

        # Max Drawdown
        if len(navs_arr) > 1:
            peak = np.maximum.accumulate(navs_arr)
            drawdown = (peak - navs_arr) / peak
            max_drawdown = drawdown.max() * 100
        else:
            max_drawdown = 0.0

        # Ensemble-specific metrics
        avg_agreement = np.mean([m['agreement'] for m in ensemble_metrics_history])
        avg_std = np.mean([m['std'] for m in ensemble_metrics_history])
        n_ood_warnings = sum(1 for m in ensemble_metrics_history if m['mode'] == 'OOD_WARNING')
        n_ood_fallbacks = sum(1 for m in ensemble_metrics_history if m['mode'] == 'OOD_FALLBACK')

        # B&H benchmark
        prices = test_df['BTC_Close'].values
        price_start_idx = self.config.window_size + context_rows
        price_end_idx = price_start_idx + len(navs_arr)
        if price_end_idx > len(prices):
            price_end_idx = len(prices)

        bnh_prices = prices[price_start_idx:price_end_idx]
        if len(bnh_prices) > 1:
            bnh_return = (bnh_prices[-1] - bnh_prices[0]) / bnh_prices[0]
            bnh_pct = bnh_return * 100
            bnh_navs = (bnh_prices / bnh_prices[0]) * 10000
            alpha = pnl_pct - bnh_pct
        else:
            bnh_pct = 0.0
            bnh_navs = np.array([10000])
            alpha = pnl_pct

        metrics = {
            'segment_id': segment_id,
            'model_type': 'ensemble',
            'aggregation': self.config.ensemble_aggregation,
            'n_members': self.config.ensemble_n_members,
            'sharpe': sharpe,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'max_drawdown': max_drawdown,
            'final_nav': navs_arr[-1] if len(navs_arr) > 0 else 10000,
            'test_rows': len(rewards_arr),
            'ensemble_avg_agreement': avg_agreement,
            'ensemble_avg_std': avg_std,
            'n_ood_warnings': n_ood_warnings,
            'n_ood_fallbacks': n_ood_fallbacks,
            'bnh_pct': bnh_pct,
            'alpha': alpha,
            'bnh_navs': bnh_navs,
        }

        if train_metrics:
            metrics['train_action_sat'] = train_metrics.get('avg_action_saturation', 0.0)
            metrics['train_entropy'] = train_metrics.get('avg_entropy', 0.0)

        print(f"  Results (Ensemble - {self.config.ensemble_aggregation}):")
        print(f"    Sharpe: {sharpe:.2f}")
        print(f"    PnL: {pnl_pct:+.2f}%")
        print(f"    Max DD: {max_drawdown:.2f}%")
        print(f"    Alpha vs B&H: {alpha:+.2f}%")
        print(f"    Avg Agreement: {avg_agreement:.2%}")
        print(f"    Avg Action Std: {avg_std:.4f}")
        print(f"    OOD Warnings: {n_ood_warnings}, Fallbacks: {n_ood_fallbacks}")

        # TensorBoard logging
        from torch.utils.tensorboard import SummaryWriter
        eval_log_dir = f"logs/wfo/eval/segment_{segment_id}_ensemble"
        os.makedirs(eval_log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=eval_log_dir)

        writer.add_scalar("eval_ensemble/sharpe", sharpe, segment_id)
        writer.add_scalar("eval_ensemble/pnl_pct", pnl_pct, segment_id)
        writer.add_scalar("eval_ensemble/max_drawdown", max_drawdown, segment_id)
        writer.add_scalar("eval_ensemble/avg_agreement", avg_agreement, segment_id)
        writer.add_scalar("eval_ensemble/avg_std", avg_std, segment_id)
        writer.add_scalar("eval_ensemble/alpha", alpha, segment_id)

        writer.close()

        torch.cuda.empty_cache()

        return metrics, navs_arr

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
        from collections import deque
        from sb3_contrib import TQC
        from src.training.batch_env import BatchCryptoEnv

        # Calculate baseline_vol from TRAIN data (avoids data leakage)
        if train_path and os.path.exists(train_path):
            train_df = pd.read_parquet(train_path)
            baseline_vol = train_df['BTC_Close'].pct_change().std()
            print(f"  Circuit Breaker calibrated on TRAIN data. Baseline Vol: {baseline_vol:.5f}")
        else:
            baseline_vol = 0.01  # Conservative fallback
            print(f"  [WARNING] train_path not provided. Using default baseline_vol: {baseline_vol:.5f}")

        # Calculate episode length from test data (full episode)
        test_df = pd.read_parquet(test_path)
        test_episode_length = len(test_df) - self.config.window_size - 1

        # Create test environment using BatchCryptoEnv with n_envs=1
        # NOTE: max_leverage=1.0 disables volatility scaling for evaluation
        # This prevents the "stuck in cash" bug where portfolio returns collapse to 0
        env = BatchCryptoEnv(
            parquet_path=test_path,
            n_envs=1,  # Single env for Gymnasium-compatible evaluation
            device='cuda' if torch.cuda.is_available() else 'cpu',
            window_size=self.config.window_size,
            episode_length=test_episode_length,  # Full episode
            commission=self.config.training_config.eval_commission,
            slippage=0.0001,
            target_volatility=self.config.training_config.target_volatility,
            vol_window=self.config.training_config.vol_window,
            max_leverage=self.config.training_config.max_leverage,  # MORL: Cohérence train/eval (fix Distributional Shift)
            price_column='BTC_Close',
            random_start=False,  # Sequential start for evaluation
        )
        
        # MORL: Fix w_cost for evaluation (Audit 2026-01-22)
        # Default: 0.5 (balanced mode). For Pareto front, iterate over EVAL_W_COST_VALUES
        env.set_eval_w_cost(DEFAULT_EVAL_W_COST)

        # Circuit Breaker state (DISABLED - too aggressive)
        cb_vol_window = 24
        cb_vol_threshold = 999.0  # Effectively disabled
        cb_max_drawdown = 1.0     # Effectively disabled
        cb_cooldown_steps = 0
        cb_nav_history = deque(maxlen=cb_vol_window + 1)
        cb_peak_nav = 10000.0
        cb_cooldown_remaining = 0
        circuit_breaker_count = 0

        # Load agent (must use CUDA - encoder uses float16 which fails on CPU)
        model = TQC.load(tqc_path, device='cuda')

        # Run evaluation using Gymnasium-compatible interface
        obs, info = env.gym_reset()
        done = False
        total_reward = 0
        rewards = []
        navs = []

        while not done:
            # Circuit Breaker: force HOLD if in cooldown
            if cb_cooldown_remaining > 0:
                action = np.array([0.0])
                cb_cooldown_remaining -= 1
            else:
                action, _ = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.gym_step(action)
            done = terminated or truncated

            # Track NAV for circuit breaker
            nav = info.get('nav', 10000.0)
            cb_nav_history.append(nav)
            cb_peak_nav = max(cb_peak_nav, nav)
            current_drawdown = (cb_peak_nav - nav) / cb_peak_nav

            # Calculate rolling volatility
            rolling_vol = 0.0
            if len(cb_nav_history) > 1:
                nav_array = np.array(cb_nav_history)
                returns = np.diff(np.log(nav_array))
                rolling_vol = np.std(returns) if len(returns) > 1 else 0.0

            # Check circuit breaker triggers
            if cb_cooldown_remaining == 0:
                vol_trigger = rolling_vol > (baseline_vol * cb_vol_threshold)
                dd_trigger = current_drawdown > cb_max_drawdown
                if vol_trigger or dd_trigger:
                    cb_cooldown_remaining = cb_cooldown_steps
                    circuit_breaker_count += 1

            total_reward += reward
            rewards.append(reward)
            navs.append(nav)

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
        resume: bool = False,
        init_model_path: Optional[str] = None
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Run full WFO pipeline for a single segment.

        Args:
            df_raw: Raw DataFrame with OHLCV data.
            segment: Segment boundaries dict.
            use_batch_env: If True, use GPU-accelerated BatchCryptoEnv for TQC.
            resume: If True, resume TQC training from tqc_last.zip.
            init_model_path: Path to model for warm start (Chain of Inheritance).

        Returns:
            Tuple of (metrics dict, train_metrics dict) for this segment.
        """
        segment_id = segment['id']
        print("\n" + "=" * 70)
        print(f"SEGMENT {segment_id}")
        print(f"Train: rows {segment['train_start']} - {segment['train_end']} ({self.config.train_months}m)")
        print(f"Eval:  rows {segment['eval_start']} - {segment['eval_end']} ({self.config.eval_months}m)")
        print(f"Test:  rows {segment['test_start']} - {segment['test_end']} ({self.config.test_months}m)")
        print("=" * 70)

        # --- Define Paths ---
        models_dir = os.path.join(self.config.models_dir, f"segment_{segment_id}")
        weights_dir = os.path.join(self.config.weights_dir, f"segment_{segment_id}")
        data_dir = os.path.join(self.config.output_dir, f"segment_{segment_id}")

        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(weights_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)

        hmm_path = os.path.join(models_dir, "hmm.pkl")
        scaler_path = os.path.join(models_dir, "scaler.pkl")
        train_path = os.path.join(data_dir, "train.parquet")
        eval_path = os.path.join(data_dir, "eval.parquet")
        test_path = os.path.join(data_dir, "test.parquet")
        metadata_path = os.path.join(data_dir, "metadata.json")
        encoder_path = os.path.join(weights_dir, "encoder.pth")

        # --- 1. Preprocessing & HMM Logic (Smart Skip) ---
        all_data_exists = (
            os.path.exists(hmm_path) and 
            os.path.exists(train_path) and 
            os.path.exists(eval_path) and
            os.path.exists(test_path) and 
            os.path.exists(metadata_path)
        )
        
        if all_data_exists:
            print(f"[SKIP] Found existing HMM and processed data for Segment {segment_id}. Loading from disk...")

            # Load Metadata (context_rows)
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            context_rows = metadata.get('context_rows', 0)
            print(f"   -> Loaded context_rows: {context_rows}")

        else:
            print(f"[EXEC] Processing data and training HMM for Segment {segment_id}...")
            # Normal flow: Preprocess -> Train HMM -> Save
            train_df, eval_df, test_df, scaler = self.preprocess_segment(df_raw, segment)

            # DEBUG: Vérification du Scaling avant HMM
            features_check = ['BTC_RSI_14', 'BTC_MACD_Hist', 'BTC_ADX_14']
            print("\n[AUDIT SCALING] Vérification des features momentum avant HMM:")
            for f in features_check:
                if f in train_df.columns:
                    stat = train_df[f]
                    print(f"  > {f}: Mean={stat.mean():.4f}, Std={stat.std():.4f}, Min={stat.min():.4f}, Max={stat.max():.4f}")
                    if abs(stat.mean()) > 1.0 or stat.std() > 5.0:
                        print(f"    ⚠️ ALERTE: {f} semble mal scalé !")
                else:
                    print(f"  ⚠️ ALERTE: {f} manquant !")

            train_df, eval_df, test_df, hmm, context_rows = self.train_hmm(train_df, eval_df, test_df, segment_id)

            # Save artifacts for future skips
            self.save_segment_data(train_df, eval_df, test_df, scaler, segment_id, context_rows)

            # Cleanup dataframes
            del train_df, eval_df, test_df, hmm
            gc.collect()

        # --- 2. MAE Logic (Smart Skip) ---
        if os.path.exists(encoder_path):
            print(f"[SKIP] Found existing MAE encoder at {encoder_path}. Skipping MAE training.")
        else:
            print(f"[EXEC] Training MAE Foundation Model for Segment {segment_id}...")
            encoder_path = self.train_mae(train_path, segment_id)

        # === 5. TQC Training: Single vs Ensemble Mode ===
        if self.config.use_ensemble:
            # === ENSEMBLE MODE ===
            ensemble_dir = os.path.join(weights_dir, "ensemble")
            model_paths, train_metrics = self.train_tqc_ensemble(
                train_path, encoder_path, segment_id,
                eval_path=eval_path
            )
            tqc_path = model_paths[0] if model_paths else None
        else:
            # === SINGLE MODEL MODE (existing code) ===
            tqc_path, train_metrics = self.train_tqc(
                train_path, encoder_path, segment_id,
                use_batch_env=use_batch_env,
                resume=resume,
                init_model_path=init_model_path,
                eval_path=eval_path  # Enable EvalCallback for in-train evaluation
            )

        # === 5.5 Guard Fail-over Logic ===
        guard_triggered = train_metrics.get('guard_early_stop', False)
        completion_ratio = train_metrics.get('completion_ratio', 1.0)
        stop_reason = train_metrics.get('guard_stop_reason', None)

        if guard_triggered:
            print(f"\n  ⚠️ GUARD INTERVENTION on Segment {segment_id}")
            print(f"     Reason: {stop_reason}")
            print(f"     Completion: {completion_ratio:.1%}")

            # Chercher le dernier checkpoint valide
            last_ckpt = self._find_last_valid_checkpoint(segment_id)
            can_recover = (
                last_ckpt is not None
                and self.config.use_checkpoint_on_failure
                and completion_ratio >= self.config.min_completion_ratio
            )

            if can_recover:
                # RECOVERY: Utiliser le checkpoint
                print(f"  → 🚑 RECOVERY: Using checkpoint {os.path.basename(last_ckpt)}")
                from sb3_contrib import TQC
                recovered_model = TQC.load(last_ckpt)
                # Sauvegarder comme modèle officiel
                recovered_model.save(os.path.join(weights_dir, "tqc.zip"))
                del recovered_model
                train_metrics['segment_status'] = 'RECOVERED'
                train_metrics['used_checkpoint'] = os.path.basename(last_ckpt)
            else:
                # FAILED: Utiliser stratégie de repli
                reason = f"ratio {completion_ratio:.1%} < {self.config.min_completion_ratio:.0%}" if last_ckpt else "no checkpoint"
                print(f"  → 💀 FAILED: Cannot recover ({reason})")
                print(f"     Using fallback strategy: '{self.config.fallback_strategy}'")
                train_metrics['segment_status'] = 'FAILED'
                train_metrics['fallback_strategy'] = self.config.fallback_strategy

                # Générer métriques de fallback (sans utiliser le modèle)
                fallback_metrics = self._run_fallback_strategy(segment, test_path)
                fallback_metrics['segment_id'] = segment_id
                fallback_metrics['model_type'] = 'fallback'
                fallback_metrics['segment_status'] = 'FAILED'
                fallback_metrics['stop_reason'] = stop_reason
                fallback_metrics['completion_ratio'] = completion_ratio

                # Sauvegarder les résultats
                self.save_results(fallback_metrics)

                return fallback_metrics, train_metrics
        else:
            train_metrics['segment_status'] = 'SUCCESS'

        # 6. Organize Artifacts (Swap & Archive) - BEFORE evaluation
        # This ensures tqc.zip=Best and tqc_last.zip=Last exist on disk
        if not self.config.use_ensemble:
            self._organize_artifacts(segment_id)

        # === 7. EVALUATION: Single vs Ensemble Mode ===
        if self.config.use_ensemble:
            # === ENSEMBLE EVALUATION ===
            ensemble_dir = os.path.join(weights_dir, "ensemble")

            # Evaluate ensemble
            metrics, navs = self.evaluate_ensemble_segment(
                test_path, encoder_path, ensemble_dir, segment_id,
                context_rows=context_rows,
                train_metrics=train_metrics,
                train_path=train_path
            )

            # Tag results
            metrics['segment_status'] = train_metrics.get('segment_status', 'SUCCESS')
            metrics['stop_reason'] = train_metrics.get('guard_stop_reason', None)
            metrics['completion_ratio'] = train_metrics.get('completion_ratio', 1.0)

            # Generate diagnostic plots
            plot_path = self.generate_segment_plots(
                segment_id, navs, metrics, train_metrics, prefix="ensemble_"
            )
            metrics['plot_path'] = plot_path

            # Save results
            self.save_results(metrics)

            # Also evaluate single best for comparison
            single_model_path = os.path.join(weights_dir, "tqc.zip")
            if os.path.exists(single_model_path):
                print(f"\n  [COMPARISON] Evaluating single model for comparison...")
                single_metrics, _ = self.evaluate_segment(
                    test_path, encoder_path, single_model_path, segment_id,
                    context_rows=context_rows,
                    train_metrics={},
                    train_path=train_path,
                )

                # Log comparison
                print(f"\n  [Comparison] Single vs Ensemble:")
                print(f"    Single Sharpe: {single_metrics['sharpe']:.2f}")
                print(f"    Ensemble Sharpe: {metrics['sharpe']:.2f}")
                print(f"    Improvement: {metrics['sharpe'] - single_metrics['sharpe']:+.2f}")

                metrics['single_sharpe'] = single_metrics['sharpe']
                metrics['single_pnl_pct'] = single_metrics['pnl_pct']
                metrics['improvement_sharpe'] = metrics['sharpe'] - single_metrics['sharpe']

            best_metrics = metrics

        else:
            # === SINGLE MODEL EVALUATION (existing code) ===
            # 7. Double Evaluation: Best Model + Last Model
            # INTENTIONAL: Evaluate both models to compare performance:
            # - 'best': Highest eval reward during training (may overfit to train data)
            # - 'last': Final model state (may be more robust, less overfitted)
            # This comparison helps identify overfitting and guides hyperparameter tuning.
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

                # Tag results with model type and segment status
                metrics['model_type'] = model_type
                metrics['segment_status'] = train_metrics.get('segment_status', 'SUCCESS')
                metrics['stop_reason'] = train_metrics.get('guard_stop_reason', None)
                metrics['completion_ratio'] = train_metrics.get('completion_ratio', 1.0)

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

    def _find_last_valid_checkpoint(self, segment_id: int) -> Optional[str]:
        """
        Find the last valid checkpoint for a segment.

        Looks for .zip files in the checkpoints directory and returns
        the most recently modified one.

        Args:
            segment_id: Segment identifier.

        Returns:
            Path to the last checkpoint, or None if not found.
        """
        import glob

        weights_dir = os.path.join(self.config.weights_dir, f"segment_{segment_id}")
        ckpt_dir = os.path.join(weights_dir, "checkpoints")

        if not os.path.exists(ckpt_dir):
            return None

        # Find all .zip checkpoint files
        checkpoints = glob.glob(os.path.join(ckpt_dir, "*.zip"))
        if not checkpoints:
            return None

        # Return the most recently modified
        return max(checkpoints, key=os.path.getmtime)

    def _run_fallback_strategy(self, segment: Dict[str, int], test_path: str) -> Dict[str, Any]:
        """
        Execute a fallback strategy for FAILED segments.

        Args:
            segment: Segment boundaries dict.
            test_path: Path to test data parquet.

        Returns:
            Metrics dict for the fallback strategy.
        """
        strategy = self.config.fallback_strategy

        if strategy == 'flat':
            # Flat = No trading = 0 returns
            return {
                'sharpe': 0.0,
                'total_return': 0.0,
                'pnl': 0.0,
                'pnl_pct': 0.0,
                'max_drawdown': 0.0,
                'total_trades': 0,
                'strategy': 'FLAT (fallback)',
                'is_fallback': True,
                'bnh_pct': 0.0,  # Will be calculated below
                'alpha': 0.0,
            }

        elif strategy == 'buy_and_hold':
            # Calculate B&H return on TEST period
            test_df = pd.read_parquet(test_path)
            prices = test_df['BTC_Close'].values

            if len(prices) > 1:
                start_price = prices[0]
                end_price = prices[-1]
                bnh_return = (end_price - start_price) / start_price
                bnh_pct = bnh_return * 100

                # Simple drawdown calculation
                cummax = np.maximum.accumulate(prices)
                drawdowns = (cummax - prices) / cummax
                max_dd = np.max(drawdowns) * 100
            else:
                bnh_return = 0.0
                bnh_pct = 0.0
                max_dd = 0.0

            return {
                'sharpe': 0.0,  # Not calculated for simple B&H
                'total_return': bnh_return,
                'pnl': bnh_return * 10000,  # Assuming 10k initial
                'pnl_pct': bnh_pct,
                'max_drawdown': max_dd,
                'total_trades': 0,
                'strategy': 'BUY_AND_HOLD (fallback)',
                'is_fallback': True,
                'bnh_pct': bnh_pct,
                'alpha': 0.0,  # B&H has no alpha vs itself
            }

        else:
            raise ValueError(f"Unknown fallback strategy: {strategy}")

    def _cleanup_failed_segment_checkpoints(self, segment_id: int) -> int:
        """
        Remove checkpoints from a FAILED segment to save disk space.

        Checkpoints from failed segments are potentially corrupted or
        in an unstable state and should not be reused.

        Args:
            segment_id: Segment identifier.

        Returns:
            Number of bytes freed.
        """
        import glob

        weights_dir = os.path.join(self.config.weights_dir, f"segment_{segment_id}")
        ckpt_dir = os.path.join(weights_dir, "checkpoints")

        if not os.path.exists(ckpt_dir):
            return 0

        freed_bytes = 0
        checkpoints = glob.glob(os.path.join(ckpt_dir, "*.zip"))

        for ckpt in checkpoints:
            try:
                size = os.path.getsize(ckpt)
                os.remove(ckpt)
                freed_bytes += size
                print(f"  🗑️ Removed failed checkpoint: {os.path.basename(ckpt)}")
            except OSError as e:
                print(f"  [WARNING] Could not remove {ckpt}: {e}")

        if freed_bytes > 0:
            print(f"  [CLEANUP] Freed {freed_bytes / 1024 / 1024:.1f} MB from failed segment {segment_id}")

        return freed_bytes

    def _get_segment_model_path(self, segment_id: int) -> str:
        """
        Get the path to the final model for a segment.

        Args:
            segment_id: Segment identifier.

        Returns:
            Path to the segment's final model (tqc.zip).
        """
        weights_dir = os.path.join(self.config.weights_dir, f"segment_{segment_id}")
        return os.path.join(weights_dir, "tqc.zip")

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
            print(f"    -> Increase w_cost (MORL) to penalize position changes")
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
        print(f"  Eval:  {self.config.eval_months} months ({self.config.eval_rows} rows) [In-Train EvalCallback]")
        print(f"  Test:  {self.config.test_months} months ({self.config.test_rows} rows) [OOS Final Evaluation]")
        print(f"  Step:  {self.config.step_months} months ({self.config.step_rows} rows)")
        print(f"  Total train window: {self.config.train_months + self.config.eval_months} months")
        print(f"  Volatility Scaling: Target={self.config.training_config.target_volatility}, Window={self.config.training_config.vol_window}")
        print(f"  GPU Acceleration: {'BatchCryptoEnv' if self.config.use_batch_env else 'SubprocVecEnv (CPU)'}")

        # Load raw data
        print(f"\nLoading raw data: {self.config.raw_data_path}")
        df_raw = self._load_raw_data(self.config.raw_data_path)
        print(f"  Shape: {df_raw.shape}")

        # Remove synthetic Funding_Rate (audit P1.2 - avoid spurious correlations)
        if 'Funding_Rate' in df_raw.columns:
            print("\n[INFO] Removing Funding_Rate column (synthetic data - audit P1.2)")
            print("  [INFO] Environment uses fixed funding_rate=0.0001 for short position costs")
            df_raw = df_raw.drop(columns=['Funding_Rate'])
            print(f"  Shape after removal: {df_raw.shape}")

        # Pre-calculate features globally (P1 optimization: ~20-30% speedup)
        print("\n[OPTIMIZATION] Pre-calculating features globally (once)...")
        self._df_features_global = self.feature_engineer.engineer_features(df_raw)
        print(f"  Features shape: {self._df_features_global.shape}")

        # Calculate warmup offset (rows lost at start due to feature engineering NaN)
        # This enables Dynamic WFO Offset - first segment shifts to accommodate training window
        warmup_offset = len(df_raw) - len(self._df_features_global)
        if warmup_offset > 0:
            print(f"  Warmup offset: {warmup_offset} rows (lost to rolling window NaN)")

        # Calculate segments with dynamic offset
        segments = self.calculate_segments(len(df_raw), warmup_offset=warmup_offset)
        print(f"\nTotal segments: {len(segments)}")

        # Filter segments if specified
        if segment_ids is not None:
            segments = [s for s in segments if s['id'] in segment_ids]
        if max_segments is not None:
            segments = segments[:max_segments]

        print(f"Running segments: {[s['id'] for s in segments]}")

        # === Chain of Inheritance Configuration ===
        if self.config.use_warm_start:
            print(f"\n[Chain of Inheritance] Enabled")
            print(f"  Pretrained model: {self.config.pretrained_model_path or 'None (cold start for Seg 0)'}")
        else:
            print(f"\n[Chain of Inheritance] Disabled (cold start for each segment)")

        # Track the last successful model for inheritance (Rollback logic)
        last_successful_model_path = self.config.pretrained_model_path

        # Run each segment
        all_metrics = []
        all_train_info = []  # Track train info for summary
        
        for segment in segments:
            segment_id = segment['id']
            
            # === Determine init_model_path for this segment ===
            init_model_path = None
            if self.config.use_warm_start:
                if segment_id == 0:
                    init_model_path = self.config.pretrained_model_path
                    if init_model_path:
                        print(f"\n[Chain] Segment 0: Using pretrained model")
                    else:
                        print(f"\n[Chain] Segment 0: Cold start (no pretrained model)")
                else:
                    if last_successful_model_path and os.path.exists(last_successful_model_path):
                        init_model_path = last_successful_model_path
                        print(f"\n[Chain] Segment {segment_id}: Inheriting from {os.path.basename(last_successful_model_path)}")
                    else:
                        print(f"\n[Chain] Segment {segment_id}: Cold start (no valid previous model)")
            
            try:
                metrics, train_metrics = self.run_segment(
                    df_raw, segment,
                    use_batch_env=self.config.use_batch_env,
                    resume=self.config.resume,
                    init_model_path=init_model_path
                )
                all_metrics.append(metrics)
                all_train_info.append(train_metrics)
                
                # === Update Chain of Inheritance based on segment status ===
                status = train_metrics.get('segment_status', 'SUCCESS')
                
                if status in ['SUCCESS', 'RECOVERED']:
                    # This segment produced a valid model -> update chain
                    current_model_path = self._get_segment_model_path(segment_id)
                    if os.path.exists(current_model_path):
                        last_successful_model_path = current_model_path
                        print(f"  ✅ Chain updated: Segment {segment_id} model is now the inheritance source")
                    else:
                        print(f"  ⚠️ Warning: Status is {status} but model file not found")
                else:
                    # FAILED: Do NOT update chain, cleanup checkpoints
                    print(f"  ⛔ Chain NOT updated (Segment {segment_id} FAILED)")
                    if self.config.cleanup_failed_checkpoints:
                        self._cleanup_failed_segment_checkpoints(segment_id)

                # Note: save_results is now called inside run_segment for each model type

                # GATE: Disabled - let WFO run all segments regardless of segment 0 performance
                # if segment['id'] == 0:
                #     if not self._check_segment0_gate(metrics, train_metrics):
                #         print("\n[WFO] Stopped at segment 0 gate.")
                #         break

            except KeyboardInterrupt:
                print("\n[USER] Interrupted by user (Ctrl+C).")
                sys.exit(0)
            except Exception as e:
                error_msg = str(e).lower()

                # Log the error with full traceback
                print(f"\n[FATAL] Segment {segment['id']} failed: {e}")
                import traceback
                traceback.print_exc()

                # OOM = Fail Fast (no point retrying)
                if isinstance(e, MemoryError) or "out of memory" in error_msg:
                    print("[FATAL] OOM detected. Stopping WFO to prevent cascading failures.")
                    sys.exit(1)

                # Numerical instability (NaN/Inf) = Fail Fast (model is corrupted)
                if "nan" in error_msg or "inf" in error_msg or "numerical" in error_msg:
                    print("[FATAL] Numerical instability detected. Model is corrupted.")
                    sys.exit(1)

                # FAIL FAST: Stop WFO on ANY training error (do not burn GPU credits)
                print("\n" + "=" * 50)
                print("[POLICY] FAIL-FAST: Stopping WFO on training error.")
                print("         Fix the issue before restarting.")
                print("         Use --segment N to resume from segment N.")
                print("=" * 50)
                sys.exit(1)

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
            
            # === Segment Status Summary ===
            if all_train_info:
                statuses = [t.get('segment_status', 'UNKNOWN') for t in all_train_info]
                n_success = statuses.count('SUCCESS')
                n_recovered = statuses.count('RECOVERED')
                n_failed = statuses.count('FAILED')
                
                print(f"\n[Segment Status]")
                print(f"  SUCCESS:   {n_success}")
                print(f"  RECOVERED: {n_recovered}")
                print(f"  FAILED:    {n_failed}")
                
                # Calculate Sharpe excluding failed segments
                if n_failed > 0 and 'is_fallback' in df_results.columns:
                    df_valid = df_results[~df_results.get('is_fallback', False).fillna(False)]
                    if len(df_valid) > 0:
                        print(f"\n[Excluding Failed Segments]")
                        print(f"  Average Sharpe: {df_valid['sharpe'].mean():.2f}")
                        print(f"  Average PnL: {df_valid['pnl_pct'].mean():+.2f}%")
            
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
        df_raw = self._load_raw_data(self.config.raw_data_path)
        print(f"  Shape: {df_raw.shape}")

        # Remove synthetic Funding_Rate (audit P1.2 - avoid spurious correlations)
        if 'Funding_Rate' in df_raw.columns:
            print("\n[INFO] Removing Funding_Rate column (synthetic data - audit P1.2)")
            print("  [INFO] Environment uses fixed funding_rate=0.0001 for short position costs")
            df_raw = df_raw.drop(columns=['Funding_Rate'])
            print(f"  Shape after removal: {df_raw.shape}")

        # Pre-calculate features globally (P1 optimization)
        print("\n[OPTIMIZATION] Pre-calculating features globally (once)...")
        self._df_features_global = self.feature_engineer.engineer_features(df_raw)
        print(f"  Features shape: {self._df_features_global.shape}")

        # Calculate warmup offset (rows lost at start due to feature engineering NaN)
        warmup_offset = len(df_raw) - len(self._df_features_global)
        if warmup_offset > 0:
            print(f"  Warmup offset: {warmup_offset} rows (lost to rolling window NaN)")

        # Calculate all segments to get their boundaries (with dynamic offset)
        all_segments = self.calculate_segments(len(df_raw), warmup_offset=warmup_offset)
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
                train_df, eval_df, test_df, scaler = self.preprocess_segment(df_raw, segment)

                # DEBUG: Vérification du Scaling avant HMM
                features_check = ['BTC_RSI_14', 'BTC_MACD_Hist', 'BTC_ADX_14']
                print("\n[AUDIT SCALING] Vérification des features momentum avant HMM:")
                for f in features_check:
                    if f in train_df.columns:
                        stat = train_df[f]
                        print(f"  > {f}: Mean={stat.mean():.4f}, Std={stat.std():.4f}, Min={stat.min():.4f}, Max={stat.max():.4f}")
                        if abs(stat.mean()) > 1.0 or stat.std() > 5.0:
                            print(f"    ⚠️ ALERTE: {f} semble mal scalé !")
                    else:
                        print(f"  ⚠️ ALERTE: {f} manquant !")

                # 2. HMM (with context buffer)
                train_df, eval_df, test_df, hmm, context_rows = self.train_hmm(train_df, eval_df, test_df, seg_id)

                # 3. Save data (with metadata)
                train_path, eval_path, test_path = self.save_segment_data(
                    train_df, eval_df, test_df, scaler, seg_id, context_rows
                )

                # Cleanup
                del train_df, eval_df, test_df, hmm
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
    parser.add_argument("--raw-data", type=str, default="data/raw_historical/multi_asset_historical.csv",
                        help="Path to raw OHLCV data")
    parser.add_argument("--segments", type=int, default=None,
                        help="Max number of segments to run")
    parser.add_argument("--segment", type=int, default=None,
                        help="Run specific segment only")
    parser.add_argument("--timesteps", type=int, default=WFOTrainingConfig.total_timesteps,
                        help="TQC training timesteps per segment")
    # === TQC Hyperparameters ===
    parser.add_argument("--learning-rate", type=float, default=None,
                        help="TQC learning rate (default: from WFOTrainingConfig)")
    parser.add_argument("--gamma", type=float, default=None,
                        help="Discount factor (default: from WFOTrainingConfig)")
    parser.add_argument("--ent-coef", type=float, default=None,
                        help="Entropy coefficient (default: auto_1.0)")
    parser.add_argument("--w-cost-scale", type=float, default=None,
                        help="MORL w_cost end value for curriculum (default: 0.1)")
    parser.add_argument("--mae-epochs", type=int, default=WFOConfig.mae_epochs,
                        help="MAE training epochs per segment")
    parser.add_argument("--train-months", type=int, default=WFOConfig.train_months,
                        help="Training window in months (default: 14)")
    parser.add_argument("--eval-months", type=int, default=WFOConfig.eval_months,
                        help="In-train eval window in months (default: 1)")
    parser.add_argument("--test-months", type=int, default=WFOConfig.test_months,
                        help="OOS test window in months (default: 3)")
    parser.add_argument("--step-months", type=int, default=WFOConfig.step_months,
                        help="Rolling step in months (default: 3)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Re-run evaluation only (skip MAE/TQC training)")
    parser.add_argument("--eval-segments", type=str, default=None,
                        help="Comma-separated segment IDs to evaluate (e.g., '0,1,2')")
    parser.add_argument("--no-batch-env", action="store_true",
                        help="Disable GPU-accelerated BatchCryptoEnv (Force CPU mode)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from tqc_last.zip (continues TensorBoard steps)")

    # === Overfitting Guard Arguments ===
    parser.add_argument("--no-overfitting-guard", action="store_true",
                        help="Disable OverfittingGuardCallbackV2 (not recommended)")
    parser.add_argument("--guard-nav-threshold", type=float, default=WFOTrainingConfig.guard_nav_threshold,
                        help="NAV multiplier threshold for Guard (default: 10.0)")
    parser.add_argument("--guard-patience", type=int, default=WFOTrainingConfig.guard_patience,
                        help="Consecutive violations before Guard stops (default: 5)")
    parser.add_argument("--guard-check-freq", type=int, default=WFOTrainingConfig.guard_check_freq,
                        help="Guard check frequency in steps (default: 25000)")

    # === Fail-over Strategy Arguments ===
    parser.add_argument("--fallback-strategy", type=str, choices=['flat', 'buy_and_hold'],
                        default='flat', help="Strategy for FAILED segments (default: flat)")
    parser.add_argument("--min-completion-ratio", type=float, default=0.30,
                        help="Min training completion for recovery (default: 0.30)")

    # === Chain of Inheritance Arguments ===
    parser.add_argument("--no-warm-start", action="store_true",
                        help="Disable warm start (cold start each segment)")
    parser.add_argument("--pretrained-model", type=str, default=None,
                        help="Path to pretrained model for Segment 0")

    # === Ensemble RL Arguments ===
    parser.add_argument("--ensemble", action="store_true",
                        help="Enable ensemble training (default: 3 members)")
    parser.add_argument("--ensemble-members", type=int, default=3,
                        help="Number of ensemble members (default: 3)")
    parser.add_argument("--ensemble-seeds", type=str, default="42,123,456",
                        help="Comma-separated seeds for ensemble members")
    parser.add_argument("--ensemble-aggregation", type=str,
                        choices=['mean', 'median', 'confidence', 'conservative', 'pessimistic_bound'],
                        default='confidence',
                        help="Ensemble aggregation method (default: confidence)")
    parser.add_argument("--no-ensemble-parallel", action="store_true",
                        help="Disable parallel training (use single GPU)")

    # === Clean WFO artefacts ===
    parser.add_argument("--clean", action="store_true",
                        help="Run clean_wfo.py (remove logs, TB, weights, encoder, data) then exit")
    parser.add_argument("--clean-dry-run", action="store_true",
                        help="With --clean: dry-run only (no deletion)")

    args = parser.parse_args()

    # Create config
    config = WFOConfig()
    config.raw_data_path = args.raw_data
    config.training_config.total_timesteps = args.timesteps  # Set TQC timesteps
    config.mae_epochs = args.mae_epochs
    config.train_months = args.train_months
    config.eval_months = args.eval_months
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

    # === Overfitting Guard Configuration ===
    # Parameters are in WFOTrainingConfig (centralized)
    tc = config.training_config
    if args.no_overfitting_guard:
        print("[WARNING] OverfittingGuardCallbackV2 DISABLED via --no-overfitting-guard")
        tc.use_overfitting_guard = False
    else:
        tc.use_overfitting_guard = True
        tc.guard_nav_threshold = args.guard_nav_threshold
        tc.guard_patience = args.guard_patience
        tc.guard_check_freq = args.guard_check_freq

    # === TQC Hyperparameters Override ===
    if args.learning_rate is not None:
        tc.learning_rate = args.learning_rate
        print(f"[INFO] Learning rate: {tc.learning_rate}")
    if args.gamma is not None:
        tc.gamma = args.gamma
        print(f"[INFO] Gamma: {tc.gamma}")
    if args.ent_coef is not None:
        tc.ent_coef = args.ent_coef
        print(f"[INFO] Entropy coef: {tc.ent_coef}")
    if args.w_cost_scale is not None:
        tc.w_cost_end = args.w_cost_scale
        print(f"[INFO] W_cost end (curriculum): {tc.w_cost_end}")

    # === Fail-over Strategy Configuration ===
    config.fallback_strategy = args.fallback_strategy
    config.min_completion_ratio = args.min_completion_ratio

    # === Chain of Inheritance Configuration ===
    if args.no_warm_start:
        print("[INFO] Warm start DISABLED: Each segment starts from scratch")
        config.use_warm_start = False
    else:
        config.use_warm_start = True
    
    if args.pretrained_model:
        if os.path.exists(args.pretrained_model):
            config.pretrained_model_path = args.pretrained_model
            print(f"[INFO] Pretrained model for Segment 0: {args.pretrained_model}")
        else:
            print(f"[WARNING] Pretrained model not found: {args.pretrained_model}")

    # === Ensemble RL Configuration ===
    if args.ensemble:
        config.use_ensemble = True
        config.ensemble_n_members = args.ensemble_members
        config.ensemble_seeds = [int(x.strip()) for x in args.ensemble_seeds.split(',')]
        config.ensemble_aggregation = args.ensemble_aggregation
        config.ensemble_parallel = not args.no_ensemble_parallel

        print(f"[ENSEMBLE] Enabled with {config.ensemble_n_members} members")
        print(f"  Seeds: {config.ensemble_seeds[:config.ensemble_n_members]}")
        print(f"  Aggregation: {config.ensemble_aggregation}")
        print(f"  Parallel training: {config.ensemble_parallel}")

    # === Clean WFO artefacts (then exit) ===
    if args.clean:
        clean_script = Path(__file__).resolve().parents[1] / "scripts" / "clean_wfo.py"
        cmd = [sys.executable, str(clean_script), "--yes"]
        if args.clean_dry_run:
            cmd.append("--dry-run")
        subprocess.run(cmd, cwd=ROOT_DIR, check=True)
        print("[WFO] Clean done. Exiting.")
        sys.exit(0)

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

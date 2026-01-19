#!/usr/bin/env python3
"""
prepare_train_eval_split.py - Prepare train/eval data split for OverfittingGuard Signal 3.

Creates two separate parquet files from a single processed data file:
- data/processed_data.parquet (Train: 80%)
- data/processed_data_eval.parquet (Eval: 20%)

With a purge window between them to avoid data leakage from technical indicators.

Usage:
    python scripts/prepare_train_eval_split.py
    python scripts/prepare_train_eval_split.py --input data/my_data.parquet --train-ratio 0.85
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

import pandas as pd


def prepare_split(
    input_path: str,
    train_output: str = "data/processed_data.parquet",
    eval_output: str = "data/processed_data_eval.parquet",
    train_ratio: float = 0.80,
    purge_window: int = 50,
    verbose: bool = True
):
    """
    Split processed data into train and eval sets with purge window.
    
    Args:
        input_path: Path to full processed data parquet file.
        train_output: Output path for training data.
        eval_output: Output path for evaluation data.
        train_ratio: Fraction of data for training (default: 0.80).
        purge_window: Hours to skip between train and eval (default: 50).
        verbose: Print statistics.
    """
    # Load data
    if verbose:
        print(f"\n{'='*60}")
        print("Train/Eval Data Split for OverfittingGuard Signal 3")
        print(f"{'='*60}")
        print(f"\nLoading: {input_path}")
    
    df = pd.read_parquet(input_path)
    n_samples = len(df)
    
    if verbose:
        print(f"Total rows: {n_samples:,}")
        print(f"Date range: {df.index.min()} → {df.index.max()}")
    
    # Calculate split index
    idx_split = int(n_samples * train_ratio)
    
    # Split with purge window
    train_df = df.iloc[:idx_split]
    eval_df = df.iloc[idx_split + purge_window:]
    
    # Ensure output directories exist
    os.makedirs(os.path.dirname(train_output) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(eval_output) or ".", exist_ok=True)
    
    # Save
    train_df.to_parquet(train_output)
    eval_df.to_parquet(eval_output)
    
    if verbose:
        print(f"\n{'─'*60}")
        print("TRAIN DATA")
        print(f"{'─'*60}")
        print(f"  Output: {train_output}")
        print(f"  Rows:   {len(train_df):,} ({len(train_df)/n_samples*100:.1f}%)")
        print(f"  Start:  {train_df.index.min()}")
        print(f"  End:    {train_df.index.max()}")
        
        print(f"\n{'─'*60}")
        print(f"PURGE WINDOW: {purge_window} hours skipped")
        print(f"{'─'*60}")
        
        print(f"\n{'─'*60}")
        print("EVAL DATA")
        print(f"{'─'*60}")
        print(f"  Output: {eval_output}")
        print(f"  Rows:   {len(eval_df):,} ({len(eval_df)/n_samples*100:.1f}%)")
        print(f"  Start:  {eval_df.index.min()}")
        print(f"  End:    {eval_df.index.max()}")
        
        # Validation
        print(f"\n{'='*60}")
        print("VALIDATION")
        print(f"{'='*60}")
        
        gap = (eval_df.index.min() - train_df.index.max()).total_seconds() / 3600
        print(f"  Gap between train end and eval start: {gap:.0f} hours")
        
        if train_df.index.max() < eval_df.index.min():
            print("  ✅ No temporal overlap - Data leakage prevented")
        else:
            print("  ❌ WARNING: Temporal overlap detected!")
        
        print(f"\n{'='*60}")
        print("USAGE")
        print(f"{'='*60}")
        print("""
In src/config/training.py:

    @dataclass
    class TQCTrainingConfig:
        data_path: str = "{train}"
        eval_data_path: str = "{eval}"
        
This enables Signal 3 (Train/Eval divergence) in OverfittingGuardCallbackV2.
""".format(train=train_output, eval=eval_output))
    
    return train_df, eval_df


def main():
    parser = argparse.ArgumentParser(
        description="Prepare train/eval data split for OverfittingGuard"
    )
    parser.add_argument(
        "--input", "-i",
        default="data/processed_data_full.parquet",
        help="Input parquet file with full processed data"
    )
    parser.add_argument(
        "--train-output",
        default="data/processed_data.parquet",
        help="Output path for training data"
    )
    parser.add_argument(
        "--eval-output",
        default="data/processed_data_eval.parquet",
        help="Output path for evaluation data"
    )
    parser.add_argument(
        "--train-ratio", "-r",
        type=float,
        default=0.80,
        help="Fraction of data for training (default: 0.80)"
    )
    parser.add_argument(
        "--purge-window", "-p",
        type=int,
        default=50,
        help="Hours to skip between train and eval (default: 50)"
    )
    
    args = parser.parse_args()
    
    # Check input exists
    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        print("\nExpected: A parquet file with processed data.")
        print("You may need to run data preprocessing first.")
        sys.exit(1)
    
    prepare_split(
        input_path=args.input,
        train_output=args.train_output,
        eval_output=args.eval_output,
        train_ratio=args.train_ratio,
        purge_window=args.purge_window,
    )


if __name__ == "__main__":
    main()

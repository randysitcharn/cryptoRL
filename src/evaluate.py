# -*- coding: utf-8 -*-
"""
evaluate.py - CLI wrapper for model evaluation and backtesting.

Provides command-line interface for evaluating trained TQC models.
Uses the unified BacktestRunner from src.evaluation.runner.

Usage:
    python src/evaluate.py --model models/best_model.zip --data data/processed.csv
    python src/evaluate.py --help
"""

import argparse

from src.evaluation.config import EvaluationConfig
from src.evaluation.runner import BacktestRunner


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate a trained TQC model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data arguments
    parser.add_argument(
        '--model', type=str, default='models/best_model.zip',
        help='Path to the trained model'
    )
    parser.add_argument(
        '--data', type=str, default='data/processed/BTC-USD_processed.csv',
        help='Path to processed data (CSV or Parquet)'
    )
    parser.add_argument(
        '--format', type=str, default='auto', choices=['auto', 'csv', 'parquet'],
        help='Data format (auto-detect from extension if "auto")'
    )

    # Environment arguments
    parser.add_argument(
        '--window', type=int, default=64,
        help='Window size for Transformer'
    )
    parser.add_argument(
        '--commission', type=float, default=0.0006,
        help='Transaction commission (0.0006 = 0.06%%)'
    )
    parser.add_argument(
        '--initial-balance', type=float, default=10000.0,
        help='Initial portfolio balance'
    )

    # Split arguments
    parser.add_argument(
        '--split', type=str, default='test', choices=['train', 'val', 'test'],
        help='Which data split to evaluate on'
    )
    parser.add_argument(
        '--train-ratio', type=float, default=0.7,
        help='Ratio of data for training'
    )
    parser.add_argument(
        '--val-ratio', type=float, default=0.15,
        help='Ratio of data for validation'
    )
    parser.add_argument(
        '--use-splitter', action='store_true',
        help='Use TimeSeriesSplitter for proper train/val/test splits'
    )

    # Output arguments
    parser.add_argument(
        '--output', type=str, default='results/',
        help='Output directory for plots'
    )
    parser.add_argument(
        '--no-plots', action='store_true',
        help='Skip generating plots'
    )
    parser.add_argument(
        '--plots', type=str, nargs='+',
        default=['portfolio_drawdown', 'actions_distribution', 'strategy_comparison',
                 'returns_distribution', 'position_timeline'],
        help='Plot types to generate'
    )

    # Other arguments
    parser.add_argument(
        '--quiet', action='store_true',
        help='Reduce output verbosity'
    )

    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()

    # Auto-detect data format
    data_format = args.format
    if data_format == 'auto':
        if args.data.endswith('.parquet'):
            data_format = 'parquet'
        else:
            data_format = 'csv'

    # Create configuration
    config = EvaluationConfig(
        data_path=args.data,
        data_format=data_format,
        model_path=args.model,
        window_size=args.window,
        commission=args.commission,
        initial_balance=args.initial_balance,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        eval_split=args.split,
        use_time_series_splitter=args.use_splitter or data_format == 'csv',
        output_dir=f"{args.output}/{args.split}" if args.output else f"results/{args.split}",
        save_plots=not args.no_plots,
        plot_types=args.plots,
        verbose=not args.quiet,
    )

    # Run evaluation
    runner = BacktestRunner(config)
    results = runner.run()

    # Print summary
    if not args.quiet:
        print(f"\nPlots saved to: {config.output_dir}/")
        for plot_type in config.plot_types:
            print(f"  - {plot_type}.png")

    return results


if __name__ == "__main__":
    main()

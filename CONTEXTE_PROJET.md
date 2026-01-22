# CONTEXTE PROJET - CryptoRL

> **Generated:** 2026-01-17 18:35 UTC | **Auditor:** Technical Audit Scan

---

## Git Status

| Property | Value |
|----------|-------|
| **Branch** | `feat/training-speed-optimization` |
| **Last Commit** | `4a84494` - refactor(scripts): nettoyage - garder uniquement 3 scripts essentiels |
| **Status** | Up to date with `origin/feat/training-speed-optimization` |

---

## Project Structure (Tree View)

```
cryptoRL/
├── configs/                    # (empty - config in src/config/)
├── data/
│   ├── processed/              # Processed parquet files
│   ├── raw/                    # Raw data placeholder
│   ├── raw_historical/         # Historical OHLCV data
│   │   ├── BTC_1h.csv
│   │   ├── ETH_1h.csv
│   │   ├── SPX_1h.csv
│   │   ├── DXY_1h.csv
│   │   ├── NASDAQ_1h.csv
│   │   └── multi_asset_historical.csv
│   └── processed_data.parquet
├── docs/
│   ├── CURRENT_SERVER.md       # Remote server connection info
│   ├── CURRICULUM_TUNING.md    # Curriculum learning documentation
│   ├── SERVER_SETUP.md         # Server setup guide
│   └── WFO_HYPERPARAMETERS.md  # WFO hyperparameters reference
├── logs/
│   ├── tensorboard/            # Legacy TensorBoard logs
│   ├── tensorboard_tqc/        # TQC-specific logs
│   ├── wfo/                    # Walk-Forward Optimization logs
│   │   └── hmm/segment_0/      # HMM training events
│   └── demo/                   # Demo run logs
├── notebooks/                  # (empty - Jupyter notebooks)
├── results/
│   ├── hmm_segments/           # 12 HMM regime visualizations
│   ├── hmm_wfo/                # 29 WFO HMM analysis files
│   │   ├── hmm_wfo_metrics.csv
│   │   └── segment_*_regimes.png
│   └── *.png                   # Various result plots
├── scripts/
│   ├── run_full_wfo.py         # Main WFO orchestration (~1600 lines)
│   ├── analyze_segment.py      # Segment analysis utility
│   └── analyze_tensorboard.py  # TensorBoard log analyzer
├── src/
│   ├── config/                 # Configuration modules
│   │   ├── base.py
│   │   ├── constants.py
│   │   └── training.py
│   ├── data/                   # Data loading
│   │   └── dataset.py
│   ├── data_engineering/       # Feature engineering
│   │   ├── features.py         # Feature calculations (FFD, HMM, etc.)
│   │   ├── loader.py           # Data loaders
│   │   ├── manager.py          # RegimeDetector (HMM)
│   │   ├── processor.py        # Data processing
│   │   └── splitter.py         # Train/test splitting
│   ├── evaluation/             # Evaluation & backtesting
│   │   ├── backtest.py
│   │   ├── runner.py
│   │   └── visualize.py
│   ├── models/                 # Neural network models
│   │   ├── foundation.py       # MAE Foundation Model (9.5K)
│   │   ├── rl_adapter.py       # FoundationFeatureExtractor (13K)
│   │   └── agent.py            # Agent wrapper (4K)
│   ├── training/               # Training infrastructure
│   │   ├── batch_env.py        # GPU-accelerated BatchCryptoEnv (unified env)
│   │   ├── callbacks.py        # Training callbacks (32K)
│   │   ├── train_agent.py      # TQC training entrypoint (40K)
│   │   ├── train_foundation.py # MAE training entrypoint (16K)
│   │   └── wrappers.py         # Risk management wrappers (11K)
│   └── utils/                  # Utilities
├── tests/                      # Unit tests
├── requirements.txt            # Python dependencies
└── ruff.toml                   # Linter configuration
```

---

## Key Files & Purposes

| File | Purpose | Size |
|------|---------|------|
| `scripts/run_full_wfo.py` | Main WFO orchestration (HMM -> MAE -> TQC -> Eval) | ~1600 lines |
| `src/training/train_agent.py` | TQC training with Foundation Model | 40 KB |
| `src/training/batch_env.py` | GPU-accelerated vectorized environment (unified) | 35 KB |
| `src/training/callbacks.py` | Curriculum learning (3-Phase), logging | 32 KB |
| `src/models/rl_adapter.py` | FoundationFeatureExtractor (MAE -> TQC) | 13 KB |
| `src/models/foundation.py` | CryptoMAE autoencoder model | 9.5 KB |

---

## Log Directories Status

| Directory | Status | Content |
|-----------|--------|---------|
| `logs/tensorboard/` | Present | 2 legacy TQC runs |
| `logs/tensorboard_tqc/` | Present | Churn analysis runs |
| `logs/wfo/` | Present | WFO training logs |
| `logs/wfo/hmm/` | Present | HMM training events (segment_0) |
| `logs/demo/` | Present | Demo/test runs (TQC_1 - TQC_9) |

---

## Results Directory Status

| Directory | Files | Content |
|-----------|-------|---------|
| `results/` | 5 | HMM visualizations, equity curves |
| `results/hmm_segments/` | 12 | Segment regime plots (00-12) |
| `results/hmm_wfo/` | 32 | Segment plots + metrics CSV + summary |

---

## Current Hyperparameters (from `docs/WFO_HYPERPARAMETERS.md`)

### WFO Windows

| Parameter | Value | Computed |
|-----------|-------|----------|
| train_months | 12 | 8,640 rows |
| test_months | 3 | 2,160 rows |
| step_months | 3 | 2,160 rows |
| hours_per_month | 720 | - |

### Training

| Parameter | Value |
|-----------|-------|
| tqc_timesteps | 30,000,000 |
| mae_epochs | 90 |
| n_envs | 1024 |
| batch_size | 2048 |
| buffer_size | 2,500,000 |
| learning_rate | 1e-4 |
| gamma | 0.95 |

### Curriculum (3-Phase)

| Phase | Progress | Churn | Smooth |
|-------|----------|-------|--------|
| 1 - Discovery | 0-10% | 0.0 -> 0.10 | 0.0 |
| 2 - Discipline | 10-30% | 0.10 -> 0.50 | 0.0 -> 0.02 |
| 3 - Consolidation | 30-100% | 0.50 (fixed) | 0.02 (fixed) |

### Reward Function

| Parameter | Value |
|-----------|-------|
| SCALE | 100.0 |
| reward_scaling | 1.0 |
| curriculum_lambda | 0.4 (max) |
| action_discretization | 0.1 |

---

## Remote Server

| Property | Value |
|----------|-------|
| Host | `158.51.110.52` |
| Port | `20941` |
| User | `root` |
| Provider | vast.ai |
| TensorBoard | Port 8081 |

```bash
# Quick connect
ssh -p 20941 root@158.51.110.52

# TensorBoard tunnel
ssh -p 20941 -L 8081:localhost:8081 root@158.51.110.52
```

---

## Recent Training Results (Segment 0)

| Metric | Value |
|--------|-------|
| Total Steps | 54,000,000 |
| Duration | 3h 13m |
| FPS | ~4,800 |
| Sharpe Ratio | 4.73 |
| PnL | +78.28% |
| Max Drawdown | 8.34% |
| Total Trades | **3** |
| B&H Return | +77.76% |
| Alpha | +0.52% |
| Market | BULLISH |

---

## Known Issue: Reward Imbalance

### Observed Amplitudes (from training logs)

| Component | Typical Value | Relative Scale |
|-----------|---------------|----------------|
| reward/pnl_component | +0.027 | 1.0x (baseline) |

### ✅ Solution Implémentée: MORL Architecture

Le problème de pénalités excessives a été résolu en passant à l'architecture MORL:

```python
# src/training/batch_env.py - Architecture MORL
reward = r_perf + curriculum_lambda * w_cost * r_cost * MAX_PENALTY_SCALE
```

où:
- `r_perf`: Log-returns (objectif performance)  
- `w_cost ∈ [0, 1]`: Paramètre MORL dans l'observation
- `curriculum_lambda ∈ [0, 0.4]`: Progression contrôlée

**Avantage:** L'agent apprend à adapter son comportement à différentes valeurs de w_cost.

---

## Architecture Diagram

```
+-------------------------------------------------------------------------+
|                         CryptoRL - WFO Pipeline                          |
+-------------------------------------------------------------------------+
|  scripts/run_full_wfo.py                                                 |
|  +-- [1] Data Loading: CSV/Parquet -> raw_training_data                  |
|  +-- [2] Feature Engineering: FFD, Z-Score, Parkinson, Garman-Klass     |
|  +-- [3] HMM Regime Detection: 4-state regime probabilities              |
|  +-- [4] MAE Pre-training: Foundation encoder (90 epochs)                |
|  +-- [5] TQC Training: BatchCryptoEnv (1024 envs, 54M steps)            |
|  +-- [6] OOS Evaluation: Test window backtest                            |
+-------------------------------------------------------------------------+
|  Training Environment (unified)                                          |
|  +-- BatchCryptoEnv (batch_env.py) - GPU/CPU, supports n_envs=1 eval    |
+-------------------------------------------------------------------------+
|  Callbacks (callbacks.py)                                                |
|  +-- ThreePhaseCurriculumCallback: churn + smooth curriculum            |
|  +-- RotatingCheckpointCallback: Disk optimization                       |
|  +-- TrainingMetricsCallback: WFO mode NAV logging                       |
|  +-- BestModelCleanerCallback: torch.compile artifact cleanup            |
+-------------------------------------------------------------------------+
```

---

*End of Project Context Report*

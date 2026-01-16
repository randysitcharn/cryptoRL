# CryptoRL - Project Context

> **Generated:** 2026-01-16 | **Branch:** `feat/training-speed-optimization`

---

## 1. Project Structure (Tree View)

```
cryptoRL/
├── configs/                    # (empty - config in src/)
├── data/
│   ├── processed/
│   ├── processed_data.parquet
│   ├── raw/
│   └── raw_historical/
│       ├── BTC_1h.csv
│       ├── ETH_1h.csv
│       ├── SPX_1h.csv
│       ├── DXY_1h.csv
│       ├── NASDAQ_1h.csv
│       └── multi_asset_historical.csv   # Main data file
├── docs/
│   ├── CURRENT_SERVER.md       # Active server credentials
│   ├── SERVER_SETUP.md         # Server initialization guide
│   └── WFO_HYPERPARAMETERS.md  # Hyperparameter documentation
├── logs/
│   ├── tensorboard/            # General TQC runs
│   ├── tensorboard_tqc/        # Churn analysis runs
│   ├── demo/tensorboard/       # Demo runs (TQC_1 - TQC_9)
│   └── wfo/
│       └── hmm/segment_0/      # WFO Segment 0 logs
├── notebooks/                  # (empty)
├── results/
│   ├── hmm_segments/           # 12 segment regime plots
│   ├── hmm_wfo/                # 31 WFO result files
│   │   ├── hmm_wfo_metrics.csv
│   │   ├── hmm_wfo_summary.png
│   │   └── segment_*_regimes.png
│   └── wfo_equity_curve.png
├── scripts/
│   ├── run_full_wfo.py         # Main WFO pipeline (1551 lines)
│   ├── analyze_regime_performance.py
│   ├── analyze_segment.py
│   ├── check_reward_balance.py
│   ├── emergency_retrain.py
│   ├── hmm_wfo_viz.py
│   ├── plot_wfo_equity.py
│   ├── test_hmm_fix.py
│   ├── visualize_hmm_per_segment.py
│   └── visualize_hmm_results.py
├── src/
│   ├── config/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── constants.py
│   │   └── training.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py
│   ├── data_engineering/
│   │   ├── __init__.py
│   │   ├── features.py          # Feature engineering
│   │   ├── historical_downloader.py
│   │   ├── loader.py
│   │   ├── manager.py           # RegimeDetector (HMM)
│   │   ├── processor.py
│   │   └── splitter.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── analyze_rewards.py
│   │   ├── backtest.py
│   │   ├── check_activity.py
│   │   ├── check_mae.py
│   │   ├── config.py
│   │   ├── export_metrics.py
│   │   ├── runner.py
│   │   └── visualize.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── agent.py             # AgentFactory
│   │   ├── foundation.py        # MAE Foundation Model
│   │   ├── rl_adapter.py
│   │   └── transformer_policy.py
│   ├── optimization/
│   │   ├── __init__.py
│   │   └── tune.py              # Optuna tuning
│   ├── training/
│   │   ├── __init__.py
│   │   ├── batch_env.py         # GPU BatchCryptoEnv (777 lines)
│   │   ├── callbacks.py         # Training callbacks (716 lines)
│   │   ├── clipped_optimizer.py
│   │   ├── env.py               # CryptoTradingEnv (649 lines)
│   │   ├── train_agent.py       # TQC training (841 lines)
│   │   ├── train_foundation.py  # MAE pre-training
│   │   └── wrappers.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── hardware.py
│   │   ├── metrics.py
│   │   └── reproducibility.py
│   ├── evaluate.py
│   └── train_demo.py
├── tests/
│   ├── debug/
│   │   ├── check_regimes.py
│   │   ├── check_shapes.py
│   │   └── debug_eth_stationarity.py
│   ├── profile_gpu_env.py
│   ├── test_agent_init.py
│   ├── test_env.py
│   ├── test_env_logic.py
│   ├── test_reproducibility.py
│   ├── test_reward.py
│   ├── test_splitting.py
│   ├── test_stationarity.py
│   └── test_vol_scaling.py
├── venv/
├── requirements.txt
├── ruff.toml
└── CONTEXTE_PROJET.md
```

---

## 2. Key Files

### Entry Points

| File | Purpose | Lines |
|------|---------|-------|
| `scripts/run_full_wfo.py` | Walk-Forward Optimization pipeline | 1,551 |
| `src/train_demo.py` | Quick training demo | ~200 |
| `src/training/train_agent.py` | TQC agent training (main) | 841 |
| `src/training/train_foundation.py` | MAE foundation model pre-training | ~300 |
| `src/evaluate.py` | Model evaluation | ~150 |

### Configuration Files

| File | Purpose |
|------|---------|
| `src/config/base.py` | Base configuration classes |
| `src/config/constants.py` | Global constants (DEVICE, SEED, EXCLUDE_COLS) |
| `src/config/training.py` | TrainingConfig, TQCTrainingConfig dataclasses |
| `requirements.txt` | Python dependencies (13 packages) |
| `ruff.toml` | Linter configuration |

### Documentation Files

| File | Purpose |
|------|---------|
| `docs/CURRENT_SERVER.md` | Active vast.ai server credentials |
| `docs/SERVER_SETUP.md` | Server initialization procedure |
| `docs/WFO_HYPERPARAMETERS.md` | Hyperparameter documentation |

---

## 3. Active Server Configuration

| Parameter | Value |
|-----------|-------|
| **SSH Host** | `142.171.48.138` |
| **SSH Port** | `24256` |
| **SSH User** | `root` |
| **Platform** | vast.ai GPU server |

### Quick Commands

```bash
# Connect
ssh -p 24256 root@142.171.48.138

# Copy data
scp -P 24256 -r data/raw_historical/ root@142.171.48.138:/workspace/cryptoRL/data/

# TensorBoard tunnel
ssh -p 24256 -L 6006:localhost:16006 root@142.171.48.138
```

---

## 4. WFO Configuration (Current)

### Window Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| train_months | **18** | Training window (was 12) |
| test_months | 3 | Test/OOS window |
| step_months | 3 | Rolling step size |
| hours_per_month | 720 | 30 days * 24 hours |

### Computed Segments

| Metric | Value |
|--------|-------|
| train_rows | 18 * 720 = 12,960 |
| test_rows | 3 * 720 = 2,160 |
| step_rows | 3 * 720 = 2,160 |

### Training Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| tqc_timesteps | 30,000,000 | 30M steps per segment |
| mae_epochs | 90 | Foundation model epochs |
| learning_rate | 1e-4 | Conservative for stability |
| buffer_size | 2,500,000 | 2.5M replay buffer |
| n_envs | 512 | Parallel environments |
| batch_size | 2,048 | Large batch for GPU |
| gamma | 0.99 | Discount factor |

### Curriculum Learning (3-Phase)

| Phase | Progress | Churn | Smooth |
|-------|----------|-------|--------|
| Discovery | 0% - 20% | 0.0 -> 0.10 | 0.0 |
| Discipline | 20% - 60% | 0.10 -> 0.50 | 0.0 -> 0.02 |
| Consolidation | 60% - 100% | 0.50 (fixed) | 0.02 (fixed) |

### GPU Acceleration

| Parameter | Value | Description |
|-----------|-------|-------------|
| use_batch_env | True | BatchCryptoEnv enabled |
| target_volatility | 0.05 | 5% vol scaling |
| vol_window | 24 | 24h rolling window |
| max_leverage | 2.0 | Conservative scaling |

---

## 5. Dependencies

| Package | Purpose |
|---------|---------|
| `gymnasium` | RL environment interface |
| `stable-baselines3[extra]` | RL algorithms |
| `sb3-contrib` | TQC algorithm |
| `torch` | Deep learning framework |
| `pandas` | Data manipulation |
| `numpy` | Numerical computing |
| `matplotlib` | Visualization |
| `yfinance` | Yahoo Finance API |
| `pandas-ta` | Technical indicators |
| `ccxt` | Crypto exchange API |
| `statsmodels` | Statistical models |
| `hmmlearn` | Hidden Markov Models |
| `pyarrow` | Parquet file support |

---

## 6. Log Directories

### TensorBoard Logs

| Path | Content |
|------|---------|
| `logs/tensorboard/` | General TQC runs (TQC_1, TQC_2) |
| `logs/tensorboard_tqc/` | Churn analysis runs |
| `logs/demo/tensorboard/` | Demo training (TQC_1 - TQC_9) |
| `logs/wfo/` | **WFO training logs** |
| `logs/wfo/hmm/segment_0/` | Segment 0 HMM events |

### Text Logs

| File | Purpose |
|------|---------|
| `logs/churn_analysis.log` | Churn analysis output |
| `logs/mae_training.log` | MAE training log |
| `logs/input_features.txt` | Feature list |
| `logs/regime_classification_report.txt` | HMM regime report |

---

## 7. Data Files

### Raw Data

| File | Format | Content |
|------|--------|---------|
| `data/raw_historical/multi_asset_historical.csv` | CSV | Combined OHLCV all assets |
| `data/raw_historical/BTC_1h.csv` | CSV | Bitcoin hourly OHLCV |
| `data/raw_historical/ETH_1h.csv` | CSV | Ethereum hourly OHLCV |
| `data/raw_historical/SPX_1h.csv` | CSV | S&P 500 hourly OHLCV |
| `data/raw_historical/DXY_1h.csv` | CSV | Dollar Index hourly OHLCV |
| `data/raw_historical/NASDAQ_1h.csv` | CSV | NASDAQ hourly OHLCV |

### Processed Data

| File | Format | Content |
|------|--------|---------|
| `data/processed_data.parquet` | Parquet | Processed features |

---

## 8. Results Directory

| Path | Files | Content |
|------|-------|---------|
| `results/hmm_segments/` | 12 | Per-segment regime visualizations |
| `results/hmm_wfo/` | 31 | WFO metrics + summary plots |
| `results/hmm_wfo/hmm_wfo_metrics.csv` | 1 | Aggregated metrics |
| `results/wfo_equity_curve.png` | 1 | Equity curve visualization |

---

## 9. Git Status

### Current Branch

```
feat/training-speed-optimization
```

### Recent Commits

| Hash | Message |
|------|---------|
| `a7439bf` | docs: add TensorBoard reconfiguration for WFO logs |
| `2673419` | feat(wfo): smart loader for CSV/Parquet input |
| `167c2ca` | config(wfo): change train period 12 -> 18 months |
| `3a24eab` | config(curriculum): extend plateau phase 20% -> 40% |
| `8747544` | fix(wfo): fail fast on OOM errors instead of continue |
| `bb8d39c` | feat(wfo): add rotating safety checkpoint (disk opt) |
| `d3a74d1` | feat(train): add TrainingMetricsCallback for WFO mode |
| `930cbfc` | fix(train): remove intermediate checkpoints in WFO mode |
| `cc93184` | fix(wfo): explicitly set eval_data_path=None to disable EvalCallback |
| `81f32ca` | fix(train): handle eval_data_path=None for WFO mode |
| `d4ab5c5` | fix(wfo): prevent log deletion for parallel execution support |
| `ca78c60` | config(wfo): revert lr to 1e-4 for stability |
| `324f484` | fix(wfo): code review fixes + n_envs=512 for dual-GPU |
| `223a762` | config(wfo): increase learning_rate 1e-4 -> 3e-4 |
| `347d676` | feat(wfo): add B&H benchmark comparison and alpha calculation |

### Untracked Files

| File | Description |
|------|-------------|
| `CONTEXTE_PROJET.md` | This file |
| `docs/WFO_HYPERPARAMETERS.md` | Hyperparameter documentation |
| `results/hmm_segments/` | Segment regime plots |
| `results/hmm_wfo/` | WFO results |
| `results/wfo_equity_curve.png` | Equity curve |

---

## 10. Architecture Diagram

```
+---------------------------------------------------------------------+
|                           CryptoRL                                   |
+---------------------------------------------------------------------+
|  scripts/run_full_wfo.py                                            |
|  +-- Walk-Forward Optimization Pipeline                             |
|      |                                                              |
|      +-- [1] Data Loading: _load_raw_data() -> CSV/Parquet         |
|      +-- [2] Feature Engineering: FeatureEngineer.compute_features() |
|      +-- [3] HMM Regime Detection: RegimeDetector.fit_transform()   |
|      +-- [4] MAE Pre-training: train_foundation.py                  |
|      +-- [5] TQC Training: train_agent.py + BatchCryptoEnv          |
|      +-- [6] OOS Evaluation: Backtest on test window                |
+---------------------------------------------------------------------+
|  Training Environments                                              |
|  +-- CryptoTradingEnv (src/training/env.py) - CPU single env       |
|  +-- BatchCryptoEnv (src/training/batch_env.py) - GPU 512 envs     |
+---------------------------------------------------------------------+
|  Callbacks (src/training/callbacks.py)                              |
|  +-- ThreePhaseCurriculumCallback: Discovery -> Discipline -> Cons. |
|  +-- RotatingCheckpointCallback: Disk optimization (keep last only) |
|  +-- TrainingMetricsCallback: WFO mode logging                      |
|  +-- EvalCallback: Periodic evaluation                              |
+---------------------------------------------------------------------+
|  Feature Engineering (src/data_engineering/)                        |
|  +-- Log Returns, Z-Score, FFD (Fractional Differentiation)        |
|  +-- Parkinson / Garman-Klass Volatility                           |
|  +-- Multi-asset: BTC, ETH, SPX, DXY, NASDAQ                       |
|  +-- HMM Regime probabilities (Prob_0, Prob_1, Prob_2, Prob_3)     |
+---------------------------------------------------------------------+
|  Evaluation (src/evaluation/)                                       |
|  +-- Backtesting with regime-aware metrics                          |
|  +-- WFO equity curve generation                                    |
|  +-- Buy & Hold benchmark comparison                                |
+---------------------------------------------------------------------+
```

---

## 11. Code Statistics

| Category | Lines |
|----------|-------|
| `scripts/run_full_wfo.py` | 1,551 |
| `src/training/train_agent.py` | 841 |
| `src/training/batch_env.py` | 777 |
| `src/training/callbacks.py` | 716 |
| `src/training/env.py` | 649 |
| **Total (key files)** | 4,534 |

---

## 12. Test Files

| Test File | Purpose |
|-----------|---------|
| `test_agent_init.py` | Agent initialization |
| `test_env.py` | Environment basic tests |
| `test_env_logic.py` | Environment logic |
| `test_reproducibility.py` | Seed reproducibility |
| `test_reward.py` | Reward function |
| `test_splitting.py` | Data splitting |
| `test_stationarity.py` | Feature stationarity |
| `test_vol_scaling.py` | Volatility scaling |
| `profile_gpu_env.py` | GPU environment profiling |

---

*End of Project Context Report*

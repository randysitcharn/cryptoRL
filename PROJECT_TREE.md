# CryptoRL - Project Structure

```
cryptoRL/
|
|-- .github/workflows/          # CI/CD
|   |-- ci.yml
|
|-- configs/                    # Configuration files (YAML, JSON)
|
|-- data/
|   |-- raw/                    # Raw market data
|   |-- raw_historical/         # Historical OHLCV data
|   |   |-- BTC_1h.csv
|   |   |-- ETH_1h.csv
|   |   |-- SPX_1h.csv
|   |   |-- DXY_1h.csv
|   |   |-- NASDAQ_1h.csv
|   |   |-- multi_asset_historical.csv
|   |-- processed/              # Processed features
|   |-- wfo/                    # Walk-Forward Optimization data
|       |-- segment_N/
|           |-- train.parquet
|           |-- eval.parquet
|           |-- test.parquet
|
|-- docs/
|   |-- audit/                  # Audit reports & findings
|   |   |-- AUDIT_AGENT_PASSIF_RED_FLAGS.md
|   |   |-- AUDIT_MORL.md
|   |   |-- DATA_PIPELINE_AUDIT_REPORT.md
|   |   |-- MASTER_PLAN_AUDIT_*.md
|   |-- design/                 # Architecture & design docs
|   |   |-- DATA_PIPELINE_DESIGN.md
|   |   |-- DROPOUT_TQC_DESIGN.md
|   |   |-- MORL_DESIGN.md
|   |   |-- WFO_OVERFITTING_GUARD.md
|   |-- server/                 # Server configuration
|   |   |-- CURRENT_SERVER.md   # Active server SSH info
|   |   |-- SERVER_SETUP.md
|   |-- CONCEPTS_CLES.md
|   |-- PROJECT_ANALYSIS.md
|
|-- logs/
|   |-- wfo/                    # WFO training logs
|   |   |-- segment_N/          # TensorBoard events
|   |-- tensorboard/            # General TensorBoard logs
|   |-- demo/                   # Demo/test runs
|
|-- models/
|   |-- wfo/                    # WFO segment models
|       |-- segment_N/
|           |-- hmm.pkl         # HMM regime detector
|           |-- scaler.pkl      # Feature scaler
|
|-- notebooks/                  # Jupyter notebooks for analysis
|
|-- results/
|   |-- hmm_audit/              # HMM audit results
|   |-- mae_audit/              # MAE audit results
|   |-- tqc_audit/              # TQC audit results
|   |   |-- YYYYMMDD_HHMMSS/
|   |       |-- report.md
|   |       |-- metrics.json
|   |       |-- plots/
|   |-- normalization_audit/
|   |-- plots/
|   |-- wfo_results.csv         # WFO summary results
|
|-- scripts/
|   |-- run_full_wfo.py         # Main WFO training script
|   |-- audit_pipeline.py       # Comprehensive audit tool
|   |-- init_server.ps1         # Server initialization (PowerShell)
|   |-- debug_inputs.py         # Observation diagnostics
|
|-- src/
|   |-- config/                 # Configuration classes
|   |   |-- __init__.py
|   |   |-- training.py         # TQCTrainingConfig, WFOTrainingConfig
|   |
|   |-- data/                   # Data loading & datasets
|   |   |-- __init__.py
|   |   |-- dataset.py          # CryptoDataset
|   |
|   |-- data_engineering/       # Feature engineering
|   |   |-- __init__.py
|   |   |-- features.py         # FeatureEngineer
|   |   |-- manager.py          # DataManager, RegimeDetector
|   |
|   |-- env/                    # Gymnasium environments
|   |   |-- __init__.py
|   |   |-- crypto_env.py       # Base CryptoTradingEnv
|   |   |-- batch_env.py        # BatchCryptoEnv (vectorized)
|   |
|   |-- models/                 # Neural network models
|   |   |-- __init__.py
|   |   |-- foundation.py       # CryptoMAE (Masked Autoencoder)
|   |   |-- film.py             # FiLM conditioning layer
|   |   |-- rl_adapter.py       # FoundationFeatureExtractor
|   |   |-- tqc_dropout_policy.py  # TQCDropoutPolicy
|   |   |-- robust_actor.py     # RobustDropoutActor
|   |
|   |-- training/               # Training utilities
|   |   |-- __init__.py
|   |   |-- train_agent.py      # TQC training setup
|   |   |-- callbacks.py        # MORLCurriculumCallback, etc.
|   |   |-- batch_env.py        # BatchCryptoEnv with MORL
|   |
|   |-- utils/                  # Utilities
|       |-- __init__.py
|       |-- metrics.py          # Sharpe, drawdown calculations
|
|-- tests/                      # Unit tests
|   |-- test_*.py
|
|-- weights/
|   |-- wfo/                    # Trained model weights
|       |-- segment_N/
|           |-- encoder.pth     # MAE encoder weights
|           |-- mae_full.pth    # Full MAE model
|           |-- tqc.zip         # TQC model (SB3 format)
|           |-- checkpoints/    # Training checkpoints
|
|-- .gitignore
|-- requirements.txt
|-- pytest.ini
|-- ruff.toml
|-- CONTEXTE_PROJET.md          # Project context (French)
|-- PROJECT_TREE.md             # This file
```

## Key Components

### Data Pipeline
```
Raw CSV -> FeatureEngineer -> RegimeDetector (HMM) -> Scaler -> Parquet
```

### Model Architecture
```
Observation (55 features)
    |
    +-- Tech Features (50) --> MAE Encoder --> Latent (256)
    |                                              |
    +-- HMM Context (5) -----> FiLM -----------> Modulated Latent
                                                    |
                                              Fusion Projector
                                                    |
                                              Actor/Critic (TQC)
```

### Training Pipeline (WFO)
```
1. Data Engineering (features, HMM regimes)
2. MAE Pre-training (self-supervised)
3. TQC Training (with frozen MAE encoder)
4. Evaluation & Checkpointing
5. Next Segment (model inheritance)
```

## Important Files

| File | Description |
|------|-------------|
| `scripts/run_full_wfo.py` | Main training script |
| `scripts/audit_pipeline.py` | Audit tool (HMM, MAE, TQC, FiLM) |
| `scripts/init_server.ps1` | Server setup automation |
| `src/config/training.py` | Training hyperparameters |
| `src/models/rl_adapter.py` | MAE-TQC integration |
| `src/training/batch_env.py` | MORL environment |
| `docs/server/CURRENT_SERVER.md` | Active server info |

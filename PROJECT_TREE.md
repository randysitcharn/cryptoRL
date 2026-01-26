# PROJECT_TREE - CryptoRL

> Dernière mise à jour: 2026-01-26

## Arborescence du code

```
cryptoRL/
│
├── src/                              # Code source principal
│   │
│   ├── config/                       # Configuration centralisée
│   │   ├── __init__.py               # Exports: DEVICE, SEED, configs
│   │   ├── base.py                   # Config de base (device, seed)
│   │   ├── constants.py              # Constantes: MAE dims, HMM_CONTEXT_SIZE, EXCLUDE_COLS
│   │   ├── training.py               # TQCTrainingConfig, WFOTrainingConfig
│   │   └── validators.py             # ModelDimensionsValidator (détecte mismatches)
│   │
│   ├── data/                         # Chargement données
│   │   ├── __init__.py
│   │   └── dataset.py                # CryptoDataset (PyTorch Dataset pour MAE)
│   │
│   ├── data_engineering/             # Feature engineering
│   │   ├── __init__.py
│   │   ├── features.py               # FeatureEngineer (FFD, Z-Score, Parkinson, GK)
│   │   ├── historical_downloader.py  # Téléchargement données Yahoo Finance
│   │   ├── loader.py                 # Chargement CSV/Parquet
│   │   ├── manager.py                # RegimeDetector (HMM 4 états)
│   │   ├── processor.py              # Pipeline de preprocessing
│   │   └── splitter.py               # WFO train/eval/test splits
│   │
│   ├── env/                          # Environnement (legacy)
│   │   ├── __init__.py
│   │   └── rewards.py                # Fonctions de reward
│   │
│   ├── evaluation/                   # Évaluation et backtesting
│   │   ├── __init__.py
│   │   └── ensemble.py               # Ensemble de modèles WFO
│   │
│   ├── models/                       # Modèles ML/RL
│   │   ├── __init__.py
│   │   ├── agent.py                  # Création agent TQC
│   │   ├── foundation.py             # CryptoMAE (Masked AutoEncoder)
│   │   ├── layers.py                 # FiLMLayer (Feature-wise Linear Modulation)
│   │   ├── rl_adapter.py             # FoundationFeatureExtractor (MAE → TQC)
│   │   ├── robust_actor.py           # RobustActor (Dropout, LayerNorm)
│   │   ├── tqc_dropout_policy.py     # TQCDropoutPolicy (Spectral Norm)
│   │   └── transformer_policy.py     # TransformerFeatureExtractor (standalone)
│   │
│   ├── training/                     # Infrastructure d'entraînement
│   │   ├── __init__.py
│   │   ├── batch_env.py              # BatchCryptoEnv (GPU vectorisé)
│   │   ├── callbacks.py              # Callbacks SB3:
│   │   │                             #   - EntropyFloorCallback
│   │   │                             #   - MORLCurriculumCallback
│   │   │                             #   - UnifiedMetricsCallback
│   │   │                             #   - OverfittingGuardCallback
│   │   │                             #   - ModelEMACallback
│   │   ├── clipped_optimizer.py      # ClippedAdamW (gradient clipping)
│   │   ├── train_agent.py            # Entraînement TQC principal
│   │   ├── train_foundation.py       # Pre-training MAE
│   │   └── wrappers.py               # Wrappers Gymnasium
│   │
│   └── utils/                        # Utilitaires
│       ├── __init__.py
│       ├── hardware.py               # HardwareManager (GPU/CPU detection)
│       ├── metrics.py                # Métriques (Sharpe, Sortino, etc.)
│       └── reproducibility.py        # Seeds, reproductibilité
│
├── scripts/                          # Scripts d'exécution
│   ├── run_full_wfo.py               # Pipeline WFO complet
│   ├── clean_wfo.py                  # Nettoyage artefacts WFO
│   ├── audit_pipeline.py             # Audits (HMM, MAE, TQC, FiLM)
│   ├── audit_env.py                  # Audit environnement
│   ├── audit_hmm_features.py         # Audit features HMM
│   ├── audit_normalization.py        # Audit normalisation
│   ├── check_magnitude.py            # Vérification magnitudes
│   ├── debug_inputs.py               # Debug observations
│   ├── debug_model_actions.py        # Debug actions modèle
│   ├── make_oracle_data.py           # Génération données oracle
│   ├── prepare_train_eval_split.py   # Préparation splits
│   ├── regenerate_data.py            # Régénération données
│   ├── run_oracle_test.py            # Test oracle
│   ├── test_film_extractor.py        # Test FiLM (dry-run)
│   └── validate_fixes.py             # Validation corrections
│
├── tests/                            # Tests unitaires (pytest)
│   │
│   ├── # Architecture & Modèles
│   ├── test_film_extractor.py        # Tests FiLM + FoundationFeatureExtractor
│   ├── test_model_dimensions.py      # Tests validation dimensions
│   ├── test_wfo_foundation_config.py # Tests config WFO + Foundation
│   ├── test_dropout_policy.py        # Tests TQCDropoutPolicy
│   ├── test_robust_actor.py          # Tests RobustActor
│   ├── test_robustness_layer.py      # Tests couches robustesse
│   │
│   ├── # Environnement & Rewards
│   ├── test_env.py                   # Tests BatchCryptoEnv
│   ├── test_env_logic.py             # Tests logique env
│   ├── test_reward.py                # Tests reward shaping
│   ├── test_morl.py                  # Tests MORL (w_cost)
│   ├── test_vol_scaling.py           # Tests volatility scaling
│   │
│   ├── # Features & Data
│   ├── test_hmm_features.py          # Tests HMM (look-ahead bias)
│   ├── test_stationarity.py          # Tests stationnarité features
│   ├── test_feature_consistency.py   # Tests cohérence features
│   ├── test_data_leakage.py          # Tests data leakage
│   ├── test_splitting.py             # Tests WFO splits
│   │
│   ├── # Callbacks & Training
│   ├── test_callbacks.py             # Tests callbacks
│   ├── test_entropy_floor.py         # Tests EntropyFloorCallback
│   ├── test_entropy_floor_integration.py # Tests intégration entropy
│   ├── test_overfitting_guard_wfo.py # Tests OverfittingGuard
│   │
│   ├── # Évaluation
│   ├── test_ensemble.py              # Tests ensemble
│   ├── test_ensemble_sanity.py       # Tests sanity ensemble
│   ├── test_oracle.py                # Tests oracle
│   ├── test_pipeline_signal.py       # Tests pipeline signal
│   │
│   ├── # Utilitaires
│   ├── test_reproducibility.py       # Tests reproductibilité
│   ├── test_observation_noise.py     # Tests bruit observations
│   ├── test_robust_trend_stats.py    # Tests stats trend
│   └── profile_gpu_env.py            # Profiling GPU
│
├── docs/                             # Documentation
│   ├── audit/                        # Rapports d'audit
│   └── server/                       # Config serveur (CURRENT_SERVER.md)
│
├── data/                             # Données
│   ├── raw_historical/               # Données brutes OHLCV
│   └── wfo/segment_X/                # Données WFO par segment
│
├── logs/                             # Logs
│   └── wfo/segment_X/                # TensorBoard par segment
│
├── weights/                          # Checkpoints
│   ├── pretrained_encoder.pth        # MAE pré-entraîné global
│   └── wfo/segment_X/                # Checkpoints par segment
│
├── results/                          # Résultats
│   └── tqc_audit/                    # Audits TQC
│
├── contexte_projet.md                # Contexte projet (pour LLM)
├── PROJECT_TREE.md                   # Ce fichier
└── requirements.txt                  # Dépendances Python
```

## Architecture MAE + FiLM

```
Observation Dict:
├── market: (B, 64, 55)     # 50 Tech + 5 HMM
├── position: (B, 1)
└── w_cost: (B, 1)

FoundationFeatureExtractor:
├── Split Input:
│   ├── Tech Features (cols 0-49) → MAE Encoder (frozen, d_model=256)
│   └── HMM Context (cols 50-54) → FiLM modulation
├── FiLM: γ, β from HMM context modulate MAE embeddings
├── Flatten: (B, 64, 256) → (B, 16384)
├── Concat: [market_flat, position, w_cost] → (B, 16386)
└── Fusion Projector: Linear(16386 → 512) + LayerNorm + LeakyReLU
```

## Fichiers clés

### Configuration
| Fichier | Description |
|---------|-------------|
| `src/config/constants.py` | MAE_D_MODEL=256, HMM_CONTEXT_SIZE=5 |
| `src/config/training.py` | TQCTrainingConfig, WFOTrainingConfig |
| `src/config/validators.py` | ModelDimensionsValidator |

### Modèles
| Fichier | Description |
|---------|-------------|
| `src/models/foundation.py` | CryptoMAE (encoder/decoder) |
| `src/models/layers.py` | FiLMLayer (γ, β modulation) |
| `src/models/rl_adapter.py` | FoundationFeatureExtractor |
| `src/models/tqc_dropout_policy.py` | TQCDropoutPolicy |

### Entraînement
| Fichier | Description |
|---------|-------------|
| `src/training/train_agent.py` | Pipeline TQC |
| `src/training/batch_env.py` | BatchCryptoEnv (GPU) |
| `src/training/callbacks.py` | EntropyFloorCallback, MORL, etc. |

### Scripts
| Fichier | Description |
|---------|-------------|
| `scripts/run_full_wfo.py` | Pipeline WFO complet |
| `scripts/audit_pipeline.py` | Audits (--mode tqc/mae/hmm/film) |
| `scripts/clean_wfo.py` | Nettoyage artefacts |

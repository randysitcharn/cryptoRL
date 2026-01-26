# CONTEXTE PROJET - CryptoRL

> **Projet:** Reinforcement Learning pour trading de cryptomonnaies
> **Dernière mise à jour:** 2026-01-26

---

## 🚨 Problème actuel: Policy Collapse

### Symptômes
- **Actions fixes:** Le TQC converge vers une position constante (ex: +85% LONG ou -1.4% SHORT)
- **Action Entropy = 0:** Aucune exploration
- **Feature Attribution ≈ 0:** Le modèle ignore les inputs (amélioration récente: attribution > 0)

### Historique des audits

| Date | Steps | Position | Attribution | Sharpe | Status |
|------|-------|----------|-------------|--------|--------|
| 25/01 | 30M | +85% LONG | 0 | +1.15 | ❌ Collapse |
| 26/01 | 1M | +4% neutre | 0 | +1.10 | ❌ Collapse |
| 26/01 | 25M | -1.4% SHORT | **>0** | -2.68 | ⚠️ Attribution OK, collapse |

### Corrections appliquées
1. **ent_coef:** `auto_0.1` → `auto_0.5` (target entropy plus élevé)
2. **EntropyFloorCallback:** `min_ent_coef=0.01` (empêche collapse total)
3. **Commit:** `56fee93`, `3c36fbc`

### Diagnostic gSDE
```
log_std mean: -0.039 → std ≈ 0.96 ✅
Actions std (stochastique): 0.798 ✅
Actions range: [-0.999, +0.999] ✅
```
Le gSDE fonctionne, mais la policy converge vers une action fixe.

---

## 🏗️ Architecture (Split Input + FiLM)

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

### Dimensions MAE (constants.py)
```python
MAE_D_MODEL = 256      # DOIT correspondre au checkpoint
MAE_N_HEADS = 4
MAE_N_LAYERS = 2
MAE_DIM_FEEDFORWARD = 1024  # 4 * d_model
```

### Validation (validators.py)
- `ModelDimensionsValidator`: Détecte les mismatches d_model/n_heads/input_dim
- Erreur claire si checkpoint incompatible avec config

---

## 📁 Structure du projet

```
cryptoRL/
├── data/
│   ├── wfo/segment_X/         # Données WFO (train/eval/test.parquet)
│   └── raw_historical/        # Données historiques OHLCV
├── logs/
│   └── wfo/segment_X/         # TensorBoard logs
├── weights/
│   └── wfo/segment_X/         # Checkpoints (encoder.pth, tqc.zip)
├── results/
│   └── tqc_audit/             # Rapports d'audit
├── scripts/
│   ├── run_full_wfo.py        # Pipeline WFO principal
│   └── audit_pipeline.py      # Audits (HMM, MAE, TQC, FiLM)
├── src/
│   ├── config/
│   │   ├── constants.py       # MAE dimensions, HMM_CONTEXT_SIZE
│   │   ├── training.py        # TQCTrainingConfig, WFOTrainingConfig
│   │   └── validators.py      # ModelDimensionsValidator
│   ├── models/
│   │   ├── foundation.py      # CryptoMAE
│   │   ├── rl_adapter.py      # FoundationFeatureExtractor + FiLM
│   │   └── layers.py          # FiLMLayer
│   └── training/
│       ├── train_agent.py     # Entraînement TQC
│       ├── batch_env.py       # BatchCryptoEnv (GPU)
│       └── callbacks.py       # EntropyFloorCallback, etc.
└── tests/
    ├── test_film_extractor.py # Tests FiLM
    └── test_hmm_features.py   # Tests HMM (look-ahead bias)
```

---

## ⚙️ Configuration actuelle

### WFO Training (WFOTrainingConfig)
| Paramètre | Valeur |
|-----------|--------|
| timesteps | 25,000,000 |
| n_envs | 1024 |
| batch_size | 512 |
| learning_rate | 3e-4 → decay |
| gamma | 0.95 |
| ent_coef | `auto_0.5` |
| sde_sample_freq | 64 |
| log_std_init | 0.0 (Shock Therapy) |

### Reward (batch_env.py)
```
Mean:   -0.033
Std:    0.059
Range:  [-0.39, +0.24]
```

### Callbacks
- `EntropyFloorCallback`: min_ent_coef=0.01
- `MORLCurriculumCallback`: w_cost curriculum
- `RotatingCheckpointCallback`: Optimisation disque
- `UnifiedMetricsCallback`: TensorBoard logging

---

## 🖥️ Serveur distant

| Propriété | Valeur |
|-----------|--------|
| Host | `172.219.157.164` |
| Port | `21130` |
| User | `root` |
| Provider | vast.ai |

**Connexion:**
```bash
ssh -p 21130 root@172.219.157.164

# Tunnel TensorBoard
ssh -p 21130 -L 8081:localhost:8081 root@172.219.157.164
```

**Script init serveur:** `scripts/init_server.ps1`

---

## 🔧 Commandes utiles

### WFO
```bash
# Clean + Launch
python scripts/run_full_wfo.py --clean
python scripts/run_full_wfo.py --segment 0 --timesteps 25000000

# Sur serveur (background)
nohup python3 scripts/run_full_wfo.py --segment 0 --timesteps 25000000 </dev/null >logs/wfo_segment0.log 2>&1 &
```

### Audit TQC
```bash
python -m scripts.audit_pipeline --mode tqc --tqc-segment 0
```

### Tests
```bash
pytest tests/ -v
python -m scripts.test_film_extractor  # Test FiLM
```

---

## 📊 Tests de diagnostic

### Test gSDE (exploration)
```python
# Sur serveur
python3 -c "
import torch
from sb3_contrib import TQC
model = TQC.load('weights/wfo/segment_0/tqc.zip')
actor = model.policy.actor
print(f'log_std mean: {actor.log_std.mean().item():.4f}')
print(f'std mean: {torch.exp(actor.log_std).mean().item():.4f}')
"
```

### Test Feature Extractor
```python
# Vérifie que MAE, FiLM, position, w_cost fonctionnent
python -m tests.test_film_extractor
```

### Test Reward Amplitude
```python
# Voir amplitude des rewards après normalisation
python3 -c "
from src.training.batch_env import BatchCryptoEnv
env = BatchCryptoEnv('data/wfo/segment_0/train.parquet', ...)
# Collecter rewards et afficher stats
"
```

---

## 🎯 Prochaines étapes

1. **Investiguer policy collapse** malgré attribution > 0
2. **Vérifier critic loss** - Q-values plates?
3. **Tester reward scaling** - amplitude trop faible?
4. **Explorer target_entropy** - valeur optimale?

---

*Document mis à jour après audits TQC du 26/01/2026*

# Configuration des Données d'Évaluation - Spécification Technique

**Version** : 1.0  
**Date** : 2026-01-19  
**Statut** : ✅ Production-Ready  
**Objectif** : Séparation temporelle stricte Train/Eval pour détection d'overfitting

---

## Table des Matières

1. [Résumé Exécutif](#1-résumé-exécutif)
2. [Contexte et Motivation](#2-contexte-et-motivation)
3. [Architecture des Données](#3-architecture-des-données)
4. [Spécification du Split](#4-spécification-du-split)
5. [Implémentation](#5-implémentation)
6. [Configuration du Training](#6-configuration-du-training)
7. [Modes d'Utilisation](#7-modes-dutilisation)
8. [Validation](#8-validation)
9. [Références](#9-références)

---

## 1. Résumé Exécutif

### 1.1 Problème

Le Signal 3 de `OverfittingGuardCallbackV2` (Train/Eval divergence) nécessite des **données d'évaluation séparées** pour détecter l'overfitting. Sans cette séparation :

- L'agent est évalué sur des données qu'il a déjà vues
- La divergence Train/Eval est artificiellement basse
- L'overfitting n'est pas détecté

### 1.2 Solution

Créer deux fichiers de données avec **séparation temporelle stricte** :

| Fichier | Contenu | Usage |
|---------|---------|-------|
| `processed_data.parquet` | 80% historique (passé) | Training |
| `processed_data_eval.parquet` | 20% historique (futur) | Évaluation |

### 1.3 Bénéfices

| Métrique | Sans Split | Avec Split |
|----------|------------|------------|
| Détection overfitting | ❌ Aveugle | ✅ Signal 3 actif |
| Data leakage | ⚠️ Risque | ✅ Évité (purge window) |
| Généralisation mesurée | ❌ Non | ✅ Sur données futures |

---

## 2. Contexte et Motivation

### 2.1 Pourquoi Séparer Train et Eval ?

En Machine Learning classique, on sépare toujours les données pour mesurer la **généralisation**. En RL Trading, c'est encore plus critique car :

1. **Non-stationnarité** : Les marchés évoluent, les patterns changent
2. **Overfitting temporel** : L'agent peut mémoriser des patterns spécifiques à une période
3. **Régimes de marché** : Bull/Bear/Sideways ont des dynamiques différentes

### 2.2 Le Problème du Data Leakage

```
❌ MAUVAIS: Même fichier pour Train et Eval
┌─────────────────────────────────────────────────────────────┐
│                    processed_data.parquet                   │
│                                                             │
│   Train: random samples ←──────→ Eval: random samples       │
│                        CHEVAUCHEMENT!                       │
│                                                             │
│   L'agent voit les mêmes données → Fausse performance       │
└─────────────────────────────────────────────────────────────┘

✅ BON: Séparation temporelle stricte
┌─────────────────────────────────────────────────────────────┐
│                    DONNÉES HISTORIQUES                      │
│                                                             │
│   ┌─────────────────┐  ┌───────┐  ┌─────────────────┐      │
│   │     TRAIN       │  │ PURGE │  │      EVAL       │      │
│   │   (80% passé)   │  │ (50h) │  │   (20% futur)   │      │
│   │                 │  │       │  │                 │      │
│   │ 2020-01→2023-06 │  │       │  │ 2023-07→2024-12 │      │
│   └─────────────────┘  └───────┘  └─────────────────┘      │
│                                                             │
│   Aucun chevauchement → Mesure vraie généralisation        │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 Rôle du Purge Window

Les indicateurs techniques (RSI, MACD, moyennes mobiles) utilisent des données passées. Sans purge :

```
                    SANS PURGE: Data Leakage via Indicateurs
                    
Train End: 2023-06-30 23:00                Eval Start: 2023-07-01 00:00
           │                                          │
           ▼                                          ▼
    ┌──────────────────────────────────────────────────────────┐
    │  RSI(14) à 2023-07-01 utilise données de 2023-06-17→07-01 │
    │                          ↑                               │
    │              INCLUT DONNÉES DE TRAIN!                    │
    └──────────────────────────────────────────────────────────┘

                    AVEC PURGE (50h): Leakage Évité
                    
Train End: 2023-06-30 23:00    PURGE: 50h    Eval Start: 2023-07-03 01:00
           │                      │                     │
           ▼                      ▼                     ▼
    ┌─────────────┐        ┌──────────┐        ┌─────────────────┐
    │   TRAIN     │        │ SKIP 50h │        │      EVAL       │
    │             │        │ (purge)  │        │                 │
    └─────────────┘        └──────────┘        └─────────────────┘
    
    Les indicateurs d'Eval n'utilisent QUE des données d'Eval
```

---

## 3. Architecture des Données

### 3.1 Schéma Global

```
data/
├── raw/
│   └── multi_asset_historical.csv     # Données brutes (OHLCV)
│
├── processed/
│   └── processed_data_full.parquet    # Données traitées complètes
│
├── processed_data.parquet             # TRAIN (80%)
└── processed_data_eval.parquet        # EVAL (20%)
```

### 3.2 Pipeline de Création

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       PIPELINE DE DONNÉES                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. TÉLÉCHARGEMENT                                                      │
│     historical_downloader.py                                            │
│            │                                                            │
│            ▼                                                            │
│     data/raw/multi_asset_historical.csv                                 │
│                                                                         │
│  2. FEATURE ENGINEERING                                                 │
│     FeatureEngineer + RegimeDetector                                    │
│            │                                                            │
│            ▼                                                            │
│     data/processed/processed_data_full.parquet                          │
│                                                                         │
│  3. SPLIT TRAIN/EVAL (ce document)                                      │
│     prepare_train_eval_split.py                                         │
│            │                                                            │
│            ├──────────────────┬──────────────────┐                      │
│            ▼                  ▼                  ▼                      │
│     processed_data.parquet   PURGE    processed_data_eval.parquet       │
│         (TRAIN 80%)          (50h)         (EVAL 20%)                   │
│                                                                         │
│  4. TRAINING                                                            │
│     train_agent.py                                                      │
│            │                                                            │
│            ├── BatchCryptoEnv(data_path=TRAIN)                          │
│            └── BatchCryptoEnv(eval_data_path=EVAL)                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Spécification du Split

### 4.1 Paramètres

| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| `train_ratio` | 0.80 | Standard ML, assez de données pour train et eval |
| `purge_window` | 50 | ~2 jours, couvre la plupart des lookback d'indicateurs |

### 4.2 Formules

```python
n_samples = len(df)                          # Total de lignes
idx_split = int(n_samples * train_ratio)     # Index de coupure

train_df = df.iloc[:idx_split]               # Début → 80%
eval_df = df.iloc[idx_split + purge_window:] # 80% + purge → Fin
```

### 4.3 Exemple Concret

```
Données complètes: 2020-01-01 → 2024-12-31 (43,800 heures)

train_ratio = 0.80
purge_window = 50

idx_split = 43,800 × 0.80 = 35,040

TRAIN: idx 0 → 35,039       = 35,040 heures (2020-01-01 → 2023-12-31)
PURGE: idx 35,040 → 35,089  = 50 heures (skipped)
EVAL:  idx 35,090 → 43,799  = 8,710 heures (2024-01-03 → 2024-12-31)
```

---

## 5. Implémentation

### 5.1 Script de Split

**Fichier** : `scripts/prepare_train_eval_split.py`

```python
#!/usr/bin/env python3
"""
prepare_train_eval_split.py - Prepare train/eval data split.

Usage:
    python scripts/prepare_train_eval_split.py
    python scripts/prepare_train_eval_split.py --input data/my_data.parquet --train-ratio 0.85
"""

import pandas as pd

def prepare_split(
    input_path: str,
    train_output: str = "data/processed_data.parquet",
    eval_output: str = "data/processed_data_eval.parquet",
    train_ratio: float = 0.80,
    purge_window: int = 50,
):
    # Load
    df = pd.read_parquet(input_path)
    n_samples = len(df)
    
    # Split
    idx_split = int(n_samples * train_ratio)
    train_df = df.iloc[:idx_split]
    eval_df = df.iloc[idx_split + purge_window:]
    
    # Save
    train_df.to_parquet(train_output)
    eval_df.to_parquet(eval_output)
    
    return train_df, eval_df
```

### 5.2 Utilisation

```bash
# Split par défaut (80/20, purge 50h)
python scripts/prepare_train_eval_split.py --input data/processed_data_full.parquet

# Split personnalisé
python scripts/prepare_train_eval_split.py \
    --input data/processed_data_full.parquet \
    --train-ratio 0.85 \
    --purge-window 100
```

### 5.3 Output Attendu

```
============================================================
Train/Eval Data Split for OverfittingGuard Signal 3
============================================================

Loading: data/processed_data_full.parquet
Total rows: 43,800
Date range: 2020-01-01 00:00:00 → 2024-12-31 23:00:00

────────────────────────────────────────────────────────────
TRAIN DATA
────────────────────────────────────────────────────────────
  Output: data/processed_data.parquet
  Rows:   35,040 (80.0%)
  Start:  2020-01-01 00:00:00
  End:    2023-12-31 23:00:00

────────────────────────────────────────────────────────────
PURGE WINDOW: 50 hours skipped
────────────────────────────────────────────────────────────

────────────────────────────────────────────────────────────
EVAL DATA
────────────────────────────────────────────────────────────
  Output: data/processed_data_eval.parquet
  Rows:   8,710 (19.9%)
  Start:  2024-01-03 01:00:00
  End:    2024-12-31 23:00:00

============================================================
VALIDATION
============================================================
  Gap between train end and eval start: 51 hours
  ✅ No temporal overlap - Data leakage prevented
```

---

## 6. Configuration du Training

### 6.1 Fichier de Configuration

**Fichier** : `src/config/training.py`

```python
@dataclass
class TQCTrainingConfig:
    # Paths - IMPORTANT: Deux fichiers séparés
    data_path: str = "data/processed_data.parquet"           # TRAIN
    eval_data_path: str = "data/processed_data_eval.parquet"  # EVAL
    
    # Eval frequency
    eval_freq: int = 50_000        # Évaluer toutes les 50k steps
    eval_episode_length: int = 720  # 1 mois d'évaluation
```

### 6.2 Création des Environnements

**Fichier** : `src/training/train_agent.py`

```python
def create_environments(config: TrainingConfig, n_envs: int = 1):
    # ==================== TRAIN ENV ====================
    train_vec_env = BatchCryptoEnv(
        parquet_path=config.data_path,      # TRAIN DATA
        n_envs=n_envs,
        random_start=True,                  # Démarrage aléatoire
        observation_noise=0.01,             # Bruit (régularisation)
    )

    # ==================== EVAL ENV ====================
    if config.eval_data_path is not None:
        eval_vec_env = BatchCryptoEnv(
            parquet_path=config.eval_data_path,  # EVAL DATA (différent!)
            n_envs=min(n_envs, 32),
            random_start=False,                  # Démarrage séquentiel
            observation_noise=0.0,               # Pas de bruit
        )
    else:
        eval_vec_env = None  # Mode WFO

    return train_vec_env, eval_vec_env, ...
```

### 6.3 Différences Train vs Eval

| Paramètre | Train Env | Eval Env | Raison |
|-----------|-----------|----------|--------|
| `parquet_path` | `data_path` | `eval_data_path` | Données différentes |
| `random_start` | `True` | `False` | Exploration vs Reproductibilité |
| `observation_noise` | `0.01` | `0.0` | Régularisation vs Mesure pure |
| `n_envs` | `1024` | `32` | Throughput vs Ressources |
| `curriculum` | Actif | Valeurs finales | Apprentissage vs Évaluation |

---

## 7. Modes d'Utilisation

### 7.1 Mode Standard (Développement)

**Use case** : Développement, debugging, hyperparameter tuning

```python
# Configuration
config = TQCTrainingConfig(
    data_path="data/processed_data.parquet",           # Train 80%
    eval_data_path="data/processed_data_eval.parquet",  # Eval 20%
)

# Signal 3 actif
guard = OverfittingGuardCallbackV2(
    eval_callback=eval_callback,  # ✅ Lié
)
# Output: "Signal 3 - Train/Eval divergence: >50% [ENABLED (via EvalCallback)]"
```

### 7.2 Mode WFO (Production)

**Use case** : Walk-Forward Optimization, production

```python
# Configuration WFO
config = TQCTrainingConfig(
    data_path="data/wfo/segment_0_train.parquet",
    eval_data_path=None,  # ❌ Désactivé pour éviter data leakage
)

# Signal 3 désactivé automatiquement
guard = OverfittingGuardCallbackV2(
    eval_callback=None,  # Signal 3 ignoré
)
# Output: "Signal 3 - Train/Eval divergence: >50% [DISABLED (no EvalCallback)]"
```

**Pourquoi désactiver en WFO ?**

En WFO, l'évaluation se fait **après** chaque segment sur des données futures. Avoir un eval pendant le training créerait un data leakage car :

1. Le modèle verrait les données d'eval avant le test final
2. Les hyperparamètres pourraient être optimisés sur l'eval
3. La mesure de généralisation WFO serait faussée

### 7.3 Tableau Récapitulatif

| Mode | `eval_data_path` | Signal 3 | `EvalCallback` | Quand utiliser |
|------|------------------|----------|----------------|----------------|
| **Standard** | `"data/eval.parquet"` | ✅ Actif | ✅ Actif | Développement |
| **WFO** | `None` | ❌ Désactivé | ❌ Désactivé | Production |
| **Debug** | `"data/eval.parquet"` | ✅ Actif | ✅ Actif | Debugging overfitting |

---

## 8. Validation

### 8.1 Checklist Avant Training

```python
import pandas as pd

# 1. Charger les données
train_df = pd.read_parquet("data/processed_data.parquet")
eval_df = pd.read_parquet("data/processed_data_eval.parquet")

# 2. Vérifier les dates
print(f"Train: {train_df.index.min()} → {train_df.index.max()}")
print(f"Eval:  {eval_df.index.min()} → {eval_df.index.max()}")

# 3. Vérifier pas de chevauchement
assert train_df.index.max() < eval_df.index.min(), "ERREUR: Chevauchement!"

# 4. Vérifier le gap (purge)
gap_hours = (eval_df.index.min() - train_df.index.max()).total_seconds() / 3600
print(f"Gap: {gap_hours:.0f} heures")
assert gap_hours >= 50, f"ERREUR: Gap insuffisant ({gap_hours}h < 50h)"

print("✅ Validation réussie - Données prêtes pour training")
```

### 8.2 Tests Automatisés

```python
def test_train_eval_no_overlap():
    """Vérifier qu'il n'y a pas de chevauchement temporel."""
    train_df = pd.read_parquet("data/processed_data.parquet")
    eval_df = pd.read_parquet("data/processed_data_eval.parquet")
    
    assert train_df.index.max() < eval_df.index.min()

def test_train_eval_purge_window():
    """Vérifier que le purge window est respecté."""
    train_df = pd.read_parquet("data/processed_data.parquet")
    eval_df = pd.read_parquet("data/processed_data_eval.parquet")
    
    gap = (eval_df.index.min() - train_df.index.max()).total_seconds() / 3600
    assert gap >= 50, f"Purge window trop petit: {gap}h"

def test_train_eval_ratio():
    """Vérifier le ratio train/eval."""
    train_df = pd.read_parquet("data/processed_data.parquet")
    eval_df = pd.read_parquet("data/processed_data_eval.parquet")
    
    total = len(train_df) + len(eval_df)
    train_ratio = len(train_df) / total
    
    assert 0.75 <= train_ratio <= 0.85, f"Ratio hors limites: {train_ratio}"
```

---

## 9. Références

### 9.1 Documents Liés

1. **OverfittingGuardCallbackV2** : `docs/OVERFITTING_GUARD_V2.md`
   - Signal 3 utilise les données d'évaluation
   
2. **Walk-Forward Optimization** : `scripts/run_full_wfo.py`
   - Mode production sans eval pendant training

3. **Data Engineering** : `src/data_engineering/`
   - Pipeline de création des features

### 9.2 Best Practices

| Practice | Implémenté | Notes |
|----------|------------|-------|
| Séparation temporelle | ✅ | Train avant Eval chronologiquement |
| Purge window | ✅ | 50h pour éviter leakage des indicateurs |
| Pas de shuffle | ✅ | Données time series, ordre préservé |
| Eval sans bruit | ✅ | `observation_noise=0.0` |
| Eval séquentiel | ✅ | `random_start=False` |

---

## Annexe A : Checklist d'Implémentation

- [x] Script `prepare_train_eval_split.py`
- [x] Configuration `eval_data_path` dans `TQCTrainingConfig`
- [x] Création `eval_env` dans `create_environments()`
- [x] Intégration avec `OverfittingGuardCallbackV2`
- [ ] Tests unitaires pour validation du split
- [x] Documentation (ce fichier)

---

## Annexe B : Troubleshooting

### B.1 "Signal 3 toujours DISABLED"

**Symptôme** : Le message affiche `[DISABLED (no EvalCallback)]`

**Causes possibles** :
1. `eval_data_path = None` dans la config
2. `eval_callback` non passé à `OverfittingGuardCallbackV2`

**Solution** :
```python
# Vérifier la config
print(f"eval_data_path: {config.eval_data_path}")

# Vérifier que EvalCallback est créé
eval_callback = next((cb for cb in callbacks if isinstance(cb, EvalCallback)), None)
print(f"EvalCallback found: {eval_callback is not None}")
```

### B.2 "Divergence toujours à 0"

**Symptôme** : `overfit/train_eval_divergence` reste à 0

**Causes possibles** :
1. Pas assez d'épisodes dans `ep_info_buffer`
2. `EvalCallback` n'a pas encore run (`last_mean_reward = -inf`)

**Solution** :
```python
# Réduire eval_freq pour avoir des données plus tôt
config.eval_freq = 10_000  # Au lieu de 50_000
```

### B.3 "FileNotFoundError: eval.parquet"

**Symptôme** : Le fichier d'évaluation n'existe pas

**Solution** :
```bash
# Créer le split
python scripts/prepare_train_eval_split.py --input data/processed_data_full.parquet
```

---

## Annexe C : Glossaire

| Terme | Définition |
|-------|------------|
| **Data Leakage** | Fuite d'information du futur vers le passé |
| **Purge Window** | Gap temporel entre train et eval pour éviter leakage |
| **Train/Eval Split** | Séparation des données en ensembles distincts |
| **WFO** | Walk-Forward Optimization (validation rolling) |
| **Généralisation** | Capacité du modèle à performer sur données non vues |

---

*Fin de la spécification technique*

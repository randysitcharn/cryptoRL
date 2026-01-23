# CONTEXTE PROJET - CryptoRL

> **Projet:** Reinforcement Learning pour trading de cryptomonnaies  
> **Dernière mise à jour:** 2026-01-17

---

## 🚀 Démarrage rapide

### Environnement virtuel

Le projet utilise un environnement virtuel Python. Par défaut, il se trouve dans `.venv/` à la racine du projet.

**Créer l'environnement virtuel (si nécessaire):**
```bash
python -m venv .venv
```

**Activer l'environnement virtuel:**

- **Windows (PowerShell):**
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```

- **Windows (CMD):**
  ```cmd
  .venv\Scripts\activate.bat
  ```

- **Linux/Mac:**
  ```bash
  source .venv/bin/activate
  ```

**Installer les dépendances:**
```bash
pip install -r requirements.txt
```

---

## 📁 Structure du projet

```
cryptoRL/
├── data/
│   ├── processed/              # Données traitées (parquet)
│   ├── raw/                   # Données brutes
│   └── raw_historical/         # Données historiques OHLCV
├── docs/                      # Documentation
├── logs/                      # Logs d'entraînement
├── results/                    # Résultats et visualisations
├── scripts/                   # Scripts d'exécution
│   └── run_full_wfo.py        # Script principal WFO
├── src/
│   ├── config/                # Configuration
│   ├── data/                   # Chargement des données
│   ├── data_engineering/      # Feature engineering (FFD, HMM, etc.)
│   ├── evaluation/            # Évaluation et backtesting
│   ├── models/                # Modèles (MAE, TQC)
│   ├── training/              # Infrastructure d'entraînement
│   └── utils/                 # Utilitaires
└── tests/                     # Tests unitaires
```

---

## 🔑 Fichiers principaux

| Fichier | Description |
|---------|-------------|
| `scripts/run_full_wfo.py` | Orchestration WFO complète (HMM → MAE → TQC → Évaluation) |
| `src/training/train_agent.py` | Entraînement TQC avec modèle Foundation |
| `src/training/batch_env.py` | Environnement vectorisé GPU/CPU |
| `src/models/foundation.py` | Modèle MAE (autoencodeur) |
| `src/models/rl_adapter.py` | Adaptateur Foundation → TQC |

---

## 🏗️ Architecture

```
WFO Pipeline (run_full_wfo.py)
├── [1] Chargement données (CSV/Parquet)
├── [2] Feature engineering (FFD, Z-Score, Parkinson, Garman-Klass)
├── [3] Détection régimes HMM (4 états)
├── [4] Pre-training MAE (90 epochs)
├── [5] Entraînement TQC (BatchCryptoEnv, 54M steps)
└── [6] Évaluation OOS (backtest fenêtre test)
```

**Environnement d'entraînement:**
- `BatchCryptoEnv` (batch_env.py) - GPU/CPU, supporte n_envs=1 pour évaluation

**Callbacks:**
- `ThreePhaseCurriculumCallback` - Curriculum learning (3 phases)
- `RotatingCheckpointCallback` - Optimisation disque
- `TrainingMetricsCallback` - Logging NAV mode WFO

---

## ⚙️ Configuration

### Paramètres WFO

| Paramètre | Valeur |
|-----------|--------|
| train_months | 12 (8,640 lignes) |
| test_months | 3 (2,160 lignes) |
| step_months | 3 (2,160 lignes) |

### Entraînement

| Paramètre | Valeur |
|-----------|--------|
| tqc_timesteps | 30,000,000 |
| mae_epochs | 90 |
| n_envs | 1024 |
| batch_size | 2048 |
| learning_rate | 1e-4 |
| gamma | 0.95 |

### Curriculum (3 phases)

| Phase | Progression | Churn | Smooth |
|-------|-------------|-------|--------|
| 1 - Discovery | 0-10% | 0.0 → 0.10 | 0.0 |
| 2 - Discipline | 10-30% | 0.10 → 0.50 | 0.0 → 0.02 |
| 3 - Consolidation | 30-100% | 0.50 (fixe) | 0.02 (fixe) |

---

## 🖥️ Serveur distant

| Propriété | Valeur |
|-----------|--------|
| Host | `158.51.110.52` |
| Port | `20941` |
| User | `root` |
| Provider | vast.ai |
| TensorBoard | Port 8081 |

**Connexion:**
```bash
ssh -p 20941 root@158.51.110.52

# Tunnel TensorBoard
ssh -p 20941 -L 8081:localhost:8081 root@158.51.110.52
```

---

## 📊 Architecture MORL

Le projet utilise une architecture MORL (Multi-Objective Reinforcement Learning) pour gérer l'équilibre entre performance et coûts:

```python
reward = r_perf + curriculum_lambda * w_cost * r_cost * MAX_PENALTY_SCALE
```

où:
- `r_perf`: Log-returns (objectif performance)
- `w_cost ∈ [0, 1]`: Paramètre MORL dans l'observation
- `curriculum_lambda ∈ [0, 0.4]`: Progression contrôlée

---

*Document simplifié - Pour plus de détails, voir la documentation dans `docs/`*

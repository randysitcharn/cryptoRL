# CONTEXTE PROJET - CryptoRL

> **Projet:** Reinforcement Learning pour trading de cryptomonnaies  
> **Dernière mise à jour:** 2026-01-17

---

## 🚀 Démarrage rapide

### Environnement virtuel

Le projet utilise un environnement virtuel Python. Par défaut, il se trouve dans `venv/` à la racine du projet.

**Créer l'environnement virtuel (si nécessaire):**
```bash
python -m venv venv
```

**Activer l'environnement virtuel:**

- **Windows (PowerShell):**
  ```powershell
  venv\Scripts\activate
  ```

- **Windows (CMD):**
  ```cmd
  venv\Scripts\activate.bat
  ```

- **Linux/Mac:**
  ```bash
  source venv/bin/activate
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
- `MORLCurriculumCallback` - Curriculum learning progressif (modulation w_cost)
- `ThreePhaseCurriculumCallback` - ⚠️ OBSOLETE (remplacé par MORLCurriculumCallback)
- `RotatingCheckpointCallback` - Optimisation disque
- `UnifiedMetricsCallback` - Logging TensorBoard unifié

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

### Curriculum (MORL)

| Phase | Progression | w_cost | Description |
|-------|-------------|--------|-------------|
| Rampe | 0-50% | 0.0 → 0.1 | Introduction progressive des coûts |
| Plateau | 50-100% | 0.1 (fixe) | Stabilisation |

**Note:** L'ancien système `ThreePhaseCurriculumCallback` (curriculum_lambda) est obsolète. Le nouveau système `MORLCurriculumCallback` module directement `w_cost` dans l'observation (architecture MORL).

---

## 🖥️ Serveur distant

| Propriété | Valeur |
|-----------|--------|
| Host | `86.127.245.129` |
| Port | `25083` |
| User | `root` |
| Provider | vast.ai |
| TensorBoard | Port 8081 |

**Connexion:**
```bash
ssh -p 25083 root@86.127.245.129

# Tunnel TensorBoard
ssh -p 25083 -L 8081:localhost:8081 root@86.127.245.129
```

---

## 📊 Architecture MORL

Le projet utilise une architecture MORL (Multi-Objective Reinforcement Learning) pour gérer l'équilibre entre performance et coûts:

```python
reward = r_perf + w_cost * r_cost * MAX_PENALTY_SCALE
```

où:
- `r_perf`: Log-returns (objectif performance)
- `w_cost ∈ [0, 1]`: Paramètre MORL dans l'observation (modulé par `MORLCurriculumCallback`)
- `MAX_PENALTY_SCALE = 0.4`: Facteur d'échelle des pénalités

**Curriculum Learning:**
- `MORLCurriculumCallback` module progressivement `w_cost` de 0.0 (performance pure) à 0.1 (équilibré) sur 50% du training
- L'agent apprend d'abord à maximiser la performance, puis à équilibrer avec les coûts

---

*Document simplifié - Pour plus de détails, voir la documentation dans `docs/`*

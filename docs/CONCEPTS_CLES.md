# Concepts Clés à Connaître - CryptoRL

> Liste des concepts fondamentaux pour comprendre et expliquer le projet CryptoRL.

---

## 1. Vue d'Ensemble du Projet

### Objectif
Créer un **agent de trading automatisé** pour les cryptomonnaies (BTC principalement) en utilisant le **Reinforcement Learning profond (Deep RL)**.

### Stack Technique
| Composant | Technologie | Rôle |
|-----------|-------------|------|
| Algorithme RL | **TQC** (Truncated Quantile Critics) | Algorithme off-policy SOTA pour actions continues |
| Feature Extractor | **MAE** (Masked Autoencoder) | Pré-entraînement non-supervisé du Transformer |
| Environnement | **BatchCryptoEnv** | Simulation GPU-vectorisée (1024 envs parallèles) |
| Framework RL | **Stable-Baselines3** | Implémentation standard TQC/PPO |
| Validation | **Walk-Forward Optimization** | Anti-data-leakage, test out-of-sample |

---

## 2. Reinforcement Learning (RL)

### Formulation MDP
| Élément | Description dans CryptoRL |
|---------|---------------------------|
| **État (s)** | Fenêtre de 64 heures de features de marché + position actuelle |
| **Action (a)** | Position cible ∈ [-1, 1] (0% → 100% long) |
| **Récompense (r)** | Log-return - pénalités (churn, downside, smoothness) |
| **Transition** | Déterministe (données historiques) |

### Algorithme TQC (Truncated Quantile Critics)
- **Famille** : Actor-Critic off-policy
- **Particularité** : Estime la distribution complète des Q-values (pas juste la moyenne)
- **Avantage** : Réduit la surestimation des Q-values → policies plus stables
- **Ref** : Kuznetsov et al., 2020 - "Controlling Overestimation Bias with Truncated Mixture of Continuous Distributional Quantile Critics"

### Concepts RL à Connaître
| Concept | Définition |
|---------|------------|
| **Policy (π)** | Fonction qui décide l'action à partir de l'état |
| **Value Function (V)** | Espérance des récompenses futures depuis un état |
| **Q-Function (Q)** | Espérance des récompenses futures depuis (état, action) |
| **Replay Buffer** | Mémoire stockant les transitions passées pour entraînement |
| **Entropy Regularization** | Bonus d'exploration pour éviter les policies déterministes |
| **Discount Factor (γ)** | Poids des récompenses futures vs immédiates (ici γ=0.95) |

---

## 3. Architecture Neurale

### Foundation Model (MAE - Masked Autoencoder)
```
Input (64×N features) → Transformer Encoder → Latent (128 dim) → Decoder → Reconstruction
                           ↓
                    [Pré-entraîné pour capturer patterns de marché]
```

- **Pré-entraînement** : L'encodeur apprend à reconstruire des séquences masquées (90 epochs)
- **Utilisation** : L'encodeur frozen devient le feature extractor pour TQC
- **Intuition** : Comme BERT pour le NLP, mais pour séries temporelles financières

### Feature Extractor pour RL
```python
Observation → MAE Encoder (frozen) → Features 128D → Actor/Critic Networks → Action
```

---

## 4. Feature Engineering

### Fractional Differentiation (FFD)
| Problème | Solution |
|----------|----------|
| Séries de prix non-stationnaires | Différenciation classique (d=1) |
| Mais perte de mémoire historique | **FFD** : différenciation d ∈ [0,1] |

- **Réf** : Lopez de Prado (2018) - "Advances in Financial Machine Learning"
- **Implémentation** : Recherche automatique du `d` minimal pour stationnarité (ADF test)

### Features Calculées
| Feature | Formule | Intuition |
|---------|---------|-----------|
| **Log-Returns** | log(P_t / P_{t-1}) | Rendement normalisé |
| **FFD** | Σ w_k × log(P_{t-k}) | Prix stationnaire avec mémoire |
| **Parkinson Vol** | (High-Low)² / (4×ln2) | Volatilité intraday |
| **Garman-Klass** | 0.5×(H-L)² - (2ln2-1)×(C-O)² | Volatilité OHLC complète |
| **Z-Score** | (X - μ) / σ sur 720h | Normalisation rolling |
| **Volume Relatif** | Vol / Mean(Vol) | Activité relative |

### Régimes de Marché (HMM)
| Régime | Description | Comportement Agent |
|--------|-------------|-------------------|
| 0 | Bull fort | Position long élevée |
| 1 | Bull modéré | Position modérée |
| 2 | Transition/Range | Position réduite |
| 3 | Bear | Cash ou short (si implémenté) |

- **Modèle** : Hidden Markov Model à 4 états
- **Features HMM** : Probabilités Prob_0, Prob_1, Prob_2, Prob_3 ajoutées aux observations

---

## 5. Fonction de Récompense

### Formule Complète
```
reward = log_returns - curriculum_λ × (churn_penalty + downside_risk + smoothness_penalty)
```

### Composantes
| Composante | Formule | Rôle |
|------------|---------|------|
| **Log Returns** | log1p(clamp(r, -0.99)) × 100 | Maximiser les profits |
| **Churn Penalty** | \|Δposition\| × cost × gate | Éviter sur-trading |
| **Downside Risk** | (negative_returns)² × 500 | Pénaliser les pertes (Sortino) |
| **Smoothness** | smooth_coef × Δposition² | Positions stables |

### Curriculum Learning (3 Phases)
| Phase | Progress | curriculum_λ | Objectif |
|-------|----------|--------------|----------|
| 1 - Discovery | 0-10% | 0.0 | Exploration libre |
| 2 - Discipline | 10-30% | 0.0 → 0.4 | Apprentissage graduel des pénalités |
| 3 - Consolidation | 30-100% | 0.4 | Stabilité |

---

## 6. Walk-Forward Optimization (WFO)

### Problème : Data Leakage
Si on entraîne sur toutes les données puis teste sur une portion, le modèle a potentiellement "vu" des patterns futurs → **overfitting**.

### Solution : WFO
```
Timeline: ═══════════════════════════════════════════════════════►
Segment 0: [TRAIN 18 mois][TEST 3 mois]
Segment 1:             [TRAIN 18 mois][TEST 3 mois]
Segment 2:                         [TRAIN 18 mois][TEST 3 mois]
...
```

### Pipeline par Segment
1. **Preprocessing** : RobustScaler fit sur TRAIN uniquement
2. **HMM** : fit sur TRAIN, predict sur TRAIN+TEST
3. **MAE** : entraîné sur TRAIN (90 epochs)
4. **TQC** : entraîné sur TRAIN (90M steps)
5. **Evaluation** : backtest sur TEST (out-of-sample)

---

## 7. Environnement de Trading (BatchCryptoEnv)

### GPU-Vectorisation
| Architecture | Parallélisme | FPS |
|--------------|--------------|-----|
| SubprocVecEnv | 31 CPU processes × 1 env | ~2,000 |
| **BatchCryptoEnv** | 1 process × 1024 GPU envs | ~50,000 |

### Mécanismes Clés
| Mécanisme | Description |
|-----------|-------------|
| **Volatility Scaling** | Position × (target_vol / current_vol) pour risque constant |
| **Action Discretization** | Actions arrondies à 0.1 → 21 niveaux (réduit bruit) |
| **Observation Noise** | Bruit gaussien 1% (régularisation anti-overfitting) |
| **Random Start** | Début aléatoire des épisodes (data augmentation) |

### Espace d'Observation
```python
{
    "market": Box(64, N_features),  # Fenêtre temporelle
    "position": Box(1,),            # Position actuelle [0, 1]
}
```

---

## 8. PLO - Pénalités Adaptatives (Avancé)

### Problème
Le coefficient de pénalité downside est **statique**. Un drawdown de 15% est pénalisé comme un de 5%.

### Solution : PLO (Predictive Lagrangian Optimization)
```
downside_risk = base_downside × downside_multiplier
                                      ↑
                               Contrôlé par PID : λ ∈ [1.0, 5.0]
```

### Contrôleur PID
| Composante | Formule | Rôle |
|------------|---------|------|
| **P (Proportionnel)** | K_p × violation | Réaction immédiate |
| **I (Intégral)** | Σ K_i × violation | Mémoire des violations |
| **D (Dérivé)** | K_d × Δviolation | Anticipation |

### Améliorations Critiques
- **Observation augmentée** : L'agent voit `risk_level` (λ normalisé)
- **Prédiction robuste** : polyfit au lieu de différence naïve
- **Quantile 90%** : VaR-style au lieu de moyenne
- **Smoothing** : max ±0.05/step pour éviter sauts brutaux

---

## 9. Métriques de Performance

### Trading
| Métrique | Formule | Cible |
|----------|---------|-------|
| **Sharpe Ratio** | E[r] / std(r) × √252 | > 1.5 |
| **Sortino Ratio** | E[r] / downside_std × √252 | > 2.0 |
| **Max Drawdown** | max(peak - trough) / peak | < 15% |
| **Alpha** | Return_agent - Return_benchmark | > 0 |

### RL
| Métrique | Description |
|----------|-------------|
| **Episode Reward** | Somme des récompenses par épisode |
| **FPS** | Steps par seconde (efficacité GPU) |
| **Policy Entropy** | Diversité des actions (exploration) |

---

## 10. Fichiers Principaux

| Fichier | Rôle | Lignes |
|---------|------|--------|
| `scripts/run_full_wfo.py` | Orchestration WFO complète | ~1600 |
| `src/training/batch_env.py` | Environnement GPU-vectorisé | ~950 |
| `src/training/callbacks.py` | Curriculum, Logging, Checkpoints | ~500 |
| `src/training/train_agent.py` | Entraînement TQC | ~800 |
| `src/data_engineering/features.py` | Feature engineering (FFD, Vol) | ~650 |
| `src/data_engineering/manager.py` | HMM Regime Detection | ~300 |

---

## 11. Termes Techniques Courants

| Terme | Définition |
|-------|------------|
| **OOS** | Out-Of-Sample (données de test non vues) |
| **Churn** | Fréquence de changement de position |
| **Drawdown** | Perte depuis le dernier pic de capital |
| **Sortino** | Sharpe modifié qui ne pénalise que les pertes |
| **B&H** | Buy & Hold (benchmark passif) |
| **Curriculum** | Augmentation progressive de la difficulté |
| **NAV** | Net Asset Value (valeur du portefeuille) |
| **Slippage** | Écart entre prix demandé et exécuté |

---

## 12. Références Essentielles

1. **Lopez de Prado (2018)** - "Advances in Financial Machine Learning"
   - Fractional Differentiation, Meta-Labeling, Purged CV

2. **Kuznetsov et al. (2020)** - "TQC: Truncated Quantile Critics"
   - Algorithme RL utilisé

3. **Stooke et al. (2020)** - "PID Lagrangian Methods"
   - Base théorique pour PLO

4. **He et al. (2022)** - "Masked Autoencoders Are Scalable Vision Learners"
   - Architecture MAE adaptée aux séries temporelles

---

*Dernière mise à jour : 2026-01-18*

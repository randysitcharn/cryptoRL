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

## 9. Dynamic Observation Noise (SOTA 2026)

### Problème
Le bruit d'observation constant (1%) ne s'adapte pas à la progression du training ni aux conditions de marché.

### Solution : Combinaison de 2 Techniques

#### 1. Noise Annealing (Time-based)
```python
annealing_factor = 1.0 - 0.5 * progress  # 100% → 50%
```
- **Début** : Fort bruit → exploration robuste
- **Fin** : Bruit réduit → précision accrue
- **Référence** : NoisyRollout (2025)

#### 2. Volatility-Adaptive (Regime-based)
```python
current_vol = torch.sqrt(ema_vars).clamp(min=1e-6)
vol_factor = (target_volatility / current_vol).clamp(0.5, 2.0)
```
- **Marché calme** : Plus de bruit (risque d'overfitting élevé)
- **Marché volatile** : Moins de bruit (bruit naturel suffisant)
- **Innovation** : Propre à CryptoRL

#### Formule Combinée
```python
final_scale = observation_noise × annealing_factor × vol_factor
noise = torch.randn_like(market) × final_scale
```

| Composante | Effet |
|------------|-------|
| `annealing_factor` | Décroissance temporelle [1.0 → 0.5] |
| `vol_factor` | Adaptation au régime [0.5 → 2.0] |

---

## 10. TQCDropoutPolicy (DroQ + STAC)

### Problème
Les réseaux TQC standards peuvent surajuster aux données d'entraînement.

### Solution : Dropout + LayerNorm (DroQ-style)

#### Architecture par Couche
```
Linear → LayerNorm → ReLU → Dropout
```

#### Pourquoi LayerNorm est CRITIQUE
- Sans LayerNorm, le dropout déstabilise l'entraînement en RL
- LayerNorm normalise les activations après chaque couche
- **Obligatoire** pour stabilité (DroQ, Hiraoka 2021)

#### Paramètres Recommandés
| Composant | Dropout Rate | Justification |
|-----------|--------------|---------------|
| **Critics** | 0.01 | Estimation Q-value robuste |
| **Actor** | 0.005 | Régularisation légère (ou 0.0 si gSDE) |

#### Conflit gSDE
Si `use_sde=True` et `actor_dropout > 0`, le dropout casse la continuité temporelle de gSDE → **Désactiver dropout actor avec gSDE**.

#### Références
- **DroQ** (Hiraoka 2021) : Dropout remplace gros ensembles de critics
- **STAC** (2026) : Dropout sur actor aussi

---

## 11. Comprendre l'Overfitting en RL

L'overfitting, c'est quand le modèle **mémorise** les données d'entraînement au lieu d'**apprendre** des patterns généralisables.

### Signe 1 : Divergence Train/Eval (le classique)

```
Train reward:  ████████████████████ +50%
Eval reward:   ████████             +15%
```

**Ce qui se passe** : Le modèle performe bien sur les données connues, mais échoue sur des données nouvelles.

**Métrique** : `overfit/train_eval_divergence` - si > 0.2, alerte.

### Signe 2 : La forme en U de la loss (U-shape)

```
Loss
  │
  │\
  │ \
  │  \____ ← Point optimal (meilleure généralisation)
  │       \____/
  │            ↑ Overfitting commence ici
  └─────────────────── Steps
```

**Ce qui se passe** :
- Phase 1 : La loss descend → le modèle apprend
- Phase 2 : La loss remonte → le modèle mémorise au lieu de généraliser

**Détection** : Analyser la trajectoire de `train/critic_loss` - si "U-shape", le meilleur checkpoint est au milieu, pas à la fin.

### Signe 3 : Saturation des actions

```
Actions au début:     [-0.3, 0.5, -0.2, 0.1, ...]  ← Diversifiées
Actions après overfit: [1.0, -1.0, 1.0, -1.0, ...] ← Extrêmes
```

**Ce qui se passe** : Le modèle devient trop "confiant" et pousse toutes ses actions vers les extrêmes (-1 ou +1), au lieu de nuancer.

**Métrique** : `overfit/action_saturation` - si > 0.8, policy collapse probable.

### Signe 4 : Variance des rewards qui explose

```
Rewards stables:    [10, 12, 11, 10, 13, 11]     CV = 0.10
Rewards instables:  [50, -30, 80, -20, 60, -40]  CV = 1.50
```

**Ce qui se passe** : Un modèle overfitté fait des paris risqués basés sur des patterns spécifiques. Ça marche parfois (gros gains), ça échoue souvent (grosses pertes).

**Métrique** : `overfit/reward_cv` - si > 0.5, instabilité.

### Signe 5 : Entropie qui s'effondre

```
Entropie haute:   Le modèle explore, hésite, essaie des choses
Entropie basse:   Le modèle est figé, toujours la même action
```

**Ce qui se passe** : L'entropie mesure la diversité des décisions. Si elle tombe à zéro, le modèle a convergé vers une politique rigide, souvent basée sur du bruit.

**Métrique** : `train/ent_coef` - surveiller si proche de 0.

### Signe 6 : NAV inconsistant entre épisodes

```
Épisode 1:  +15% return  (a appris ce segment)
Épisode 2:  -5% return   (nouveau segment, ne généralise pas)
Épisode 3:  +12% return  (re-mémorise)
Épisode 4:  -8% return   (re-échoue sur nouveau)
```

**Ce qui se passe** : Le modèle performe bien sur certains régimes de marché mais échoue sur d'autres.

**Signe de convergence saine** : Les returns inter-épisodes doivent être **stables** (CV < 0.2).

### Résumé des seuils d'alerte

| Signe | Métrique | Seuil d'alerte |
|-------|----------|----------------|
| Train >> Eval | `train_eval_divergence` | > 0.2 |
| Loss en U | `critic_loss` trajectory | "U-shape" |
| Actions saturées | `action_saturation` | > 0.8 |
| Rewards instables | `reward_cv` | > 0.5 |
| Entropie effondrée | `ent_coef` | proche de 0 |
| NAV inconsistant | CV épisodes | > 0.3 |

### Comment CryptoRL combat l'overfitting

| Mécanisme | Effet |
|-----------|-------|
| **Observation noise** | Empêche la mémorisation des patterns exacts |
| **Curriculum learning** | Augmente progressivement la difficulté |
| **PLO** | Pénalise les comportements extrêmes (drawdown, churn) |
| **Walk-Forward Optimization** | Entraîne sur segment N, teste sur segment N+1 |
| **TQCDropoutPolicy** | Dropout + LayerNorm pour régularisation |
| **OverfittingGuardV2** | Détecte et stoppe automatiquement l'overfitting |

---

## 12. OverfittingGuardCallbackV2 (5 Signaux)

### Problème
La détection d'overfitting via un seul signal (NAV) est fragile.

### Solution : 5 Signaux Indépendants

| Signal | Détecte | Méthode |
|--------|---------|---------|
| **1. NAV Threshold** | Returns irréalistes (+400%) | `max_nav > 5× initial` |
| **2. Weight Stagnation** | Convergence/collapse | CV des poids < 0.01 |
| **3. Train/Eval Divergence** | Overfitting classique | `train_rew - eval_rew > 50%` |
| **4. Action Saturation** | Policy collapse | >80% actions à ±1 |
| **5. Reward Variance** | Mémorisation | Variance → 0 |

### Logique de Décision
```python
STOP si:
  - Un signal atteint 'patience' violations consécutives (défaut: 3)
  - OU 2+ signaux actifs simultanément
```

### Signal 3 : Prérequis
- Nécessite des **données d'évaluation séparées** temporellement
- Lit `ep_info_buffer` (train) et `EvalCallback.last_mean_reward` (eval)
- **Désactivé en mode WFO** (évite data leakage)

### Métriques TensorBoard
```
overfit/max_nav_ratio
overfit/weight_delta, overfit/weight_cv
overfit/train_eval_divergence
overfit/action_saturation
overfit/reward_variance, overfit/reward_cv
overfit/violations_*, overfit/active_signals
```

---

## 13. Séparation Données Train/Eval

### Problème : Data Leakage
Si train et eval partagent les mêmes données, la mesure de généralisation est faussée.

### Solution : Split Temporel Strict + Purge

```
┌─────────────────────────────────────────────────────────────────┐
│                    DONNÉES HISTORIQUES                          │
│                                                                 │
│   ┌───────────────────┐  ┌───────┐  ┌─────────────────────┐    │
│   │      TRAIN        │  │ PURGE │  │       EVAL          │    │
│   │     (80%)         │  │ (50h) │  │      (20%)          │    │
│   │ 2020-01→2023-06   │  │       │  │ 2023-07→2024-12     │    │
│   └───────────────────┘  └───────┘  └─────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Purge Window (50h)
- Les indicateurs techniques (RSI, MACD) utilisent des données passées
- Sans purge, les features d'Eval contiennent des infos de Train
- 50h couvre la plupart des lookback d'indicateurs

### Fichiers de Données
| Fichier | Contenu | Usage |
|---------|---------|-------|
| `processed_data.parquet` | 80% historique | Training |
| `processed_data_eval.parquet` | 20% historique | Évaluation |

### Différences Train vs Eval Env
| Paramètre | Train | Eval |
|-----------|-------|------|
| `random_start` | True | False |
| `observation_noise` | 0.01 | 0.0 |
| `curriculum` | Actif | Valeurs finales |

---

## 14. Intégration WFO du Guard (Fail-over)

### Problème
Sans Guard en WFO, le training peut continuer même si le modèle a divergé.

### Solution : Guard Partiel en WFO

| Signal | Actif en WFO ? | Raison |
|--------|----------------|--------|
| 1. NAV | ✅ Oui | Pas de dépendance externe |
| 2. Weights | ✅ Oui | Lecture poids du modèle |
| 3. Divergence | ❌ Non | Pas d'EvalCallback |
| 4. Saturation | ✅ Oui | Lecture actions locales |
| 5. Variance | ✅ Oui | Lecture rewards locales |

### Politique Fail-over

```
Guard déclenche l'arrêt
        │
        ▼
┌───────────────────────────────┐
│ completion_ratio >= 30% ?    │
│ ET checkpoint valide ?       │
└───────────────────────────────┘
        │
   ┌────┴────┐
  OUI       NON
   │         │
   ▼         ▼
RECOVERED   FAILED
(dernier    (stratégie
checkpoint) de repli)
```

### Statuts de Segment
| Statut | Description |
|--------|-------------|
| `SUCCESS` | Training complet sans intervention |
| `RECOVERED` | Arrêt Guard, checkpoint valide utilisé |
| `FAILED` | Arrêt Guard, pas de checkpoint valide |

### Chain of Inheritance
Le Segment N+1 hérite des poids du **dernier segment valide** (pas forcément N si N a FAILED).

```
Segment 0 (SUCCESS) → Segment 1 (FAILED) → Segment 2 (SUCCESS)
         │                   │                      ↑
         └───────────────────┴──────────────────────┘
              Segment 2 utilise les poids de Segment 0
```

---

## 15. Métriques de Performance

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

## 16. Fichiers Principaux

| Fichier | Rôle | Lignes |
|---------|------|--------|
| `scripts/run_full_wfo.py` | Orchestration WFO complète | ~1600 |
| `src/training/batch_env.py` | Environnement GPU-vectorisé + Dynamic Noise | ~1200 |
| `src/training/callbacks.py` | Curriculum, PLO, OverfittingGuardV2 | ~1700 |
| `src/training/train_agent.py` | Entraînement TQC | ~880 |
| `src/models/tqc_dropout_policy.py` | TQCDropoutPolicy (DroQ + LayerNorm) | ~420 |
| `src/models/rl_adapter.py` | FoundationFeatureExtractor | ~330 |
| `src/data_engineering/features.py` | Feature engineering (FFD, Vol) | ~650 |
| `scripts/prepare_train_eval_split.py` | Split train/eval avec purge | ~100 |

---

## 17. Termes Techniques Courants

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
| **Annealing** | Réduction progressive d'un paramètre (bruit, learning rate) |
| **DroQ** | Dropout + LayerNorm pour RL (Hiraoka 2021) |
| **CV** | Coefficient de Variation = σ/μ (mesure de dispersion relative) |
| **Purge Window** | Gap temporel entre train et eval pour éviter leakage |
| **Fail-over** | Stratégie de repli en cas d'échec (checkpoint ou stratégie passive) |
| **Policy Collapse** | Dégénérescence de la policy (actions bloquées à ±1) |
| **Weight Stagnation** | Poids du réseau qui ne bougent plus (signe de convergence/collapse) |

---

## 18. Références Essentielles

1. **Lopez de Prado (2018)** - "Advances in Financial Machine Learning"
   - Fractional Differentiation, Meta-Labeling, Purged CV

2. **Kuznetsov et al. (2020)** - "TQC: Truncated Quantile Critics"
   - Algorithme RL principal

3. **Stooke et al. (2020)** - "PID Lagrangian Methods"
   - Base théorique pour PLO

4. **He et al. (2022)** - "Masked Autoencoders Are Scalable Vision Learners"
   - Architecture MAE adaptée aux séries temporelles

5. **Hiraoka et al. (2021)** - "DroQ: Dropout Q-functions"
   - Dropout + LayerNorm pour régularisation en RL

6. **NoisyRollout (2025)** - arXiv:2504.13055
   - Noise annealing pour observation noise

7. **GRADSTOP (2025)** - arXiv:2508.19028
   - Early stopping sans validation set (adapté pour Signal 2)

8. **FineFT (2025)** - arXiv:2512.23773
   - Détection policy collapse via action saturation (Signal 4)

9. **Sparse-Reg (2025)** - arXiv:2506.17155
   - Variance rewards comme proxy généralisation (Signal 5)

---

*Dernière mise à jour : 2026-01-21*

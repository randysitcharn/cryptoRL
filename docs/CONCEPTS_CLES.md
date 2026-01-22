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
| Robustesse | **Ensemble RL** (Multi-Seed + Confidence) | Agrégation de 3 modèles pondérée par incertitude |
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

## 8. MORL - Multi-Objective RL (Avancé)

### Problème
Les pénalités statiques (churn, smoothness) ne s'adaptent pas au contexte du marché.

### Solution : MORL avec paramètre w_cost
```python
reward = r_perf + w_cost * r_cost * MAX_PENALTY_SCALE
```

- **w_cost = 0** : Mode scalping (ignorer les coûts, maximiser profit)
- **w_cost = 1** : Mode B&H (minimiser les coûts, conservateur)

L'agent apprend une politique conditionnée : `π(a|s, w_cost)` et peut s'adapter dynamiquement.

### Distribution de w_cost (Training)
| Probabilité | Valeur | Stratégie |
|-------------|--------|-----------|
| 20% | 0.0 | Scalping pur |
| 20% | 1.0 | B&H pur |
| 60% | Uniforme [0,1] | Exploration |

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

## 11. Ensemble RL - Agrégation Multi-Modèles

### Problème
Un seul modèle TQC présente :
- **Variance inter-seeds** : Performances très variables selon l'initialisation
- **Sensibilité aux outliers** : Un mauvais gradient peut dérailler l'entraînement
- **Décisions "all-in"** : Pas de notion de confiance dans les actions

### Solution : Multi-Seed Ensemble + Confidence-Weighted Aggregation

#### Architecture
```
Observation → [TQC seed=42] → action_0, spread_0
           → [TQC seed=123] → action_1, spread_1
           → [TQC seed=456] → action_2, spread_2
                    ↓
         Confidence-Weighted Aggregation
                    ↓
              Final Action
```

#### Méthode d'Agrégation (Softmax Temperature)
```python
# Spread = incertitude (max - min des quantiles TQC)
# Plus le spread est faible, plus le modèle est confiant

spread_norm = spread / EMA(spread)  # Calibration volatilité
weight_i = exp(-spread_norm / τ) / Σ(exp(-spread_norm / τ))  # Softmax
final_action = Σ(weight_i × action_i)
```

| Paramètre | Valeur | Effet |
|-----------|--------|-------|
| **τ = 1.0** | Standard | Pondération proportionnelle |
| **τ = 0.5** | Agressif | "Tue" les modèles incertains |
| **τ = 2.0** | Doux | Différences atténuées |

### Diversité Forcée (Audit Gemini)
Pour éviter que les modèles convergent vers la même solution :

| Membre | Seed | Gamma | Learning Rate |
|--------|------|-------|---------------|
| 0 | 42 | 0.94 | 5e-5 |
| 1 | 123 | 0.95 | 1e-4 |
| 2 | 456 | 0.96 | 2e-4 |

### Méthodes d'Agrégation Disponibles

| Méthode | Formule | Usage |
|---------|---------|-------|
| **confidence** (défaut) | Softmax(-spread/τ) weighted | Pondère par confiance TQC |
| **mean** | Σ(action_i) / N | Moyenne simple |
| **median** | Médiane des actions | Résistant aux outliers |
| **conservative** | argmin(\|action_i\|) | Plus risk-averse |

### Filtre d'Agrément
```python
agreement = 1.0 - std(actions)  # 1 = accord parfait, 0 = désaccord total

if agreement < min_agreement:
    final_action = 0.0  # Hold si trop de désaccord
```

### ROI Attendu (Audit Gemini)
> *"Diviser le Max Drawdown par 2 via l'ensemble permet souvent de doubler le levier (et donc le profit) à risque constant."*

| Métrique | Single Model | Ensemble 3 | Amélioration |
|----------|--------------|------------|--------------|
| Sharpe | X | ≥ X | ≥ +10% |
| Max DD | Y% | ≤ Y% | ≤ -10% |
| Variance | σ | σ/2 | -50% |

### Training Parallèle (2 GPUs)
```
GPU 0: TQC seed=42   ─┐
GPU 1: TQC seed=123  ─┼→ ~20 min total (vs 30 min séquentiel)
Then GPU 0: seed=456 ─┘
```

### Références
- **DroQ** (Hiraoka 2021) - Dropout comme mini-ensemble implicite
- **TQC** (Kuznetsov 2020) - Quantiles pour incertitude
- Design complet : `docs/design/ENSEMBLE_RL_DESIGN.md`

---

## 12. Comprendre l'Overfitting en RL

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
| **MORL (w_cost)** | Apprend à s'adapter aux différents régimes de coûts |
| **Walk-Forward Optimization** | Entraîne sur segment N, teste sur segment N+1 |
| **TQCDropoutPolicy** | Dropout + LayerNorm pour régularisation |
| **OverfittingGuardV2** | Détecte et stoppe automatiquement l'overfitting |
| **Polyak Averaging (EMA)** | Moyenne mobile exponentielle des poids pour généralisation |
| **Ensemble RL** | Agrégation multi-modèles réduit variance et risque systémique |

---

## 13. Polyak Averaging (EMA) - Robustesse & Généralisation

### Problème
Les poids du policy network pendant l'entraînement sont **instables** et peuvent surajuster aux derniers batches vus. Un poids optimal à l'étape N peut être "perdu" si le modèle continue à s'entraîner.

**Note importante** : SB3/TQC utilise déjà Polyak Averaging pour les **target networks des critics** (via `tau=0.005`), mais **pas** pour le policy network (actor). Cette section décrit une extension custom pour le policy.

### Solution : Exponential Moving Average (EMA)

Polyak Averaging maintient une **moyenne mobile exponentielle** des poids du réseau au fil du temps.

#### Formule
```python
θ_ema = τ × θ_current + (1 - τ) × θ_ema
```

où :
- `θ_current` : Poids actuels du policy network
- `θ_ema` : Poids EMA (Exponential Moving Average)
- `τ` : Coefficient de mise à jour (typiquement 0.005 = lent, 0.01 = moyen)

#### Intuition
```
Étape 1000:  θ_ema = moyenne des poids [1..1000]
Étape 2000:  θ_ema = moyenne des poids [1..2000] (pondéré récursivement)
Étape 5000:  θ_ema = moyenne des poids [1..5000]
```

Les poids EMA "lissent" les fluctuations et capturent une **version robuste** du modèle qui généralise mieux.

#### Analogie
Imagine un trader qui change de stratégie chaque jour (poids actuels) vs un trader qui garde une **moyenne mobile** de ses stratégies récentes (poids EMA). La moyenne est plus stable et généralise mieux aux nouvelles situations.

### Paramètre τ (Tau)

| τ | Vitesse | Usage |
|---|---------|-------|
| **0.001** | Très lent | Modèles très stables, longue convergence |
| **0.005** | Lent | **Défaut CryptoRL** (match TQC target network) |
| **0.01** | Moyen | Équilibre stabilité/réactivité |
| **0.05** | Rapide | Modèles instables, adaptation rapide |

**Règle** : Plus τ est petit, plus l'EMA est stable mais moins réactif aux nouvelles améliorations.

### Utilisation dans CryptoRL

#### 1. Pendant l'entraînement
- À chaque étape (ou N étapes), mise à jour de l'EMA
- Les poids actuels continuent de s'entraîner normalement
- L'EMA est une **copie en arrière-plan** qui ne perturbe pas le training

#### 2. Pour l'évaluation
- **Charger les poids EMA** dans le policy avant évaluation
- Utiliser ces poids pour inférence (plus robustes)
- Les poids actuels restent disponibles pour continuer l'entraînement

#### 3. Avantages pour Trading RL

| Problème | Solution EMA |
|----------|--------------|
| Policy "chase noise" en fin d'entraînement | EMA ignore les fluctuations récentes |
| Variance élevée entre épisodes | EMA réduit la variance (weights smoothing) |
| Overfitting aux dernières données vues | EMA pondère toutes les étapes d'entraînement |

### Métriques TensorBoard

```
ema/weight_diff_relative  # Distance L2 entre θ_current et θ_ema (normalisée)
```

Si `weight_diff_relative` augmente :
- Le modèle continue à apprendre (poids actuels divergent de l'EMA)
- Normal en début d'entraînement
- Anormal si l'augmentation persiste (signe d'instabilité)

### Références

- **Polyak & Juditsky (1992)** - "Acceleration of Stochastic Approximation by Averaging"
  - Base théorique de la méthode
- **Lillicrap et al. (2015)** - DDPG utilise τ=0.001 pour target networks
- **TQC** - Utilise τ=0.005 pour les target critics (même valeur recommandée pour EMA policy)

### Différence avec Target Network de SB3

| Concept | Objectif | Réseau | Mise à jour | Implémenté dans SB3 ? |
|---------|----------|--------|-------------|----------------------|
| **Target Network** | Stabilité du critic (Q-function) | Critics | τ = 0.005 (lent) | ✅ **Oui** (automatique) |
| **EMA Policy** | Robustesse du policy (actor) | Actor | τ = 0.005 (recommandé) | ❌ **Non** (custom callback) |

**Détail** :
- SB3/TQC utilise `tau` pour mettre à jour automatiquement les **target critics** :
  ```python
  θ_target_critic = τ × θ_critic + (1-τ) × θ_target_critic
  ```
- Mais **aucun mécanisme** n'existe pour l'actor. Le `EMACallback` proposé dans le plan est une **extension custom** qui applique la même logique au policy network.

---

## 14. OverfittingGuardCallbackV2 (5 Signaux)

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

## 15. Séparation Données Train/Eval

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

## 16. Intégration WFO du Guard (Fail-over)

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

## 17. Métriques de Performance

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

## 18. Fichiers Principaux

| Fichier | Rôle | Lignes |
|---------|------|--------|
| `scripts/run_full_wfo.py` | Orchestration WFO complète | ~1600 |
| `src/training/batch_env.py` | Environnement GPU-vectorisé + MORL + Dynamic Noise | ~1100 |
| `src/training/callbacks.py` | Curriculum, OverfittingGuardV2, EMA | ~1500 |
| `src/training/train_agent.py` | Entraînement TQC | ~880 |
| `src/evaluation/ensemble.py` | Ensemble RL (Multi-Seed + Confidence Aggregation) | ~900 |
| `src/models/tqc_dropout_policy.py` | TQCDropoutPolicy (DroQ + LayerNorm) | ~420 |
| `src/models/rl_adapter.py` | FoundationFeatureExtractor | ~330 |
| `src/data_engineering/features.py` | Feature engineering (FFD, Vol) | ~650 |
| `scripts/prepare_train_eval_split.py` | Split train/eval avec purge | ~100 |

---

## 19. Termes Techniques Courants

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
| **Polyak Averaging (EMA)** | Moyenne mobile exponentielle des poids pour robustesse (τ=0.005) |
| **Ensemble RL** | Agrégation de plusieurs modèles pour réduire variance et améliorer robustesse |
| **Quantile Spread** | Écart max-min des quantiles TQC, proxy d'incertitude |
| **Confidence-Weighted** | Pondération inversement proportionnelle à l'incertitude |
| **Softmax Temperature** | Paramètre τ contrôlant la "dureté" de la pondération softmax |

---

## 20. Références Essentielles

1. **Lopez de Prado (2018)** - "Advances in Financial Machine Learning"
   - Fractional Differentiation, Meta-Labeling, Purged CV

2. **Kuznetsov et al. (2020)** - "TQC: Truncated Quantile Critics"
   - Algorithme RL principal

3. **Abels et al. (ICML 2019)** - "Dynamic Weights in Multi-Objective Deep RL"
   - Base théorique pour MORL (w_cost)

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

10. **Polyak & Juditsky (1992)** - "Acceleration of Stochastic Approximation by Averaging"
    - Base théorique de Polyak Averaging (EMA) pour robustesse des poids

11. **Ensemble RL through Classifier Models (2025)** - arXiv:2502.17518
    - Agrégation d'ensemble via classifiers, base conceptuelle pour multi-seed

---

*Dernière mise à jour : 2026-01-21*

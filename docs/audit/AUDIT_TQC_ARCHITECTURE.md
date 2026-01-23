# Audit TQC - Architecture & Plan de Diagnostic

**Date**: 2026-01-23  
**Auteur**: Analyse architecturale TQC  
**Statut**: 📋 Plan d'audit proposé

---

## 🎯 Objectif

Le TQC (Truncated Quantile Critics) est le **"Chef d'Orchestre"** du système CryptoRL. S'il est mal réglé, peu importe la qualité du HMM ou du MAE, le système perdra de l'argent.

Ce document répond aux **4 questions architecturales** critiques pour construire un outil de diagnostic parfait.

---

## 🏗️ 1. L'Architecture des Entrées (Input Fusion)

### Question : Quelle est la stratégie de fusion actuelle ?

**Réponse : Option A (Concatenation avec Encoder MAE)**

Le TQC utilise une architecture **hiérarchique en deux étapes** :

#### Étape 1 : Feature Extraction (FoundationFeatureExtractor)

```python
# Observation Dict reçue par TQC
observation = {
    "market": (batch, 64, 43),  # Fenêtre de 64 steps × 43 features
    "position": (batch, 1),     # Position actuelle ∈ [-1, 1]
    "w_cost": (batch, 1)        # Préférence MORL ∈ [0, 1]
}

# Pipeline FoundationFeatureExtractor
market (64, 43) 
  → MAE Encoder (frozen) 
  → (64, 128) embeddings
  → Flatten 
  → (8192) market_features
  → LayerNorm
  → Concat avec position
  → (8193) combined
  → Linear(8193 → 512)
  → LayerNorm
  → LeakyReLU
  → (512) features
```

**Référence** : `src/models/rl_adapter.py:FoundationFeatureExtractor`

#### Étape 2 : Policy Network (TQCDropoutPolicy)

```python
# Les features (512D) sont ensuite passées à l'Actor/Critic
features (512)
  → Actor MLP [256, 256] (avec dropout 0.01, LayerNorm)
  → Action ∈ [-1, 1]

features (512) + action
  → Critic MLP [256, 256] (avec dropout 0.01, LayerNorm)
  → n_quantiles=25 Q-values
```

**Référence** : `src/models/tqc_dropout_policy.py:TQCDropoutPolicy`

### Question : Est-ce que le TQC reçoit les embeddings bruts du MAE ou seulement sa prédiction ?

**Réponse : ✅ Embeddings bruts (recommandé)**

Le TQC reçoit les **embeddings complets** du MAE encoder :
- **Input** : `market` (64, 43) - fenêtre temporelle complète
- **Output MAE** : `(64, 128)` - embeddings par timestep
- **Flatten** : `(8192)` - concaténation de tous les timesteps

**Avantage** : Le TQC a accès à toute l'information temporelle encodée, pas juste une prédiction ponctuelle.

### ⚠️ Point d'Attention : HMM Features

**Question critique** : Où sont les features HMM (`HMM_Prob_0`, `HMM_Prob_1`, `HMM_Prob_2`, `HMM_Prob_3`, `HMM_Entropy`) ?

**Réponse** : Les features HMM sont **intégrées dans le `market` tensor** avant l'encodage MAE.

**Flux de données** :
```
Raw OHLCV 
  → Feature Engineering (FFD, Volatility, etc.)
  → HMM Regime Detection
  → Ajout colonnes HMM_Prob_* et HMM_Entropy
  → RobustScaler (fit sur train uniquement)
  → market tensor (64, 43) incluant HMM features
  → MAE Encoder
  → TQC
```

**Référence** : `src/data_engineering/manager.py:RegimeDetector.get_belief_states_df()`

**Vérification nécessaire** : S'assurer que les features HMM sont bien présentes dans le `market` tensor et que le MAE les encode correctement.

---

## 🧠 2. La Nature du "Cerveau" (Core Policy)

### Question : Quel type de modèle est le TQC ?

**Réponse : Reinforcement Learning (TQC - Truncated Quantile Critics)**

Le TQC est un algorithme **Actor-Critic off-policy** qui :

1. **N'apprend PAS** à prédire une erreur du MAE
2. **N'est PAS** un meta-learner supervisé
3. **Apprend une politique** `π(a|s, w_cost)` pour maximiser la récompense MORL

#### Architecture TQC

```
┌─────────────────────────────────────────────────────────┐
│                    TQC (SB3)                            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────┐         ┌──────────────┐            │
│  │ Actor (π)     │         │ Critics (Q)   │            │
│  │               │         │              │            │
│  │ Input:        │         │ Input:       │            │
│  │  features(512)│         │  features(512)│          │
│  │               │         │  + action    │            │
│  │ MLP:          │         │              │            │
│  │  [256, 256]   │         │ MLP:         │            │
│  │  + Dropout    │         │  [256, 256]  │            │
│  │  + LayerNorm  │         │  + Dropout    │            │
│  │               │         │  + LayerNorm  │            │
│  │ Output:       │         │              │            │
│  │  action ∈ [-1,1]│        │ Output:      │            │
│  │               │         │  n_quantiles=25│          │
│  │               │         │  Q-values     │            │
│  └──────────────┘         └──────────────┘            │
│                                                         │
│  Loss: Actor-Critic avec Truncated Quantiles          │
│  - Critic Loss: Huber loss sur quantiles              │
│  - Actor Loss: -Q(s, π(s)) (policy gradient)         │
│  - Entropy Bonus: Exploration (gSDE)                  │
└─────────────────────────────────────────────────────────┘
```

**Référence** : `docs/design/MODELES_RL_DESIGN.md:3.1 TQC`

#### Algorithme d'Apprentissage

```python
# Boucle d'entraînement TQC
for step in range(total_timesteps):
    # 1. Sample action from policy
    action = π(observation) + gSDE_noise
    
    # 2. Execute in environment
    next_obs, reward, done = env.step(action)
    
    # 3. Store transition
    replay_buffer.add(obs, action, reward, next_obs, done)
    
    # 4. Update TQC (off-policy)
    if step > learning_starts:
        batch = replay_buffer.sample(batch_size=2048)
        
        # Critic update: Estimate Q-distribution (25 quantiles)
        critic_loss = huber_loss(quantiles, target_quantiles)
        
        # Actor update: Maximize Q(s, π(s))
        actor_loss = -Q(s, π(s)) + entropy_bonus
        
        # Soft update target networks (τ=0.005)
        θ_target = τ * θ + (1-τ) * θ_target
```

**Référence** : `src/training/train_agent.py:train()`

---

## 🎯 3. La Sortie Décisionnelle (Action Space)

### Question : Que contrôle le TQC exactement ?

**Réponse : Action directe (Position cible)**

Le TQC contrôle directement l'**action** (position cible) :

```python
action ∈ [-1, 1]  # -1 = 100% short, 0 = cash, +1 = 100% long
```

#### Types de contrôle possibles :

| Type | Description | Implémenté ? |
|------|-------------|--------------|
| **Signal de Confiance** | Multiplicateur de taille | ❌ Non |
| **Signal "Go / No-Go"** | Filtre binaire (Trade/Cash) | ❌ Non |
| **Side Correction** | Peut inverser le MAE | ✅ **OUI** |
| **Action Directe** | Position cible ∈ [-1, 1] | ✅ **OUI** |

#### Architecture de Décision

```
TQC Policy π(s, w_cost)
  ↓
Action ∈ [-1, 1]
  ↓
Action Discretization (optionnel, 21 niveaux si action_discretization=0.1)
  ↓
New Position = discretize(action)
  ↓
Execute Trade: Δposition = new_position - current_position
  ↓
Reward = r_perf + w_cost * r_cost * MAX_PENALTY_SCALE
```

**Référence** : `src/training/batch_env.py:BatchCryptoEnv._calculate_reward()`

#### Peut-il inverser le MAE ?

**OUI**, le TQC peut complètement ignorer ou inverser toute suggestion du MAE car :

1. Le MAE est **frozen** (pas de gradient)
2. Le TQC apprend **end-to-end** la politique optimale
3. Si le MAE suggère "Long" mais le TQC voit un piège (via HMM entropy élevée, volatilité, etc.), il peut shorter

**Exemple** :
```python
# MAE encodeur suggère pattern bullish (via embeddings)
# Mais TQC voit :
#   - HMM_Entropy élevée (régime incertain)
#   - Position déjà longue (surcharge)
#   - w_cost élevé (coûts importants)
# → TQC peut décider action = -0.5 (short partiel)
```

**Référence** : `src/models/rl_adapter.py:FoundationFeatureExtractor` (encoder frozen)

---

## 📉 4. La Loss Function (Si Supervisé)

### Question : Si le TQC est entraîné, qu'est-ce qu'il optimise ?

**Réponse : Reinforcement Learning (pas de loss supervisée)**

Le TQC est entraîné via **Reinforcement Learning**, donc il optimise :

#### Objectif : Maximiser la récompense cumulée (discountée)

```python
# Objectif RL
J(π) = E[Σ γ^t * r_t]

# Où la reward est MORL (Multi-Objective)
r_t = r_perf + w_cost * r_cost * MAX_PENALTY_SCALE

# Avec :
r_perf = log1p(clamp(returns, -0.2, 0.2)) * SCALE  # SCALE=100
r_cost = -|Δposition| * SCALE
w_cost ∈ [0, 1]  # Préférence MORL (dans observation)
MAX_PENALTY_SCALE = 2.0
```

**Référence** : `src/training/batch_env.py:BatchCryptoEnv._calculate_reward()`

#### Loss Functions TQC

**Critic Loss** (Huber loss sur quantiles) :
```python
# Pour chaque critic (n_critics=2)
# Pour chaque quantile (n_quantiles=25)
quantile_loss = huber_loss(
    predicted_quantile,
    target_quantile  # Calculé via Bellman backup
)

# Truncation : Drop top_quantiles_to_drop=2 quantiles extrêmes
critic_loss = mean(quantile_losses[keep_quantiles])
```

**Actor Loss** (Policy gradient) :
```python
# Maximize Q(s, π(s))
actor_loss = -mean(Q(s, π(s))) + entropy_bonus

# Où Q(s, a) = mean(truncated_quantiles)
```

**Référence** : `sb3_contrib.tqc.policies.TQCPolicy` (implémentation SB3)

#### Ce que le TQC n'optimise PAS

| Métrique | Optimisé ? | Raison |
|----------|-----------|--------|
| **Accuracy du trade** | ❌ Non | Pas de labels supervisés |
| **PnL pondéré** | ✅ Indirectement | Via reward r_perf |
| **Calibration (BCE)** | ❌ Non | Pas de classification |
| **Sharpe Ratio** | ✅ Indirectement | Via reward (log-returns) |

**Note** : Le TQC optimise le **PnL via la reward**, mais pas directement. La reward est une approximation du PnL (log-returns) avec pénalités de coûts.

---

## 📋 Plan de Diagnostic TQC Proposé

### A. Analyse de la "Méta-Confiance" (Calibration Audit)

**Objectif** : Vérifier si le TQC sait quand il a raison.

#### A.1 Reliability Diagram

**Test** : Quand le TQC est "confiant" (Q-values élevées), est-ce qu'on gagne vraiment ?

```python
# Méthodologie
1. Collecter (Q_value, actual_return) pour chaque step
2. Binner Q_values en déciles [0-10%, 10-20%, ..., 90-100%]
3. Pour chaque bin, calculer :
   - Mean Q_value (confiance prédite)
   - Mean actual_return (réalité)
   - Win rate (% de trades profitables)
4. Plot Reliability Diagram:
   - Axe X: Q_value bins
   - Axe Y: Actual return / Win rate
   - Ligne idéale: y = x (calibration parfaite)
```

**Métriques** :
- **ECE (Expected Calibration Error)** : Écart moyen entre confiance et réalité
- **Brier Score** : Score de calibration (0 = parfait)
- **Overconfidence** : Si Q_values > actual_returns systématiquement

#### A.2 Entropy Correlation

**Test** : Le TQC baisse-t-il sa confiance quand `HMM_Entropy` est élevée ?

```python
# Méthodologie
1. Collecter (HMM_Entropy, Q_value_std, action_magnitude) pour chaque step
2. Calculer corrélations :
   - corr(HMM_Entropy, Q_value_std)  # Incertitude TQC vs incertitude HMM
   - corr(HMM_Entropy, |action|)     # Position size vs incertitude HMM
3. Plot scatter:
   - Axe X: HMM_Entropy
   - Axe Y: Q_value_std (ou |action|)
   - Attendu: Corrélation négative (entropy élevée → confiance faible)
```

**Vérification** : Si corrélation faible/nulle, le TQC ignore le HMM (problème d'architecture).

---

### B. Analyse d'Attribution (Feature Importance)

**Objectif** : Comprendre pourquoi le TQC prend une décision.

#### B.1 SHAP Values / Gradient Attribution

**Test** : Quelles features influencent le plus les décisions TQC ?

```python
# Méthodologie
1. Sample N observations (N=1000)
2. Pour chaque observation:
   - Calculer gradients: ∇_features Q(s, a)
   - Ou utiliser SHAP: shap_values = explainer.shap_values(obs)
3. Agrégation:
   - Mean |gradient| par feature
   - Feature importance ranking
4. Vérifications spécifiques:
   - HMM_Prob_* : Impact sur action ?
   - HMM_Entropy : Impact sur |action| ?
   - Position : Impact sur action (hold vs trade) ?
   - w_cost : Impact sur action (scalping vs B&H) ?
```

**Métriques** :
- **Feature Importance Ranking** : Top 10 features les plus influentes
- **HMM Sensitivity** : Si HMM features sont en bas du ranking → problème
- **Position Sensitivity** : Si position n'influence pas → problème (agent ignore son état)

#### B.2 Ablation Study

**Test** : Que se passe-t-il si on retire certaines features ?

```python
# Méthodologie
1. Baseline: TQC avec toutes les features
2. Ablations:
   - Sans HMM features (Prob_*, Entropy)
   - Sans position
   - Sans w_cost
   - Sans MAE embeddings (features brutes uniquement)
3. Comparer:
   - Sharpe Ratio
   - Win Rate
   - Max Drawdown
   - Action distribution
```

**Vérification** : Si retirer HMM ne change rien → TQC ignore HMM (problème).

---

### C. Analyse de la "Value Add" (PnL Uplift)

**Objectif** : Test financier ultime - le TQC ajoute-t-il de la valeur ?

#### C.1 Baseline Comparison

**Test** : Comparer PnL avec/sans TQC

```python
# Méthodologie
1. Courbe A (Naive) : Suivre aveuglément le MAE
   - action = sign(MAE_prediction) * 1.0  # Full size
   
2. Courbe B (TQC) : Utiliser la politique TQC
   - action = π(observation)
   
3. Courbe C (Oracle) : Action parfaite (look-ahead)
   - action = sign(future_return) * 1.0
   
4. Métriques comparatives:
   - Sharpe Ratio
   - Total Return
   - Max Drawdown
   - Win Rate
   - Calmar Ratio
```

**Métrique clé** : **Delta = Sharpe_TQC - Sharpe_Naive**

Si Delta < 0 → TQC détruit de la valeur (problème critique).

#### C.2 Regime-Specific Performance

**Test** : Le TQC performe-t-il mieux dans certains régimes HMM ?

```python
# Méthodologie
1. Segmenter les trades par HMM state dominant
   - State 0 (Crash): Trades où HMM_Prob_0 > 0.5
   - State 1 (Downtrend): Trades où HMM_Prob_1 > 0.5
   - State 2 (Range): Trades où HMM_Prob_2 > 0.5
   - State 3 (Uptrend): Trades où HMM_Prob_3 > 0.5

2. Calculer métriques par régime:
   - Sharpe Ratio par régime
   - Win Rate par régime
   - Avg Return par régime

3. Vérification:
   - TQC devrait outperformer dans régimes incertains (entropy élevée)
   - TQC devrait être conservateur en Crash (State 0)
```

**Vérification** : Si TQC performe mal dans certains régimes → problème de calibration.

---

### D. Analyse de la Distribution des Actions

**Objectif** : Vérifier que le TQC explore correctement l'espace d'action.

#### D.1 Action Distribution Analysis

```python
# Méthodologie
1. Collecter toutes les actions prédites par TQC
2. Analyser:
   - Histogramme des actions
   - Saturation: % d'actions à ±1.0 (ou proche)
   - Mode: Distribution unimodale vs multimodale
   - Entropy des actions: H(π) = -Σ π(a) log π(a)

3. Vérifications:
   - Si > 95% actions saturées → Policy collapse
   - Si distribution trop étroite → Pas d'exploration
   - Si entropy trop faible → Policy trop déterministe
```

**Référence** : `docs/audit/AUDIT_SMALL_POSITIONS.md` (déjà identifié)

---

### E. Analyse de la Convergence

**Objectif** : Vérifier que le TQC apprend correctement.

#### E.1 Training Curves Analysis

```python
# Métriques à monitorer (déjà dans TensorBoard)
1. Q-values:
   - Mean Q-value (devrait augmenter)
   - Std Q-value (devrait diminuer si convergence)
   
2. Actor/Critic Loss:
   - Critic loss (devrait diminuer)
   - Actor loss (devrait converger)
   
3. Entropy:
   - Policy entropy (devrait diminuer progressivement)
   
4. Rewards:
   - Mean reward (devrait augmenter)
   - Reward variance (devrait diminuer)
```

**Vérification** : Si Q-values divergent ou loss explose → problème de stabilité.

---

### F. Analyse des Quantiles (Risk Awareness)

**Objectif** : Vérifier que le TQC est "conscient du risque" grâce à ses quantiles.

#### F.1 Inter-Quantile Range (IQR) vs HMM_Entropy

**Test** : Quand `HMM_Entropy` est élevée (incertitude marché), l'IQR du TQC doit augmenter (incertitude modèle).

**Méthodologie** :
```python
# Pour chaque step:
1. Extraire les 25 quantiles du critic TQC
   - quantiles = model.policy.critic(obs, action)
   - Shape: (batch, n_critics=2, n_quantiles=25)
   - Moyenne sur les critics: (batch, 25)

2. Calculer IQR = Q90 - Q10
   - q10_idx = int(0.10 * 25) = 2
   - q90_idx = int(0.90 * 25) = 22
   - iqr = quantiles[q90_idx] - quantiles[q10_idx]

3. Extraire HMM_Entropy depuis le dataframe de test

4. Calculer corrélation: corr(HMM_Entropy, IQR)
   - Attendu: Corrélation positive (high entropy → high IQR)
   - Si corrélation < 0.3 → TQC ignore l'incertitude du marché (OVERCONFIDENCE)
```

**Métriques** :
- **Corrélation (HMM_Entropy, IQR)** : Doit être ≥ 0.3
- **Overconfidence Flag** : Si corrélation < 0.3, TQC est trop confiant

**Interprétation** :
- ✅ **Corrélation ≥ 0.3** : TQC répond correctement à l'incertitude du marché
- ⚠️ **Corrélation < 0.3** : TQC est overconfident (dangerous) - ignore l'incertitude HMM

**Plot** : Scatter plot `HMM_Entropy` (axe X) vs `IQR` (axe Y) avec ligne de corrélation.

**Référence** : `scripts/audit_pipeline.py:analyze_tqc_quantiles()`

---

## 🚀 Prochaines Étapes

### 1. Implémentation du Script d'Audit

**Fichier** : `scripts/audit_tqc.py`

**Fonctions à implémenter** :
- `analyze_tqc_calibration(model, test_data)` → Reliability Diagram
- `analyze_tqc_attribution(model, test_data)` → SHAP/Gradient Attribution
- `analyze_tqc_value_add(model, test_data, baseline)` → PnL Uplift
- `analyze_tqc_action_distribution(model, test_data)` → Action Analysis
- `analyze_tqc_regime_performance(model, test_data)` → Regime-Specific

### 2. Collecte de Données

**Prérequis** :
- Modèle TQC entraîné (`weights/wfo/segment_X/tqc.zip`)
- Données de test (parquet avec HMM features)
- Baseline MAE (pour comparaison)

### 3. Génération de Rapports

**Output** :
- `results/audit_tqc/calibration_report.md`
- `results/audit_tqc/attribution_report.md`
- `results/audit_tqc/value_add_report.md`
- `results/audit_tqc/plots/*.png`

---

## 📚 Références

- **TQC Paper** : Kuznetsov et al. (2020) - "Controlling Overestimation Bias with Truncated Mixture of Continuous Distributional Quantile Critics"
- **DroQ Paper** : Hiraoka et al. (2021) - "Dropout Q-Functions for Doubly Efficient RL"
- **MORL Paper** : Abels et al. (2019) - "Multi-Objective Reinforcement Learning"
- **Design Docs** :
  - `docs/design/MODELES_RL_DESIGN.md`
  - `docs/design/MORL_DESIGN.md`
  - `docs/design/DROPOUT_TQC_DESIGN.md`
- **Code Sources** :
  - `src/models/rl_adapter.py` - FoundationFeatureExtractor
  - `src/models/tqc_dropout_policy.py` - TQCDropoutPolicy
  - `src/training/train_agent.py` - Training loop
  - `src/training/batch_env.py` - Environment & Reward

---

**Statut** : ✅ Architecture cartographiée - Prêt pour implémentation audit

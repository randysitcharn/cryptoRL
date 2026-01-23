# Master Plan: Audit Environnement BatchCryptoEnv - CryptoRL

**Date**: 2026-01-23  
**Méthode**: State-of-the-Art Audit Framework  
**Objectif**: Audit exhaustif et critique de l'environnement de trading BatchCryptoEnv  
**Référence**: Gymnasium VecEnv, Abels et al. (2019) MORL, Lopez de Prado (2018)

---

## 📋 Méta-Informations

- **Complexité totale estimée**: 45 points
- **Nombre de prompts atomiques**: 16
- **Chemins parallélisables**: 
  - Batch 1: P1.1 ‖ P1.2 ‖ P1.3 ‖ P1.4
  - Batch 2: P2.1 ‖ P2.2 ‖ P2.3 ‖ P2.4
  - Batch 3: P3.1 ‖ P3.2 ‖ P3.3
  - Batch 4: P4.1 ‖ P4.2
  - Batch 5: P5

---

## 🎯 Phase 0 : Clarification (Pré-Analyse)

| Question | Réponse | Statut |
|----------|---------|--------|
| L'objectif final est-il mesurable/vérifiable ? | Rapport d'audit avec scores par composant, findings critiques (P0/P1/P2), recommandations priorisées, métriques quantitatives | ✅ |
| Les contraintes techniques sont-elles explicites ? | Python 3.10+, PyTorch 2.x, GPU CUDA, SB3 VecEnv interface, Gymnasium spaces | ✅ |
| Le scope est-il borné ? | BatchCryptoEnv uniquement - reward, trading mechanics, MORL, vectorization | ✅ |

**Scope IN**:
- `BatchCryptoEnv` class (`src/training/batch_env.py`)
- Observation space (market, position, w_cost)
- Action space et discretization
- Reward function (MORL scalarization, log returns, penalties)
- Trading mechanics (position management, fees, slippage, funding)
- Volatility scaling (EMA variance, risk parity)
- Domain randomization (commission, slippage)
- Episode management (reset, termination, bankruptcy)
- GPU vectorization et performance
- Observation noise (anti-overfitting)
- Data handling (window stacking, feature extraction)

**Scope OUT**:
- Feature engineering (FFD, HMM) → déjà audité
- Data pipeline orchestration → déjà audité
- RL agent (TQC, policy) → scope modèles RL
- Callbacks RL → scope modèles RL

---

## 🌳 Arbre de Décomposition

```
Root: "Audit Environnement SOTA"
│
├─→ P1: Audit Architecture & Design (parallèle)
│   ├─‖ P1.1: Audit Observation Space (ATOMIC)
│   ├─‖ P1.2: Audit Action Space & Discretization (ATOMIC)
│   ├─‖ P1.3: Audit MORL Implementation (ATOMIC)
│   └─‖ P1.4: Audit Episode Management (ATOMIC)
│
├─→ P2: Audit Trading Mechanics (parallèle, dépend P1)
│   ├─‖ P2.1: Audit Reward Function (ATOMIC)
│   ├─‖ P2.2: Audit Position Management (ATOMIC)
│   ├─‖ P2.3: Audit Cost Model (ATOMIC)
│   └─‖ P2.4: Audit Volatility Scaling (ATOMIC)
│
├─→ P3: Audit Robustness & Performance (parallèle, dépend P2)
│   ├─‖ P3.1: Audit Numerical Stability (ATOMIC)
│   ├─‖ P3.2: Audit GPU Vectorization (ATOMIC)
│   └─‖ P3.3: Audit Domain Randomization (ATOMIC)
│
├─→ P4: Audit Data & Integration (parallèle, dépend P3)
│   ├─‖ P4.1: Audit Data Handling (ATOMIC)
│   └─‖ P4.2: Audit Observation Noise (ATOMIC)
│
└─→ P5: Synthèse & Recommandations (ATOMIC, dépend P4)
```

**Légende**: → séquentiel | ‖ parallèle

---

## 📝 Prompts Exécutables

---

### Batch 1 : Audit Architecture & Design

---

### Étape 1.1 : Audit Observation Space

**ID**: `P1.1`  
**Dépendances**: Aucune  
**Parallélisable avec**: P1.2, P1.3, P1.4  
**Score complexité**: 6 (design space + validation théorique)

**Prompt Optimisé**:
```text
## Audit Observation Space - BatchCryptoEnv

### Persona
Tu es un expert en design d'environnements RL pour le trading avec 10+ ans d'expérience. Tu connais les pièges classiques (information leakage, non-stationnarité, curse of dimensionality).

### Contexte
- Fichier: `src/training/batch_env.py` (lignes 185-207)
- Observation space: Dict{
  - "market": Box(shape=(64, n_features)) - fenêtre temporelle
  - "position": Box(low=-1.0, high=1.0, shape=(1,)) - position actuelle
  - "w_cost": Box(low=0.0, high=1.0, shape=(1,)) - paramètre MORL
}
- Window size: 64 steps
- Features: n_features colonnes du DataFrame (excluant EXCLUDE_COLS)

### Tâche
Auditer l'observation space selon les standards SOTA:

1. **Complétude de l'Information**
   - Vérifier que toutes les informations nécessaires sont présentes
   - Analyser si des features manquantes sont critiques (volume, spread, order book depth)
   - Valider que la position est bien incluse (nécessaire pour éviter churn)

2. **Window Size Justification**
   - Valider que window_size=64 est optimal (pas trop court, pas trop long)
   - Comparer avec la littérature (64-128 steps = standard)
   - Tester l'impact de différentes tailles (32, 64, 128, 256)

3. **MORL w_cost Integration**
   - Vérifier que w_cost est bien visible dans l'observation
   - Valider que la distribution de sampling (20/60/20) est correcte
   - Analyser l'impact sur la capacité de l'agent à conditionner sa politique

4. **Information Leakage**
   - Vérifier qu'il n'y a pas de look-ahead bias
   - Valider que les features sont bien calculées avec seulement les données passées
   - Tester avec un oracle (future data) pour détecter le leakage

5. **Normalization & Scaling**
   - Vérifier que les features sont normalisées (z-score, min-max)
   - Analyser l'impact de features non-normalisées sur l'apprentissage
   - Valider la cohérence des échelles entre features

6. **Stationnarité**
   - Tester la stationnarité des features (ADF, KPSS)
   - Analyser l'impact de la non-stationnarité sur l'apprentissage
   - Proposer des transformations si nécessaire

### Livrables
1. Rapport d'audit avec scores par composant (0-10)
2. Tests de look-ahead bias (oracle test)
3. Analyse de stationnarité (ADF, KPSS)
4. Comparaison window sizes (grid search)
5. Liste de findings (P0/P1/P2) avec recommandations
6. Code de validation reproductible
```

**Métriques de Succès**:
- ✅ Aucun look-ahead bias détecté (tests oracle passés)
- ✅ Window size optimal (test grid search)
- ✅ w_cost bien intégré dans observation
- ✅ Features normalisées et stationnaires
- ✅ Observation space complet (pas de features critiques manquantes)

---

### Étape 1.2 : Audit Action Space & Discretization

**ID**: `P1.2`  
**Dépendances**: Aucune  
**Parallélisable avec**: P1.1, P1.3, P1.4  
**Score complexité**: 5 (action space design)

**Prompt Optimisé**:
```text
## Audit Action Space & Discretization

### Persona
Tu es un expert en action spaces pour RL continu avec expertise en discretization et churn reduction.

### Contexte
- Action space: Box(low=-1.0, high=1.0, shape=(1,))
- Discretization: action_discretization=0.1 (21 niveaux: -1.0, -0.9, ..., 0.9, 1.0)
- Fichier: `src/training/batch_env.py` (lignes 208-212, 668-675)
- Mapping: action → position_pct (direct mapping avec vol scaling)

### Tâche
Auditer l'action space et la discretization selon les standards SOTA:

1. **Action Space Design**
   - Valider que [-1, 1] est approprié pour le trading (long/short/cash)
   - Comparer avec d'autres designs (discret, multi-discret, hierarchical)
   - Analyser l'impact sur l'exploration

2. **Discretization Strategy**
   - Valider que discretization=0.1 réduit bien le churn
   - Tester différents niveaux (0.0, 0.05, 0.1, 0.2)
   - Analyser le compromis granularité vs churn

3. **Volatility Scaling Integration**
   - Vérifier que vol scaling est appliqué AVANT discretization
   - Valider que effective_actions = raw_actions * vol_scalar
   - Analyser l'impact sur la granularité effective

4. **Edge Cases**
   - Tester les actions extrêmes (-1.0, 0.0, 1.0)
   - Valider que les actions sont bien clampées
   - Analyser le comportement avec vol scaling extrême

5. **Churn Reduction**
   - Mesurer l'impact de discretization sur le turnover
   - Comparer avec/sans discretization
   - Valider que le churn est réduit sans perte de performance

### Livrables
1. Rapport d'audit avec scores par aspect
2. Tests de churn avec/sans discretization
3. Grid search sur discretization levels
4. Analyse de vol scaling impact
5. Liste de findings (P0/P1/P2) avec recommandations
6. Code de validation reproductible
```

**Métriques de Succès**:
- ✅ Discretization réduit le churn de >30%
- ✅ Vol scaling appliqué correctement (avant discretization)
- ✅ Action space approprié pour le trading
- ✅ Edge cases gérés (clamping, vol extremes)

---

### Étape 1.3 : Audit MORL Implementation

**ID**: `P1.3`  
**Dépendances**: Aucune  
**Parallélisable avec**: P1.1, P1.2, P1.4  
**Score complexité**: 7 (MORL theory + implementation)

**Prompt Optimisé**:
```text
## Audit MORL Implementation

### Persona
Tu es un chercheur en MORL avec expertise en scalarization methods et conditioned networks (Abels et al., 2019).

### Contexte
- Architecture: Conditioned Network avec w_cost ∈ [0, 1]
- Scalarization: reward = r_perf + w_cost * r_cost * MAX_PENALTY_SCALE
- Distribution: 20% w=0, 20% w=1, 60% uniform
- Fichier: `src/training/batch_env.py` (lignes 398-483, 539-557, 606-614)

### Tâche
Auditer l'implémentation MORL selon les standards SOTA:

1. **Scalarization Method**
   - Valider que la scalarisation linéaire est appropriée
   - Comparer avec d'autres méthodes (Tchebycheff, weighted sum, Pareto)
   - Analyser l'impact sur le Pareto front

2. **w_cost Distribution**
   - Valider que la distribution 20/60/20 explore bien les extrêmes
   - Tester différentes distributions (uniform, beta, custom)
   - Analyser l'impact sur la diversité des politiques

3. **MAX_PENALTY_SCALE Calibration**
   - Vérifier que MAX_PENALTY_SCALE=2.0 équilibre r_perf et r_cost
   - Analyser l'ordre de grandeur (r_perf vs r_cost * MAX_PENALTY_SCALE)
   - Tester différents scalings (0.5, 1.0, 2.0, 5.0)

4. **Conditioned Network**
   - Vérifier que w_cost est bien dans l'observation
   - Valider que l'agent peut conditionner sa politique sur w_cost
   - Analyser l'impact sur la capacité d'apprentissage

5. **Evaluation Mode**
   - Vérifier que set_eval_w_cost() fonctionne correctement
   - Valider la reproductibilité avec w_cost fixe
   - Analyser le Pareto front généré

6. **Theoretical Validation**
   - Comparer avec la littérature (Abels 2019, Hayes 2022)
   - Valider que l'implémentation est conforme aux standards
   - Identifier les écarts et justifier

### Livrables
1. Rapport d'audit avec validation théorique
2. Tests de calibration MAX_PENALTY_SCALE
3. Analyse de distribution w_cost (diversité)
4. Validation conditioned network (w_cost impact)
5. Liste de findings (P0/P1/P2) avec recommandations
6. Code de validation reproductible
```

**Métriques de Succès**:
- ✅ Scalarization conforme à Abels 2019
- ✅ MAX_PENALTY_SCALE calibré (r_perf ≈ r_cost * MAX_PENALTY_SCALE)
- ✅ Distribution w_cost explore les extrêmes (>80% coverage)
- ✅ Conditioned network fonctionne (w_cost impact visible)
- ✅ Evaluation mode reproductible

---

### Étape 1.4 : Audit Episode Management

**ID**: `P1.4`  
**Dépendances**: Aucune  
**Parallélisable avec**: P1.1, P1.2, P1.3  
**Score complexité**: 5 (episode lifecycle)

**Prompt Optimisé**:
```text
## Audit Episode Management

### Persona
Tu es un expert en gestion d'épisodes pour RL avec expertise en termination conditions et auto-reset.

### Contexte
- Episode length: 2048 steps
- Termination: time limit (episode_length) ou bankruptcy (nav <= 0)
- Reset: random_start=True (training) ou False (evaluation)
- Fichier: `src/training/batch_env.py` (lignes 485-559, 797-830)

### Tâche
Auditer la gestion d'épisodes selon les standards SOTA:

1. **Episode Length**
   - Valider que episode_length=2048 est optimal
   - Comparer avec d'autres longueurs (1024, 2048, 4096)
   - Analyser l'impact sur l'apprentissage (horizon effectif)

2. **Termination Conditions**
   - Vérifier que time limit est bien appliqué
   - Valider que bankruptcy (nav <= 0) est détecté
   - Analyser l'impact du bankruptcy penalty (-1.0)

3. **Reset Strategy**
   - Valider que random_start explore bien l'espace temporel
   - Vérifier que sequential start (eval) est reproductible
   - Analyser l'impact sur la diversité des épisodes

4. **Auto-Reset**
   - Vérifier que les envs terminés sont bien reset
   - Valider que les stats d'épisode sont capturées avant reset
   - Analyser l'impact sur le monitoring SB3

5. **Episode Boundaries**
   - Vérifier qu'il n'y a pas de leakage entre épisodes
   - Valider que les états sont bien réinitialisés
   - Analyser l'impact sur la cohérence

6. **Edge Cases**
   - Tester avec données courtes (< episode_length)
   - Valider le comportement avec bankruptcy immédiat
   - Analyser les cas limites (max_start < min_start)

### Livrables
1. Rapport d'audit avec validation des conditions
2. Tests de termination (time limit, bankruptcy)
3. Analyse de reset strategy (diversité)
4. Tests d'edge cases
5. Liste de findings (P0/P1/P2) avec recommandations
6. Code de validation reproductible
```

**Métriques de Succès**:
- ✅ Episode length optimal (test grid search)
- ✅ Termination conditions correctes (time limit + bankruptcy)
- ✅ Reset strategy explore l'espace (>90% coverage)
- ✅ Auto-reset fonctionne (stats capturées)
- ✅ Pas de leakage entre épisodes

---

### Batch 2 : Audit Trading Mechanics

---

### Étape 2.1 : Audit Reward Function

**ID**: `P2.1`  
**Dépendances**: P1.3  
**Parallélisable avec**: P2.2, P2.3, P2.4  
**Score complexité**: 8 (reward design critique)

**Prompt Optimisé**:
```text
## Audit Reward Function

### Persona
Tu es un quant researcher senior avec 10+ ans d'expérience en reward design pour trading RL. Tu connais les pièges classiques (reward hacking, non-stationnarité, scale mismatch).

### Contexte
- Reward: r_perf + w_cost * r_cost * MAX_PENALTY_SCALE
- r_perf: log1p(safe_returns) * SCALE (SCALE=100.0)
- r_cost: -position_deltas * SCALE
- MAX_PENALTY_SCALE: 0.0 (désactivé actuellement)
- Fichier: `src/training/batch_env.py` (lignes 398-483)

### Tâche
Auditer la reward function selon les standards SOTA:

1. **Log Returns Justification**
   - Valider que log1p() est approprié (vs simple returns)
   - Analyser l'impact sur la distribution des rewards
   - Vérifier la stabilité numérique (clamp à -0.99)

2. **SCALE Calibration**
   - Valider que SCALE=100.0 est optimal
   - Analyser l'ordre de grandeur des rewards
   - Tester différents scalings (10, 50, 100, 200)

3. **MAX_PENALTY_SCALE**
   - Analyser pourquoi MAX_PENALTY_SCALE=0.0 (désactivé)
   - Valider que r_perf et r_cost sont du même ordre de grandeur
   - Tester différents scalings (0.5, 1.0, 2.0, 5.0)

4. **Reward Hacking Detection**
   - Identifier les stratégies de reward hacking possibles
   - Tester avec des actions extrêmes (churn excessif, positions fixes)
   - Analyser la robustesse de la reward

5. **Non-Stationnarité**
   - Vérifier que la reward est stationnaire (même distribution dans le temps)
   - Analyser l'impact de la volatilité sur la reward
   - Proposer des normalisations si nécessaire

6. **Theoretical Validation**
   - Comparer avec la littérature (log returns standard en finance)
   - Valider que la reward aligne avec l'objectif (maximiser Sharpe/Sortino)
   - Identifier les écarts et justifier

### Livrables
1. Rapport d'audit avec validation théorique
2. Tests de calibration SCALE et MAX_PENALTY_SCALE
3. Tests de reward hacking (stratégies extrêmes)
4. Analyse de stationnarité (distribution temporelle)
5. Liste de findings (P0/P1/P2) avec recommandations
6. Code de validation reproductible
```

**Métriques de Succès**:
- ✅ Log returns justifiés théoriquement
- ✅ SCALE calibré (rewards dans [-10, 10] typiquement)
- ✅ MAX_PENALTY_SCALE équilibre r_perf et r_cost
- ✅ Pas de reward hacking détecté
- ✅ Reward stationnaire (distribution stable)

---

### Étape 2.2 : Audit Position Management

**ID**: `P2.2`  
**Dépendances**: P1.2  
**Parallélisable avec**: P2.1, P2.3, P2.4  
**Score complexité**: 7 (trading mechanics)

**Prompt Optimisé**:
```text
## Audit Position Management

### Persona
Tu es un quant trader avec expertise en execution algorithms et position sizing pour le trading algorithmique.

### Contexte
- Position: direct mapping action → position_pct (avec vol scaling)
- Long: position_pct > 0, Short: position_pct < 0, Cash: position_pct = 0
- Execution: seulement si position_changed (optimisation)
- Fichier: `src/training/batch_env.py` (lignes 677-726)

### Tâche
Auditer la gestion des positions selon les standards SOTA:

1. **Position Calculation**
   - Vérifier que target_exposures = target_positions (direct mapping)
   - Valider que target_units = target_values / old_prices
   - Analyser l'impact du vol scaling sur les positions

2. **Trade Execution**
   - Vérifier que les trades sont exécutés uniquement si position_changed
   - Valider que units_delta est calculé correctement
   - Analyser l'impact sur les coûts (pas de trade inutile)

3. **Short Selling**
   - Vérifier que les positions négatives sont bien gérées
   - Valider que le cash augmente lors d'un short (proceeds)
   - Analyser l'impact du funding rate sur les shorts

4. **Position Limits**
   - Vérifier que les positions sont bien clampées [-1, 1]
   - Valider que max_leverage est respecté (via vol scaling)
   - Analyser les edge cases (position = ±1, cash = 0)

5. **NAV Calculation**
   - Vérifier que NAV = cash + positions * price
   - Valider que les calculs sont cohérents (long/short/cash)
   - Analyser l'impact des coûts sur le NAV

6. **Edge Cases**
   - Tester avec position flipping (long → short direct)
   - Valider le comportement avec cash insuffisant
   - Analyser les cas de bankruptcy (nav <= 0)

### Livrables
1. Rapport d'audit avec validation des calculs
2. Tests de position calculation (long/short/cash)
3. Tests de trade execution (position_changed logic)
4. Tests d'edge cases (position limits, bankruptcy)
5. Liste de findings (P0/P1/P2) avec recommandations
6. Code de validation reproductible
```

**Métriques de Succès**:
- ✅ Position calculation correcte (NAV = cash + positions * price)
- ✅ Trade execution optimisée (seulement si position_changed)
- ✅ Short selling fonctionne (cash augmente, funding appliqué)
- ✅ Position limits respectés (clamping, max_leverage)
- ✅ Edge cases gérés (position flipping, bankruptcy)

---

### Étape 2.3 : Audit Cost Model

**ID**: `P2.3`  
**Dépendances**: P2.2  
**Parallélisable avec**: P2.1, P2.2, P2.4  
**Score complexité**: 7 (cost modeling)

**Prompt Optimisé**:
```text
## Audit Cost Model

### Persona
Tu es un expert en modélisation de coûts de transaction avec expertise en commission, slippage, et market impact pour le trading haute fréquence.

### Contexte
- Commission: commission_rate * abs(delta_position) (linéaire)
- Slippage: slippage_rate * abs(delta_position) (linéaire)
- Funding: funding_rate * |position| * price (pour shorts uniquement)
- Domain Randomization: commission et slippage varient par env
- Fichier: `src/training/batch_env.py` (lignes 287-320, 689-726)

### Tâche
Auditer le modèle de coûts selon les standards SOTA:

1. **Commission Model**
   - Valider que commission=0.0006 (0.06%) est réaliste
   - Comparer avec les exchanges réels (Binance, Coinbase)
   - Analyser l'impact de la linéarité (vs tiered fees)

2. **Slippage Model**
   - Valider que slippage=0.0001 (0.01%) est réaliste
   - Analyser la linéarité (vs market impact non-linéaire)
   - Comparer avec la littérature (Almgren-Chriss, square-root law)

3. **Funding Rate**
   - Valider que funding_rate=0.0001 (0.01%/step) est réaliste
   - Vérifier que funding s'applique uniquement aux shorts
   - Analyser l'impact sur les stratégies long/short

4. **Domain Randomization**
   - Valider que la randomisation réduit l'overfitting
   - Analyser la distribution (Beta pour commission, Uniform pour slippage)
   - Tester l'impact sur la robustesse

5. **Slippage Noise**
   - Valider que slippage_noise_std=0.00002 capture la variabilité
   - Analyser l'impact sur le réalisme
   - Comparer avec les modèles de market impact

6. **Cost Realism**
   - Comparer avec les coûts réels observés
   - Identifier les simplifications et leur impact
   - Proposer des améliorations si nécessaire

### Livrables
1. Rapport d'audit avec validation du réalisme
2. Comparaison avec exchanges réels
3. Tests de domain randomization (robustesse)
4. Analyse de cost impact sur performance
5. Liste de findings (P0/P1/P2) avec recommandations
6. Code de validation reproductible
```

**Métriques de Succès**:
- ✅ Commission réaliste (0.06% aligné avec exchanges)
- ✅ Slippage réaliste (0.01% acceptable pour backtesting)
- ✅ Funding rate réaliste (0.01%/step ≈ 0.24%/day)
- ✅ Domain randomization réduit l'overfitting (>10% amélioration)
- ✅ Cost model documenté avec limitations

---

### Étape 2.4 : Audit Volatility Scaling

**ID**: `P2.4`  
**Dépendances**: P2.2  
**Parallélisable avec**: P2.1, P2.2, P2.3  
**Score complexité**: 6 (risk parity)

**Prompt Optimisé**:
```text
## Audit Volatility Scaling

### Persona
Tu es un quant researcher avec expertise en risk parity et volatility targeting pour le trading algorithmique.

### Contexte
- Target volatility: target_volatility=0.01 (1% par step)
- Volatility estimation: EMA variance (vol_window=24)
- Volatility scaling: vol_scalar = target_vol / current_vol
- Max leverage: max_leverage=5.0
- Volatility floor: min_vol = target_vol / max_leverage
- Fichier: `src/training/batch_env.py` (lignes 642-657, 738-739)

### Tâche
Auditer le volatility scaling selon les standards SOTA:

1. **Target Volatility**
   - Valider que target_volatility=0.01 est optimal
   - Comparer avec d'autres targets (0.005, 0.01, 0.02, 0.05)
   - Analyser l'impact sur le risk-adjusted return

2. **Volatility Estimation**
   - Valider que EMA variance est approprié (vs rolling std)
   - Vérifier que vol_window=24 est optimal
   - Analyser la réactivité (fast vs slow EMA)

3. **Scaling Formula**
   - Vérifier que vol_scalar = target_vol / current_vol
   - Valider que le clamping [0.1, max_leverage] est correct
   - Analyser l'impact sur la position sizing

4. **Volatility Floor**
   - Valider que min_vol = target_vol / max_leverage prévient le cash trap
   - Vérifier que le floor est appliqué correctement
   - Analyser l'impact sur les périodes de faible volatilité

5. **Max Leverage**
   - Valider que max_leverage=5.0 est approprié
   - Analyser l'impact sur le risque (VaR, CVaR)
   - Comparer avec les limites réglementaires

6. **Edge Cases**
   - Tester avec volatilité très faible (cash trap)
   - Valider le comportement avec volatilité très élevée
   - Analyser les cas de division par zéro (vol = 0)

### Livrables
1. Rapport d'audit avec validation théorique
2. Tests de calibration target_volatility
3. Analyse de volatility estimation (EMA vs rolling)
4. Tests d'edge cases (vol extremes, cash trap)
5. Liste de findings (P0/P1/P2) avec recommandations
6. Code de validation reproductible
```

**Métriques de Succès**:
- ✅ Target volatility optimal (test grid search)
- ✅ Volatility estimation réactive (EMA approprié)
- ✅ Scaling formula correcte (risk parity)
- ✅ Volatility floor prévient le cash trap
- ✅ Max leverage approprié (risque contrôlé)

---

### Batch 3 : Audit Robustness & Performance

---

### Étape 3.1 : Audit Numerical Stability

**ID**: `P3.1`  
**Dépendances**: P2.1, P2.4  
**Parallélisable avec**: P3.2, P3.3  
**Score complexité**: 7 (numerical safety)

**Prompt Optimisé**:
```text
## Audit Numerical Stability

### Persona
Tu es un ingénieur spécialisé en numerical stability pour deep learning, expert en issues float32/64, gradient explosion, et NaN debugging.

### Contexte
- Points critiques identifiés:
  - log1p(safe_returns) avec clamp à -0.99
  - Division par volatilité (vol scaling)
  - Division par old_navs (step_returns)
  - Multiplication par SCALE=100.0
- Fichier: `src/training/batch_env.py` (lignes 444-445, 649-656, 732)

### Tâche
Auditer la stabilité numérique selon les standards SOTA:

1. **Log Returns Safety**
   - Vérifier que clamp(-0.99) prévient log(0)
   - Analyser l'impact sur les returns extrêmes (flash crash)
   - Tester avec returns = -1.0 (edge case)

2. **Division by Zero**
   - Vérifier que old_navs > 0 (pas de division par zéro)
   - Valider que current_vol > 0 (volatility floor)
   - Analyser les cas de bankruptcy (nav = 0)

3. **Overflow/Underflow**
   - Vérifier que SCALE=100.0 ne cause pas d'overflow
   - Analyser l'impact de rewards extrêmes
   - Tester avec float32 vs float64

4. **NaN/Inf Detection**
   - Identifier toutes les opérations pouvant produire NaN/Inf
   - Valider que les protections sont en place
   - Tester avec données corrompues (NaN dans features)

5. **Gradient Stability**
   - Analyser l'impact sur les gradients (explosion/collapse)
   - Vérifier que les rewards sont dans une plage stable
   - Tester avec gradient clipping

6. **Edge Cases**
   - Tester avec returns extrêmes (±50%)
   - Valider le comportement avec volatilité = 0
   - Analyser les cas de bankruptcy immédiat

### Livrables
1. Rapport d'audit avec tests de stabilité
2. Tests de division par zéro (tous les cas)
3. Tests de NaN/Inf (données corrompues)
4. Tests d'overflow/underflow (rewards extrêmes)
5. Liste de findings (P0/P1/P2) avec recommandations
6. Code de validation reproductible
```

**Métriques de Succès**:
- ✅ Pas de division par zéro (tous les cas testés)
- ✅ Log returns safe (clamp prévient log(0))
- ✅ Pas de NaN/Inf (données corrompues gérées)
- ✅ Overflow/underflow contrôlés (rewards stables)
- ✅ Gradient stability validée

---

### Étape 3.2 : Audit GPU Vectorization

**ID**: `P3.2`  
**Dépendances**: P2.2  
**Parallélisable avec**: P3.1, P3.3  
**Score complexité**: 6 (performance optimization)

**Prompt Optimisé**:
```text
## Audit GPU Vectorization

### Persona
Tu es un ingénieur performance avec expertise en GPU programming (CUDA, PyTorch) et vectorization pour RL.

### Contexte
- Architecture: GPU-vectorized batch environment
- n_envs: 512-1024 environnements parallèles
- Performance: 2k → 50k steps/s (vs SubprocVecEnv)
- Fichier: `src/training/batch_env.py` (lignes 223-285, 616-795)

### Tâche
Auditer la vectorization GPU selon les standards SOTA:

1. **Tensor Operations**
   - Vérifier que toutes les opérations sont vectorisées (pas de loops)
   - Analyser l'utilisation de torch.where() vs conditionals
   - Valider que les opérations sont sur GPU (device)

2. **Memory Management**
   - Vérifier que les tensors sont pré-alloués (pas de création à chaque step)
   - Analyser l'utilisation mémoire (n_envs × tensor_size)
   - Valider que les tensors sont contigus (contiguous())

3. **Data Transfer**
   - Vérifier que CPU ↔ GPU transfers sont minimisés
   - Analyser l'impact de .cpu().numpy() dans _get_observations()
   - Optimiser si nécessaire (async transfers)

4. **Batch Operations**
   - Valider que les opérations batch sont efficaces
   - Analyser l'utilisation de broadcasting
   - Vérifier que les opérations sont parallélisables

5. **Performance Profiling**
   - Profiler les opérations critiques (step_wait, _calculate_rewards)
   - Identifier les bottlenecks
   - Proposer des optimisations

6. **Scalability**
   - Tester avec différents n_envs (128, 512, 1024, 2048)
   - Analyser l'impact sur la performance (throughput)
   - Valider que la scalabilité est linéaire

### Livrables
1. Rapport d'audit avec profiling
2. Tests de performance (throughput vs n_envs)
3. Analyse de memory usage
4. Identification des bottlenecks
5. Liste de findings (P0/P1/P2) avec recommandations
6. Code de validation reproductible
```

**Métriques de Succès**:
- ✅ Toutes les opérations vectorisées (pas de loops)
- ✅ Memory management optimal (pré-allocation)
- ✅ CPU ↔ GPU transfers minimisés
- ✅ Performance > 10k steps/s (n_envs=512)
- ✅ Scalabilité linéaire (n_envs jusqu'à 2048)

---

### Étape 3.3 : Audit Domain Randomization

**ID**: `P3.3`  
**Dépendances**: P2.3  
**Parallélisable avec**: P3.1, P3.2  
**Score complexité**: 5 (regularization)

**Prompt Optimisé**:
```text
## Audit Domain Randomization

### Persona
Tu es un expert en domain randomization et sim-to-real transfer pour RL, avec expertise en anti-overfitting.

### Contexte
- Domain Randomization: commission et slippage varient par env
- Commission: Beta distribution [commission_min, commission_max]
- Slippage: Uniform distribution [slippage_min, slippage_max]
- Slippage noise: Normal(0, slippage_noise_std) additif
- Sampling: per-episode (pas per-step)
- Fichier: `src/training/batch_env.py` (lignes 287-320, 531-533, 697-706)

### Tâche
Auditer le domain randomization selon les standards SOTA:

1. **Randomization Strategy**
   - Valider que per-episode sampling est approprié (vs per-step)
   - Analyser l'impact sur le réalisme (exchange behavior)
   - Comparer avec d'autres stratégies (curriculum, adaptive)

2. **Distribution Selection**
   - Valider que Beta pour commission est approprié (skewed center)
   - Vérifier que Uniform pour slippage est optimal
   - Analyser l'impact sur la diversité

3. **Range Calibration**
   - Valider que [0.02%, 0.08%] pour commission est réaliste
   - Vérifier que [0.005%, 0.015%] pour slippage est approprié
   - Analyser l'impact sur la robustesse

4. **Slippage Noise**
   - Valider que slippage_noise_std=0.00002 capture la variabilité
   - Analyser l'impact sur le réalisme (market impact)
   - Comparer avec les modèles de market impact

5. **Anti-Overfitting Effectiveness**
   - Tester avec/sans domain randomization
   - Analyser l'impact sur la généralisation (train vs test)
   - Valider que l'overfitting est réduit (>10% amélioration)

6. **Training vs Evaluation**
   - Vérifier que randomization est désactivé en eval (training flag)
   - Valider la reproductibilité en mode eval
   - Analyser l'impact sur les métriques

### Livrables
1. Rapport d'audit avec validation de l'efficacité
2. Tests avec/sans domain randomization (généralisation)
3. Analyse de distribution (diversité)
4. Tests de calibration (ranges)
5. Liste de findings (P0/P1/P2) avec recommandations
6. Code de validation reproductible
```

**Métriques de Succès**:
- ✅ Domain randomization réduit l'overfitting (>10% amélioration)
- ✅ Distributions appropriées (Beta, Uniform)
- ✅ Ranges réalistes (alignés avec exchanges)
- ✅ Randomization désactivé en eval (reproductibilité)
- ✅ Per-episode sampling approprié

---

### Batch 4 : Audit Data & Integration

---

### Étape 4.1 : Audit Data Handling

**ID**: `P4.1`  
**Dépendances**: P1.1, P3.2  
**Parallélisable avec**: P4.2  
**Score complexité**: 6 (data pipeline)

**Prompt Optimisé**:
```text
## Audit Data Handling

### Persona
Tu es un expert en data pipelines pour RL avec expertise en windowing, feature extraction, et data leakage.

### Contexte
- Data loading: pd.read_parquet() → torch.tensor
- Window stacking: _get_batch_windows() avec window_size=64
- Feature extraction: exclude EXCLUDE_COLS, handle NaN
- Data slicing: start_idx/end_idx pour train/val split
- Fichier: `src/training/batch_env.py` (lignes 149-179, 561-600, 830-870)

### Tâche
Auditer le data handling selon les standards SOTA:

1. **Data Loading**
   - Vérifier que le chargement est efficace (parquet vs CSV)
   - Valider que les données sont bien transférées sur GPU
   - Analyser l'impact mémoire (n_steps × n_features)

2. **Window Stacking**
   - Vérifier que _get_batch_windows() est optimisé (pas de loops)
   - Valider que les windows sont correctes (pas de look-ahead)
   - Analyser l'impact sur la performance (pre-allocated offsets)

3. **Feature Extraction**
   - Vérifier que EXCLUDE_COLS exclut bien les colonnes non-numériques
   - Valider que NaN sont bien gérés (nan_to_num)
   - Analyser l'impact sur la qualité des features

4. **Data Slicing**
   - Vérifier que start_idx/end_idx fonctionnent correctement
   - Valider que les slices sont cohérents (train/val/test)
   - Analyser l'impact sur la reproductibilité

5. **Data Leakage**
   - Vérifier qu'il n'y a pas de look-ahead bias
   - Valider que les windows utilisent seulement les données passées
   - Tester avec un oracle (future data) pour détecter le leakage

6. **Edge Cases**
   - Tester avec données courtes (< window_size)
   - Valider le comportement avec données manquantes
   - Analyser les cas de données corrompues

### Livrables
1. Rapport d'audit avec validation du data handling
2. Tests de data leakage (oracle test)
3. Tests de window stacking (correctness)
4. Tests d'edge cases (données courtes, manquantes)
5. Liste de findings (P0/P1/P2) avec recommandations
6. Code de validation reproductible
```

**Métriques de Succès**:
- ✅ Data loading efficace (parquet, GPU transfer)
- ✅ Window stacking optimisé (pas de loops)
- ✅ Pas de data leakage (tests oracle passés)
- ✅ Feature extraction correcte (NaN gérés)
- ✅ Data slicing cohérent (train/val/test)

---

### Étape 4.2 : Audit Observation Noise

**ID**: `P4.2`  
**Dépendances**: P1.1, P3.3  
**Parallélisable avec**: P4.1  
**Score complexité**: 6 (regularization)

**Prompt Optimisé**:
```text
## Audit Observation Noise

### Persona
Tu es un expert en regularization pour RL avec expertise en observation noise et anti-overfitting (NoisyRollout 2025).

### Contexte
- Observation Noise: Dynamic (Annealing + Volatility-Adaptive)
- Annealing: 1.0 → 0.5 (progress-based)
- Volatility-Adaptive: vol_factor = target_vol / current_vol (clamped [0.5, 2.0])
- Combined: final_scale = observation_noise * annealing_factor * vol_factor
- Training flag: désactivé en eval
- Fichier: `src/training/batch_env.py` (lignes 571-590, 602-604)

### Tâche
Auditer l'observation noise selon les standards SOTA:

1. **Noise Strategy**
   - Valider que l'annealing est approprié (1.0 → 0.5)
   - Vérifier que le vol-adaptive est innovant (CryptoRL)
   - Analyser l'impact sur la robustesse

2. **Volatility-Adaptive**
   - Valider que vol_factor = target_vol / current_vol est correct
   - Vérifier que le clamping [0.5, 2.0] prévient l'explosion
   - Analyser l'impact sur les régimes de marché

3. **Annealing Schedule**
   - Valider que progress-based annealing est optimal
   - Comparer avec d'autres schedules (linear, exponential, cosine)
   - Analyser l'impact sur l'apprentissage

4. **Anti-Overfitting Effectiveness**
   - Tester avec/sans observation noise
   - Analyser l'impact sur la généralisation (train vs test)
   - Valider que l'overfitting est réduit (>10% amélioration)

5. **Training vs Evaluation**
   - Vérifier que noise est désactivé en eval (training flag)
   - Valider la reproductibilité en mode eval
   - Analyser l'impact sur les métriques

6. **Noise Level Calibration**
   - Valider que observation_noise est calibré (typiquement 0.01-0.05)
   - Tester différents niveaux (0.0, 0.01, 0.05, 0.1)
   - Analyser l'impact sur la performance

### Livrables
1. Rapport d'audit avec validation de l'efficacité
2. Tests avec/sans observation noise (généralisation)
3. Analyse de calibration (noise level)
4. Tests de vol-adaptive (régimes de marché)
5. Liste de findings (P0/P1/P2) avec recommandations
6. Code de validation reproductible
```

**Métriques de Succès**:
- ✅ Observation noise réduit l'overfitting (>10% amélioration)
- ✅ Volatility-adaptive innovant (CryptoRL)
- ✅ Annealing approprié (1.0 → 0.5)
- ✅ Noise désactivé en eval (reproductibilité)
- ✅ Noise level calibré (0.01-0.05 optimal)

---

### Batch 5 : Synthèse & Recommandations

---

### Étape 5 : Synthèse & Recommandations

**ID**: `P5`  
**Dépendances**: P1, P2, P3, P4  
**Score complexité**: 8 (synthesis + prioritization)

**Prompt Optimisé**:
```text
## Synthèse & Recommandations - Audit Environnement SOTA

### Persona
Tu es un architecte technique senior avec expertise en synthèse d'audits, priorisation, et roadmap planning.

### Contexte
- Audits complétés: P1.1-P1.4, P2.1-P2.4, P3.1-P3.3, P4.1-P4.2
- Findings collectés: P0 (critiques), P1 (importants), P2 (mineurs)
- Objectif: Synthèse, priorisation, roadmap

### Tâche
Synthétiser tous les audits et produire un rapport final:

1. **Executive Summary**
   - Score global Environnement (0-10)
   - Top 5 findings critiques
   - Recommandations prioritaires
   - Impact estimé des corrections

2. **Findings Aggregation**
   - Regrouper les findings par catégorie:
     - Architecture & Design
     - Trading Mechanics
     - Robustness & Performance
     - Data & Integration
   - Prioriser (P0 > P1 > P2)
   - Estimer l'effort de correction

3. **Risk Matrix**
   - Probabilité vs Impact pour chaque finding
   - Identifier les risques critiques
   - Proposer un plan de mitigation

4. **Roadmap de Correction**
   - Phase 1: P0 (bloquants) - 1-2 semaines
   - Phase 2: P1 (importants) - 2-4 semaines
   - Phase 3: P2 (améliorations) - 4-8 semaines
   - Dépendances entre corrections

5. **Métriques de Succès**
   - Définir des KPIs pour valider les corrections
   - Implémenter des tests de régression
   - Valider que les corrections améliorent la qualité

6. **Comparaison SOTA**
   - Comparer avec les implémentations SOTA (OpenAI Gym, FinRL)
   - Identifier les gaps
   - Proposer des améliorations futures

### Livrables
1. Rapport de synthèse complet (Executive Summary)
2. Risk matrix avec priorisation
3. Roadmap de correction (phases, dépendances)
4. Métriques de succès (KPIs)
5. Comparaison SOTA (gaps, améliorations)
6. Code de validation (tests de régression)
```

**Métriques de Succès**:
- ✅ Score global Environnement > 8/10
- ✅ Tous les P0 corrigés (0 findings critiques)
- ✅ Roadmap claire avec dépendances
- ✅ Métriques de succès définies et mesurables

---

## 📊 Matrice de Risque (Template)

| ID | Finding | Prob | Impact | Priority | Effort | Status |
|----|---------|------|--------|----------|--------|--------|
| P1.1-X | Observation space missing feature X | H/M/L | H/M/L | P0/P1/P2 | S/M/L | ⏳/✅/❌ |
| ... | ... | ... | ... | ... | ... | ... |

**Légende**:
- **Prob**: Probabilité (H=High, M=Medium, L=Low)
- **Impact**: Impact sur la qualité (H=High, M=Medium, L=Low)
- **Priority**: P0 (Bloquant), P1 (Important), P2 (Amélioration)
- **Effort**: S (Small <1j), M (Medium 1-3j), L (Large >3j)
- **Status**: ⏳ (À faire), ✅ (Fait), ❌ (Rejeté)

---

## 🎯 Métriques de Succès Globales

1. **Score Global Environnement**: > 8/10
2. **Findings P0**: 0 (tous corrigés)
3. **Data Leakage**: 0 détecté (tests oracle passés)
4. **Numerical Stability**: 100% (pas de NaN/Inf)
5. **Performance**: > 10k steps/s (n_envs=512)
6. **Reward Calibration**: r_perf ≈ r_cost * MAX_PENALTY_SCALE
7. **MORL Implementation**: Conforme Abels 2019
8. **GPU Vectorization**: Toutes opérations vectorisées

---

## 📚 Références SOTA

1. **Abels et al. (2019)**: "Dynamic Weights in Multi-Objective Deep RL"
2. **Lopez de Prado (2018)**: "Advances in Financial Machine Learning"
3. **Hayes et al. (2022)**: "MORL Guide - Best Practices"
4. **Gymnasium**: "VecEnv Interface Documentation"
5. **NoisyRollout (2025)**: "Observation Noise for RL Robustness"
6. **Almgren & Chriss (2000)**: "Optimal Execution of Portfolio Transactions"

---

## ✅ Checklist d'Exécution

### Phase 1: Architecture & Design
- [ ] P1.1: Audit Observation Space
- [ ] P1.2: Audit Action Space & Discretization
- [ ] P1.3: Audit MORL Implementation
- [ ] P1.4: Audit Episode Management

### Phase 2: Trading Mechanics
- [ ] P2.1: Audit Reward Function
- [ ] P2.2: Audit Position Management
- [ ] P2.3: Audit Cost Model
- [ ] P2.4: Audit Volatility Scaling

### Phase 3: Robustness & Performance
- [ ] P3.1: Audit Numerical Stability
- [ ] P3.2: Audit GPU Vectorization
- [ ] P3.3: Audit Domain Randomization

### Phase 4: Data & Integration
- [ ] P4.1: Audit Data Handling
- [ ] P4.2: Audit Observation Noise

### Phase 5: Synthèse
- [ ] P5: Synthèse & Recommandations

---

**Date de création**: 2026-01-23  
**Dernière mise à jour**: 2026-01-23  
**Version**: 1.0

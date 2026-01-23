# Master Plan: Audit HMM (Hidden Markov Model) - CryptoRL

**Date**: 2026-01-22  
**Méthode**: State-of-the-Art Audit Framework  
**Objectif**: Audit exhaustif et critique du système HMM pour détection de régimes de marché  
**Référence**: Hamilton (1989), Rabiner (1989), fHMM (R), Shu et al. (2024)

---

## 📋 Méta-Informations

- **Complexité totale estimée**: 42 points
- **Nombre de prompts atomiques**: 15
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
| Les contraintes techniques sont-elles explicites ? | Python 3.10+, hmmlearn, sklearn, données horaires BTC, WFO compatible | ✅ |
| Le scope est-il borné ? | HMM uniquement (RegimeDetector) - features, training, prediction, alignment | ✅ |

**Scope IN**:
- `RegimeDetector` class (`src/data_engineering/manager.py`)
- HMM features engineering (HMM_Trend, HMM_Vol, HMM_Momentum, HMM_RiskOnOff, HMM_VolRatio)
- GMM-HMM configuration (n_components=4, n_mix=2, transition_penalty)
- K-Means warm start initialization
- Archetype Alignment (Hungarian Algorithm)
- Quality validation logic
- WFO integration (fit on train, predict on test)
- TensorBoard logging

**Scope OUT**:
- Feature engineering général (FFD, Z-Score) → déjà audité
- Data pipeline orchestration → déjà audité
- RL agent utilisant Prob_* features → scope modèles RL

---

## 🌳 Arbre de Décomposition

```
Root: "Audit HMM SOTA"
│
├─→ P1: Audit Architecture & Design (parallèle)
│   ├─‖ P1.1: Audit Features Engineering (ATOMIC)
│   ├─‖ P1.2: Audit GMM-HMM Configuration (ATOMIC)
│   ├─‖ P1.3: Audit Initialization Strategy (ATOMIC)
│   └─‖ P1.4: Audit Archetype Alignment (ATOMIC)
│
├─→ P2: Audit Validation & Quality (parallèle, dépend P1)
│   ├─‖ P2.1: Audit Quality Metrics (ATOMIC)
│   ├─‖ P2.2: Audit Model Selection (ATOMIC)
│   ├─‖ P2.3: Audit Convergence & Stability (ATOMIC)
│   └─‖ P2.4: Audit Retry Logic (ATOMIC)
│
├─→ P3: Audit Data Leakage & WFO (parallèle, dépend P2)
│   ├─‖ P3.1: Audit WFO Integration (ATOMIC)
│   ├─‖ P3.2: Audit Context Buffer (ATOMIC)
│   └─‖ P3.3: Audit Feature Lookback Windows (ATOMIC)
│
├─→ P4: Audit Robustness & Reproducibility (parallèle, dépend P3)
│   ├─‖ P4.1: Audit Reproducibility (ATOMIC)
│   └─‖ P4.2: Audit Edge Cases & Numerical Stability (ATOMIC)
│
└─→ P5: Synthèse & Recommandations (ATOMIC, dépend P4)
```

**Légende**: → séquentiel | ‖ parallèle

---

## 📝 Prompts Exécutables

---

### Batch 1 : Audit Architecture & Design

---

### Étape 1.1 : Audit Features Engineering

**ID**: `P1.1`  
**Dépendances**: Aucune  
**Parallélisable avec**: P1.2, P1.3, P1.4  
**Score complexité**: 7 (domaine spécialisé + validation théorique)

**Prompt Optimisé**:
```text
## Audit HMM Features Engineering

### Persona
Tu es un quant researcher senior avec 10+ ans d'expérience en modélisation de régimes de marché. Tu as publié sur HMM pour time series financières et connais les pièges classiques (look-ahead bias, stationnarité, multicollinéarité).

### Contexte
- Projet: CryptoRL - Trading RL pour cryptomonnaies
- HMM Features: HMM_Trend, HMM_Vol, HMM_Momentum, HMM_RiskOnOff, HMM_VolRatio
- Fichier: `src/data_engineering/manager.py` (lignes 122-194)
- Window: 168h (7 jours) pour smoothing
- Data: BTC hourly, multi-asset (SPX, DXY)

### Tâche
Auditer la qualité et la validité des features HMM selon les standards SOTA:

1. **Stationnarité & Look-Ahead Bias**
   - Vérifier que les rolling windows n'utilisent pas de données futures
   - Valider que les features sont stationnaires (ADF test, KPSS test)
   - Vérifier l'absence de leakage temporel

2. **Feature Engineering Quality**
   - HMM_Trend: MA(LogRet, 168h) - valider la pertinence de la fenêtre
   - HMM_Vol: MA(Parkinson, 168h) - vérifier la cohérence avec la volatilité réelle
   - HMM_Momentum: RSI(14) / 100 - valider la normalisation et les bornes
   - HMM_RiskOnOff: MA(SPX_ret - DXY_ret, 168h) - vérifier la corrélation avec BTC
   - HMM_VolRatio: Vol(24h) / Vol(168h) - valider le ratio comme early warning

3. **Clipping & Numerical Stability**
   - Vérifier que les clips sont justifiés théoriquement
   - Valider que les bornes ne tronquent pas trop de signal
   - Tester la stabilité numérique (NaN, Inf)

4. **Multicollinéarité**
   - Calculer la matrice de corrélation entre features
   - Identifier les features redondantes (VIF > 5)
   - Recommander des features alternatives si nécessaire

5. **Feature Selection vs Domain Knowledge**
   - Comparer avec la littérature (Hamilton 1989, Ang & Timmermann 2002)
   - Valider que HMM_Funding a bien été retiré (audit P1.2)
   - Proposer des features additionnelles SOTA si pertinentes

### Livrables
1. Rapport d'audit avec scores par feature (0-10)
2. Tests de stationnarité (ADF, KPSS) avec résultats
3. Matrice de corrélation et VIF
4. Liste de findings (P0/P1/P2) avec recommandations
5. Code de validation reproductible
```

**Métriques de Succès**:
- ✅ Toutes les features sont stationnaires (p-value < 0.05 ADF)
- ✅ Aucun look-ahead bias détecté
- ✅ VIF < 5 pour toutes les features
- ✅ Clipping justifié théoriquement
- ✅ Features alignées avec la littérature SOTA

---

### Étape 1.2 : Audit GMM-HMM Configuration

**ID**: `P1.2`  
**Dépendances**: Aucune  
**Parallélisable avec**: P1.1, P1.3, P1.4  
**Score complexité**: 8 (théorie HMM + hyperparamètres)

**Prompt Optimisé**:
```text
## Audit GMM-HMM Configuration

### Persona
Tu es un chercheur en statistique computationnelle avec expertise en HMM et GMM. Tu connais les pièges classiques (overfitting, underfitting, identifiability, label switching).

### Contexte
- Modèle: GMMHMM (hmmlearn)
- Configuration actuelle:
  - n_components: 4 (Crash, Downtrend, Range, Uptrend)
  - n_mix: 2 (mixture components)
  - covariance_type: 'diag'
  - n_iter: 200
  - min_covar: 1e-3
  - transition_penalty: 0.1 (Sticky HMM)
- Fichier: `src/data_engineering/manager.py` (lignes 504-512, 262-297)

### Tâche
Auditer la configuration GMM-HMM selon les standards SOTA:

1. **Model Selection (n_components, n_mix)**
   - Valider que n_components=4 est optimal (AIC, BIC, ICL)
   - Tester n_components ∈ {3, 4, 5, 6} avec information criteria
   - Valider que n_mix=2 est suffisant (pas d'overfitting)
   - Comparer avec la littérature (4 régimes = standard en finance)

2. **Covariance Structure**
   - Valider 'diag' vs 'full' vs 'tied' vs 'spherical'
   - Vérifier la stabilité numérique (condition number)
   - Tester si 'full' améliore la qualité sans overfitting

3. **EM Algorithm Configuration**
   - Valider n_iter=200 (convergence garantie?)
   - Vérifier le tolerance (monitor_.tol)
   - Analyser l'historique de convergence (log-likelihood)
   - Détecter les cas de non-convergence

4. **Regularization (min_covar)**
   - Valider min_covar=1e-3 (pas trop restrictif?)
   - Tester min_covar ∈ {1e-4, 1e-3, 1e-2}
   - Vérifier que la régularisation n'écrase pas le signal

5. **Sticky HMM (Transition Penalty)**
   - Valider la formule: A_sticky = A × (1-p) + I × p
   - Vérifier que penalty=0.1 est optimal (test grid search)
   - Analyser l'impact sur la durée moyenne des régimes
   - Comparer avec la littérature (Shu et al. 2024)

6. **Identifiability & Label Switching**
   - Vérifier que le modèle est identifiable (pas de modes dégénérés)
   - Valider que l'Archetype Alignment résout le label switching
   - Tester la stabilité du mapping entre runs

### Livrables
1. Rapport d'audit avec scores par hyperparamètre
2. Courbes AIC/BIC/ICL pour différents n_components
3. Analyse de convergence (log-likelihood history)
4. Grid search results pour transition_penalty
5. Liste de findings (P0/P1/P2) avec recommandations
6. Code de validation reproductible
```

**Métriques de Succès**:
- ✅ n_components=4 optimal selon AIC/BIC
- ✅ Convergence EM garantie (monitor_.converged > 95%)
- ✅ min_covar optimal (pas de singularité, pas d'over-regularization)
- ✅ transition_penalty=0.1 optimal (grid search)
- ✅ Modèle identifiable (pas de modes dégénérés)

---

### Étape 1.3 : Audit Initialization Strategy

**ID**: `P1.3`  
**Dépendances**: Aucune  
**Parallélisable avec**: P1.1, P1.2, P1.4  
**Score complexité**: 6 (initialization heuristics)

**Prompt Optimisé**:
```text
## Audit HMM Initialization Strategy

### Persona
Tu es un expert en optimisation non-convexe avec expertise en initialization heuristics pour modèles génératifs (HMM, GMM, VAE).

### Contexte
- Stratégie: K-Means warm start
- Fichier: `src/data_engineering/manager.py` (lignes 196-245, 489-521)
- Flow: K-Means → inject centers → add noise → fit HMM
- Random state: self.random_state + attempt * 17 (retry logic)

### Tâche
Auditer la stratégie d'initialisation selon les standards SOTA:

1. **K-Means Warm Start**
   - Valider que K-Means améliore la convergence vs random init
   - Vérifier que n_init=10 est suffisant pour K-Means
   - Tester si K-Means++ est meilleur que K-Means standard
   - Analyser l'inertia K-Means (qualité des clusters)

2. **Noise Injection**
   - Valider que noise ~ N(0, 0.1) est optimal
   - Vérifier que le noise différencie bien les mixture components
   - Tester différents niveaux de noise (0.05, 0.1, 0.2)
   - Analyser l'impact sur la convergence

3. **Reproducibility Issues**
   - Identifier les sources de non-reproductibilité:
     - K-Means random_state changeant entre retries
     - HMM random_state changeant entre retries
     - Noise injection avec np.random.seed()
   - Proposer une solution déterministe

4. **Alternative Initialization Methods**
   - Comparer avec d'autres méthodes SOTA:
     - Spectral initialization
     - Moment matching
     - Variational Bayes initialization
   - Évaluer si une méthode alternative est meilleure

5. **Retry Logic Impact**
   - Analyser si les retries améliorent vraiment la qualité
   - Vérifier que le best selection (max active states) est optimal
   - Proposer une métrique de qualité secondaire (separation_score)

### Livrables
1. Rapport d'audit avec comparaison init methods
2. Tests de reproductibilité (même seed → même résultat)
3. Analyse de convergence avec différentes initializations
4. Liste de findings (P0/P1/P2) avec recommandations
5. Code de validation reproductible
```

**Métriques de Succès**:
- ✅ K-Means warm start améliore la convergence (log-likelihood initiale +20%)
- ✅ Initialization 100% reproductible (même seed → même résultat)
- ✅ Noise injection optimal (test grid search)
- ✅ Retry logic justifié (améliore la qualité dans >80% des cas)

---

### Étape 1.4 : Audit Archetype Alignment

**ID**: `P1.4`  
**Dépendances**: Aucune  
**Parallélisable avec**: P1.1, P1.2, P1.3  
**Score complexité**: 7 (optimisation combinatoire + validation métier)

**Prompt Optimisé**:
```text
## Audit Archetype Alignment (Hungarian Algorithm)

### Persona
Tu es un expert en optimisation combinatoire avec expertise en assignment problems et validation de modèles pour la finance.

### Contexte
- Méthode: Hungarian Algorithm (scipy.optimize.linear_sum_assignment)
- Archétypes fixes:
  - State 0: Crash (-0.50%/h, 4.0%/h vol)
  - State 1: Downtrend (-0.10%/h, 1.5%/h vol)
  - State 2: Range (0.00%/h, 0.5%/h vol)
  - State 3: Uptrend (+0.15%/h, 2.0%/h vol)
- Distance: Euclidienne pondérée (w_ret=1.0, w_vol=2.0)
- Fichier: `src/data_engineering/manager.py` (lignes 299-362)

### Tâche
Auditer l'Archetype Alignment selon les standards SOTA:

1. **Archetype Calibration**
   - Valider que les archétypes sont réalistes pour BTC hourly
   - Comparer avec les régimes observés historiquement (2020-2024)
   - Vérifier que les archétypes couvrent bien l'espace des régimes
   - Tester si des archétypes additionnels sont nécessaires

2. **Distance Metric**
   - Valider la distance euclidienne pondérée
   - Vérifier que w_vol=2.0 est optimal (vol plus discriminante)
   - Comparer avec d'autres métriques (Mahalanobis, cosine)
   - Tester si la normalisation (z-scores) est correcte

3. **Hungarian Algorithm Correctness**
   - Vérifier que l'algorithme trouve bien l'optimal global
   - Valider que le mapping est unique (pas d'ambiguïté)
   - Tester les cas edge (états très proches, archétypes mal calibrés)

4. **Semantic Drift Resolution**
   - Valider que l'alignment résout vraiment le semantic drift
   - Tester la stabilité du mapping entre segments WFO
   - Vérifier que Prob_0 signifie toujours "Crash" entre segments
   - Analyser les cas où l'alignment échoue

5. **Alternative Alignment Methods**
   - Comparer avec d'autres méthodes:
     - Maximum likelihood alignment
     - Wasserstein distance
     - Procrustes analysis
   - Évaluer si une méthode alternative est meilleure

6. **Validation Métier**
   - Vérifier que les régimes alignés ont du sens économiquement
   - Analyser la cohérence avec les événements de marché (crashes, bull runs)
   - Valider que les transitions entre régimes sont réalistes

### Livrables
1. Rapport d'audit avec validation des archétypes
2. Tests de stabilité du mapping entre segments WFO
3. Comparaison avec méthodes alternatives
4. Analyse de cohérence métier (événements de marché)
5. Liste de findings (P0/P1/P2) avec recommandations
6. Code de validation reproductible
```

**Métriques de Succès**:
- ✅ Archétypes réalistes (match avec régimes historiques >80%)
- ✅ Mapping stable entre segments WFO (>95% de cohérence)
- ✅ Alignment résout le semantic drift (Prob_0 = Crash toujours)
- ✅ Distance metric optimale (test grid search)

---

### Batch 2 : Audit Validation & Quality

---

### Étape 2.1 : Audit Quality Metrics

**ID**: `P2.1`  
**Dépendances**: P1.1, P1.2  
**Parallélisable avec**: P2.2, P2.3, P2.4  
**Score complexité**: 6 (métriques de qualité)

**Prompt Optimisé**:
```text
## Audit HMM Quality Metrics

### Persona
Tu es un expert en validation de modèles génératifs avec expertise en métriques de qualité pour HMM (separation, persistence, interpretability).

### Contexte
- Métriques actuelles:
  - n_active_states: nombre d'états avec proportion > 5%
  - state_proportions: distribution des états
  - state_mean_returns: mean return par état
  - separation_score: std(mean_returns)
  - is_valid: n_active_states >= 3
- Fichier: `src/data_engineering/manager.py` (lignes 364-419)

### Tâche
Auditer les métriques de qualité selon les standards SOTA:

1. **Completeness of Metrics**
   - Identifier les métriques manquantes SOTA:
     - Regime persistence (durée moyenne des régimes)
     - Transition matrix entropy
     - State separation (distance entre états)
     - Predictive power (corrélation Prob_* avec future returns)
     - Calibration (Prob_* bien calibrées?)
   - Proposer une suite complète de métriques

2. **Validation Thresholds**
   - Valider min_proportion=0.05 (5% minimum par état)
   - Vérifier que is_valid (n_active >= 3) est suffisant
   - Tester différents seuils et leur impact

3. **Separation Score**
   - Valider que std(mean_returns) est une bonne métrique
   - Comparer avec d'autres métriques de séparation:
     - Silhouette score
     - Davies-Bouldin index
     - Distance inter-états (Mahalanobis)
   - Proposer une métrique composite

4. **Predictive Power Validation**
   - Tester si Prob_* prédit les future returns (1h, 24h, 168h ahead)
   - Calculer l'information mutuelle entre Prob_* et future returns
   - Valider que le HMM capture bien les régimes prédictifs

5. **Calibration Analysis**
   - Vérifier que Prob_* sont bien calibrées (reliability diagram)
   - Tester si Prob_0 élevé correspond vraiment à des crashes
   - Analyser les cas de mauvais calibrage

### Livrables
1. Rapport d'audit avec métriques SOTA complètes
2. Tests de predictive power (corrélation avec future returns)
3. Reliability diagrams (calibration)
4. Liste de findings (P0/P1/P2) avec recommandations
5. Code de validation reproductible
```

**Métriques de Succès**:
- ✅ Suite complète de métriques SOTA implémentée
- ✅ Predictive power validé (corrélation Prob_* vs future returns > 0.1)
- ✅ Calibration validée (reliability diagram proche de la diagonale)
- ✅ Separation score optimal (états bien distincts)

---

### Étape 2.2 : Audit Model Selection

**ID**: `P2.2`  
**Dépendances**: P1.2  
**Parallélisable avec**: P2.1, P2.3, P2.4  
**Score complexité**: 7 (model selection theory)

**Prompt Optimisé**:
```text
## Audit HMM Model Selection

### Persona
Tu es un statisticien avec expertise en model selection (AIC, BIC, cross-validation) et overfitting detection.

### Contexte
- Sélection actuelle: n_components=4 fixe (domain knowledge)
- Pas de model selection automatique
- Fichier: `src/data_engineering/manager.py`

### Tâche
Auditer la stratégie de model selection selon les standards SOTA:

1. **Information Criteria**
   - Implémenter AIC, BIC, ICL pour HMM
   - Tester n_components ∈ {2, 3, 4, 5, 6, 7, 8}
   - Valider que n_components=4 est optimal selon les critères
   - Comparer avec la littérature (4 régimes = standard)

2. **Cross-Validation**
   - Implémenter time series cross-validation (pas de shuffle)
   - Tester différents n_components avec CV
   - Valider que n_components=4 minimise l'erreur de prédiction

3. **Overfitting Detection**
   - Analyser si n_mix=2 est suffisant (pas d'overfitting)
   - Tester n_mix ∈ {1, 2, 3, 4}
   - Valider avec train/validation split

4. **Robustness Testing**
   - Tester la stabilité du modèle sur différents segments temporels
   - Valider que n_components=4 est robuste (pas de drift)
   - Analyser les cas où le modèle dégénère

5. **Alternative Model Selection**
   - Comparer avec d'autres méthodes:
     - Variational Bayes (automatic model selection)
     - Reversible Jump MCMC
     - Non-parametric HMM
   - Évaluer si une méthode alternative est meilleure

### Livrables
1. Rapport d'audit avec courbes AIC/BIC/ICL
2. Résultats de cross-validation
3. Tests d'overfitting (train/validation)
4. Liste de findings (P0/P1/P2) avec recommandations
5. Code de validation reproductible
```

**Métriques de Succès**:
- ✅ n_components=4 optimal selon AIC/BIC/ICL
- ✅ Cross-validation confirme n_components=4
- ✅ Pas d'overfitting détecté (n_mix=2 suffisant)
- ✅ Modèle robuste sur différents segments (>90% de stabilité)

---

### Étape 2.3 : Audit Convergence & Stability

**ID**: `P2.3`  
**Dépendances**: P1.2  
**Parallélisable avec**: P2.1, P2.2, P2.4  
**Score complexité**: 6 (convergence analysis)

**Prompt Optimisé**:
```text
## Audit HMM Convergence & Stability

### Persona
Tu es un expert en algorithmes EM avec expertise en analyse de convergence et détection de modes locaux.

### Contexte
- Algorithme: EM (Expectation-Maximization)
- Configuration: n_iter=200, monitor_.tol (default)
- Fichier: `src/data_engineering/manager.py` (lignes 523-526, 561)

### Tâche
Auditer la convergence et la stabilité selon les standards SOTA:

1. **Convergence Analysis**
   - Analyser l'historique de log-likelihood (monitor_.history)
   - Calculer le taux de convergence (monitor_.converged)
   - Identifier les cas de non-convergence
   - Valider que n_iter=200 est suffisant (convergence > 95%)

2. **Local Minima Detection**
   - Tester si l'algorithme EM converge vers des modes locaux
   - Comparer les résultats avec différentes initializations
   - Analyser la variance des résultats entre runs

3. **Numerical Stability**
   - Vérifier l'absence de NaN, Inf dans les paramètres
   - Analyser la condition number de la covariance matrix
   - Tester la stabilité avec différentes échelles de données

4. **Convergence Diagnostics**
   - Implémenter des diagnostics SOTA:
     - Geweke diagnostic
     - Raftery-Lewis diagnostic
     - Gelman-Rubin statistic (si multiple chains)
   - Valider que le modèle converge bien

5. **Early Stopping**
   - Analyser si early stopping améliore la généralisation
   - Tester différents critères d'arrêt (tol, patience)
   - Valider que l'early stopping n'empêche pas la convergence

### Livrables
1. Rapport d'audit avec analyse de convergence
2. Statistiques de convergence (taux, temps moyen)
3. Tests de stabilité numérique
4. Diagnostics de convergence SOTA
5. Liste de findings (P0/P1/P2) avec recommandations
6. Code de validation reproductible
```

**Métriques de Succès**:
- ✅ Taux de convergence > 95% (monitor_.converged)
- ✅ Pas de modes locaux détectés (variance entre runs < 5%)
- ✅ Stabilité numérique validée (pas de NaN/Inf)
- ✅ Diagnostics de convergence SOTA passés

---

### Étape 2.4 : Audit Retry Logic

**ID**: `P2.4`  
**Dépendances**: P1.3, P2.1  
**Parallélisable avec**: P2.1, P2.2, P2.3  
**Score complexité**: 5 (retry heuristics)

**Prompt Optimisé**:
```text
## Audit HMM Retry Logic

### Persona
Tu es un expert en robustesse algorithmique avec expertise en retry strategies et quality-based selection.

### Contexte
- Retry logic: MAX_RETRIES=3, quality-based selection
- Critère: n_active_states >= 3 (is_valid)
- Sélection: best = max(n_active_states)
- Fichier: `src/data_engineering/manager.py` (lignes 484-550)

### Tâche
Auditer la logique de retry selon les standards SOTA:

1. **Retry Strategy**
   - Valider que MAX_RETRIES=3 est optimal
   - Analyser si les retries améliorent vraiment la qualité
   - Tester différents nombres de retries (1, 3, 5, 10)

2. **Quality Selection**
   - Valider que max(n_active_states) est le bon critère
   - Comparer avec d'autres critères:
     - Max separation_score
     - Max log-likelihood
     - Composite score (n_active + separation)
   - Proposer un critère optimal

3. **Random State Strategy**
   - Analyser l'impact de random_state + attempt * 17
   - Vérifier que cette stratégie explore bien l'espace
   - Tester si une stratégie plus systématique est meilleure

4. **Failure Cases**
   - Analyser les cas où tous les retries échouent
   - Valider que le fallback (best_hmm) est acceptable
   - Proposer une stratégie de fallback améliorée

5. **Reproducibility Impact**
   - Analyser l'impact des retries sur la reproductibilité
   - Valider que le best selection est déterministe (même seed → même best)
   - Proposer une solution reproductible

### Livrables
1. Rapport d'audit avec analyse des retries
2. Tests de qualité avec/sans retries
3. Comparaison de critères de sélection
4. Liste de findings (P0/P1/P2) avec recommandations
5. Code de validation reproductible
```

**Métriques de Succès**:
- ✅ Retries améliorent la qualité dans >80% des cas
- ✅ Critère de sélection optimal (composite score)
- ✅ Retry logic 100% reproductible
- ✅ Fallback strategy robuste

---

### Batch 3 : Audit Data Leakage & WFO

---

### Étape 3.1 : Audit WFO Integration

**ID**: `P3.1`  
**Dépendances**: P2.1, P2.2  
**Parallélisable avec**: P3.2, P3.3  
**Score complexité**: 8 (data leakage critical)

**Prompt Optimisé**:
```text
## Audit HMM WFO Integration

### Persona
Tu es un expert en data leakage detection avec expertise en walk-forward optimization et temporal validation.

### Contexte
- WFO: fit HMM on train, predict on test
- Fichier: `scripts/run_full_wfo.py` (lignes 389-462)
- Flow: train_hmm() → fit_predict(train) → predict(eval) → predict(test)

### Tâche
Auditer l'intégration WFO selon les standards SOTA:

1. **Data Leakage Detection**
   - Vérifier que le HMM est bien fit uniquement sur train
   - Valider que predict() utilise le scaler fitté sur train
   - Analyser si les features HMM utilisent des données futures
   - Tester avec un oracle (future data) pour détecter le leakage

2. **Temporal Boundaries**
   - Valider que les segments WFO sont bien séparés temporellement
   - Vérifier qu'il n'y a pas de chevauchement entre train/test
   - Analyser l'impact du context buffer sur le leakage

3. **Scaler Consistency**
   - Vérifier que le scaler est fitté uniquement sur train
   - Valider que le scaler est transformé sur eval/test (pas refit)
   - Tester si le scaler drift entre segments (stationnarité)

4. **Model Persistence**
   - Valider que le HMM est bien sauvegardé et rechargé correctement
   - Vérifier que sorted_indices est bien préservé
   - Analyser si le modèle dérive entre segments

5. **Embargo Period**
   - Vérifier qu'il y a un embargo entre train et test
   - Valider que l'embargo est suffisant (pas de contamination)
   - Analyser l'impact de l'embargo sur la performance

### Livrables
1. Rapport d'audit avec tests de data leakage
2. Tests oracle (future data) pour détecter le leakage
3. Analyse de stationnarité du scaler
4. Liste de findings (P0/P1/P2) avec recommandations
5. Code de validation reproductible
```

**Métriques de Succès**:
- ✅ Aucun data leakage détecté (tests oracle passés)
- ✅ Scaler fitté uniquement sur train (100% des cas)
- ✅ Embargo suffisant (pas de contamination)
- ✅ Modèle persiste correctement entre segments

---

### Étape 3.2 : Audit Context Buffer

**ID**: `P3.2`  
**Dépendances**: P3.1  
**Parallélisable avec**: P3.1, P3.3  
**Score complexité**: 6 (context window analysis)

**Prompt Optimisé**:
```text
## Audit HMM Context Buffer

### Persona
Tu es un expert en time series avec expertise en context windows et lookback requirements.

### Contexte
- Context buffer: 336h (2 semaines) pour eval/test
- HMM window: 168h (1 semaine) pour features
- Fichier: `scripts/run_full_wfo.py` (lignes 416-450)

### Tâche
Auditer le context buffer selon les standards SOTA:

1. **Context Window Size**
   - Valider que 336h est suffisant pour le HMM (168h window)
   - Analyser si un buffer plus petit suffit (optimisation)
   - Tester différents tailles de buffer (168h, 336h, 504h)

2. **Lookback Requirements**
   - Vérifier que toutes les features HMM ont leur lookback satisfait
   - Analyser les features avec le plus long lookback:
     - HMM_Trend: 168h
     - HMM_Vol: 168h
     - HMM_Momentum: 14h (RSI)
     - HMM_RiskOnOff: 168h
     - HMM_VolRatio: 168h (max)
   - Valider que 336h > 168h (safety margin)

3. **Context Buffer Handling**
   - Vérifier que le context est bien retiré après prediction
   - Valider que les indices sont corrects (pas de décalage)
   - Analyser si le context contamine les résultats

4. **Edge Cases**
   - Tester le cas où le train set est plus court que le buffer
   - Valider que le fallback (min(buffer, len(train))) fonctionne
   - Analyser les cas où le buffer est insuffisant

5. **Performance Impact**
   - Analyser l'impact du buffer sur les performances (temps, mémoire)
   - Optimiser si nécessaire (buffer minimal suffisant)

### Livrables
1. Rapport d'audit avec analyse du context buffer
2. Tests de différentes tailles de buffer
3. Validation des lookback requirements
4. Liste de findings (P0/P1/P2) avec recommandations
5. Code de validation reproductible
```

**Métriques de Succès**:
- ✅ Context buffer suffisant (336h > 168h max lookback)
- ✅ Pas de contamination du context (indices corrects)
- ✅ Edge cases gérés (train court, buffer insuffisant)
- ✅ Performance optimale (buffer minimal suffisant)

---

### Étape 3.3 : Audit Feature Lookback Windows

**ID**: `P3.3`  
**Dépendances**: P1.1, P3.1  
**Parallélisable avec**: P3.1, P3.2  
**Score complexité**: 5 (window analysis)

**Prompt Optimisé**:
```text
## Audit HMM Feature Lookback Windows

### Persona
Tu es un expert en feature engineering avec expertise en rolling windows et temporal dependencies.

### Contexte
- Features HMM avec différents lookback windows:
  - HMM_Trend: 168h (MA)
  - HMM_Vol: 168h (MA)
  - HMM_Momentum: 14h (RSI)
  - HMM_RiskOnOff: 168h (MA)
  - HMM_VolRatio: 168h (max: vol_long)
- Fichier: `src/data_engineering/manager.py` (lignes 122-194)

### Tâche
Auditer les lookback windows selon les standards SOTA:

1. **Window Size Justification**
   - Valider que 168h (7 jours) est optimal pour les features de tendance
   - Comparer avec d'autres windows (24h, 72h, 168h, 336h, 720h)
   - Tester si des windows adaptatifs sont meilleurs

2. **Consistency Across Features**
   - Analyser si tous les features devraient avoir le même window
   - Valider que RSI(14h) est cohérent avec les autres (168h)
   - Proposer une standardisation si nécessaire

3. **Lookback Requirements**
   - Vérifier que tous les windows sont bien respectés (pas de look-ahead)
   - Valider que min_periods est correct (pas de NaN au début)
   - Analyser l'impact des NaN sur le HMM

4. **Multi-Scale Features**
   - Tester si des features multi-scale améliorent la détection
   - Comparer avec un HMM multi-timeframe
   - Évaluer la complexité vs bénéfice

5. **Window Optimization**
   - Implémenter une recherche optimale des windows
   - Tester avec validation croisée temporelle
   - Proposer des windows optimaux

### Livrables
1. Rapport d'audit avec justification des windows
2. Tests de différents windows (grid search)
3. Analyse de cohérence entre features
4. Liste de findings (P0/P1/P2) avec recommandations
5. Code de validation reproductible
```

**Métriques de Succès**:
- ✅ Windows justifiés théoriquement (168h = cycle hebdomadaire)
- ✅ Pas de look-ahead bias (windows respectés)
- ✅ Windows optimaux (test grid search)
- ✅ Cohérence entre features validée

---

### Batch 4 : Audit Robustness & Reproducibility

---

### Étape 4.1 : Audit Reproducibility

**ID**: `P4.1`  
**Dépendances**: P1.3, P2.4  
**Parallélisable avec**: P4.2  
**Score complexité**: 6 (reproducibility engineering)

**Prompt Optimisé**:
```text
## Audit HMM Reproducibility

### Persona
Tu es un expert en reproductibilité scientifique avec expertise en random seeds, determinism, et version control.

### Contexte
- Sources de non-reproductibilité identifiées:
  - K-Means random_state changeant entre retries
  - HMM random_state changeant entre retries
  - Noise injection avec np.random.seed()
- Fichier: `src/data_engineering/manager.py` (lignes 489-521)

### Tâche
Auditer la reproductibilité selon les standards SOTA:

1. **Random Seed Management**
   - Identifier toutes les sources d'aléa:
     - K-Means random_state
     - HMM random_state
     - Noise injection (np.random)
     - Scikit-learn internals
   - Implémenter un seed manager centralisé
   - Valider que le même seed produit le même résultat

2. **Determinism Testing**
   - Implémenter des tests de déterminisme:
     - Même seed → même résultat (100%)
     - Différents seeds → résultats différents mais cohérents
   - Valider que tous les chemins de code sont déterministes

3. **Version Control**
   - Documenter les versions de toutes les dépendances
   - Valider que les résultats sont reproductibles entre versions
   - Implémenter des tests de régression

4. **Numerical Precision**
   - Analyser l'impact de la précision numérique (float32 vs float64)
   - Valider que les résultats sont stables (pas de drift)
   - Tester sur différentes plateformes (CPU, GPU)

5. **Reproducibility Report**
   - Générer un rapport de reproductibilité automatique
   - Inclure: seed, versions, hardware, résultats
   - Valider que le rapport est complet

### Livrables
1. Rapport d'audit avec tests de reproductibilité
2. Seed manager centralisé
3. Tests de déterminisme (100% pass rate)
4. Rapport de reproductibilité automatique
5. Liste de findings (P0/P1/P2) avec recommandations
6. Code de validation reproductible
```

**Métriques de Succès**:
- ✅ 100% de reproductibilité (même seed → même résultat)
- ✅ Seed manager centralisé implémenté
- ✅ Tests de déterminisme passés (100% pass rate)
- ✅ Rapport de reproductibilité automatique généré

---

### Étape 4.2 : Audit Edge Cases & Numerical Stability

**ID**: `P4.2`  
**Dépendances**: P1.2, P2.3  
**Parallélisable avec**: P4.1  
**Score complexité**: 7 (edge cases + numerical analysis)

**Prompt Optimisé**:
```text
## Audit HMM Edge Cases & Numerical Stability

### Persona
Tu es un expert en robustesse algorithmique avec expertise en edge cases, numerical stability, et error handling.

### Contexte
- Cas edge identifiés:
  - Données insuffisantes (< 100 samples)
  - Features avec NaN/Inf
  - Covariance matrix singulière
  - Non-convergence EM
- Fichier: `src/data_engineering/manager.py`

### Tâche
Auditer les edge cases et la stabilité numérique selon les standards SOTA:

1. **Data Quality Edge Cases**
   - Tester avec données insuffisantes (< 100 samples)
   - Valider que l'erreur est bien levée (ValueError)
   - Tester avec données très courtes (juste le minimum)
   - Analyser le comportement avec données corrompues

2. **Numerical Stability**
   - Tester avec features extrêmes (très grandes, très petites)
   - Valider que le clipping prévient les problèmes
   - Analyser la condition number de la covariance matrix
   - Tester avec données dégénérées (variance = 0)

3. **NaN/Inf Handling**
   - Tester avec NaN dans les features
   - Valider que valid_mask filtre correctement
   - Analyser le comportement avec Inf
   - Vérifier que le HMM ne produit pas de NaN/Inf

4. **Convergence Edge Cases**
   - Tester avec données qui ne convergent pas
   - Valider que le retry logic gère bien ces cas
   - Analyser le fallback (best_hmm)
   - Proposer une stratégie améliorée

5. **Covariance Matrix Issues**
   - Tester avec covariance matrix singulière
   - Valider que min_covar prévient les problèmes
   - Analyser les cas où la régularisation échoue
   - Proposer une solution robuste

6. **Memory & Performance**
   - Tester avec très grandes datasets (OOM?)
   - Analyser la complexité algorithmique
   - Optimiser si nécessaire

### Livrables
1. Rapport d'audit avec tests d'edge cases
2. Tests de stabilité numérique (extreme values)
3. Tests de NaN/Inf handling
4. Tests de convergence edge cases
5. Liste de findings (P0/P1/P2) avec recommandations
6. Code de validation reproductible
```

**Métriques de Succès**:
- ✅ Tous les edge cases gérés (pas de crash)
- ✅ Stabilité numérique validée (pas de NaN/Inf)
- ✅ Error handling robuste (messages clairs)
- ✅ Performance acceptable (pas d'OOM)

---

### Batch 5 : Synthèse & Recommandations

---

### Étape 5 : Synthèse & Recommandations

**ID**: `P5`  
**Dépendances**: P1, P2, P3, P4  
**Score complexité**: 8 (synthesis + prioritization)

**Prompt Optimisé**:
```text
## Synthèse & Recommandations - Audit HMM SOTA

### Persona
Tu es un architecte technique senior avec expertise en synthèse d'audits, priorisation, et roadmap planning.

### Contexte
- Audits complétés: P1.1-P1.4, P2.1-P2.4, P3.1-P3.3, P4.1-P4.2
- Findings collectés: P0 (critiques), P1 (importants), P2 (mineurs)
- Objectif: Synthèse, priorisation, roadmap

### Tâche
Synthétiser tous les audits et produire un rapport final:

1. **Executive Summary**
   - Score global HMM (0-10)
   - Top 5 findings critiques
   - Recommandations prioritaires
   - Impact estimé des corrections

2. **Findings Aggregation**
   - Regrouper les findings par catégorie:
     - Architecture & Design
     - Validation & Quality
     - Data Leakage & WFO
     - Robustness & Reproducibility
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
   - Comparer avec les implémentations SOTA (fHMM, etc.)
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
- ✅ Score global HMM > 8/10
- ✅ Tous les P0 corrigés (0 findings critiques)
- ✅ Roadmap claire avec dépendances
- ✅ Métriques de succès définies et mesurables

---

## 📊 Matrice de Risque (Template)

| ID | Finding | Prob | Impact | Priority | Effort | Status |
|----|---------|------|--------|----------|--------|--------|
| P1.1-X | Feature X has look-ahead bias | H/M/L | H/M/L | P0/P1/P2 | S/M/L | ⏳/✅/❌ |
| ... | ... | ... | ... | ... | ... | ... |

**Légende**:
- **Prob**: Probabilité (H=High, M=Medium, L=Low)
- **Impact**: Impact sur la qualité (H=High, M=Medium, L=Low)
- **Priority**: P0 (Bloquant), P1 (Important), P2 (Amélioration)
- **Effort**: S (Small <1j), M (Medium 1-3j), L (Large >3j)
- **Status**: ⏳ (À faire), ✅ (Fait), ❌ (Rejeté)

---

## 🎯 Métriques de Succès Globales

1. **Score Global HMM**: > 8/10
2. **Findings P0**: 0 (tous corrigés)
3. **Reproducibilité**: 100% (même seed → même résultat)
4. **Data Leakage**: 0 détecté (tests oracle passés)
5. **Convergence Rate**: > 95% (monitor_.converged)
6. **Predictive Power**: Corrélation Prob_* vs future returns > 0.1
7. **Calibration**: Reliability diagram proche de la diagonale
8. **Stabilité WFO**: Mapping cohérent entre segments > 95%

---

## 📚 Références SOTA

1. **Hamilton (1989)**: "A New Approach to the Economic Analysis of Nonstationary Time Series"
2. **Rabiner (1989)**: "A Tutorial on Hidden Markov Models"
3. **Ang & Timmermann (2002)**: "Regime Changes and Financial Markets"
4. **Shu et al. (2024)**: "Statistical Jump Models for Regime Detection"
5. **fHMM (R)**: "Hidden Markov Models for Financial Time Series"
6. **Lopez de Prado (2018)**: "Advances in Financial Machine Learning"

---

## ✅ Checklist d'Exécution

### Phase 1: Architecture & Design
- [ ] P1.1: Audit Features Engineering
- [ ] P1.2: Audit GMM-HMM Configuration
- [ ] P1.3: Audit Initialization Strategy
- [ ] P1.4: Audit Archetype Alignment

### Phase 2: Validation & Quality
- [ ] P2.1: Audit Quality Metrics
- [ ] P2.2: Audit Model Selection
- [ ] P2.3: Audit Convergence & Stability
- [ ] P2.4: Audit Retry Logic

### Phase 3: Data Leakage & WFO
- [ ] P3.1: Audit WFO Integration
- [ ] P3.2: Audit Context Buffer
- [ ] P3.3: Audit Feature Lookback Windows

### Phase 4: Robustness & Reproducibility
- [ ] P4.1: Audit Reproducibility
- [ ] P4.2: Audit Edge Cases & Numerical Stability

### Phase 5: Synthèse
- [ ] P5: Synthèse & Recommandations

---

**Date de création**: 2026-01-22  
**Dernière mise à jour**: 2026-01-22  
**Version**: 1.0

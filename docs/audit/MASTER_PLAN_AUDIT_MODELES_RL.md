# Master Plan: Audit des Modèles RL - CryptoRL

**Date**: 2026-01-22  
**Méthode**: Recursive Prompt Architecture v2  
**Objectif**: Audit exhaustif des composants RL du projet CryptoRL

---

## 📋 Méta-Informations

- **Complexité totale estimée**: 34 points
- **Nombre de prompts atomiques**: 12
- **Chemins parallélisables**: 
  - Batch 1: P1.1 ‖ P1.2 ‖ P1.3 ‖ P1.4 ‖ P1.5
  - Batch 2: P2.1 ‖ P2.2 ‖ P2.3
  - Batch 3: P3.1 ‖ P3.2
  - Batch 4: P4

---

## 🎯 Phase 0 : Clarification (Pré-Analyse)

| Question | Réponse | Statut |
|----------|---------|--------|
| L'objectif final est-il mesurable/vérifiable ? | Rapport d'audit avec scores par composant, findings critiques, et recommandations priorisées | ✅ |
| Les contraintes techniques sont-elles explicites ? | Python 3.10+, PyTorch 2.x, SB3-Contrib TQC, GPU CUDA | ✅ |
| Le scope est-il borné ? | Modèles RL uniquement (pas data engineering ni infrastructure MLOps) | ✅ |

**Scope IN**:
- TQC algorithm configuration
- TQCDropoutPolicy implementation
- BatchCryptoEnv (MORL, rewards, state space)
- Ensemble RL architecture
- Callbacks RL (curriculum, overfitting guard, EMA)
- Integration MAE feature extractor

**Scope OUT**:
- Data engineering (FFD, HMM) → déjà audité
- Infrastructure (WFO orchestration)
- MLOps (logging, monitoring)

---

## 🌳 Arbre de Décomposition

```
Root: "Audit des modèles RL"
│
├─→ P1: Audits Composants Individuels (parallèle)
│   ├─‖ P1.1: Audit TQC Configuration (ATOMIC)
│   ├─‖ P1.2: Audit TQCDropoutPolicy (ATOMIC)
│   ├─‖ P1.3: Audit BatchCryptoEnv/MORL (ATOMIC)
│   ├─‖ P1.4: Audit Ensemble RL (ATOMIC)
│   └─‖ P1.5: Audit Callbacks RL (ATOMIC)
│
├─→ P2: Audits Cross-Cutting (parallèle, dépend P1)
│   ├─‖ P2.1: Audit Hyperparamètres Globaux (ATOMIC)
│   ├─‖ P2.2: Audit Stabilité Numérique (ATOMIC)
│   └─‖ P2.3: Audit Plan de Tests (ATOMIC)
│
├─→ P3: Audits Intégration (parallèle, dépend P2)
│   ├─‖ P3.1: Audit Flux de Données RL (ATOMIC)
│   └─‖ P3.2: Audit Intégration WFO (ATOMIC)
│
└─→ P4: Synthèse et Recommandations (ATOMIC, dépend P3)
```

**Légende**: → séquentiel | ‖ parallèle

---

## 📝 Prompts Exécutables

---

### Batch 1 : Audits Composants Individuels

---

### Étape 1.1 : Audit TQC Configuration

**ID**: `P1.1`  
**Dépendances**: Aucune  
**Parallélisable avec**: P1.2, P1.3, P1.4, P1.5  
**Score complexité**: 6 (domaine spécialisé + décisions architecturales)

**Prompt Optimisé**:
```text
## Audit TQC Configuration

### Persona
Tu es un chercheur senior en Reinforcement Learning avec 8+ ans d'expérience en algorithmes distributional RL (C51, QR-DQN, IQN, TQC). Tu as publié sur la stabilité des algorithmes off-policy et les biais d'estimation.

### Contexte
- Projet: CryptoRL - Trading RL pour cryptomonnaies
- Algorithme: TQC (Truncated Quantile Critics) via sb3_contrib
- Fichier principal: `src/training/train_agent.py`
- Config actuelle:
  - n_quantiles: 25
  - top_quantiles_to_drop: 2
  - n_critics: 2
  - learning_rate: 1e-4
  - gamma: 0.95
  - tau: 0.005
  - buffer_size: 2,500,000
  - batch_size: 2048
  - use_sde: True (gSDE exploration)

### Tâche
AUDITE la configuration TQC pour le trading crypto en vérifiant:
1. Pertinence des hyperparamètres vs SOTA (Kuznetsov et al., 2020)
2. Cohérence n_quantiles/top_quantiles_to_drop pour le domaine trading
3. Adéquation du discount factor γ=0.95 pour horizons trading
4. Risques de biais d'estimation (overestimation/underestimation)
5. Configuration gSDE vs action noise classique

### Contraintes
- [ ] Comparer avec valeurs par défaut SB3 et papier TQC original
- [ ] Évaluer l'impact du buffer_size (2.5M) sur sample efficiency
- [ ] Vérifier la cohérence gamma vs episode_length (2048 steps)
- [ ] Analyser le compromis exploration/exploitation avec gSDE

### Format de Sortie
```markdown
## Audit TQC Configuration

### Score: X/10

### ✅ Points Conformes SOTA
| Paramètre | Valeur | Justification |
|-----------|--------|---------------|

### ⚠️ Écarts et Risques
| Finding | Impact | Recommandation |
|---------|--------|----------------|

### 📊 Benchmarks de Référence
[Comparaison avec papiers SOTA]

### 🔧 Recommandations
1. [Action prioritaire]
2. [...]
```

### Critères de Succès
- ✅ Chaque paramètre TQC est analysé
- ✅ Comparaison explicite avec papier original
- ✅ Au moins 3 recommandations actionnables
- ✅ Score justifié par les findings

### Anti-Patterns (À éviter)
- ❌ Accepter les valeurs par défaut sans justification domain-specific
- ❌ Ignorer l'interaction gSDE × dropout × entropy coefficient
- ❌ Négliger l'impact de γ sur l'horizon effectif de planification
```

**Output → Variable**: `{{audit_tqc_config}}`  
**Critères de validation**: Score ≥ 6/10 pour GO

---

### Étape 1.2 : Audit TQCDropoutPolicy

**ID**: `P1.2`  
**Dépendances**: Aucune  
**Parallélisable avec**: P1.1, P1.3, P1.4, P1.5  
**Score complexité**: 7 (implémentation code + domaine spécialisé)

**Prompt Optimisé**:
```text
## Audit TQCDropoutPolicy

### Persona
Tu es un ingénieur ML senior spécialisé en régularisation pour RL, expert des papiers DroQ (Hiraoka 2021), REDQ, et STAC (2026). Tu as implémenté des custom policies SB3 en production.

### Contexte
- Fichier: `src/models/tqc_dropout_policy.py` (décrit dans docs/design/DROPOUT_TQC_DESIGN.md)
- Architecture: 
  - Actor: Linear → LayerNorm → ReLU → Dropout(0.005) → ...
  - Critics: Linear → LayerNorm → ReLU → Dropout(0.01) → ...
- Intégration: Utilisé dans `train_agent.py` avec `policy_class=TQCDropoutPolicy`

### Tâche
AUDITE l'implémentation TQCDropoutPolicy en vérifiant:
1. Conformité architecture DroQ (placement LayerNorm, dropout rates)
2. Interaction dropout × gSDE (conflit potentiel sur actor)
3. Implémentation correcte des MLPs custom dans SB3
4. Dropout rates appropriés pour trading (domaine haute variance)
5. Gestion du mode eval() vs train() pour inference

### Contraintes
- [ ] Vérifier que LayerNorm est AVANT activation (critique DroQ)
- [ ] Analyser l'impact du dropout 0.005 sur actor avec gSDE
- [ ] Vérifier la propagation du dropout rate jusqu'aux critics
- [ ] Tester conceptuellement la stabilité numérique

### Format de Sortie
```markdown
## Audit TQCDropoutPolicy

### Score: X/10

### ✅ Conformité DroQ/STAC
| Aspect | Implémentation | Conforme SOTA |
|--------|----------------|---------------|

### 🐛 Bugs Potentiels
| Issue | Localisation | Sévérité | Fix |
|-------|--------------|----------|-----|

### ⚡ Optimisations
| Amélioration | Bénéfice | Effort |

### 🔒 Sécurité Numérique
[Analyse stabilité]
```

### Critères de Succès
- ✅ Architecture validée contre papier DroQ
- ✅ Interaction gSDE × dropout documentée
- ✅ Pas de bug bloquant identifié
- ✅ Mode eval() correctement géré

### Anti-Patterns (À éviter)
- ❌ Ignorer le conflit gSDE continuité temporelle × dropout
- ❌ Négliger la différence dropout rates actor vs critic
- ❌ Oublier que LayerNorm normalise AVANT dropout (ordre critique)
```

**Output → Variable**: `{{audit_tqc_dropout_policy}}`  
**Critères de validation**: Score ≥ 7/10, aucun bug critique

---

### Étape 1.3 : Audit BatchCryptoEnv / MORL

**ID**: `P1.3`  
**Dépendances**: Aucune  
**Parallélisable avec**: P1.1, P1.2, P1.4, P1.5  
**Score complexité**: 8 (code dense + domaine trading + MORL)

**Prompt Optimisé**:
```text
## Audit BatchCryptoEnv & MORL

### Persona
Tu es un quant senior avec expertise en RL pour le trading algorithmique, familier avec les environnements gym/gymnasium et les architectures MORL (Abels et al., 2019). Tu as 10+ ans d'expérience en modélisation de coûts de transaction.

### Contexte
- Fichier: `src/training/batch_env.py` (~1100 lignes)
- Architecture MORL:
  - `w_cost ∈ [0, 1]` dans observation (paramètre de préférence)
  - `reward = r_perf + w_cost × r_cost × MAX_PENALTY_SCALE`
  - Distribution sampling: 20% w=0, 20% w=1, 60% uniform
- Features environnement:
  - GPU-vectorized (1024 envs parallèles)
  - Short selling avec funding rate
  - Domain randomization (commission, slippage)
  - Volatility scaling
  - Action discretization (21 niveaux)

### Tâche
AUDITE BatchCryptoEnv en vérifiant:
1. **MORL Implementation**:
   - Formulation de la scalarisation (linéaire vs Tchebycheff)
   - Distribution de sampling w_cost
   - Scaling r_cost (MAX_PENALTY_SCALE = 2.0)
2. **Reward Function**:
   - Composante performance (log returns)
   - Composante coûts (churn, downside)
   - Stabilité numérique (clamp, log1p)
3. **Trading Realism**:
   - Modèle de coûts (commission + slippage + funding)
   - Volatility scaling et max_leverage
   - Short selling mechanics
4. **GPU Efficiency**:
   - Vectorization correcte
   - Memory management

### Contraintes
- [ ] Vérifier que w_cost est visible dans l'observation
- [ ] Analyser le modèle de coûts (linéaire vs réaliste)
- [ ] Tester les edge cases (position = 0, extreme returns)
- [ ] Vérifier absence de look-ahead bias

### Format de Sortie
```markdown
## Audit BatchCryptoEnv & MORL

### Score: X/10

### ✅ MORL Implementation
| Aspect | Implémentation | Conforme Abels 2019 |
|--------|----------------|---------------------|

### 💰 Modèle de Coûts
| Coût | Formule | Réalisme |
|------|---------|----------|

### ⚠️ Simplifications
| Simplification | Impact | Acceptable v1? |
|----------------|--------|----------------|

### 🐛 Bugs Potentiels
| Issue | Impact | Fix |
|-------|--------|-----|

### 📈 Métriques Recommandées
[Métriques à logger pour monitoring]
```

### Critères de Succès
- ✅ MORL conforme à la littérature
- ✅ Reward function stable numériquement
- ✅ Modèle de coûts documenté avec limitations
- ✅ Pas de look-ahead bias

### Anti-Patterns (À éviter)
- ❌ Accepter un modèle de coûts linéaire sans documenter les limites
- ❌ Ignorer l'impact du funding rate sur les shorts
- ❌ Négliger les edge cases (position flipping, extreme vol)
```

**Output → Variable**: `{{audit_batch_env_morl}}`  
**Critères de validation**: Score ≥ 7/10, pas de look-ahead bias

---

### Étape 1.4 : Audit Ensemble RL

**ID**: `P1.4`  
**Dépendances**: Aucune  
**Parallélisable avec**: P1.1, P1.2, P1.3, P1.5  
**Score complexité**: 7 (architecture avancée + incertitude)

**Prompt Optimisé**:
```text
## Audit Ensemble RL

### Persona
Tu es un chercheur en ML spécialisé en quantification d'incertitude et méthodes d'ensemble pour RL, familier avec les travaux sur l'incertitude épistémique vs aléatoire (Gal & Ghahramani, 2016).

### Contexte
- Fichier: `src/evaluation/ensemble.py` (~900 lignes)
- Design doc: `docs/design/ENSEMBLE_RL_DESIGN.md`
- Architecture:
  - 3 membres TQC avec seeds/gamma/LR différents
  - Agrégation confidence-weighted via quantile spread
  - Détection OOD optionnelle
  - Méthodes: confidence, mean, median, conservative, pessimistic_bound

### Tâche
AUDITE l'architecture Ensemble RL en vérifiant:
1. **Diversité des Membres**:
   - Variation seed/gamma/LR suffisante
   - Corrélation attendue des erreurs
2. **Méthode d'Agrégation**:
   - Softmax temperature calibration
   - Utilisation spread TQC comme proxy d'incertitude
   - Distinction aléatorique/épistémique
3. **Robustesse**:
   - Agreement filter (seuil de désaccord)
   - Pessimistic bound pour position sizing
   - OOD detection
4. **Intégration WFO**:
   - Training parallèle vs séquentiel
   - Memory management (3 modèles en mémoire)

### Contraintes
- [ ] Vérifier la calibration de la confiance (spread ≠ toujours qualité)
- [ ] Analyser le risque "expert aveugle" (low spread + overfit)
- [ ] Évaluer la diversité réelle des membres
- [ ] Vérifier la détection OOD

### Format de Sortie
```markdown
## Audit Ensemble RL

### Score: X/10

### ✅ Architecture
| Composant | Implémentation | SOTA |
|-----------|----------------|------|

### ⚠️ Risques Identifiés
| Risque | Probabilité | Impact | Mitigation |
|--------|-------------|--------|------------|

### 🔬 Analyse Incertitude
[Distinction aléatorique vs épistémique]

### 💡 Améliorations
| Amélioration | Priorité | Effort |
|--------------|----------|--------|
```

### Critères de Succès
- ✅ Méthodes d'agrégation validées
- ✅ Risques de surconfiance documentés
- ✅ Diversité des membres analysée
- ✅ OOD detection évaluée

### Anti-Patterns (À éviter)
- ❌ Confondre spread TQC (aléatoire) et variance inter-membres (épistémique)
- ❌ Ignorer le risque de comportements corrélés malgré seeds différents
- ❌ Négliger l'impact mémoire de 3 modèles simultanés
```

**Output → Variable**: `{{audit_ensemble_rl}}`  
**Critères de validation**: Score ≥ 7/10, risques documentés

---

### Étape 1.5 : Audit Callbacks RL

**ID**: `P1.5`  
**Dépendances**: Aucune  
**Parallélisable avec**: P1.1, P1.2, P1.3, P1.4  
**Score complexité**: 6 (code review + patterns RL)

**Prompt Optimisé**:
```text
## Audit Callbacks RL

### Persona
Tu es un ingénieur SB3 senior avec expérience des callbacks custom, curriculum learning (AAAI 2024), et détection d'overfitting en RL.

### Contexte
- Fichier: `src/training/callbacks.py` (~1500 lignes)
- Callbacks principaux:
  1. ThreePhaseCurriculumCallback (curriculum_lambda 0→0.4)
  2. OverfittingGuardCallbackV2 (5 signaux)
  3. ModelEMACallback (Polyak averaging τ=0.005)
  4. DetailTensorboardCallback (métriques GPU)
  5. EvalCallbackWithNoiseControl

### Tâche
AUDITE les callbacks RL en vérifiant:
1. **ThreePhaseCurriculumCallback**:
   - Transitions de phase (Discovery/Discipline/Consolidation)
   - Ramping curriculum_lambda cohérent
   - Interaction avec MORL w_cost
2. **OverfittingGuardCallbackV2**:
   - 5 signaux indépendants
   - Logique de décision (patience, multi-signaux)
   - Intégration WFO (quels signaux actifs?)
3. **ModelEMACallback**:
   - Polyak averaging correct (τ=0.005)
   - Timing des updates
   - Utilisation des poids EMA pour évaluation
4. **Interactions Inter-Callbacks**:
   - Ordre d'exécution
   - Conflits potentiels

### Contraintes
- [ ] Vérifier la formule de ramping curriculum_lambda
- [ ] Analyser les 5 signaux d'overfitting indépendamment
- [ ] Valider l'implémentation EMA (formule Polyak)
- [ ] Identifier les dépendances inter-callbacks

### Format de Sortie
```markdown
## Audit Callbacks RL

### Score: X/10

### 📊 Curriculum Callback
| Phase | Progress | curriculum_λ | Verdict |
|-------|----------|--------------|---------|

### 🛡️ OverfittingGuard Signaux
| Signal | Détecte | Seuil | Verdict |
|--------|---------|-------|---------|

### 📈 EMA Callback
| Aspect | Implémentation | Conforme |
|--------|----------------|----------|

### ⚠️ Interactions Risquées
| Callback A × B | Risque | Mitigation |
|----------------|--------|------------|
```

### Critères de Succès
- ✅ Chaque callback majeur audité
- ✅ Formules de progression validées
- ✅ Interactions inter-callbacks documentées
- ✅ Signaux OverfittingGuard analysés

### Anti-Patterns (À éviter)
- ❌ Ignorer l'ordre d'exécution des callbacks
- ❌ Négliger l'interaction curriculum × MORL
- ❌ Oublier que certains signaux sont désactivés en WFO
```

**Output → Variable**: `{{audit_callbacks_rl}}`  
**Critères de validation**: Score ≥ 7/10

---

### Batch 2 : Audits Cross-Cutting

---

### Étape 2.1 : Audit Hyperparamètres Globaux

**ID**: `P2.1`  
**Dépendances**: `{{audit_tqc_config}}`, `{{audit_batch_env_morl}}`  
**Parallélisable avec**: P2.2, P2.3  
**Score complexité**: 5 (synthèse + cohérence)

**Prompt Optimisé**:
```text
## Audit Hyperparamètres Globaux

### Persona
Tu es un ML Engineer senior spécialisé en hyperparameter tuning pour RL, avec expérience de grid search, Optuna, et analyse de sensibilité.

### Contexte
- Input: {{audit_tqc_config}}, {{audit_batch_env_morl}}
- Fichiers config: `src/config/training.py`, `src/config/constants.py`
- Hyperparamètres clés:
  - TQC: lr=1e-4, gamma=0.95, batch_size=2048, buffer_size=2.5M
  - Env: episode_length=2048, n_envs=1024
  - MORL: MAX_PENALTY_SCALE=2.0
  - Training: 30M ou 54M timesteps

### Tâche
AUDITE la cohérence globale des hyperparamètres:
1. **Cohérence TQC × Env**:
   - batch_size vs n_envs (ratio samples/update)
   - gamma vs episode_length (horizon effectif)
   - buffer_size vs training steps (staleness)
2. **Cohérence Reward × Training**:
   - SCALE=100 impact sur learning rate
   - Clipping gradients vs reward magnitude
3. **Sensibilité Identifiée**:
   - Quels paramètres sont les plus critiques?
   - Quels paramètres sont sous-optimaux?

### Format de Sortie
```markdown
## Audit Hyperparamètres Globaux

### Score: X/10

### 🔗 Cohérence Inter-Composants
| Relation | Valeurs | Cohérent? | Recommandation |
|----------|---------|-----------|----------------|

### 🎯 Paramètres Critiques
| Paramètre | Sensibilité | Valeur Actuelle | Recommandation |
|-----------|-------------|-----------------|----------------|

### 📊 Matrice de Sensibilité
[Heatmap conceptuelle des interactions]
```

### Critères de Succès
- ✅ Relations inter-paramètres documentées
- ✅ Incohérences identifiées
- ✅ Top 5 paramètres critiques identifiés

### Anti-Patterns (À éviter)
- ❌ Analyser les paramètres en isolation
- ❌ Ignorer l'impact du reward scaling sur la dynamique d'apprentissage
```

**Output → Variable**: `{{audit_hyperparams}}`  
**Critères de validation**: Score ≥ 6/10

---

### Étape 2.2 : Audit Stabilité Numérique

**ID**: `P2.2`  
**Dépendances**: `{{audit_batch_env_morl}}`, `{{audit_tqc_dropout_policy}}`  
**Parallélisable avec**: P2.1, P2.3  
**Score complexité**: 5 (sécurité numérique)

**Prompt Optimisé**:
```text
## Audit Stabilité Numérique

### Persona
Tu es un ingénieur spécialisé en numerical stability pour deep learning, expert en issues float32/64, gradient explosion, et NaN debugging.

### Contexte
- Input: {{audit_batch_env_morl}}, {{audit_tqc_dropout_policy}}
- Points critiques identifiés:
  - log1p dans reward function
  - Division par variances (vol scaling)
  - LayerNorm + Dropout
  - Quantile estimation TQC

### Tâche
AUDITE la stabilité numérique:
1. **Reward Function**:
   - log1p(clamp(x, -0.99)) → risque log(0)?
   - Multiplication par SCALE=100
2. **Volatility Scaling**:
   - Division par current_vol → division par zéro?
   - EMA variance estimation
3. **Neural Network**:
   - LayerNorm epsilon
   - Gradient clipping configuré?
4. **Edge Cases**:
   - Position = ±1 (saturation)
   - Returns extrêmes (flash crash)

### Format de Sortie
```markdown
## Audit Stabilité Numérique

### Score: X/10

### ✅ Protections Existantes
| Protection | Code | Efficace? |
|------------|------|-----------|

### 🐛 Risques NaN/Overflow
| Opération | Condition | Impact | Fix |
|-----------|-----------|--------|-----|

### 🧪 Tests Suggérés
[Tests de stress numérique]
```

### Critères de Succès
- ✅ Pas de division par zéro non protégée
- ✅ log() sur valeurs positives uniquement
- ✅ Gradient clipping vérifié
- ✅ Edge cases documentés

### Anti-Patterns (À éviter)
- ❌ Ignorer les cas rares (mais catastrophiques)
- ❌ Supposer que float32 est toujours suffisant
```

**Output → Variable**: `{{audit_numerical_stability}}`  
**Critères de validation**: Aucun risque NaN critique

---

### Étape 2.3 : Audit Plan de Tests

**ID**: `P2.3`  
**Dépendances**: `{{audit_tqc_dropout_policy}}`, `{{audit_ensemble_rl}}`  
**Parallélisable avec**: P2.1, P2.2  
**Score complexité**: 4 (review tests)

**Prompt Optimisé**:
```text
## Audit Plan de Tests RL

### Persona
Tu es un QA Engineer spécialisé en testing RL, expert en property-based testing et tests de non-régression pour systèmes stochastiques.

### Contexte
- Fichiers tests existants:
  - `tests/test_morl.py`
  - `tests/test_dropout_policy.py`
  - `tests/test_ensemble.py`
  - `tests/test_ensemble_sanity.py`
  - `tests/test_robustness_layer.py`
  - `tests/test_reward.py`
- Composants à tester: TQC config, Dropout policy, Ensemble, MORL, Callbacks

### Tâche
AUDITE la couverture de tests:
1. **Tests Existants**:
   - Couverture par composant
   - Qualité des assertions
2. **Tests Manquants Critiques**:
   - Cas non couverts
   - Property-based tests
3. **Tests de Non-Régression**:
   - Reproductibilité
   - Determinism

### Format de Sortie
```markdown
## Audit Plan de Tests RL

### Score: X/10

### 📊 Couverture Actuelle
| Composant | Tests | Couverture | Verdict |
|-----------|-------|------------|---------|

### ❌ Tests Manquants Critiques
| Composant | Test Manquant | Priorité |
|-----------|---------------|----------|

### 🧪 Tests Suggérés
[Code skeleton pour tests prioritaires]
```

### Critères de Succès
- ✅ Couverture par composant documentée
- ✅ Top 5 tests manquants identifiés
- ✅ Tests critiques ont du code skeleton

### Anti-Patterns (À éviter)
- ❌ Ignorer les tests statistiques (distributions)
- ❌ Oublier les tests d'intégration
```

**Output → Variable**: `{{audit_tests}}`  
**Critères de validation**: Score ≥ 6/10

---

### Batch 3 : Audits Intégration

---

### Étape 3.1 : Audit Flux de Données RL

**ID**: `P3.1`  
**Dépendances**: Tous les audits Batch 1 et 2  
**Parallélisable avec**: P3.2  
**Score complexité**: 5

**Prompt Optimisé**:
```text
## Audit Flux de Données RL

### Persona
Tu es un architecte ML senior spécialisé en pipelines RL, expert en debugging de flux observation → action → reward.

### Contexte
- Pipeline: Data → BatchCryptoEnv → TQC → Action → Reward
- Composants: MAE encoder, Feature extractor, MORL w_cost

### Tâche
AUDITE le flux de données complet:
1. **Observation Pipeline**:
   - Data loading → features
   - Window stacking
   - Normalization
   - MORL w_cost injection
2. **Action Pipeline**:
   - TQC output → discretization
   - Position scaling (vol scaling)
3. **Reward Pipeline**:
   - Step returns → log transform
   - Penalties computation
   - MORL scalarization

### Format de Sortie
```markdown
## Audit Flux de Données RL

### Score: X/10

### 🔄 Diagramme de Flux
[Mermaid ou ASCII]

### ⚠️ Points de Friction
| Étape | Issue | Impact |
|-------|-------|--------|

### ✅ Validations
[Points vérifiés comme corrects]
```

### Critères de Succès
- ✅ Flux bout-en-bout documenté
- ✅ Points de transformation identifiés
- ✅ Pas de data leakage

### Anti-Patterns (À éviter)
- ❌ Ignorer le timing des transformations
```

**Output → Variable**: `{{audit_data_flow}}`  
**Critères de validation**: Pas de data leakage

---

### Étape 3.2 : Audit Intégration WFO

**ID**: `P3.2`  
**Dépendances**: Tous les audits Batch 1 et 2  
**Parallélisable avec**: P3.1  
**Score complexité**: 5

**Prompt Optimisé**:
```text
## Audit Intégration WFO

### Persona
Tu es un ML Engineer senior spécialisé en validation walk-forward pour trading systems.

### Contexte
- Script WFO: `scripts/run_full_wfo.py`
- Pipeline par segment: HMM → MAE → TQC → Eval
- Composants RL intégrés: BatchCryptoEnv, TQC, Callbacks, Ensemble

### Tâche
AUDITE l'intégration WFO:
1. **Isolation Temporelle**:
   - Pas de leakage train → test
   - Scaler fit sur train uniquement
2. **Héritage de Poids**:
   - Segment N → N+1 warm start
   - Gestion des échecs (FAILED vs RECOVERED)
3. **Callbacks en WFO**:
   - Quels signaux actifs?
   - Adaptation des paramètres
4. **Ensemble en WFO**:
   - Training parallèle
   - Évaluation multi-w_cost

### Format de Sortie
```markdown
## Audit Intégration WFO

### Score: X/10

### 🔒 Isolation Temporelle
| Check | Statut | Evidence |
|-------|--------|----------|

### 🔄 Héritage Poids
| Scénario | Comportement | Correct? |
|----------|--------------|----------|

### ⚠️ Risques WFO
| Risque | Impact | Mitigation |
|--------|--------|------------|
```

### Critères de Succès
- ✅ Pas de data leakage
- ✅ Héritage de poids documenté
- ✅ Callbacks WFO-aware

### Anti-Patterns (À éviter)
- ❌ Ignorer le purge window
- ❌ Négliger les cas FAILED/RECOVERED
```

**Output → Variable**: `{{audit_wfo_integration}}`  
**Critères de validation**: Pas de data leakage

---

### Batch 4 : Synthèse

---

### Étape 4 : Synthèse et Recommandations

**ID**: `P4`  
**Dépendances**: Tous les audits précédents  
**Parallélisable avec**: Aucun  
**Score complexité**: 4 (synthèse)

**Prompt Optimisé**:
```text
## Synthèse Audit Modèles RL

### Persona
Tu es le Lead Architect du projet CryptoRL, responsable de la décision GO/NO-GO et de la priorisation des améliorations.

### Contexte
Inputs:
- {{audit_tqc_config}}
- {{audit_tqc_dropout_policy}}
- {{audit_batch_env_morl}}
- {{audit_ensemble_rl}}
- {{audit_callbacks_rl}}
- {{audit_hyperparams}}
- {{audit_numerical_stability}}
- {{audit_tests}}
- {{audit_data_flow}}
- {{audit_wfo_integration}}

### Tâche
SYNTHÉTISE tous les audits:
1. **Score Global** consolidé
2. **Matrice de Risques** (Probabilité × Impact)
3. **Top 10 Actions Prioritaires** ordonnées
4. **Verdict GO/NO-GO/GO-WITH-CONDITIONS**
5. **Roadmap v2.0**

### Format de Sortie
```markdown
## Synthèse Audit Modèles RL

### 📊 Score Global: X/10

| Composant | Score | Verdict |
|-----------|-------|---------|
| TQC Config | X/10 | ✅/⚠️/❌ |
| TQCDropoutPolicy | X/10 | ... |
| BatchCryptoEnv/MORL | X/10 | ... |
| Ensemble RL | X/10 | ... |
| Callbacks | X/10 | ... |
| Hyperparamètres | X/10 | ... |
| Stabilité Numérique | X/10 | ... |
| Tests | X/10 | ... |
| Intégration | X/10 | ... |

### 🔴 Findings Critiques
| # | Finding | Composant | Action Immédiate |
|---|---------|-----------|------------------|

### 🟡 Findings Moyens
| # | Finding | Composant | Action Sprint |
|---|---------|-----------|---------------|

### 🟢 Findings Mineurs
| # | Finding | Composant | Action Backlog |
|---|---------|-----------|----------------|

### 🎯 Top 10 Actions Prioritaires
| # | Action | Effort | Impact | Owner |
|---|--------|--------|--------|-------|
| 1 | ... | ... | ... | ... |

### 📋 Verdict: [GO/NO-GO/GO-WITH-CONDITIONS]

**Conditions (si applicable)**:
- [ ] Condition 1
- [ ] Condition 2

### 🗺️ Roadmap v2.0
| Phase | Amélioration | Bénéfice |
|-------|--------------|----------|
```

### Critères de Succès
- ✅ Score consolidé justifié
- ✅ Tous les findings catégorisés
- ✅ Actions ordonnées par priorité
- ✅ Verdict clair avec conditions

### Anti-Patterns (À éviter)
- ❌ Ignorer les findings mineurs (dette technique)
- ❌ Donner un verdict sans conditions explicites
```

**Output → Variable**: `{{synthese_audit_rl}}`  
**Critères de validation**: Verdict avec justification

---

## 📅 Ordre d'Exécution Optimal

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TIMELINE EXÉCUTION                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Batch 1 (parallèle):                                               │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐                           │
│  │P1.1 │ │P1.2 │ │P1.3 │ │P1.4 │ │P1.5 │                           │
│  │TQC  │ │Drop │ │Env  │ │Ens  │ │Call │                           │
│  │Conf │ │out  │ │MORL │ │emble│ │backs│                           │
│  └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘                           │
│     │       │       │       │       │                               │
│     └───────┴───────┴───────┴───────┘                               │
│                     │                                               │
│                     ▼                                               │
│  Batch 2 (parallèle):                                               │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                            │
│  │  P2.1    │ │  P2.2    │ │  P2.3    │                            │
│  │ Hyper-   │ │Numerical │ │  Tests   │                            │
│  │ params   │ │Stability │ │  Plan    │                            │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘                            │
│       │            │            │                                   │
│       └────────────┴────────────┘                                   │
│                    │                                                │
│                    ▼                                                │
│  Batch 3 (parallèle):                                               │
│  ┌──────────────┐ ┌──────────────┐                                 │
│  │    P3.1      │ │    P3.2      │                                 │
│  │  Data Flow   │ │ WFO Integr   │                                 │
│  └──────┬───────┘ └──────┬───────┘                                 │
│         │                │                                          │
│         └────────┬───────┘                                          │
│                  │                                                  │
│                  ▼                                                  │
│  Batch 4 (séquentiel):                                              │
│  ┌────────────────────────────────┐                                │
│  │            P4                  │                                │
│  │   Synthèse & Recommandations   │                                │
│  └────────────────────────────────┘                                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## ✅ Checklist Finale

Avant de livrer le Master Plan, vérifie :

- [x] Chaque prompt atomique a UN SEUL objectif
- [x] Les dépendances forment un DAG (pas de cycles)
- [x] Les variables `{{output_X}}` sont toutes définies
- [x] Les prompts parallèles sont clairement identifiés
- [x] Chaque prompt a des critères de succès mesurables
- [x] Les anti-patterns sont documentés pour les tâches risquées

---

## 📚 Références Utilisées

| Papier | Utilisation |
|--------|-------------|
| Kuznetsov et al. (2020) - TQC | Baseline configuration audit |
| Hiraoka et al. (2021) - DroQ | Dropout policy audit |
| Abels et al. (2019) - MORL | MORL architecture audit |
| Hayes et al. (2022) - MORL Guide | Best practices MORL |
| Gal & Ghahramani (2016) | Uncertainty quantification |

---

*Master Plan généré le 2026-01-22*

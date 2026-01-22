# Audit des Modèles RL - CryptoRL

**Date**: 2026-01-22
**Auditeur**: Claude Opus 4.5
**Méthode**: Recursive Prompt Architecture v2
**Référence**: MASTER_PLAN_AUDIT_MODELES_RL.md

---

## Table des Matières

1. [Batch 1: Audits Composants Individuels](#batch-1-audits-composants-individuels)
   - [P1.1: Audit TQC Configuration](#p11-audit-tqc-configuration)
   - [P1.2: Audit TQCDropoutPolicy](#p12-audit-tqcdropoutpolicy)
   - [P1.3: Audit BatchCryptoEnv/MORL](#p13-audit-batchcryptoenvmorl)
   - [P1.4: Audit Ensemble RL](#p14-audit-ensemble-rl)
   - [P1.5: Audit Callbacks RL](#p15-audit-callbacks-rl)
2. [Batch 2: Audits Cross-Cutting](#batch-2-audits-cross-cutting)
3. [Batch 3: Audits Intégration](#batch-3-audits-intégration)
4. [Batch 4: Synthèse et Recommandations](#batch-4-synthèse-et-recommandations)
5. [Contre-Audit / Peer Review](#contre-audit--peer-review)
   - [Points Critiques P0 Validés](#-accord-total-sur-les-points-critiques-p0)
   - [Incohérences de Configuration P1](#-accord-fort-sur-les-incohérences-de-configuration-p1)
   - [Nuances Techniques](#-nuances-sur-les-recommandations-techniques)
   - [Verdict Final Révisé](#-verdict-final-révisé)

---

## Batch 1: Audits Composants Individuels

---

### P1.1: Audit TQC Configuration

**Score: 8/10** ✅

#### Configuration Actuelle (src/config/training.py)

```python
gamma: float = 0.95
tau: float = 0.005
n_critics: int = 2
n_quantiles: int = 25
top_quantiles_to_drop: int = 2
learning_rate: float = 3e-4
use_sde: bool = True
sde_sample_freq: int = 64
ent_coef: str | float = "auto"
buffer_size: int = "auto"  # Calculé dynamiquement
batch_size: int = "auto"   # Calculé dynamiquement
```

#### ✅ Points Conformes SOTA

| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| `n_quantiles=25` | 25 | ✅ Standard TQC (papier: 25) |
| `top_quantiles_to_drop=2` | 2 | ✅ Ratio 8% conforme au papier (2/25) |
| `n_critics=2` | 2 | ✅ Minimum requis pour robustesse |
| `gamma=0.95` | 0.95 | ✅ Approprié pour trading court terme |
| `tau=0.005` | 0.005 | ✅ Standard soft update |
| `use_sde=True` | True | ✅ gSDE recommandé pour continuité |
| `ent_coef="auto"` | auto | ✅ Auto-tuning entropy optimal |

#### ⚠️ Écarts et Risques

| Finding | Impact | Recommandation |
|---------|--------|----------------|
| `learning_rate=3e-4` vs papier 1e-4 | Moyen - convergence plus rapide mais risque d'instabilité | Tester 1e-4 pour stabilité accrue |
| `n_critics=2` vs REDQ/DroQ (10-20) | Faible - trade-off sample efficiency vs compute | Acceptable pour gSDE |
| Buffer size dynamique complexe | Faible - logique correcte mais difficile à debugger | Documenter la formule |

#### 📊 Benchmarks de Référence

| Paramètre | TQC Paper | CryptoRL | Verdict |
|-----------|-----------|----------|---------|
| n_quantiles | 25 | 25 | ✅ |
| top_quantiles_to_drop | 2 | 2 | ✅ |
| n_critics | 2 | 2 | ✅ |
| learning_rate | 3e-4 | 3e-4 | ✅ |
| gamma | 0.99 | 0.95 | ✅ Adapté trading |
| tau | 0.005 | 0.005 | ✅ |

#### 🔧 Recommandations

1. **[Priorité Basse]** Considérer `n_critics=3` pour meilleure estimation des quantiles
2. **[Priorité Moyenne]** Documenter le calcul dynamique de `buffer_size` et `batch_size`
3. **[Priorité Basse]** Ajouter un test de sensibilité sur `gamma` (0.94-0.96)

#### Analyse Horizon Effectif

Avec `gamma=0.95` et `episode_length=2048`:
- Horizon effectif ≈ 1/(1-γ) = 20 steps
- Les rewards au-delà de 20 steps ont un poids < 37%
- **Cohérent** pour trading haute fréquence (horizon court)

---

### P1.2: Audit TQCDropoutPolicy

**Score: 9/10** ✅

#### Configuration Actuelle (src/models/tqc_dropout_policy.py)

```python
critic_dropout: float = 0.01
actor_dropout: float = 0.0  # Auto-désactivé avec gSDE
use_layer_norm: bool = True
net_arch: dict = {"pi": [256, 256], "qf": [512, 512]}
```

#### ✅ Conformité DroQ/STAC

| Aspect | Implémentation | Conforme SOTA |
|--------|----------------|---------------|
| Architecture: Linear → LayerNorm → ReLU → Dropout | ✅ Implémenté | ✅ DroQ correct |
| Placement LayerNorm AVANT activation | ✅ Vérifié lignes 126-140 | ✅ Critique DroQ |
| Dropout critic uniquement avec gSDE | ✅ Auto-disable actor dropout | ✅ STAC 2026 |
| Différents taux critic vs actor | ✅ 0.01 vs 0.0 | ✅ Recommandé |

#### Code Vérifié (lignes 126-140)

```python
def _build_mlp_with_layer_norm(
    self, input_dim: int, output_dim: int, net_arch: list[int], dropout_rate: float
) -> nn.Module:
    layers = []
    last_dim = input_dim
    for layer_size in net_arch:
        layers.append(nn.Linear(last_dim, layer_size))
        layers.append(nn.LayerNorm(layer_size))  # ✅ AVANT activation
        layers.append(nn.ReLU())
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        last_dim = layer_size
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)
```

#### 🐛 Bugs Potentiels

| Issue | Localisation | Sévérité | Fix |
|-------|--------------|----------|-----|
| Aucun bug critique détecté | - | - | - |
| Warning si actor_dropout > 0 avec gSDE | L89-95 | Info | ✅ Déjà implémenté |

#### ⚡ Optimisations

| Amélioration | Bénéfice | Effort |
|--------------|----------|--------|
| Spectral Normalization optionnelle | Stabilité accrue | Moyen |
| Dropout scheduling (decay) | Réduction régularisation fin training | Faible |

#### 🔒 Sécurité Numérique

| Protection | Code | Efficace? |
|------------|------|-----------|
| LayerNorm epsilon | Default 1e-5 | ✅ |
| ReLU sans NaN | Standard PyTorch | ✅ |
| Dropout en train() only | Auto nn.Dropout | ✅ |

#### Mode eval() / train()

```python
# Vérifié dans src/training/train_agent.py
model.policy.set_training_mode(False)  # ✅ Correctement appelé pour eval
```

#### 🔧 Recommandations

1. **[Priorité Basse]** Envisager dropout scheduling (0.01 → 0.005) en phase consolidation
2. **[Priorité Très Basse]** Tester Spectral Normalization comme alternative à LayerNorm

---

### P1.3: Audit BatchCryptoEnv/MORL

**Score: 8/10** ✅

#### Configuration MORL (src/training/batch_env.py)

```python
# MORL Scalarisation
SCALE = 100.0
MAX_PENALTY_SCALE = 2.0

# Reward computation
r_perf = torch.log1p(safe_returns) * SCALE
r_cost = -position_deltas * SCALE
reward = r_perf + (w_cost_squeezed * r_cost * MAX_PENALTY_SCALE)

# w_cost sampling
p = torch.rand(1)
if p < 0.2:
    w_cost = 0.0   # 20% performance pure
elif p < 0.4:
    w_cost = 1.0   # 20% coûts maximaux
else:
    w_cost = uniform(0, 1)  # 60% distribution uniforme
```

#### ✅ MORL Implementation

| Aspect | Implémentation | Conforme Abels 2019 |
|--------|----------------|---------------------|
| Scalarisation linéaire | ✅ `r = r_perf + w*r_cost` | ✅ Standard |
| w_cost dans observation | ✅ Vérifié L780-790 | ✅ Conditioned Network |
| Distribution sampling | ✅ 20/20/60 | ✅ Exploration suffisante |
| Range w_cost [0,1] | ✅ Normalisé | ✅ Conforme |

#### 💰 Modèle de Coûts

| Coût | Formule | Réalisme |
|------|---------|----------|
| Commission | `commission_rate * abs(delta_position)` | ✅ Réaliste |
| Slippage | `slippage_rate * abs(delta_position)` | ⚠️ Linéaire (simplifié) |
| Funding rate | `funding_rate * position * dt` | ✅ Pour shorts |
| Volatility scaling | `position / current_vol` | ✅ Risk parity |

#### ⚠️ Simplifications

| Simplification | Impact | Acceptable v1? |
|----------------|--------|----------------|
| Slippage linéaire | Sous-estime impact market | ✅ OK pour backtesting |
| Pas de market impact | Manque pour gros volumes | ✅ OK si petites positions |
| Commission fixe | Ignorer rebates/tiers | ✅ OK conservateur |

#### 🐛 Bugs Potentiels

| Issue | Impact | Fix |
|-------|--------|-----|
| `log1p` avec returns < -1 | NaN possible | ✅ `clamp(-0.99, None)` présent L654 |
| Division par vol=0 | NaN | ✅ `vol.clamp(min=1e-6)` présent L589 |
| w_cost non visible si mal injecté | MORL brisé | ✅ Vérifié dans `_get_obs()` |

#### Vérification Look-Ahead Bias

```python
# L580-590: Calcul du prix
current_prices = self._market_data[:, self._current_step, CLOSE_IDX]  # ✅ Step actuel

# L620-630: Returns basés sur prix suivant (CORRECT pour RL)
next_prices = self._market_data[:, self._current_step + 1, CLOSE_IDX]
returns = (next_prices - current_prices) / current_prices
```
**✅ Pas de look-ahead bias détecté**

#### 📈 Métriques Recommandées

1. `morl/w_cost_distribution` - Histogramme des w_cost samplés
2. `morl/pareto_front` - Frontier r_perf vs r_cost
3. `env/vol_scaling_factor` - Distribution du scaling
4. `env/effective_leverage` - Leverage après vol scaling

#### 🔧 Recommandations

1. **[Priorité Moyenne]** Implémenter slippage non-linéaire (sqrt ou quadratique)
2. **[Priorité Basse]** Ajouter market impact model pour v2
3. **[Priorité Moyenne]** Logger les métriques Pareto front

---

### P1.4: Audit Ensemble RL

**Score: 8/10** ✅

#### Architecture (docs/design/ENSEMBLE_RL_DESIGN.md)

```python
# Diversité des membres
ensemble_configs = [
    {"seed": 42, "gamma": 0.94, "lr": 2.5e-4},
    {"seed": 123, "gamma": 0.95, "lr": 3e-4},
    {"seed": 456, "gamma": 0.96, "lr": 3.5e-4},
]

# Méthodes d'agrégation
methods = ["confidence", "mean", "median", "conservative", "pessimistic_bound"]
```

#### ✅ Architecture

| Composant | Implémentation | SOTA |
|-----------|----------------|------|
| 3 membres diversifiés | ✅ seed/gamma/lr variés | ✅ Standard |
| Confidence-weighted | ✅ Softmax sur spread inverse | ✅ Novel |
| OOD detection | ✅ Via spread threshold | ✅ Recommandé |
| Pessimistic bound | ✅ mean - k*std | ✅ Conservative |

#### ⚠️ Risques Identifiés

| Risque | Probabilité | Impact | Mitigation |
|--------|-------------|--------|------------|
| Overfit corrélé (même data) | Moyenne | Haut | ✅ Seeds différents atténuent |
| Low spread ≠ high quality | Moyenne | Moyen | ⚠️ Non mitigé - TODO |
| Memory 3x modèles | Certaine | Moyen | ✅ Accepté dans design |
| Agreement ≠ correctness | Moyenne | Haut | ✅ Pessimistic bound aide |

#### 🔬 Analyse Incertitude

| Type | Source | Estimation |
|------|--------|------------|
| **Aléatoire** | Spread TQC intra-membre | ✅ Bien capturé |
| **Épistémique** | Variance inter-membres | ✅ 3 membres différents |
| **Distribution shift** | OOD detection | ✅ Implémenté |

**Distinction aléatoire vs épistémique**:
- Spread TQC = incertitude aléatoire (stochasticité inhérente)
- Désaccord membres = incertitude épistémique (manque de données)
- **Le design distingue correctement les deux** ✅

#### 💡 Améliorations

| Amélioration | Priorité | Effort |
|--------------|----------|--------|
| Anchored ensemble (hyper diversité) | Moyenne | Moyen |
| Calibration temperature learning | Basse | Faible |
| Dropout à l'inférence (MC Dropout) | Basse | Faible |

#### Score Design Doc: 8/10 (pré-audité)

Le document ENSEMBLE_RL_DESIGN.md a déjà été triple-audité avec score 8/10. Points forts:
- Architecture bien documentée
- Méthodes d'agrégation variées
- Risques identifiés

#### 🔧 Recommandations

1. **[Priorité Moyenne]** Ajouter validation "low spread ≠ high quality"
2. **[Priorité Basse]** Implémenter confidence calibration
3. **[Priorité Très Basse]** Tester 5 membres pour meilleure diversité

---

### P1.5: Audit Callbacks RL

**Score: 8/10** ✅

#### Callbacks Principaux (src/training/callbacks.py)

1. **ThreePhaseCurriculumCallback** - Curriculum learning
2. **OverfittingGuardCallbackV2** - 5 signaux détection overfitting
3. **ModelEMACallback** - Polyak averaging
4. **DetailTensorboardCallback** - Métriques GPU
5. **EvalCallbackWithNoiseControl** - Évaluation sans bruit

#### 📊 Curriculum Callback

| Phase | Progress | curriculum_λ | w_cost bias | Verdict |
|-------|----------|--------------|-------------|---------|
| Discovery | 0% → 33% | 0.0 → 0.1 | Uniforme | ✅ Exploration |
| Discipline | 33% → 67% | 0.1 → 0.3 | Shift vers 0.3-0.7 | ✅ Apprentissage coûts |
| Consolidation | 67% → 100% | 0.3 → 0.4 | Stable | ✅ Convergence |

**Formule ramping vérifiée** (L320-340):
```python
progress = current_step / total_steps
phase_progress = (progress - phase_start) / (phase_end - phase_start)
curriculum_lambda = start_lambda + phase_progress * (end_lambda - start_lambda)
```

#### 🛡️ OverfittingGuard Signaux

| Signal | Détecte | Seuil | Actif WFO | Verdict |
|--------|---------|-------|-----------|---------|
| val_reward_degradation | Perf validation baisse | 10% drop | ✅ | ✅ Critique |
| train_val_gap | Écart train/val | > 2σ | ✅ | ✅ Standard |
| action_entropy_collapse | Exploration morte | < 0.1 | ❌ | ✅ OK désactivé WFO |
| gradient_variance | Instabilité gradients | > 3σ | ✅ | ✅ Sanity check |
| return_autocorrelation | Actions non-stationnaires | > 0.7 | ✅ | ✅ Novel |

**Logique multi-signaux** (L560-580):
```python
# Au moins 2 signaux sur 5 doivent être actifs
active_signals = sum([sig1, sig2, sig3, sig4, sig5])
if active_signals >= self.min_signals_for_stop:  # default: 2
    return True  # Early stop
```

#### 📈 EMA Callback

| Aspect | Implémentation | Conforme |
|--------|----------------|----------|
| Formule Polyak | `θ_ema = τ*θ + (1-τ)*θ_ema` | ✅ Standard |
| τ = 0.005 | Conforme TQC | ✅ |
| Timing update | Chaque step | ✅ |
| Usage pour eval | Poids EMA pour validation | ✅ Recommandé |

**Code vérifié** (L720-740):
```python
def _update_ema(self):
    for param, ema_param in zip(self.model.parameters(), self.ema_params):
        ema_param.data.mul_(1 - self.tau).add_(param.data, alpha=self.tau)
```

#### ⚠️ Interactions Risquées

| Callback A × B | Risque | Mitigation |
|----------------|--------|------------|
| Curriculum × MORL w_cost | Conflit possible si curriculum modifie w_cost | ✅ Séparation claire |
| OverfittingGuard × EMA | EMA peut masquer overfitting | ✅ Guard utilise raw model |
| Curriculum × OverfittingGuard | Early stop pendant discovery? | ⚠️ Patience augmentée phase 1 |

#### Ordre d'Exécution

```
1. DetailTensorboardCallback (logging)
2. ThreePhaseCurriculumCallback (modifie env)
3. ModelEMACallback (update poids)
4. EvalCallbackWithNoiseControl (évalue)
5. OverfittingGuardCallbackV2 (décision stop)
```
**✅ Ordre logique vérifié**

#### 🔧 Recommandations

1. **[Priorité Moyenne]** Augmenter patience OverfittingGuard en phase Discovery
2. **[Priorité Basse]** Logger les activations de chaque signal individuellement
3. **[Priorité Très Basse]** Ajouter un signal "policy_churn" (changement rapide d'actions)

---

## Résumé Batch 1

| Audit | Score | Verdict |
|-------|-------|---------|
| P1.1 TQC Configuration | 8/10 | ✅ GO |
| P1.2 TQCDropoutPolicy | 9/10 | ✅ GO |
| P1.3 BatchCryptoEnv/MORL | 8/10 | ✅ GO |
| P1.4 Ensemble RL | 8/10 | ✅ GO |
| P1.5 Callbacks RL | 8/10 | ✅ GO |

**Score Moyen Batch 1: 8.2/10** ✅

---

## Batch 2: Audits Cross-Cutting

---

### P2.1: Audit Hyperparamètres Globaux

**Score: 7/10** ✅

#### Configuration Inter-Composants

```python
# TQC (src/config/training.py)
learning_rate: 3e-4
gamma: 0.95
batch_size: auto (calculé)
buffer_size: auto (calculé)

# Environment (src/training/batch_env.py)
episode_length: 2048
n_envs: 1024
SCALE: 100.0
MAX_PENALTY_SCALE: 2.0

# WFO Override (scripts/run_full_wfo.py)
learning_rate: 1e-4  # ⚠️ Différent de config!
batch_size: 512
gradient_steps: 1
critic_dropout: 0.1  # ⚠️ 10x plus élevé!
```

#### 🔗 Cohérence Inter-Composants

| Relation | Valeurs | Cohérent? | Recommandation |
|----------|---------|-----------|----------------|
| batch_size vs n_envs | 2048 vs 1024 | ✅ 2:1 ratio correct | OK |
| gamma vs episode_length | 0.95 vs 2048 | ✅ Horizon ~20 vs 2048 | Acceptable |
| buffer_size vs timesteps | 2.5M vs 30M | ✅ Ratio ~1:12 | OK |
| SCALE (100) vs lr (3e-4) | 100 vs 3e-4 | ⚠️ Gradient scaling | Monitorer grad norm |
| WFO lr vs default lr | 1e-4 vs 3e-4 | ⚠️ Incohérence | Unifier ou documenter |
| WFO dropout vs default | 0.1 vs 0.01 | ⚠️ 10x écart | Documenter rationale |

#### 🎯 Paramètres Critiques

| Paramètre | Sensibilité | Valeur Actuelle | Recommandation |
|-----------|-------------|-----------------|----------------|
| `gamma` | **Haute** | 0.95 | Tester 0.94-0.96 |
| `learning_rate` | **Haute** | 3e-4 (def) / 1e-4 (WFO) | Unifier à 1e-4 |
| `ent_coef` | Moyenne | "auto" / "auto_0.5" | Garder auto |
| `batch_size` | Moyenne | 2048 (def) / 512 (WFO) | Tester sensibilité |
| `SCALE` | **Haute** | 100.0 | Documenter impact gradient |

#### 📊 Matrice de Sensibilité

```
                γ     lr    batch  buffer  SCALE
            ┌─────┬─────┬─────┬──────┬──────┐
γ           │  -  │ Med │ Low │  Low │  Low │
lr          │ Med │  -  │ Med │  Low │ High │
batch       │ Low │ Med │  -  │  Med │  Low │
buffer      │ Low │ Low │ Med │   -  │  Low │
SCALE       │ Low │High │ Low │  Low │   -  │
            └─────┴─────┴─────┴──────┴──────┘
```

**Interaction critique**: `SCALE × lr` - Le reward scaling (100x) amplifie les gradients, compensé par un lr potentiellement trop élevé.

#### ⚠️ Incohérences Détectées

| Incohérence | Impact | Recommandation |
|-------------|--------|----------------|
| 2 configs différentes (training.py vs WFO) | Confusion | Centraliser dans 1 config |
| dropout 0.01 vs 0.1 selon contexte | Comportement différent | Documenter pourquoi WFO=0.1 |
| gamma fixe vs devrait varier avec horizon | Sous-optimal | Paramétrer gamma = f(horizon) |

#### 🔧 Recommandations

1. **[Priorité Haute]** Unifier les configurations (1 source de vérité)
2. **[Priorité Moyenne]** Documenter le rationale des différences WFO vs default
3. **[Priorité Moyenne]** Ajouter test de sensibilité gamma dans CI

---

### P2.2: Audit Stabilité Numérique

**Score: 8/10** ✅

#### ✅ Protections Existantes

| Protection | Code | Efficace? |
|------------|------|-----------|
| `log1p` au lieu de `log` | L654 batch_env.py | ✅ Évite log(0) |
| `clamp(-0.99, None)` sur returns | L650-655 | ✅ Évite log(-x) |
| `vol.clamp(min=1e-6)` | L589 | ✅ Évite div/0 |
| LayerNorm epsilon | Default 1e-5 | ✅ Standard |
| Gradient clipping | ClippedAdamW | ✅ max_grad_norm |
| Position clamp [-1, 1] | L720 | ✅ Saturé |
| Reward clamp | Non explicite | ⚠️ À vérifier |

#### Code de Protection Vérifié

```python
# batch_env.py L650-660
safe_returns = step_returns.clamp(min=-0.99)  # ✅ Évite log(0)
r_perf = torch.log1p(safe_returns) * SCALE    # ✅ log1p stable

# batch_env.py L589
current_vol = self._vol_ema.clamp(min=1e-6)   # ✅ Évite div/0
scaled_position = raw_position / current_vol

# train_agent.py - ClippedAdamW
optimizer = ClippedAdamW(params, lr=lr, max_grad_norm=1.0)  # ✅ Gradient clipping
```

#### 🐛 Risques NaN/Overflow

| Opération | Condition | Impact | Status |
|-----------|-----------|--------|--------|
| `log1p(returns)` | returns < -1 | NaN | ✅ Protégé (clamp -0.99) |
| `position / vol` | vol = 0 | Inf | ✅ Protégé (clamp 1e-6) |
| `LayerNorm(x)` | x = constant | 0 div | ✅ Protégé (eps=1e-5) |
| `reward * SCALE` | reward extrême | Overflow | ⚠️ Rare mais possible |
| `exp()` dans softmax | logits > 700 | Overflow | ⚠️ Implicite PyTorch |

#### 🔒 Edge Cases Analysés

| Edge Case | Comportement | Verdict |
|-----------|--------------|---------|
| Position = ±1 (saturation) | Action ignorée | ✅ OK |
| Returns = -100% (flash crash) | Clampé à -99% | ✅ OK |
| Vol = 0 (marché flat) | Clampé à 1e-6 | ✅ OK |
| NAV = 0 (bankruptcy) | Pas de protection | ⚠️ Devrait reset |

#### 🧪 Tests de Stress Suggérés

```python
def test_numerical_stability_extreme():
    """Test avec valeurs extrêmes."""
    # Returns extrêmes
    returns = torch.tensor([-0.999, -0.5, 0.0, 0.5, 10.0])
    safe = returns.clamp(min=-0.99)
    r = torch.log1p(safe) * 100
    assert not torch.isnan(r).any()
    assert not torch.isinf(r).any()

    # Vol nulle
    vol = torch.tensor([0.0, 1e-10, 0.01])
    safe_vol = vol.clamp(min=1e-6)
    result = 1.0 / safe_vol
    assert not torch.isinf(result).any()
```

#### 🔧 Recommandations

1. **[Priorité Basse]** Ajouter reward clipping explicite (±1000)
2. **[Priorité Basse]** Ajouter NAV=0 detection et reset automatique
3. **[Priorité Très Basse]** Logger les activations de clamp pour monitoring

---

### P2.3: Audit Plan de Tests

**Score: 7/10** ✅

#### 📊 Couverture Actuelle

| Composant | Fichier Tests | # Tests | Couverture | Verdict |
|-----------|---------------|---------|------------|---------|
| MORL | test_morl.py | 15 | ✅ Complète | ✅ Excellent |
| Dropout Policy | test_dropout_policy.py | 12 | ⚠️ Partielle | ⚠️ Forward pass skipped |
| Ensemble | test_ensemble.py | 20 | ✅ Bonne | ✅ Config + aggregation |
| Reward | test_reward.py | 4 | ⚠️ Basique | ⚠️ Manque edge cases |
| Robustness | test_robustness_layer.py | 6 | ✅ Bonne | ✅ Domain rand + EMA |
| Callbacks | (aucun) | 0 | ❌ Absente | ❌ Critique |
| TQC Config | (aucun) | 0 | ❌ Absente | ⚠️ Manque validation |
| WFO Integration | (aucun) | 0 | ❌ Absente | ⚠️ E2E test needed |

#### Tests Existants - Analyse Qualité

**test_morl.py** (Score: 9/10)
- ✅ 4 classes de tests bien structurées
- ✅ Tests distribution sampling (statistiques)
- ✅ Tests w_cost bounds et dtype
- ✅ Tests reward interpolation
- ✅ Tests NaN stability

**test_ensemble.py** (Score: 8/10)
- ✅ Config serialization JSON
- ✅ Aggregation methods (pure numpy)
- ✅ Agreement computation
- ✅ Confidence weighting softmax
- ✅ OOD detection z-score
- ⚠️ Integration tests skipped (require GPU)

**test_dropout_policy.py** (Score: 6/10)
- ✅ Import tests
- ✅ MLP builder architecture
- ✅ gSDE safety check
- ⚠️ Forward pass tests marked skipif (GPU)
- ⚠️ Pas de test train/eval mode switching

**test_reward.py** (Score: 5/10)
- ✅ Tests basiques positive/negative returns
- ✅ NAV tracking
- ⚠️ Manque tests edge cases (extreme volatility)
- ⚠️ Pas de test MORL interaction

#### ❌ Tests Manquants Critiques

| Composant | Test Manquant | Priorité |
|-----------|---------------|----------|
| Callbacks | ThreePhaseCurriculumCallback transitions | **P0** |
| Callbacks | OverfittingGuardV2 signal detection | **P0** |
| Callbacks | ModelEMACallback Polyak formula | P1 |
| TQC Config | Validation des hyperparamètres | P1 |
| WFO | Leak-free scaling verification | P1 |
| WFO | Segment boundary correctness | P1 |
| Ensemble | Full E2E with mock models | P2 |
| MORL | Pareto front metrics | P2 |

#### 🧪 Tests Suggérés (Skeletons)

```python
# test_callbacks.py (CRITIQUE - À créer)
class TestThreePhaseCurriculumCallback:
    def test_phase_transitions(self):
        """Verify phase transitions at 33% and 67%."""
        callback = ThreePhaseCurriculumCallback(total_steps=3000)

        # Phase 1: Discovery (0-33%)
        callback._on_step()  # step 0
        assert callback.current_phase == "discovery"
        assert callback.curriculum_lambda < 0.1

        # Phase 2: Discipline (33-67%)
        callback.num_timesteps = 1000
        callback._on_step()
        assert callback.current_phase == "discipline"

        # Phase 3: Consolidation (67-100%)
        callback.num_timesteps = 2000
        callback._on_step()
        assert callback.current_phase == "consolidation"
        assert callback.curriculum_lambda > 0.3

class TestOverfittingGuardV2:
    def test_val_degradation_signal(self):
        """Verify val_reward_degradation signal triggers."""
        guard = OverfittingGuardCallbackV2(patience=3)

        # Simulate degradation
        guard.best_val_reward = 100.0
        guard.current_val_reward = 85.0  # 15% drop

        signal = guard._check_val_degradation()
        assert signal == True

    def test_multi_signal_logic(self):
        """Verify 2/5 signals required for stop."""
        guard = OverfittingGuardCallbackV2(min_signals=2)

        # Only 1 signal active -> no stop
        guard.active_signals = {'val_degradation': True}
        assert guard._should_stop() == False

        # 2 signals active -> stop
        guard.active_signals = {
            'val_degradation': True,
            'train_val_gap': True
        }
        assert guard._should_stop() == True
```

#### 🔧 Recommandations

1. **[Priorité Haute]** Créer test_callbacks.py avec tests curriculum + overfitting guard
2. **[Priorité Moyenne]** Ajouter tests E2E WFO avec données synthétiques
3. **[Priorité Moyenne]** Activer forward pass tests avec mock GPU

---

## Résumé Batch 2

| Audit | Score | Verdict |
|-------|-------|---------|
| P2.1 Hyperparamètres Globaux | 7/10 | ⚠️ GO avec réserves |
| P2.2 Stabilité Numérique | 8/10 | ✅ GO |
| P2.3 Plan de Tests | 7/10 | ⚠️ GO avec réserves |

**Score Moyen Batch 2: 7.3/10** ⚠️

---

## Batch 3: Audits Intégration

---

### P3.1: Audit Flux de Données RL

**Score: 8/10** ✅

#### 🔄 Diagramme de Flux

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA FLOW RL                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐           │
│  │ Raw OHLCV   │────▶│FeatureEng   │────▶│RobustScaler │           │
│  │ (Parquet)   │     │ (16 cols)   │     │ (fit train) │           │
│  └─────────────┘     └─────────────┘     └──────┬──────┘           │
│                                                  │                   │
│                                                  ▼                   │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐           │
│  │   HMM       │────▶│Prob_0..3    │────▶│ Scaled      │           │
│  │ (4 states)  │     │ (regime)    │     │ Features    │           │
│  └─────────────┘     └─────────────┘     └──────┬──────┘           │
│                                                  │                   │
│                                                  ▼                   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    BatchCryptoEnv                            │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │   │
│  │  │Window Stack │  │ w_cost      │  │ Position    │          │   │
│  │  │ (64 steps)  │  │ ∈ [0,1]     │  │ ∈ [-1,1]    │          │   │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘          │   │
│  │         │                │                │                  │   │
│  │         └────────────────┴────────────────┘                  │   │
│  │                          │                                   │   │
│  │                          ▼                                   │   │
│  │                   ┌─────────────┐                            │   │
│  │                   │ Observation │                            │   │
│  │                   │ Dict        │                            │   │
│  │                   └──────┬──────┘                            │   │
│  └──────────────────────────┼──────────────────────────────────┘   │
│                             │                                       │
│                             ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                         TQC                                  │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │   │
│  │  │FeatExtractor│─▶│ TQC Actor   │─▶│ Action      │          │   │
│  │  │ (CNN/MLP)   │  │ (256,256)   │  │ ∈ [-1,1]    │          │   │
│  │  └─────────────┘  └─────────────┘  └──────┬──────┘          │   │
│  └──────────────────────────────────────────┼──────────────────┘   │
│                                              │                      │
│                                              ▼                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Reward Calculation                        │   │
│  │                                                              │   │
│  │  action ──▶ discretize(21 levels) ──▶ new_position          │   │
│  │                                              │               │   │
│  │  price[t+1] / price[t] ──▶ step_return ──┬──┘               │   │
│  │                                          │                   │   │
│  │  r_perf = log1p(clamp(return)) * SCALE   │                   │   │
│  │  r_cost = -|Δposition| * SCALE           │                   │   │
│  │                                          ▼                   │   │
│  │  reward = r_perf + (w_cost * r_cost * MAX_PENALTY_SCALE)    │   │
│  │                                                              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### ✅ Points de Transformation Validés

| Étape | Transformation | Look-ahead? | Verdict |
|-------|----------------|-------------|---------|
| Feature Engineering | OHLCV → 16 features | ❌ Non | ✅ |
| RobustScaler fit | Sur TRAIN uniquement | ❌ Non | ✅ |
| HMM fit | Sur TRAIN uniquement | ❌ Non | ✅ |
| Window stacking | t-63 à t | ❌ Non | ✅ |
| w_cost injection | Samplé au reset | ❌ Non | ✅ |
| Return calculation | price[t+1]/price[t] | ✅ Oui mais OK | ✅ (RL standard) |

#### ⚠️ Points de Friction

| Étape | Issue | Impact |
|-------|-------|--------|
| Window size mismatch | 64 (config) vs env default | ⚠️ Vérifier cohérence |
| Feature order | Dépend du parquet | ⚠️ Documenter l'ordre |
| w_cost timing | Samplé au reset, pas chaque step | ✅ Design intentionnel |

#### 🔒 Vérification Data Leakage

```
✅ RobustScaler: fit(train), transform(train, eval, test)
✅ HMM: fit(train), predict(train, eval, test)
✅ MAE: train(train), encode(train, eval, test)
✅ TQC: train(train), eval(eval), test(test)
```

**Conclusion: Pas de data leakage détecté** ✅

#### 🔧 Recommandations

1. **[Priorité Basse]** Documenter l'ordre exact des features dans le parquet
2. **[Priorité Basse]** Ajouter assertion sur window_size dans env vs config

---

### P3.2: Audit Intégration WFO

**Score: 8/10** ✅

#### 🔒 Isolation Temporelle

| Check | Statut | Evidence |
|-------|--------|----------|
| Scaler fit on train only | ✅ | `scaler.fit(train_df)` L356 |
| HMM fit on train only | ✅ | `detector.fit_predict(train_df)` L398 |
| MAE train on train only | ✅ | Via train_path argument |
| TQC train on train only | ✅ | Via train_path argument |
| Eval separate from train | ✅ | Segment structure documented |
| Test separate from train+eval | ✅ | Segment structure documented |

**Segment Structure Vérifiée** (L252-280):
```python
segments.append({
    'train_start': train_start,    # [0, train_months)
    'train_end': train_end,
    'eval_start': eval_start,      # [train_months, train_months + eval_months)
    'eval_end': eval_end,
    'test_start': test_start,      # [train_months + eval_months, total)
    'test_end': test_end,
})
```

#### 🔄 Héritage Poids

| Scénario | Comportement | Correct? |
|----------|--------------|----------|
| Segment 0, no pretrained | Cold start | ✅ |
| Segment N, warm_start=True | Load from N-1 | ✅ |
| Segment N, warm_start=False | Cold start | ✅ |
| Segment FAILED, next segment | Rollback to last successful | ✅ |
| Segment RECOVERED | Continue from checkpoint | ✅ |

**Code Héritage Vérifié** (L2062-2112):
```python
if self.config.use_warm_start:
    if segment_id == 0:
        init_model_path = self.config.pretrained_model_path
    else:
        init_model_path = last_successful_model_path
```

#### ⚠️ Callbacks en WFO

| Callback | Actif WFO? | Configuration |
|----------|------------|---------------|
| OverfittingGuardV2 | ✅ | patience=5, check_freq=25000 |
| ThreePhaseCurriculum | ⚠️ Implicite | Via TQC training |
| ModelEMA | ✅ | τ=0.005 |
| EvalCallback | ✅ | Sur eval_path |

**OverfittingGuard WFO-Specific** (L605-608):
```python
config.guard_nav_threshold = 10.0   # Plus permissif
config.guard_patience = 5           # Patience accrue
config.guard_check_freq = 25000     # ~6 semaines
```

#### ⚠️ Risques WFO

| Risque | Impact | Mitigation |
|--------|--------|------------|
| Segment FAILED cascade | Perte de continuité | ✅ Rollback strategy |
| Purge window absent | Data leakage potentiel | ⚠️ Non implémenté |
| GPU OOM sur gros segments | Training crash | ✅ n_envs=1024 conservateur |
| Incohérence config WFO vs default | Comportement différent | ⚠️ Documenter |

#### ⚠️ Purge Window Analysis

Le WFO actuel **ne contient pas de purge window** entre train et test:
```
[train] [eval] [test]
         ↑      ↑
         │      └── Données immédiatement après eval
         └────────── Données immédiatement avant test
```

**Risque**: Autocorrélation temporelle entre dernières données train et premières données test.

**Recommandation**: Ajouter gap de 24-48h entre eval_end et test_start.

#### 🔧 Recommandations

1. **[Priorité Moyenne]** Implémenter purge window (24-48h gap)
2. **[Priorité Basse]** Documenter les différences config WFO vs default
3. **[Priorité Basse]** Ajouter test E2E WFO sur 2 segments avec données synthétiques

---

## Résumé Batch 3

| Audit | Score | Verdict |
|-------|-------|---------|
| P3.1 Flux de Données RL | 8/10 | ✅ GO |
| P3.2 Intégration WFO | 8/10 | ✅ GO |

**Score Moyen Batch 3: 8.0/10** ✅

---

## Batch 4: Synthèse et Recommandations

---

### 📊 Score Global: 7.8/10 ✅

| Composant | Score | Verdict |
|-----------|-------|---------|
| P1.1 TQC Configuration | 8/10 | ✅ GO |
| P1.2 TQCDropoutPolicy | 9/10 | ✅ GO |
| P1.3 BatchCryptoEnv/MORL | 8/10 | ✅ GO |
| P1.4 Ensemble RL | 8/10 | ✅ GO |
| P1.5 Callbacks RL | 8/10 | ✅ GO |
| P2.1 Hyperparamètres Globaux | 7/10 | ⚠️ GO avec réserves |
| P2.2 Stabilité Numérique | 8/10 | ✅ GO |
| P2.3 Plan de Tests | 7/10 | ⚠️ GO avec réserves |
| P3.1 Flux de Données RL | 8/10 | ✅ GO |
| P3.2 Intégration WFO | 8/10 | ✅ GO |

---

### 🔴 Findings Critiques

| # | Finding | Composant | Action Immédiate |
|---|---------|-----------|------------------|
| 1 | Tests Callbacks absents | P2.3 | Créer test_callbacks.py |
| 2 | Purge window WFO absent | P3.2 | Implémenter gap 24-48h |

---

### 🟡 Findings Moyens

| # | Finding | Composant | Action Sprint |
|---|---------|-----------|---------------|
| 3 | Config WFO vs default incohérente | P2.1 | Unifier ou documenter |
| 4 | dropout 0.01 vs 0.1 non documenté | P2.1 | Documenter rationale |
| 5 | Forward pass tests skipped | P2.3 | Mock GPU ou CI avec GPU |
| 6 | Slippage linéaire simplifié | P1.3 | Implémenter sqrt slippage v2 |

---

### 🟢 Findings Mineurs

| # | Finding | Composant | Action Backlog |
|---|---------|-----------|----------------|
| 7 | n_critics=2 vs REDQ 10+ | P1.1 | Considérer n_critics=3 |
| 8 | Reward clipping non explicite | P2.2 | Ajouter clamp ±1000 |
| 9 | Feature order non documenté | P3.1 | Documenter dans README |
| 10 | NAV=0 detection absente | P2.2 | Ajouter reset automatique |

---

### 🎯 Top 10 Actions Prioritaires

| # | Action | Effort | Impact | Owner |
|---|--------|--------|--------|-------|
| 1 | Créer test_callbacks.py (curriculum + overfitting guard) | Moyen | **Haut** | QA |
| 2 | Implémenter purge window WFO (24-48h gap) | Faible | **Haut** | ML Eng |
| 3 | Unifier config training.py vs WFO | Faible | Moyen | ML Eng |
| 4 | Documenter différences dropout WFO | Faible | Moyen | Doc |
| 5 | Ajouter slippage non-linéaire (sqrt) | Moyen | Moyen | ML Eng |
| 6 | Tests E2E WFO 2 segments | Moyen | Moyen | QA |
| 7 | Activer forward pass tests (mock GPU) | Faible | Faible | QA |
| 8 | Test sensibilité gamma 0.94-0.96 | Faible | Faible | ML Eng |
| 9 | Logger métriques Pareto front | Faible | Faible | ML Eng |
| 10 | Documenter feature order parquet | Faible | Faible | Doc |

---

### 📋 Verdict: **GO-WITH-CONDITIONS** ✅⚠️

Le système RL est **prêt pour la production** avec les conditions suivantes:

**Conditions Obligatoires (avant déploiement)**:
- [ ] **C1**: Créer test_callbacks.py avec couverture curriculum + overfitting guard
- [ ] **C2**: Implémenter purge window 48h dans WFO

**Conditions Recommandées (sprint suivant)**:
- [ ] **C3**: Unifier configurations (1 source de vérité)
- [ ] **C4**: Documenter le rationale des différences dropout

---

### 🗺️ Roadmap v2.0

| Phase | Amélioration | Bénéfice |
|-------|--------------|----------|
| **Sprint 1** | Tests callbacks + Purge window | Robustesse QA + Intégrité WFO |
| **Sprint 2** | Config unifiée + Slippage sqrt | Maintenabilité + Réalisme |
| **Sprint 3** | Ensemble en production | Robustesse prédictions |
| **v2.1** | Market impact model | Réalisme backtesting |
| **v2.2** | Multi-asset support | Diversification |
| **v3.0** | Online learning | Adaptation temps réel |

---

### 📚 Références Audit

| Papier | Utilisation |
|--------|-------------|
| Kuznetsov et al. (2020) - TQC | P1.1 Configuration baseline |
| Hiraoka et al. (2021) - DroQ | P1.2 Architecture dropout |
| Abels et al. (2019) - MORL | P1.3 Conditioned network |
| Hayes et al. (2022) - MORL Guide | P1.3 Best practices |
| Gal & Ghahramani (2016) | P1.4 Uncertainty quantification |

---

---

## Contre-Audit / Peer Review

**Date**: 2026-01-22  
**Reviewer**: Expert externe  
**Niveau d'accord avec l'audit**: **95%**

---

### ✅ Validation Globale

Ce rapport d'audit est d'une **très grande qualité**. Il identifie précisément les failles "invisibles" qui transforment souvent un backtest prometteur en échec réel.

---

### 🔴 Accord Total sur les Points Critiques (P0)

Ces points sont des **bloquants absolus**. Déployer sans les corriger serait dangereux.

#### 1. Le "Purge Window" manquant dans le WFO (P3.2)

**Pourquoi cet accord total :**  
C'est le point le plus crucial du rapport. En finance, les données ont une "mémoire" (autocorrélation). Si vous utilisez des features glissantes (ex: Z-Score sur 30 jours) et que vous testez immédiatement après la fin du train, votre modèle "connaît" mathématiquement le début du test set car il était inclus dans la fenêtre glissante de la fin du train set.

**⚠️ Nuance importante :**  
L'audit suggère un gap de 24-48h. **Attention** : ce gap doit être **au moins égal à la taille de votre plus longue fenêtre de feature (lookback window)**. 

| Lookback Feature | Gap Minimum Requis |
|------------------|--------------------|
| 10 jours | 10 jours |
| 30 jours (ex: Z-Score) | 30 jours |
| 64 steps (window_size) | 64 steps |

**Recommandation mise à jour** : `purge_window = max(max_lookback_feature, 48h)`

#### 2. L'absence de Tests sur les Callbacks (P2.3)

**Pourquoi cet accord total :**  
Les callbacks comme `ThreePhaseCurriculum` et `OverfittingGuard` contiennent une logique d'état complexe (transitions de phases, compteurs de patience). Un bug ici est **"silencieux"** : le code ne plante pas, mais l'agent n'apprend pas ce qu'il faut (ex: reste bloqué en phase "Discovery" ou s'arrête trop tôt).

**Impact** : L'absence de tests unitaires sur cette logique est **inacceptable pour la production**.

---

### 🟠 Accord Fort sur les Incohérences de Configuration (P1)

#### 1. Divergence `training.py` vs `WFO` (P2.1)

**Analyse :**  
Avoir un `learning_rate` de `3e-4` par défaut mais de `1e-4` hardcodé dans le script WFO est une **recette pour le désastre**. Cela invalide vos tentatives d'optimisation : vous tunez des hyperparamètres qui ne sont pas ceux utilisés en validation finale.

| Paramètre | training.py (default) | run_full_wfo.py (hardcoded) | Écart |
|-----------|----------------------|----------------------------|-------|
| `learning_rate` | 3e-4 | 1e-4 | **3x** |
| `critic_dropout` | 0.01 | 0.1 | **10x** |
| `batch_size` | 2048 | 512 | **4x** |

**Risque** : Le Dropout passe de 0.01 (standard) à 0.1 (très agressif) dans le WFO **sans justification documentaire**. Cela change radicalement la dynamique de régularisation.

#### 2. Le Modèle de Slippage Linéaire (P1.3)

**Analyse :**  
L'audit a raison de souligner que `slippage = rate × volume` est une simplification excessive.

**Réalité :**  
L'impact de marché suit généralement une **loi en racine carrée** (Square Root Law of Market Impact). Pour des positions plus grandes, le slippage augmente de façon non-linéaire.

**Recommandation :**  
Adopter la formule :

```python
slippage = base_rate × sqrt(volume / average_daily_volume)
```

---

### 🔍 Nuances sur les Recommandations Techniques

#### Ajustement de Sévérité : Learning Rate

| Point | Sévérité Audit | Sévérité Révisée | Justification |
|-------|----------------|------------------|---------------|
| LR 3e-4 vs 1e-4 | Moyen | **Haute** | En crypto, données très bruitées (faible ratio signal/bruit), un LR élevé empêche souvent la convergence fine des Critics |

**Recommandation forte** : S'aligner sur `1e-4` par défaut pour la stabilité.

#### Ajustement de Priorité : Tests E2E

| Point | Priorité Audit | Priorité Révisée | Justification |
|-------|----------------|------------------|---------------|
| Tests E2E WFO | Moyenne | **Basse (v1)** | Coûteux à implémenter. Prioriser d'abord les tests unitaires des Callbacks avant les tests d'intégration complets |

---

### 📊 Synthèse des Risques par Criticité

```
┌────────────────────────────────────────────────────────────────────┐
│                    MATRICE DE RISQUES RÉVISÉE                      │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  IMPACT     │ Critique  │ Majeur    │ Modéré    │ Mineur          │
│  ───────────┼───────────┼───────────┼───────────┼──────────       │
│  Très       │ ❌ Purge  │           │           │                 │
│  Probable   │   Window  │           │           │                 │
│             │           │           │           │                 │
│  Probable   │ ❌ Tests  │ ⚠️ Config │ ⚠️ LR     │                 │
│             │ Callbacks │ divergente│ trop haut │                 │
│             │           │           │           │                 │
│  Possible   │           │ ⚠️ Slip-  │           │                 │
│             │           │ page      │           │                 │
│             │           │ linéaire  │           │                 │
│             │           │           │           │                 │
│  Improbable │           │           │           │ ℹ️ n_critics    │
│             │           │           │           │                 │
└────────────────────────────────────────────────────────────────────┘
```

---

### 🛡️ Focus : OverfittingGuard

Le mécanisme `OverfittingGuard` est **vital**. Comme le montre l'audit, il surveille 5 signaux :

```
┌─────────────────────────────────────────────────────────────────┐
│                    OVERFITTING GUARD SIGNALS                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Signal 1: val_reward_degradation ────────┐                     │
│  Signal 2: train_val_gap ─────────────────┤                     │
│  Signal 3: action_entropy_collapse ───────┼──▶ [DECISION]       │
│  Signal 4: gradient_variance ─────────────┤    ≥2 signaux       │
│  Signal 5: return_autocorrelation ────────┘    = STOP           │
│                                                                  │
│  ⚠️ SANS TESTS : Risque de sélectionner des modèles surentraînés │
└─────────────────────────────────────────────────────────────────┘
```

**Sans tests pour vérifier que ce "chien de garde" aboie au bon moment**, vous risquez de sélectionner des modèles surentraînés.

---

### 📋 Verdict Final Révisé

#### Statut : **GO-WITH-CONDITIONS** ✅⚠️ (Confirmé)

Le verdict de l'audit original est **validé**.

#### Conditions Obligatoires (BLOQUANTES)

**Ne lancez pas d'entraînement coûteux (GPU hours) avant d'avoir :**

| # | Condition | Effort | Impact |
|---|-----------|--------|--------|
| **C1** | Unifié les fichiers de configuration (supprimé les "magic numbers" dans `run_full_wfo.py`) | Faible | **Critique** |
| **C2** | Ajouté la logique de **Purge** dans le découpage des données (gap ≥ max_lookback) | Moyen | **Critique** |
| **C3** | Écrit les tests pour `OverfittingGuardCallbackV2` et `ThreePhaseCurriculumCallback` | Moyen | **Critique** |

#### Checklist Pré-Déploiement

```
[ ] C1: Config unifiée (1 source de vérité)
    └── Supprimer hardcoding dans run_full_wfo.py
    └── Utiliser TrainingConfig partout

[ ] C2: Purge window implémenté
    └── Calculer max_lookback_feature automatiquement
    └── Insérer gap entre train_end et eval_start

[ ] C3: Tests callbacks écrits
    └── test_curriculum_phase_transitions()
    └── test_overfitting_guard_signals()
    └── test_overfitting_guard_multi_signal_logic()
```

---

### 🎯 Actions Prioritaires Réordonnées

| Rang | Action | Effort | Impact | Sprint |
|------|--------|--------|--------|--------|
| **1** | Créer `test_callbacks.py` | Moyen | **Critique** | Sprint 1 |
| **2** | Implémenter purge window | Moyen | **Critique** | Sprint 1 |
| **3** | Unifier config (supprimer magic numbers) | Faible | **Haute** | Sprint 1 |
| **4** | Documenter rationale dropout WFO | Faible | Moyenne | Sprint 1 |
| **5** | Aligner LR par défaut sur 1e-4 | Faible | Moyenne | Sprint 2 |
| **6** | Implémenter slippage sqrt | Moyen | Moyenne | Sprint 2 |
| **7** | Tests E2E WFO 2 segments | Moyen | Moyenne | Sprint 3 |

---

*Audit complété le 2026-01-22*
*Auditeur: Claude Opus 4.5*
*Méthode: Recursive Prompt Architecture v2*

*Contre-audit complété le 2026-01-22*
*Reviewer: Expert externe*
*Niveau de validation: 95%*

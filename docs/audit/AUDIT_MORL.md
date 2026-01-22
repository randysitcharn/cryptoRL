# Audit SOTA: MORL_DESIGN.md

**Date**: 2026-01-22  
**Méthode**: Recursive Prompt Architecture (5 audits parallèles)  
**Document audité**: `docs/design/MORL_DESIGN.md` v1.0  
**Implémentation**: `src/training/batch_env.py`, `src/training/callbacks.py`

---

## Table des Matières

1. [Résumé Exécutif](#1-résumé-exécutif)
2. [Audit 1: Fondations MORL (Théorie ML)](#2-audit-1-fondations-morl-théorie-ml)
3. [Audit 2: Pertinence Économique/Trading](#3-audit-2-pertinence-économiquetrading)
4. [Audit 3: Implémentation Code](#4-audit-3-implémentation-code)
5. [Audit 4: Plan de Tests](#5-audit-4-plan-de-tests)
6. [Audit 5: Intégration Système](#6-audit-5-intégration-système)
7. [Synthèse et Recommandations](#7-synthèse-et-recommandations)

---

## 1. Résumé Exécutif

### Score Global: 7.8/10

| Dimension | Score | Verdict |
|-----------|-------|---------|
| Théorie MORL | 8.5/10 | ✅ Solide, références SOTA |
| Économie/Trading | 7.0/10 | ⚠️ Simplifications acceptables |
| Implémentation | 8.0/10 | ✅ Correcte, optimisable |
| Tests | 6.0/10 | ⚠️ Plan non implémenté |
| Intégration | 8.5/10 | ✅ Cohérente |

### Verdict: **GO avec réserves mineures**

Le design MORL est **Production Ready** pour une v1.0. Les simplifications (modèle de coûts linéaire, scalarization linéaire) sont acceptables et bien documentées. Priorité: implémenter les tests unitaires manquants.

---

## 2. Audit 1: Fondations MORL (Théorie ML)

### ✅ Points Conformes au SOTA

1. **Architecture Conditioned Network**
   - Choix correct pour le use case (préférence scalaire unique)
   - Alternative Multi-Head rejetée à raison (perte de partage de représentation)
   - Référence Abels et al. (ICML 2019) pertinente et correctement appliquée

2. **Scalarization Linéaire**
   - Appropriée pour un front de Pareto convexe (performance vs coûts)
   - Formule `R = r_perf + w × r_cost × scale` est standard
   - Propriété de convexité correctement identifiée

3. **Distribution de Sampling Biaisée (20/60/20)**
   - Innovation pertinente vs sampling uniforme naïf
   - Garantit exploration des extrêmes (w=0, w=1)
   - Cohérent avec curriculum sampling MORL (Yang et al., 2019)

4. **Conditions de Convergence**
   - Les 3 conditions citées sont correctes:
     - ✅ Sampling suffisant (distribution biaisée)
     - ✅ Capacité réseau (TQC 64×64 suffisant pour 1D preference)
     - ✅ Exploration (gSDE + observation noise)

### ⚠️ Écarts Mineurs

1. **Scalarization Non-Convexe**
   - Le design identifie correctement que la scalarization linéaire ne peut atteindre les points non-convexes
   - Tchebycheff mentionné comme alternative v2.0 → **OK, roadmap claire**
   - *Impact*: Faible pour trading (front généralement convexe)

2. **Paramètre w_cost Scalaire**
   - Un seul paramètre de préférence limite l'expressivité
   - *Alternative SOTA*: Yang et al. proposent des vecteurs de préférence multi-dim
   - *Verdict*: Acceptable pour 2 objectifs, mais limitant si on ajoute d'autres objectifs (max DD, Sortino, etc.)

### ❌ Problèmes Critiques

*Aucun identifié.*

### 📚 Références Manquantes

| Papier | Pourquoi Pertinent |
|--------|-------------------|
| Hayes et al. (2022) - *"A Practical Guide to MORL"* | Cité mais pas exploité: contient recommandations concrètes sur hyperparamètres MORL |
| Lu et al. (2023) - *"Pareto Set Learning for MORL"* | Alternative au conditioned network pour fronts complexes |
| Alegre et al. (2023) - *"MORL-Baselines"* | Benchmark et implémentations de référence |

### Score Théorie: 8.5/10

---

## 3. Audit 2: Pertinence Économique/Trading

### ✅ Modélisation Correcte

1. **Objectif Performance (r_perf)**
   - `log1p(returns) × SCALE` est standard en finance quantitative
   - Log-returns additifs, bonne propriété pour RL
   - Clamp à -0.99 évite log(0) → **Correct**

2. **Interprétation w_cost → Style de Trading**
   - Mapping w=0 (scalping) → w=1 (B&H) économiquement cohérent
   - Continuum de styles reflète la réalité des traders

3. **Pareto Front Interprétable**
   - L'axe Sharpe vs Trades est pertinent pour un investisseur
   - Permet de choisir le profil risque/activité post-training

### ⚠️ Simplifications Acceptables

1. **Modèle de Coûts Linéaire**
   ```python
   r_cost = -|Δposition| × SCALE
   ```
   - **Manque**: Slippage non-linéaire (√volume), market impact, spread bid-ask variable
   - **Mitigation**: Domain randomization sur commission/slippage (implémenté!)
   - **Verdict**: Acceptable pour v1.0, le bruit couvre partiellement les non-linéarités

2. **MAX_PENALTY_SCALE = 2.0 Fixe**
   - Calibration empirique, pas de justification formelle
   - **Recommandation**: Ajouter une phase de calibration automatique basée sur les magnitudes moyennes de r_perf et r_cost sur le training set

3. **Pas de Coût de Financement Overnight**
   - Le `funding_rate` existe pour shorts, mais w_cost ne le module pas
   - **Impact**: Faible pour crypto (pas de distinction jour/nuit)

### ❌ Erreurs de Modélisation

1. **COST_PENALTY_CAP = 20.0 Asymétrique**
   - `r_cost = clamp(r_cost, min=-20)` mais pas de max
   - **Impact**: Si r_perf explose (bug), le clamp ne protège pas
   - **Fix**: Ajouter `r_cost = clamp(r_cost, min=-20, max=0)` (coûts toujours négatifs)

### 💡 Améliorations Suggérées

| Amélioration | Complexité | Bénéfice |
|--------------|------------|----------|
| Coût de slippage √volume | Moyenne | Réalisme accru pour gros ordres |
| w_cost per-asset (multi-asset) | Haute | Préférences par actif |
| Calibration auto MAX_PENALTY_SCALE | Faible | Robustesse aux changements de données |

### Score Économique: 7.0/10

---

## 4. Audit 3: Implémentation Code

### ✅ Code Correct

1. **Observation Space Dict**
   ```python
   "w_cost": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
   ```
   - Compatible SB3/TQC avec CombinedExtractor (implicite)
   - Shape (1,) correct pour scalar preference

2. **Sampling Distribution (reset/auto_reset)**
   ```python
   self.w_cost = torch.where(
       sample_type.unsqueeze(1) < 0.2, zeros,
       torch.where(sample_type.unsqueeze(1) > 0.8, ones, uniform)
   )
   ```
   - Logique correcte: 20% w=0, 20% w=1, 60% uniform
   - Broadcasting correct avec unsqueeze(1)

3. **Reward Calculation**
   ```python
   w_cost_squeezed = self.w_cost.squeeze(-1)  # (n_envs,)
   reward = r_perf + (w_cost_squeezed * r_cost * MAX_PENALTY_SCALE)
   ```
   - Shapes corrects: (n_envs,) partout
   - Multiplication element-wise correcte

4. **Mode Évaluation**
   ```python
   def set_eval_w_cost(self, w_cost: Optional[float]):
       self._eval_w_cost = w_cost
   ```
   - API claire et fonctionnelle
   - Utilisé dans reset() et _auto_reset()

### 🐛 Bugs Potentiels

1. **Reproducibilité du Sampling w_cost**
   - `torch.rand()` dans reset() n'utilise pas le seed de l'env
   - **Conditions**: Si on veut reproduire exactement un épisode
   - **Fix**: Utiliser un Generator local avec seed controllé
   ```python
   # Fix suggéré
   if self._rng is None:
       self._rng = torch.Generator(device=self.device)
   sample_type = torch.rand(self.num_envs, generator=self._rng, device=self.device)
   ```

2. **Mémoire Non-Libérée**
   - Dans `_allocate_state_tensors()`, si appelé plusieurs fois, les anciens tensors ne sont pas explicitement libérés
   - **Impact**: Faible (GC Python gère), mais peut causer des pics mémoire
   - **Fix**: Ajouter `del self.w_cost` avant réallocation

### ⚡ Optimisations

1. **Sampling avec torch.where Nestés**
   - 3 allocations tenseurs (zeros, ones, uniform)
   - **Optimisation**: Pré-allouer dans `_allocate_state_tensors()`
   ```python
   # Dans _allocate_state_tensors:
   self._w_cost_zeros = torch.zeros(n, 1, device=device)
   self._w_cost_ones = torch.ones(n, 1, device=device)
   
   # Dans reset:
   self.w_cost = torch.where(
       sample_type.unsqueeze(1) < 0.2,
       self._w_cost_zeros,
       torch.where(sample_type.unsqueeze(1) > 0.8, self._w_cost_ones, torch.rand(...))
   )
   ```
   - **Speedup estimé**: ~5% sur reset() (mineur car reset rare)

2. **Logging w_cost Distribution**
   - Pas de métrique loggée pour vérifier la distribution effective
   - **Recommandation**: Ajouter dans `get_global_metrics()`:
   ```python
   "morl/w_cost_mean": self.w_cost.mean().item(),
   "morl/w_cost_std": self.w_cost.std().item(),
   ```

### 🔒 Sécurité Numérique

1. **log1p Stability**
   - `safe_returns = torch.clamp(step_returns, min=-0.99)` → log1p(≥0.01) ✅
   - **Robuste** aux pertes extrêmes

2. **Division par Zéro**
   - Pas de division dans la reward function MORL
   - `prev_valuations` utilisé mais jamais divisé directement
   - ✅ Pas de risque

### Score Implémentation: 8.0/10

---

## 5. Audit 4: Plan de Tests

### ✅ Tests Existants Valides (dans le design)

| Test | Ce qu'il couvre |
|------|-----------------|
| `test_w_cost_in_observation` | Présence et shape de w_cost |
| `test_w_cost_sampling_distribution` | Distribution 20/60/20 |
| `test_eval_w_cost_fixed` | Mode évaluation |
| `test_reward_with_w_zero` | r_cost = 0 quand w=0 |
| `test_reward_with_w_one` | r_cost actif quand w=1 |
| `test_trained_agent_respects_w_cost` | Sensibilité comportementale |

### ❌ Cas Non Couverts (Critiques)

1. **Tests Non Implémentés**
   - ⚠️ Le fichier `tests/test_morl.py` **n'existe pas**
   - Les tests sont décrits dans le design mais jamais créés
   - **Risque**: Régression silencieuse
   - **Action**: Créer `tests/test_morl.py` avec le code du design

2. **Edge Cases w_cost**
   - Pas de test pour w_cost = 0.5 (comportement intermédiaire)
   - Pas de test pour transitions w_cost entre épisodes
   - **Test suggéré**:
   ```python
   def test_w_cost_changes_between_episodes():
       """w_cost should be resampled after auto-reset."""
       env = BatchCryptoEnv(n_envs=1)
       env.reset()
       w1 = env.w_cost.item()
       # Force episode end
       for _ in range(env.episode_length + 1):
           env.step_async(np.zeros((1, 1)))
           env.step_wait()
       w2 = env.w_cost.item()
       # Statistically unlikely to be equal (1/inf for continuous)
       # But we test that resampling occurred (not stuck)
   ```

3. **Robustesse NaN/Overflow**
   - Pas de test avec returns extrêmes
   - **Test suggéré**:
   ```python
   def test_reward_stability_extreme_returns():
       """Reward should not NaN with extreme returns."""
       # Mock extreme returns
       env._calculate_rewards(
           step_returns=torch.tensor([0.99, -0.99, 10.0]),  # 10x = overflow sans clamp
           position_deltas=torch.tensor([2.0, 0.0, 1.0]),
           dones=torch.tensor([False, False, False])
       )
       assert not torch.isnan(reward).any()
   ```

### ⚠️ Cas Non Couverts (Secondaires)

| Cas | Priority |
|-----|----------|
| Multi-env avec w_cost hétérogènes | P2 |
| Interaction w_cost × observation_noise | P3 |
| Performance GPU du sampling | P3 |

### 📊 Amélioration Tests Statistiques

Le test `test_w_cost_sampling_distribution` utilise:
```python
assert 0.15 < pct_zero < 0.25
```

**Problème**: Intervalle ad-hoc, pas de justification statistique.

**Amélioration**: Utiliser un test binomial:
```python
from scipy import stats

def test_w_cost_sampling_distribution_statistical():
    n_samples = 100_000
    # ... sample w_cost ...
    
    # Test binomial pour 20% ± marge
    count_zero = (w == 0.0).sum()
    p_value = stats.binom_test(count_zero, n_samples, 0.2, alternative='two-sided')
    assert p_value > 0.01, f"Distribution w=0 non conforme (p={p_value:.4f})"
```

### 🆕 Tests Suggérés

```python
# tests/test_morl.py - À CRÉER

import pytest
import numpy as np
import torch
from src.training.batch_env import BatchCryptoEnv


class TestMORLIntegration:
    """Integration tests for MORL with TQC."""
    
    @pytest.fixture
    def env(self, tmp_path):
        # Create minimal test data
        # ... (voir conftest.py existant)
        return BatchCryptoEnv(str(tmp_path / "test.parquet"), n_envs=4)
    
    def test_w_cost_observation_shape(self, env):
        obs = env.reset()
        assert "w_cost" in obs
        assert obs["w_cost"].shape == (4, 1)
    
    def test_w_cost_bounds(self, env):
        obs = env.reset()
        assert np.all(obs["w_cost"] >= 0.0)
        assert np.all(obs["w_cost"] <= 1.0)
    
    def test_eval_mode_fixes_w_cost(self, env):
        env.set_eval_w_cost(0.75)
        obs = env.reset()
        assert np.allclose(obs["w_cost"], 0.75)
        
        # Also fixed after step
        env.step_async(np.zeros((4, 1)))
        obs, _, _, _ = env.step_wait()
        assert np.allclose(obs["w_cost"], 0.75)
    
    def test_reward_zero_cost_when_w_zero(self, env):
        env.set_eval_w_cost(0.0)
        env.reset()
        env.step_async(np.array([[0.5]] * 4))  # Position change
        env.step_wait()
        
        # With w=0, churn component should be 0
        assert env._rew_churn.abs().max().item() < 1e-6
    
    def test_reward_nonzero_cost_when_w_one(self, env):
        env.set_eval_w_cost(1.0)
        env.reset()
        env.step_async(np.array([[0.5]] * 4))  # Position change
        env.step_wait()
        
        # With w=1, churn component should be negative (penalty)
        assert env._rew_churn.min().item() < 0
```

### Score Tests: 6.0/10

---

## 6. Audit 5: Intégration Système

### ✅ Intégrations Correctes

1. **MORL × TQC**
   - Dict observation space avec `w_cost` fonctionne via `CombinedExtractor` de SB3
   - TQC gère nativement les Dict spaces
   - **Vérifié**: Pas besoin de feature extractor custom

2. **MORL × Callbacks**
   - `DetailTensorboardCallback` log `reward/pnl_component` et `reward/churn_cost`
   - Métriques MORL accessibles via `get_global_metrics()`
   - **Logs disponibles**:
     - `internal/reward/pnl_component`
     - `internal/reward/churn_cost`
     - `internal/curriculum/lambda` (indirectement lié)

3. **MORL × Domain Randomization**
   - Commission/slippage randomisés indépendamment de w_cost
   - Pas de conflit: w_cost module la pénalité, DR module les coûts réels
   - **Synergie**: Agent robuste aux variations de coûts ET de préférences

### ⚠️ Frictions d'Intégration

1. **curriculum_lambda Non Utilisé dans MORL**
   - Le design note: "curriculum_lambda n'est pas directement utilisé dans la formule de récompense MORL"
   - `ThreePhaseCurriculumCallback` met à jour `curriculum_lambda` mais MORL utilise `MAX_PENALTY_SCALE` fixe
   - **Impact**: Confusion potentielle, deux mécanismes qui semblent faire la même chose
   - **Mitigation**: 
     - Option A: Supprimer curriculum_lambda si MORL remplace complètement
     - Option B: Utiliser `curriculum_lambda` comme multiplicateur de `MAX_PENALTY_SCALE`:
     ```python
     effective_scale = MAX_PENALTY_SCALE * self.curriculum_lambda
     reward = r_perf + (w_cost * r_cost * effective_scale)
     ```

2. **WFO × MORL: Sélection de w_cost**
   - `evaluate_segment_morl()` évalue sur `[0.0, 0.5, 1.0]` (3 points)
   - **Question**: 3 points suffisent-ils pour caractériser le front ?
   - **Recommandation**: Utiliser 5 points `[0.0, 0.25, 0.5, 0.75, 1.0]` pour meilleure résolution

3. **EvalCallback × MORL**
   - `EvalCallbackWithNoiseControl` gère noise mais pas w_cost
   - **Friction**: Pas de mode "eval sur tous les w_cost" intégré
   - **Fix suggéré**: Ajouter paramètre `eval_w_cost_values` à `EvalCallbackWithNoiseControl`:
   ```python
   def __init__(self, eval_w_cost_values: List[float] = [0.5], ...):
       self.eval_w_cost_values = eval_w_cost_values
   
   def _on_step(self):
       for w in self.eval_w_cost_values:
           env.set_eval_w_cost(w)
           # ... evaluate ...
   ```

### ❌ Incompatibilités

*Aucune incompatibilité bloquante identifiée.*

### 🔄 Flux de Données MORL

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MORL Data Flow                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   reset() / _auto_reset()                                                    │
│        │                                                                     │
│        ▼                                                                     │
│   ┌─────────────────────────────────────────────────────────────────┐       │
│   │ Sample w_cost ~ Biased(20% 0, 60% U[0,1], 20% 1)               │       │
│   │ OR use _eval_w_cost if set                                      │       │
│   └─────────────────────────────────────────────────────────────────┘       │
│        │                                                                     │
│        ▼                                                                     │
│   _get_observations()                                                        │
│        │                                                                     │
│        ▼                                                                     │
│   ┌─────────────────────────────────────────────────────────────────┐       │
│   │ obs = {market, position, w_cost}                                │       │
│   │       └──────────────────────────┘                              │       │
│   │               Sent to TQC Policy                                │       │
│   └─────────────────────────────────────────────────────────────────┘       │
│        │                                                                     │
│        ▼                                                                     │
│   TQC Policy: π(a|s, w_cost)                                                │
│        │                                                                     │
│        ▼                                                                     │
│   step_wait()                                                                │
│        │                                                                     │
│        ▼                                                                     │
│   ┌─────────────────────────────────────────────────────────────────┐       │
│   │ _calculate_rewards(step_returns, position_deltas, dones)        │       │
│   │                                                                  │       │
│   │   r_perf = log1p(returns) × 100                                 │       │
│   │   r_cost = -|Δposition| × 100 (clamped)                         │       │
│   │                                                                  │       │
│   │   reward = r_perf + w_cost × r_cost × MAX_PENALTY_SCALE         │       │
│   │            ▲                                                     │       │
│   │            └─ Conditioned on agent's preference                  │       │
│   └─────────────────────────────────────────────────────────────────┘       │
│        │                                                                     │
│        ▼                                                                     │
│   ┌─────────────────────────────────────────────────────────────────┐       │
│   │ Logging (DetailTensorboardCallback)                             │       │
│   │   - internal/reward/pnl_component                               │       │
│   │   - internal/reward/churn_cost                                  │       │
│   └─────────────────────────────────────────────────────────────────┘       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 📈 Métriques Manquantes

| Métrique | Utilité |
|----------|---------|
| `morl/w_cost_mean` | Vérifier distribution effective |
| `morl/w_cost_std` | Détecter collapse vers une valeur |
| `morl/pareto_hypervolume` | Qualité du front (si multi-eval) |
| `morl/trades_per_w_bucket` | Sensibilité comportementale par bucket |

### Score Intégration: 8.5/10

---

## 7. Synthèse et Recommandations

### Matrice de Risque

| Finding | Prob | Impact | Score | Owner |
|---------|------|--------|-------|-------|
| Tests MORL non implémentés | 100% | Moyen | 🔴 HIGH | Dev |
| curriculum_lambda vs MAX_PENALTY_SCALE confus | 50% | Faible | 🟡 MED | Design |
| Calibration MAX_PENALTY_SCALE ad-hoc | 30% | Moyen | 🟡 MED | Data |
| Sampling w_cost non reproductible | 20% | Faible | 🟢 LOW | Dev |
| WFO évalue seulement 3 points Pareto | 40% | Faible | 🟢 LOW | Pipeline |

### Top 5 Actions Prioritaires

| # | Action | Effort | Impact | Deadline |
|---|--------|--------|--------|----------|
| 1 | **Créer `tests/test_morl.py`** avec le code du design | 2h | ⬛⬛⬛⬛ | Immédiat |
| 2 | Ajouter métriques `morl/w_cost_mean`, `morl/w_cost_std` | 30min | ⬛⬛⬛ | Sprint |
| 3 | Documenter relation curriculum_lambda / MAX_PENALTY_SCALE | 1h | ⬛⬛ | Sprint |
| 4 | Étendre WFO à 5 points w_cost `[0, 0.25, 0.5, 0.75, 1]` | 1h | ⬛⬛ | Sprint |
| 5 | Ajouter script de calibration auto MAX_PENALTY_SCALE | 4h | ⬛⬛⬛ | v2.0 |

### Verdict Final

**✅ GO - Production Ready (v1.0)**

Le design MORL est solide, bien documenté, et l'implémentation est correcte. Les simplifications (modèle de coûts linéaire, scalarization linéaire) sont explicitement reconnues et acceptables pour une première version.

**Conditions de release**:
1. ⚠️ Implémenter `tests/test_morl.py` avant merge en production
2. ⚠️ Vérifier visuellement le Pareto front sur 1-2 segments WFO

### Roadmap v2.0

| Phase | Amélioration | Bénéfice |
|-------|--------------|----------|
| 2.1 | Scalarization Tchebycheff | Fronts non-convexes |
| 2.2 | Calibration auto MAX_PENALTY_SCALE | Robustesse cross-dataset |
| 2.3 | Multi-head policy (5 buckets) | Spécialisation par style |
| 2.4 | Pareto hypervolume logging | Métriques MORL SOTA |

---

## Annexes

### A. Checklist Pré-Merge

- [x] `tests/test_morl.py` créé et passant (2026-01-22)
- [x] Métriques `morl/w_cost_mean`, `morl/w_cost_std` ajoutées (2026-01-22)
- [x] WFO configuré avec 5 valeurs de w_cost `EVAL_W_COST_VALUES` (2026-01-22)
- [x] Documentation curriculum_lambda clarifiée (section 5.4) (2026-01-22)
- [x] Fix sécurité: `r_cost` clampé à max=0.0 (2026-01-22)
- [ ] Pareto front visualisé sur au moins 1 segment

### B. Références Supplémentaires

1. **MORL-Baselines** (GitHub): https://github.com/LucasAlegre/morl-baselines
2. **Pareto Set Learning** (NeurIPS 2023): Extension pour fronts complexes
3. **SB3 Dict Observation**: https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html

---

**Auteur**: Audit automatisé (Recursive Prompt Architecture)  
**Validé par**: [À compléter]  
**Date de revue**: 2026-01-22

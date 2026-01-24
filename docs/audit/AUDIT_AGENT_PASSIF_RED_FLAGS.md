# Audit "Deep Dive" - Agent Passif (Attribution 0.0, Action Mean 0.0)

**Date:** 2026-01-24  
**Mission:** Identifier les causes de la passivité de l'agent (Attribution 0.0, Action Mean 0.0)

---

## 🔴 RED FLAG #1 : Déséquilibre Massif Reward/Cost (CRITIQUE)

### Localisation
`src/training/batch_env.py`, ligne 424-507 (`_calculate_rewards`)

### Problème Identifié

**Calcul du coût d'un changement de position de 0 → 1 :**

```python
# Ligne 477
r_cost = -position_deltas * SCALE  # SCALE = 10.0
# Pour delta = 1.0 : r_cost = -1.0 * 10 = -10.0

# Ligne 481 : Clampé à COST_PENALTY_CAP = 2.0
r_cost = torch.clamp(r_cost, min=-COST_PENALTY_CAP, max=0.0)  # → -2.0

# Ligne 493 : Application MORL avec w_cost=1 et MAX_PENALTY_SCALE=0.4
reward = r_perf + (w_cost_squeezed * r_cost * MAX_PENALTY_SCALE)
# → reward = r_perf + (1.0 * -2.0 * 0.4) = r_perf - 0.8
```

**Comparaison avec un return moyen de 0.1% (0.001) :**

```python
# Ligne 469
r_perf = torch.log1p(safe_returns) * SCALE
# Pour return = 0.001 : r_perf = log1p(0.001) * 10 ≈ 0.00998 ≈ 0.01
```

### Impact

**Ratio Pénalité/Gain = -0.8 / 0.01 = -80x**

L'agent est **instantanément puni 80 fois plus fort** qu'il ne peut gagner en moyenne. Même avec `w_cost=0` (scalping mode), si l'agent explore et que certains envs ont `w_cost>0`, il subit des pénalités massives.

### Preuve dans le Code

```python:424:507:src/training/batch_env.py
SCALE = 10.0
MAX_PENALTY_SCALE = 0.4
COST_PENALTY_CAP = 2.0

# Pour un changement de position de 0 à 1 :
r_cost = -1.0 * 10.0 = -10.0  # Clampé à -2.0
penalty = -2.0 * 0.4 = -0.8   # Avec w_cost=1

# Pour un return de 0.1% :
r_perf = log1p(0.001) * 10 ≈ 0.01

# Ratio = -0.8 / 0.01 = -80x (PÉNALITÉ 80x PLUS FORTE)
```

### Recommandation

**Option A (Recommandée) :** Réduire `COST_PENALTY_CAP` de 2.0 à 0.1
```python
COST_PENALTY_CAP = 0.1  # Au lieu de 2.0
# Nouvelle pénalité max : -0.1 * 0.4 = -0.04
# Ratio = -0.04 / 0.01 = -4x (acceptable pour exploration)
```

**Option B :** Réduire `MAX_PENALTY_SCALE` de 0.4 à 0.05
```python
MAX_PENALTY_SCALE = 0.05  # Au lieu de 0.4
# Pénalité max : -2.0 * 0.05 = -0.1
# Ratio = -0.1 / 0.01 = -10x (encore élevé mais mieux)
```

**Option C (Hybride) :** Combiner les deux
```python
COST_PENALTY_CAP = 0.2
MAX_PENALTY_SCALE = 0.2
# Pénalité max : -0.2 * 0.2 = -0.04
# Ratio = -0.04 / 0.01 = -4x
```

---

## 🟡 RED FLAG #2 : curriculum_lambda Non Utilisé (Incohérence)

### Localisation
`src/training/batch_env.py`, ligne 1129-1152 (`set_progress`)

### Problème Identifié

`curriculum_lambda` est calculé et mis à jour mais **jamais utilisé dans `_calculate_rewards`**.

```python
# Ligne 1143-1152 : curriculum_lambda est mis à jour
if self.progress <= 0.15:
    self.curriculum_lambda = 0.0
elif self.progress <= 0.75:
    self.curriculum_lambda = 0.4 * phase_progress
else:
    self.curriculum_lambda = 0.4

# MAIS ligne 493 : curriculum_lambda n'est PAS utilisé
reward = r_perf + (w_cost_squeezed * r_cost * MAX_PENALTY_SCALE)
# Devrait être :
# reward = r_perf + (w_cost_squeezed * r_cost * MAX_PENALTY_SCALE * self.curriculum_lambda)
```

### Impact

- Les coûts sont appliqués **immédiatement** dès le début de l'entraînement (même si `w_cost=0` dans certains envs)
- Le curriculum learning ne fonctionne pas comme prévu
- L'agent subit des pénalités maximales dès le début, même en phase d'exploration

### Recommandation

**Intégrer `curriculum_lambda` dans le calcul de reward :**

```python
# Ligne 493, modifier en :
effective_penalty_scale = MAX_PENALTY_SCALE * self.curriculum_lambda
reward = r_perf + (w_cost_squeezed * r_cost * effective_penalty_scale)
```

Cela permettrait :
- Phase 1 (0-15%) : `curriculum_lambda=0.0` → Pas de pénalité (exploration pure)
- Phase 2 (15-75%) : `curriculum_lambda` monte progressivement → Introduction graduelle des coûts
- Phase 3 (75-100%) : `curriculum_lambda=0.4` → Pénalités complètes

---

## 🟢 VÉRIFICATION #1 : INPUT SPLIT (OK)

### Localisation
`src/models/rl_adapter.py`, ligne 100-107 et 320-328

### Analyse

**Calcul de `mae_input_dim` :**
```python
# Ligne 103
if self.use_film:
    self.mae_input_dim = self.n_features - HMM_CONTEXT_SIZE  # 43 - 5 = 38
```

**Slicing dans `forward` :**
```python
# Ligne 323
mae_obs = market_obs[:, :, :-HMM_CONTEXT_SIZE]  # (B, 64, 38) ✓
hmm_context = market_obs[:, -1, -HMM_CONTEXT_SIZE:].float()  # (B, 5) ✓
```

**Verdict :** ✅ Le slicing est **robuste** et cohérent. `mae_input_dim` est calculé correctement pour matcher les poids pré-entraînés.

---

## 🟢 VÉRIFICATION #2 : CONTRÔLE POLICY (OK)

### Localisation
`src/models/tqc_dropout_policy.py`

### Analyse

**1. Spectral Normalization sur Actor :**
```python
# Ligne 340
use_spectral_norm_actor: bool = False,  # Default False (conservative)
```
✅ **OK** : `spectral_norm` n'est **pas** appliqué à l'Actor par défaut (configurable via `use_spectral_norm_actor`).

**2. Valeur par défaut de `log_std_init` :**
```python
# Ligne 131 (DropoutActor)
log_std_init: float = -1.0,  # FIX: Hardcoded -1 for larger positions (was -3)

# Ligne 342 (TQCDropoutPolicy)
log_std_init: float = -1.0,  # FIX: -1 gives std≈0.37 (vs SB3 default -3 giving std≈0.05)

# Ligne 91 (training.py)
log_std_init: float = -1.0  # FIX: Increased init exploration
```
✅ **OK** : `log_std_init = -1.0` donne `std ≈ 0.37` (vs `-3.0` → `std ≈ 0.05`), ce qui est **correct** pour l'exploration.

**Verdict :** ✅ Aucun problème détecté. La policy est correctement configurée pour l'exploration.

---

## 🟢 VÉRIFICATION #3 : FREEZE STATUS (OK)

### Localisation
`src/config/training.py`, ligne 68  
`src/models/rl_adapter.py`, ligne 60

### Analyse

```python
# training.py ligne 68
freeze_encoder: bool = True  # Par défaut

# rl_adapter.py ligne 60
freeze_encoder: bool = True,  # Paramètre par défaut
```

**Verdict :** ✅ `freeze_encoder = True` par défaut, ce qui est **attendu** pour préserver les représentations pré-entraînées. Ce n'est **pas** la cause de la passivité.

---

## 📊 RÉSUMÉ DES RED FLAGS

| # | Sévérité | Localisation | Problème | Impact |
|---|-----------|--------------|----------|--------|
| **1** | 🔴 **CRITIQUE** | `batch_env.py:477-493` | Déséquilibre Reward/Cost (-80x) | Agent puni 80x plus fort qu'il ne peut gagner → **Passivité totale** |
| **2** | 🟡 **MOYEN** | `batch_env.py:493` | `curriculum_lambda` non utilisé | Pénalités appliquées dès le début, pas de phase d'exploration |

### Actions Recommandées (Priorité)

1. **URGENT** : Réduire `COST_PENALTY_CAP` de 2.0 à 0.1 dans `batch_env.py:462`
2. **URGENT** : Intégrer `curriculum_lambda` dans le calcul de reward (ligne 493)
3. **Optionnel** : Ajuster `MAX_PENALTY_SCALE` si nécessaire après test

### Test de Validation

Après corrections, vérifier dans TensorBoard :
- `reward/pnl_component` : Devrait être positif en moyenne
- `reward/churn_cost` : Devrait être négatif mais **proportionnel** à `r_perf`
- `action_mean` : Devrait sortir de 0.0 après quelques milliers de steps
- `curriculum/lambda` : Devrait être 0.0 en début d'entraînement

---

## 🔍 CALCULS DÉTAILLÉS (RED FLAG #1)

### Scénario : Agent change de position de 0 à 1

**Inputs :**
- `position_deltas = 1.0`
- `step_returns = 0.001` (0.1% return moyen)
- `w_cost = 1.0` (worst case, B&H mode)
- `SCALE = 10.0`
- `MAX_PENALTY_SCALE = 0.4`
- `COST_PENALTY_CAP = 2.0`

**Calculs :**

```python
# 1. Performance reward
r_perf = log1p(0.001) * 10.0
r_perf = 0.000998 * 10.0 ≈ 0.01

# 2. Cost penalty
r_cost = -1.0 * 10.0 = -10.0
r_cost = clamp(-10.0, min=-2.0, max=0.0) = -2.0

# 3. MORL scalarization
penalty = 1.0 * (-2.0) * 0.4 = -0.8

# 4. Total reward
reward = 0.01 + (-0.8) = -0.79
```

**Résultat :** L'agent reçoit une récompense de **-0.79** pour un changement de position, même avec un return positif de 0.1%.

**Ratio Pénalité/Gain :** `-0.8 / 0.01 = -80x`

### Scénario : Agent reste en cash (position = 0)

**Inputs :**
- `position_deltas = 0.0`
- `step_returns = 0.001`
- `w_cost = 1.0`

**Calculs :**

```python
r_perf = 0.01
r_cost = 0.0
penalty = 0.0
reward = 0.01 + 0.0 = 0.01  # POSITIF !
```

**Résultat :** L'agent reçoit une récompense de **+0.01** en restant en cash.

**Conclusion :** L'agent apprend que **rester en cash (reward = +0.01) est meilleur que trader (reward = -0.79)**. C'est la cause directe de la passivité.

---

## 🎯 CORRECTIONS PROPOSÉES

### Correction #1 : Réduire COST_PENALTY_CAP

```python
# batch_env.py ligne 462
COST_PENALTY_CAP = 0.1  # Au lieu de 2.0
```

**Nouveau calcul :**
```python
r_cost = clamp(-10.0, min=-0.1, max=0.0) = -0.1
penalty = 1.0 * (-0.1) * 0.4 = -0.04
reward = 0.01 + (-0.04) = -0.03
```

**Ratio :** `-0.04 / 0.01 = -4x` (acceptable pour exploration)

### Correction #2 : Intégrer curriculum_lambda

```python
# batch_env.py ligne 493
effective_penalty_scale = MAX_PENALTY_SCALE * self.curriculum_lambda
reward = r_perf + (w_cost_squeezed * r_cost * effective_penalty_scale)
```

**Phase 1 (0-15%) :** `curriculum_lambda=0.0` → Pas de pénalité
**Phase 2 (15-75%) :** `curriculum_lambda` monte progressivement
**Phase 3 (75-100%) :** `curriculum_lambda=0.4` → Pénalités complètes

---

**Fin du rapport d'audit**

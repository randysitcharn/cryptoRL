# Audit: Positions Trop Petites

**Date:** 2026-01-22
**Statut:** Investigation terminée
**Auteur:** Agent Codeur

---

## Problème Observé

Le modèle TQC output des positions entre **±0.1 et ±0.3** malgré un action space de **[-1, 1]**.

Symptômes:
- Positions moyennes: ±0.1 à ±0.2
- Drawdown: 92-96%
- Le modèle "apprend" à ne pas trader

---

## Hypothèse Testée: "Le Piège du Volatility Scaling"

**Question:** Est-ce que le `vol_scalar` écrase mathématiquement les actions brutes?

### Mécanisme du Vol Scaling

```python
# batch_env.py lignes 644-655
min_vol = target_volatility / max_leverage  # 0.05 / 2.0 = 0.025
current_vol = sqrt(ema_vars)
clipped_vol = clamp(current_vol, min=min_vol)
vol_scalar = clamp(target_vol / clipped_vol, min=0.1, max=max_leverage)
effective_action = raw_action * vol_scalar
```

### Résultats des Tests

| Test | raw_action | current_vol | vol_scalar | effective | position |
|------|------------|-------------|------------|-----------|----------|
| Grande action | 0.80 | 0.048 → 0.026 | 1.0 → 1.8 | 0.8 → 1.45 | **1.0** ✓ |
| Petite action | 0.10 | 0.048 → 0.026 | 1.0 → 1.8 | 0.1 → 0.18 | **0.1-0.2** |

### Conclusion de l'Hypothèse

**INVALIDÉE** ❌

Le vol_scalar **AMPLIFIE** les actions, il ne les écrase pas:
- `current_vol` (~3-5%) est **égal ou inférieur** au `target_vol` (5%)
- `vol_scalar = target/current = 0.05/0.03 = 1.5-2.0` (amplification)
- Avec `raw_action=0.8`, on obtient `effective=1.2` (clippé à 1.0)

---

## Audit de la Reward Function

### Structure du Reward (batch_env.py)

```python
MAX_PENALTY_SCALE = 0.0  # Churn désactivé
SCALE = 100.0

# Performance (PnL)
r_perf = log1p(step_returns) * SCALE

# Coût (turnover) - DÉSACTIVÉ
r_cost = -position_deltas * SCALE * MAX_PENALTY_SCALE  # = 0

# Reward final
reward = r_perf  # Pure PnL
```

### Constats

- ✅ **Pas de pénalité action²** cachée
- ✅ **Churn penalty désactivé** (`MAX_PENALTY_SCALE = 0`)
- ✅ **Commission raisonnable** (0.02-0.08%)
- ✅ **Reward = PnL pur** (log returns × 100)

---

## Configuration Vérifiée

| Paramètre | Valeur | Fichier | Impact |
|-----------|--------|---------|--------|
| `target_volatility` | 0.05 (5%) | training.py:44 | Raisonnable pour crypto |
| `max_leverage` | 2.0 | training.py:46 | Scalar max = 2x |
| `commission` | 0.0002 (0.02%) | training.py:32 | Très bas |
| `action_discretization` | 0.1 | training.py:41 | Steps de 10% |
| `log_std_init` | -1.0 | training.py:91 | std initial ≈ 0.37 |
| `ent_coef` | "auto_1.0" | training.py:77 | Haute entropie |

---

## Diagnostic Final

### Root Cause Identifiée

**Le modèle APPREND que petites positions = moins de pertes.**

Avec 92-96% de drawdown, le signal de reward est:
- **Trading** → Pertes → Reward négatif
- **Ne pas trader** → Moins de pertes → Reward moins négatif

Le modèle fait exactement ce qu'on lui demande: **minimiser les pertes**.

### Pourquoi log_std_init=-1 n'a pas suffi?

1. `log_std_init=-1` donne `std ≈ 0.37` au départ
2. Les actions initiales sont dans `[-0.37, +0.37]`
3. Le critic apprend rapidement que `|action| > 0.1` → pertes
4. L'actor suit le critic → collapse vers petites actions
5. L'entropie reste élevée mais centrée autour de 0

---

## Solutions Possibles

### Option A: Reward Shaping - Bonus pour Trading
```python
# Ajouter un bonus pour positions non-nulles
trading_bonus = abs(position_pct) * BONUS_SCALE
reward = r_perf + trading_bonus
```
**Risque:** Peut encourager le trading pour le trading (overtrading)

### Option B: Pénalité d'Inaction
```python
# Pénaliser les positions proches de zéro
inaction_penalty = -exp(-abs(position_pct) * 10) * PENALTY_SCALE
reward = r_perf + inaction_penalty
```
**Risque:** Force le trading même quand inapproprié

### Option C: Minimum Position Constraint
```python
# Forcer une taille minimum de position
min_position = 0.3
if abs(effective_action) < min_position:
    effective_action = sign(effective_action) * min_position
```
**Risque:** Perte de granularité, peut amplifier les pertes

### Option D: Revoir les Features/Données
- Vérifier que les features ont un pouvoir prédictif
- Tester sur des données synthétiques avec signal clair
- Valider que le modèle peut apprendre une stratégie simple

---

## Recommandation

Avant de modifier le reward, **valider que le problème n'est pas les données**:

1. Créer un environnement "toy" avec signal prédictif évident
2. Vérifier que le modèle apprend à trader dans ce cas
3. Si oui → problème de features/données
4. Si non → problème d'architecture/hyperparamètres

---

## Fichiers Modifiés (Debug)

- `src/training/batch_env.py`: Ajout de logging vol_scalar (temporaire)
- `src/config/training.py`: `log_std_init = -1.0`
- `src/models/tqc_dropout_policy.py`: `log_std_init = -1.0` (hardcoded)

**Note:** Le code de debug (`_step_counter`, fichier vol_debug.txt) devra être retiré après investigation.

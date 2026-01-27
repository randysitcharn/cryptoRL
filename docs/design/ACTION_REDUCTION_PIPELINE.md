# Action Reduction Pipeline

Ce document liste **tous les éléments du code qui peuvent réduire l'action** entre la sortie de la policy et la position finale.

## Pipeline complet

```
Policy Network (mean, log_std)
    ↓
StateDependentNoise / SquashedGaussian (tanh squashing)
    ↓
Hardtanh clip_mean (±2.0)
    ↓
clamp(-1, 1) [batch_env.py:685]
    ↓
× vol_scalar [batch_env.py:702]
    ↓
clamp(-1, 1) [batch_env.py:702]
    ↓
action_discretization [batch_env.py:714-717]
    ↓
Position finale
```

---

## 1. Policy (SB3/TQC) - `src/models/robust_actor.py`

| Élément | Fichier:Ligne | Effet | Configurable |
|---------|---------------|-------|--------------|
| **tanh squashing** | `robust_actor.py:65,100` | `squash_output=True` → actions bornées [-1,+1] | Non (hardcodé) |
| **Hardtanh clip_mean** | `robust_actor.py:110` | `clip_mean=2.0` → mean clippé à ±2 avant tanh | Oui (param) |
| **SquashedDiagGaussian** | `robust_actor.py:114` | Distribution Gaussienne avec tanh intégré | Non |
| **StateDependentNoise** | `robust_actor.py:95-101` | gSDE avec `squash_output=True` | Non |

### Code concerné:
```python
# robust_actor.py:94-101
if self.use_sde:
    action_dist = StateDependentNoiseDistribution(
        action_dim,
        full_std=full_std,
        use_expln=use_expln,
        learn_features=True,
        squash_output=True,  # ← TANH ICI
    )
```

```python
# robust_actor.py:107-111
if clip_mean > 0.0:
    self.mu = nn.Sequential(
        self.mu,
        nn.Hardtanh(min_val=-clip_mean, max_val=clip_mean),  # ← CLIP MEAN
    )
```

---

## 2. Environment - `src/training/batch_env.py`

| Élément | Fichier:Ligne | Effet | Status actuel |
|---------|---------------|-------|---------------|
| **clamp initial** | `batch_env.py:685` | `clamp(actions, -1, 1)` | Toujours actif |
| **vol_scalars** | `batch_env.py:697-702` | `effective = raw * (target_vol / current_vol)` | `target=1.0` (désactivé) |
| **clamp après vol** | `batch_env.py:702` | `clamp(effective, -1, 1)` | Toujours actif |
| **action_discretization** | `batch_env.py:714-717` | `round(action/disc)*disc` | `disc=0.0` (désactivé) |

### Code concerné:
```python
# batch_env.py:685
raw_actions = torch.clamp(actions, -1.0, 1.0)

# batch_env.py:693-702
min_vol = self.target_volatility / self.max_leverage
current_vol = torch.sqrt(self.ema_vars)
clipped_vol = torch.clamp(current_vol, min=min_vol)
self.vol_scalars = torch.clamp(
    self.target_volatility / clipped_vol,
    min=0.1,
    max=self.max_leverage
)
effective_actions = torch.clamp(raw_actions * self.vol_scalars, -1.0, 1.0)

# batch_env.py:714-720
if self.action_discretization > 0:
    target_positions = torch.round(
        effective_actions / self.action_discretization
    ) * self.action_discretization
    target_positions = torch.clamp(target_positions, -1.0, 1.0)
else:
    target_positions = effective_actions
```

### Configuration (constants.py):
```python
DEFAULT_TARGET_VOLATILITY: float = 1.0  # 1.0 = désactivé
DEFAULT_VOL_WINDOW: int = 24
DEFAULT_MAX_LEVERAGE: float = 2.0
```

---

## 3. Wrappers - `src/training/wrappers.py`

| Élément | Fichier:Ligne | Effet | Utilisé |
|---------|---------------|-------|---------|
| **Action smoothing** | `wrappers.py:116` | `smoothed = α*action + (1-α)*prev` | Non (wrapper optionnel) |

### Code concerné:
```python
# wrappers.py:116
smoothed = self.alpha * action + (1 - self.alpha) * self._prev_action
```

---

## 4. Entropy Auto-Tuning (SB3 interne)

| Élément | Config | Effet |
|---------|--------|-------|
| **ent_coef** | `"auto_0.5"` | Coefficient multiplicateur de l'entropy bonus |
| **target_entropy** | `0.0` | Cible d'entropie (0 = neutre, -1 = exploitation) |
| **log_std learnable** | Oui | Le réseau apprend log_std, peut collapse vers -∞ |

### Problème:
Même avec `target_entropy=0.0`, le réseau peut apprendre que `mean ≈ 0` est optimal car:
- Petites positions = pas de risque = pas de perte
- L'entropy bonus n'est pas assez fort pour contrer

### Configuration (training.py):
```python
ent_coef: Union[str, float] = "auto_0.5"
target_entropy: Union[str, float] = 0.0
log_std_init: float = 3.0  # std initial = exp(3) ≈ 20
```

---

## 5. Ensemble - `src/evaluation/ensemble.py` (optionnel)

| Élément | Fichier:Ligne | Effet |
|---------|---------------|-------|
| **safety_scale** | `ensemble.py:291` | `action * safety_scale` |
| **scaling_factor** | `ensemble.py:351,442` | Réduction basée sur incertitude |

---

## Clarification: UN SEUL tanh (pas de multiple passage)

```
Hardtanh ≠ tanh !

- Hardtanh(x) = clip(x, -2, +2) → Fonction LINÉAIRE, pas de saturation
- tanh(x) → Fonction SIGMOÏDE, sature pour |x| > 2
```

Le pipeline exact :
```
mean_raw → Hardtanh(±2.0) → mean_clipped
                              ↓
                   tanh(mean_clipped + std * noise) → action
```

**Confirmation** : `rl_adapter.py:209` dit explicitement "NO TANH to avoid vanishing gradient with TQC's internal Tanh"

---

## Diagnostic: Pourquoi policy collapse?

### Réductions actives actuellement:
1. ✅ **tanh squashing** - ACTIF (UN SEUL tanh, dans StateDependentNoiseDistribution)
2. ✅ **Hardtanh clip_mean=2.0** - ACTIF (clip linéaire, PAS un tanh)
3. ❌ **vol_scalars** - DÉSACTIVÉ (target=1.0)
4. ❌ **action_discretization** - DÉSACTIVÉ (=0.0)
5. ✅ **Entropy auto-tuning** - ACTIF (peut causer mean collapse)

### Cause principale identifiée:
```
action = tanh(mean + std * noise)
```

Si le réseau apprend que `mean ≈ 0` est "safe" (pas de perte), alors:
- Même avec `std = 20`, `tanh(0 + 20*noise)` donne des valeurs proches de ±1
- Mais le réseau **apprend rapidement** à réduire le mean vers 0
- L'entropy bonus n'est pas suffisant pour contrer ce comportement
- Le problème n'est PAS le tanh, mais l'incentive (pas de pénalité pour inaction)

---

## Solutions proposées

| Solution | Difficulté | Effet attendu |
|----------|------------|---------------|
| **Inaction penalty** | Facile | Pénalise `\|pos\| < 0.1` |
| **Augmenter target_entropy** | Facile | Force plus d'exploration |
| **Désactiver tanh** | Difficile | Actions non bornées, log_prob incorrect |
| **Trading bonus** | Moyen | Récompense pour prendre des positions |
| **Epsilon-greedy** | Moyen | 10% actions aléatoires |

### Recommandation: Inaction Penalty
```python
# Dans batch_env.py, après calcul du reward
inaction_penalty = -0.01 * (1.0 - torch.abs(position_pct))
reward = reward + inaction_penalty
```

Effet:
- `position = 0` → penalty = -0.01
- `position = ±1` → penalty = 0

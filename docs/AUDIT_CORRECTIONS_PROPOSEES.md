# Rapport d'Audit : Corrections Proposées pour CryptoRL WFO

**Date** : 21 janvier 2026  
**Version** : 1.0  
**Objectif** : Document destiné à l'audit par un LLM externe

---

## 1. Contexte et Diagnostic

### 1.1 Résultats WFO Actuels

| Segment | Sharpe | PnL % | DD % | Trades | Alpha vs B&H | Marché | Status |
|---------|--------|-------|------|--------|--------------|--------|--------|
| 0 | -2.95 | -32.1% | 38.3% | 228 | **+4.6%** | 🔴 BEAR | SUCCESS |
| 1 | +3.09 | +35.3% | 10.1% | 149 | **-3.8%** | 🟢 BULL | RECOVERED |
| 2 | -4.84 | -63.1% | 71.3% | **969** | **-182%** | 🟢 BULL | SUCCESS |
| 3 | -0.95 | -11.1% | 29.3% | 438 | **+6.8%** | 🔴 BEAR | SUCCESS |
| 4 | -3.86 | -43.4% | 49.8% | **883** | **-43%** | ⚪ RANGE | SUCCESS |

### 1.2 Métriques TensorBoard Clés (Extraites du Serveur)

| Segment | Position Moyenne | Action Saturation | Entropy | Churn Multiplier PLO |
|---------|------------------|-------------------|---------|----------------------|
| 0 | +0.54 (LONG) | 0.18 → 0.24 | 0.078 → 0.020 | **1.0** (jamais actif) |
| 2 | +0.41 (LONG) | 0.34 → **0.47** | **0.015** | **1.0** (jamais actif) |
| 4 | -0.46 (SHORT) | 0.38 → 0.43 | 0.021 | **1.0** (jamais actif) |

### 1.3 Problèmes Identifiés

#### Problème 1 : Mismatch Train/Eval sur le Volatility Scaling

**Fichier** : `scripts/run_full_wfo.py`, ligne 732

```python
# Actuel (ligne 732)
max_leverage=1.0,  # Disable vol scaling (was: self.config.max_leverage)
```

**Impact** :
- En training : `max_leverage=2.0` → les positions sont amplifiées jusqu'à 2x via le `vol_scalar`
- En évaluation : `max_leverage=1.0` → les positions sont brutes, sans amplification
- Un agent qui apprend à faire des ajustements de 0.05 en training (amplifiés à 0.10) se retrouve à faire 0.05 en évaluation, créant un churn **différent** de celui appris

#### Problème 2 : PLO Churn Jamais Activé

**Observation** : `churn_multiplier = 1.0` sur **tous** les segments

**Cause** : Le `turnover_threshold` (0.08 = 8%) n'est jamais dépassé car le calcul de turnover utilise `metric_turnover = avg(current_position_deltas)` qui est proche de 0 en moyenne, même si le nombre de trades en évaluation est très élevé (200-900+).

**Fichier** : `src/training/callbacks.py`, lignes 1123-1131

```python
# Actuel
current_deltas = real_env.current_position_deltas  # Shape: (n_envs,)
avg_turnover = current_deltas.mean().item()
```

**Impact** : Le système PLO Churn est conçu pour augmenter la pénalité quand le turnover dépasse un seuil, mais ce seuil n'est jamais atteint car :
1. `current_position_deltas` est le delta **instantané** (step actuel vs step précédent)
2. En moyenne sur 1024 envs, ce delta est très petit
3. Le turnover **cumulé** par épisode n'est pas mesuré

#### Problème 3 : Alpha Négatif dans les Marchés Bull

**Segment 2** : Marché BULL (+119% B&H) mais l'agent fait -63% (Alpha = -182%)

**Analyse** :
- Position moyenne = +0.41 → L'agent est bien LONG
- Mais 969 trades en 3 mois = **overtrading massif** (~10 trades/jour)
- Chaque trade coûte ~0.05% (commission + slippage)
- Coût total ≈ 48% en frais (969 × 0.05%)

**Cause racine** : L'agent n'est pas pénalisé pour l'overtrading car le PLO Churn est inactif.

#### Problème 4 : Entropy Collapse

**Observation** : `ent_coef` descend à 0.015 (segment 2)

**Impact** : La politique devient quasi-déterministe, répétant les mêmes actions sans exploration, ce qui amplifie l'overtrading appris.

---

## 2. Corrections Proposées

### 2.1 CORRECTION 1 : Aligner Volatility Scaling Train/Eval

**Fichier** : `scripts/run_full_wfo.py`

**Avant** (ligne 732) :
```python
max_leverage=1.0,  # Disable vol scaling (was: self.config.max_leverage)
```

**Après** :
```python
max_leverage=self.config.max_leverage,  # Cohérence train/eval
```

**Justification** :
- Le volatility scaling est un composant clé de la stratégie apprise
- Le désactiver en évaluation crée un mismatch distribution → l'agent ne voit pas l'environnement qu'il a appris
- Le commentaire original mentionne "stuck in cash bug" mais ce bug devrait être résolu par le `vol_floor` introduit dans `batch_env.py`

**Risques** :
- Si le bug "stuck in cash" réapparaît, il faudra investiguer `vol_floor` dans `_calculate_volatility`
- Possible augmentation de la variance des résultats en évaluation

---

### 2.2 CORRECTION 2 : Réformer le Calcul de Turnover pour PLO Churn

**Fichier** : `src/training/callbacks.py`

**Avant** (lignes 1123-1133) :
```python
# TURNOVER MEASUREMENT
current_deltas = real_env.current_position_deltas
avg_turnover = current_deltas.mean().item()

self.turnover_history.append(avg_turnover)
if len(self.turnover_history) > self.prediction_horizon:
    self.turnover_history.pop(0)

# Average turnover over window
metric_turnover = np.mean(self.turnover_history[-20:]) if len(self.turnover_history) >= 20 else avg_turnover
```

**Après** :
```python
# TURNOVER MEASUREMENT - v2: Cumulative per Episode
current_deltas = real_env.current_position_deltas
sum_turnover = current_deltas.sum().item()  # Somme sur tous les envs (pas moyenne)
num_envs = real_env.num_envs

# Normaliser par le nombre d'envs pour obtenir turnover moyen par env
avg_turnover_per_env = sum_turnover / num_envs

self.turnover_history.append(avg_turnover_per_env)
if len(self.turnover_history) > self.prediction_horizon:
    self.turnover_history.pop(0)

# Turnover cumulé sur fenêtre glissante (plus sensible)
metric_turnover = np.sum(self.turnover_history[-20:]) if len(self.turnover_history) >= 20 else np.sum(self.turnover_history)
```

**Alternative** : Mesurer le turnover comme `total_trades / episode_length` à la fin de chaque épisode.

**Justification** :
- Le turnover **instantané** moyen est toujours proche de 0 car la plupart des steps n'ont pas de changement de position
- Le turnover **cumulé** sur une fenêtre reflète mieux le coût réel de l'overtrading
- Avec 969 trades sur 2095 steps, le turnover moyen par step est ~0.46, ce qui dépasserait facilement le seuil de 0.08

**Risques** :
- Changement de sémantique du `turnover_threshold` → potentiellement recalibrer le seuil
- Le PLO pourrait devenir trop agressif si mal calibré

---

### 2.3 CORRECTION 3 : Récompense Basée sur l'Alpha (Optionnel - Refonte Majeure)

**Fichier** : `src/training/batch_env.py`

**Avant** (lignes 406-410) :
```python
# 1. BASE REWARD: Log Returns (always active)
safe_returns = torch.clamp(step_returns, min=-0.99)
log_returns = torch.log1p(safe_returns) * SCALE
```

**Après** :
```python
# 1. BASE REWARD: Alpha vs Buy & Hold (excess return)
safe_returns = torch.clamp(step_returns, min=-0.99)

# Market return (B&H = hold 100% long)
market_return = (self.prices[self.current_steps] - self.prices[self.current_steps - 1]) / self.prices[self.current_steps - 1]
market_return = torch.clamp(market_return, min=-0.99)

# Alpha = portfolio return - market return
alpha = safe_returns - market_return
log_alpha = torch.log1p(torch.abs(alpha)) * torch.sign(alpha) * SCALE
```

**Justification** :
- L'objectif explicite est "battre B&H" → la récompense doit refléter cet objectif
- Avec des log-returns absolus, l'agent peut être récompensé même s'il sous-performe le marché
- Avec alpha, l'agent est pénalisé pour toute sous-performance vs B&H

**Risques** :
- Changement majeur de la fonction de récompense → nécessite re-tuning complet
- En marché BEAR, B&H perd → l'agent doit aussi perdre moins, ce qui peut encourager le shorting
- Nécessite une période de validation plus longue

**Recommandation** : Tester d'abord les corrections 1 et 2 avant cette refonte.

---

### 2.4 CORRECTION 4 : Augmenter les Coefficients de Pénalité

**Fichier** : `scripts/run_full_wfo.py`

**Avant** (lignes 83-84) :
```python
churn_coef: float = 0.5    # Max target après curriculum (réduit)
smooth_coef: float = 1e-5  # Très bas (curriculum monte à 0.00005 max)
```

**Après** :
```python
churn_coef: float = 1.0    # Doublé pour pénaliser l'overtrading
smooth_coef: float = 0.01  # Augmenté 1000x pour lisser les positions
```

**Fichier** : `src/training/callbacks.py`

**Avant** (lignes 619-623) :
```python
PHASES = [
    {'end_progress': 0.1, 'churn': (0.0, 0.10), 'smooth': (0.0, 0.0)},
    {'end_progress': 0.3, 'churn': (0.10, 0.50), 'smooth': (0.0, 0.005)},
    {'end_progress': 1.0, 'churn': (0.50, 0.50), 'smooth': (0.005, 0.005)},
]
```

**Après** :
```python
PHASES = [
    {'end_progress': 0.05, 'churn': (0.0, 0.20), 'smooth': (0.0, 0.0)},      # Phase 1: 5% (réduit)
    {'end_progress': 0.15, 'churn': (0.20, 1.00), 'smooth': (0.0, 0.01)},    # Phase 2: Ramp rapide
    {'end_progress': 1.0, 'churn': (1.00, 1.00), 'smooth': (0.01, 0.01)},    # Phase 3: Max penalties
]
```

**Justification** :
- Le `churn_coef` actuel (0.5) est insuffisant pour contrebalancer les gains de trading fréquent
- Le `smooth_coef` (1e-5) est quasi-nul et n'empêche pas les changements brusques
- Le curriculum actuel atteint le max seulement à 30% du training, laissant 70% sans progression

**Risques** :
- Si les pénalités sont trop fortes, l'agent pourrait ne plus trader du tout ("flat agent")
- Nécessite un monitoring du nombre de trades minimum par épisode

---

### 2.5 CORRECTION 5 : Réduire le Nombre de Timesteps

**Fichier** : `scripts/run_full_wfo.py`

**Avant** (ligne 73) :
```python
tqc_timesteps: int = 90_000_000  # 90M steps
```

**Après** :
```python
tqc_timesteps: int = 30_000_000  # 30M steps (réduit pour éviter overfitting)
```

**Justification** :
- Les logs montrent que l'`action_saturation` monte à 0.47 vers la fin du training
- L'`entropy` collapse à 0.015 indique une politique sur-ajustée
- Le modèle "best" est souvent trouvé avant 50% du training (signal "RECOVERED")

**Risques** :
- Potentiellement insuffisant pour apprendre des patterns complexes
- À combiner avec early stopping basé sur validation

---

## 3. Plan d'Implémentation Recommandé

### Phase 1 : Corrections Conservatrices (Quick Wins)

1. **CORRECTION 1** : Aligner vol scaling train/eval
2. **CORRECTION 4** : Augmenter `churn_coef` et `smooth_coef`
3. **CORRECTION 5** : Réduire timesteps à 30M

**Temps estimé** : 15 minutes de modification, 8-12h de re-training WFO

### Phase 2 : Correction du PLO Churn

4. **CORRECTION 2** : Réformer le calcul de turnover

**Temps estimé** : 30 minutes de modification, tests unitaires requis

### Phase 3 : Refonte Reward (Si Phase 1-2 insuffisantes)

5. **CORRECTION 3** : Alpha-based reward

**Temps estimé** : 2-4h de modification, re-tuning complet nécessaire

---

## 4. Métriques de Succès Post-Correction

| Métrique | Seuil Minimum | Objectif |
|----------|---------------|----------|
| Alpha moyen sur 5 segments | > -10% | > 0% |
| Sharpe moyen | > 0 | > 1.0 |
| Trades par segment | < 500 | < 200 |
| Action Saturation fin training | < 0.40 | < 0.30 |
| Entropy fin training | > 0.05 | > 0.10 |
| PLO Churn Multiplier activé | > 1.0 sur ≥1 segment | > 2.0 si violation |

---

## 5. Questions pour l'Auditeur

1. **Sur la Correction 1** : Le mismatch train/eval est-il la cause principale du gap de performance, ou y a-t-il d'autres facteurs (ex: stochasticité de l'environnement) ?

2. **Sur la Correction 2** : Le changement de sémantique du turnover (instantané → cumulé) pourrait-il créer des effets secondaires non anticipés dans le PID controller ?

3. **Sur la Correction 3** : L'utilisation de l'alpha comme récompense pourrait-elle créer un problème de "moving target" si le marché change de régime mid-épisode ?

4. **Sur la Correction 4** : Les valeurs proposées (`churn_coef=1.0`, `smooth_coef=0.01`) sont-elles calibrées correctement par rapport au `SCALE=100` de la reward function ?

5. **Architecture** : Le système PLO actuel (3 contrôleurs PID indépendants) est-il adapté, ou faudrait-il un contrôleur multi-objectif (ex: MORL) ?

---

## 6. Annexes

### 6.1 Code Source Pertinent

**Reward Function** : `src/training/batch_env.py` lignes 363-493  
**Curriculum Phases** : `src/training/callbacks.py` lignes 619-623  
**PLO Churn** : `src/training/callbacks.py` lignes 1045-1205  
**Evaluation** : `scripts/run_full_wfo.py` lignes 700-954

### 6.2 Données du Serveur

```
SSH: ssh -p 20941 root@158.51.110.52
Résultats: /workspace/cryptoRL/results/wfo_results.csv
Logs: /workspace/cryptoRL/logs/wfo/
```

### 6.3 Configuration Actuelle (WFOConfig)

```python
tqc_timesteps: 90_000_000
learning_rate: 1e-4
buffer_size: 2_500_000
n_envs: 1024
batch_size: 512
gamma: 0.95
ent_coef: "auto_0.5"
churn_coef: 0.5
smooth_coef: 1e-5
target_volatility: 0.05
max_leverage: 2.0
observation_noise: 0.01
critic_dropout: 0.1
```

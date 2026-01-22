# Curriculum Learning - Tuning Guide (MORL Architecture)

> **Date:** 2026-01-21 | **Architecture:** MORL avec curriculum_lambda

---

## 1. Architecture Actuelle

L'architecture MORL utilise deux paramètres indépendants :
- `w_cost ∈ [0, 1]`: Paramètre MORL dans l'observation (vu par l'agent)
- `curriculum_lambda ∈ [0, 0.4]`: Progression du curriculum (contrôle noise annealing et futures pénalités)

**Note importante** : Dans l'implémentation actuelle (v1.0), `curriculum_lambda` n'est **pas** utilisé directement dans la formule de récompense MORL. Ces deux paramètres sont indépendants.

### Formule de Reward (Implémentation Actuelle)
```python
reward = r_perf + w_cost * r_cost * MAX_PENALTY_SCALE
```

où:
- `r_perf`: Log-returns (objectif performance)
- `r_cost`: Pénalité de changement de position
- `w_cost ∈ [0, 1]`: Paramètre MORL dans l'observation (conditioned network)
- `MAX_PENALTY_SCALE = 2.0`: Facteur de calibration fixe

**Pourquoi cette séparation ?**
- `w_cost` (vu par l'agent) module déjà dynamiquement l'importance des coûts
- Ajouter `curriculum_lambda` créerait une double modulation potentiellement instable
- Le curriculum actuel se concentre sur l'exploration progressive (noise decay)

**Roadmap v2.0** : Si nécessaire, envisager:
```python
effective_scale = MAX_PENALTY_SCALE * curriculum_lambda
reward = r_perf + (w_cost * r_cost * effective_scale)
```
Ceci introduirait les coûts progressivement pendant le training.

---

## 2. Configuration des Phases

```python
PHASES = [
    # Phase 1: Pure Exploration (0% -> 15%)
    # curriculum_lambda = 0.0 - Exploration pure, noise maximal
    {'end_progress': 0.15, 'lambda': (0.0, 0.0)},
    
    # Phase 2: Discipline (15% -> 75%)
    # curriculum_lambda: 0.0 -> 0.4 - Ramp progressif (noise decay)
    {'end_progress': 0.75, 'lambda': (0.0, 0.4)},
    
    # Phase 3: Consolidation (75% -> 100%)
    # curriculum_lambda = 0.4 - Valeur finale stable
    {'end_progress': 1.0, 'lambda': (0.4, 0.4)},
]
```

**Note** : Les phases ont été étendues (15% → 75% pour Phase 2) pour un ramping plus progressif.

### Visualisation

```
curriculum_lambda
    0.40 |                   ************************************
    0.30 |                  *
    0.20 |                 *
    0.10 |                *
    0.00 |***************
         +----------------------------------------------------
         0%   15%        30%                              100%
         |     |          |                                 |
      Pure  Discipline        Consolidation (70%)
    Exploration
```

| Phase | Progress | curriculum_lambda | Objectif |
|-------|----------|-------------------|----------|
| Pure Exploration | 0% - 15% | 0.0 | Apprendre à trader profitablement |
| Discipline | 15% - 30% | 0.0 -> 0.4 | Introduire les coûts progressivement |
| Consolidation | 30% - 100% | 0.4 | Optimiser profit+coûts |

---

## 3. MORL: Paramètre w_cost

L'agent voit `w_cost` dans son observation et apprend à adapter son comportement:

| w_cost | Comportement | Usage |
|--------|--------------|-------|
| 0.0 | Scalping (ignorer coûts) | Trading haute fréquence |
| 0.5 | Équilibré | Comportement standard |
| 1.0 | B&H (minimiser coûts) | Trading conservateur |

### Distribution pendant l'entraînement
```python
# 20% extremes + 60% uniform pour exploration complète
if random() < 0.2:
    w_cost = 0.0  # 10%: scalping pur
elif random() < 0.4:
    w_cost = 1.0  # 10%: B&H pur
else:
    w_cost = uniform(0, 1)  # 80%: exploration
```

---

## 4. Indicateurs de Diagnostic

### Problèmes potentiels

| Indicateur | Seuil | Action |
|------------|-------|--------|
| Position near-zero (>80%) | `sum(|pos| < 0.05) / total` | Réduire curriculum_lambda max |
| Entropy collapse (<0.05) | `train/ent_coef` | Augmenter ent_coef |
| NAV négatif constant | | Vérifier r_perf domine |

### Script de Diagnostic

```python
from tensorboard.backend.event_processing import event_accumulator

ea = event_accumulator.EventAccumulator('logs/wfo/segment_0/WFO_seg0_1')
ea.Reload()

# Vérifier que performance domine
pnls = [e.value for e in ea.Scalars('rewards/pnl')]
costs = [abs(e.value) for e in ea.Scalars('rewards/cost_penalty')]

ratio = np.mean(costs) / np.mean([abs(p) for p in pnls])
print(f'Ratio |Cost|/|PnL|: {ratio:.2f}')

if ratio > 0.5:
    print('WARNING: Cost penalty domine - réduire curriculum_lambda')
else:
    print('OK: PnL domine le signal de reward')
```

---

## 5. Avantages de MORL vs PLO (Architecture Précédente)

| Aspect | MORL Actuel | PLO (Ancien) |
|--------|-------------|--------------|
| Hyperparamètres | 1 (MAX_PENALTY_SCALE) | 10+ (λ, seuils, PID gains) |
| Adaptabilité | w_cost ajustable post-training | Fixe après training |
| Complexité | Simple | 3 contrôleurs PID |
| Robustesse | Agent apprend toutes préférences | Sensible aux seuils |

---

## 6. Recommandations

### Pour nouveaux runs WFO

1. **Garder** curriculum_lambda max à 0.4 (équilibre profit/coûts)
2. **Monitorer** le ratio Cost/PnL pendant le training
3. **Ajuster** w_cost en inférence selon les conditions de marché

### Ajustements en Inférence

| Condition Marché | w_cost Recommandé |
|------------------|-------------------|
| Bull fort | 0.2 (plus de trading) |
| Range/Volatile | 0.5 (équilibré) |
| Bear/Crash | 0.8 (conservateur) |

---

*Fichier: docs/design/CURRICULUM_TUNING.md*
*Mise à jour: 2026-01-21 (Transition vers MORL)*

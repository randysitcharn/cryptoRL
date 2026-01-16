# Curriculum Learning - Tuning Guide

> **Date:** 2026-01-16 | **Branch:** `feat/training-speed-optimization`

---

## 1. Probleme Observe

Avec la configuration initiale (ramp jusqu'a 60%), l'agent apprenait a **minimiser le churn plutot qu'a maximiser le profit**.

### Symptomes
```
Ratio |Churn|/|PnL| = 0.12  <- semblait OK
NAV en baisse constante (-5.4% puis crash a -51.8% drawdown)
Position moyenne proche de 0 (38% du temps < 5%)
```

### Diagnostic
L'agent a appris trop tot a eviter le churn, AVANT de maitriser le trading rentable. Le signal de penalite churn dominait l'apprentissage pendant la phase critique.

---

## 2. Configuration Actuelle

```python
PHASES = [
    # Phase 1: Discovery (0% -> 10%) - exploration libre
    {'end_progress': 0.1, 'churn': (0.0, 0.10), 'smooth': (0.0, 0.0)},
    # Phase 2: Discipline (10% -> 30%) - ramp-up rapide vers max
    {'end_progress': 0.3, 'churn': (0.10, 0.50), 'smooth': (0.0, 0.02)},
    # Phase 3: Consolidation (30% -> 100%) - LONG PLATEAU at max
    {'end_progress': 1.0, 'churn': (0.50, 0.50), 'smooth': (0.02, 0.02)},
]
```

### Visualisation

```
churn_coef
    0.50 |                   ************************************
    0.40 |                  *
    0.30 |                 *
    0.20 |                *
    0.10 |     ***********
    0.00 |*****
         +----------------------------------------------------
         0%   10%         30%                              100%
         |     |           |                                 |
      Discovery Discipline        Consolidation (70%)
```

| Phase | Progress | churn_coef | smooth_coef | Objectif |
|-------|----------|------------|-------------|----------|
| Discovery | 0% - 10% | 0.00 -> 0.10 | 0.00 | Explorer librement |
| Discipline | 10% - 30% | 0.10 -> 0.50 | 0.00 -> 0.02 | Apprendre les contraintes |
| Consolidation | 30% - 100% | 0.50 (fixe) | 0.02 (fixe) | Optimiser profit+contraintes |

---

## 3. Indicateurs de Diagnostic

### Churn_coef trop eleve si:

| Indicateur | Seuil Critique | Comment Verifier |
|------------|----------------|------------------|
| Ratio Churn/PnL | > 0.5 | `mean(|churn_penalty|) / mean(|pnl|)` |
| Position near-zero | > 80% | `sum(|pos| < 0.05) / total` |
| Entropy collapse | < 0.05 | `train/ent_coef` dans TensorBoard |
| Std position | < 0.02 | Agent ne bouge plus |

### Script de Diagnostic

```python
# Extraire depuis TensorBoard
from tensorboard.backend.event_processing import event_accumulator

ea = event_accumulator.EventAccumulator('logs/wfo/segment_0/WFO_seg0_1')
ea.Reload()

pnls = [e.value for e in ea.Scalars('rewards/pnl')]
churns = [abs(e.value) for e in ea.Scalars('rewards/churn_penalty')]

ratio = np.mean(churns) / np.mean([abs(p) for p in pnls])
print(f'Ratio |Churn|/|PnL|: {ratio:.2f}')

if ratio > 0.5:
    print('WARNING: Churn penalty trop dominant')
else:
    print('OK: PnL domine le signal de reward')
```

---

## 4. Alternatives Considerees

### Option A: Baisser le max churn_coef

```python
# Churn comme regularizer leger (pas signal dominant)
PHASES = [
    {'end_progress': 0.1, 'churn': (0.0, 0.05), 'smooth': (0.0, 0.0)},
    {'end_progress': 0.3, 'churn': (0.05, 0.25), 'smooth': (0.0, 0.02)},
    {'end_progress': 1.0, 'churn': (0.25, 0.25), 'smooth': (0.02, 0.02)},
]
```

**Avantage:** PnL reste toujours dominant
**Inconvenient:** Agent peut churner excessivement

### Option B: Curriculum Adaptatif

```python
# N'augmente churn que si l'agent est profitable
def _on_step(self):
    avg_pnl = self.get_recent_pnl()
    if avg_pnl > 0:
        self.current_churn = min(self.current_churn + 0.001, self.max_churn)
    # else: garde churn bas pour laisser l'agent apprendre
```

**Avantage:** S'adapte au niveau de l'agent
**Inconvenient:** Complexite, risque de ne jamais augmenter si agent faible

### Option C: Ratio Fixe (Clipping)

```python
# Dans env.py - churn penalty toujours < 20% du |PnL|
churn_penalty = min(raw_churn_penalty, 0.2 * abs(pnl_reward))
```

**Avantage:** Garantit que PnL domine toujours
**Inconvenient:** Peut rendre churn inefficace sur gros PnL

---

## 5. Recommandations

### Pour WFO Segment 0 (config actuelle)

1. **Garder** la ramp rapide (30%) pour long plateau
2. **Monitorer** le ratio Churn/PnL pendant le training
3. **Si ratio > 0.3**: considerer baisser max churn a 0.30

### Ajustements Futurs

| Si Observation | Action |
|----------------|--------|
| Agent ne trade pas (pos ~0) | Baisser max churn a 0.30 |
| Agent churne trop (>50 trades/episode) | Augmenter max churn a 0.60 |
| NAV stable mais pas de profit | Reduire smooth_coef |
| Entropy collapse (<0.05) | Reduire tous les coefs de penalite |

---

## 6. Historique des Configurations

| Date | Config | Resultat |
|------|--------|----------|
| 2026-01-13 | ramp 80%, max 0.50 | Non teste (change avant run) |
| 2026-01-15 | ramp 60%, max 0.50 | -9.4% PnL, agent passif |
| 2026-01-16 | ramp 30%, max 0.50 | En cours de test |

---

*Fichier: docs/CURRICULUM_TUNING.md*

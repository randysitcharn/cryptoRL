# PLO - Predictive Lagrangian Optimization pour Churn Adaptatif

**Version** : 1.1 (Post-Audit)  
**Date** : 2026-01-18  
**Statut** : Spécification technique - Production-Ready

> ⚠️ **Version 1.1** : Corrige le "Paradoxe du Profit Gate" identifié lors de l'audit technique.

---

## Table des Matières

1. [Résumé Exécutif](#1-résumé-exécutif)
2. [Contexte et Motivation](#2-contexte-et-motivation)
3. [Architecture Actuelle](#3-architecture-actuelle)
4. [Spécification PLO Churn](#4-spécification-plo-churn)
5. [Implémentation](#5-implémentation)
6. [Paramètres et Tuning](#6-paramètres-et-tuning)
7. [Validation et Tests](#7-validation-et-tests)

---

## 1. Résumé Exécutif

### Objectif

Implémenter un système de **pénalité churn adaptative** où le coefficient augmente dynamiquement quand le taux de trading dépasse un seuil, créant une pression progressive pour réduire le turnover excessif.

### Pourquoi PLO sur Churn ?

| Situation | Besoin | Solution PLO |
|-----------|--------|--------------|
| Marché calme | Peu de trades nécessaires | λ = 1.0 (pénalité normale) |
| Hyperactivité | Agent trade frénétiquement | λ augmente → freine le trading |
| Opportunité rare | Repositionnement justifié | Gate réduit l'impact (profit-gated) |

### Métrique Cible : Turnover Rate

Le **turnover** mesure l'activité de trading :

```
turnover_rate = Σ|Δposition| / n_steps
             = Volume de changements de position par step
```

- **Turnover élevé** = trading excessif (coûts de transaction élevés)
- **Turnover faible** = positions stables (coûts minimisés)

---

## 2. Contexte et Motivation

### 2.1 Formule Actuelle de Churn

```python
raw_churn = |Δposition| × cost_rate × churn_coef × SCALE
churn_penalty = raw_churn × gate
```

| Composante | Description |
|------------|-------------|
| `Δposition` | Changement absolu de position |
| `cost_rate` | Commission + slippage (~0.07%) |
| `churn_coef` | Coefficient de pénalité (0 → 0.5 via curriculum) |
| `gate` | Profit-gate : réduit la pénalité si step non profitable |

### 2.2 Le Gate Existant

Le système actuel a déjà un mécanisme intelligent :

```python
gate = clamp(step_returns / TARGET_PNL, 0, 1)
# Si step_returns <= 0 : gate = 0 → pas de pénalité churn
# Si step_returns >= 0.5% : gate = 1 → pénalité complète
```

**Logique** : Ne pas pénaliser le trading si ce step n'est pas profitable (l'agent n'aurait pas dû trader de toute façon).

### 2.3 Problème Résiduel

Même avec le gate, le `churn_coef` reste **statique**. Un agent hyperactif qui trade profitablement (gate = 1) n'est pas davantage pénalisé :

```
Agent A: 10 trades/jour profitable → Pénalité = 10 × base
Agent B: 50 trades/jour profitable → Pénalité = 50 × base (5× plus)

Mais l'Agent B pourrait faire mieux avec moins de trades !
→ On veut augmenter λ pour le forcer à réduire son turnover.
```

### 2.4 Solution Proposée

Ajouter un multiplicateur PLO **au-dessus** du gate, avec un **"leak minimum"** pour éviter le paradoxe du gate :

```python
# Gate effectif avec leak minimum quand PLO actif
effective_gate = max(gate, min_leak × (λ - 1) / 4)
churn_penalty = raw_churn × effective_gate × churn_multiplier
#                           ↑                ↑
#                    Leak minimum si λ > 1   PLO contrôle λ ∈ [1.0, 5.0]
```

---

## 3. Architecture Actuelle

### 3.1 Position dans le Reward

```
reward = log_returns - curriculum_λ × (churn + downside) - smoothness
                                       ↑
                                 PLO cible ici
```

### 3.2 Curriculum Actuel de Churn

```
Phase 1 (0-10%):  churn_coef = 0.0 → 0.10
Phase 2 (10-30%): churn_coef = 0.10 → 0.50
Phase 3 (30-100%): churn_coef = 0.50
```

Le PLO s'appliquera **après** le curriculum et le gate :
`effective_churn = raw_churn × gate × churn_multiplier`

### 3.3 Schéma du Flow Actuel

```
|Δposition| ──┐
              │
cost_rate ───┼──▶ raw_churn = |Δpos| × cost × coef × SCALE
              │
churn_coef ──┘
                        │
                        ▼
step_returns ──▶ gate = clamp(returns / 0.5%, 0, 1)
                        │
                        ▼
              churn_penalty = raw_churn × gate
                        │
                        ▼
              curriculum_λ × churn_penalty ──▶ Reward
```

**Avec PLO**, on ajoute un multiplicateur ET un "leak minimum" :

```
              ┌─────────────────────────────────────────────────────────┐
              │  CORRECTION AUDIT: "Leak Minimum"                       │
              │                                                         │
              │  Si λ > 1 (PLO actif), on force au moins 20% du        │
              │  signal PLO même si gate = 0.                           │
              │                                                         │
              │  Évite le "Paradoxe du Profit Gate" où un agent        │
              │  churne à perte sans être pénalisé.                     │
              └─────────────────────────────────────────────────────────┘

              effective_gate = max(gate, 0.2 × (λ - 1) / 4)
                        │
                        ▼
              churn_penalty = raw_churn × effective_gate × churn_multiplier
                                                           ↑
                                                     PLO contrôle
```

---

## 4. Spécification PLO Churn

### 4.1 Métrique de Contrainte : Turnover Rate

```python
# Calcul du turnover moyen sur une fenêtre glissante
turnover_history.append(position_deltas.mean().item())
avg_turnover = mean(turnover_history[-50:])  # Moyenne sur 50 steps

# Contrainte
g(t) = max(0, avg_turnover - turnover_threshold)
```

**Paramètre clé** : `turnover_threshold = 0.08` (8% de changement moyen par step ≈ 2 repositionnements/jour)

| Turnover moyen | Interprétation | Action PLO |
|----------------|----------------|------------|
| < 0.08 | Trading modéré | λ → 1.0 |
| 0.08 - 0.15 | Trading actif | λ = 1.5 - 3.0 |
| > 0.15 | Hyperactivité | λ → 5.0 |

### 4.2 Particularité : Composante Prédictive

Contrairement à PLO Smoothness, PLO Churn **inclut** une prédiction :
- Le turnover a une inertie (patterns de trading persistants)
- Prédire permet d'anticiper une spirale de trading

```python
# Prédiction via polyfit si turnover augmente
slope, intercept = np.polyfit(x, y, 1)

# AUDIT FIX: Ignorer les pentes très faibles pour éviter le jitter
MIN_SLOPE_THRESHOLD = 0.001  # 0.1% par step minimum

if slope > MIN_SLOPE_THRESHOLD:  # Turnover en hausse significative
    predicted_turnover = slope * 15 + intercept
    predicted_violation = max(0, predicted_turnover - threshold)
```

### 4.3 Contrôleur PID + Prédiction

```
λ(t) = λ_min + P(t) + I(t) + D(t)

P(t) = K_p × g_eff(t)                       [Proportionnel]
I(t) = I(t-1) + K_i × g(t)                  [Intégral]
D(t) = K_d × (g(t) - g(t-1))                [Dérivé]

g_eff(t) = max(g(t), 0.6 × ĝ(t+h))          [Violation effective]
```

### 4.4 Diagramme du Contrôleur

```
                    ┌─────────────────────────────────────────────────────┐
                    │            PLO CHURN CONTROLLER                     │
                    │                                                     │
   |Δposition|(t) ─▶│  ┌─────────────────────────────────────────────┐   │
   (batch mean)     │  │                                             │   │
                    │  │  turnover_history.append(mean(|Δpos|))      │   │
                    │  │  avg_turnover = mean(history[-50:])         │   │
                    │  │                                             │   │
                    │  └──────────────────┬──────────────────────────┘   │
                    │                     │                              │
                    │                     ▼                              │
                    │  ┌─────────────────────────────────────────────┐   │
                    │  │     ROBUST PREDICTION (Polyfit)             │   │
                    │  │                                             │   │
                    │  │  slope, intercept = np.polyfit(x, y, 1)     │   │
                    │  │  if slope > 0:  # Turnover increasing       │   │
                    │  │      predicted = slope × 15 + intercept     │   │
   threshold ──────▶│  │      ĝ = max(0, predicted - threshold)      │   │
                    │  │  g_eff = max(g, 0.6 × ĝ)                    │   │
                    │  │                                             │   │
                    │  └──────────────────┬──────────────────────────┘   │
                    │                     │                              │
                    │                     ▼                              │
                    │  ┌─────────────────────────────────────────────┐   │
                    │  │         PID CONTROLLER + SMOOTHING          │   │
                    │  │                                             │   │
                    │  │  target_λ = λ_min + P + I + D               │   │
                    │  │  Δλ = clip(target_λ - λ_prev, ±0.08)        │   │
                    │  │  λ = λ_prev + Δλ                            │   │
                    │  │                                             │   │
                    │  └──────────────────┬──────────────────────────┘   │
                    │                     │                              │
                    └─────────────────────┼──────────────────────────────┘
                                          │
                                          ▼
                               churn_multiplier = λ
                                          │
                                          ▼
                              ┌───────────────────────┐
                              │  OBSERVATION AUGMENT  │
                              │                       │
                              │  obs["churn_level"] = │
                              │    (λ - 1.0) / 4.0    │
                              └───────────────────────┘
```

---

## 5. Implémentation

### 5.1 Fichiers à Modifier

| Fichier | Modification |
|---------|--------------|
| `src/training/batch_env.py` | Ajouter `churn_multiplier` + observation `churn_level` |
| `src/training/callbacks.py` | Ajouter `PLOChurnCallback` |
| `src/training/train_agent.py` | Intégrer le callback |

### 5.2 Modification de batch_env.py

```python
class BatchCryptoEnv:
    def __init__(self, ...):
        # ... existing code ...
        
        # PLO Churn: Multiplicateur adaptatif
        self.churn_multiplier = 1.0
        
        # Modifier observation_space pour inclure churn_level
        self.observation_space = spaces.Dict({
            "market": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(window_size, self.n_features),
                dtype=np.float32
            ),
            "position": spaces.Box(
                low=-1.0, high=1.0,
                shape=(1,),
                dtype=np.float32
            ),
            "churn_level": spaces.Box(
                low=0.0, high=1.0,
                shape=(1,),
                dtype=np.float32
            )
        })
    
    def set_churn_multiplier(self, value: float) -> None:
        """Setter pour PLO Churn callback."""
        self.churn_multiplier = max(1.0, min(value, 10.0))
    
    @property
    def current_position_deltas(self) -> torch.Tensor:
        """Retourne les deltas de position actuels pour le callback."""
        return torch.abs(self.position_pcts - self.prev_position_pcts)
    
    def _get_observations(self) -> Dict[str, np.ndarray]:
        """Get observations for all envs as numpy (SB3 compatible)."""
        market = self._get_batch_windows(self.current_steps)

        if self.observation_noise > 0 and self.training:
            noise = torch.randn_like(market) * self.observation_noise
            market = market + noise

        position = self.position_pcts.unsqueeze(1)
        
        # ═══════════════════════════════════════════════════════════════════
        # OBSERVATION AUGMENTÉE: Churn Level (PLO λ normalisé)
        # ═══════════════════════════════════════════════════════════════════
        churn_level_value = (self.churn_multiplier - 1.0) / 4.0
        churn_level = torch.full(
            (self.num_envs, 1), 
            churn_level_value, 
            device=self.device
        )

        return {
            "market": market.cpu().numpy(),
            "position": position.cpu().numpy(),
            "churn_level": churn_level.cpu().numpy()
        }
    
    def _calculate_rewards(self, step_returns, position_deltas, dones):
        SCALE = 100.0
        TARGET_PNL = 0.005
        MIN_GATE_LEAK = 0.2  # Leak minimum pour éviter le paradoxe du gate
        
        # ... calcul de log_returns ...
        
        # ═══════════════════════════════════════════════════════════════════
        # CHURN avec Gate + PLO + LEAK MINIMUM + SAFETY CLIP
        # ═══════════════════════════════════════════════════════════════════
        
        # 1. Gate profit-conditionné de base
        base_gate = torch.clamp(step_returns / TARGET_PNL, min=0.0, max=1.0)
        self.last_gate_mean = base_gate.mean().item()
        
        # 2. Calcul de base
        cost_rate = self.commission + self.slippage
        raw_churn = position_deltas * cost_rate * self._current_churn_coef * SCALE
        
        # ═══════════════════════════════════════════════════════════════════
        # 3. CORRECTION AUDIT: "Leak Minimum" pour éviter le paradoxe du gate
        # Si λ > 1 (PLO actif), on force au moins 20% du signal PLO
        # même si le step n'est pas profitable (gate = 0).
        # Cela évite qu'un agent "churne à perte" sans être pénalisé.
        # ═══════════════════════════════════════════════════════════════════
        if self.churn_multiplier > 1.0:
            # Leak proportionnel à l'activation du PLO: 0% si λ=1, 20% si λ=5
            plo_intensity = (self.churn_multiplier - 1.0) / 4.0  # [0, 1]
            min_gate = MIN_GATE_LEAK * plo_intensity
            effective_gate = torch.clamp(base_gate, min=min_gate)
        else:
            effective_gate = base_gate
        
        # 4. Application du gate effectif
        gated_churn = raw_churn * effective_gate
        
        # 5. Application du multiplicateur PLO
        plo_churn = gated_churn * self.churn_multiplier
        
        # 6. SAFETY CLIP
        CHURN_PENALTY_CAP = 15.0
        safe_churn = torch.clamp(plo_churn, max=CHURN_PENALTY_CAP)
        
        churn_penalty = safe_churn
        
        # 7. Observabilité
        self._rew_churn = -churn_penalty * self.curriculum_lambda
        self._raw_churn_before_clip = plo_churn.mean().item()
        self._effective_gate_mean = effective_gate.mean().item()  # NOUVEAU: monitoring du leak
        
        # ... reste du calcul de reward ...
```

### 5.3 PLOChurnCallback

```python
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback


class PLOChurnCallback(BaseCallback):
    """
    Predictive Lagrangian Optimization pour Churn adaptatif.
    
    Augmente la pénalité de turnover quand l'agent trade excessivement.
    
    Inclut une composante prédictive car le turnover a une inertie
    (patterns de trading persistants).
    """
    
    def __init__(
        self,
        # Contrainte Turnover
        turnover_threshold: float = 0.08,  # 8% de changement moyen par step (~2 repos/jour)
        turnover_lambda_min: float = 1.0,
        turnover_lambda_max: float = 5.0,
        # Gains PID
        turnover_Kp: float = 2.5,
        turnover_Ki: float = 0.08,
        turnover_Kd: float = 0.4,
        # Prédiction
        prediction_horizon: int = 50,
        use_prediction: bool = True,
        # Stabilité
        integral_max: float = 2.0,
        decay_rate: float = 0.995,
        max_lambda_change: float = 0.08,
        # Logging
        log_freq: int = 100,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        
        self.turnover_threshold = turnover_threshold
        self.turnover_lambda_min = turnover_lambda_min
        self.turnover_lambda_max = turnover_lambda_max
        self.turnover_Kp = turnover_Kp
        self.turnover_Ki = turnover_Ki
        self.turnover_Kd = turnover_Kd
        self.prediction_horizon = prediction_horizon
        self.use_prediction = use_prediction
        self.integral_max = integral_max
        self.decay_rate = decay_rate
        self.max_lambda_change = max_lambda_change
        self.log_freq = log_freq
        
        # État du contrôleur
        self.turnover_integral = 0.0
        self.turnover_prev_violation = 0.0
        self.turnover_lambda = 1.0
        
        # Buffer pour prédiction
        self.turnover_history = []
        
    def _on_step(self) -> bool:
        real_env = get_underlying_batch_env(self.model.env)
        
        if not hasattr(real_env, 'current_position_deltas'):
            return True
        
        # ═══════════════════════════════════════════════════════════════════
        # PROTECTION CURRICULUM
        # Ne pas activer le PLO si churn_coef ≈ 0
        # ═══════════════════════════════════════════════════════════════════
        curriculum_active = True
        if hasattr(real_env, '_current_churn_coef'):
            if real_env._current_churn_coef < 0.05:
                self.turnover_integral *= 0.9  # Decay rapide
                curriculum_active = False
        
        # ═══════════════════════════════════════════════════════════════════
        # MESURE DU TURNOVER
        # ═══════════════════════════════════════════════════════════════════
        current_deltas = real_env.current_position_deltas
        avg_turnover = current_deltas.mean().item()
        
        self.turnover_history.append(avg_turnover)
        if len(self.turnover_history) > self.prediction_horizon:
            self.turnover_history.pop(0)
        
        # Turnover moyen sur la fenêtre
        metric_turnover = np.mean(self.turnover_history[-20:]) if len(self.turnover_history) >= 20 else avg_turnover
        max_turnover = current_deltas.max().item()
        violation = max(0.0, metric_turnover - self.turnover_threshold)
        
        # ═══════════════════════════════════════════════════════════════════
        # PRÉDICTION (si curriculum actif)
        # ═══════════════════════════════════════════════════════════════════
        predicted_violation = 0.0
        slope = 0.0
        
        if curriculum_active and self.use_prediction and len(self.turnover_history) >= 15:
            y = np.array(self.turnover_history[-15:])
            x = np.arange(len(y))
            slope, intercept = np.polyfit(x, y, 1)
            
            # AUDIT FIX: Seuil minimum pour ignorer le bruit
            # Ne prédire que si turnover en hausse significative (> 0.1% par step)
            MIN_SLOPE = 0.001
            if slope > MIN_SLOPE:
                future_turnover = slope * (len(y) + 10) + intercept
                predicted_violation = max(0.0, future_turnover - self.turnover_threshold)
        
        # ═══════════════════════════════════════════════════════════════════
        # CONTRÔLEUR PID
        # ═══════════════════════════════════════════════════════════════════
        if curriculum_active:
            if violation > 0 or predicted_violation > 0:
                effective_violation = max(violation, 0.6 * predicted_violation)
                
                P = self.turnover_Kp * effective_violation
                self.turnover_integral += self.turnover_Ki * violation
                self.turnover_integral = np.clip(self.turnover_integral, 0, self.integral_max)
                I = self.turnover_integral
                D = self.turnover_Kd * (violation - self.turnover_prev_violation)
                
                target_lambda = self.turnover_lambda_min + P + I + D
                target_lambda = np.clip(target_lambda, self.turnover_lambda_min, self.turnover_lambda_max)
            else:
                # Decay vers λ_min
                target_lambda = max(self.turnover_lambda_min, self.turnover_lambda * self.decay_rate)
                self.turnover_integral *= 0.995
            
            # Smoothing
            change = np.clip(target_lambda - self.turnover_lambda, 
                            -self.max_lambda_change, self.max_lambda_change)
            self.turnover_lambda = self.turnover_lambda + change
        
        self.turnover_prev_violation = violation
        
        # Appliquer à l'environnement
        if hasattr(real_env, 'set_churn_multiplier'):
            real_env.set_churn_multiplier(self.turnover_lambda)
        
        # Logging
        if self.num_timesteps % self.log_freq == 0:
            self.logger.record("plo_churn/turnover_violation", violation)
            self.logger.record("plo_churn/turnover_predicted", predicted_violation)
            self.logger.record("plo_churn/turnover_multiplier", self.turnover_lambda)
            self.logger.record("plo_churn/turnover_integral", self.turnover_integral)
            self.logger.record("plo_churn/turnover_slope", slope)
            self.logger.record("plo_churn/metric_turnover", metric_turnover)
            self.logger.record("plo_churn/max_turnover", max_turnover)
            self.logger.record("plo_churn/curriculum_active", float(curriculum_active))
        
        return True
    
    def _on_training_start(self) -> None:
        if self.verbose > 0:
            print(f"\n[PLO Churn] Configuration:")
            print(f"  Turnover threshold: {self.turnover_threshold:.2f}")
            print(f"  Lambda range: [{self.turnover_lambda_min}, {self.turnover_lambda_max}]")
            print(f"  PID gains: Kp={self.turnover_Kp}, Ki={self.turnover_Ki}, Kd={self.turnover_Kd}")
            print(f"  Prediction: {'polyfit (robust)' if self.use_prediction else 'disabled'}")
```

### 5.4 Intégration dans train_agent.py

```python
from src.training.callbacks import PLOChurnCallback

callbacks = [
    ThreePhaseCurriculumCallback(total_timesteps=config.total_timesteps, verbose=1),
    PLOChurnCallback(
        turnover_threshold=0.08,
        turnover_lambda_min=1.0,
        turnover_lambda_max=5.0,
        turnover_Kp=2.5,
        turnover_Ki=0.08,
        turnover_Kd=0.4,
        use_prediction=True,
        verbose=1
    ),
    # ... autres callbacks
]
```

---

## 6. Paramètres et Tuning

### 6.1 Paramètres Recommandés

| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| `turnover_threshold` | 0.08 | 8% de changement moyen (~2 repos/jour, ~50%/an de coûts) |
| `turnover_lambda_min` | 1.0 | Pas de sur-pénalisation de base |
| `turnover_lambda_max` | 5.0 | Plafond conservateur |
| `turnover_Kp` | 2.5 | Réaction modérée |
| `turnover_Ki` | 0.08 | Mémoire pour patterns persistants |
| `turnover_Kd` | 0.4 | Amortissement modéré |
| `decay_rate` | 0.995 | Retour lent (~320 steps) |
| `max_lambda_change` | 0.08 | Équilibre réactivité/stabilité |

### 6.2 Analyse du Seuil de Turnover

| Turnover/step | Trades/jour équivalent | Coût annuel (~0.07%/trade) | Profil |
|---------------|------------------------|----------------------------|--------|
| 0.05 | ~1-2 | ~30%/an | Très conservateur |
| **0.08** | ~2-3 | **~50%/an** | **Recommandé** |
| 0.15 | ~3-5 | ~90%/an | Actif (coûteux) |
| 0.30 | ~7-10 | ~180%/an | Hyperactif (ruineux) |

### 6.3 Comparaison des 3 PLO

| Aspect | PLO Drawdown | PLO Smoothness | PLO Churn |
|--------|--------------|----------------|-----------|
| Métrique | Drawdown (%) | Jerk (Δ²pos) | Turnover (|Δpos|) |
| Seuil | 10% | 0.40 | 0.08 |
| Prédiction | Oui | Non | Oui |
| Decay rate | 0.995 | 0.99 | 0.995 |
| max_λ_change | 0.05 | 0.1 | 0.08 |
| Réactivité | Prudente | Immédiate | Modérée |
| Fenêtre | 50 steps | Instantané | 20 steps |

---

## 7. Validation et Tests

### 7.1 Tests Unitaires

```python
def test_plo_churn_increases_on_high_turnover():
    """λ doit augmenter quand turnover > threshold."""
    callback = PLOChurnCallback(turnover_threshold=0.08)
    # Simuler turnover = 0.30 (hyperactif)
    # ... vérifier que turnover_lambda > 1.0

def test_plo_churn_leak_minimum():
    """PLO doit forcer un leak minimum si λ > 1."""
    # Simuler step non profitable (base_gate = 0) avec λ = 5.0
    # → effective_gate doit être 0.2 (20% leak)
    # → pénalité effective = raw_churn × 0.2 × 5.0 = raw_churn × 1.0
    
def test_plo_churn_no_leak_if_lambda_1():
    """Pas de leak si λ = 1.0 (PLO inactif)."""
    # Simuler step non profitable (base_gate = 0) avec λ = 1.0
    # → effective_gate = 0 (pas de leak)
    # → pénalité effective = 0

def test_plo_churn_prediction():
    """Prédiction active seulement si slope > 0."""
    callback = PLOChurnCallback()
    # Simuler turnover décroissant
    # ... vérifier que predicted_violation = 0

def test_plo_churn_curriculum_protection():
    """λ ne doit pas monter si churn_coef ≈ 0."""
    callback = PLOChurnCallback()
    # Simuler Phase 1 (churn_coef < 0.05)
    # ... vérifier que turnover_integral décroît
```

### 7.2 Métriques TensorBoard

| Métrique | Description |
|----------|-------------|
| `plo_churn/turnover_violation` | Excès de turnover |
| `plo_churn/turnover_predicted` | Violation prédite |
| `plo_churn/turnover_multiplier` | λ actuel |
| `plo_churn/turnover_integral` | Terme I accumulé |
| `plo_churn/turnover_slope` | Pente de la régression |
| `plo_churn/metric_turnover` | Turnover moyen 20 steps |
| `plo_churn/curriculum_active` | Flag d'activité |
| `plo_churn/effective_gate_mean` | **NOUVEAU (v1.1)**: Gate effectif avec leak |

### 7.3 Critères de Succès

| Métrique | Cible |
|----------|-------|
| Turnover moyen | < 0.10 |
| Trades/jour | 2-3 |
| Coûts de transaction | < 5% du PnL |
| % steps avec turnover > 0.40 | < 10% |

---

## 8. Audit et Corrections (v1.1)

### 8.1 Risque Critique Identifié : "Paradoxe du Profit Gate"

**Problème** : Dans la version originale, si l'agent "churne à perte" (trade frénétiquement avec des returns négatifs) :

```
step_returns < 0 → gate = 0 → churn_penalty = 0
```

Le PLO détecte le turnover élevé et monte λ à 5.0, mais `pénalité × 0 = 0`. Le mécanisme est **impuissant**.

**Impact** : L'agent pourrait apprendre à ignorer le signal `churn_level` tant qu'il perd de l'argent.

### 8.2 Correction Appliquée : "Leak Minimum"

**Solution** : Si le PLO est actif (λ > 1), on force un "leak" minimum dans le gate :

```python
if self.churn_multiplier > 1.0:
    plo_intensity = (self.churn_multiplier - 1.0) / 4.0  # [0, 1]
    min_gate = 0.2 * plo_intensity  # 0% si λ=1, 20% si λ=5
    effective_gate = max(base_gate, min_gate)
```

| λ | plo_intensity | min_gate | Effet |
|---|---------------|----------|-------|
| 1.0 | 0% | 0% | Gate normal (pas de leak) |
| 2.0 | 25% | 5% | Léger leak |
| 3.0 | 50% | 10% | Leak modéré |
| 5.0 | 100% | 20% | Leak maximum |

### 8.3 Autres Corrections

| Issue | Correction |
|-------|------------|
| **Polyfit bruité** | Ajout d'un seuil minimum `MIN_SLOPE = 0.001` pour ignorer les pentes négligeables |
| **Cycle de vie `prev_position_pcts`** | Documenté explicitement : doit être mis à jour **après** le calcul des rewards |

### 8.4 Risque Accepté : Batch Mean

Le calcul utilise `current_deltas.mean()` sur tout le batch. Un seul env "fou" (turnover 100%) ne déclenche pas le PLO si les 63 autres sont calmes.

**Décision** : Accepté. Le PLO régule la *politique moyenne*, pas les instances individuelles. C'est cohérent avec l'approche Policy Gradient.

---

## Annexe A : Interaction Gate + PLO (Corrigée v1.1)

Le flow complet avec le **leak minimum** :

```
|Δposition| ────────────────────────────┐
                                        │
cost_rate × churn_coef × SCALE ────────┼──▶ raw_churn
                                        │
                                        ▼
step_returns ──▶ base_gate = clamp(ret/0.5%, 0, 1)
                        │
                        ▼
              ┌─────────────────────────────────────────────┐
              │  LEAK MINIMUM (v1.1)                        │
              │                                             │
              │  if λ > 1:                                  │
              │      plo_intensity = (λ - 1) / 4           │
              │      min_gate = 0.2 × plo_intensity         │
              │      effective_gate = max(base_gate, min)   │
              │  else:                                      │
              │      effective_gate = base_gate             │
              └─────────────────────────────────────────────┘
                        │
                        ▼
                 gated_churn = raw_churn × effective_gate
                        │
                        ▼
                 plo_churn = gated_churn × churn_multiplier ◀── PLO
                        │
                        ▼
                 safe_churn = clamp(plo_churn, max=15.0) ◀── SAFETY
                        │
                        ▼
                 curriculum_λ × safe_churn ──▶ Reward
```

**Séquence de protection (v1.1)** :
1. **Base Gate** : Calcul classique (0 si step non profitable)
2. **Leak Minimum** : Force un minimum si PLO actif (évite paradoxe)
3. **PLO** : Multiplicateur adaptatif selon le turnover
4. **Safety Clip** : Plafond pour éviter les NaN
5. **Curriculum** : Montée progressive des pénalités

---

## Annexe B : Utilisation Combinée des 3 PLO

Si vous utilisez **les trois PLO** (Drawdown + Smoothness + Churn), l'observation complète devient :

```python
self.observation_space = spaces.Dict({
    "market": ...,
    "position": ...,
    "risk_level": ...,      # PLO Drawdown (λ_dd)
    "smooth_level": ...,    # PLO Smoothness (λ_smooth)
    "churn_level": ...      # PLO Churn (λ_churn)
})
```

**Indépendance** : Les trois contrôleurs sont totalement indépendants et peuvent être activés/désactivés individuellement.

**Priorité suggérée** :
1. **PLO Drawdown** : Le plus critique (risque financier)
2. **PLO Churn** : Coûts de transaction
3. **PLO Smoothness** : Qualité des exécutions

---

**Conclusion** : Ce PLO Churn cible spécifiquement l'hyperactivité de trading. La version 1.1 corrige le "Paradoxe du Profit Gate" via un mécanisme de "leak minimum", garantissant que le PLO reste efficace même quand l'agent churne à perte. Il s'intègre harmonieusement avec le gate profit-conditionné existant tout en évitant ses pièges logiques.

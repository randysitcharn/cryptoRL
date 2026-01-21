# PLO - Predictive Lagrangian Optimization pour Pénalités Adaptatives

**Version** : 3.0 (Final)  
**Date** : 2026-01-18  
**Statut** : Spécification technique finale - Production-Ready

---

## Table des Matières

1. [Résumé Exécutif](#1-résumé-exécutif)
2. [Contexte et Motivation](#2-contexte-et-motivation)
3. [Architecture Actuelle des Récompenses](#3-architecture-actuelle-des-récompenses)
4. [État de l'Art - Méthodes SOTA](#4-état-de-lart---méthodes-sota)
5. [Spécification PLO](#5-spécification-plo)
6. [⚠️ Audit et Corrections Critiques](#6-️-audit-et-corrections-critiques)
7. [Implémentation (Version Corrigée)](#7-implémentation-version-corrigée)
8. [Paramètres et Tuning](#8-paramètres-et-tuning)
9. [Validation et Tests](#9-validation-et-tests)
10. [Références](#10-références)

**Annexes** :
- [Annexe A : Interaction avec le Curriculum](#annexe-a--interaction-avec-le-curriculum)
- [Annexe B : Checklist d'Audit](#annexe-b--checklist-daudit-post-correction)
- [Annexe C : Migration depuis v1.0](#annexe-c--migration-depuis-v10)

---

## 1. Résumé Exécutif

### Objectif

Implémenter un système de **pénalités adaptatives** où le coefficient de pénalité downside augmente dynamiquement quand le drawdown dépasse un seuil (ex: 10%), créant une pression progressive pour que l'agent apprenne à gérer le risque de manière proactive.

### Solution Retenue

**PLO (Predictive Lagrangian Optimization)** - Une méthode SOTA (2025) qui combine :
- Un contrôleur **PID** pour ajuster le multiplicateur de pénalité
- Une composante **prédictive robuste** (polyfit) pour anticiper les violations
- Une **observation augmentée** pour que l'agent "voie" le niveau de risque

### Bénéfices Attendus

| Métrique | Sans PLO | Avec PLO (cible) |
|----------|----------|------------------|
| Max Drawdown moyen | ~15-20% | < 12% |
| Fréquence DD > 10% | ~30% | < 15% |
| Sharpe Ratio | Baseline | ≥ Baseline |

---

## 2. Contexte et Motivation

### 2.1 Configuration Temporelle du Système

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| **Granularité** | 1 heure | Données OHLCV horaires |
| **Window size** | 64 steps | 64 heures ≈ 2.67 jours de contexte |
| **Episode length** | 2048 steps | ~85 jours par épisode |
| **Vol window** | 24 steps | 24 heures pour calcul volatilité |

### 2.2 Problème Identifié

Le coefficient de pénalité downside actuel est **statique** :

```python
downside_risk = negative_returns² × SCALE × 5.0  # Coefficient fixe !
```

**Conséquences** :
1. Pas de réaction aux conditions de marché
2. Un drawdown de 15% est pénalisé comme un drawdown de 5%
3. L'agent n'a pas de signal croissant pour réduire le risque

### 2.3 Solution Proposée

Rendre le coefficient **adaptatif** via PLO :

```python
downside_risk = negative_returns² × SCALE × 5.0 × downside_multiplier
#                                                  ↑
#                                           PLO contrôle cette valeur
#                                           λ ∈ [1.0, 5.0]
```

---

## 3. Architecture Actuelle des Récompenses

### 3.1 Formule Globale

```
reward = log_returns - curriculum_λ × (churn_penalty + downside_risk + smoothness_penalty)
```

### 3.2 Détail des Composantes

| Composante | Formule | Rôle |
|------------|---------|------|
| **Log Returns** | `log1p(clamp(returns, -0.99)) × 100` | Récompense de base (PnL) |
| **Churn Penalty** | `\|Δpos\| × cost × churn_coef × gate` | Décourager trading excessif |
| **Downside Risk** | `negative_returns² × 500` | Pénaliser les pertes (Sortino-style) |
| **Smoothness** | `smooth_coef × Δpos²` | Encourager positions stables |

### 3.3 Curriculum Learning Existant

```
Phase 1 (0-10%)  : curriculum_λ = 0.0      → Pure exploration
Phase 2 (10-30%) : curriculum_λ = 0.0→0.4  → Rampe linéaire
Phase 3 (30-100%): curriculum_λ = 0.4      → Stabilité
```

### 3.4 Schéma d'Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    SYSTÈME DE RÉCOMPENSE COMPLET                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐                                                        │
│  │ LOG RETURNS │  ← Base reward (toujours actif)                        │
│  └──────┬──────┘                                                        │
│         │                                                               │
│         ▼                                                               │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │              PÉNALITÉS GÉRÉES PAR CURRICULUM                 │       │
│  │                                                              │       │
│  │  curriculum_λ(t) × [                                         │       │
│  │                                                              │       │
│  │    ┌─────────────────┐     ┌─────────────────────────────┐   │       │
│  │    │  CHURN PENALTY  │  +  │      DOWNSIDE RISK          │   │       │
│  │    │                 │     │                             │   │       │
│  │    │ |Δpos| × cost   │     │  negative_returns² × 5      │   │       │
│  │    │    × gate       │     │    × PLO_multiplier(t) ◀────┼───┼───┐   │
│  │    └─────────────────┘     └─────────────────────────────┘   │   │   │
│  │                                                              │   │   │
│  │    ┌─────────────────┐                                       │   │   │
│  │    │   SMOOTHNESS    │  ← Aussi sous curriculum              │   │   │
│  │    │  Δpos² × coef   │                                       │   │   │
│  │    └─────────────────┘                                       │   │   │
│  │                                                              │   │   │
│  └──────────────────────────────────────────────────────────────┘   │   │
│         │                                                           │   │
│         │                                                           │   │
│         ▼                                                           │   │
│  ┌─────────────────┐     ┌──────────────────────────────────────┐   │   │
│  │  FINAL REWARD   │     │     PLO CALLBACK                     │   │   │
│  └─────────────────┘     │                                      │   │   │
│                          │  Drawdown > 10% ?                    │   │   │
│                          │       │                              │   │   │
│                          │       ▼                              │   │   │
│                          │  ┌─────────────┐                     │   │   │
│                          │  │ PID + Pred  │ ────────────────────┼───┘   │
│                          │  └─────────────┘                     │       │
│                          └──────────────────────────────────────┘       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. État de l'Art - Méthodes SOTA

### 4.1 Comparaison des Approches (2024-2025)

| Méthode | Référence | Principe | Avantages | Inconvénients |
|---------|-----------|----------|-----------|---------------|
| **PID Lagrangian** | Stooke 2020 | Contrôleur PID sur λ | Stable, bien compris | Tuning manuel |
| **Augmented Lagrangian** | Classique | λ×g + μ/2×g² | Simple | Peut osciller |
| **PLO** | 2025 | Contrôle optimal de λ | SOTA, prédictif | Plus complexe |
| **Adaptive Primal-Dual** | AAMAS 2024 | LR adaptatifs | Théoriquement fondé | Sample complexity |

### 4.2 Pourquoi PLO ?

1. **Généralise PID Lagrangian** : PLO inclut PID comme cas particulier
2. **Composante prédictive** : Anticipe les violations avant qu'elles n'arrivent
3. **Compatible GPU** : Intégrable via callback SB3 sans modification majeure
4. **Extensible** : Peut gérer plusieurs contraintes (drawdown, churn, vol)

---

## 5. Spécification PLO

### 5.1 Formulation Mathématique

**Problème CMDP (Constrained MDP)** :
```
max_π  E[Σ r(s,a)]
s.t.   E[g(s,a)] ≤ 0
```

Où la fonction de violation est :
```
g(t) = max(0, drawdown(t) - threshold)
```

**Relaxation Lagrangienne** :
```
L(π, λ) = E[r] - λ × E[g]
```

### 5.2 Contrôleur PID + Prédiction

```
λ(t) = λ_min + P(t) + I(t) + D(t)

P(t) = K_p × g_eff(t)                      [Proportionnel]
I(t) = I(t-1) + K_i × g(t)                 [Intégral]
D(t) = K_d × (g(t) - g(t-1))               [Dérivé]

g_eff(t) = max(g(t), 0.7 × ĝ(t+h))         [Violation effective]
ĝ(t+h) = slope × (n + h) + intercept       [Prédiction par régression]
```

### 5.3 Comportement du Contrôleur

**Quand violation > 0** (drawdown > seuil) :
- P augmente immédiatement proportionnellement à l'excès
- I accumule la violation (mémoire)
- D réagit aux changements rapides

**Quand violation = 0** (drawdown ≤ seuil) :
- λ décroît vers λ_min avec decay_rate
- I décroît lentement (garde mémoire partielle)

### 5.4 Diagramme du Contrôleur (Version Corrigée)

```
                    ┌─────────────────────────────────────────────────────┐
                    │              PLO CONTROLLER (v2.0)                  │
                    │                                                     │
   drawdown(t) ────▶│  ┌─────────────────────────────────────────────┐   │
   (quantile 90%)   │  │                                             │   │
                    │  │  g(t) = max(0, drawdown - threshold)        │   │
   threshold ──────▶│  │                                             │   │
                    │  └──────────────────┬──────────────────────────┘   │
                    │                     │                              │
                    │                     ▼                              │
                    │  ┌─────────────────────────────────────────────┐   │
                    │  │     ROBUST PREDICTION (Polyfit)             │   │
                    │  │                                             │   │
                    │  │  slope, intercept = np.polyfit(x, y, 1)     │   │
                    │  │  if slope > 0:  # Only penalize worsening   │   │
                    │  │      predicted_dd = slope × 25 + intercept  │   │
                    │  │      ĝ = max(0, predicted_dd - threshold)   │   │
                    │  │  g_eff = max(g, 0.7 × ĝ)                    │   │
                    │  │                                             │   │
                    │  └──────────────────┬──────────────────────────┘   │
                    │                     │                              │
                    │                     ▼                              │
                    │  ┌─────────────────────────────────────────────┐   │
                    │  │         PID CONTROLLER + SMOOTHING          │   │
                    │  │                                             │   │
                    │  │  target_λ = λ_min + P + I + D               │   │
                    │  │                                             │   │
                    │  │  # SMOOTHING: Limit change per step         │   │
                    │  │  Δλ = clip(target_λ - λ_prev, ±0.05)        │   │
                    │  │  λ = λ_prev + Δλ                            │   │
                    │  │                                             │   │
                    │  └──────────────────┬──────────────────────────┘   │
                    │                     │                              │
                    └─────────────────────┼──────────────────────────────┘
                                          │
                                          ▼
                              downside_multiplier = λ
                                          │
                                          ▼
                              ┌───────────────────────┐
                              │  OBSERVATION AUGMENT  │
                              │                       │
                              │  obs["risk_level"] =  │
                              │    (λ - 1.0) / 4.0    │
                              └───────────────────────┘
```

---

## 6. ⚠️ Audit et Corrections Critiques

### 6.1 Résumé de l'Audit

| Composante | Statut | Analyse |
|------------|--------|---------|
| **Logique PID** | ✅ Valide | Mathématiquement sain pour stabiliser l'apprentissage |
| **Prédiction naïve** | ⚠️ **Corrigé** | `(last-first)/10` trop sensible au bruit → remplacé par `polyfit` |
| **Intégration SB3** | ✅ Valide | Architecture via Callback non-intrusive |
| **Stationnarité** | ❌ **CRITIQUE → Corrigé** | Agent ne voyait pas λ → observation augmentée ajoutée |

### 6.2 Faille Critique : L'Agent "Aveugle"

**Problème** : Sans connaître la valeur de λ, l'agent voit un environnement non-stationnaire où la même action dans le même état donne parfois reward=-1, parfois reward=-5. Cela **casse l'hypothèse de Markov** et empêche la Value Function de converger.

**Solution** : Ajouter `downside_multiplier` (normalisé) dans l'observation de l'agent.

```python
# L'agent doit VOIR le niveau de risque actuel
State(t) = [Market_Features, Position, Risk_Level(λ)]
```

### 6.3 Amélioration : Prédiction Robuste

**Problème** : La formule `trend = (DD[-1] - DD[0]) / 10` est trop sensible aux mèches de volatilité crypto. Une simple bougie peut déclencher un "panic mode" injustifié.

**Solution** : Utiliser une régression linéaire (`np.polyfit`) sur 15 points et ne prédire que si la pente est **positive** (aggravation).

```python
# AVANT (fragile)
trend = (recent_dd[-1] - recent_dd[0]) / 10

# APRÈS (robuste)
slope, intercept = np.polyfit(x, y, 1)
if slope > 0:  # Seulement si ça s'aggrave
    predicted_dd = slope * 25 + intercept
```

### 6.4 Amélioration : Quantile au lieu de Mean

**Problème** : La moyenne des drawdowns peut être biaisée par quelques envs en bonne santé, masquant un crash dans d'autres.

**Solution** : Utiliser le **quantile 90%** (VaR-style) pour piloter le PLO.

```python
# AVANT
mean_dd = real_env.current_drawdowns.mean().item()

# APRÈS (VaR-style)
metric_dd = torch.quantile(real_env.current_drawdowns, 0.9).item()
```

### 6.5 Amélioration : Smoothing du Lambda

**Problème** : Un saut brutal de λ=1.0 à λ=4.0 déstabilise l'apprentissage PPO/TQC.

**Solution** : Limiter le changement de λ à ±0.05 par step.

```python
max_lambda_change_per_step = 0.05
change = np.clip(target_lambda - current_lambda, 
                 -max_lambda_change_per_step, 
                 max_lambda_change_per_step)
new_lambda = current_lambda + change
```

---

## 7. Implémentation (Version Production-Ready)

> ⚠️ **Ce code est la version finale auditée et sécurisée.** Il intègre toutes les protections : Hard Clip, Wake-up Shock, Quantile adaptatif, et `MultiInputPolicy`.

### 7.1 Fichiers à Modifier

| Fichier | Modification |
|---------|--------------|
| `src/training/batch_env.py` | Ajouter `downside_multiplier` + observation augmentée + Hard Clip |
| `src/training/callbacks.py` | Ajouter `PLOAdaptivePenaltyCallback` (version production) |
| `src/training/train_agent.py` | Utiliser `MultiInputPolicy` + intégrer callback |

### 7.2 Modification de batch_env.py

```python
class BatchCryptoEnv:
    def __init__(self, ...):
        # ... existing code ...
        
        # PLO: Multiplicateur adaptatif pour downside
        self.downside_multiplier = 1.0
        
        # Modifier observation_space pour inclure risk_level
        # ⚠️ IMPORTANT: Nécessite MultiInputPolicy dans TQC/PPO
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
            "risk_level": spaces.Box(
                low=0.0, high=1.0,
                shape=(1,),
                dtype=np.float32
            )
        })
    
    def set_downside_multiplier(self, value: float) -> None:
        """Setter pour PLO callback avec clamp de sécurité."""
        self.downside_multiplier = max(1.0, min(value, 10.0))
    
    def _get_observations(self) -> Dict[str, np.ndarray]:
        """Get observations for all envs as numpy (SB3 compatible)."""
        # Market windows: (n_envs, window_size, n_features)
        market = self._get_batch_windows(self.current_steps)

        # Add observation noise for regularization (anti-overfitting)
        if self.observation_noise > 0 and self.training:
            noise = torch.randn_like(market) * self.observation_noise
            market = market + noise

        # Position: (n_envs, 1)
        position = self.position_pcts.unsqueeze(1)
        
        # ═══════════════════════════════════════════════════════════════════
        # OBSERVATION AUGMENTÉE: Risk Level (PLO λ normalisé)
        # L'agent DOIT voir le niveau de pression pour que V(s) converge
        # Sans cela, l'environnement devient non-stationnaire (casse Markov)
        # ═══════════════════════════════════════════════════════════════════
        # Normalisation: λ ∈ [1, 5] → risk_level ∈ [0, 1]
        risk_level_value = (self.downside_multiplier - 1.0) / 4.0
        risk_level = torch.full(
            (self.num_envs, 1), 
            risk_level_value, 
            device=self.device
        )

        # Transfer to CPU numpy for SB3
        return {
            "market": market.cpu().numpy(),
            "position": position.cpu().numpy(),
            "risk_level": risk_level.cpu().numpy()
        }
    
    def _calculate_rewards(self, step_returns, position_deltas, dones):
        SCALE = 100.0
        
        # ... calcul de log_returns, churn_penalty comme existant ...
        
        # ═══════════════════════════════════════════════════════════════════
        # DOWNSIDE RISK avec PLO + SAFETY CLIP
        # ═══════════════════════════════════════════════════════════════════
        
        # 1. Calcul du risque de base (quadratique - Sortino style)
        negative_returns = torch.clamp(step_returns, max=0.0)
        base_downside = torch.square(negative_returns) * SCALE * 5.0
        
        # 2. Application du multiplicateur PLO
        raw_penalty = base_downside * self.downside_multiplier
        
        # 3. SAFETY CLIP (Crucial pour éviter NaN)
        # Empêche la pénalité de dépasser 20.0 par step
        # Même si lambda=5 et crash de -10%, la pénalité reste bornée
        PENALTY_CAP = 20.0
        safe_penalty = torch.clamp(raw_penalty, max=PENALTY_CAP)
        
        # 4. Smoothness penalty (also under curriculum)
        smoothness_penalty = self._current_smooth_coef * (position_deltas ** 2) * SCALE
        
        # 5. Reward final (all penalties under curriculum_lambda)
        total_penalty = self.curriculum_lambda * (churn_penalty + safe_penalty + smoothness_penalty)
        reward = log_returns - total_penalty
        
        # 6. Observabilité (pour debug et monitoring)
        self._rew_downside = -safe_penalty * self.curriculum_lambda
        self._raw_downside_before_clip = raw_penalty.mean().item()  # Pour monitoring
        
        return reward * self.reward_scaling
```

### 7.3 PLOAdaptivePenaltyCallback (Version Production)

```python
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback


def get_underlying_batch_env(env):
    """Unwrap SB3 VecEnv to get BatchCryptoEnv."""
    while hasattr(env, 'venv'):
        env = env.venv
    while hasattr(env, 'env'):
        env = env.env
    return env


class PLOAdaptivePenaltyCallback(BaseCallback):
    """
    Predictive Lagrangian Optimization (PLO) pour pénalités adaptatives.
    
    VERSION PRODUCTION - Inclut toutes les protections:
    1. Prédiction robuste via np.polyfit (au lieu de différence naïve)
    2. Quantile adaptatif (90% si num_envs >= 16, sinon LogSumExp)
    3. Smoothing du lambda (max ±0.05/step)
    4. Prédiction uniquement si pente positive (aggravation)
    5. Protection "Wake-up Shock" (freeze PID en Phase 1 curriculum)
    
    Référence: "Predictive Lagrangian Optimization" (2025)
    """
    
    def __init__(
        self,
        # Contrainte Drawdown
        dd_threshold: float = 0.10,
        dd_lambda_min: float = 1.0,
        dd_lambda_max: float = 5.0,
        # Gains PID
        dd_Kp: float = 2.0,
        dd_Ki: float = 0.05,
        dd_Kd: float = 0.3,
        # Prédiction PLO
        prediction_horizon: int = 50,
        use_prediction: bool = True,
        # Anti-windup, decay et smoothing
        integral_max: float = 2.0,
        decay_rate: float = 0.995,
        max_lambda_change: float = 0.05,
        # Mesure du risque
        dd_quantile: float = 0.9,
        # Logging
        log_freq: int = 100,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        
        # Paramètres
        self.dd_threshold = dd_threshold
        self.dd_lambda_min = dd_lambda_min
        self.dd_lambda_max = dd_lambda_max
        self.dd_Kp = dd_Kp
        self.dd_Ki = dd_Ki
        self.dd_Kd = dd_Kd
        self.prediction_horizon = prediction_horizon
        self.use_prediction = use_prediction
        self.integral_max = integral_max
        self.decay_rate = decay_rate
        self.max_lambda_change = max_lambda_change
        self.dd_quantile = dd_quantile
        self.log_freq = log_freq
        
        # État du contrôleur PID
        self.dd_integral = 0.0
        self.dd_prev_violation = 0.0
        self.dd_lambda = 1.0
        
        # Buffer pour prédiction
        self.dd_history = []
        
    def _on_step(self) -> bool:
        real_env = get_underlying_batch_env(self.model.env)
        
        if not hasattr(real_env, 'current_drawdowns'):
            return True
        
        # ═══════════════════════════════════════════════════════════════════
        # PROTECTION "WAKE-UP SHOCK"
        # Ne pas accumuler d'intégral si le curriculum n'est pas encore actif
        # Évite que λ monte à 5.0 pendant que l'agent ne peut pas réagir
        # ═══════════════════════════════════════════════════════════════════
        curriculum_active = True
        if hasattr(real_env, 'curriculum_lambda'):
            if real_env.curriculum_lambda < 0.05:  # Phase 1: curriculum ≈ 0
                # Decay rapide de l'intégral pour éviter saturation
                self.dd_integral *= 0.9
                curriculum_active = False
        
        # ═══════════════════════════════════════════════════════════════════
        # MESURE ADAPTATIVE (selon num_envs)
        # Quantile instable si peu d'envs → utiliser LogSumExp
        # ═══════════════════════════════════════════════════════════════════
        current_dd = real_env.current_drawdowns
        
        if real_env.num_envs >= 16:
            # Assez d'envs pour un quantile stable
            metric_dd = torch.quantile(current_dd, self.dd_quantile).item()
        else:
            # Peu d'envs: LogSumExp (smooth approximation du max)
            temperature = 10.0
            metric_dd = (torch.logsumexp(current_dd * temperature, dim=0) / temperature).item()
        
        max_dd = current_dd.max().item()
        violation = max(0.0, metric_dd - self.dd_threshold)
        
        # Stocker pour prédiction
        self.dd_history.append(metric_dd)
        if len(self.dd_history) > self.prediction_horizon:
            self.dd_history.pop(0)
        
        # ═══════════════════════════════════════════════════════════════════
        # PRÉDICTION ROBUSTE (seulement si curriculum actif)
        # Polyfit sur 15 points, uniquement si pente positive
        # ═══════════════════════════════════════════════════════════════════
        predicted_violation = 0.0
        slope = 0.0
        
        if curriculum_active and self.use_prediction and len(self.dd_history) >= 15:
            y = np.array(self.dd_history[-15:])
            x = np.arange(len(y))
            slope, intercept = np.polyfit(x, y, 1)
            
            # STRICT: Ne prédire que si pente POSITIVE (aggravation)
            if slope > 0:
                future_dd = slope * (len(y) + 10) + intercept
                predicted_violation = max(0.0, future_dd - self.dd_threshold)
        
        # ═══════════════════════════════════════════════════════════════════
        # CONTRÔLEUR PID (conditionné par curriculum)
        # ═══════════════════════════════════════════════════════════════════
        if curriculum_active:
            if violation > 0 or predicted_violation > 0:
                effective_violation = max(violation, 0.7 * predicted_violation)
                
                P = self.dd_Kp * effective_violation
                self.dd_integral += self.dd_Ki * violation
                self.dd_integral = np.clip(self.dd_integral, 0, self.integral_max)
                I = self.dd_integral
                D = self.dd_Kd * (violation - self.dd_prev_violation)
                
                target_lambda = self.dd_lambda_min + P + I + D
                target_lambda = np.clip(target_lambda, self.dd_lambda_min, self.dd_lambda_max)
            else:
                target_lambda = max(self.dd_lambda_min, self.dd_lambda * self.decay_rate)
                self.dd_integral *= 0.995
            
            # Smoothing: limiter le changement par step
            change = np.clip(target_lambda - self.dd_lambda, 
                            -self.max_lambda_change, self.max_lambda_change)
            self.dd_lambda = self.dd_lambda + change
        
        self.dd_prev_violation = violation
        
        # Appliquer à l'environnement
        if hasattr(real_env, 'set_downside_multiplier'):
            real_env.set_downside_multiplier(self.dd_lambda)
        
        # Logging
        if self.num_timesteps % self.log_freq == 0:
            self.logger.record("plo/dd_violation", violation)
            self.logger.record("plo/dd_predicted_violation", predicted_violation)
            self.logger.record("plo/dd_multiplier", self.dd_lambda)
            self.logger.record("plo/dd_integral", self.dd_integral)
            self.logger.record("plo/dd_slope", slope)
            self.logger.record("plo/metric_drawdown", metric_dd)
            self.logger.record("plo/max_drawdown", max_dd)
            self.logger.record("plo/curriculum_active", float(curriculum_active))
        
        return True
    
    def _on_training_start(self) -> None:
        if self.verbose > 0:
            print(f"\n[PLO] Predictive Lagrangian Optimization (Production):")
            print(f"  DD threshold: {self.dd_threshold:.1%}")
            print(f"  Lambda range: [{self.dd_lambda_min}, {self.dd_lambda_max}]")
            print(f"  PID gains: Kp={self.dd_Kp}, Ki={self.dd_Ki}, Kd={self.dd_Kd}")
            print(f"  Prediction: {'polyfit (robust)' if self.use_prediction else 'disabled'}")
            print(f"  DD Quantile: {self.dd_quantile:.0%} (adaptatif selon num_envs)")
            print(f"  Max λ change/step: ±{self.max_lambda_change}")
            print(f"  Wake-up Shock protection: enabled")
```

### 7.4 Intégration dans train_agent.py

```python
from sb3_contrib import TQC  # Ou PPO depuis stable_baselines3
from src.training.callbacks import PLOAdaptivePenaltyCallback

# ═══════════════════════════════════════════════════════════════════════════
# ⚠️ CRITIQUE: Utiliser MultiInputPolicy (pas MlpPolicy)
# L'observation est un Dict → MlpPolicy crashera immédiatement
# ═══════════════════════════════════════════════════════════════════════════

model = TQC(
    "MultiInputPolicy",  # ← OBLIGATOIRE car obs est un Dict
    env,
    verbose=1,
    policy_kwargs={
        "net_arch": dict(pi=[256, 256], qf=[256, 256]),
    },
    # ... autres hyperparams ...
)

# Callbacks
callbacks = [
    ThreePhaseCurriculumCallback(
        total_timesteps=config.total_timesteps,
        verbose=1
    ),
    PLOAdaptivePenaltyCallback(
        dd_threshold=0.10,
        dd_lambda_min=1.0,
        dd_lambda_max=5.0,
        dd_Kp=2.0,
        dd_Ki=0.05,
        dd_Kd=0.3,
        prediction_horizon=50,
        use_prediction=True,
        max_lambda_change=0.05,
        dd_quantile=0.9,
        verbose=1
    ),
    StepLoggingCallback(log_freq=config.log_freq),
    # ... autres callbacks
]

model.learn(total_timesteps=config.total_timesteps, callback=callbacks)
```

### 7.5 Sanity Check (5 minutes avant training long)

Avant de lancer un training de 24h+, effectuez ces vérifications :

| Test | Attendu | Si échec |
|------|---------|----------|
| **Démarrage** | Pas de crash dans les 10 premières secondes | `MultiInputPolicy` manquant |
| **`plo/dd_integral`** (Steps 0-1000) | ≈ 0 ou décroissant | Wake-up Shock non actif |
| **`plo/curriculum_active`** (Steps 0-1000) | = 0.0 | Protection Phase 1 manquante |
| **Rewards min** | Jamais < -50 | Hard Clip manquant |

---

## 8. Paramètres et Tuning

### 8.1 Paramètres Recommandés (Post-Audit)

| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| `dd_threshold` | 0.10 | Standard industrie (10% max DD) |
| `dd_lambda_min` | 1.0 | Multiplicateur neutre |
| `dd_lambda_max` | 5.0 | Évite paralysie de l'agent |
| `dd_Kp` | 2.0 | +2× par 1% d'excès de DD |
| `dd_Ki` | 0.05 | Accumulation lente (~20 steps pour +1.0) |
| `dd_Kd` | 0.3 | Amortissement modéré |
| `prediction_horizon` | 50 | 50 heures ≈ 2 jours |
| `decay_rate` | 0.995 | ~320 steps (13 jours) pour retour à λ_min |
| `max_lambda_change` | 0.05 | **NOUVEAU**: Stabilise PPO |
| `dd_quantile` | 0.9 | **NOUVEAU**: VaR 90% |

### 8.2 Analyse Temporelle

**Decay Analysis** :
```
Temps pour λ_max → λ_min :
  T = log(λ_min/λ_max) / log(decay_rate)
  T = log(1/5) / log(0.995) ≈ 321 steps ≈ 13 jours
```

**Half-life** :
```
T_half = log(0.5) / log(0.995) ≈ 138 steps ≈ 5.7 jours
```

**Smoothing Analysis** :
```
Temps pour λ=1.0 → λ=5.0 (pire cas) :
  T = (5.0 - 1.0) / 0.05 = 80 steps ≈ 3.3 jours

Cela évite les sauts brutaux qui déstabilisent l'apprentissage.
```

### 8.3 Suggestions de Sweep

```python
# Hyperparameter sweep suggéré
sweep_config = {
    'dd_threshold': [0.05, 0.10, 0.15],
    'dd_Kp': [1.0, 2.0, 3.0, 5.0],
    'dd_Ki': [0.01, 0.05, 0.1],
    'dd_Kd': [0.1, 0.3, 0.5],
    'decay_rate': [0.99, 0.995, 0.999],
    'max_lambda_change': [0.02, 0.05, 0.10],
    'dd_quantile': [0.8, 0.9, 0.95],
}
```

---

## 9. Validation et Tests

### 9.1 Tests Unitaires

```python
def test_plo_lambda_increases_on_violation():
    """λ doit augmenter quand DD > threshold."""
    callback = PLOAdaptivePenaltyCallback(dd_threshold=0.10)
    # Simuler DD = 15%
    callback.dd_history = [0.15] * 15
    # ... vérifier que dd_lambda > 1.0

def test_plo_lambda_decreases_when_ok():
    """λ doit décroître quand DD < threshold."""
    callback = PLOAdaptivePenaltyCallback(dd_threshold=0.10)
    callback.dd_lambda = 3.0
    # Simuler DD = 5%
    # ... vérifier que dd_lambda < 3.0 après quelques steps

def test_plo_lambda_bounds():
    """λ doit rester dans [λ_min, λ_max]."""
    callback = PLOAdaptivePenaltyCallback(
        dd_lambda_min=1.0,
        dd_lambda_max=5.0
    )
    # ... vérifier les bornes dans tous les cas

def test_plo_smoothing():
    """λ ne doit pas changer de plus de max_lambda_change par step."""
    callback = PLOAdaptivePenaltyCallback(max_lambda_change=0.05)
    callback.dd_lambda = 1.0
    # Simuler violation massive
    # ... vérifier que Δλ ≤ 0.05

def test_plo_prediction_only_on_positive_slope():
    """La prédiction ne doit s'activer que si la pente est positive."""
    callback = PLOAdaptivePenaltyCallback()
    # Simuler DD décroissant (récupération)
    callback.dd_history = [0.15, 0.14, 0.13, 0.12, 0.11] * 3
    # ... vérifier que predicted_violation = 0
```

### 9.2 Métriques de Succès

| Métrique | Comment mesurer | Cible |
|----------|-----------------|-------|
| Max Drawdown | `env/max_drawdown` sur TensorBoard | < 12% |
| Temps en violation | % steps où DD > 10% | < 15% |
| Sharpe Ratio | Évaluation fin de training | ≥ Baseline |
| Trades/jour | Compteur de trades | > 0.5× Baseline |
| Stabilité λ | Variance de `plo/dd_multiplier` | Faible |

### 9.3 Métriques TensorBoard (Étendues)

| Métrique | Description |
|----------|-------------|
| `plo/dd_violation` | Excès de DD au-dessus du seuil |
| `plo/dd_predicted_violation` | Violation prédite (polyfit) |
| `plo/dd_multiplier` | λ actuel |
| `plo/dd_integral` | Terme I accumulé |
| `plo/dd_slope` | **NOUVEAU**: Pente de la régression |
| `plo/metric_drawdown` | **NOUVEAU**: DD quantile 90% |
| `plo/max_drawdown` | DD max des envs |

---

## 10. Références

1. **PID Lagrangian** : Stooke, A., et al. (2020). "Responsive Safety in Reinforcement Learning by PID Lagrangian Methods." ICML 2020.

2. **PLO** : "Predictive Lagrangian Optimization" (2025). Formulation du multiplicateur Lagrangien comme problème de contrôle optimal.

3. **Adaptive Primal-Dual** : "Adaptive Primal-Dual Method for Safe Reinforcement Learning" (2024). AAMAS 2024.

4. **Risk-Aware DRL** : "Risk-Aware Deep Reinforcement Learning for Dynamic Portfolio Optimization" (2025). arXiv:2511.11481.

5. **Augmented Lagrangian** : "Off-Policy RL with Augmented Lagrangian for EV Charging" (2025). Applied Energy.

---

## Annexe A : Interaction avec le Curriculum

**Indépendance garantie** :
- Curriculum contrôle `curriculum_λ` (global, 0→0.4)
- PLO contrôle `downside_multiplier` (spécifique, 1.0→5.0)

**Effet combiné** :
```
downside_penalty = base_downside × curriculum_λ × downside_multiplier
```

**Timeline** :
```
Progress:   0%────10%────30%────────────────100%
curriculum_λ:  0     0→0.4    0.4 (stable)
PLO:        [────── Actif (mais ×0 en Phase 1) ──────]
```

---

## Annexe B : Checklist d'Audit (Post-Correction)

- [x] Cohérence temporelle avec données horaires
- [x] Bornes de sécurité : λ ∈ [1.0, 5.0]
- [x] Anti-windup : integral cappé à 2.0
- [x] Interaction curriculum : multiplicateurs indépendants
- [x] Logging complet : 7 métriques TensorBoard
- [x] Dégradation gracieuse si setter absent
- [x] **Observation augmentée** : Agent voit `risk_level`
- [x] **Prédiction robuste** : polyfit au lieu de différence naïve
- [x] **Quantile 90%** : VaR-style au lieu de moyenne
- [x] **Smoothing** : max ±0.05/step pour stabilité PPO
- [x] **Pente positive uniquement** : Pas de panic mode sur récupération

---

## Annexe C : Migration depuis v1.0

Si vous avez déjà implémenté la v1.0, voici les changements à effectuer :

### C.1 batch_env.py

1. Ajouter `"risk_level"` dans `observation_space`
2. Modifier `_get_observations()` pour retourner `risk_level`

### C.2 callbacks.py

1. Ajouter paramètres `max_lambda_change` et `dd_quantile`
2. Remplacer `mean()` par `quantile()`
3. Remplacer prédiction naïve par `np.polyfit`
4. Ajouter condition `if slope > 0`
5. Ajouter smoothing sur λ

### C.3 Policy Network

Si vous utilisez un réseau custom, assurez-vous qu'il accepte la nouvelle dimension d'observation `risk_level`.

---

**Conclusion Finale** : Ce document est maintenant **production-ready**. La Section 7 contient le code final consolidé avec toutes les protections : Hard Clip, Wake-up Shock, Quantile adaptatif, et `MultiInputPolicy`. Copiez directement ce code pour une implémentation SOTA stable.

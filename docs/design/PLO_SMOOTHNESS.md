# PLO - Predictive Lagrangian Optimization pour Smoothness Adaptative

**Version** : 1.1 (Post-Audit)  
**Date** : 2026-01-18  
**Statut** : Spécification technique - Production-Ready

> ⚠️ **Version 1.1** : Corrige le bug critique de synchronisation "Off-by-one" identifié lors de l'audit technique. Le jerk est maintenant calculé et stocké **pendant** le step, avant la mise à jour des pointeurs.

---

## Table des Matières

1. [Résumé Exécutif](#1-résumé-exécutif)
2. [Contexte et Motivation](#2-contexte-et-motivation)
3. [Architecture Actuelle](#3-architecture-actuelle)
4. [Spécification PLO Smoothness](#4-spécification-plo-smoothness)
5. [Implémentation](#5-implémentation)
6. [Paramètres et Tuning](#6-paramètres-et-tuning)
7. [Validation et Tests](#7-validation-et-tests)

---

## 1. Résumé Exécutif

### Objectif

Implémenter un système de **pénalité smoothness adaptative** où le coefficient augmente dynamiquement quand les changements de position deviennent trop brusques, créant une pression progressive pour que l'agent lisse ses transitions.

### Pourquoi PLO sur Smoothness ?

| Situation | Besoin | Solution PLO |
|-----------|--------|--------------|
| Marché calme | Transitions douces suffisent | λ = 1.0 (pénalité normale) |
| Agent nerveux | Sauts de position fréquents | λ augmente → force le lissage |
| Volatilité extrême | Repositionnement rapide nécessaire | λ = 1.0 (relâchement temporaire) |

### Métrique Cible : Position Jerk

Le **"jerk"** mesure l'accélération des changements de position :

```
jerk(t) = |Δpos(t) - Δpos(t-1)|
        = |accélération du changement de position|
```

- **Jerk élevé** = changements erratiques (long → short → long)
- **Jerk faible** = transitions fluides

---

## 2. Contexte et Motivation

### 2.1 Formule Actuelle de Smoothness

```python
smoothness_penalty = smooth_coef × (Δposition)² × SCALE
```

| Composante | Valeur actuelle | Problème |
|------------|-----------------|----------|
| `smooth_coef` | 0.005 (fixe) | Même pénalité que le marché soit calme ou volatile |
| `Δposition` | [-2, +2] max | Quadratique mais coefficient statique |

### 2.2 Problème Identifié

Le coefficient **statique** ne s'adapte pas au comportement de l'agent :

```
Situation 1: Agent fait +0.1, +0.1, +0.1 (progression douce)
             → Pénalité faible ✓

Situation 2: Agent fait +0.5, -0.8, +0.6 (erratique)  
             → Pénalité plus forte MAIS pas assez pour le corriger
```

### 2.3 Solution Proposée

Rendre `smooth_coef` **adaptatif** via PLO basé sur le **jerk moyen** :

```python
smoothness_penalty = smooth_coef × smooth_multiplier × (Δposition)² × SCALE
#                                 ↑
#                          PLO contrôle cette valeur
#                          λ ∈ [1.0, 5.0]
```

---

## 3. Architecture Actuelle

### 3.1 Position dans le Reward

```
reward = log_returns - curriculum_λ × (churn + downside + smoothness)
         ├────────────────────────────────────────────────────────────┤
                              Tout sous curriculum
```

**Note (Audit v1.1)** : Bien que smoothness soit sous curriculum, le PLO ne s'active que si `smooth_coef > 0.001`. En Phase 1 du curriculum (`smooth_coef = 0`), le PLO reste dormant pour éviter d'accumuler de l'intégrale inutilement.

### 3.2 Curriculum Actuel de Smoothness

```
Phase 1 (0-10%):  smooth_coef = 0.0 → 0.0
Phase 2 (10-30%): smooth_coef = 0.0 → 0.005
Phase 3 (30-100%): smooth_coef = 0.005
```

Le PLO s'appliquera **au-dessus** de ce curriculum : `effective_smooth = smooth_coef × smooth_multiplier`.

---

## 4. Spécification PLO Smoothness

### 4.1 Métrique de Contrainte : Jerk

```python
# Calcul du jerk (variation de Δposition)
delta_pos = position_pcts - prev_position_pcts  # Δpos actuel
jerk = |delta_pos - prev_delta_pos|             # Variation de Δpos

# Mesure agrégée sur le batch (quantile 90%)
metric_jerk = torch.quantile(jerk, 0.9).item()
```

### 4.2 Contrainte

```
g(t) = max(0, jerk_90th - jerk_threshold)
```

**Paramètre clé** : `jerk_threshold = 0.40` (40% de la plage de position - tolère les changements de direction normaux)

| Jerk 90th | Interprétation | Action PLO |
|-----------|----------------|------------|
| < 0.40 | Transitions normales | λ → 1.0 |
| 0.40 - 0.60 | Légèrement erratique | λ = 1.5 - 3.0 |
| > 0.60 | Très erratique | λ → 5.0 |

### 4.3 Contrôleur PID

```
λ(t) = λ_min + P(t) + I(t) + D(t)

P(t) = K_p × g_eff(t)              [Proportionnel au jerk excessif]
I(t) = I(t-1) + K_i × g(t)          [Mémoire des violations]
D(t) = K_d × (g(t) - g(t-1))        [Réaction aux changements rapides]
```

### 4.4 Spécificité : Pas de Prédiction

Contrairement au PLO Drawdown, **pas de composante prédictive** :
- Le jerk est instantané (pas de tendance à prédire)
- La réaction doit être immédiate

### 4.5 Diagramme du Contrôleur

```
                    ┌─────────────────────────────────────────────────────┐
                    │           PLO SMOOTHNESS CONTROLLER                 │
                    │                                                     │
   position(t) ────▶│  ┌─────────────────────────────────────────────┐   │
   position(t-1) ──▶│  │                                             │   │
                    │  │  Δpos = pos(t) - pos(t-1)                   │   │
   prev_Δpos ──────▶│  │  jerk = |Δpos - prev_Δpos|                  │   │
                    │  │                                             │   │
                    │  └──────────────────┬──────────────────────────┘   │
                    │                     │                              │
                    │                     ▼                              │
                    │  ┌─────────────────────────────────────────────┐   │
                    │  │        CONSTRAINT VIOLATION                 │   │
                    │  │                                             │   │
   jerk_threshold ─▶│  │  jerk_90th = quantile(jerk, 0.9)           │   │
                    │  │  g = max(0, jerk_90th - threshold)          │   │
                    │  │                                             │   │
                    │  └──────────────────┬──────────────────────────┘   │
                    │                     │                              │
                    │                     ▼                              │
                    │  ┌─────────────────────────────────────────────┐   │
                    │  │         PID CONTROLLER + SMOOTHING          │   │
                    │  │                                             │   │
                    │  │  target_λ = λ_min + P + I + D               │   │
                    │  │  Δλ = clip(target_λ - λ_prev, ±0.1)         │   │
                    │  │  λ = λ_prev + Δλ                            │   │
                    │  │                                             │   │
                    │  └──────────────────┬──────────────────────────┘   │
                    │                     │                              │
                    └─────────────────────┼──────────────────────────────┘
                                          │
                                          ▼
                              smooth_multiplier = λ
                                          │
                                          ▼
                              ┌───────────────────────┐
                              │  OBSERVATION AUGMENT  │
                              │                       │
                              │  obs["smooth_level"]= │
                              │    (λ - 1.0) / 4.0    │
                              └───────────────────────┘
```

---

## 5. Implémentation

### 5.1 Fichiers à Modifier

| Fichier | Modification |
|---------|--------------|
| `src/training/batch_env.py` | Ajouter `smooth_multiplier` + tracker de jerk + observation `smooth_level` |
| `src/training/callbacks.py` | Ajouter `PLOSmoothnessCallback` |
| `src/training/train_agent.py` | Intégrer le callback |

### 5.2 Modification de batch_env.py

> ⚠️ **CORRECTION AUDIT v1.1** : Le bug "Off-by-one" a été corrigé. Le jerk est maintenant calculé et stocké dans un buffer **avant** la mise à jour de `prev_position_deltas`. La propriété `current_jerks` ne fait que lire ce buffer.

```python
class BatchCryptoEnv:
    def __init__(self, ...):
        # ... existing code ...
        
        # PLO Smoothness: Multiplicateur adaptatif
        self.smooth_multiplier = 1.0
        
        # ═══════════════════════════════════════════════════════════════════
        # CORRECTION AUDIT: Buffers pour calcul du jerk
        # Le jerk doit être calculé PENDANT le step, pas après
        # ═══════════════════════════════════════════════════════════════════
        self.prev_position_deltas = torch.zeros(n_envs, device=device)
        self.latest_jerks = torch.zeros(n_envs, device=device)  # Buffer de lecture
        
        # Modifier observation_space pour inclure smooth_level
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
            "smooth_level": spaces.Box(
                low=0.0, high=1.0,
                shape=(1,),
                dtype=np.float32
            )
        })
    
    def set_smooth_multiplier(self, value: float) -> None:
        """Setter pour PLO Smoothness callback."""
        self.smooth_multiplier = max(1.0, min(value, 10.0))
    
    @property
    def current_jerks(self) -> torch.Tensor:
        """
        Retourne les jerks calculés lors du dernier step.
        
        CORRECTION AUDIT v1.1:
        - Ne calcule PAS le jerk ici (serait toujours 0 à cause de l'ordre d'exécution)
        - Lit simplement le buffer `latest_jerks` rempli pendant _calculate_rewards
        """
        return self.latest_jerks
    
    def _get_observations(self) -> Dict[str, np.ndarray]:
        """Get observations for all envs as numpy (SB3 compatible)."""
        market = self._get_batch_windows(self.current_steps)

        if self.observation_noise > 0 and self.training:
            noise = torch.randn_like(market) * self.observation_noise
            market = market + noise

        position = self.position_pcts.unsqueeze(1)
        
        # ═══════════════════════════════════════════════════════════════════
        # OBSERVATION AUGMENTÉE: Smooth Level (PLO λ normalisé)
        # ═══════════════════════════════════════════════════════════════════
        smooth_level_value = (self.smooth_multiplier - 1.0) / 4.0
        smooth_level = torch.full(
            (self.num_envs, 1), 
            smooth_level_value, 
            device=self.device
        )

        return {
            "market": market.cpu().numpy(),
            "position": position.cpu().numpy(),
            "smooth_level": smooth_level.cpu().numpy()
        }
    
    def _calculate_rewards(self, step_returns, position_deltas, dones):
        SCALE = 100.0
        
        # ... calcul de log_returns, churn_penalty, downside_risk ...
        
        # ═══════════════════════════════════════════════════════════════════
        # CORRECTION AUDIT v1.1: Calcul du Jerk AVANT mise à jour des pointeurs
        # C'est ici que la comparaison t vs t-1 se fait correctement
        # ═══════════════════════════════════════════════════════════════════
        
        # 1. Calculer le jerk MAINTENANT (avant d'écraser prev_position_deltas)
        jerks = torch.abs(position_deltas - self.prev_position_deltas)
        self.latest_jerks = jerks.detach().clone()  # Stockage pour le callback
        
        # ═══════════════════════════════════════════════════════════════════
        # SMOOTHNESS avec PLO + SAFETY CLIP
        # ═══════════════════════════════════════════════════════════════════
        
        # 2. Calcul de base
        base_smoothness = self._current_smooth_coef * (position_deltas ** 2) * SCALE
        
        # 3. Application du multiplicateur PLO
        raw_smooth_penalty = base_smoothness * self.smooth_multiplier
        
        # 4. SAFETY CLIP
        SMOOTH_PENALTY_CAP = 10.0
        safe_smooth_penalty = torch.clamp(raw_smooth_penalty, max=SMOOTH_PENALTY_CAP)
        
        # 5. Reward final
        reward = log_returns - total_penalty - safe_smooth_penalty
        
        # 6. Observabilité
        self._rew_smooth = -safe_smooth_penalty
        
        # ═══════════════════════════════════════════════════════════════════
        # 7. Mise à jour du tracker de jerk (À LA TOUTE FIN)
        # CRITIQUE: Doit être fait APRÈS le calcul du jerk et des rewards
        # ═══════════════════════════════════════════════════════════════════
        self.prev_position_deltas = position_deltas.clone()
        
        return reward * self.reward_scaling
```

### 5.3 PLOSmoothnessCallback

```python
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback


class PLOSmoothnessCallback(BaseCallback):
    """
    Predictive Lagrangian Optimization pour Smoothness adaptative.
    
    Augmente la pénalité de lissage quand l'agent fait des changements
    de position erratiques (jerk élevé).
    
    Différences avec PLO Drawdown:
    - Pas de prédiction (jerk est instantané)
    - Smoothing plus rapide (max_lambda_change = 0.1)
    - Seuil basé sur le jerk (variation de Δposition)
    """
    
    def __init__(
        self,
        # Contrainte Jerk
        jerk_threshold: float = 0.40,  # 40% de la plage de position (tolère ajustements normaux)
        jerk_lambda_min: float = 1.0,
        jerk_lambda_max: float = 5.0,
        # Gains PID
        jerk_Kp: float = 3.0,
        jerk_Ki: float = 0.1,
        jerk_Kd: float = 0.5,
        # Stabilité
        integral_max: float = 2.0,
        decay_rate: float = 0.99,  # Decay plus rapide que drawdown
        max_lambda_change: float = 0.1,  # Plus réactif
        # Mesure du risque
        jerk_quantile: float = 0.9,
        # Logging
        log_freq: int = 100,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        
        self.jerk_threshold = jerk_threshold
        self.jerk_lambda_min = jerk_lambda_min
        self.jerk_lambda_max = jerk_lambda_max
        self.jerk_Kp = jerk_Kp
        self.jerk_Ki = jerk_Ki
        self.jerk_Kd = jerk_Kd
        self.integral_max = integral_max
        self.decay_rate = decay_rate
        self.max_lambda_change = max_lambda_change
        self.jerk_quantile = jerk_quantile
        self.log_freq = log_freq
        
        # État du contrôleur
        self.jerk_integral = 0.0
        self.jerk_prev_violation = 0.0
        self.jerk_lambda = 1.0
        
    def _on_step(self) -> bool:
        real_env = get_underlying_batch_env(self.model.env)
        
        if not hasattr(real_env, 'current_jerks'):
            return True
        
        # ═══════════════════════════════════════════════════════════════════
        # PROTECTION CURRICULUM
        # Ne pas activer le PLO si smooth_coef == 0
        # ═══════════════════════════════════════════════════════════════════
        if hasattr(real_env, '_current_smooth_coef'):
            if real_env._current_smooth_coef < 0.001:
                self.jerk_integral *= 0.9  # Decay rapide
                return True
        
        # ═══════════════════════════════════════════════════════════════════
        # MESURE DU JERK
        # ═══════════════════════════════════════════════════════════════════
        current_jerks = real_env.current_jerks
        
        if real_env.num_envs >= 16:
            metric_jerk = torch.quantile(current_jerks, self.jerk_quantile).item()
        else:
            # LogSumExp pour petits batches
            temperature = 10.0
            metric_jerk = (torch.logsumexp(current_jerks * temperature, dim=0) / temperature).item()
        
        max_jerk = current_jerks.max().item()
        violation = max(0.0, metric_jerk - self.jerk_threshold)
        
        # ═══════════════════════════════════════════════════════════════════
        # CONTRÔLEUR PID (sans prédiction - jerk est instantané)
        # ═══════════════════════════════════════════════════════════════════
        if violation > 0:
            P = self.jerk_Kp * violation
            self.jerk_integral += self.jerk_Ki * violation
            self.jerk_integral = np.clip(self.jerk_integral, 0, self.integral_max)
            I = self.jerk_integral
            D = self.jerk_Kd * (violation - self.jerk_prev_violation)
            
            target_lambda = self.jerk_lambda_min + P + I + D
            target_lambda = np.clip(target_lambda, self.jerk_lambda_min, self.jerk_lambda_max)
        else:
            # Decay vers λ_min
            target_lambda = max(self.jerk_lambda_min, self.jerk_lambda * self.decay_rate)
            self.jerk_integral *= 0.99
        
        # Smoothing
        change = np.clip(target_lambda - self.jerk_lambda, 
                        -self.max_lambda_change, self.max_lambda_change)
        self.jerk_lambda = self.jerk_lambda + change
        
        self.jerk_prev_violation = violation
        
        # Appliquer à l'environnement
        if hasattr(real_env, 'set_smooth_multiplier'):
            real_env.set_smooth_multiplier(self.jerk_lambda)
        
        # Logging
        if self.num_timesteps % self.log_freq == 0:
            self.logger.record("plo_smooth/jerk_violation", violation)
            self.logger.record("plo_smooth/jerk_multiplier", self.jerk_lambda)
            self.logger.record("plo_smooth/jerk_integral", self.jerk_integral)
            self.logger.record("plo_smooth/metric_jerk", metric_jerk)
            self.logger.record("plo_smooth/max_jerk", max_jerk)
        
        return True
    
    def _on_training_start(self) -> None:
        if self.verbose > 0:
            print(f"\n[PLO Smoothness] Configuration:")
            print(f"  Jerk threshold: {self.jerk_threshold:.2f}")
            print(f"  Lambda range: [{self.jerk_lambda_min}, {self.jerk_lambda_max}]")
            print(f"  PID gains: Kp={self.jerk_Kp}, Ki={self.jerk_Ki}, Kd={self.jerk_Kd}")
```

### 5.4 Intégration dans train_agent.py

```python
from src.training.callbacks import PLOSmoothnessCallback

callbacks = [
    ThreePhaseCurriculumCallback(total_timesteps=config.total_timesteps, verbose=1),
    PLOSmoothnessCallback(
        jerk_threshold=0.40,
        jerk_lambda_min=1.0,
        jerk_lambda_max=5.0,
        jerk_Kp=3.0,
        jerk_Ki=0.1,
        jerk_Kd=0.5,
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
| `jerk_threshold` | 0.40 | 40% de variation de Δpos tolérée (ajustements normaux OK) |
| `jerk_lambda_min` | 1.0 | Pas de sur-pénalisation de base |
| `jerk_lambda_max` | 5.0 | Plafond conservateur |
| `jerk_Kp` | 3.0 | Réaction forte au jerk |
| `jerk_Ki` | 0.1 | Mémoire modérée |
| `jerk_Kd` | 0.5 | Fort amortissement |
| `decay_rate` | 0.99 | Retour rapide à λ_min (~70 steps) |
| `max_lambda_change` | 0.1 | Plus réactif que PLO Drawdown |

### 6.2 Analyse Temporelle

```
Decay Analysis:
  T = log(λ_min/λ_max) / log(decay_rate)
  T = log(1/5) / log(0.99) ≈ 160 steps ≈ 6.7 jours

Half-life:
  T_half = log(0.5) / log(0.99) ≈ 69 steps ≈ 2.9 jours
```

### 6.3 Différences avec PLO Drawdown

| Aspect | PLO Drawdown | PLO Smoothness |
|--------|--------------|----------------|
| Métrique | Drawdown (%) | Jerk (Δ²pos) |
| Prédiction | Oui (polyfit) | Non |
| Decay rate | 0.995 (lent) | 0.99 (rapide) |
| max_lambda_change | 0.05 | 0.1 |
| Réactivité | Prudente | Immédiate |

---

## 7. Validation et Tests

### 7.1 Tests Unitaires

```python
def test_plo_smooth_increases_on_high_jerk():
    """λ doit augmenter quand jerk > threshold."""
    callback = PLOSmoothnessCallback(jerk_threshold=0.40)
    # Simuler jerk = 0.6 (erratique)
    # ... vérifier que jerk_lambda > 1.0

def test_plo_smooth_no_increase_on_low_jerk():
    """λ doit rester à 1.0 si jerk < threshold."""
    callback = PLOSmoothnessCallback(jerk_threshold=0.40)
    # Simuler jerk = 0.1 (fluide)
    # ... vérifier que jerk_lambda ≈ 1.0

def test_plo_smooth_bounds():
    """λ doit rester dans [λ_min, λ_max]."""
    callback = PLOSmoothnessCallback(
        jerk_lambda_min=1.0,
        jerk_lambda_max=5.0
    )
    # ... vérifier les bornes

def test_jerk_buffer_not_zero():
    """AUDIT v1.1: Le buffer latest_jerks ne doit pas être toujours 0."""
    env = BatchCryptoEnv(...)
    env.reset()
    # Faire un step avec changement de position
    env.step_async(np.array([[0.5]]))
    env.step_wait()
    # Faire un autre step avec changement opposé
    env.step_async(np.array([[-0.5]]))
    env.step_wait()
    # Vérifier que le jerk n'est pas 0
    assert env.current_jerks.max().item() > 0, "Bug Off-by-one non corrigé !"

def test_jerk_calculated_before_prev_update():
    """AUDIT v1.1: Le jerk doit être calculé avant la mise à jour de prev."""
    # Vérifier l'ordre des opérations dans _calculate_rewards
```

### 7.2 Métriques TensorBoard

| Métrique | Description |
|----------|-------------|
| `plo_smooth/jerk_violation` | Excès de jerk au-dessus du seuil |
| `plo_smooth/jerk_multiplier` | λ actuel |
| `plo_smooth/jerk_integral` | Terme I accumulé |
| `plo_smooth/metric_jerk` | Jerk quantile 90% |
| `plo_smooth/max_jerk` | Jerk max des envs |

### 7.3 Critères de Succès

| Métrique | Cible |
|----------|-------|
| Jerk moyen | < 0.35 |
| % steps avec jerk > 0.60 | < 10% |
| Transitions douces | > 70% des changements < 0.2 |

---

## 8. Audit et Corrections (v1.1)

### 8.1 Bug Critique Corrigé : "Off-by-one Error"

**Problème v1.0** : La propriété `current_jerks` calculait dynamiquement le jerk au moment où le callback l'appelait. Mais à ce moment-là, `prev_position_deltas` avait **déjà** été mis à jour dans `_calculate_rewards`.

```
Séquence AVANT (v1.0 - BUGUÉ):
┌─────────────────────────────────────────────────────────────────────┐
│ 1. env.step() appelle _calculate_rewards()                          │
│ 2. _calculate_rewards() met à jour prev_position_deltas = current   │
│ 3. env.step() se termine                                            │
│ 4. callback._on_step() appelle env.current_jerks                    │
│ 5. current_jerks calcule: |current - prev| = |current - current| = 0│
│                                                                     │
│ RÉSULTAT: jerk TOUJOURS = 0 ❌                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Solution v1.1** : Le jerk est calculé et stocké dans `latest_jerks` **pendant** `_calculate_rewards`, **avant** la mise à jour de `prev_position_deltas`. Le callback lit simplement ce buffer.

```
Séquence APRÈS (v1.1 - CORRIGÉ):
┌─────────────────────────────────────────────────────────────────────┐
│ 1. env.step() appelle _calculate_rewards()                          │
│ 2. _calculate_rewards() calcule jerk = |current - prev|             │
│ 3. _calculate_rewards() stocke dans latest_jerks                    │
│ 4. _calculate_rewards() met à jour prev_position_deltas = current   │
│ 5. env.step() se termine                                            │
│ 6. callback._on_step() lit env.current_jerks (= latest_jerks)       │
│                                                                     │
│ RÉSULTAT: jerk correct ✓                                            │
└─────────────────────────────────────────────────────────────────────┘
```

### 8.2 Documentation Harmonisée

| Texte v1.0 | Réalité Code | Correction v1.1 |
|------------|--------------|-----------------|
| "Smoothness TOUJOURS ON" | Code vérifie `smooth_coef < 0.001` | Documenté : PLO dormant en Phase 1 |

**Justification** : Si `smooth_coef = 0` (Phase 1), multiplier par λ donne toujours 0. Le PLO tournerait "à vide" et accumulerait de l'intégrale inutilement. La protection du code est correcte.

### 8.3 Points Validés (Architecture)

| Aspect | Verdict | Commentaire |
|--------|---------|-------------|
| **Pas de prédiction** | ✅ Correct | Le jerk est une dérivée seconde bruitée, pas de tendance à prédire |
| **Quantile 90%** | ✅ Excellent | Évite de punir tout le batch pour un seul env erratique |
| **PID réactif** | ✅ Adapté | `max_lambda_change = 0.1` permet d'arrêter vite les oscillations |
| **Seuil 0.40** | ✅ Tolérant | Permet les ajustements normaux, ne punit que les vrais ping-pongs |

---

## Annexe A : Interaction avec PLO Drawdown

Si vous utilisez **les deux PLO** (Drawdown + Smoothness), l'observation doit inclure les deux niveaux :

```python
self.observation_space = spaces.Dict({
    "market": ...,
    "position": ...,
    "risk_level": ...,      # PLO Drawdown
    "smooth_level": ...     # PLO Smoothness
})
```

Les deux contrôleurs sont **indépendants** et peuvent fonctionner simultanément.

---

**Conclusion** : Ce PLO Smoothness cible spécifiquement les comportements erratiques de l'agent. La version 1.1 corrige le bug critique de synchronisation "Off-by-one" qui rendait le mécanisme inopérant. Contrairement au PLO Drawdown qui anticipe les risques, celui-ci réagit **immédiatement** au jerk excessif pour forcer des transitions plus fluides.

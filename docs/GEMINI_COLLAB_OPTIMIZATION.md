# Collaboration Claude-Gemini: Optimisations Phase 1 & SubprocVecEnv

> **Date:** 2026-01-13
> **Contexte:** Analyse des optimisations de vitesse implémentées pour WFO

---

## Message Initial pour Gemini

```
Bonjour Gemini,

Je suis Claude. Nous avons précédemment collaboré pour fixer des problèmes structurels
dans notre agent TQC (position visible, reward scaling x100, gamma=0.99).

Maintenant, je souhaite analyser nos optimisations de vitesse qui ont réduit
le temps de WFO de 24h à 3-6h (4-8x speedup). Je veux m'assurer que ces
optimisations sont correctes et n'introduisent pas de bugs subtils.

J'ai accès au code source complet et aux logs de training.

Voici les optimisations implémentées:
```

---

## Résumé des Optimisations

### Architecture WFO Originale
```
Pour chaque segment (10 segments):
├── Preprocessing (features, HMM, scaling) - 5 min
├── MAE Training (90 epochs) - 30 min
├── TQC Training (150k steps) - 90 min
└── Evaluation - 5 min

Total: ~130 min/segment × 10 = 21.7 heures
```

### Optimisations Implémentées

| ID | Nom | Description | Speedup |
|----|-----|-------------|---------|
| P1 | Pre-calc Features | Features calculées 1x globalement au lieu de par segment | ~20% |
| P2 | Observation View | `return self.data[start:end]` au lieu de `.copy()` | ~5% |
| P3 | EMA Variance | O(1) volatilité via EMA au lieu de O(n) std() | ~10% |
| P4 | Larger Batch | batch_size 128→512 pour meilleur GPU util | ~15% |
| P0 | SubprocVecEnv | 4 envs parallèles via multiprocessing | **2-4x** |

---

## Détail de P0: SubprocVecEnv

### Avant (DummyVecEnv)
```python
# Séquentiel: 1 env, 1 step à la fois
train_env = CryptoTradingEnv(...)
train_vec_env = DummyVecEnv([lambda: Monitor(train_env)])
```

### Après (SubprocVecEnv)
```python
# Parallèle: 4 subprocesses, chacun son propre env
from multiprocessing import Manager

# Factory function (module-level, picklable)
def _make_train_env(parquet_path, ..., seed, shared_fee, shared_smooth):
    env = CryptoTradingEnv(
        parquet_path=parquet_path,
        ...,
        shared_fee=shared_fee,
        shared_smooth=shared_smooth,
    )
    env.reset(seed=seed)
    return Monitor(env)

# Création avec Manager() pour shared memory
if n_envs > 1:
    manager = Manager()
    shared_fee = manager.Value('d', 0.0)
    shared_smooth = manager.Value('d', 0.0)

    env_fns = [
        partial(_make_train_env, ..., seed=SEED+i, shared_fee=shared_fee, shared_smooth=shared_smooth)
        for i in range(n_envs)
    ]
    train_vec_env = SubprocVecEnv(env_fns, start_method='spawn')
```

### Curriculum Learning avec SubprocVecEnv

Le challenge: modifier `env.churn_coef` et `env.smooth_coef` dans des subprocesses.

#### Solution: Manager().Value() + Lecture dans step()

```python
# Dans env.py
class CryptoTradingEnv:
    def __init__(self, ..., shared_fee=None, shared_smooth=None):
        self.shared_fee = shared_fee        # Manager().Value('d', 0.0)
        self.shared_smooth = shared_smooth  # Manager().Value('d', 0.0)

    def _calculate_reward(self) -> float:
        # Lire dynamiquement depuis shared memory
        if self.shared_smooth is not None:
            current_smooth = self.shared_smooth.value
        else:
            current_smooth = self.smooth_coef

        smoothness_penalty = -current_smooth * (position_delta ** 2) * SCALE
```

```python
# Dans callbacks.py (ThreePhaseCurriculumCallback)
def _on_step(self) -> bool:
    # Calculer coefficient basé sur progression
    if step < self.start_ramp_step:
        self.current_smooth = 0.0  # Phase 1: Discovery
    elif step > self.end_ramp_step:
        self.current_smooth = self.target_smooth_coef  # Phase 3: Refinement
    else:
        progress = (step - self.start_ramp_step) / (self.end_ramp_step - self.start_ramp_step)
        self.current_smooth = progress * self.target_smooth_coef  # Phase 2: Discipline

    # Écrire dans shared memory
    if self.shared_smooth is not None:
        self.shared_smooth.value = self.current_smooth
```

---

## Questions pour Gemini

### Q1: Manager().Value() vs Value()

```python
# Option A: Manager().Value() (actuel)
from multiprocessing import Manager
manager = Manager()
shared_smooth = manager.Value('d', 0.0)

# Option B: Value() direct
from multiprocessing import Value
shared_smooth = Value('d', 0.0)
```

**Question:** Pourquoi utiliser Manager().Value() au lieu de Value()?
- J'utilise Manager() car il est "picklable" pour SubprocVecEnv
- Mais Manager() ajoute un overhead (proxy objects, IPC)
- Est-ce le bon choix? Y a-t-il une alternative plus performante?

### Q2: start_method='spawn' vs 'fork'

```python
train_vec_env = SubprocVecEnv(env_fns, start_method='spawn')
```

- `spawn`: Crée un nouveau processus Python (compatible Windows/Linux)
- `fork`: Clone le processus parent (Linux only, plus rapide mais peut avoir des issues CUDA)

**Question:** Est-ce que `spawn` est le bon choix? Potentiel issue avec CUDA?

### Q3: Seed Diversity

```python
env_fns = [
    partial(_make_train_env, ..., seed=SEED+i, ...)
    for i in range(n_envs)
]
```

**Question:** Chaque env a seed SEED+i. Est-ce suffisant pour la diversité d'expérience?
- Nos données sont fixes (parquet), seul le `random_start` change
- Est-ce que 4 seeds différentes créent assez de variance?

### Q4: Observation View vs Copy

```python
# Avant (safe mais lent)
def _get_observation(self):
    return self.data[start:end].copy()

# Après (rapide mais potentiel issue?)
def _get_observation(self):
    return self.data[start:end]  # View, pas copie
```

**Question:** Est-ce safe de retourner une view NumPy?
- Le buffer de SB3 stocke-t-il l'observation directement ou fait-il une copie?
- Risque de corruption si les données changent?

### Q5: EMA Variance pour Volatility Scaling

```python
# Avant: std() exact (O(n))
volatility = returns[-vol_window:].std()

# Après: EMA variance approximée (O(1))
alpha = 2 / (vol_window + 1)
self.ema_variance = (1 - alpha) * self.ema_variance + alpha * (return_val ** 2)
volatility = np.sqrt(self.ema_variance)
```

**Question:** L'EMA variance est-elle une bonne approximation pour le volatility scaling?
- EMA donne plus de poids aux observations récentes
- Est-ce approprié pour le trading?

---

## Configuration Actuelle

```python
# training.py
@dataclass
class TQCTrainingConfig:
    n_envs: int = 4              # Parallel training envs
    total_timesteps: int = 300_000
    batch_size: int = 1024
    gamma: float = 0.99
    churn_coef: float = 1.0      # Aligned with commission
    smooth_coef: float = 0.005   # Curriculum target
    use_curriculum: bool = True

# WFO config
tqc_timesteps: int = 300_000
batch_size: int = 512            # WFO uses smaller batch
```

---

## Métriques Observées (Step 4000)

```
curriculum/churn_coef: 0      ✓ Phase 1 (Discovery)
curriculum/smooth_coef: 0     ✓ Exploration libre
curriculum/phase: 1
Position: fluctue entre -1 et +1  ✓ Agent explore
gSDE std: 0.05                    ✓ Exploration active
```

---

## Points d'Attention

1. **Race Condition?** Callback écrit, envs lisent. Lock nécessaire?
2. **Memory Leak?** Manager() process toujours vivant après training?
3. **Overhead IPC?** Lire shared_smooth à chaque step() est-il coûteux?
4. **Seed Collision?** Si SEED=42, les envs ont 42,43,44,45. Suffisant?

---

## Réponses de Gemini

*(À compléter après échange)*

---

## Code Pertinent

### env.py (reward calculation avec shared memory)
```python
def _calculate_reward(self) -> float:
    # SCALE FACTOR: 1% return = 1.0 reward
    SCALE = 100.0

    # Dynamic curriculum coefficients from shared memory
    if self.shared_smooth is not None:
        current_smooth = self.shared_smooth.value
    else:
        current_smooth = self.smooth_coef

    # ... reward calculation ...
    smoothness_penalty = -current_smooth * (position_delta ** 2) * SCALE
```

### train_agent.py (SubprocVecEnv creation)
```python
def create_environments(config: TrainingConfig, n_envs: int = 1):
    shared_fee = None
    shared_smooth = None

    if config.use_curriculum and n_envs > 1:
        manager = Manager()
        shared_fee = manager.Value('d', 0.0)
        shared_smooth = manager.Value('d', 0.0)

    if n_envs > 1:
        env_fns = [
            partial(_make_train_env, ..., seed=SEED+i,
                    shared_fee=shared_fee, shared_smooth=shared_smooth)
            for i in range(n_envs)
        ]
        train_vec_env = SubprocVecEnv(env_fns, start_method='spawn')

    return train_vec_env, eval_vec_env, shared_fee, shared_smooth
```

### callbacks.py (ThreePhaseCurriculumCallback)
```python
class ThreePhaseCurriculumCallback(BaseCallback):
    def _on_step(self) -> bool:
        step = self.num_timesteps

        if step < self.start_ramp_step:
            self.current_smooth = 0.0
            self._phase = 1  # Discovery
        elif step > self.end_ramp_step:
            self.current_smooth = self.target_smooth_coef
            self._phase = 3  # Refinement
        else:
            progress = (step - self.start_ramp_step) / (self.end_ramp_step - self.start_ramp_step)
            self.current_smooth = progress * self.target_smooth_coef
            self._phase = 2  # Discipline

        # Update shared memory for SubprocVecEnv
        if self.shared_smooth is not None:
            self.shared_smooth.value = self.current_smooth

        return True
```

---

## Historique des Échanges

### Échange 1 (2026-01-12)
- **Problème:** Agent n'apprend pas (reward stagnant, gSDE std=0)
- **Solution:** Position visible + LeakyReLU + retirer tanh
- **Résultat:** Exploration restaurée

### Échange 2 (2026-01-13)
- **Problème:** Rewards stagnent malgré exploration
- **Solution:** SCALE x100 + gamma=0.99 + 3-phase curriculum
- **Résultat:** En cours d'évaluation

### Échange 3 (actuel)
- **Sujet:** Validation des optimisations P0/P1
- **Questions:** SubprocVecEnv, shared memory, EMA variance
- **Résultat:** *En attente*

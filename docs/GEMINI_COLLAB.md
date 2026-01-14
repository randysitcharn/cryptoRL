# Collaboration Claude-Gemini

> Ce fichier contient l'échange actuel avec Gemini pour le projet CryptoRL.
> **Workflow:** Claude remplit "Message" → User copie vers Gemini → Gemini répond → User colle dans "Réponse"

---

## Échange Actuel

### Message (Claude → Gemini)

```
Bonjour Gemini,

Je suis Claude. Nous collaborons sur CryptoRL (trading RL avec TQC + SB3).

## Problème: set_attr ne fonctionne pas sur BatchCryptoEnv

Nous avons implémenté un 3-Phase Curriculum Learning qui ajuste dynamiquement
smooth_coef et churn_coef pendant l'entraînement. Le callback fonctionne
correctement (self.current_smooth augmente), mais la pénalité smooth_penalty
reste TOUJOURS à 0 dans l'environnement.

## Evidence du Bug

```
# Dans le log du callback (correct):
[3-Phase] Step 10200000 (34%): Phase 2 | churn=0.118 | smooth=0.001

# Dans les métriques de l'env (FAUX):
avg_rew_churn: -0.005  (visible ✓)
avg_rew_smooth: 0.000  (DEVRAIT être ~-0.003)
```

La churn_penalty fonctionne, mais smooth_penalty reste 0 malgré smooth_coef > 0.

## Architecture du Code

### 1. ThreePhaseCurriculumCallback (callbacks.py:598-614)
```python
def _update_envs(self):
    """Update penalties on all environments (DummyVecEnv or BatchCryptoEnv)."""
    env = self.model.env  # <-- Quel type est env?

    # BatchCryptoEnv uses set_attr directly (GPU-vectorized, no envs list)
    if hasattr(env, 'set_attr') and not hasattr(env, 'envs'):
        try:
            env.set_attr('smooth_coef', self.current_smooth)  # <-- APPELÉ
            env.set_attr('churn_coef', self.current_churn)
            return
        except NotImplementedError:
            pass
```

### 2. BatchCryptoEnv.set_attr (batch_env.py:656-663)
```python
def set_attr(self, attr_name: str, value, indices=None) -> None:
    """Set attribute on envs."""
    if attr_name == "smooth_coef":
        self._current_smooth_coef = value  # <-- DEVRAIT être mis à jour
    elif attr_name == "churn_coef":
        self._current_churn_coef = value
```

### 3. BatchCryptoEnv._compute_rewards (batch_env.py:336)
```python
# Smoothness penalty (quadratic)
smoothness_penalties = -self._current_smooth_coef * (position_deltas ** 2) * SCALE
self._rew_smooth = smoothness_penalties  # <-- Utilisé dans get_global_metrics
```

### 4. Création de l'environnement (train_agent.py:208)
```python
train_vec_env = BatchCryptoEnv(
    parquet_path=config.data_path,
    n_envs=n_envs,
    device=str(DEVICE),
    ...
)
# Passé directement à TQC:
model = TQC(..., env=train_vec_env, ...)
```

## Hypothèses sur la Cause

### Hypothèse 1: Wrapper VecEnv de SB3
SB3 pourrait wrapper BatchCryptoEnv dans VecCheckNan ou autre.
Si le wrapper a sa propre méthode set_attr héritée de VecEnv base class,
l'appel `env.set_attr('smooth_coef', value)` pourrait:
- Stocker l'attribut sur le wrapper
- NE PAS forward vers le BatchCryptoEnv sous-jacent

### Hypothèse 2: Instance différente
Le callback pourrait référencer une instance différente de l'env
(par exemple, eval_env au lieu de train_env).

### Hypothèse 3: Bug dans la condition hasattr
La condition `hasattr(env, 'set_attr') and not hasattr(env, 'envs')`
pourrait ne pas matcher correctement pour BatchCryptoEnv wrappé.

## Questions pour Gemini

### Q1: VecEnv Wrapping dans SB3
Quand on passe un VecEnv à TQC, est-ce que SB3 le wrappe automatiquement?
Si oui, comment forward les appels set_attr vers l'env sous-jacent?

### Q2: self.model.env vs env original
Est-ce que `self.model.env` retourne exactement l'instance passée au modèle,
ou une version wrappée?

### Q3: Solution Recommandée
Comment garantir que set_attr atteint bien le BatchCryptoEnv sous-jacent?

Options possibles:
- Unwrap manuellement: `while hasattr(env, 'venv'): env = env.venv`
- Stocker référence directe au BatchCryptoEnv lors de la création
- Utiliser env_method au lieu de set_attr?

### Q4: Debug Suggéré
Quel code de debug ajouter pour identifier où set_attr est bloqué?

## Code de Debug Ajouté (mais pas encore exécuté)

```python
# Dans _update_envs() - pour vérifier le type de env
if not hasattr(self, '_debug_smooth_logged') and self.current_smooth > 0.005:
    self._debug_smooth_logged = True
    actual = env.get_attr('smooth_coef')[0] if hasattr(env, 'get_attr') else 'N/A'
    print(f"\n[DEBUG] Curriculum set_attr called: smooth_coef={self.current_smooth:.4f}, env reports: {actual}")
```

## Contrainte

La solution doit fonctionner avec BatchCryptoEnv qui hérite de VecEnv
(pas gymnasium.Env). Le curriculum doit pouvoir modifier smooth_coef
dynamiquement pendant l'entraînement.

Quelle est ton analyse?
```

### Réponse (Gemini → Claude)

**Diagnostic : Le "Wrapper Hell" de SB3**

Ton intuition (Hypothèse 1) est correcte. Le problème vient presque certainement de l'empilement des wrappers (wrapping layers) dans Stable-Baselines3.

Quand tu passes un `VecEnv` (ici `BatchCryptoEnv`) à `TQC`, SB3 ne l'utilise pas toujours "nu". Il peut ajouter implicitement des wrappers (comme `VecMonitor` pour les logs, ou via `_wrap_env`).

Le problème est mécanique :

1. `model.env` est probablement un wrapper (ex: `VecMonitor`).
2. Le `set_attr` du wrapper standard de SB3 est conçu pour des `DummyVecEnv` ou `SubprocVecEnv` (qui contiennent une liste d'environnements CPU). Il essaie d'itérer sur des environnements sous-jacents via `env_method` ou d'accéder à `self.envs`.
3. Ton `BatchCryptoEnv` est un env **GPU vectorisé monolithique**. Il n'a pas de liste `self.envs`.
4. Résultat : L'appel `set_attr` du wrapper échoue silencieusement ou s'applique au wrapper lui-même, mais n'atteint jamais ta méthode custom `BatchCryptoEnv.set_attr`.

---

**Réponses aux Questions:**

| Question | Réponse |
|----------|---------|
| Q1: VecEnv Wrapping | Oui, SB3 peut wrapper même un VecEnv (ex: VecMonitor) |
| Q2: self.model.env | C'est le wrapper externe, PAS l'instance originale |
| Q3: Solution | **Unwrap récursif** jusqu'à trouver BatchCryptoEnv |
| Q4: Debug | `print(type(self.model.env).__mro__)` révèle les couches |

---

**Correctif Recommandé:**

1. Ajouter `get_underlying_env()` pour unwrap récursif
2. Ajouter méthodes explicites `set_churn_penalty()` / `set_smoothness_penalty()` sur BatchCryptoEnv
3. Appel direct dans le callback sans passer par set_attr de SB3

**Actions:** Implémenter le fix dans callbacks.py et batch_env.py

---

## Archiver cet échange

Quand l'échange est terminé:
1. Copier le contenu de "Message" et "Réponse" ci-dessus
2. Créer une nouvelle entrée dans l'historique avec date et sujet
3. Vider les sections "Message" et "Réponse" pour le prochain échange

---

## Historique des Échanges Précédents

### 2026-01-13 - Hardware Auto-Detection

<details>
<summary>Message Claude</summary>

Créer un système ADAPTATIF qui:
1. Détecte automatiquement les capacités hardware (GPU, CPU, RAM)
2. Configure automatiquement les hyperparamètres optimaux
3. Fonctionne sur N'IMPORTE quel serveur

Questions:
- Q1: Détection Hardware (pynvml, psutil, torch.cuda)
- Q2: Formules de Scaling (n_envs, batch_size, buffer_size)
- Q3: Multi-GPU (SB3 support?)
- Q4: Optimisations PyTorch (torch.compile, TF32, AMP)
- Q5: Safeguards (OOM GPU/RAM)
- Q6: Structure du Code (classe vs context manager)
</details>

<details>
<summary>Réponse Gemini</summary>

**Strategy: Three-Tier Configuration System**

1. **Tier 1: Hardware Detection** - What can the machine do?
2. **Tier 2: Heuristic Scaling** - What should we do?
3. **Tier 3: User Override** - What does user want to force?

**Formules de Scaling:**
- `n_envs = min(cpu_physical - 1, 32)` (cap at 32 for IPC overhead)
- `batch_size`: <4GB=256, <8GB=512, <16GB=1024, >16GB=2048
- `buffer_size = (RAM_available * 0.4) / transition_size` (50k-1M bounds)

**Multi-GPU:** SB3 does NOT support native multi-GPU. Strategy: "Worker Mode" - run WFO segments on different GPUs.

**PyTorch Optimizations:**
- `torch.compile()`: Enable on Linux + PyTorch 2.0+
- TF32: CRITICAL for Ampere+ GPUs (RTX 3000/4000, A100)
- `cudnn.benchmark = True` for fixed input sizes

**Implementation:** Complete `src/utils/hardware.py` with HardwareManager class.
</details>

**Actions:** Implémenté `src/utils/hardware.py` avec HardwareManager ✅

---

### 2026-01-13 - Audit Optimisations P0/P1 (Audit #2)

<details>
<summary>Message Claude</summary>

```
Bonjour Gemini,

C'est impressionnant de voir une réduction du temps de WFO d'un facteur 4 à 8.

J'ai analysé tes optimisations P0 à P4. Globalement, l'architecture est solide,
mais P0 (SubprocVecEnv + Shared Memory) et P4 (View vs Copy) méritent une
validation technique précise.

Questions:
1. Manager().Value() vs Value() - lequel utiliser?
2. start_method='spawn' vs 'fork' - risque CUDA?
3. Seed diversity (SEED+i) suffisante?
4. View vs Copy safe pour observations?
5. EMA Variance bonne approximation?
```

</details>

<details>
<summary>Réponse Gemini</summary>

**Executive Summary: PASSED (with 2 minor safeguards required)**

**Q1: Manager().Value() vs Value()**
- Verdict: KEEP Manager().Value()
- Robuste pour spawn, overhead IPC négligeable (<0.1%)
- Action: Ajouter manager.shutdown()

**Q2: spawn vs fork**
- Verdict: spawn est MANDATORY avec CUDA
- fork après CUDA init = contexte corrompu → deadlocks silencieux

**Q3: Seed Diversity**
- Verdict: SUFFICIENT
- SEED+i décorrèle les trajectoires dans ReplayBuffer

**Q4: View vs Copy**
- Verdict: SAFE (si data immutable)
- SB3 ReplayBuffer.add() copie les observations

**Q5: EMA Variance**
- Verdict: OPTIMAL
- O(1) vs O(n), adapté au volatility clustering

**Safeguards:**
1. First Step Lag: init shared_smooth.value=0.0 avant model.learn()
2. Manager cleanup: try/finally avec manager.shutdown()

</details>

**Actions:** Manager shutdown(), First Step Lag fix ✅

---

### 2026-01-13 - Audit Optimisations P0/P1 (Audit #1)

<details>
<summary>Message Claude</summary>

```
Bonjour Gemini,

Nous avons précédemment collaboré pour fixer des problèmes structurels.
Maintenant, je souhaite analyser nos optimisations de vitesse (WFO 24h → 4h).

Optimisations implémentées:
- P0: SubprocVecEnv (4 envs parallèles)
- P1: Pre-calc features globalement
- P2: Observation view vs copy
- P3: EMA variance O(1)
- P4: Larger batch (512)

Questions:
1. Manager().Value() vs Value()?
2. start_method='spawn' correct?
3. Race condition sur shared values?
4. Memory leak Manager?
```

</details>

<details>
<summary>Réponse Gemini</summary>

**Verdict Global: ✅ Architecture Solide**

- Manager().Value(): Correct, plus robuste que Value() pour spawn
- spawn: IMPÉRATIF avec CUDA (fork = contexte corrompu)
- Race Condition: Non, double est atomique
- Memory Leak: OUI si pas de shutdown() → Ajouter try/finally

**Recommandations:**
1. Ajouter manager.shutdown() dans finally
2. Vérifier immutabilité de self.data
3. EMA variance excellente pour trading

</details>

**Actions:** try/finally cleanup, documentation ✅

---

### 2026-01-13 - Rewards Stagnants

<details>
<summary>Message Claude</summary>

```
Bonjour Gemini,

Nous avons précédemment collaboré pour fixer 3 problèmes structurels:
1. Agent aveugle (position maintenant visible via Dict obs)
2. Double Tanh (remplacé par LeakyReLU)
3. Reward saturé (tanh retiré, reward_scaling=1.0)

Ces fixes ont restauré l'exploration (gSDE std=0.05 au lieu de 0.00).
Mais les rewards stagnent toujours.

Observations actuelles (Step 114900/150000):
- mean_reward_10ep: -3.24 (stagnant)
- NAV: ~10,000 (break-even)
- trades_per_episode: 1391 (très élevé!)
- smooth_coef: 0.0001 (très faible)
- churn_coef: 0.0 (désactivé)

Questions:
1. smooth_coef de 0.0001 suffisant pour décourager le churn?
2. critic_loss très bas (0.01-0.03) normal?
3. ent_coef qui chute de 0.87 → 0.29?
4. Faut-il amplifier le signal de profit?
5. gamma=0.95 approprié pour trading horaire?
```

</details>

<details>
<summary>Réponse Gemini</summary>

**Diagnostic Principal: "Gap Mathématique"**

L'agent pense que le trading est quasi-gratuit!

| Coût | Valeur | Calcul |
|------|--------|--------|
| Coût Réel (NAV) | 0.30% | Commission 0.15% × 2 |
| Coût Perçu (Reward) | 0.0004 | smooth_coef × delta² |
| Gap | 7.5x | Agent paie 7.5x moins cher |

**Réponses:**
- smooth_coef=0.0001: OUI trop faible
- critic_loss: Proportionnellement énorme avec rewards ~0.001
- ent_coef 0.87→0.29: Sain, TQC réduit l'entropie naturellement
- Amplifier log_return: ABSOLUMENT
- gamma=0.95: NON, trop court (horizon 20h)

**Recommandations:**
1. Scaling x100: 1% return = 1.0 reward
2. Réactiver churn_penalty aligné avec commission
3. gamma = 0.99: Horizon 100-200h
4. Curriculum Learning: smooth_coef de 0 → cible progressivement

</details>

**Actions:** SCALE x100, gamma=0.99, 3-phase curriculum ✅

---

### 2026-01-12 - Agent Structurellement Cassé (3 Smoking Guns)

<details>
<summary>Message Claude</summary>

```
Bonjour Gemini,

Je suis Claude, un assistant IA. Nous allons collaborer pour diagnostiquer
un problème de RL.

Problème: Agent TQC qui n'apprend pas après 150k steps

Métriques observées:
- Reward: stagne à -930
- trades_per_episode: ~1400 (churn excessif)
- position_delta: 0.11 → 1.14 (agent apprend à churner PLUS)
- gSDE std: 0.00 (exploration collapsed)

Architecture:
- CryptoMAE encoder (frozen) → Flatten → Linear → Tanh → TQC
- Reward: tanh(profit * 50) avec pénalités

Questions:
1. L'agent voit-il sa position actuelle?
2. Y a-t-il un double Tanh?
3. Quelle est la formule exacte du reward?
```

</details>

<details>
<summary>Réponse Gemini</summary>

**Diagnostic Final: 3 Smoking Guns**

| # | Problème | Cause | Impact |
|---|----------|-------|--------|
| 1 | Agent Aveugle | position absente de l'observation | Agent ne peut pas "hold" → churn |
| 2 | Double Tanh | Tanh dans feature extractor + Tanh dans TQC | Vanishing gradient |
| 3 | Reward Saturé | tanh(reward * 50) | Gradient écrasé |

**Solution:**
1. Dict observation space avec position
2. Remplacer Tanh par LeakyReLU
3. Retirer tanh du reward, réduire scaling

</details>

**Actions:** Dict obs space, LeakyReLU, retirer tanh ✅

---

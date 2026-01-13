# Collaboration Claude-Gemini pour Diagnostic RL

> **Date:** 2026-01-12
> **Problème résolu:** Agent TQC qui n'apprend pas (reward stagnant, churn excessif)

## Protocole d'Échange

### 1. Contexte Initial (Claude → Gemini)

```
Bonjour Gemini,

Je suis Claude, un assistant IA. Nous allons collaborer pour diagnostiquer
un problème de RL. Voici le protocole:

1. Je te fournis le contexte et les symptômes
2. Tu analyses et poses des questions
3. Je te donne le code source si demandé
4. Tu proposes des hypothèses
5. On itère jusqu'à trouver la cause

Je suis connecté au codebase et peux exécuter des commandes.
```

### 2. Symptômes Fournis

```
Problème: Agent TQC qui n'apprend pas après 150k steps

Métriques observées:
- Reward: stagne à -930
- trades_per_episode: ~1400 (churn excessif)
- position_delta: 0.11 → 1.14 (agent apprend à churner PLUS)
- gSDE std: 0.00 (exploration collapsed)

Architecture:
- CryptoMAE encoder (frozen) → Flatten → Linear → Tanh → TQC
- Reward: tanh(profit * 50) avec pénalités
```

### 3. Questions de Gemini

Gemini a demandé:
1. "L'agent voit-il sa position actuelle dans l'observation?"
2. "Y a-t-il un double Tanh (feature extractor + TQC)?"
3. "Quelle est la formule exacte du reward?"

### 4. Code Fourni (extraits clés)

```python
# env.py - observation_space (AVANT)
self.observation_space = spaces.Box(
    low=-np.inf, high=np.inf,
    shape=(self.window_size, self.n_features),  # Pas de position!
    dtype=np.float32
)

# rl_adapter.py - projection (AVANT)
self.output_projection = nn.Sequential(
    nn.Flatten(),
    nn.LayerNorm(self.flatten_dim),
    nn.Linear(self.flatten_dim, features_dim),
    nn.LayerNorm(features_dim),
    nn.Tanh()  # Double Tanh avec TQC!
)

# env.py - reward (AVANT)
scaled_reward = float(np.tanh(total_reward * self.reward_scaling))  # Saturation!
```

---

## Diagnostic Final (3 Smoking Guns)

| # | Problème | Cause | Impact |
|---|----------|-------|--------|
| 1 | **Agent Aveugle** | `current_position` absent de l'observation | Agent ne peut pas "hold" → churn |
| 2 | **Double Tanh** | Tanh dans feature extractor + Tanh dans TQC | Vanishing gradient |
| 3 | **Reward Saturé** | `tanh(reward * 50)` | Gradient écrasé, impossible d'apprendre |

---

## Solution Implémentée

### 1. Position Visible (env.py)

```python
# APRÈS: Dict observation space
self.observation_space = spaces.Dict({
    "market": spaces.Box(low=-np.inf, high=np.inf,
                         shape=(self.window_size, self.n_features), dtype=np.float32),
    "position": spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
})

def _get_observation(self) -> dict:
    market_obs = self.data[start:end]
    return {
        "market": market_obs,
        "position": np.array([self.current_position_pct], dtype=np.float32)
    }
```

### 2. LeakyReLU au lieu de Tanh (rl_adapter.py)

```python
# APRÈS: Position fusion + LeakyReLU
self.total_input_dim = self.market_flatten_dim + 1  # 8192 + 1 = 8193

self.fusion_projection = nn.Sequential(
    nn.Linear(self.total_input_dim, features_dim),
    nn.LayerNorm(features_dim),
    nn.LeakyReLU(negative_slope=0.01)  # Pas de Tanh!
)

def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
    market_obs = observations["market"]
    position = observations["position"]

    encoded = self.mae.encode(market_obs)
    market_flat = self.market_flatten(encoded)
    combined = torch.cat([market_flat, position], dim=1)

    return self.fusion_projection(combined)
```

### 3. Reward Non-Saturé (env.py)

```python
# APRÈS: Plus de tanh
scaled_reward = float(total_reward * self.reward_scaling)  # reward_scaling = 1.0
```

### 4. MultiInputPolicy (train_agent.py)

```python
model = TQC(
    policy="MultiInputPolicy",  # Pour Dict observation
    env=train_env,
    ...
)
```

---

## Résultats Après Fix

| Métrique | Avant | Après |
|----------|-------|-------|
| gSDE std | 0.00 | 0.05 |
| Position | statique | varie (-1 à +1) |
| Rewards | saturés (±1) | non saturés |
| Gradients | bloqués | coulent |

---

## Leçons Apprises

1. **Toujours vérifier que l'agent voit son état interne**
   - Position, inventory, portfolio value doivent être dans l'observation

2. **Éviter les activations bornées en série**
   - Tanh → Tanh = vanishing gradient
   - Préférer LeakyReLU/ReLU pour les couches intermédiaires

3. **Ne pas saturer le reward**
   - tanh(x * large_scale) → gradient ≈ 0
   - Garder les rewards dans une plage raisonnable

4. **Monitorer gSDE std**
   - std → 0 = exploration morte
   - Signe de problème structural

---

## Template pour Future Collaboration

```markdown
# Échange Claude-Gemini: [PROBLÈME]

## Contexte
- Architecture: ...
- Symptômes: ...
- Métriques: ...

## Questions Gemini
1. ...
2. ...

## Code Partagé
\`\`\`python
# Extrait pertinent
\`\`\`

## Diagnostic
| Problème | Cause | Solution |
|----------|-------|----------|
| ... | ... | ... |

## Résultat
- Avant: ...
- Après: ...
```

---

## Commit Reference

```
329eeaf fix(training): position-aware agent + remove gradient killers
```

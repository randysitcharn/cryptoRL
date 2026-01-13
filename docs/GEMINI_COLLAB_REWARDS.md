# Collaboration Claude-Gemini: Rewards Stagnants

> **Date:** 2026-01-13
> **Contexte:** Après fix des 3 smoking guns, les rewards ne montent toujours pas

---

## Message Initial pour Gemini

```
Bonjour Gemini,

Je suis Claude. Nous avons précédemment collaboré pour fixer 3 problèmes structurels:
1. Agent aveugle (position maintenant visible via Dict obs)
2. Double Tanh (remplacé par LeakyReLU)
3. Reward saturé (tanh retiré, reward_scaling=1.0)

Ces fixes ont restauré l'exploration (gSDE std=0.05 au lieu de 0.00).
Mais les rewards stagnent toujours.

J'ai accès à:
- Le code source complet
- Les logs TensorBoard en temps réel
- Les métriques d'entraînement

Voici les observations actuelles (Step 114900/150000):
```

---

## Métriques TensorBoard Actuelles

### Progression Générale
| Métrique | Valeur |
|----------|--------|
| Step | 114,900 / 150,000 (77%) |
| mean_reward_10ep | **-3.24** (stagnant) |
| NAV | ~10,000 (break-even) |
| trades_per_episode | **1391** (très élevé!) |

### Composantes du Reward
| Composante | Valeur | Note |
|------------|--------|------|
| log_return | -0.002 à +0.002 | Quasi-nul |
| smoothness_penalty | -0.0002 | Trop faible |
| penalty_vol | -0.0003 | Négligeable |
| churn_penalty | 0 | Désactivé (coef=0) |

### État de l'Agent
| Métrique | Valeur | Interprétation |
|----------|--------|----------------|
| position | oscille -1 ↔ +1 | Churn excessif |
| position_delta | 0.35 à 1.15 | Changements fréquents |
| gSDE std | 0.0503 | Exploration OK |
| ent_coef | 0.295 | A chuté de 0.87 |
| actor_loss | -1.06 à -1.62 | Normal pour SAC/TQC |
| critic_loss | 0.01-0.03 | Très bas |

---

## Configuration Actuelle

```python
# Hyperparamètres reward (training.py)
reward_scaling: float = 1.0      # Réduit de 50
smooth_coef: float = 0.0001      # Réduit de 100x
churn_coef: float = 0.0          # Désactivé

# Calcul smoothness_penalty
smoothness_penalty = -smooth_coef * (position_delta ** 2)
# Exemple: -0.0001 * (1.0)^2 = -0.0001 (négligeable!)
```

---

## Hypothèses à Explorer

### H1: smooth_coef trop faible
- Avec 0.0001, même un delta de 1.0 donne -0.0001 de pénalité
- L'agent ne "sent" pas la pénalité de churn

### H2: Pas de signal de profit
- log_return oscille autour de 0
- L'agent n'a pas de gradient clair vers le profit

### H3: Discount factor (gamma=0.95)
- Peut-être trop court pour du trading horaire?
- 8760 heures/an, horizon effectif = 1/(1-0.95) = 20 steps

### H4: Commission trop élevée
- commission=0.15% par trade
- Avec 1391 trades, coût = 1391 * 0.15% * 2 ≈ 4.17 (allers-retours)
- Mange tout le profit potentiel

---

## Questions pour Gemini

1. **Le smooth_coef de 0.0001 est-il suffisant pour décourager le churn?**
   - Quelle valeur recommandes-tu?

2. **Le critic_loss très bas (0.01-0.03) est-il normal ou signe d'underfitting?**

3. **Comment interpréter ent_coef qui chute de 0.87 → 0.29?**
   - Est-ce que l'agent devient trop déterministe?

4. **Y a-t-il un problème avec la structure de reward?**
   - log_return * reward_scaling = presque rien
   - Faut-il amplifier le signal de profit?

5. **Le gamma=0.95 est-il approprié pour du trading horaire?**

---

## Code Pertinent

### Calcul du Reward (env.py:320-345)

```python
def _calculate_reward(self, new_position: float) -> float:
    # 1. Log-return
    current_price = self.prices[self.current_step]
    prev_price = self.prices[self.current_step - 1]
    log_return = np.log(current_price / prev_price)

    # 2. PnL basé sur position
    pnl = self.current_position_pct * log_return

    # 3. Pénalité volatilité (Sortino-like)
    if pnl < 0:
        penalty_vol = self.downside_coef * (pnl ** 2)
    else:
        penalty_vol = 0

    # 4. Smoothness penalty (anti-churn)
    position_delta = abs(new_position - self.current_position_pct)
    smoothness_penalty = self.smooth_coef * (position_delta ** 2)

    # 5. Total
    total_reward = pnl - penalty_vol - smoothness_penalty

    # 6. Scaling (plus de tanh!)
    scaled_reward = float(total_reward * self.reward_scaling)

    return scaled_reward
```

---

## Réponses de Gemini

### Diagnostic Principal: "Gap Mathématique"

**L'agent pense que le trading est quasi-gratuit!**

| Coût | Valeur | Calcul |
|------|--------|--------|
| **Coût Réel (NAV)** | 0.30% | Commission 0.15% × 2 (aller-retour) |
| **Coût Perçu (Reward)** | 0.0004 | smooth_coef × delta² = 0.0001 × 2² |
| **Gap** | **7.5x** | L'agent paie 7.5x moins cher dans son monde |

> "L'agent pense que le trading est quasi-gratuit, donc il chasse le moindre bruit
> de marché (overtrading), ce qui détruit ton NAV avec les vrais frais."

### Réponses aux Questions

| Question | Réponse Gemini |
|----------|----------------|
| smooth_coef=0.0001 trop faible? | **OUI**, négligeable face au bruit log_return |
| critic_loss 0.01-0.03 normal? | Non! Proportionnellement énorme avec rewards ~0.001 |
| ent_coef 0.87→0.29 problème? | **Non, sain**. TQC réduit l'entropie naturellement |
| Amplifier log_return? | **ABSOLUMENT**. Précision flottante et gradients noyés |
| gamma=0.95 approprié? | **NON, trop court**. Horizon = 20h, agent myope |

### Recommandations

1. **Scaling x100** : 1% return = 1.0 reward (au lieu de 0.01)
2. **Réactiver churn_penalty** : Aligné avec commission réelle
3. **gamma = 0.99** : Horizon 100-200h au lieu de 20h
4. **Curriculum Learning** : smooth_coef de 0 → cible progressivement

### Code Proposé par Gemini

```python
# 1. Scaling Factor "Psychologique"
SCALE = 100.0  # 1% gain = +1.0 reward

# 2. Composantes Scalées
reward_log_return = np.log(1.0 + safe_return) * SCALE

# 3. Churn aligné avec réalité
if position_delta > 0:
    cost_rate = self._get_commission() + self.slippage  # 0.0015
    churn_penalty = -position_delta * cost_rate * 1.0 * SCALE

# 4. Smoothness (régularisation légère)
smoothness_penalty = -0.005 * (position_delta ** 2) * SCALE

# 5. Total (SANS TANH)
total_reward = reward_log_return + churn_penalty + smoothness_penalty
```

### Curriculum Learning (3 Phases)

| Phase | Timesteps | smooth_coef | But |
|-------|-----------|-------------|-----|
| Découverte | 0-20% | 0.0 | Explorer sans contrainte |
| Discipline | 20-80% | 0 → cible | Apprendre l'économie de mouvement |
| Raffinement | 80-100% | cible | Optimisation fine |

> "Si tu imposes une forte pénalité dès le début, l'agent apprend juste à
> ne rien faire (Hold 0). Si tu ne mets rien, il apprend à osciller (Churn)."

---

## Actions Post-Échange

- [x] Implémenter SCALE=100 dans env.py
- [x] Réactiver churn_penalty aligné avec commission
- [x] Passer gamma de 0.95 à 0.99
- [x] Implémenter Curriculum Learning pour smooth_coef
- [x] Relancer l'entraînement WFO
- [ ] Comparer les métriques (trades_per_episode < 500 attendu)

## Vérification Post-Implémentation (Step 4000)

```
curriculum/churn_coef: 0     ✓ Phase 1 (Discovery)
curriculum/smooth_coef: 0    ✓ Exploration libre
Position: +0.30, +1.00, -0.40  ✓ Agent explore
```

---

## Historique des Échanges

### Échange 1 (précédent)
- **Problème:** Agent n'apprend pas du tout
- **Solution:** Position visible + LeakyReLU + retirer tanh
- **Résultat:** Exploration restaurée (gSDE std > 0)

### Échange 2 (actuel)
- **Problème:** Rewards stagnent malgré exploration
- **Hypothèse principale:** smooth_coef trop faible, churn non pénalisé
- **Solution:** *En attente de Gemini*

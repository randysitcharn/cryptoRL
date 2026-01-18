# Améliorations Futures - CryptoRL

Liste des améliorations prévues pour le projet, priorisées par importance.

---

## P0 - Haute Priorité

### [ ] Short Selling Support
**Fichier:** `src/training/batch_env.py` (lignes 522-525)

**Problème actuel:**
```python
# Map [-1, 1] to exposure [0, 1]
target_exposures = (target_positions + 1.0) / 2.0
```
Actuellement, action=-1 = 0% (cash), action=1 = 100% long. Pas de positions short.

**Solution proposée:**
- Mapping symétrique : action=-1 = -100% short, action=0 = cash, action=1 = +100% long
- Modifier le calcul de NAV pour supporter positions négatives
- Ajouter paramètre `allow_short: bool = True` pour activer/désactiver

**Impact:** Permet à l'agent de profiter des marchés baissiers.

---

### [ ] Curriculum Lambda Max Tuning
**Fichier:** `src/training/batch_env.py` (ligne 843)

**Problème actuel:**
```python
# Phase 3: Stability - fixed discipline
self.curriculum_lambda = 0.4
```
Le lambda max est hardcodé à 0.4.

**Solution proposée:**
- Rendre configurable via paramètre `curriculum_lambda_max: float = 0.4`
- Expérimenter avec valeurs 0.3, 0.5, 0.6 pour trouver l'optimum
- Logger la valeur dans TensorBoard pour analyse

**Impact:** Permet de tuner le ratio PnL/Penalties selon les résultats OOS.

---

## P1 - Moyenne Priorité

### [ ] Funding Rate pour Shorts
**Fichier:** `src/training/batch_env.py`

**Description:**
Ajouter un coût de funding réaliste pour les positions short (comme sur les perpetual futures).

**Solution proposée:**
- Paramètre `funding_rate: float = 0.0001` (0.01% par step, ~0.24%/jour)
- Appliquer uniquement sur positions négatives
- Déduire du cash à chaque step

**Impact:** Rend le short selling plus réaliste et évite l'abus de positions short longue durée.

---

### [ ] Smooth Coef Tuning
**Fichier:** `src/training/callbacks.py` (ligne 597)

**Problème actuel:**
```python
{'end_progress': 0.3, 'churn': (0.10, 0.50), 'smooth': (0.0, 0.005)},
```
`smooth_coef` réduit à 0.005 pour "unblock trading".

**Solution proposée:**
- Monitorer le nombre de trades par épisode
- Si < 10 trades/épisode, c'est OK
- Si agent ne trade jamais, augmenter progressivement (0.01, 0.02)

**Impact:** Balance entre réduction du churn et capacité à trader.

---

## P2 - Basse Priorité

### [ ] Observation Noise Adaptive
**Fichier:** `src/training/batch_env.py`

**Description:**
Réduire le bruit d'observation progressivement pendant le training (curriculum-style).

**Solution proposée:**
- Début: `observation_noise = 0.02` (2%)
- Fin: `observation_noise = 0.005` (0.5%)
- Décroissance linéaire basée sur `progress`

**Impact:** Exploration forte au début, précision à la fin.

---

### [ ] Multi-Asset Support
**Fichier:** `src/training/batch_env.py`

**Description:**
Étendre BatchCryptoEnv pour gérer un portefeuille multi-assets (BTC + ETH).

**Solution proposée:**
- Action space: `Box(-1, 1, shape=(n_assets,))`
- Positions indépendantes par asset
- Contrainte: somme des expositions <= max_leverage

**Impact:** Permet la diversification et les stratégies de spread.

---

## Notes

- Les items P0 sont bloquants pour les prochaines expérimentations
- Les items P1 améliorent le réalisme de la simulation
- Les items P2 sont des extensions futures

---

*Dernière mise à jour: 2026-01-18*

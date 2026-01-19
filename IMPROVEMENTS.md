# Améliorations Futures - CryptoRL

Liste des améliorations prévues pour le projet, priorisées par importance.

---

## P0 - Haute Priorité

### [x] Short Selling Support ✅ IMPLÉMENTÉ

**Fichier:** `src/training/batch_env.py` (lignes 681-684)

**Statut:** ✅ **IMPLÉMENTÉ** (2026-01-19)

**Implémentation actuelle:**
```python
# Direct mapping: -1=100% short, 0=cash, +1=100% long
target_exposures = target_positions
target_values = old_navs * target_exposures
target_units = target_values / old_prices
```

**Fonctionnalités:**
- ✅ Mapping symétrique : action=-1 = -100% short, action=0 = cash, action=1 = +100% long
- ✅ Calcul NAV supporte positions négatives (`cash + positions * prices`)
- ✅ Action space `[-1, 1]` et position space `[-1, 1]`
- ✅ Funding rate pour positions short (voir P1 ci-dessous)

**Impact:** L'agent peut profiter des marchés baissiers.

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

### [x] Funding Rate pour Shorts ✅ IMPLÉMENTÉ

**Fichier:** `src/training/batch_env.py` (lignes 702-706)

**Statut:** ✅ **IMPLÉMENTÉ** (2026-01-19)

**Implémentation actuelle:**
```python
# 6b. Apply funding cost for short positions (perpetual futures style)
if self.funding_rate > 0:
    short_mask = self.positions < 0
    funding_cost = torch.abs(self.positions) * old_prices * self.funding_rate
    self.cash = torch.where(short_mask, self.cash - funding_cost, self.cash)
```

**Fonctionnalités:**
- ✅ Paramètre `funding_rate: float = 0.0001` (0.01% par step, ~0.24%/jour)
- ✅ Appliqué uniquement sur positions négatives (`positions < 0`)
- ✅ Déduit du cash à chaque step
- ✅ Configurable via constructeur de `BatchCryptoEnv`

**Impact:** Short selling réaliste avec coût de funding style perpetual futures.

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

### [x] Data Augmentation - Dynamic Noise (Annealing + Volatility-Adaptive)

**Fichier:** `src/training/batch_env.py`

**Statut:** ✅ **IMPLÉMENTÉ** (2026-01-19) - Voir `docs/AUDIT_OBSERVATION_NOISE.md`

**Problème actuel:**
```python
noise = torch.randn_like(market) * self.observation_noise  # Bruit fixe à 1%
```
Le bruit est constant quelle que soit la volatilité du marché et la progression du training.

**Solution approuvée (combinée):**
```python
if self.observation_noise > 0 and self.training:
    # 1. ANNEALING (Time-based) - Standard NoisyRollout 2025
    annealing_factor = 1.0 - 0.5 * self.progress
    
    # 2. ADAPTIVE (Regime-based) - Innovation CryptoRL
    current_vol = torch.sqrt(self.ema_vars).clamp(min=1e-6)
    target_vol = getattr(self, 'target_volatility', 0.015)
    vol_factor = (target_vol / current_vol).clamp(0.5, 2.0)  # CRITIQUE: garde-fous
    
    # 3. INJECTION COMBINÉE
    final_scale = self.observation_noise * annealing_factor * vol_factor
    noise = torch.randn_like(market) * final_scale.unsqueeze(1).unsqueeze(2)
    market = market + noise
```

**Intuition:** 
- Annealing: Exploration forte au début, précision à la fin (standard industriel)
- Volatility-Adaptive: Plus de bruit en marché calme, moins en marché volatile

**Impact:** Meilleure généralisation, convergence plus stable.

---

## P2 - Basse Priorité

### [x] ~~Observation Noise Adaptive~~ (Fusionné dans P1)

**Statut:** ✅ **FUSIONNÉ** dans "Dynamic Noise" (P1) - Voir audit 2026-01-19

L'annealing fait maintenant partie de la solution combinée approuvée.

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

### [ ] Data Augmentation - Magnitude Scaling

**Fichier:** `src/training/batch_env.py`

**Description:**
Multiplier les observations par un facteur aléatoire pour simuler différentes conditions de volatilité.

**Solution proposée:**
```python
if self.training and self.magnitude_scaling:
    scale = torch.empty(n_envs, 1, 1, device=self.device).uniform_(0.9, 1.1)
    market = market * scale
```

**Intuition:** Un mouvement de +5% et un mouvement de +5.5% sont essentiellement le même signal.

**Impact:** Simule différentes conditions de volatilité, préserve la structure relative des données.

---

### [ ] Data Augmentation - Time Warping

**Fichier:** `src/training/batch_env.py`

**Description:**
Étirer/compresser temporellement certaines portions de la série temporelle.

**Intuition:** Un rallye de 3 jours et un de 5 jours peuvent être le même pattern, juste à vitesse différente.

**Attention:** Complexe à implémenter. Peut casser les relations temporelles importantes (ex: momentum sur 24h).

**Impact:** Crée de la variété structurelle pour les patterns de chartisme.

---

## P3 - Futur

### [ ] Data Augmentation - Synthetic Episode Generation

**Fichier:** Nouveau module à créer

**Description:**
Générer des épisodes synthétiques avec des modèles génératifs (GANs, Diffusion Models) entraînés sur les données historiques.

**Impact:** Haute valeur si bien fait, mais effort très élevé. À considérer uniquement si les autres techniques sont insuffisantes.

---

### [ ] HMM Relative Artifacts + A/B Testing
**Fichier:** `src/data_engineering/features.py`

**Problème actuel:**
Les artifacts HMM sont fixes (probabilités de régime absolues).

**Solution proposée:**
- Passer à des artifacts relatifs (ex: changement de probabilité, distance au centroïde du régime, temps passé dans le régime actuel)
- Implémenter un framework A/B testing pour comparer les performances agent avec vs sans features HMM
- Métriques à comparer : Sharpe OOS, max drawdown, stabilité des performances

**Impact:** Valider objectivement l'apport du HMM et potentiellement améliorer la qualité des features de régime.

---

### [ ] 3 HMM Timeframes
**Fichier:** `src/data_engineering/features.py`

**Description:**
Entraîner plusieurs HMM sur différents timeframes pour capturer les régimes à plusieurs échelles temporelles.

**Solution proposée:**
- À définir (multi-timeframe, hiérarchique, ou ensemble)

**Impact:** Potentiellement capturer des régimes de marché à court, moyen et long terme.

---

### [x] A/B Testing: gSDE vs Actor Noise ✅ IMPLÉMENTÉ

**Fichier:** `src/training/train_agent.py`, `src/config/training.py`

**Statut:** ✅ **IMPLÉMENTÉ** (2026-01-19)

**Description:**
Support pour deux approches d'exploration pour TQC:
1. **gSDE (generalized State-Dependent Exploration):** Bruit dans l'espace des paramètres, corrélé au state (défaut)
2. **Actor Noise (OrnsteinUhlenbeckActionNoise):** Bruit sur les actions, indépendant du state

**Configuration:**
```python
# Config A: gSDE (défaut)
use_sde: bool = True

# Config B: Actor Noise
use_sde: bool = False
use_action_noise: bool = True      # Active OU noise quand gSDE off
action_noise_sigma: float = 0.1    # Écart-type du bruit (0.05-0.3)
action_noise_theta: float = 0.15   # Taux de retour à la moyenne
```

**Usage CLI:**
```bash
# Défaut: gSDE activé
python -m src.training.train_agent

# Alternative: OrnsteinUhlenbeck noise
python -m src.training.train_agent --no-sde --action-noise-sigma 0.1 --action-noise-theta 0.15
```

**Métriques à comparer (A/B testing):**
- Sharpe OOS (Walk-Forward)
- Max Drawdown
- Stabilité inter-folds
- Convergence speed (timesteps to plateau)
- Action smoothness (churn)

**Impact:** Permet de tester quelle stratégie d'exploration fonctionne mieux pour le trading RL.

---

## Propositions REJETÉES (Audit 2026-01-19)

Les propositions suivantes ont été évaluées et rejetées lors de l'audit. Voir `docs/AUDIT_OBSERVATION_NOISE.md` pour les justifications complètes.

### [x] ~~Feature-Specific Noise~~ 🔴 REJETÉ

**Raison:** Complexité de maintenance trop élevée pour gain marginal.

**Détails:**
- Mapping features → groupes fragile et difficile à maintenir
- Valeurs (0.5%, 2%, 1%, 0%) purement heuristiques sans validation
- Couplage fort avec pipeline de features
- ROI insuffisant : +5% estimé vs. effort permanent

**Alternative:** Reporter après validation des techniques approuvées (Dynamic Noise).

---

### [x] ~~SNI (Selective Noise Injection)~~ 🔴 REJETÉ

**Raison:** Changement architectural trop profond, hors scope.

**Détails:**
- Nécessite modification du forward pass ou architecture dual-path
- Impact sur toute la chaîne d'entraînement (TQC, callbacks)
- Paper original (NeurIPS 2024) testé sur CoinRun, pas finance
- Risque de régression élevé
- Effort : 1+ jour vs. quelques heures pour solutions approuvées

**Alternative:** Créer ticket de recherche pour évaluation future.

---

## Data Augmentation - Techniques à ÉVITER

| Technique | Pourquoi l'éviter |
|-----------|-------------------|
| **Flip temporel** | Le temps a une direction. Un pattern inversé temporellement devient complètement différent. |
| **Shuffling des features** | Les colonnes ont une sémantique fixe. Le modèle apprend que colonne 0 = prix. |
| **Mixup/CutMix** | Mélanger deux contextes de marché crée une chimère irréaliste (mi-bull mi-bear). |
| **Bruit trop fort (>5%)** | Détruit le signal. Le modèle apprend à ignorer les observations. |

---

## Notes

- Les items P0 sont bloquants pour les prochaines expérimentations
- Les items P1 améliorent le réalisme de la simulation
- Les items P2 sont des extensions futures
- Les items P3 sont des pistes de recherche à long terme
- Note: `random_start=True` (déjà implémenté) est une forme de **Window Slicing** (data augmentation)

---

*Dernière mise à jour: 2026-01-19*
*Audit Observation Noise: 2026-01-19 - Voir `docs/AUDIT_OBSERVATION_NOISE.md`*
*Mise à jour Short Selling + Funding Rate: 2026-01-19 - Marqués comme implémentés*

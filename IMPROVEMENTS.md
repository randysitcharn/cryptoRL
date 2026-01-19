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

### [ ] Data Augmentation - Volatility-Adaptive Noise

**Fichier:** `src/training/batch_env.py`

**Problème actuel:**
```python
noise = torch.randn_like(market) * self.observation_noise  # Bruit fixe à 1%
```
Le bruit est constant quelle que soit la volatilité du marché.

**Solution proposée:**
```python
volatility = torch.sqrt(self.ema_vars)  # Vol courante (déjà calculée)
noise_scale = self.observation_noise * (self.target_volatility / volatility)
noise = torch.randn_like(market) * noise_scale.unsqueeze(1).unsqueeze(2)
```

**Intuition:** Plus de bruit en marché calme (où l'overfitting est facile), moins en marché volatile (déjà bruité naturellement).

**Impact:** Meilleure généralisation sans détruire le signal en période volatile.

---

### [ ] Data Augmentation - Feature-Specific Noise

**Fichier:** `src/training/batch_env.py`

**Problème actuel:**
Le même niveau de bruit est appliqué à toutes les features, mais certaines sont plus sensibles que d'autres.

**Solution proposée:**
```python
noise_scales = {
    'price_features': 0.005,   # 0.5% - Très sensible
    'volume_features': 0.02,   # 2% - Plus tolérant
    'momentum_features': 0.01, # 1% - Modéré
    'regime_probs': 0.0        # 0% - Sortie de modèle, pas de bruit
}
```

**Impact:** Préserve les features sensibles tout en régularisant les autres.

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

*Dernière mise à jour: 2026-01-18*

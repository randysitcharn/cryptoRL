# Data Pipeline Audit Report - CryptoRL

**Date**: 2026-01-22
**Méthode**: Recursive Prompt Architecture
**Scope**: Pipeline de données (Ingestion → Features → HMM → Scaling → Export)
**Contre-Audit**: 2026-01-22 - Senior Staff Engineer Review ✅

---

## Executive Summary

**Score Global: 7.5/10** | **Validation Externe: 95% accord**

### Forces
1. **Architecture solide** - Pipeline bien structuré avec séparation des responsabilités
2. **FFD SOTA** - Implémentation correcte de Lopez de Prado (2018) avec recherche adaptative du d optimal
3. **HMM robuste** - Archetype Alignment via Hungarian Algorithm résout le Semantic Drift

### Faiblesses Critiques
1. **[P0] RobustScaler Data Leakage** - fit() sur tout le dataset au lieu de train seulement ✅ **VALIDÉ CRITIQUE**
2. **[P0] Purge Window insuffisant** - 50h < 720h (Z-Score window) ✅ **VALIDÉ CRITIQUE**
3. **[P1] Funding Rate synthétique** - Agent apprend sur données "inventées" ✅ **VALIDÉ**

### ⚠️ BLOCAGE ENTRAÎNEMENT
> **Ne lancer AUCUN entraînement avant correction des P0.** Tout résultat actuel est invalide.

---

## Contre-Audit & Validation Externe

**Reviewer**: Senior Staff Engineer
**Date**: 2026-01-22
**Verdict**: Audit d'excellente qualité - 95% accord total

### Points P0 : Validation Totale 🔴

Ces deux erreurs invalident totalement les résultats d'un backtest. Si elles ne sont pas corrigées, l'agent apprend à "voir le futur".

#### 1. Leakage du RobustScaler - CONFIRMÉ FATAL

| Aspect | Analyse |
|--------|---------|
| **Verdict** | Erreur classique mais fatale |
| **Raison** | Le `RobustScaler` utilise la médiane et l'IQR (Interquartile Range). En faisant un `.fit()` sur tout le dataset, la médiane de t₀ est influencée par les prix de t₊₁₀₀₀. L'agent sait implicitement si le prix actuel est "haut" ou "bas" par rapport au futur. |
| **Action** | Impératif de `fit` uniquement sur `train_set` et `transform` le `test_set` avec statistiques figées du train |

#### 2. Purge Window (50h vs 720h) - CONFIRMÉ MATHÉMATIQUEMENT

| Aspect | Analyse |
|--------|---------|
| **Verdict** | Mathématiquement incontestable |
| **Calcul** | Si Z-Score sur 30 jours (720h) comme feature, la valeur à t contient de l'information de t-720 à t. Si test set commence à T, les données entre T-720 et T-1 sont partagées avec train set. |
| **Conséquence** | Le début du validation set est pollué par la fin du training set. Performances artificiellement élevées au début de chaque période de test. |

### Points P1 : Accord avec Nuances 🟠

#### 1. Funding Rate Synthétique - CONFIRMÉ DANGEREUX

| Aspect | Analyse |
|--------|---------|
| **Verdict** | Dangereux pour le RL |
| **Raison** | Le Funding Rate est souvent corrélé à l'euphorie (prix haut + vol haute). En générant un processus Ornstein-Uhlenbeck décorrélé de la réalité, feature bruitée fournie à l'agent. |
| **Impact** | Au mieux: apprend à l'ignorer (gaspillage capacité réseau). Au pire: trouve corrélations fallacieuses. |
| **Recommandation** | Si pas d'historique Binance, **supprimer** la feature plutôt qu'en inventer une fausse. |

#### 2. HMM Fit sur tout le segment - NUANCÉ

| Aspect | Analyse |
|--------|---------|
| **Verdict** | "Moins pire" que le scaler dans un contexte WFO |
| **Raison** | Walk-Forward Optimization refit le modèle périodiquement |
| **Amélioration puriste** | Pour suivre Lopez de Prado strictement, HMM devrait être entraîné en mode "Expanding Window" uniquement sur le passé disponible pour éviter biais rétrospectif sur la définition des régimes. |

### Points Techniques Validés 🟡

#### Garman-Klass Négatif - CONFIRMÉ

| Aspect | Analyse |
|--------|---------|
| **Verdict** | Point très fin mais juste |
| **Raison** | En théorie, si H≈L et que le gap d'ouverture est énorme par rapport au range intra-bougie, le terme sous la racine peut devenir négatif. |
| **Solution** | Passer à **Rogers-Satchell** (plus robuste aux gaps et au drift) |

#### FFD Performance - CONFIRMÉ

| Aspect | Analyse |
|--------|---------|
| **Verdict** | Critique valide sur la lenteur |
| **Solution** | Implémentation via FFT ou vectorisation avec `numpy`/`numba` accélérerait le pipeline de 10x à 100x |
| **Note** | Le seuil `min_d_floor=0.30` est arbitraire. Préférable de laisser l'ADF test décider ou imposer test de corrélation. |

### Points Manquants Ajoutés par Contre-Audit 🆕

Ces points n'étaient pas dans l'audit initial et sont ajoutés suite à la review:

#### 1. [P1-NEW] Embargo vs Purge

| Aspect | Détail |
|--------|--------|
| **Concept** | L'audit parle de "Purge" (retirer données chevauchantes Train/Test). C'est bien. |
| **Manque** | L'**Embargo** - Après un trade de test, il faut éliminer quelques échantillons *après* le test set avant de reprendre le train suivant. |
| **Raison** | Laisser "retomber" les corrélations temporelles des labels (surtout si Triple Barrier Method utilisée) |
| **Priorité** | P1 |

#### 2. [P1-NEW] Stationnarité du Scaler en Production

| Aspect | Détail |
|--------|--------|
| **Problème** | Même en fitant `RobustScaler` uniquement sur Train, risque si régime de volatilité change drastiquement (ex: passage 2017→2018 en crypto). |
| **Symptôme** | Scaler calibré sur 2017 va exploser (valeurs > 10 ou < -10) en 2018. |
| **Solution** | Utiliser **Rolling Scaler** ou **Dynamic Z-Score** comme input réseau, plutôt que scaler global statique par fold. Plus robuste pour la production. |
| **Priorité** | P1 |

---

## Étape 1a : Audit Téléchargement + Validation Multi-Assets

**Fichier**: `src/data_engineering/loader.py` (lignes 42-168)

### Téléchargement (_download_asset)

| Aspect | Status | Risque | Recommandation |
|--------|--------|--------|----------------|
| Auto-adjust | ✅ Correct | Faible | Dividendes/splits gérés automatiquement |
| MultiIndex | ✅ Géré | Faible | `columns.get_level_values(0)` ligne 76 |
| Timeout/Retry | ❌ Absent | **Moyen** | Ajouter `tenacity` avec exponential backoff |
| Rate Limiting | ❌ Absent | Moyen | Yahoo peut ban si trop de requêtes |
| Limite 730j | ✅ Respectée | Faible | `min(days, 729)` ligne 54 |

### Validation (_validate_raw_data)

| Check | Implementation | Edge Cases | Score |
|-------|----------------|------------|-------|
| Prix ≤ 0 | → NaN → ffill | ✅ Géré | 8/10 |
| Volumes < 0 | → 0 | ✅ Géré | 9/10 |
| Duplicats index | keep='last' | ⚠️ Pourquoi pas 'first'? | 7/10 |
| Gaps temporels | 24h crypto, 72h macro | ✅ Approprié | 9/10 |
| ffill final | Sans limite | ⚠️ Propagation longue possible | 6/10 |

### Bugs Potentiels

1. **[BUG-1a-1]** Pas de retry automatique - si Yahoo échoue, le ticker est silencieusement ignoré
2. **[BUG-1a-2]** Pas de validation du format des colonnes retournées - si yfinance change son API
3. **[BUG-1a-3]** DataFrame vide retourné sans raise - peut causer erreurs downstream

### Edge Cases Non Gérés

- Si `^GSPC` retourne DataFrame vide → skip silencieux (ligne 276)
- Si BTC-USD < 100 lignes → pas de validation minimum
- Timezone: index est naive (pas de tz-aware), suppose UTC

---

## Étape 1b : Audit Synchronisation Master Index + Funding Rate

**Fichier**: `src/data_engineering/loader.py` (lignes 170-376)

### Synchronisation

| Aspect | Implementation | Risque Data Leakage | Score |
|--------|----------------|---------------------|-------|
| Master Index | BTC-USD (24/7) | 🟢 None | 9/10 |
| Forward-fill | ffill only (pas bfill) | 🟢 Évite look-ahead | 10/10 |
| Timestamp Macro | floor('h') ligne 285 | 🟢 Correct | 9/10 |
| Concaténation | pd.concat(axis=1) | 🟢 Alignement garanti | 9/10 |

### Règle "Ne jamais supprimer de lignes"

| Comportement | Économiquement Correct | Impact HMM |
|--------------|------------------------|------------|
| Week-end ffill | ⚠️ Prix stables artificiels | HMM voit "fausse" consolidation |
| Nuit US ffill | ✅ Acceptable | Minimal |

### Funding Rate Synthétique (Ornstein-Uhlenbeck)

| Paramètre | Valeur | Réalisme BTC Perp | Recommandation |
|-----------|--------|-------------------|----------------|
| mu | 0.0001 (0.01%/h) | ✅ Historiquement correct | - |
| theta | 0.1 | ✅ Mean reversion ~10h | - |
| sigma | 0.0002 | ⚠️ Faible | Augmenter à 0.0005 |
| Clip | [-0.001, 0.003] | ⚠️ Range réel: [-0.03, 0.10] | Élargir |
| Seed | 42 (fixe) | ⚠️ Même funding tous les runs | Randomiser ou charger réel |

### Questions Ouvertes

1. **Pourquoi funding synthétique?** - Les vrais funding rates sont disponibles via Binance API
2. **Impact agent?** - Apprend sur signal corrélé artificiellement à la vol
3. **Utilisé dans reward?** - À vérifier dans `batch_env.py`

---

## Étape 2a : Audit FFD (Fractional Differentiation)

**Fichier**: `src/data_engineering/features.py` (lignes 144-284)

### FFD Implementation

| Composant | Conforme AFML | Performance | Risque |
|-----------|---------------|-------------|--------|
| Poids (_get_weights_ffd) | ✅ Formule récursive correcte | O(window) | Faible |
| Application (_ffd) | ✅ Fixed-Width Window | O(n × window) | ⚠️ Lent (boucle Python) |
| find_min_d | ✅ ADF test itératif | O(d_range × n) | Faible |
| min_d_floor | 0.30 | ⚠️ Arbitraire | Justification empirique? |

### Détails Techniques

| Élément | Valeur | Commentaire |
|---------|--------|-------------|
| threshold | 1e-5 | Standard pour troncature des poids |
| ffd_window | 100h (~4 jours) | ⚠️ Court pour données horaires |
| ADF maxlag | 1 | Conservateur, OK |
| ADF regression | 'c' (constant) | Standard |
| Fallback d | 1.0 | Différenciation complète si ADF échoue |

### Cache FFD_D_OPTIMAL_CACHE

- Cache vide `{}` ligne 36 → force ADF test par segment ✅
- Pas de persistance entre runs → recalculé à chaque fois ⚠️
- Thread-safety: aucune protection (OK si single-threaded)

### Valeurs Typiques Attendues

| Asset | d_optimal attendu | Justification |
|-------|-------------------|---------------|
| BTC | 0.30-0.50 | Prix très persistant |
| ETH | 0.35-0.55 | Similaire BTC |
| SPX | 0.40-0.60 | Moins persistant que crypto |

### Optimisations Suggérées

1. **Vectorisation** - Remplacer boucle Python par `np.convolve`
2. **Caching** - Persister d_optimal par asset et période
3. **Window adaptatif** - Ajuster ffd_window selon volatilité

---

## Étape 2b : Audit Indicateurs de Volatilité

**Fichier**: `src/data_engineering/features.py` (lignes 290-408)

### Volatility Indicators

| Indicateur | Formule OK | Bias | Variance | Score |
|------------|------------|------|----------|-------|
| Parkinson | ✅ sqrt(1/(4*ln2) × log²(H/L)) | Low | Low | 9/10 |
| Garman-Klass | ✅ sqrt(0.5×log²(H/L) - 0.386×log²(C/O)) | Medium | ⚠️ Peut être négatif | 7/10 |
| Z-Score | ✅ (P - μ) / σ | None | N/A | 9/10 |

### Edge Cases Matrix

| Condition | Parkinson | GK | Z-Score |
|-----------|-----------|-----|---------|
| H = L (Doji) | 0 ✅ | Peut être négatif ⚠️ | Normal |
| C = O | Normal | ⚠️ Terme négatif | Normal |
| Std → 0 | Normal | Normal | Explose (ε=1e-8 trop petit) |

### Problème GK Négatif

```python
# Ligne 363: peut produire valeur négative sous racine
gk = np.sqrt(0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2))
# Si |log_co| > 1.14 × |log_hl| → négatif sous racine → NaN
```

**Recommandation**: Utiliser Rogers-Satchell qui ne peut pas être négatif:
```python
rs = log_hl × (log_ho + log_hc) + log_lc × (log_lo + log_lc)
```

### Z-Score Window

- 720h = 30 jours → horizon long terme ✅
- Epsilon 1e-8 → négligeable vs std typique ✅
- Calculé sur Close brut (pas log) → correct pour comparaison cross-asset ✅

---

## Étape 2c : Audit Volume Features + Sanitization

**Fichier**: `src/data_engineering/features.py` (lignes 71-526)

### Volume Features

| Feature | Formula | Missing Data Handling | Risk |
|---------|---------|----------------------|------|
| Vol_LogRet | log(v_t/v_{t-1}) | 0→1 (neutral) ✅ | Low |
| Vol_ZScore | Z(vol, 336h) | Zero Padding ⚠️ | Medium |

### Sanitization (_sanitize_prices)

| Step | Action | Justification | Concern |
|------|--------|---------------|---------|
| Prix=0 | →NaN→ffill→bfill | Évite log(0) | ⚠️ bfill = look-ahead |
| Ordre ffill→bfill | Priorité passé | ✅ Correct | - |
| Logging | Print count | ✅ Debug utile | - |

### Validation (_validate_features)

| Aspect | Implementation | Issue |
|--------|----------------|-------|
| Seuil | \|value\| > 10 | ⚠️ Pourquoi 10? FFD peut dépasser |
| Action | Warning only | ⚠️ Pas de clipping automatique |

### Log-Returns Clipping

| Event | Real Return | Clipped Return | Info Lost |
|-------|-------------|----------------|-----------|
| COVID Mar 2020 | -50% daily | -20% | ✅ OUI (mais hourly OK) |
| LUNA May 2022 | -99% daily | -20% | ✅ OUI |
| FTX Nov 2022 | -25% daily | -20% | Partiel |

**Justification**: ±20%/h est extrême. Un move > 20%/h est probablement une erreur de données.

### Pipeline Order

1. Sanitize → ✅ Avant log pour éviter log(0)
2. LogRet → ✅ Avec clip
3. Volume → ✅ Zero padding si absent
4. Parkinson → ✅ Rolling 24h
5. GK → ✅ Rolling 24h
6. ZScore → ✅ Rolling 720h
7. FFD → ✅ Le plus coûteux en dernier
8. Clean → ✅ dropna
9. Validate → ✅ Check extrêmes

---

## Étape 3a : Audit HMM Features + K-Means Init

**Fichier**: `src/data_engineering/manager.py` (lignes 122-248)

### HMM Features Analysis

| Feature | Information Content | Redundancy | Data Quality |
|---------|---------------------|------------|--------------|
| HMM_Trend | Trend direction (168h MA) | ⚠️ Avec FFD? | ✅ |
| HMM_Vol | Volatility level (Parkinson 168h) | ⚠️ Avec Parkinson? | ✅ |
| HMM_Momentum | RSI 14 [0,1] | Unique | ✅ |
| HMM_Funding | Funding 24h MA | Unique | ⚠️ Synthétique |
| HMM_RiskOnOff | SPX - DXY (168h) | Unique | ✅ |
| HMM_VolRatio | vol_short/vol_long | Unique | ⚠️ Instable si vol_long→0 |

### Clipping des Features

| Feature | Clip Range | Justification | Issue |
|---------|------------|---------------|-------|
| HMM_Trend | [-0.05, 0.05] | ±5%/h max | ✅ Raisonnable |
| HMM_Vol | [0, 0.2] | Max 20%/h | ⚠️ Large, jamais atteint |
| HMM_Momentum | [0, 1] | RSI borné | ✅ Naturel |
| HMM_Funding | [-0.005, 0.005] | ±0.5% | ✅ Réaliste |
| HMM_RiskOnOff | [-0.02, 0.02] | ±2% | ✅ Cohérent |
| HMM_VolRatio | [0.2, 5.0] | Ratio 0.2x-5x | ⚠️ Flash crash peut dépasser |

### K-Means Initialization

| Aspect | Value | Assessment |
|--------|-------|------------|
| n_clusters | 4 (fixé) | ⚠️ Pas de recherche Elbow/Silhouette |
| n_init | 10 | ✅ Robuste |
| Noise injection | σ=0.1 | ✅ Magnitude appropriée pour z-scores |

### Problème: Funding Synthétique dans HMM

Le HMM apprend des patterns basés sur un signal **inventé**. Impact:
- Corrélation artificielle Funding ↔ Volatilité
- Agent peut sur-apprendre ce pattern inexistant en réalité

**Recommandation**: Remplacer par vrais funding rates Binance ou supprimer HMM_Funding

---

## Étape 3b : Audit Fit HMM + Transition Penalty

**Fichier**: `src/data_engineering/manager.py` (lignes 250-600)

### GMMHMM Configuration

| Parameter | Value | Justification | SOTA |
|-----------|-------|---------------|------|
| n_components | 4 | Domain knowledge | ✅ |
| n_mix | 2 | Flexibility | ⚠️ Risque overfit |
| covariance_type | 'diag' | Stability | ✅ |
| n_iter | 200 | Convergence | ✅ Suffisant |
| min_covar | 1e-3 | Regularization | ✅ |
| init_params | 'stc' | Pas 'm' (K-Means inject) | ✅ |

### Transition Penalty (Sticky HMM)

| Aspect | Implementation | Reference |
|--------|----------------|-----------|
| Formule | A_sticky = A × (1-p) + I × p | ✅ Conforme |
| penalty | 0.1 default | Shu et al. 2024 |
| Renormalisation | Lignes somment à 1 | ✅ |

| penalty | Diag Average | Regime Duration | Reactivity |
|---------|--------------|-----------------|------------|
| 0.0 | ~0.3-0.5 | ~5-10h | High |
| 0.1 | ~0.4-0.6 | ~10-20h | Medium |
| 0.3 | ~0.6-0.8 | ~20-50h | Low |

### Retry Logic

| Aspect | Implementation | Issue |
|--------|----------------|-------|
| MAX_RETRIES | 3 | ✅ Raisonnable |
| Critère | n_active_states >= 3 | ✅ |
| Random state | +17 par attempt | ⚠️ Non reproductible |
| Best selection | Max active states | ⚠️ Pas de métrique de qualité secondaire |

### Reproducibility Issues

1. **[ISSUE]** Pas de seed fixe entre runs → résultats différents
2. **[ISSUE]** Retry avec random_state changeant → non déterministe
3. **[ISSUE]** K-Means + HMM random → double source d'aléa

---

## Étape 3c : Audit Archetype Alignment + Quality Validation

**Fichier**: `src/data_engineering/manager.py` (lignes 302-422)

### Archetype Calibration

| Archetype | mean_ret | mean_vol | BTC 2020-2024 Observé | Match |
|-----------|----------|----------|----------------------|-------|
| Crash | -0.50%/h | 4.0%/h | Mar 2020: -15%/day | ⚠️ Surestimé |
| Downtrend | -0.10%/h | 1.5%/h | Bear 2022: -0.05%/h | ✅ OK |
| Range | 0.00%/h | 0.5%/h | Consolidation | ✅ OK |
| Uptrend | +0.15%/h | 2.0%/h | Bull 2021: +0.10%/h | ⚠️ Surestimé |

### Hungarian Algorithm

| Aspect | Implementation | Alternative | Recommendation |
|--------|----------------|-------------|----------------|
| Distance | Euclidean pondérée | Mahalanobis | Keep Euclidean (simpler) |
| Weights | [1.0, 2.0] (ret, vol) | Learned | ⚠️ Pourquoi 2x vol? |
| Inverse transform | ✅ De z-scores vers brut | - | Correct |

### Quality Metrics

| Metric | Threshold | Rationale | Improvement |
|--------|-----------|-----------|-------------|
| n_active >= 3 | 3/4 states | Allow 1 inactive | Consider require 4 |
| separation_score | None (info only) | std(mean_returns) | Add threshold > 0.001 |
| min_proportion | 5% | Avoid empty states | ✅ OK |

### Semantic Drift Risk

| Scenario | Without Alignment | With Alignment |
|----------|-------------------|----------------|
| Prob_0 meaning | Varies per segment | Always "Crash-like" |
| Cross-segment comparison | ❌ Invalid | ✅ Valid |
| Temporal consistency | ❌ Drift | ✅ Stable |

**Risque résiduel**: Si le marché évolue (2017 ≠ 2024), les archétypes fixes peuvent ne plus être représentatifs.

---

## Étape 4a : Audit Pipeline Orchestration + Data Leakage

**Fichier**: `src/data_engineering/manager.py` (lignes 852-997)

### Data Leakage Analysis

| Component | Train-Test Separation | Severity | Fix Required |
|-----------|----------------------|----------|--------------|
| FFD find_min_d | Uses full segment | 🟡 Low | Optional (ADF test only) |
| HMM fit | Uses full segment | 🟡 Low | In WFO: refit per segment |
| RobustScaler | **Uses full dataset** | 🔴 **CRITICAL** | **MANDATORY** |
| Z-Score | Rolling (causal) | 🟢 None | No |
| Parkinson/GK | Point-wise | 🟢 None | No |
| LogRet clip | Point-wise | 🟢 None | No |

### Pipeline Execution Order

```
1. Load           → ✅ No leakage
2. Features       → ✅ Causal indicators (rolling backward only)
3. HMM            → ⚠️ Full segment (OK in WFO where refit)
4. Clean (dropna) → ✅ No leakage
5. Scale          → 🔴 LEAKAGE (fit on full data)
6. Export         → ✅ No leakage
```

### Critical Fix Required

**Problème ligne 953**:
```python
df[cols_to_scale] = self.scaler.fit_transform(df[cols_to_scale])
# ↑ fit() utilise TOUT le dataset, y compris le futur
```

**Fix in WFO**:
```python
# In run_full_wfo.py, fit scaler on TRAIN only:
scaler.fit(train_df[cols_to_scale])
train_df[cols_to_scale] = scaler.transform(train_df[cols_to_scale])
test_df[cols_to_scale] = scaler.transform(test_df[cols_to_scale])
```

### Source Priority

| Priority | Source | Duration | Status |
|----------|--------|----------|--------|
| 1 | raw_historical/ | 8 years | ✅ Si disponible |
| 2 | HistoricalDownloader | 8 years | ⚠️ Requires API key |
| 3 | Yahoo Finance | 730 days | Fallback |

---

## Étape 4b : Audit Scaling Strategy + Serialization

**Fichier**: `src/data_engineering/manager.py` (lignes 830-1043)

### Scaling Strategy

| Aspect | Current | Alternative | Recommendation |
|--------|---------|-------------|----------------|
| Scaler | RobustScaler | QuantileTransformer | Keep Robust (better for fat tails) |
| Outliers | Already clipped ±20% | - | Redundant but safe |
| Target range | ~[-2, 2] | - | Good for NN |

### Exclude List Analysis

| Column Category | Excluded | Rationale | Concern |
|-----------------|----------|-----------|---------|
| OHLC bruts | YES | Pas des features | ✅ Correct |
| Volumes bruts | YES | Utilise Vol_* | ✅ Correct |
| Log-Returns | YES | "Clippés ±20%" | ⚠️ Asymétrie avec FFD scalé |
| Prob_* | YES | "[0,1] range" | ✅ Correct |
| HMM_* | NO (pas listé) | Intermédiaires | ⚠️ Devraient être exclus |

**Issue**: Les colonnes HMM_* ne sont pas dans `exclude_from_scaling` mais sont des intermédiaires qui ne devraient pas être scalés.

### Serialization

| Format | Usage | Compatibility | Risk |
|--------|-------|---------------|------|
| pickle | Scaler | Python version dependent | Medium |
| parquet | Data | Cross-platform | Low |

### Consistency Checks

- `cols_to_scale` calculé dynamiquement → peut changer entre runs
- Pas de validation scaler columns vs DataFrame columns
- Si nouvelles colonnes ajoutées → scaler incompatible

---

## Étape 5 : Audit Splitting Strategy

**Fichier**: `src/data_engineering/splitter.py`

### Split Strategy

| Set | Ratio | Bars (60k example) | Purge |
|-----|-------|-------------------|-------|
| Train | 70% | 42,000 | - |
| Gap | - | 50 | ✅ |
| Val | 15% | 8,950 | - |
| Gap | - | 50 | ✅ |
| Test | 15% | 8,950 | - |

### Purge Window Analysis

| Indicator | Window Size | Purge Required | Current Purge | Sufficient |
|-----------|-------------|----------------|---------------|------------|
| Z-Score | 720h | 720h | 50h | ❌ **NO** |
| FFD | 100h | 100h | 50h | ❌ **NO** |
| HMM Smoothing | 168h | 168h | 50h | ❌ **NO** |
| Parkinson | 24h | 24h | 50h | ✅ YES |
| GK | 24h | 24h | 50h | ✅ YES |
| Vol_ZScore | 336h | 336h | 50h | ❌ **NO** |

### Critical Issue

**purge_window=50h << max(indicator_windows)=720h**

Les indicateurs à longue fenêtre contaminent le test set:
- Z-Score utilise 720h de passé → 670h de train "fuient" dans val
- Le modèle peut implicitement voir des patterns futurs via ces features

### Fix Required

```python
# Calculer purge comme max de toutes les fenêtres
MAX_INDICATOR_WINDOW = 720  # Z-Score
purge_window = MAX_INDICATOR_WINDOW + 50  # Safety margin
```

### Usage Check

| Question | Answer |
|----------|--------|
| Used in production pipeline? | ❌ Non, DataManager.pipeline() ne l'utilise pas |
| Used in run_full_wfo.py? | ❌ Non, WFO a sa propre logique |
| Used in tests only? | ✅ Probablement code legacy |

---

## Étape 6 : Synthèse et Recommandations

### Risk Matrix (Mise à jour post contre-audit)

| ID | Finding | Prob | Impact | Priority | Validation |
|----|---------|------|--------|----------|------------|
| 4a-1 | Scaler fit on full dataset | H | H | **P0** | ✅ **CONFIRMÉ FATAL** |
| 5-1 | Purge window 50h < 720h | H | H | **P0** | ✅ **CONFIRMÉ FATAL** |
| 1b-1 | Synthetic funding rate | H | M | P1 | ✅ Confirmé dangereux |
| 3a-1 | HMM_Funding uses synthetic data | H | M | P1 | ✅ Confirmé dangereux |
| **NEW-1** | **Embargo manquant (post-test gap)** | M | M | **P1** | 🆕 Ajouté contre-audit |
| **NEW-2** | **Stationnarité scaler production** | M | M | **P1** | 🆕 Ajouté contre-audit |
| 3b-2 | HMM fit sur tout segment (hindsight) | M | M | P1 | ⚠️ Nuancé (OK si WFO) |
| 1a-1 | No retry on network error | M | L | P2 | - |
| 2b-1 | GK can be negative → NaN | M | L | P2 | ✅ Rogers-Satchell |
| 3b-1 | Non-reproducible HMM (random retry) | M | L | P2 | - |
| 4b-1 | HMM_* columns scaled (should exclude) | L | L | P2 | - |
| 2a-1 | FFD boucle Python lente | L | L | P3 | ✅ FFT/numba |
| 2a-2 | min_d_floor=0.30 arbitraire | L | L | P3 | ✅ Confirmé |

### Data Leakage Report (Validé par contre-audit)

#### Critical (P0) - BLOQUANTS ⛔

1. **RobustScaler** fit on full dataset ✅ **CONFIRMÉ FATAL**
   - **Location**: `manager.py:953`
   - **Fix**: Fit on train only, transform train+test separately
   - **Impact**: Test performance artificially inflated
   - **Validation**: La médiane de t₀ est influencée par les prix de t₊₁₀₀₀. L'agent sait implicitement si le prix est "haut" ou "bas" par rapport au futur.

2. **Purge Window** too short ✅ **CONFIRMÉ MATHÉMATIQUEMENT**
   - **Location**: `splitter.py:28`
   - **Fix**: Increase to 720h (max indicator window)
   - **Impact**: Rolling features leak future info
   - **Validation**: Si Z-Score sur 720h, les données entre T-720 et T-1 sont partagées. Début du validation set pollué par fin du training set.

#### Moderate (P1) - Validés avec nuances

3. **FFD find_min_d** uses full segment for ADF test
   - **Location**: `features.py:200`
   - **Fix**: Use expanding window ADF on train portion
   - **Impact**: Minor (ADF is statistical test, not prediction)

4. **HMM fit** on full segment ⚠️ **NUANCÉ**
   - **Location**: `manager.py:529`
   - **Fix**: Already handled in WFO (refit per segment)
   - **Impact**: Mitigated in production
   - **Validation**: "Moins pire" en contexte WFO. Pour être puriste (Lopez de Prado), utiliser Expanding Window uniquement sur passé disponible.

#### Nouveaux Points (Ajoutés par contre-audit) 🆕

5. **Embargo manquant** (différent de Purge)
   - **Concept**: Après un trade de test, éliminer quelques échantillons *après* le test set avant de reprendre le train suivant
   - **Raison**: Laisser "retomber" les corrélations temporelles des labels (surtout si Triple Barrier Method)
   - **Fix**: Ajouter paramètre `embargo_window` en plus de `purge_window`

6. **Stationnarité du Scaler en production**
   - **Problème**: Même en fitant sur Train uniquement, si régime de volatilité change (2017→2018), scaler explose (valeurs >10 ou <-10)
   - **Fix**: Utiliser **Rolling Scaler** ou **Dynamic Z-Score** plutôt que scaler global statique par fold
   - **Impact**: Robustesse en production

---

## Action Plan (Validé par contre-audit)

### ⛔ P0 - BLOQUANTS (Ne lancer AUCUN entraînement avant correction)

> **VERDICT CONTRE-AUDIT**: Ces deux points rendent tout résultat actuel invalide.

- [ ] **Fix scaler leakage** in `DataManager.pipeline()` and `run_full_wfo.py`
  ```python
  # Fit on train only
  scaler.fit(train_df[cols_to_scale])
  train_df[cols_to_scale] = scaler.transform(train_df[cols_to_scale])
  val_df[cols_to_scale] = scaler.transform(val_df[cols_to_scale])
  test_df[cols_to_scale] = scaler.transform(test_df[cols_to_scale])
  ```

- [ ] **Increase purge_window** to 720h in all split logic
  ```python
  purge_window = 720  # = max(zscore_window, ffd_window, hmm_smoothing)
  ```

### P1 - High Priority (v1.1)

- [ ] **Replace synthetic funding** with real Binance funding rates
  - Use `ccxt` library to fetch `BTC/USDT:USDT` funding history
  - **Alternative validée**: Supprimer la feature plutôt qu'en inventer une fausse

- [ ] **Ajouter Embargo** (🆕 contre-audit)
  ```python
  # Après le test set, ajouter un gap avant de reprendre le train suivant
  embargo_window = 24  # heures, pour laisser retomber corrélations labels
  ```

- [ ] **Rolling Scaler pour production** (🆕 contre-audit)
  ```python
  # Utiliser Dynamic Z-Score ou Rolling Scaler plutôt que scaler statique
  # Plus robuste aux changements de régime de volatilité
  ```

- [ ] **HMM Expanding Window** (nuancé par contre-audit)
  ```python
  # Optionnel si WFO utilisé, mais recommandé pour pureté Lopez de Prado
  # HMM entraîné uniquement sur passé disponible (Expanding Window)
  ```

- [ ] **Add retry logic** in `loader.py:_download_asset()`
  ```python
  from tenacity import retry, stop_after_attempt, wait_exponential
  @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
  def _download_asset(self, ticker, days=730):
      ...
  ```

- [ ] **Fix HMM reproducibility**
  ```python
  # Use fixed seeds, log all random states
  random_state = config.GLOBAL_SEED + segment_id * 1000
  ```

### P2 - Medium Priority

- [ ] **Use Rogers-Satchell** instead of Garman-Klass for volatility (✅ validé)
- [ ] **Add HMM_* to exclude_from_scaling** list
- [ ] **Validate scaler columns** match DataFrame columns before transform
- [ ] **Add minimum rows check** after download (e.g., require > 1000 rows)

### P3 - Optimisations (validées par contre-audit)

- [ ] **Vectoriser FFD** avec FFT ou numba (gain 10x-100x)
- [ ] **Revoir min_d_floor=0.30** - laisser ADF test décider ou imposer test de corrélation

---

## Missing Tests

### Unit Tests
- [ ] `test_ffd_weights_formula.py` - Verify weights match Lopez de Prado
- [ ] `test_ffd_stationarity.py` - Verify ADF passes after FFD
- [ ] `test_parkinson_edge_cases.py` - H=L (doji) case
- [ ] `test_gk_negative.py` - Verify NaN handling when GK negative
- [ ] `test_hmm_archetype_alignment.py` - Verify Hungarian mapping stable
- [ ] `test_scaler_no_leakage.py` - **CRITICAL** Verify scaler fit on train only
- [ ] `test_purge_window_sufficient.py` - Verify purge >= max(indicator_windows)

### Integration Tests
- [ ] `test_pipeline_end_to_end.py` - Full pipeline smoke test
- [ ] `test_wfo_data_separation.py` - No data leakage between folds
- [ ] `test_hmm_semantic_consistency.py` - Prob_0 always "crash-like"
- [ ] `test_reproducibility.py` - Same seed → same output

---

## Monitoring Suggestions

### Metrics to Log
| Metric | Location | Alert Threshold |
|--------|----------|-----------------|
| Feature NaN rate | After features | > 1% |
| FFD d_optimal | Per asset | Outside [0.2, 0.8] |
| HMM n_active_states | After fit | < 3 |
| HMM separation_score | After fit | < 0.0005 |
| Scaler median/IQR | After fit | Drift > 20% from baseline |
| Regime distribution | After predict | Any regime < 5% |

### Alerts
- [ ] Alert if `n_active_states < 3`
- [ ] Alert if scaler columns mismatch DataFrame
- [ ] Alert if download returns < 1000 rows
- [ ] Alert if FFD returns d=1.0 (full differentiation)
- [ ] Alert if HMM doesn't converge

---

## Appendix: Code Quality Scores

| File | Lines | Complexity | Test Coverage | Score |
|------|-------|------------|---------------|-------|
| loader.py | 405 | Medium | Low | 7/10 |
| features.py | 648 | High | Low | 7/10 |
| manager.py | 1053 | High | Medium | 7/10 |
| splitter.py | 89 | Low | Low | 8/10 |
| constants.py | 54 | Low | N/A | 9/10 |

---

---

## Conclusion Contre-Audit

### Verdict Final

**C'est un "Go" pour appliquer les correctifs.**

L'audit initial est de qualité excellente avec 95% d'accord sur les conclusions. Les failles critiques identifiées (P0) sont des "deal-breakers" pour un système de trading algorithmique.

### Checklist Avant Entraînement

| Action | Statut | Impact |
|--------|--------|--------|
| Corriger `RobustScaler` (Fit on Train ONLY) | ⬜ À FAIRE | Invalide tous résultats |
| Corriger `Purge Window` (Doit être > 720h) | ⬜ À FAIRE | Invalide tous résultats |
| Remplacer/Supprimer Funding Rate synthétique | ⬜ v1.1 | Amélioration qualité |
| Ajouter Embargo | ⬜ v1.1 | Amélioration qualité |
| Rolling Scaler pour production | ⬜ v1.1 | Robustesse production |

### Timeline Recommandée

1. **Immédiat**: Corrections P0 (scaler + purge) - sans ces corrections, tout résultat est invalide
2. **v1.1**: Funding rate, optimisations FFD, Embargo, Rolling Scaler
3. **v1.2**: Rogers-Satchell, HMM expanding window

---

*Généré par Audit Pipeline Data - Recursive Prompt Architecture*
*Date: 2026-01-22*
*Contre-Audit: 2026-01-22 - Senior Staff Engineer Review ✅*

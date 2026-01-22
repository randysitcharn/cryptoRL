# Data Pipeline Design Document

**Version**: 2.0
**Date**: 2026-01-22
**Status**: Production Ready (Post-Audit)

---

## Executive Summary

Le Data Pipeline CryptoRL transforme des données OHLCV multi-actifs brutes en features stationnaires prêtes pour l'entraînement d'agents RL. Le pipeline intègre des mesures anti-leakage conformes aux recommandations de Lopez de Prado (AFML).

### Caractéristiques Clés

- **Multi-Asset**: BTC, ETH, SPX, DXY, NASDAQ (5 actifs)
- **Leak-Free**: Purge window 720h, scaler fit on train only
- **Regime-Aware**: HMM 4 états avec alignement sémantique
- **Stationnaire**: Fractional Differentiation adaptative

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA PIPELINE ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   SOURCES   │───>│   LOADER    │───>│  FEATURES   │───>│    HMM      │  │
│  │             │    │             │    │             │    │             │  │
│  │ - Yahoo     │    │ Download    │    │ Log-Returns │    │ Regime      │  │
│  │ - Polygon   │    │ Sync        │    │ Volatility  │    │ Detection   │  │
│  │ - Binance   │    │ Validate    │    │ FFD         │    │ Alignment   │  │
│  │ - CSV       │    │             │    │ Z-Score     │    │             │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│                                                                    │        │
│                                                                    v        │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   OUTPUT    │<───│   SCALER    │<───│   CLEANUP   │<───│  Prob_0-3   │  │
│  │             │    │             │    │             │    │             │  │
│  │ Parquet     │    │ RobustScale │    │ Drop NaN    │    │ Smart Sort  │  │
│  │ Scaler.pkl  │    │ Train Only  │    │ Validate    │    │             │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Components

### 1. MultiAssetDownloader (`loader.py`)

Télécharge et synchronise les données multi-actifs depuis Yahoo Finance.

```python
class MultiAssetDownloader:
    """
    Crypto (Master Timeframe): BTC-USD, ETH-USD (24/7)
    Macro (Slave Timeframe): SPX, DXY, NASDAQ (5/7)
    """
```

#### Responsabilités

| Fonction | Description |
|----------|-------------|
| `_download_asset()` | Télécharge OHLCV via yfinance avec retry (tenacity) |
| `_validate_raw_data()` | Valide prix > 0, volumes >= 0, gaps < 24h/72h |
| `_synchronize_dataframes()` | Aligne sur index BTC-USD via forward-fill |
| `download_multi_asset()` | Pipeline complet: download → sync → validate |

#### Data Sources Priority

```
1. raw_historical/multi_asset_historical.csv  [8 années, Polygon/Binance]
2. HistoricalDownloader                       [API keys required]
3. Yahoo Finance                              [730 jours max, fallback]
```

#### Synchronisation Strategy

```
BTC-USD (Master Index, 24/7)
    │
    ├── ETH-USD      → reindex + forward-fill
    ├── SPX (^GSPC)  → reindex + forward-fill (weekends)
    ├── DXY          → reindex + forward-fill (weekends)
    └── NASDAQ       → reindex + forward-fill (weekends)
```

Les actifs Macro sont forward-filled car les marchés traditionnels ferment le week-end et la nuit. L'index BTC-USD (24/7) sert de référence temporelle.

---

### 2. FeatureEngineer (`features.py`)

Transforme les prix bruts en features stationnaires pour le RL.

#### Pipeline de Features

```
┌──────────────────────────────────────────────────────────────────┐
│                    FEATURE ENGINEERING PIPELINE                   │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  [1] SANITIZE           [2] LOG-RETURNS        [3] VOLUME        │
│  ────────────           ─────────────          ────────          │
│  Price=0 → NaN          r = ln(Ct/Ct-1)        Vol_LogRet        │
│  ffill → bfill          clip(±20%)             Vol_ZScore(336h)  │
│                                                                  │
│  [4] VOLATILITY         [5] Z-SCORE            [6] FFD           │
│  ──────────────         ───────────            ─────             │
│  Parkinson(24h)         Z = (P-μ)/σ            d via ADF test    │
│  Garman-Klass(24h)      window=720h            window=100        │
│                                                                  │
│  [7] VALIDATE                                                    │
│  ────────────                                                    │
│  Check |x| > 10                                                  │
│  Warning if extreme                                              │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

#### Features Générées

| Feature | Formule | Window | Assets |
|---------|---------|--------|--------|
| `{ASSET}_LogRet` | ln(C_t / C_{t-1}) | 1 | All |
| `{ASSET}_Parkinson` | √(1/(4ln2) × ln²(H/L)) | 24h | All |
| `{ASSET}_GK` | √(0.5×ln²(H/L) - 0.386×ln²(C/O)) | 24h | All |
| `{ASSET}_ZScore` | (P - μ) / σ | 720h | All |
| `{ASSET}_Fracdiff` | FFD(log(P), d*) | 100 | All |
| `{ASSET}_Vol_LogRet` | ln(V_t / V_{t-1}) | 1 | All |
| `{ASSET}_Vol_ZScore` | (V - μ) / σ | 336h | All |

#### Fractional Differentiation (FFD)

Implémentation conforme à Lopez de Prado (2018) pour obtenir la stationnarité tout en préservant la mémoire.

```python
# Recherche adaptative du d optimal
d* = min{d ∈ [0.30, 1.0] | ADF(FFD(series, d)) < 0.05}

# Formule FFD (Fixed-Width Window)
y_t = Σ(k=0 to window) w_k × x_{t-k}

# Poids FFD
w_k = w_{k-1} × (d - k + 1) / k
```

| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| `d_range` | [0.0, 1.0, 0.05] | Recherche par pas de 0.05 |
| `min_d_floor` | 0.30 | Évite faux positifs ADF |
| `window` | 100 | ~4 jours, compromis mémoire/stationnarité |
| `threshold` | 1e-5 | Troncature des poids |

---

### 3. RegimeDetector (`manager.py`)

Détecte les régimes de marché via GMM-HMM pour conditionner la policy RL.

#### Architecture HMM

```
┌─────────────────────────────────────────────────────────────┐
│                      HMM REGIME DETECTION                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  HMM Features (5)                  HMM States (4)           │
│  ─────────────────                 ──────────────           │
│  • HMM_Trend (168h)                State 0: Crash           │
│  • HMM_Vol (168h)                  State 1: Downtrend       │
│  • HMM_Momentum (RSI14)            State 2: Range           │
│  • HMM_RiskOnOff (SPX-DXY)         State 3: Uptrend         │
│  • HMM_VolRatio (24h/168h)                                  │
│                                                             │
│  Training Flow:                                             │
│  K-Means → GMM-HMM → Sticky → Archetype Align → Prob_*     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### HMM Features

| Feature | Calcul | Rôle |
|---------|--------|------|
| `HMM_Trend` | MA(LogRet, 168h) | Direction du marché |
| `HMM_Vol` | MA(Parkinson, 168h) | Niveau de volatilité |
| `HMM_Momentum` | RSI(14) / 100 | Surachat/survente |
| `HMM_RiskOnOff` | MA(SPX_ret - DXY_ret, 168h) | Appétit pour le risque |
| `HMM_VolRatio` | Vol(24h) / Vol(168h) | Accélération de volatilité |

> **Note**: `HMM_Funding` a été retiré (données synthétiques causant des corrélations fallacieuses).

#### State Archetypes

Les archétypes sont fixés pour permettre une interprétation sémantique stable entre segments WFO.

| State | Name | Mean Return | Mean Vol | Description |
|-------|------|-------------|----------|-------------|
| 0 | Crash | -0.50%/h | 4.0%/h | Panique, flash crash |
| 1 | Downtrend | -0.10%/h | 1.5%/h | Bear market |
| 2 | Range | 0.00%/h | 0.5%/h | Consolidation |
| 3 | Uptrend | +0.15%/h | 2.0%/h | Bull market |

#### Archetype Alignment (Hungarian Algorithm)

Résout le problème de "Semantic Drift" où le même état HMM peut représenter des régimes différents entre segments.

```python
# Distance euclidienne pondérée
dist(state_i, archetype_j) = √(w_ret × (ret_i - ret_j)² + w_vol × (vol_i - vol_j)²)

# Poids
w_ret = 1.0
w_vol = 2.0  # Volatilité plus discriminante

# Mapping optimal via scipy.optimize.linear_sum_assignment
mapping = hungarian(cost_matrix)
```

#### GMM-HMM Parameters

| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| `n_components` | 4 | 4 régimes de marché (domain knowledge) |
| `n_mix` | 2 | Flexibilité sans overfitting |
| `covariance_type` | 'diag' | Stabilité numérique |
| `n_iter` | 200 | Convergence garantie |
| `min_covar` | 1e-3 | Régularisation |
| `transition_penalty` | 0.1 | Sticky HMM (10% boost diagonal) |

---

### 4. DataManager (`manager.py`)

Orchestre le pipeline complet de préparation des données.

#### Pipeline Steps

```
STEP 1: LOAD DATA
    │   Priority: CSV > HistoricalDownloader > Yahoo
    │
    v
STEP 2: FEATURE ENGINEERING
    │   Log-returns, Volatility, Z-Score, FFD
    │
    v
STEP 3: REGIME DETECTION
    │   HMM fit → Archetype Alignment → Prob_0/1/2/3
    │
    v
STEP 4: CLEANUP
    │   Drop NaN rows
    │
    v
STEP 5: SCALING (Leak-Free)
    │   RobustScaler fit on train only
    │
    v
STEP 6: EXPORT
        Parquet + Scaler pickle
```

#### Leak-Free Scaling

```python
def pipeline(self, ..., train_end_idx: Optional[int] = None):
    """
    Args:
        train_end_idx: Index de fin du train set pour fit scaler on train only.
                      Si None, utilise fit_transform sur tout le dataset (legacy).
    """
    if train_end_idx is not None:
        # CORRECT: Fit on train only
        self.scaler.fit(df.iloc[:train_end_idx][cols_to_scale])
        df[cols_to_scale] = self.scaler.transform(df[cols_to_scale])
    else:
        # LEGACY: Warning issued
        warnings.warn("Scaler fit on full dataset - data leakage risk!")
        df[cols_to_scale] = self.scaler.fit_transform(df[cols_to_scale])
```

#### Columns Excluded from Scaling

```python
exclude_from_scaling = [
    # Raw OHLC prices
    'BTC_Close', 'ETH_Close', 'SPX_Close', 'DXY_Close', 'NASDAQ_Close',
    # Raw volumes
    'BTC_Volume', 'ETH_Volume', ...
    # Log-returns (already clipped ±20%)
    'BTC_LogRet', 'ETH_LogRet', ...
    # HMM probabilities (already [0, 1])
    'Prob_0', 'Prob_1', 'Prob_2', 'Prob_3',
]
```

---

### 5. TimeSeriesSplitter (`splitter.py`)

Divise les données chronologiquement avec purge et embargo.

#### Split Strategy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TIME SERIES SPLIT WITH PURGE                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  [========= TRAIN (70%) =========][PURGE][=== VAL (15%) ===][PURGE][TEST]  │
│                                   720h                      720h   (15%)   │
│                                                                             │
│  Purge Window = 720h (max indicator window = Z-Score)                       │
│  Embargo Window = 24h (for WFO, after test before next train)               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Parameters

| Paramètre | Default | Justification |
|-----------|---------|---------------|
| `train_ratio` | 0.70 | 70% pour training |
| `val_ratio` | 0.15 | 15% pour validation |
| `purge_window` | 720 | >= max(indicator_windows) |
| `embargo_window` | 24 | Décorrélation des labels |

---

## Window Sizes & Constants

### Feature Windows

| Constante | Valeur | Usage |
|-----------|--------|-------|
| `ZSCORE_WINDOW` | 720h (30j) | Rolling Z-Score prix |
| `VOL_ZSCORE_WINDOW` | 336h (14j) | Rolling Z-Score volume |
| `HMM_SMOOTHING_WINDOW` | 168h (7j) | Lissage features HMM |
| `FFD_WINDOW` | 100h (~4j) | Fractional Differentiation |
| `VOL_WINDOW` | 24h (1j) | Parkinson/GK volatility |

### Purge & Embargo

| Constante | Valeur | Usage |
|-----------|--------|-------|
| `MAX_INDICATOR_WINDOW` | 720h | Max de toutes les fenêtres |
| `DEFAULT_PURGE_WINDOW` | 720h | Gap train→val, val→test |
| `DEFAULT_EMBARGO_WINDOW` | 24h | Gap après test (WFO) |

---

## Data Leakage Prevention

### Sources de Leakage Potentielles

| Source | Risque | Mitigation |
|--------|--------|------------|
| **Scaler** | Médiane/IQR incluent futur | Fit on train only (`train_end_idx`) |
| **Purge** | Indicateurs rolling fuient | Purge >= 720h (max window) |
| **HMM** | Fit sur tout le segment | Refit par segment en WFO |
| **FFD** | d optimal sur tout | ADF test est statistique, impact mineur |

### Checklist Anti-Leakage

- [x] `RobustScaler.fit()` sur train uniquement
- [x] `purge_window >= MAX_INDICATOR_WINDOW` (720h)
- [x] `embargo_window` après test set (24h)
- [x] Fixed-width windows (pas de look-ahead)
- [x] Forward-fill uniquement (pas de back-fill)
- [x] Warning si mode legacy utilisé

---

## Output Format

### Parquet Schema

```
data/processed_data.parquet
├── Index: DatetimeIndex (hourly, UTC)
├── Raw Columns (excluded from features):
│   ├── {ASSET}_Close, {ASSET}_Open, {ASSET}_High, {ASSET}_Low
│   └── {ASSET}_Volume
├── Engineered Features (scaled):
│   ├── {ASSET}_Fracdiff
│   ├── {ASSET}_Parkinson
│   ├── {ASSET}_GK
│   ├── {ASSET}_ZScore
│   ├── {ASSET}_Vol_LogRet
│   └── {ASSET}_Vol_ZScore
├── Returns (clipped, not scaled):
│   └── {ASSET}_LogRet
└── Regime Probabilities (not scaled):
    ├── Prob_0 (Crash)
    ├── Prob_1 (Downtrend)
    ├── Prob_2 (Range)
    └── Prob_3 (Uptrend)
```

### Scaler Pickle

```python
# data/scaler.pkl
{
    'scaler': RobustScaler(),  # Fitted on train only
    'columns': ['BTC_Fracdiff', 'BTC_Parkinson', ...]  # Scaled columns
}
```

---

## Integration with WFO

Le pipeline s'intègre avec Walk-Forward Optimization via `run_full_wfo.py`.

### WFO Segment Structure

```
Segment N:
┌──────────────┬───────┬──────────────┬───────┬───────────┬─────────┐
│    TRAIN     │ PURGE │     EVAL     │ PURGE │    TEST   │ EMBARGO │
│   (13 mo)    │ 720h  │    (1 mo)    │ 720h  │   (3 mo)  │   24h   │
└──────────────┴───────┴──────────────┴───────┴───────────┴─────────┘
```

### Per-Segment Processing

```python
# In run_full_wfo.py:
scaler = RobustScaler()
scaler.fit(train_df[cols_to_scale])  # Train only!
train_df[cols_to_scale] = scaler.transform(train_df[cols_to_scale])
eval_df[cols_to_scale] = scaler.transform(eval_df[cols_to_scale])
test_df[cols_to_scale] = scaler.transform(test_df[cols_to_scale])
```

---

## Error Handling

### Validation Checks

| Check | Location | Action |
|-------|----------|--------|
| Prix <= 0 | `_validate_raw_data` | Replace with NaN, forward-fill |
| Volume < 0 | `_validate_raw_data` | Clip to 0 |
| Gaps > 24h (crypto) | `_validate_raw_data` | Warning |
| Gaps > 72h (macro) | `_validate_raw_data` | Warning |
| Features |x| > 10 | `_validate_features` | Warning |
| HMM < 3 active states | `_validate_hmm` | Retry (max 3) |
| Download failure | `_download_asset` | Retry with exponential backoff |

### Retry Logic

```python
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True
)
def _download_with_retry(self, ticker, start_date):
    return yf.download(...)
```

---

## Performance Considerations

### Optimizations Actuelles

- K-Means warm start pour HMM (convergence rapide)
- Forward-fill vectorisé (pandas)
- Parquet format (compression, I/O rapide)

### Optimisations Futures (P3)

- [ ] FFD via FFT ou numba (gain 10x-100x)
- [ ] Cache d_optimal par asset
- [ ] Parallel feature computation

---

## References

1. **Lopez de Prado, M. (2018)**. *Advances in Financial Machine Learning*. Wiley.
   - Fractional Differentiation (Chapter 5)
   - Purge/Embargo (Chapter 7)

2. **Hamilton, J.D. (1989)**. *A New Approach to the Economic Analysis of Nonstationary Time Series*.
   - HMM for regime detection

3. **Rabiner, L.R. (1989)**. *A Tutorial on Hidden Markov Models*.
   - HMM fundamentals

4. **Audit Report**: `docs/audit/DATA_PIPELINE_AUDIT_REPORT.md`
   - P0: Scaler leakage fix
   - P0: Purge window 720h
   - P1: Synthetic funding removal

---

*Document généré le 2026-01-22*
*Version 2.0 - Post-Audit*

**Statut Implémentation** : ✅ COMPLET

**Fichiers implémentés** :
- `src/data_engineering/loader.py` : MultiAssetDownloader ✅
- `src/data_engineering/features.py` : FeatureEngineer ✅
- `src/data_engineering/manager.py` : RegimeDetector + DataManager ✅
- `src/data_engineering/splitter.py` : TimeSeriesSplitter ✅

**Fonctionnalités implémentées** :
- ✅ Download multi-actifs avec synchronisation
- ✅ Feature engineering (Log-Returns, Volatility, Z-Score, FFD)
- ✅ HMM Regime Detection avec archetype alignment
- ✅ Leak-free scaling (fit on train only)
- ✅ Purge window 720h dans splitter
- ✅ Support WFO avec isolation temporelle

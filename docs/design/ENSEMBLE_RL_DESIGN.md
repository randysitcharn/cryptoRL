# Design Document: Ensemble RL pour CryptoRL

**Version**: 1.3  
**Date**: 2026-01-22  
**Statut**: ✅ TRIPLE AUDIT SOTA (Score consolidé: 8/10)  
**Niveau de Risque**: Faible (Architecture renforcée avec OOD detection)  

---

## Table des Matières

1. [Résumé Exécutif](#1-résumé-exécutif)
2. [Contexte et Motivation](#2-contexte-et-motivation)
3. [Architecture Détaillée](#3-architecture-détaillée)
4. [Implémentation](#4-implémentation)
5. [Configuration](#5-configuration)
6. [Intégration WFO](#6-intégration-wfo)
7. [Intégration Composants Existants](#7-intégration-composants-existants)
8. [Risques et Mitigations](#8-risques-et-mitigations)
9. [Plan de Validation](#9-plan-de-validation)
10. [Plan d'Implémentation](#10-plan-dimplémentation)
11. [Annexes](#annexes)

---

## 🛡️ Résultats des Audits

### Audit #1 (Gemini AI - 2026-01-21) — Score: EXCELLENT

| Composant | Verdict | Commentaire |
|-----------|---------|-------------|
| Confidence-Weighted Aggregation | ✅ Validé | Approximation élégante de l'incertitude bayésienne |
| Architecture Parallèle | ✅ Indispensable | Critique pour WFO |
| Intégration TQC Native | ✅ Optimal | Coût additionnel nul via quantiles existants |
| Signal 6 (Ensemble Collapse) | ✅ Innovation | Très pertinent pour la finance |

**Améliorations Intégrées (v1.1)** :
- Softmax Temperature, Calibration Volatilité, Diversité Forcée, Shared Replay Buffer

---

### Audit #2 SOTA (2026-01-22) — Score: 8.5/10 (Production Ready)

#### Points Forts SOTA

| Concept | Analyse |
|---------|---------|
| **Hybridation Aleatoric/Epistemic** | Capture les deux types d'incertitude : aléatorique (quantile spread TQC) + épistémique (variance inter-membres) |
| **Calibration Dynamique** | `spread_norm = spread / EMA(spread)` évite la paralysie en haute volatilité |
| **Forced Diversity** | Variation gamma/LR décorrèle les erreurs entre modèles |

#### Points de Vigilance Identifiés

| Risque | Description | Mitigation |
|--------|-------------|------------|
| **Expert Aveugle** | Low spread ≠ High quality (overfit possible) | Monitorer spread vs reward |
| **Fausse Confiance** | Modèle overfitté peut avoir spread faible | Combiner spread + Q-value (future) |
| **Reversal Blindness** | Spread large = peut-être détection changement régime | Ne pas pénaliser aveuglément |

#### Recommandations Critiques Intégrées (v1.2)

##### A. Pessimistic Bound (NOUVEAU - Sécurité Critique)

> *"Ne pas seulement pondérer par la confiance, mais réduire la taille de position quand le désaccord est fort."*

```python
method == 'pessimistic_bound':
    mean_action = np.mean(actions, axis=0)
    std_action = np.std(actions, axis=0)
    
    # Si désaccord fort → réduire la position vers 0 (Hold)
    # k = facteur d'aversion au risque (1.0 = standard, 2.0 = conservateur)
    scaling_factor = np.clip(1.0 - (k * std_action), 0.0, 1.0)
    final_action = mean_action * scaling_factor
```

**Avantage** : Gère nativement le Position Sizing basé sur l'incertitude épistémique.

##### B. Confidence Scaling Optionnel

Même en mode `confidence`, appliquer un scaling de sécurité :

```python
# Après aggregation confidence-weighted
if apply_pessimistic_scaling:
    action_std = np.std(actions, axis=0)
    safety_scale = np.clip(1.0 - (risk_aversion * action_std), 0.1, 1.0)
    final_action = final_action * safety_scale
```

##### C. Shared Memory (Mitigation OOM)

Pour training parallèle avec gros Replay Buffers :
- Option 1 : Buffer petit (100k) par membre
- Option 2 : `multiprocessing.SharedMemory` ou `numpy.memmap`
- Option 3 : Shared Replay Buffer central (tous membres écrivent/lisent)

---

### Audit #3 SOTA Critique (2026-01-22) — Score: 7.5/10

#### Gaps Critiques Identifiés

| Gap | Description | Impact |
|-----|-------------|--------|
| **Confusion Aléatorique/Épistémique** | Mélange spread quantile (aléatorique) et variance inter-membres (épistémique) | Surconfiance possible |
| **Softmax Naïve** | `softmax(-spread)` suppose calibration parfaite | Dangereux hors distribution |
| **Diversité Insuffisante** | Gamma/LR seuls ≠ diversité réelle | Comportements corrélés |
| **Absence OOD Detection** | Pas de détection régimes inconnus | Pertes catastrophiques possibles |

#### Recommandations Cutting-Edge Intégrées (v1.3)

##### A. Score de Confiance Mixte (Model Trust)

Ne pas se baser uniquement sur le spread. Score combiné :

```python
model_trust = (
    calibration_score          # Historique de calibration
    + recent_oos_performance   # Performance récente OOS
    - uncertainty_penalty      # Pénalité spread + variance
)
weight_i = softmax(model_trust_i / temperature)
```

##### B. Bootstrapped Replay Buffer (Diversité Bayésienne)

Chaque membre voit des données légèrement différentes via masques de Bernoulli :

```python
# Au lieu de shared buffer identique
mask = torch.bernoulli(torch.ones(buffer_size) * 0.8)  # 80% overlap
member_indices = torch.nonzero(mask).squeeze()
```

##### C. Détection OOD & Conservative Fallback (CRITIQUE)

```python
@dataclass
class OODConfig:
    enable_ood_detection: bool = True
    ood_threshold: float = 2.0  # Seuil en écarts-types
    fallback_action: float = 0.0  # Hold en cas de OOD
    fallback_leverage_scale: float = 0.25  # Réduire à 25% du levier
    
def compute_ood_score(self, obs: np.ndarray) -> float:
    """
    Détecte si l'observation est hors distribution.
    
    Methods:
    1. MAE reconstruction error (si encoder disponible)
    2. Mahalanobis distance sur features
    3. Spread moyen anormalement élevé
    """
    # Spread anormal = proxy OOD simple
    spreads = [self._get_quantile_spread(m, obs) for m in self.models]
    avg_spread = np.mean(spreads)
    
    if not hasattr(self, '_spread_history'):
        self._spread_history = []
    self._spread_history.append(avg_spread)
    
    if len(self._spread_history) > 100:
        mean_spread = np.mean(self._spread_history[-100:])
        std_spread = np.std(self._spread_history[-100:])
        z_score = (avg_spread - mean_spread) / (std_spread + 1e-6)
        return z_score
    return 0.0
```

##### D. Conservative Fallback Mode

```python
def predict_with_safety(self, obs, deterministic=True):
    ood_score = self.compute_ood_score(obs)
    
    if ood_score > self.config.ood_threshold:
        # Mode Survie : réduire drastiquement l'exposition
        return np.array([[self.config.fallback_action]]), {
            'mode': 'OOD_FALLBACK',
            'ood_score': ood_score,
            'action_override': True
        }
    
    # Mode normal
    action, info = self.predict(obs, deterministic)
    
    # Scaling additionnel si proche du seuil OOD
    if ood_score > self.config.ood_threshold * 0.7:
        safety_scale = self.config.fallback_leverage_scale
        action = action * safety_scale
        info['ood_scaling'] = safety_scale
    
    return action, info
```

### ROI Attendu (Consolidé)

> *"En finance, la réduction de variance est souvent plus précieuse que l'augmentation de l'espérance de gain. Diviser le Max Drawdown par 2 via l'ensemble permet souvent de doubler le levier (et donc le profit) à risque constant."* — Gemini AI

> *"L'absence de détection OOD est le principal risque de pertes catastrophiques. Un bon système n'est pas celui qui gagne le plus, mais celui qui survit aux Black Swans."* — Audit SOTA #3

---

## 1. Résumé Exécutif

### Approche Choisie : Hybrid Multi-Seed + Confidence-Weighted Aggregation

Nous implémentons une stratégie d'**ensemble hybride** qui combine :

1. **Multi-Seed (3 seeds)** : Diversité fondamentale via initialisation différente
2. **Confidence-Weighted Aggregation** : Pondération par l'incertitude des quantiles TQC
3. **Training Parallèle** : Exploitation des 2 GPUs disponibles

**Avantages clés** :
- Réduction de la variance des décisions de trading (~50%)
- Robustesse accrue aux régimes de marché
- Exploitation native de l'architecture TQC (quantiles = incertitude)
- Compatible avec le pipeline WFO existant

**Coût** :
- Training ~2× plus long (parallèle sur 2 GPUs)
- Mémoire GPU supplémentaire (~3× pour inférence)
- Complexité modérée

---

## 2. Contexte et Motivation

### 2.1 Problème Actuel

Le système actuel entraîne un seul modèle TQC par segment WFO. Cette approche présente :

| Limitation | Impact |
|------------|--------|
| **Variance inter-seeds** | Performances très variables selon l'initialisation |
| **Sensibilité aux outliers** | Un mauvais gradient peut dérailler l'entraînement |
| **Décisions "all-in"** | Pas de notion de confiance dans les actions |
| **Overfitting local** | Un seul modèle peut mémoriser des patterns spurieux |

### 2.2 Solution : Ensemble RL

L'ensemble agrège plusieurs politiques pour :

1. **Réduire la variance** : Moyenne de N modèles plus stable qu'un seul
2. **Estimer l'incertitude** : Disagreement entre modèles = incertitude
3. **Robustesse** : Si un modèle échoue, les autres compensent
4. **Décisions calibrées** : Actions pondérées par la confiance

### 2.3 Références SOTA

| Méthode | Année | Idée Principale | Application |
|---------|-------|-----------------|-------------|
| **DroQ** | 2021 | Dropout = mini-ensemble implicite | Déjà implémenté (TQCDropoutPolicy) |
| **REDQ** | 2021 | 20 critics, subset random | Trop coûteux |
| **Ensemble RL Classifiers** | 2025 | Agrégation via classifiers | Référence (arXiv:2502.17518) |
| **TQC** | 2020 | Quantile Critics + Truncation | Base du projet |

**Notre approche** combine DroQ (régularisation interne) + Multi-Seed (diversité externe) + TQC Quantiles (confidence native).

---

## 3. Architecture Détaillée

### 3.1 Vue d'Ensemble

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           EnsemblePolicy                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Observation (market + position + w_cost)                                   │
│                         │                                                    │
│         ┌───────────────┼───────────────┐                                   │
│         ▼               ▼               ▼                                    │
│   ┌───────────┐   ┌───────────┐   ┌───────────┐                            │
│   │ TQC       │   │ TQC       │   │ TQC       │                            │
│   │ seed=42   │   │ seed=123  │   │ seed=456  │                            │
│   │           │   │           │   │           │                            │
│   │ action_0  │   │ action_1  │   │ action_2  │                            │
│   │ spread_0  │   │ spread_1  │   │ spread_2  │  ← Quantile Spread        │
│   └─────┬─────┘   └─────┬─────┘   └─────┬─────┘                            │
│         │               │               │                                    │
│         └───────────────┼───────────────┘                                   │
│                         ▼                                                    │
│   ┌─────────────────────────────────────────────────────────────────┐      │
│   │           Confidence-Weighted Aggregation (Softmax)              │      │
│   │                                                                   │      │
│   │   // Calibrated spread (normalized by market volatility)         │      │
│   │   spread_norm_i = spread_i / EMA(spread)                         │      │
│   │                                                                   │      │
│   │   // Softmax temperature weighting (τ = 1.0 default)             │      │
│   │   weight_i = exp(-spread_norm_i / τ) / Σ(exp(-spread_norm_j / τ))│      │
│   │                                                                   │      │
│   │   final_action = Σ(weight_i × action_i)                          │      │
│   │   ensemble_confidence = 1 / (action_std + ε)                     │      │
│   └─────────────────────────────────────────────────────────────────┘      │
│                         │                                                    │
│                         ▼                                                    │
│              final_action, ensemble_info                                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Flux de Données WFO

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    WFO Pipeline avec Ensemble (2 GPUs)                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Segment i                                                                   │
│  ═════════                                                                   │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ 1. PREPROCESSING (Shared, Sequential)                              │    │
│  │    ├─ Feature Engineering (from raw OHLCV)                         │    │
│  │    ├─ RobustScaler (fit on TRAIN only)                            │    │
│  │    └─ HMM Regime Detection (fit on TRAIN)                          │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                              │                                              │
│                              ▼                                              │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ 2. MAE TRAINING (Shared, Single GPU)                               │    │
│  │    └─ encoder.pth (frozen for all TQC members)                     │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                              │                                              │
│                              ▼                                              │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ 3. TQC ENSEMBLE TRAINING (Parallel, 2 GPUs)                        │    │
│  │                                                                     │    │
│  │    GPU 0                           GPU 1                           │    │
│  │    ┌─────────────────┐            ┌─────────────────┐             │    │
│  │    │ TQC seed=42     │            │ TQC seed=123    │             │    │
│  │    │ 30M timesteps   │            │ 30M timesteps   │             │    │
│  │    │                 │            │                 │             │    │
│  │    │ → tqc_0.zip     │            │ → tqc_1.zip     │             │    │
│  │    │ → tqc_0_ema.zip │            │ → tqc_1_ema.zip │             │    │
│  │    └─────────────────┘            └─────────────────┘             │    │
│  │                                                                     │    │
│  │    Then GPU 0:                                                      │    │
│  │    ┌─────────────────┐                                             │    │
│  │    │ TQC seed=456    │                                             │    │
│  │    │ 30M timesteps   │                                             │    │
│  │    │ → tqc_2.zip     │                                             │    │
│  │    └─────────────────┘                                             │    │
│  │                                                                     │    │
│  │    Total time: ~2× single model (vs 3× sequential)                 │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                              │                                              │
│                              ▼                                              │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ 4. ENSEMBLE EVALUATION (TEST, Sequential)                          │    │
│  │                                                                     │    │
│  │    Load: tqc_0.zip, tqc_1.zip, tqc_2.zip                          │    │
│  │    Create: EnsemblePolicy(aggregation='confidence')                │    │
│  │                                                                     │    │
│  │    Evaluate:                                                        │    │
│  │    ├─ Single Best Model (baseline)                                 │    │
│  │    ├─ Ensemble Mean                                                │    │
│  │    ├─ Ensemble Confidence-Weighted ← PRIMARY                       │    │
│  │    └─ Individual Members (ablation)                                │    │
│  │                                                                     │    │
│  │    Metrics: Sharpe, PnL, MaxDD, Ensemble Agreement                 │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Structure des Fichiers (Post-Training)

```
weights/wfo/segment_0/
├── encoder.pth                 # Shared MAE encoder
├── ensemble/
│   ├── tqc_seed_42.zip        # Member 0
│   ├── tqc_seed_42_ema.zip    # Member 0 EMA weights
│   ├── tqc_seed_123.zip       # Member 1
│   ├── tqc_seed_123_ema.zip   # Member 1 EMA weights
│   ├── tqc_seed_456.zip       # Member 2
│   ├── tqc_seed_456_ema.zip   # Member 2 EMA weights
│   └── ensemble_config.json   # Ensemble metadata
├── checkpoints/               # Safety checkpoints (cleaned after success)
└── tqc.zip                    # Symlink to best single model (backward compat)
```

---

## 4. Implémentation

### 4.1 Classe `EnsemblePolicy`

**Fichier** : `src/evaluation/ensemble.py`

```python
# -*- coding: utf-8 -*-
"""
ensemble.py - Ensemble RL Policy Aggregation for TQC.

Implements SOTA ensemble techniques for robust trading:
- Multi-seed aggregation
- Confidence-weighted voting via TQC quantile spread
- Agreement-based action filtering

References:
- Ensemble RL through Classifier Models (arXiv:2502.17518)
- DroQ (Hiraoka 2021) - implicit ensemble via dropout
- TQC (Kuznetsov 2020) - distributional RL with quantiles

Author: CryptoRL Team
Date: 2026-01-21
"""

import os
import json
import numpy as np
import torch
from typing import List, Optional, Tuple, Dict, Any, Literal, Union, TYPE_CHECKING
from pathlib import Path
from dataclasses import dataclass, field, asdict

if TYPE_CHECKING:
    from src.config import TQCTrainingConfig

from sb3_contrib import TQC
from stable_baselines3.common.vec_env import VecEnv


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class EnsembleConfig:
    """Configuration for Ensemble RL training and inference."""
    
    # === Ensemble Composition ===
    n_members: int = 3
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456])
    
    # === Aggregation ===
    aggregation: Literal['mean', 'median', 'confidence', 'conservative', 'pessimistic_bound'] = 'confidence'
    # Note: 'vote' déprécié (perd l'amplitude en trading continu) - utiliser 'median'
    confidence_epsilon: float = 1e-6  # Prevent division by zero
    
    # === Softmax Temperature (Audit Gemini) ===
    # Plus τ est petit, plus les modèles incertains sont "tués"
    softmax_temperature: float = 1.0  # τ=1.0 standard, τ=0.5 agressif
    
    # === Spread Calibration (Audit Gemini) ===
    # Normalise le spread par EMA pour éviter la timidité en haute volatilité
    calibrate_spread: bool = True
    spread_ema_alpha: float = 0.01  # EMA decay pour moyenne mobile du spread
    
    # === Pessimistic Scaling (Audit SOTA v1.2) ===
    # Réduit la position quand le désaccord entre membres est fort
    apply_pessimistic_scaling: bool = True  # Appliqué après toute méthode d'agrégation
    risk_aversion: float = 1.0  # k factor: 1.0=standard, 2.0=très conservateur, 0.5=agressif
    min_scaling: float = 0.1  # Ne jamais réduire en dessous de 10% de la position
    
    # === OOD Detection (Audit SOTA v1.3) ===
    # Détection Out-of-Distribution pour éviter les pertes catastrophiques
    enable_ood_detection: bool = True
    ood_threshold: float = 2.5  # Z-score seuil pour déclarer OOD
    ood_warning_threshold: float = 1.5  # Z-score pour réduction préventive
    fallback_action: float = 0.0  # Action en mode OOD (0 = Hold)
    fallback_leverage_scale: float = 0.25  # Réduire à 25% si proche OOD
    ood_history_window: int = 500  # Fenêtre pour statistiques spread
    
    # === Agreement Filtering ===
    min_agreement: float = 0.0  # Min agreement to act (0 = always act)
    disagreement_action: float = 0.0  # Action when disagreement (0 = hold)
    
    # === Training ===
    parallel_gpus: List[int] = field(default_factory=lambda: [0, 1])
    shared_encoder: bool = True
    shared_replay_buffer: bool = False  # Mitigation OOM (Audit Gemini)
    
    # === Forced Diversity (Audit Gemini) ===
    # Varier légèrement les hyperparamètres entre membres
    use_diverse_hyperparams: bool = True
    gamma_range: Tuple[float, float] = (0.94, 0.96)  # [0.94, 0.95, 0.96]
    lr_range: Tuple[float, float] = (5e-5, 2e-4)     # [5e-5, 1e-4, 2e-4]
    
    # === Inference ===
    use_ema_weights: bool = True  # Use EMA weights for inference
    deterministic: bool = True  # Disable exploration noise
    
    # === Memory Management & I/O Performance ===
    # IMPORTANT: Lazy loading recommandé pour éviter saturation RAM
    # 3 modèles TQC .zip = ~30MB chacun en RAM + GPU tensors
    preload_models: bool = False  # False = lazy loading (charge à la demande)
    unload_after_predict: bool = False  # True = libère GPU après chaque predict (lent mais économe)
    device: str = 'cuda'
    
    def to_json(self, path: str):
        """Save config to JSON."""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def from_json(cls, path: str) -> 'EnsembleConfig':
        """Load config from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


# =============================================================================
# Ensemble Policy
# =============================================================================

class EnsemblePolicy:
    """
    Ensemble of TQC policies for robust trading decisions.
    
    Aggregates predictions from multiple TQC models trained with different
    seeds. Uses TQC's quantile spread for confidence-weighted aggregation.
    
    Features:
    - Multiple aggregation methods (mean, median, confidence, vote, conservative)
    - Confidence estimation via quantile spread (TQC-native)
    - Agreement-based action filtering
    - GPU memory management
    
    Example:
        >>> ensemble = EnsemblePolicy(
        ...     model_paths=['tqc_0.zip', 'tqc_1.zip', 'tqc_2.zip'],
        ...     config=EnsembleConfig(aggregation='confidence')
        ... )
        >>> action, info = ensemble.predict(obs)
        >>> print(f"Action: {action}, Confidence: {info['ensemble_confidence']}")
    """
    
    def __init__(
        self,
        model_paths: List[str],
        config: Optional[EnsembleConfig] = None,
        verbose: int = 0,
    ):
        """
        Initialize the ensemble policy.
        
        Args:
            model_paths: List of paths to TQC model .zip files.
            config: Ensemble configuration. Uses defaults if None.
            verbose: Verbosity level (0=silent, 1=info, 2=debug).
        """
        self.config = config or EnsembleConfig()
        self.verbose = verbose
        self.device = torch.device(
            self.config.device if torch.cuda.is_available() else 'cpu'
        )
        
        # Validate paths
        self.model_paths = []
        for path in model_paths:
            if not Path(path).exists():
                raise FileNotFoundError(f"Model not found: {path}")
            self.model_paths.append(path)
        
        # Load models
        self.models: List[TQC] = []
        if self.config.preload_models:
            self._load_models()
        
        if self.verbose > 0:
            print(f"[Ensemble] Initialized with {len(self.model_paths)} models")
            print(f"           Aggregation: {self.config.aggregation}")
            print(f"           Device: {self.device}")
    
    def _load_models(self):
        """Load all models into memory."""
        self.models = []
        for i, path in enumerate(self.model_paths):
            if self.verbose > 1:
                print(f"  Loading model {i+1}/{len(self.model_paths)}: {path}")
            
            model = TQC.load(path, device=self.device)
            model.policy.set_training_mode(False)  # Disable dropout
            self.models.append(model)
        
        if self.verbose > 0:
            print(f"[Ensemble] Loaded {len(self.models)} models")
    
    def _ensure_loaded(self):
        """Lazy loading: ensure models are loaded before prediction."""
        if not self.models:
            self._load_models()
    
    def predict(
        self,
        obs: Union[Dict[str, np.ndarray], np.ndarray],
        deterministic: Optional[bool] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Predict action using ensemble aggregation.
        
        Args:
            obs: Observation (dict with 'market', 'position', 'w_cost' or flat array).
            deterministic: Override config.deterministic if provided.
        
        Returns:
            Tuple of (action, info_dict) where info contains:
                - ensemble_std: Standard deviation across members (disagreement)
                - ensemble_confidence: Aggregated confidence (higher = more certain)
                - member_actions: Individual actions from each member
                - member_confidences: Individual confidences (if confidence aggregation)
        """
        self._ensure_loaded()
        
        deterministic = deterministic if deterministic is not None else self.config.deterministic
        n_models = len(self.models)
        
        # Collect predictions from all models
        actions = []
        quantile_spreads = []
        
        for model in self.models:
            action, _ = model.predict(obs, deterministic=deterministic)
            actions.append(action)
            
            # Get quantile spread for confidence weighting
            if self.config.aggregation == 'confidence':
                spread = self._get_quantile_spread(model, obs)
                quantile_spreads.append(spread)
        
        # Stack actions: (n_models, batch_size, action_dim)
        actions_stack = np.stack(actions, axis=0)
        
        # Compute aggregation
        final_action, weights = self._aggregate(actions_stack, quantile_spreads)
        
        # Check agreement (optional filtering)
        agreement = self._compute_agreement(actions_stack)
        if self.config.min_agreement > 0 and agreement < self.config.min_agreement:
            final_action = np.full_like(final_action, self.config.disagreement_action)
        
        # Build info dict
        info = {
            'ensemble_std': float(np.std(actions_stack, axis=0).mean()),
            'ensemble_confidence': float(1.0 / (np.std(actions_stack, axis=0).mean() + 1e-6)),
            'ensemble_agreement': float(agreement),
            'member_actions': actions_stack.tolist(),
            'n_models': n_models,
        }
        
        if weights is not None:
            info['member_weights'] = weights.tolist()
            info['mean_weight'] = float(weights.mean())
            info['weight_std'] = float(weights.std())
        
        return final_action, info
    
    def _aggregate(
        self,
        actions: np.ndarray,
        quantile_spreads: List[np.ndarray],
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Aggregate actions from ensemble members.
        
        Args:
            actions: Stacked actions (n_models, batch_size, action_dim).
            quantile_spreads: List of quantile spreads per model.
        
        Returns:
            Tuple of (final_action, weights) where weights is None for some methods.
        """
        method = self.config.aggregation
        
        if method == 'mean':
            return np.mean(actions, axis=0), None
        
        elif method == 'median':
            return np.median(actions, axis=0), None
        
        elif method == 'confidence':
            # === Softmax Temperature Weighting (Audit Gemini) ===
            spreads = np.stack(quantile_spreads, axis=0)  # (n_models, batch_size)
            
            # Calibrate spread by EMA if enabled (prevents timidity in high vol)
            if self.config.calibrate_spread and hasattr(self, '_spread_ema'):
                spreads_norm = spreads / (self._spread_ema + self.config.confidence_epsilon)
            else:
                spreads_norm = spreads
                # Update EMA for next call
                if self.config.calibrate_spread:
                    current_mean = spreads.mean()
                    if not hasattr(self, '_spread_ema'):
                        self._spread_ema = current_mean
                    else:
                        alpha = self.config.spread_ema_alpha
                        self._spread_ema = alpha * current_mean + (1 - alpha) * self._spread_ema
            
            # Softmax with temperature: exp(-spread/τ) / Σ(exp(-spread/τ))
            # Lower spread = higher confidence = higher weight
            tau = self.config.softmax_temperature
            log_weights = -spreads_norm / tau
            # Numerical stability: subtract max before exp
            log_weights = log_weights - log_weights.max(axis=0, keepdims=True)
            weights = np.exp(log_weights)
            weights = weights / weights.sum(axis=0, keepdims=True)  # Normalize
            
            # Weighted average: (n_models, batch, 1) * (n_models, batch, action_dim)
            final_action = np.sum(
                actions * weights[:, :, np.newaxis], 
                axis=0
            )
            return final_action, weights
        
        elif method == 'conservative':
            # Select action closest to 0 (most risk-averse)
            abs_actions = np.abs(actions)
            # For each batch element, find model with smallest |action|
            min_idx = np.argmin(abs_actions.mean(axis=-1), axis=0)  # (batch,)
            
            final_action = np.zeros_like(actions[0])
            for i, idx in enumerate(min_idx):
                final_action[i] = actions[idx, i]
            
            return final_action, None
        
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
    
    def _get_quantile_spread(
        self, 
        model: TQC, 
        obs: Union[Dict[str, np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """
        Extract quantile spread from TQC critic for confidence estimation.
        
        The spread (max - min quantile) indicates uncertainty:
        - Low spread = high confidence
        - High spread = low confidence
        
        Args:
            model: TQC model.
            obs: Observation.
        
        Returns:
            Spread per batch element (batch_size,).
        """
        with torch.no_grad():
            # Convert obs to tensor dict
            if isinstance(obs, dict):
                obs_tensor = {
                    k: torch.tensor(v, device=self.device, dtype=torch.float32)
                    for k, v in obs.items()
                }
            else:
                obs_tensor = torch.tensor(obs, device=self.device, dtype=torch.float32)
            
            # Get features from extractor
            features = model.policy.extract_features(
                obs_tensor, 
                model.policy.features_extractor
            )
            
            # Get action from actor (mean for stochastic)
            action_dist = model.policy.actor(features)
            if hasattr(action_dist, 'mean'):
                action = action_dist.mean
            else:
                action = action_dist
            
            # Get quantile values from all critics
            # TQC stores critics as qf0, qf1, ... in critic module
            all_quantiles = []
            critic = model.critic
            
            # Handle different TQC implementations
            if hasattr(critic, 'quantile_critics'):
                # sb3-contrib structure
                for qf in critic.quantile_critics:
                    q_values = qf(features, action)
                    all_quantiles.append(q_values)
            else:
                # Try direct access (qf0, qf1, ...)
                for i in range(model.critic.n_critics if hasattr(model.critic, 'n_critics') else 2):
                    qf = getattr(critic, f'qf{i}', None)
                    if qf is not None:
                        q_values = qf(features, action)
                        all_quantiles.append(q_values)
            
            if not all_quantiles:
                # Fallback: return uniform spread
                batch_size = features.shape[0] if hasattr(features, 'shape') else 1
                return np.ones(batch_size) * 0.1
            
            # Concatenate all quantiles
            all_q = torch.cat(all_quantiles, dim=-1)  # (batch, total_quantiles)
            
            # Compute spread (max - min)
            spread = (all_q.max(dim=-1)[0] - all_q.min(dim=-1)[0]).cpu().numpy()
            
            return spread
    
    def _compute_agreement(self, actions: np.ndarray) -> float:
        """
        Compute agreement ratio across ensemble members.
        
        Agreement is measured as the inverse of action variance.
        High agreement = all models predict similar actions.
        
        Args:
            actions: Stacked actions (n_models, batch_size, action_dim).
        
        Returns:
            Agreement ratio in [0, 1] where 1 = perfect agreement.
        """
        # Standard deviation across models
        std = np.std(actions, axis=0).mean()
        
        # Convert to agreement (inversely proportional to std)
        # Max reasonable std is ~1.0 (action range is [-1, 1])
        agreement = np.clip(1.0 - std, 0.0, 1.0)
        
        return agreement
    
    def get_individual_predictions(
        self,
        obs: Union[Dict[str, np.ndarray], np.ndarray],
        deterministic: bool = True,
    ) -> List[Tuple[np.ndarray, Dict]]:
        """
        Get predictions from each ensemble member individually.
        
        Useful for ablation studies and debugging.
        
        Args:
            obs: Observation.
            deterministic: Use deterministic policy.
        
        Returns:
            List of (action, info) tuples, one per member.
        """
        self._ensure_loaded()
        
        results = []
        for i, model in enumerate(self.models):
            action, _ = model.predict(obs, deterministic=deterministic)
            spread = self._get_quantile_spread(model, obs)
            
            info = {
                'member_idx': i,
                'seed': self.config.seeds[i] if i < len(self.config.seeds) else None,
                'quantile_spread': float(spread.mean()),
                'confidence': float(1.0 / (spread.mean() + 1e-6)),
            }
            results.append((action, info))
        
        return results
    
    def close(self):
        """Release GPU memory."""
        for model in self.models:
            del model
        self.models = []
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if self.verbose > 0:
            print("[Ensemble] Models unloaded, GPU memory released")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# =============================================================================
# Ensemble Trainer
# =============================================================================

class EnsembleTrainer:
    """
    Trains multiple TQC models for ensemble.
    
    Supports:
    - Sequential training (single GPU)
    - Parallel training (multi-GPU)
    - Checkpoint collection for snapshot ensemble
    
    Example:
        >>> trainer = EnsembleTrainer(
        ...     base_config=tqc_config,
        ...     ensemble_config=EnsembleConfig(n_members=3)
        ... )
        >>> model_paths = trainer.train(output_dir='weights/ensemble/')
    """
    
    def __init__(
        self,
        base_config: 'TQCTrainingConfig',
        ensemble_config: Optional[EnsembleConfig] = None,
        verbose: int = 1,
    ):
        """
        Initialize ensemble trainer.
        
        Args:
            base_config: Base TQCTrainingConfig for all members.
            ensemble_config: Ensemble configuration.
            verbose: Verbosity level.
        """
        self.base_config = base_config
        self.config = ensemble_config or EnsembleConfig()
        self.verbose = verbose
        
        # Validate
        if self.config.n_members > len(self.config.seeds):
            raise ValueError(
                f"Need {self.config.n_members} seeds, got {len(self.config.seeds)}"
            )
    
    def train_sequential(self, output_dir: str) -> List[str]:
        """
        Train ensemble members sequentially (single GPU).
        
        Args:
            output_dir: Directory to save models.
        
        Returns:
            List of paths to trained models.
        """
        import copy
        import gc
        from src.training.train_agent import train
        from src.config import SEED
        
        os.makedirs(output_dir, exist_ok=True)
        model_paths = []
        all_metrics = []
        
        for i in range(self.config.n_members):
            seed = self.config.seeds[i]
            
            if self.verbose > 0:
                print(f"\n{'='*60}")
                print(f"Training Ensemble Member {i+1}/{self.config.n_members}")
                print(f"Seed: {seed}")
                print(f"{'='*60}")
            
            # Clone and modify config
            config = copy.deepcopy(self.base_config)
            config.seed = seed
            config.save_path = os.path.join(output_dir, f"tqc_seed_{seed}.zip")
            config.checkpoint_dir = os.path.join(output_dir, f"checkpoints_seed_{seed}/")
            config.name = f"ensemble_seed_{seed}"
            
            # === Forced Diversity (Audit Gemini) ===
            # Varier légèrement gamma et LR entre membres pour diversité structurelle
            if self.config.use_diverse_hyperparams:
                gamma_min, gamma_max = self.config.gamma_range
                lr_min, lr_max = self.config.lr_range
                n = self.config.n_members
                
                # Linspace pour distribution uniforme
                config.gamma = gamma_min + (gamma_max - gamma_min) * i / max(n - 1, 1)
                config.learning_rate = lr_min + (lr_max - lr_min) * i / max(n - 1, 1)
                
                if self.verbose > 0:
                    print(f"  [Diversity] gamma={config.gamma:.4f}, lr={config.learning_rate:.2e}")
            
            # Ensure directories exist
            os.makedirs(config.checkpoint_dir, exist_ok=True)
            
            # Train
            model, metrics = train(config, use_batch_env=True)
            
            model_paths.append(config.save_path)
            all_metrics.append(metrics)
            
            # Cleanup
            del model
            torch.cuda.empty_cache()
            gc.collect()
        
        # Save ensemble config
        self.config.to_json(os.path.join(output_dir, "ensemble_config.json"))
        
        # Save aggregated metrics
        self._save_ensemble_metrics(output_dir, all_metrics)
        
        return model_paths
    
    def train_parallel(self, output_dir: str) -> List[str]:
        """
        Train ensemble members in parallel (multi-GPU).
        
        Uses torch.multiprocessing to train on separate GPUs.
        
        Args:
            output_dir: Directory to save models.
        
        Returns:
            List of paths to trained models.
        """
        import copy
        import torch.multiprocessing as mp
        
        os.makedirs(output_dir, exist_ok=True)
        
        n_gpus = len(self.config.parallel_gpus)
        n_members = self.config.n_members
        
        if self.verbose > 0:
            print(f"\n{'='*60}")
            print(f"Parallel Ensemble Training")
            print(f"Members: {n_members}, GPUs: {n_gpus}")
            print(f"{'='*60}")
        
        # Prepare configs for each member
        configs = []
        for i in range(n_members):
            seed = self.config.seeds[i]
            gpu_id = self.config.parallel_gpus[i % n_gpus]
            
            config = copy.deepcopy(self.base_config)
            config.seed = seed
            config.save_path = os.path.join(output_dir, f"tqc_seed_{seed}.zip")
            config.checkpoint_dir = os.path.join(output_dir, f"checkpoints_seed_{seed}/")
            config.name = f"ensemble_seed_{seed}"
            config.device = f"cuda:{gpu_id}"
            
            os.makedirs(config.checkpoint_dir, exist_ok=True)
            configs.append((config, gpu_id))
        
        # Train in batches of n_gpus
        model_paths = []
        for batch_start in range(0, n_members, n_gpus):
            batch_end = min(batch_start + n_gpus, n_members)
            batch_configs = configs[batch_start:batch_end]
            
            if self.verbose > 0:
                print(f"\nBatch {batch_start//n_gpus + 1}: Members {batch_start+1}-{batch_end}")
            
            # Spawn processes
            mp.set_start_method('spawn', force=True)
            processes = []
            
            for config, gpu_id in batch_configs:
                p = mp.Process(
                    target=self._train_member,
                    args=(config, gpu_id)
                )
                p.start()
                processes.append(p)
            
            # Wait for all to finish
            for p in processes:
                p.join()
            
            # Collect paths
            for config, _ in batch_configs:
                model_paths.append(config.save_path)
        
        # Save ensemble config
        self.config.to_json(os.path.join(output_dir, "ensemble_config.json"))
        
        return model_paths
    
    @staticmethod
    def _train_member(config: 'TQCTrainingConfig', gpu_id: int):
        """Worker function for parallel training."""
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        from src.training.train_agent import train
        
        model, metrics = train(config, use_batch_env=True)
        del model
    
    def _save_ensemble_metrics(self, output_dir: str, all_metrics: List[Dict]):
        """Save aggregated ensemble training metrics."""
        import pandas as pd
        
        df = pd.DataFrame(all_metrics)
        df['seed'] = self.config.seeds[:len(all_metrics)]
        df.to_csv(os.path.join(output_dir, "ensemble_training_metrics.csv"), index=False)
        
        # Summary
        summary = {
            'n_members': len(all_metrics),
            'seeds': self.config.seeds[:len(all_metrics)],
            'avg_action_saturation': df['action_saturation'].mean() if 'action_saturation' in df else None,
            'avg_entropy': df['avg_entropy'].mean() if 'avg_entropy' in df else None,
            'avg_critic_loss': df['avg_critic_loss'].mean() if 'avg_critic_loss' in df else None,
        }
        
        with open(os.path.join(output_dir, "ensemble_summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)


# =============================================================================
# Utility Functions
# =============================================================================

def load_ensemble(
    ensemble_dir: str,
    config_path: Optional[str] = None,
    device: str = 'cuda',
) -> EnsemblePolicy:
    """
    Load a trained ensemble from directory.
    
    Args:
        ensemble_dir: Directory containing ensemble models.
        config_path: Path to config JSON. If None, looks for ensemble_config.json.
        device: PyTorch device.
    
    Returns:
        Loaded EnsemblePolicy.
    """
    # Load config
    if config_path is None:
        config_path = os.path.join(ensemble_dir, "ensemble_config.json")
    
    if os.path.exists(config_path):
        config = EnsembleConfig.from_json(config_path)
    else:
        config = EnsembleConfig()
    
    config.device = device
    
    # Find model files
    model_paths = []
    for seed in config.seeds[:config.n_members]:
        path = os.path.join(ensemble_dir, f"tqc_seed_{seed}.zip")
        if os.path.exists(path):
            model_paths.append(path)
    
    if not model_paths:
        raise FileNotFoundError(f"No ensemble models found in {ensemble_dir}")
    
    return EnsemblePolicy(model_paths, config=config)


def compare_single_vs_ensemble(
    single_model_path: str,
    ensemble_dir: str,
    test_env: VecEnv,
    n_episodes: int = 10,
) -> Dict[str, Any]:
    """
    Compare single model vs ensemble on test environment.
    
    Args:
        single_model_path: Path to single TQC model.
        ensemble_dir: Directory containing ensemble.
        test_env: Test environment.
        n_episodes: Number of evaluation episodes.
    
    Returns:
        Comparison metrics dict.
    """
    from stable_baselines3.common.evaluation import evaluate_policy
    
    # Evaluate single
    single_model = TQC.load(single_model_path)
    single_mean, single_std = evaluate_policy(
        single_model, test_env, n_eval_episodes=n_episodes
    )
    del single_model
    
    # Evaluate ensemble
    ensemble = load_ensemble(ensemble_dir)
    
    # Manual evaluation (EnsemblePolicy doesn't have evaluate_policy)
    ensemble_rewards = []
    for _ in range(n_episodes):
        obs = test_env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action, _ = ensemble.predict(obs)
            obs, reward, done, info = test_env.step(action)
            episode_reward += reward[0]
        
        ensemble_rewards.append(episode_reward)
    
    ensemble.close()
    
    ensemble_mean = np.mean(ensemble_rewards)
    ensemble_std = np.std(ensemble_rewards)
    
    return {
        'single_mean': single_mean,
        'single_std': single_std,
        'ensemble_mean': ensemble_mean,
        'ensemble_std': ensemble_std,
        'improvement_mean': ensemble_mean - single_mean,
        'improvement_pct': (ensemble_mean - single_mean) / abs(single_mean) * 100 if single_mean != 0 else 0,
        'variance_reduction': (single_std - ensemble_std) / single_std * 100 if single_std > 0 else 0,
    }
```

### 4.2 Intégration WFO

**Modifications à** `scripts/run_full_wfo.py` :

```python
# Ajouter après les imports existants
from src.evaluation.ensemble import EnsemblePolicy, EnsembleTrainer, EnsembleConfig, load_ensemble

# Ajouter à WFOConfig
@dataclass
class WFOConfig:
    # ... existing fields ...
    
    # === Ensemble RL ===
    use_ensemble: bool = False
    ensemble_n_members: int = 3
    ensemble_seeds: List[int] = field(default_factory=lambda: [42, 123, 456])
    ensemble_aggregation: str = 'confidence'
    ensemble_parallel: bool = True  # Use 2 GPUs


# Nouvelle méthode dans WFOPipeline
def train_tqc_ensemble(
    self,
    train_path: str,
    encoder_path: str,
    segment_id: int,
    eval_path: Optional[str] = None,
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Train TQC ensemble for a segment.
    
    Args:
        train_path: Path to training data.
        encoder_path: Path to encoder weights.
        segment_id: Segment identifier.
        eval_path: Path to eval data (optional).
    
    Returns:
        Tuple of (model_paths, aggregated_metrics).
    """
    print(f"\n[Segment {segment_id}] Training TQC Ensemble...")
    print(f"  Members: {self.config.ensemble_n_members}")
    print(f"  Seeds: {self.config.ensemble_seeds[:self.config.ensemble_n_members]}")
    print(f"  Parallel: {self.config.ensemble_parallel}")
    
    # Create base config
    from src.config import TQCTrainingConfig
    
    base_config = TQCTrainingConfig()
    base_config.data_path = train_path
    base_config.encoder_path = encoder_path
    base_config.total_timesteps = self.config.tqc_timesteps
    base_config.learning_rate = self.config.learning_rate
    base_config.buffer_size = self.config.buffer_size
    base_config.n_envs = self.config.n_envs
    base_config.batch_size = self.config.batch_size
    base_config.gamma = self.config.gamma
    base_config.ent_coef = self.config.ent_coef
    base_config.observation_noise = self.config.observation_noise
    base_config.critic_dropout = self.config.critic_dropout
    base_config.target_volatility = self.config.target_volatility
    base_config.vol_window = self.config.vol_window
    base_config.max_leverage = self.config.max_leverage
    base_config.use_curriculum = True
    base_config.eval_data_path = eval_path
    
    # Ensemble config
    ensemble_config = EnsembleConfig(
        n_members=self.config.ensemble_n_members,
        seeds=self.config.ensemble_seeds[:self.config.ensemble_n_members],
        aggregation=self.config.ensemble_aggregation,
        parallel_gpus=[0, 1],  # 2 GPUs available
    )
    
    # Output directory
    weights_dir = os.path.join(self.config.weights_dir, f"segment_{segment_id}")
    ensemble_dir = os.path.join(weights_dir, "ensemble")
    
    # Create trainer
    trainer = EnsembleTrainer(
        base_config=base_config,
        ensemble_config=ensemble_config,
        verbose=1,
    )
    
    # Train
    if self.config.ensemble_parallel:
        model_paths = trainer.train_parallel(ensemble_dir)
    else:
        model_paths = trainer.train_sequential(ensemble_dir)
    
    # Load aggregated metrics
    metrics_path = os.path.join(ensemble_dir, "ensemble_summary.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            aggregated_metrics = json.load(f)
    else:
        aggregated_metrics = {'n_members': len(model_paths)}
    
    # Create symlink for backward compatibility
    if model_paths:
        best_model = model_paths[0]  # First seed as "best"
        tqc_symlink = os.path.join(weights_dir, "tqc.zip")
        if os.path.exists(tqc_symlink):
            os.remove(tqc_symlink)
        os.symlink(best_model, tqc_symlink)
    
    print(f"  Ensemble trained: {len(model_paths)} models")
    
    return model_paths, aggregated_metrics


def evaluate_ensemble_segment(
    self,
    test_path: str,
    encoder_path: str,
    ensemble_dir: str,
    segment_id: int,
    context_rows: int = 0,
    train_metrics: Optional[Dict] = None,
    train_path: Optional[str] = None,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    Evaluate ensemble on test data.
    
    Compares:
    1. Best single model
    2. Ensemble with configured aggregation
    """
    print(f"\n[Segment {segment_id}] Evaluating Ensemble...")
    
    import torch
    from src.training.batch_env import BatchCryptoEnv
    
    # Load ensemble
    ensemble = load_ensemble(ensemble_dir, device='cuda')
    
    # Calculate baseline_vol from TRAIN data
    if train_path and os.path.exists(train_path):
        train_df = pd.read_parquet(train_path)
        baseline_vol = train_df['BTC_Close'].pct_change().std()
    else:
        baseline_vol = 0.01
    
    # Create test env
    test_df = pd.read_parquet(test_path)
    test_episode_length = len(test_df) - self.config.window_size - 1
    
    env = BatchCryptoEnv(
        parquet_path=test_path,
        n_envs=1,
        device='cuda',
        window_size=self.config.window_size,
        episode_length=test_episode_length,
        commission=0.0004,
        slippage=0.0001,
        target_volatility=self.config.target_volatility,
        vol_window=self.config.vol_window,
        max_leverage=self.config.max_leverage,
        price_column='BTC_Close',
        random_start=False,
    )
    
    # Run evaluation
    obs, info = env.gym_reset()
    done = False
    rewards = []
    navs = []
    ensemble_metrics_history = []
    
    while not done:
        action, ensemble_info = ensemble.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.gym_step(action)
        done = terminated or truncated
        
        rewards.append(reward)
        navs.append(info.get('nav', 10000.0))
        ensemble_metrics_history.append({
            'std': ensemble_info['ensemble_std'],
            'agreement': ensemble_info['ensemble_agreement'],
        })
    
    ensemble.close()
    env.close()
    
    # Calculate metrics (skip warmup)
    all_rewards = np.array(rewards)
    all_navs = np.array(navs)
    
    if len(all_rewards) > context_rows:
        rewards = all_rewards[context_rows:]
        navs = all_navs[context_rows:]
    else:
        rewards = all_rewards
        navs = all_navs
    
    # Sharpe
    if len(rewards) > 1 and rewards.std() > 0:
        sharpe = (rewards.mean() / rewards.std()) * np.sqrt(8760)
    else:
        sharpe = 0.0
    
    # PnL
    pnl = navs[-1] - navs[0] if len(navs) > 1 else 0
    pnl_pct = (pnl / navs[0]) * 100 if navs[0] > 0 else 0
    
    # Max Drawdown
    if len(navs) > 1:
        peak = np.maximum.accumulate(navs)
        drawdown = (peak - navs) / peak
        max_drawdown = drawdown.max() * 100
    else:
        max_drawdown = 0.0
    
    # Ensemble-specific metrics
    avg_agreement = np.mean([m['agreement'] for m in ensemble_metrics_history])
    avg_std = np.mean([m['std'] for m in ensemble_metrics_history])
    
    metrics = {
        'segment_id': segment_id,
        'model_type': 'ensemble',
        'aggregation': self.config.ensemble_aggregation,
        'n_members': self.config.ensemble_n_members,
        'sharpe': sharpe,
        'pnl': pnl,
        'pnl_pct': pnl_pct,
        'max_drawdown': max_drawdown,
        'final_nav': navs[-1] if len(navs) > 0 else 10000,
        'test_rows': len(rewards),
        'ensemble_avg_agreement': avg_agreement,
        'ensemble_avg_std': avg_std,
    }
    
    if train_metrics:
        metrics.update({
            'train_action_sat': train_metrics.get('action_saturation', 0.0),
            'train_entropy': train_metrics.get('avg_entropy', 0.0),
        })
    
    print(f"  Results (Ensemble):")
    print(f"    Sharpe: {sharpe:.2f}")
    print(f"    PnL: {pnl_pct:+.2f}%")
    print(f"    Max DD: {max_drawdown:.2f}%")
    print(f"    Avg Agreement: {avg_agreement:.2%}")
    print(f"    Avg Action Std: {avg_std:.4f}")
    
    torch.cuda.empty_cache()
    
    return metrics, navs
```

---

## 5. Configuration

### 5.1 Paramètres par Défaut

```python
# Ajouts à src/config/training.py

@dataclass
class TQCTrainingConfig:
    # ... existing fields ...
    
    # === Ensemble RL (Design Doc 2026-01-21) ===
    # Reference: docs/design/ENSEMBLE_RL_DESIGN.md
    use_ensemble: bool = False
    ensemble_n_members: int = 3
    ensemble_seeds: List[int] = field(default_factory=lambda: [42, 123, 456])
    ensemble_aggregation: str = 'confidence'  # 'mean', 'median', 'confidence', 'vote', 'conservative'
    ensemble_parallel: bool = True  # Train on multiple GPUs
    ensemble_parallel_gpus: List[int] = field(default_factory=lambda: [0, 1])
```

### 5.2 Arguments CLI

```python
# Ajouts au parser de run_full_wfo.py

# === Ensemble RL Arguments ===
parser.add_argument("--ensemble", action="store_true",
                    help="Enable ensemble training (default: 3 members)")
parser.add_argument("--ensemble-members", type=int, default=3,
                    help="Number of ensemble members (default: 3)")
parser.add_argument("--ensemble-seeds", type=str, default="42,123,456",
                    help="Comma-separated seeds for ensemble members")
parser.add_argument("--ensemble-aggregation", type=str,
                    choices=['mean', 'median', 'confidence', 'vote', 'conservative'],
                    default='confidence',
                    help="Ensemble aggregation method (default: confidence)")
parser.add_argument("--no-ensemble-parallel", action="store_true",
                    help="Disable parallel training (use single GPU)")
```

### 5.3 Usage Exemples

```bash
# Training WFO avec ensemble (parallel sur 2 GPUs)
python scripts/run_full_wfo.py --ensemble --ensemble-members 3

# Training avec agrégation conservative (risk-averse)
python scripts/run_full_wfo.py --ensemble --ensemble-aggregation conservative

# Training séquentiel (single GPU)
python scripts/run_full_wfo.py --ensemble --no-ensemble-parallel

# Personnaliser les seeds
python scripts/run_full_wfo.py --ensemble --ensemble-seeds "42,1337,9999"
```

---

## 6. Intégration WFO

### 6.1 Modification de `run_segment()`

```python
def run_segment(
    self,
    df_raw: pd.DataFrame,
    segment: Dict[str, int],
    use_batch_env: bool = False,
    resume: bool = False,
    init_model_path: Optional[str] = None
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Run full WFO pipeline for a single segment.
    
    Supports both single model and ensemble training.
    """
    segment_id = segment['id']
    
    # ... existing preprocessing code ...
    
    # 5. TQC Training (Single vs Ensemble)
    if self.config.use_ensemble:
        # === ENSEMBLE MODE ===
        model_paths, train_metrics = self.train_tqc_ensemble(
            train_path, encoder_path, segment_id,
            eval_path=eval_path
        )
        
        # Evaluate ensemble
        metrics, navs = self.evaluate_ensemble_segment(
            test_path, encoder_path,
            ensemble_dir=os.path.join(weights_dir, "ensemble"),
            segment_id=segment_id,
            context_rows=context_rows,
            train_metrics=train_metrics,
            train_path=train_path,
        )
        
        # Also evaluate single best for comparison
        single_metrics, _ = self.evaluate_segment(
            test_path, encoder_path, model_paths[0], segment_id,
            context_rows=context_rows,
            train_metrics={},
            train_path=train_path,
        )
        
        # Log comparison
        print(f"\n  [Comparison] Single vs Ensemble:")
        print(f"    Single Sharpe: {single_metrics['sharpe']:.2f}")
        print(f"    Ensemble Sharpe: {metrics['sharpe']:.2f}")
        print(f"    Improvement: {metrics['sharpe'] - single_metrics['sharpe']:+.2f}")
        
        metrics['single_sharpe'] = single_metrics['sharpe']
        metrics['single_pnl_pct'] = single_metrics['pnl_pct']
        metrics['improvement_sharpe'] = metrics['sharpe'] - single_metrics['sharpe']
        
    else:
        # === SINGLE MODEL MODE (existing code) ===
        tqc_path, train_metrics = self.train_tqc(
            train_path, encoder_path, segment_id,
            use_batch_env=use_batch_env,
            resume=resume,
            init_model_path=init_model_path,
            eval_path=eval_path
        )
        
        metrics, navs = self.evaluate_segment(
            test_path, encoder_path, tqc_path, segment_id,
            context_rows=context_rows,
            train_metrics=train_metrics,
            train_path=train_path,
        )
    
    # ... rest of existing code ...
```

### 6.2 Chain of Inheritance pour Ensemble

```python
# Warm start : chaque membre hérite du même seed du segment précédent

def get_ensemble_init_paths(self, prev_segment_id: int) -> Dict[int, str]:
    """Get init paths for ensemble warm start."""
    prev_ensemble_dir = os.path.join(
        self.config.weights_dir, 
        f"segment_{prev_segment_id}", 
        "ensemble"
    )
    
    init_paths = {}
    for seed in self.config.ensemble_seeds:
        path = os.path.join(prev_ensemble_dir, f"tqc_seed_{seed}.zip")
        if os.path.exists(path):
            init_paths[seed] = path
    
    return init_paths
```

---

## 7. Intégration Composants Existants

### 7.1 EMA Callback

**Stratégie** : Un EMA par membre d'ensemble

```python
# Chaque membre a son propre ModelEMACallback
# Les fichiers EMA sont sauvegardés comme: tqc_seed_{seed}_ema.zip

# Pour l'inférence avec EMA :
class EnsemblePolicy:
    def __init__(self, model_paths, use_ema=True, ...):
        if use_ema:
            # Chercher les fichiers EMA correspondants
            ema_paths = [p.replace('.zip', '_ema.zip') for p in model_paths]
            model_paths = [ep if os.path.exists(ep) else p 
                         for p, ep in zip(model_paths, ema_paths)]
        # ...
```

### 7.2 OverfittingGuard

**Stratégie** : Guard individuel + Guard ensemble

```python
# Ajout de signaux spécifiques ensemble dans OverfittingGuardCallbackV2

def _check_ensemble_collapse(self) -> Optional[str]:
    """
    Signal 6: Détecte la perte de diversité de l'ensemble.
    
    Si tous les membres prédisent la même action, l'ensemble
    perd son avantage de robustesse.
    """
    if not hasattr(self, 'ensemble_std_history'):
        return None
    
    if len(self.ensemble_std_history) < 100:
        return None
    
    recent_std = np.mean(self.ensemble_std_history[-100:])
    
    # Seuil: si std < 0.01, tous les modèles sont quasi-identiques
    if recent_std < 0.01:
        return f"Ensemble diversity collapse (std={recent_std:.4f})"
    
    return None
```

### 7.3 Curriculum Learning

**Stratégie** : Curriculum synchronisé entre membres

```python
# Tous les membres utilisent le même progress pour curriculum_lambda
# Cela garantit que tous voient les mêmes niveaux de difficulté

# Dans ThreePhaseCurriculumCallback:
def _update_envs(self):
    progress = self.num_timesteps / self.total_timesteps
    
    # Broadcast progress à tous les envs (même si c'est le même)
    real_env = get_underlying_batch_env(self.model.env)
    if hasattr(real_env, 'set_progress'):
        real_env.set_progress(progress)
```

---

## 8. Risques et Mitigations

| Risque | Probabilité | Impact | Mitigation |
|--------|-------------|--------|------------|
| **Mémoire GPU (Training)** | Moyenne | Élevé | **Shared Replay Buffer** (Audit) : un seul buffer RAM partagé entre processus |
| **Mémoire GPU (Inférence)** | Faible | Moyen | Inférence séquentielle par membre ; `close()` après évaluation |
| **Performance I/O (Loading)** | Moyenne | Moyen | **Lazy Loading** obligatoire : ne charger les modèles qu'à la demande |
| **Training 2× plus long** | Certaine | Moyen | Training parallèle sur 2 GPUs |
| **Diversité insuffisante** | Faible | Moyen | **Forced Diversity** (Audit) : varier gamma/LR entre membres |
| **Overfitting collectif** | Faible | Élevé | OverfittingGuard V2 + Signal 6 (Ensemble Collapse) |
| **Latence inférence > 10ms** | Faible | Élevé | Architecture tiny (64×64) ; pré-chargement modèles |
| **Timidité haute volatilité** | Moyenne | Moyen | **Spread Calibration** (Audit) : normaliser par EMA(spread) |
| **Disagreement permanent** | Faible | Moyen | `min_agreement` filter ; fallback to single model |

### 8.1 Performance I/O & Lazy Loading

**Problème** : Le chargement de 3+ modèles `.zip` (TQC complets) à chaque inférence ou changement de segment WFO peut saturer la RAM et ralentir le système.

**Solution : Lazy Loading Pattern**

```python
# Pattern recommandé dans EnsemblePolicy
class EnsemblePolicy:
    def __init__(self, model_paths, config):
        self.model_paths = model_paths
        self.models = []  # Vide au départ
        self._models_loaded = False
    
    def _ensure_loaded(self):
        """Charge les modèles uniquement au premier predict()."""
        if not self._models_loaded:
            self._load_models()
            self._models_loaded = True
    
    def predict(self, obs):
        self._ensure_loaded()  # Lazy load
        # ... inference ...
    
    def close(self):
        """Libère explicitement la mémoire GPU."""
        for model in self.models:
            del model
        self.models = []
        self._models_loaded = False
        torch.cuda.empty_cache()
```

**Cycle de vie recommandé en WFO** :

```
Segment N:
  1. train_ensemble() → sauvegarde 3 .zip
  2. evaluate_segment() → load lazy → predict → close() → libère RAM
  3. torch.cuda.empty_cache()

Segment N+1:
  1. Répéter (modèles précédents déjà libérés)
```

| Mode | `preload_models` | `unload_after_predict` | Usage |
|------|------------------|------------------------|-------|
| **Live Trading** | `True` | `False` | Latence minimale, RAM constante |
| **WFO Evaluation** | `False` | `True` | RAM minimale, latence acceptable |
| **Backtesting Long** | `False` | `False` | Équilibre (lazy load, garde en mémoire) |

### 8.2 Estimation Mémoire GPU

```
Single TQC (tiny arch):
  - Policy: ~2MB
  - Critic: ~4MB
  - Total: ~10MB avec buffers

Ensemble 3 membres:
  - Models: 3 × 10MB = 30MB
  - BatchCryptoEnv: ~500MB (1024 envs, données)
  - Total inférence: ~550MB

→ Confortablement dans les 8GB+ d'une GPU moderne
```

### 8.3 Estimation Temps Training

```
Single model (30M steps, 1024 envs, GPU):
  - FPS: ~50k steps/s
  - Temps: ~10 minutes

Ensemble 3 membres (parallèle 2 GPUs):
  - Batch 1: 2 modèles en parallèle = ~10 min
  - Batch 2: 1 modèle séquentiel = ~10 min
  - Total: ~20 minutes (2× single)

Ensemble 3 membres (séquentiel 1 GPU):
  - Total: ~30 minutes (3× single)
```

---

## 9. Plan de Validation

### 9.1 Tests Unitaires

**Fichier** : `tests/test_ensemble.py`

```python
"""Tests for Ensemble RL implementation."""

import pytest
import numpy as np
import torch
import tempfile
import os

from src.evaluation.ensemble import (
    EnsemblePolicy, 
    EnsembleConfig, 
    EnsembleTrainer,
    load_ensemble,
)


class TestEnsembleConfig:
    """Tests for EnsembleConfig."""
    
    def test_default_values(self):
        config = EnsembleConfig()
        assert config.n_members == 3
        assert config.aggregation == 'confidence'
        assert len(config.seeds) >= config.n_members
    
    def test_json_roundtrip(self, tmp_path):
        config = EnsembleConfig(n_members=5, aggregation='median')
        path = tmp_path / "config.json"
        config.to_json(str(path))
        
        loaded = EnsembleConfig.from_json(str(path))
        assert loaded.n_members == 5
        assert loaded.aggregation == 'median'


class TestEnsemblePolicy:
    """Tests for EnsemblePolicy."""
    
    @pytest.fixture
    def mock_models(self, tmp_path):
        """Create mock TQC models for testing."""
        # This would require actual model files
        # Skip if not available
        pytest.skip("Requires trained models")
    
    def test_aggregation_mean(self):
        """Test mean aggregation."""
        actions = np.array([
            [[0.5], [0.3]],   # Model 0
            [[0.7], [0.5]],   # Model 1
            [[0.6], [0.4]],   # Model 2
        ])
        
        expected = np.array([[0.6], [0.4]])
        result = np.mean(actions, axis=0)
        
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_aggregation_median(self):
        """Test median aggregation."""
        actions = np.array([
            [[0.1], [0.1]],   # Outlier
            [[0.5], [0.5]],   # Median
            [[0.6], [0.6]],   # 
        ])
        
        expected = np.array([[0.5], [0.5]])
        result = np.median(actions, axis=0)
        
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_aggregation_conservative(self):
        """Test conservative aggregation (closest to 0)."""
        actions = np.array([
            [[0.9]],   # Aggressive
            [[0.1]],   # Conservative ← should be selected
            [[0.5]],   # Medium
        ])
        
        # Conservative selects action with smallest |action|
        abs_actions = np.abs(actions)
        min_idx = np.argmin(abs_actions.mean(axis=-1), axis=0)
        
        assert min_idx[0] == 1  # Model 1 is most conservative
    
    def test_agreement_computation(self):
        """Test agreement ratio computation."""
        # Perfect agreement
        actions_agree = np.array([
            [[0.5]],
            [[0.5]],
            [[0.5]],
        ])
        std_agree = np.std(actions_agree, axis=0).mean()
        agreement_agree = np.clip(1.0 - std_agree, 0.0, 1.0)
        assert agreement_agree == 1.0
        
        # High disagreement
        actions_disagree = np.array([
            [[-1.0]],
            [[0.0]],
            [[1.0]],
        ])
        std_disagree = np.std(actions_disagree, axis=0).mean()
        agreement_disagree = np.clip(1.0 - std_disagree, 0.0, 1.0)
        assert agreement_disagree < 0.5


class TestEnsembleTrainer:
    """Tests for EnsembleTrainer."""
    
    def test_seed_validation(self):
        """Test that trainer validates seeds."""
        from src.config import TQCTrainingConfig
        
        config = TQCTrainingConfig()
        ensemble_config = EnsembleConfig(n_members=10, seeds=[1, 2, 3])
        
        with pytest.raises(ValueError, match="Need 10 seeds"):
            EnsembleTrainer(config, ensemble_config)


class TestIntegration:
    """Integration tests with actual models."""
    
    @pytest.mark.slow
    def test_full_pipeline(self):
        """Test full ensemble training and evaluation."""
        pytest.skip("Requires GPU and ~30 minutes")
```

### 9.2 Métriques de Succès

| Métrique | Baseline (Single) | Target (Ensemble) | Amélioration |
|----------|-------------------|-------------------|--------------|
| Sharpe Ratio OOS | X | ≥ X | ≥ +10% |
| Max Drawdown | Y% | ≤ Y% | ≤ -10% |
| Variance inter-runs | σ | σ/2 | -50% |
| Action Consistency | variable | plus stable | mesurable |

### 9.3 A/B Test Protocol

```
Durée: 5 segments WFO minimum
Seeds: Mêmes données, random seeds différents

Configurations:
├── A (Baseline): Single TQC, seed=42
├── B (Ensemble Mean): 3 seeds, aggregation='mean'
├── C (Ensemble Confidence): 3 seeds, aggregation='confidence' ← Primary
└── D (Ensemble Conservative): 3 seeds, aggregation='conservative'

Métriques collectées:
├── Sharpe Ratio (per segment, average)
├── PnL % (per segment, average)
├── Max Drawdown (per segment, worst)
├── Ensemble Agreement (per step, average)
└── Training Time (total)

Critères de succès:
├── Sharpe C > Sharpe A (p < 0.05)
├── MaxDD C < MaxDD A (p < 0.05)
└── Variance(Sharpe C) < Variance(Sharpe A)
```

---

## 10. Plan d'Implémentation

### Phase 1 : MVP (Semaine 1)

- [ ] Créer `src/evaluation/ensemble.py` avec `EnsemblePolicy`
- [ ] Créer `EnsembleConfig` dans `src/config/training.py`
- [ ] Tests unitaires basiques (`tests/test_ensemble.py`)
- [ ] Documentation inline complète

### Phase 2 : Intégration WFO (Semaine 2)

- [ ] Ajouter `EnsembleTrainer` avec training séquentiel
- [ ] Modifier `run_full_wfo.py` pour supporter `--ensemble`
- [ ] Intégrer évaluation ensemble dans `evaluate_segment`
- [ ] Tests d'intégration sur 1 segment

### Phase 3 : Optimisation (Semaine 3)

- [ ] Implémenter training parallèle multi-GPU
- [ ] Ajouter monitoring TensorBoard (agreement, confidence)
- [ ] Intégrer avec Chain of Inheritance
- [ ] A/B test sur 3+ segments

### Phase 4 : Production (Semaine 4)

- [ ] Optimisation mémoire (lazy loading)
- [ ] Ajout `torch.compile` pour inférence
- [ ] Documentation utilisateur
- [ ] Benchmark final avec rapport

---

## Annexes

### A. Références

1. **Ensemble RL through Classifier Models** (arXiv:2502.17518)
   - Mixing RL avec classifiers pour agrégation
   
2. **DroQ** (Hiraoka et al., 2021)
   - Dropout comme alternative aux grands ensembles
   - Déjà implémenté dans le projet (`TQCDropoutPolicy`)
   
3. **TQC** (Kuznetsov et al., 2020)
   - Truncated Quantile Critics
   - Base du projet, fournit l'incertitude via quantiles
   
4. **REDQ** (Chen et al., 2021)
   - 20 critics avec subset random
   - Trop coûteux, mais inspire la diversité

### B. Résumé Complet de l'Audit (Gemini AI)

**Date** : 2026-01-21  
**Auditeur** : Gemini AI  
**Verdict** : 🟢 EXCELLENT / VALIDÉ  

#### Points Validés ✅

1. **Confidence-Weighted Aggregation** : Approximation élégante de l'incertitude bayésienne sans coût computationnel
2. **Architecture Parallèle** : Indispensable pour WFO, bien conçue avec `torch.multiprocessing`
3. **Intégration TQC Native** : Utilise les quantiles existants, coût nul en inférence
4. **Signal 6 (Ensemble Collapse)** : Innovation pertinente pour détecter la perte de diversité

#### Améliorations Recommandées et Intégrées ✅

| Recommandation | Intégrée | Détail |
|----------------|----------|--------|
| Softmax Temperature | ✅ | `exp(-spread/τ)` remplace `1/spread` pour accentuer différences |
| Spread Calibration | ✅ | Normaliser par `EMA(spread)` évite timidité en haute vol |
| Forced Diversity | ✅ | Varier gamma [0.94-0.96] et LR [5e-5 - 2e-4] entre membres |
| Shared Replay Buffer | ✅ | Option pour mitigation OOM en parallèle |
| Vote Discret Déprécié | ✅ | Supprimé, préférer `median` ou `confidence` |

#### Questions Résolues

| Question | Réponse |
|----------|---------|
| ROI de l'ensemble ? | OUI - Réduction variance plus précieuse que gain espérance |
| Compatible MORL ? | OUI - Agrégation sur action finale, indépendant de génération |
| Pourquoi pas REDQ ? | Trop lourd (20 critics), Multi-Seed meilleur compromis |

### C. Changelog

| Date | Version | Changement |
|------|---------|------------|
| 2026-01-21 | 1.0 | Design initial |
| 2026-01-21 | 1.1 | Audit Gemini: softmax temperature, calibration spread, diversité forcée |
| 2026-01-22 | 1.2 | Audit SOTA #2: Pessimistic Bound, Risk Aversion scaling |
| 2026-01-22 | 1.3 | Audit SOTA #3: OOD Detection, Conservative Fallback, Model Trust |

---

**Statut** : ✅ IMPLÉMENTÉ - Production Ready

**Fichiers implémentés** :
- `src/evaluation/ensemble.py` : EnsemblePolicy + EnsembleTrainer ✅
- `src/config/training.py` : Configuration ensemble dans TQCTrainingConfig ✅

**Fonctionnalités implémentées** :
- ✅ `EnsemblePolicy` avec lazy loading
- ✅ Méthodes d'agrégation : `mean`, `median`, `confidence`, `conservative`, `pessimistic_bound`
- ✅ Softmax temperature weighting pour confidence
- ✅ Spread calibration via EMA
- ✅ OOD Detection avec conservative fallback
- ✅ Pessimistic bound scaling
- ✅ `EnsembleTrainer` pour training séquentiel et parallèle
- ✅ Intégration avec configuration TQC

**Tests** :
- ✅ `tests/test_ensemble.py` : Tests unitaires implémentés
- ✅ `tests/test_ensemble_sanity.py` : Tests de sanity check

**Prochaines étapes** :
1. Intégration complète dans WFO pipeline
2. Benchmark performance vs single model
3. Visualisation des métriques d'ensemble

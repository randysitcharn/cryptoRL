# Intégration OverfittingGuardCallbackV2 dans Walk-Forward Optimization

**Version** : 1.3  
**Date** : 2026-01-19  
**Statut** : PRÊT POUR IMPLÉMENTATION  
**Objectif** : Activer la détection d'overfitting intra-train pendant le WFO

---

## Table des Matières

1. [Contexte](#1-contexte)
2. [Problématique](#2-problématique)
3. [Solution Proposée](#3-solution-proposée)
4. [Architecture](#4-architecture)
5. [Implémentation](#5-implémentation)
6. [Gestion de l'Arrêt Prématuré (Fail-over)](#6-gestion-de-larrêt-prématuré-fail-over)
7. [Continuité WFO (Chain of Inheritance)](#7-continuité-wfo-chain-of-inheritance)
8. [Configuration](#8-configuration)
9. [Tests de Validation](#9-tests-de-validation)
10. [Références](#10-références)

---

## 1. Contexte

### 1.1 État Actuel

Le projet dispose de deux mécanismes de protection contre l'overfitting :

| Composant | Fichier | Usage Actuel |
|-----------|---------|--------------|
| `OverfittingGuardCallbackV2` | `src/training/callbacks.py` | Utilisé dans `train_agent.py` (mode standard) |
| Walk-Forward Optimization | `scripts/run_full_wfo.py` | Validation out-of-sample post-training |

### 1.2 OverfittingGuardCallbackV2 - Rappel

Callback SOTA avec 5 signaux de détection :

```
Signal 1: NAV Threshold      → Détecte returns irréalistes (+400%)
Signal 2: Weight Stagnation  → Détecte convergence/collapse du réseau
Signal 3: Train/Eval Diverg. → Détecte overfitting classique (nécessite EvalCallback)
Signal 4: Action Saturation  → Détecte policy collapse (actions bloquées à ±1)
Signal 5: Reward Variance    → Détecte mémorisation (variance → 0)
```

### 1.3 WFO - Rappel

Pipeline Walk-Forward par segment :

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         SEGMENT N                                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌─────────────────────────────┐    ┌─────────────────────────────┐   │
│   │          TRAIN              │    │           TEST              │   │
│   │        (18 mois)            │    │         (3 mois)            │   │
│   │                             │    │                             │   │
│   │  MAE + TQC Training         │    │  Évaluation OOS             │   │
│   │  (actuellement SANS Guard)  │    │  (après training)           │   │
│   └─────────────────────────────┘    └─────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Problématique

### 2.1 Le Problème

Actuellement, `run_full_wfo.py` **n'utilise PAS** `OverfittingGuardCallbackV2` :

```python
# run_full_wfo.py, ligne 515
config.eval_data_path = None  # WFO mode: disable EvalCallback
```

**Conséquence** : Le training peut continuer même si l'agent :
- Atteint des NAV irréalistes (Signal 1)
- A des poids qui stagnent (Signal 2)
- Sature ses actions à ±1 (Signal 4)
- A des rewards sans variance (Signal 5)

### 2.2 Pourquoi C'est Intentionnel (Actuellement)

Le WFO désactive `EvalCallback` pour éviter le **data leakage** :

```
❌ RISQUE AVEC EVAL STANDARD EN WFO:

Segment TRAIN: 2020-01 → 2021-06
Segment TEST:  2021-07 → 2021-09

Si EvalCallback utilise des données de 2021-07+, l'agent
"voit" le futur pendant le training = DATA LEAKAGE
```

### 2.3 Ce Qu'on Perd

Sans `OverfittingGuardCallbackV2` en WFO :

| Signal | Disponible ? | Conséquence |
|--------|--------------|-------------|
| Signal 1 (NAV) | ❌ Non | Training continue même avec +1000% NAV |
| Signal 2 (Weights) | ❌ Non | Pas de détection de collapse |
| Signal 3 (Divergence) | ❌ Non | Normal (pas d'eval) |
| Signal 4 (Saturation) | ❌ Non | Policy collapse non détecté |
| Signal 5 (Variance) | ❌ Non | Mémorisation non détectée |

**Résultat** : On brûle du GPU sur des trainings qui auraient dû s'arrêter tôt.

---

## 3. Solution Proposée

### 3.1 Approche : Intégration Partielle

Activer `OverfittingGuardCallbackV2` en WFO avec **4 signaux sur 5** :

```
┌─────────────────────────────────────────────────────────────────────────┐
│              OverfittingGuardCallbackV2 en Mode WFO                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Signal 1: NAV Threshold       ✅ ACTIF   (pas de dépendance externe)  │
│   Signal 2: Weight Stagnation   ✅ ACTIF   (lecture poids du modèle)    │
│   Signal 3: Train/Eval Diverg.  ❌ DÉSACTIVÉ (pas d'EvalCallback)       │
│   Signal 4: Action Saturation   ✅ ACTIF   (lecture actions locales)    │
│   Signal 5: Reward Variance     ✅ ACTIF   (lecture rewards locales)    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Option Avancée : Split Intra-Train (Déconseillé)

> **⚠️ AVIS AUDIT : DÉCONSEILLÉ POUR L'INSTANT**
> 
> Réduire les données d'entraînement de 10% (de 18 mois à ~16 mois) pour gagner un signal 
> de validation est un compromis risqué en séries temporelles financières où la diversité 
> des régimes de marché est clé. Mieux vaut s'en tenir aux signaux 1, 2, 4, 5.

Pour activer **Signal 3**, créer un holdout **DANS** les données TRAIN :

```
                        SEGMENT TRAIN (18 mois)
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│   ┌───────────────────────────────────┐  ┌───┐  ┌─────────────────┐    │
│   │         TRAIN-TRAIN               │  │ P │  │   TRAIN-EVAL    │    │
│   │           (90%)                   │  │ U │  │     (10%)       │    │
│   │                                   │  │ R │  │                 │    │
│   │   TQC apprend ici                 │  │ G │  │  EvalCallback   │    │
│   │                                   │  │ E │  │  lit ici        │    │
│   └───────────────────────────────────┘  └───┘  └─────────────────┘    │
│                                                                         │
│   Mois 1-2 ─────────────────────────────────────────────────── Mois 18 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

                        SEGMENT TEST (3 mois)
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │                    ÉVALUATION FINALE OOS                        │  │
│   │                                                                 │  │
│   │   Jamais vu pendant training (ni TRAIN-TRAIN ni TRAIN-EVAL)    │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│   Mois 19 ─────────────────────────────────────────────────── Mois 21  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Avantage** : Signal 3 actif sans data leakage  
**Inconvénient** : Moins de données pour le training (90% au lieu de 100%)

---

## 4. Architecture

### 4.1 Flux Actuel (Sans Guard)

```
run_full_wfo.py
      │
      ├── train_tqc()
      │       │
      │       └── train_agent.train()
      │               │
      │               └── model.learn(callback=[
      │                       CheckpointCallback,    ✅
      │                       CurriculumCallback,    ✅
      │                       PLOCallbacks,          ✅
      │                       # PAS de OverfittingGuard ❌
      │                   ])
      │
      └── evaluate_segment()  # Évaluation POST-training
```

### 4.2 Flux Proposé (Avec Guard)

```
run_full_wfo.py
      │
      ├── train_tqc()
      │       │
      │       └── train_agent.train()
      │               │
      │               └── model.learn(callback=[
      │                       CheckpointCallback,         ✅
      │                       CurriculumCallback,         ✅
      │                       PLOCallbacks,               ✅
      │                       OverfittingGuardCallbackV2, ✅ NOUVEAU
      │                   ])
      │
      └── evaluate_segment()  # Évaluation POST-training
```

### 4.3 Modifications Requises

| Fichier | Modification |
|---------|--------------|
| `run_full_wfo.py` | Ajouter flag `use_overfitting_guard` dans `WFOConfig` |
| `train_agent.py` | Supporter création de `OverfittingGuardCallbackV2` sans `EvalCallback` |
| `callbacks.py` | Aucune (déjà supporte `eval_callback=None`) |

---

## 5. Implémentation

### 5.1 Modification de WFOConfig

```python
# run_full_wfo.py

@dataclass
class WFOConfig:
    # ... existing fields ...
    
    # === NEW: Overfitting Guard ===
    use_overfitting_guard: bool = True  # Activer OverfittingGuard en WFO
    
    # Guard thresholds (WFO-specific, peut être plus permissif)
    guard_nav_threshold: float = 10.0       # 10x au lieu de 5x (WFO plus long)
    guard_patience: int = 5                 # Plus de patience en WFO
    guard_check_freq: int = 25_000          # Réactivité accrue (~6 semaines de données)
    guard_action_saturation: float = 0.95   # Seuil saturation (95% = policy collapse)
    guard_reward_variance: float = 1e-5     # Seuil variance (très permissif)
```

### 5.2 Modification de train_tqc()

```python
# run_full_wfo.py, dans WFOPipeline.train_tqc()

def train_tqc(self, train_path: str, encoder_path: str, segment_id: int, ...):
    # ... existing code ...
    
    # Configure TQC training
    config = TrainingConfig()
    # ... existing config ...
    
    # NEW: Enable OverfittingGuard in WFO mode
    if self.config.use_overfitting_guard:
        config.use_overfitting_guard = True
        config.guard_nav_threshold = self.config.guard_nav_threshold
        config.guard_patience = self.config.guard_patience
        config.guard_check_freq = self.config.guard_check_freq
        config.guard_action_saturation = self.config.guard_action_saturation
        config.guard_reward_variance = self.config.guard_reward_variance
        # Signal 3 reste désactivé (pas d'eval_data_path)
    
    # Train
    model, train_metrics = train(config, ...)
```

### 5.3 Modification de train_agent.py

```python
# src/training/train_agent.py, dans create_callbacks()

def create_callbacks(config, env, eval_env=None, ...):
    callbacks = []
    
    # ... existing callbacks ...
    
    # OverfittingGuard (v2)
    if getattr(config, 'use_overfitting_guard', False):
        from src.training.callbacks import OverfittingGuardCallbackV2
        
        # En WFO: pas d'EvalCallback, Signal 3 sera désactivé automatiquement
        eval_cb = None
        if eval_env is not None:
            # Mode standard: chercher EvalCallback existant
            eval_cb = next((cb for cb in callbacks if isinstance(cb, EvalCallback)), None)
        
        guard = OverfittingGuardCallbackV2(
            nav_threshold=getattr(config, 'guard_nav_threshold', 5.0),
            patience=getattr(config, 'guard_patience', 3),
            check_freq=getattr(config, 'guard_check_freq', 10_000),
            action_saturation_threshold=getattr(config, 'guard_action_saturation', 0.95),
            reward_variance_threshold=getattr(config, 'guard_reward_variance', 1e-4),
            eval_callback=eval_cb,  # None en WFO = Signal 3 désactivé
            verbose=1
        )
        callbacks.append(guard)
        print(f"  [Guard] OverfittingGuardCallbackV2 enabled (Signal 3: {'ON' if eval_cb else 'OFF'})")
    
    return callbacks
```

### 5.4 Vérification dans callbacks.py

Le callback gère déjà le cas `eval_callback=None` :

```python
# src/training/callbacks.py, ligne ~1475

def _check_train_eval_divergence(self) -> Optional[str]:
    """
    Check if training reward diverges from evaluation reward.
    
    v2.3: Reads from ep_info_buffer (train) and EvalCallback (eval).
    Returns None if eval_callback is not set (WFO mode).
    """
    # Signal désactivé si pas d'EvalCallback
    if self.eval_callback is None:
        return None  # ✅ Déjà géré
    
    # ... rest of the method ...
```

---

## 6. Gestion de l'Arrêt Prématuré (Fail-over)

### 6.1 Problématique

Lorsque `OverfittingGuardCallbackV2` déclenche un arrêt prématuré (ex: NAV > 10x au step 1M sur 90M prévus), que fait le pipeline WFO ?

**Risques identifiés :**
- Trader avec un modèle "immature" ou partiellement entraîné
- Utiliser un modèle dans un état corrompu ou divergent
- Perdre toute information sur la raison de l'échec

### 6.2 Politique de Fail-over Recommandée

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ARBRE DE DÉCISION POST-ARRÊT                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Guard déclenche l'arrêt                                                   │
│           │                                                                 │
│           ▼                                                                 │
│   ┌───────────────────────────────────┐                                     │
│   │ completion_ratio >= 30% ?         │                                     │
│   │ ET checkpoint valide disponible ? │                                     │
│   └───────────────────────────────────┘                                     │
│           │                                                                 │
│     ┌─────┴─────┐                                                           │
│     │           │                                                           │
│    OUI         NON                                                          │
│     │           │                                                           │
│     ▼           ▼                                                           │
│  ┌──────────┐  ┌──────────────────────┐                                     │
│  │ Utiliser │  │ Marquer segment      │                                     │
│  │ dernier  │  │ comme FAILED         │                                     │
│  │ checkpoint│  │                      │                                     │
│  │ (RECOVERED)│ │ Utiliser stratégie  │                                     │
│  └──────────┘  │ de REPLI (Flat/B&H)  │                                     │
│                └──────────────────────┘                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

> **⚠️ Note importante sur `min_completion_ratio`** : Un modèle entraîné sur seulement 10% 
> des données a peu de chances d'être robuste. Le seuil par défaut est fixé à **30%** pour 
> garantir un minimum de convergence avant de considérer un checkpoint comme "récupérable".

### 6.3 Implémentation du Fail-over

```python
# run_full_wfo.py, dans WFOPipeline.run_segment()

def run_segment(self, df_raw, segment, ...):
    # ... training ...
    model, train_metrics = self.train_tqc(...)
    
    # NEW: Check if Guard triggered early stop
    guard_triggered = train_metrics.get('guard_early_stop', False)
    stop_reason = train_metrics.get('guard_stop_reason', None)
    completion_ratio = train_metrics.get('completion_ratio', 1.0)
    
    if guard_triggered:
        logger.warning(f"⚠️ SEGMENT {segment.id} STOPPED EARLY: {stop_reason}")
        logger.warning(f"   Completion ratio: {completion_ratio:.1%}")
        
        # Vérifier si le modèle est suffisamment entraîné pour être récupérable
        last_valid_checkpoint = self._find_last_valid_checkpoint(segment.id)
        can_recover = (
            last_valid_checkpoint 
            and self.config.use_checkpoint_on_failure
            and completion_ratio >= self.config.min_completion_ratio
        )
        
        if can_recover:
            logger.info(f"  → Using last valid checkpoint: {last_valid_checkpoint}")
            model = TQC.load(last_valid_checkpoint)
            train_metrics['used_checkpoint'] = True
            train_metrics['segment_status'] = 'RECOVERED'
        else:
            # Marquer le segment comme FAILED et utiliser stratégie de repli
            logger.warning(f"  → Cannot recover (ratio {completion_ratio:.1%} < {self.config.min_completion_ratio:.0%})")
            logger.warning(f"  → Using fallback strategy: {self.config.fallback_strategy}")
            train_metrics['segment_status'] = 'FAILED'
            train_metrics['fallback_strategy'] = self.config.fallback_strategy
            
            # Utiliser stratégie de repli pour le backtest
            metrics = self._run_fallback_strategy(segment, self.config.fallback_strategy)
            return metrics, train_metrics
    else:
        train_metrics['segment_status'] = 'SUCCESS'
    
    # Continue with evaluation...
    metrics = self.evaluate_segment(model, test_path, ...)
    return metrics, train_metrics


def _run_fallback_strategy(self, segment, strategy: str) -> dict:
    """
    Exécute une stratégie de repli pour les segments FAILED.
    
    Args:
        segment: Le segment WFO
        strategy: 'flat' (pas de trading) ou 'buy_and_hold'
    
    Returns:
        Métriques simulées pour ce segment
    """
    if strategy == 'flat':
        # Pas de trading = returns de 0
        return {
            'segment_id': segment.id,
            'sharpe': 0.0,
            'total_return': 0.0,
            'max_drawdown': 0.0,
            'strategy': 'FLAT (fallback)',
            'is_fallback': True
        }
    elif strategy == 'buy_and_hold':
        # Calculer le B&H sur la période TEST
        test_return = self._calculate_buy_and_hold(segment.test_start, segment.test_end)
        return {
            'segment_id': segment.id,
            'sharpe': None,  # Non applicable pour B&H simple
            'total_return': test_return,
            'max_drawdown': None,
            'strategy': 'BUY_AND_HOLD (fallback)',
            'is_fallback': True
        }
    else:
        raise ValueError(f"Unknown fallback strategy: {strategy}")
```

### 6.4 Logging de la Raison d'Arrêt (Télémétrie)

Le pipeline doit stocker **pourquoi** chaque segment s'est arrêté pour l'analyse post-mortem.
Cette information doit être présente dans le fichier de résultats final (`wfo_results.json`).

```python
# Structure du log d'arrêt par segment
segment_log = {
    'segment_id': 5,
    'status': 'FAILED',           # SUCCESS | RECOVERED | FAILED
    'stop_reason': 'Signal 1: NAV > 10.0 (observed: 15.3)',
    'stopped_at_step': 1_234_567,
    'total_planned_steps': 90_000_000,
    'completion_ratio': 0.0137,   # 1.37%
    'checkpoint_used': None,      # ou 'segment_05_step_1000000.zip'
    'fallback_strategy': 'flat',  # si FAILED
    'timestamp': '2026-01-19T14:32:15Z'
}
```

**Fichier de résultats final (`wfo_results.json`)** :

```python
# Le fichier doit contenir stop_reason pour permettre le filtrage post-WFO
wfo_results = {
    'config': { ... },
    'segments': [
        {
            'id': 0,
            'status': 'SUCCESS',
            'stop_reason': None,
            'metrics': { 'sharpe': 1.2, 'total_return': 0.15, ... }
        },
        {
            'id': 1,
            'status': 'FAILED',
            'stop_reason': 'Signal 4: Action saturation > 0.95',
            'fallback_strategy': 'flat',
            'metrics': { 'sharpe': 0.0, 'total_return': 0.0, 'is_fallback': True }
        },
        # ...
    ],
    'summary': {
        'total_segments': 10,
        'successful': 8,
        'recovered': 1,
        'failed': 1,
        'aggregate_sharpe': 0.95,
        'aggregate_sharpe_excluding_failed': 1.05  # Pour comparaison
    }
}
```

Cette structure permet de :
- Filtrer les résultats post-WFO (ex: "Performance sans les segments crashés")
- Identifier les patterns de failure (quels signaux déclenchent le plus souvent)
- Ajuster les seuils du Guard si trop de faux positifs

### 6.5 Récapitulatif des Statuts de Segment

| Statut | Description | Action Évaluation |
|--------|-------------|-------------------|
| `SUCCESS` | Training complet sans intervention | Évaluation normale sur TEST |
| `RECOVERED` | Arrêt Guard, checkpoint valide utilisé (ratio ≥ 30%) | Évaluation avec avertissement |
| `FAILED` | Arrêt Guard, pas de checkpoint ou ratio < 30% | Stratégie de repli (Flat/B&H) |

### 6.6 Configuration Fail-over

```python
@dataclass
class WFOConfig:
    # ... existing fields ...
    
    # Fail-over configuration
    use_checkpoint_on_failure: bool = True   # Utiliser checkpoint si Guard arrête
    min_completion_ratio: float = 0.30       # Min 30% du training pour être "récupérable"
    checkpoint_freq: int = 1_000_000         # Fréquence des checkpoints (pour recovery)
    fallback_strategy: str = 'flat'          # 'flat' ou 'buy_and_hold' pour segments FAILED
```

> **⚠️ Pourquoi 30% et pas 10% ?** Un modèle TQC complexe entraîné sur seulement 10% des 
> données (9M steps sur 90M) n'a pas eu le temps de converger correctement. À 30% (27M steps), 
> le modèle a généralement passé les phases critiques d'exploration initiale et possède une 
> policy minimalement cohérente.

---

## 7. Continuité WFO (Chain of Inheritance)

### 7.1 Problématique Critique

En WFO standard, le **Segment N+1** initialise souvent ses poids à partir de l'état final du **Segment N** (warm start). Cela permet un apprentissage continu et une adaptation progressive aux régimes de marché.

**La Section 6 définit ce qui se passe pour les *résultats* du Segment N en cas d'échec, mais pas ce qui se passe pour l'*initialisation* du Segment N+1.**

### 7.2 Scénario de Risque

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    RUPTURE DE LA CHAÎNE WFO                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Segment N-1          Segment N              Segment N+1                   │
│   ┌─────────┐          ┌─────────┐           ┌─────────┐                   │
│   │ SUCCESS │ ──────▶  │ FAILED  │ ──────▶   │    ?    │                   │
│   │         │  init    │ (ratio  │   init    │         │                   │
│   │ model   │  from    │  <30%)  │   from    │ CRASH?  │                   │
│   │  OK     │  N-1     │ no model│   ???     │ ou      │                   │
│   └─────────┘          └─────────┘           │ cold    │                   │
│                                              │ start?  │                   │
│                                              └─────────┘                   │
│                                                                             │
│   ❌ Si N+1 essaie de charger le modèle N : CRASH (fichier inexistant)     │
│   ❌ Si N+1 recommence de zéro : Perte de l'apprentissage continu          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.3 Solution : Rollback d'Initialisation

Le pipeline doit maintenir une variable `last_successful_model` qui pointe toujours vers le dernier modèle valide (SUCCESS ou RECOVERED).

```python
# run_full_wfo.py, dans WFOPipeline.run_all_segments()

def run_all_segments(self, df_raw, segments):
    """
    Exécute tous les segments WFO avec gestion de l'héritage des poids.
    """
    all_results = []
    last_successful_model_path = None  # Track du dernier modèle valide
    
    for i, segment in enumerate(segments):
        logger.info(f"═══ Segment {i}/{len(segments)-1} ═══")
        
        # Déterminer le modèle d'initialisation
        if i == 0:
            # Premier segment : cold start ou modèle pré-entraîné
            init_model_path = self.config.pretrained_model_path
        else:
            # Segments suivants : utiliser le dernier modèle valide
            init_model_path = last_successful_model_path
            
            if init_model_path is None:
                logger.warning(f"⚠️ No valid model from previous segments, using cold start")
        
        # Exécuter le segment
        metrics, train_metrics = self.run_segment(
            df_raw, segment, 
            init_model_path=init_model_path,
            ...
        )
        
        # Mettre à jour le tracking du dernier modèle valide
        segment_status = train_metrics.get('segment_status', 'SUCCESS')
        
        if segment_status in ['SUCCESS', 'RECOVERED']:
            # Ce segment a produit un modèle valide
            last_successful_model_path = self._get_segment_model_path(segment.id)
            logger.info(f"  ✓ Updated last_successful_model: {last_successful_model_path}")
        else:
            # Segment FAILED : on garde l'ancien last_successful_model
            logger.warning(f"  ✗ Segment FAILED, keeping previous model for inheritance")
            
            # IMPORTANT: Nettoyer les checkpoints pourris de ce segment
            self._cleanup_failed_segment_checkpoints(segment.id)
        
        all_results.append({
            'segment': segment,
            'metrics': metrics,
            'train_metrics': train_metrics,
            'init_model_used': init_model_path
        })
    
    return all_results
```

### 7.4 Diagramme de la Chaîne Corrigée

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CHAÎNE WFO AVEC ROLLBACK                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Segment 0          Segment 1          Segment 2          Segment 3       │
│   ┌─────────┐        ┌─────────┐        ┌─────────┐        ┌─────────┐     │
│   │ SUCCESS │───────▶│ SUCCESS │───────▶│ FAILED  │        │ SUCCESS │     │
│   │         │ model  │         │ model  │         │        │         │     │
│   │ model_0 │   0    │ model_1 │   1    │ (crash) │        │ model_3 │     │
│   └─────────┘        └─────────┘        └─────────┘        └─────────┘     │
│                            │                  │                  ▲         │
│                            │                  │                  │         │
│                            └──────────────────┴──────────────────┘         │
│                                      │                                      │
│                            last_successful_model = model_1                  │
│                            (utilisé pour init Segment 3)                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.5 Nettoyage des Checkpoints Pourris

Quand un segment est marqué `FAILED`, les checkpoints générés avant l'arrêt sont potentiellement corrompus ou dans un état instable. Ils doivent être supprimés pour :
- Éviter de saturer le disque inutilement
- Empêcher une réutilisation accidentelle

```python
def _cleanup_failed_segment_checkpoints(self, segment_id: int):
    """
    Supprime les checkpoints d'un segment FAILED.
    
    Ne supprime PAS le modèle final si le segment est RECOVERED 
    (car ce modèle est valide et utilisable).
    """
    checkpoint_dir = os.path.join(self.output_dir, f"segment_{segment_id:02d}", "checkpoints")
    
    if os.path.exists(checkpoint_dir):
        # Lister tous les fichiers .zip dans le dossier
        checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.zip"))
        
        for ckpt in checkpoints:
            logger.info(f"  🗑️ Removing failed checkpoint: {os.path.basename(ckpt)}")
            os.remove(ckpt)
        
        logger.info(f"  Cleaned up {len(checkpoints)} checkpoints from failed segment {segment_id}")
```

### 7.6 Configuration de l'Héritage

```python
@dataclass
class WFOConfig:
    # ... existing fields ...
    
    # Chain of Inheritance
    use_warm_start: bool = True              # Hériter des poids du segment précédent
    pretrained_model_path: str = None        # Modèle de départ pour Segment 0
    cleanup_failed_checkpoints: bool = True  # Supprimer les checkpoints des segments FAILED
```

---

## 8. Configuration

### 8.1 Paramètres Recommandés pour WFO

| Paramètre | Valeur Standard | Valeur WFO | Raison |
|-----------|-----------------|------------|--------|
| `nav_threshold` | 5.0 | 10.0 | WFO = 90M steps, plus de temps pour accumuler |
| `patience` | 3 | 5 | Réduire faux positifs sur long training |
| `check_freq` | 10,000 | **25,000** | Réactivité accrue sans overhead notable |
| `action_saturation` | 0.95 | **0.95** | Si 95% bang-bang, le modèle ne fait plus de finesse |
| `reward_variance` | 1e-4 | 1e-5 | Plus permissif (détecte seulement "mort cérébrale") |

> **⚠️ Note sur `check_freq`** : La fréquence de 25,000 steps offre un bon compromis entre 
> réactivité et performance. Avec 18 mois de données (~576k minutes si 1 step = 1 minute),
> 25k steps représente environ 6 semaines de données. Si le modèle diverge, on le détecte
> rapidement sans attendre 3 mois (ce qui serait le cas avec 50k).
> 
> **Calcul** : 25k / 576k ≈ 4.3% d'une époque, ce qui est inférieur à la limite de 10% recommandée.

### 8.2 CLI Arguments

```bash
# Activer (défaut)
python scripts/run_full_wfo.py --segment 0

# Désactiver explicitement
python scripts/run_full_wfo.py --segment 0 --no-overfitting-guard

# Personnaliser les seuils
python scripts/run_full_wfo.py --segment 0 \
    --guard-nav-threshold 8.0 \
    --guard-patience 4
```

---

## 9. Tests de Validation

### 9.1 Test Unitaire : Guard Sans EvalCallback

```python
# tests/test_overfitting_guard_wfo.py

def test_guard_without_eval_callback():
    """Verify Signal 3 is gracefully disabled when no EvalCallback."""
    from src.training.callbacks import OverfittingGuardCallbackV2
    
    guard = OverfittingGuardCallbackV2(
        eval_callback=None,  # WFO mode
        verbose=0
    )
    
    # Signal 3 should return None (no violation, no error)
    result = guard._check_train_eval_divergence()
    assert result is None, "Signal 3 should be disabled without EvalCallback"
```

### 9.2 Test Intégration : WFO avec Guard

```python
def test_wfo_segment_with_guard():
    """Run a mini WFO segment with OverfittingGuard enabled."""
    from scripts.run_full_wfo import WFOPipeline, WFOConfig
    
    config = WFOConfig()
    config.use_overfitting_guard = True
    config.tqc_timesteps = 10_000  # Mini run
    
    pipeline = WFOPipeline(config)
    
    # Should not crash
    metrics, train_metrics = pipeline.run_segment(df_raw, segment, use_batch_env=True)
    
    # Guard metrics should be logged
    assert 'guard_violations' in train_metrics or True  # Optionnel
```

### 9.3 Test Early Stopping

```python
def test_guard_triggers_early_stop():
    """Verify Guard can stop training when NAV explodes."""
    # Create mock env with artificially high NAV
    # ...
    
    guard = OverfittingGuardCallbackV2(
        nav_threshold=2.0,  # Trigger at 2x
        patience=1
    )
    
    # Simulate steps with high NAV
    # ...
    
    # Should return False (stop training)
    assert guard._on_step() == False
```

### 9.4 Test Fail-over avec Checkpoint

```python
def test_failover_uses_last_checkpoint():
    """Verify pipeline uses last valid checkpoint when Guard stops training."""
    from scripts.run_full_wfo import WFOPipeline, WFOConfig
    
    config = WFOConfig()
    config.use_overfitting_guard = True
    config.use_checkpoint_on_failure = True
    config.guard_nav_threshold = 1.5  # Trigger rapidement
    config.checkpoint_freq = 1000     # Checkpoints fréquents pour le test
    
    pipeline = WFOPipeline(config)
    
    # Run segment qui va déclencher le Guard
    metrics, train_metrics = pipeline.run_segment(df_volatile, segment, ...)
    
    # Vérifier le comportement fail-over
    assert train_metrics['guard_early_stop'] == True
    assert train_metrics['segment_status'] in ['RECOVERED', 'FAILED']
    
    if train_metrics['segment_status'] == 'RECOVERED':
        assert 'used_checkpoint' in train_metrics
        assert train_metrics['used_checkpoint'] == True
```

### 9.5 Test Log de Raison d'Arrêt

```python
def test_stop_reason_logged():
    """Verify stop reason is properly logged for post-mortem analysis."""
    # ... setup ...
    
    metrics, train_metrics = pipeline.run_segment(...)
    
    if train_metrics.get('guard_early_stop'):
        assert 'guard_stop_reason' in train_metrics
        assert train_metrics['guard_stop_reason'] is not None
        # Reason should be descriptive
        assert 'Signal' in train_metrics['guard_stop_reason']
```

### 9.6 Test Chain of Inheritance

```python
def test_chain_of_inheritance_after_failure():
    """Verify Segment N+1 uses model from N-1 when N fails."""
    from scripts.run_full_wfo import WFOPipeline, WFOConfig
    
    config = WFOConfig()
    config.use_overfitting_guard = True
    config.use_warm_start = True
    config.guard_nav_threshold = 1.5  # Trigger failure on segment 1
    
    pipeline = WFOPipeline(config)
    
    # Run 3 segments where segment 1 will fail
    results = pipeline.run_all_segments(df_raw, segments[:3])
    
    # Segment 0: SUCCESS
    assert results[0]['train_metrics']['segment_status'] == 'SUCCESS'
    
    # Segment 1: FAILED (triggered by Guard)
    assert results[1]['train_metrics']['segment_status'] == 'FAILED'
    
    # Segment 2: Should have initialized from Segment 0's model (not Segment 1)
    assert results[2]['init_model_used'] == results[0]['model_path']
    assert results[2]['train_metrics']['segment_status'] in ['SUCCESS', 'RECOVERED']
```

### 9.7 Test Cleanup de Checkpoints

```python
def test_failed_segment_checkpoints_cleaned():
    """Verify checkpoints are deleted for FAILED segments."""
    # ... setup with segment that will fail ...
    
    pipeline.run_segment(df_volatile, segment, ...)
    
    # Checkpoints directory should be empty or not exist
    checkpoint_dir = os.path.join(pipeline.output_dir, f"segment_{segment.id:02d}", "checkpoints")
    
    if os.path.exists(checkpoint_dir):
        remaining_files = os.listdir(checkpoint_dir)
        assert len(remaining_files) == 0, f"Found orphan checkpoints: {remaining_files}"
```

---

## 10. Références

### 10.1 Fichiers du Projet

| Fichier | Description |
|---------|-------------|
| `src/training/callbacks.py` | Définition de `OverfittingGuardCallbackV2` |
| `scripts/run_full_wfo.py` | Pipeline WFO à modifier |
| `src/training/train_agent.py` | Création des callbacks |
| `docs/OVERFITTING_GUARD_V2.md` | Spécification technique du callback |

### 10.2 Documentation Liée

- `docs/OVERFITTING_GUARD_V2.md` - Détails des 5 signaux
- `docs/EVAL_DATA_SPLIT.md` - Split train/eval (mode standard)
- `docs/PLO_ADAPTIVE_PENALTIES.md` - Callbacks PLO (compatibles)

---

## Annexe A : Checklist d'Implémentation

### Configuration
- [ ] Ajouter `use_overfitting_guard` dans `WFOConfig`
- [ ] Ajouter paramètres `guard_*` dans `WFOConfig` (notamment `guard_check_freq=25_000`)
- [ ] Ajouter paramètres fail-over dans `WFOConfig` (`min_completion_ratio=0.30`, `fallback_strategy`)
- [ ] Ajouter paramètres Chain of Inheritance (`use_warm_start`, `pretrained_model_path`, `cleanup_failed_checkpoints`)
- [ ] Ajouter arguments CLI (`--no-overfitting-guard`, `--guard-*`, `--fallback-strategy`)

### Implémentation Core
- [ ] Modifier `train_tqc()` pour passer les paramètres Guard
- [ ] Modifier `train_agent.py` pour créer le callback
- [ ] Implémenter la logique fail-over dans `run_segment()`
- [ ] Ajouter `_find_last_valid_checkpoint()`
- [ ] Ajouter `_run_fallback_strategy()` (Flat et Buy & Hold)

### Chain of Inheritance (CRITIQUE)
- [ ] Implémenter `run_all_segments()` avec tracking de `last_successful_model_path`
- [ ] Passer `init_model_path` à `run_segment()` pour le warm start
- [ ] Ajouter `_cleanup_failed_segment_checkpoints()` pour le nettoyage disque
- [ ] Logger `init_model_used` dans les résultats de chaque segment

### Télémétrie
- [ ] S'assurer que `train_metrics` contient `guard_early_stop`, `guard_stop_reason`, `completion_ratio`
- [ ] Ajouter `stop_reason` dans `wfo_results.json` pour chaque segment
- [ ] Ajouter `aggregate_sharpe_excluding_failed` dans le summary
- [ ] Logger `init_model_used` pour tracer l'héritage des poids

### Tests
- [ ] Tester sur un segment court (10k steps)
- [ ] Tester le fail-over avec `min_completion_ratio` (vérifier seuil 30%)
- [ ] Tester les deux stratégies de repli (Flat, B&H)
- [ ] Tester la Chain of Inheritance après un segment FAILED
- [ ] Tester le nettoyage des checkpoints pourris
- [ ] Valider les logs TensorBoard (`guard/*`)

### Documentation
- [ ] Documenter dans `IMPROVEMENTS.md`

---

## Annexe B : Option Avancée - Split Intra-Train

> **⚠️ Rappel : Cette option est DÉCONSEILLÉE** (voir section 3.2)

Si Signal 3 est requis en WFO malgré tout, voici l'approche :

### B.1 Modification de preprocess_segment()

```python
def preprocess_segment(self, df_raw, segment, intra_train_split: float = 0.9):
    """
    Args:
        intra_train_split: Fraction of TRAIN for actual training (rest = holdout eval)
    """
    # ... existing preprocessing ...
    
    if intra_train_split < 1.0:
        # Split TRAIN into TRAIN-TRAIN and TRAIN-EVAL
        split_idx = int(len(train_df) * intra_train_split)
        purge = 50  # Purge window
        
        train_train_df = train_df.iloc[:split_idx]
        train_eval_df = train_df.iloc[split_idx + purge:]
        
        # Save both
        train_train_path = os.path.join(data_dir, "train_train.parquet")
        train_eval_path = os.path.join(data_dir, "train_eval.parquet")
        
        train_train_df.to_parquet(train_train_path)
        train_eval_df.to_parquet(train_eval_path)
        
        return train_train_path, train_eval_path, test_path
```

### B.2 Modification de train_tqc()

```python
def train_tqc(self, train_path, train_eval_path, ...):
    config = TrainingConfig()
    config.data_path = train_path           # TRAIN-TRAIN (90%)
    config.eval_data_path = train_eval_path  # TRAIN-EVAL (10%)
    
    # EvalCallback sera créé, Signal 3 actif
```

### B.3 Trade-offs

| Aspect | Sans Split Intra-Train | Avec Split Intra-Train |
|--------|------------------------|------------------------|
| Données training | 100% du TRAIN | 90% du TRAIN |
| Signal 3 | ❌ Désactivé | ✅ Actif |
| Complexité | Simple | Moyenne |
| Risque leakage | Aucun | Aucun (TRAIN-EVAL ⊂ TRAIN) |

---

## Annexe C : Métriques TensorBoard Attendues

Avec Guard actif, les métriques suivantes apparaîtront dans TensorBoard :

```
guard/
├── active_signals      # Nombre de signaux en violation (0-5)
├── nav_max             # NAV maximum observé
├── weight_cv           # Coefficient de variation des poids
├── weight_delta        # Delta moyen des poids
├── action_saturation   # Ratio d'actions saturées
├── reward_variance     # Variance des rewards
├── reward_cv           # CV des rewards
└── violations/
    ├── nav             # Compteur violations Signal 1
    ├── weight          # Compteur violations Signal 2
    ├── saturation      # Compteur violations Signal 4
    └── variance        # Compteur violations Signal 5
```

---

*Fin de la spécification technique*

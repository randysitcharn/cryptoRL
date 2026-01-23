# Liste des Callbacks TensorBoard

Ce document liste tous les callbacks et mécanismes de logging TensorBoard dans le projet cryptoRL.

## 📋 Vue d'ensemble

Le projet utilise une architecture unifiée pour le logging TensorBoard :
1. **UnifiedMetricsCallback** : Callback principal utilisant uniquement `logger.record()` / `logger.record_mean()` (automatiquement loggés par SB3)
2. **SummaryWriter direct** : Utilisé uniquement pour HMM/MAE training (hors scope RL)

**Architecture "Single Source of Truth"** : Toutes les métriques RL passent par le logger SB3 pour garantir la synchronisation des timesteps et éviter les doublons.

---

## 🔄 Callbacks SB3 (via logger.record)

Ces callbacks utilisent le système de logging de Stable-Baselines3, qui écrit automatiquement dans TensorBoard.

### 1. **UnifiedMetricsCallback** ⭐ NOUVEAU
**Fichier** : `src/training/callbacks.py` (lignes 134-465)

**Description** : Callback unifié remplaçant `TensorBoardStepCallback`, `StepLoggingCallback`, et `DetailTensorboardCallback`. Centralise tout le logging via `logger.record()` avec des namespaces standardisés.

**Architecture optimisée** :
- **Métriques légères** (chaque step) : Buffer pour lissage (NAV, position, drawdown)
- **Métriques lourdes** (uniquement à `log_freq`) : Gradients, Q-values TQC, `get_global_metrics()`

**Métriques loggées (Épurées - Signal uniquement)** :

**Portfolio** :
- `portfolio/nav` - Portfolio NAV
- `portfolio/position_pct` - Position en pourcentage

**Risk** :
- `risk/max_drawdown` - Max drawdown en pourcentage

**Rewards (Agrégées)** :
- `rewards/pnl_component` - Composante PnL du reward (signal principal)
- `rewards/total_penalties` - Somme agrégée de toutes les pénalités (churn + smoothness + downside_risk)
- ❌ Composantes individuelles supprimées (trop granulaires)

**Strategy** :
- `strategy/churn_ratio` - Ratio churn/PnL par épisode

**Debug TQC (Essentielles pour diagnostic Gamma)** :
- `debug/q_values_mean` - Moyenne des Q-values TQC
- `debug/q_values_std` - Écart-type des Q-values TQC
- `debug/grad_actor_norm` - Norme L2 des gradients de l'actor
- `debug/grad_critic_norm` - Norme L2 des gradients du critic

**Fréquence** :
- Métriques légères : Buffer à chaque step, loggées à `log_freq`
- Métriques lourdes : Calculées et loggées uniquement à `log_freq` (performance)

**Optimisations** :
- Utilise `logger.record_mean()` pour les métriques bufferisées (courbes lisses)
- Calcul des gradients uniquement à `log_freq` (évite 10-20% de ralentissement)
- Polling direct GPU via `get_global_metrics()` (pas via infos)
- Monitoring TQC intégré pour diagnostic Gamma (Q-values)

**Console logging** : Optionnel (flag `verbose`), format identique à l'ancien `StepLoggingCallback`

**Métriques supprimées (bruit)** :
- ❌ `portfolio/nav_std` (redondant avec max_drawdown)
- ❌ `strategy/price` (bruit visuel, prix normalisés)
- ❌ `rewards/churn_cost`, `rewards/smoothness`, `rewards/downside_risk` (agrégées en `total_penalties`)
- ❌ `debug/q_values_min`, `debug/q_values_max` (outliers, std suffit)
- ❌ `time/fps_live` (si doublon avec SB3)

---

### 2. **ThreePhaseCurriculumCallback**
**Fichier** : `src/training/callbacks.py` (lignes 618-684)

**Métriques loggées** :
- `curriculum/phase` - Phase actuelle (1, 2, ou 3)
- `curriculum/progress` - Progression totale (0.0 à 1.0)
- `curriculum/lambda` - Valeur de `curriculum_lambda` de l'environnement
- `observation_noise/effective_scale` - Échelle effective du bruit d'observation

**Fréquence** : À chaque step

**Phases** :
- Phase 1 : 0-15% (Discovery)
- Phase 2 : 15-75% (Discipline)
- Phase 3 : 75-100% (Refinement)

---

### 3. **OverfittingGuardCallbackV2**
**Fichier** : `src/training/callbacks.py` (lignes 868-1318)

**Métriques loggées** :
- `overfit/max_nav_ratio` - Ratio NAV max / NAV initial
- `overfit/weight_delta` - Delta moyen des poids
- `overfit/weight_cv` - Coefficient de variation des poids
- `overfit/train_eval_divergence` - Divergence train/eval
- `overfit/action_saturation` - Ratio d'actions saturées
- `overfit/reward_variance` - Variance des rewards
- `overfit/reward_cv` - Coefficient de variation des rewards
- `overfit/violations_{name}` - Compteurs de violations par signal
- `overfit/active_signals` - Nombre de signaux actifs

**Fréquence** : Tous les `check_freq` steps (défaut: 10,000)

**Signaux de détection** :
1. NAV threshold (retours irréalistes)
2. Weight stagnation (convergence/collapse)
3. Train/Eval divergence (overfitting classique)
4. Action saturation (collapse de la politique)
5. Reward variance (mémorisation)

---

### 4. **ModelEMACallback**
**Fichier** : `src/training/callbacks.py` (lignes 1324-1500)

**Métriques loggées** :
- `ema/weight_diff_l2` - Différence L2 entre poids actuels et EMA

**Fréquence** : Tous les 10,000 steps

**Fonctionnalité** : Maintient une copie EMA (Exponential Moving Average) des poids du modèle pour éviter l'overfitting.

---

## 📊 SummaryWriter Direct (Hors Scope RL)

Ces mécanismes utilisent `SummaryWriter` directement pour des cas spécifiques (HMM, MAE) qui ne sont pas dans le scope du training RL.

### 5. **train_foundation.py (MAE Training)**
**Fichier** : `src/training/train_foundation.py` (lignes 466-551)

**Métriques loggées** :
- `loss/train_total` - Loss totale d'entraînement
- `loss/train_recon` - Loss de reconstruction
- `loss/train_aux` - Loss auxiliaire (si supervised)
- `loss/val_total` - Loss totale de validation
- `loss/val_recon` - Loss de reconstruction (validation)
- `loss/val_aux` - Loss auxiliaire (validation)
- `loss/best_val` - Meilleure loss de validation
- `time/epoch_seconds` - Temps par epoch
- `accuracy/val_direction` - Précision de direction (si supervised)
- `hparam/*` - Hyperparamètres (via `add_hparams`)

**Fréquence** : À chaque epoch

**Contexte** : Entraînement du modèle MAE (Masked Autoencoder).

---

### 6. **DataManager.fit_predict (HMM Training)**
**Fichier** : `src/data_engineering/manager.py` (lignes 460-689)

**Métriques loggées** :
- `hmm/log_likelihood` - Log-likelihood par itération EM
- `hmm/log_likelihood_delta` - Delta de log-likelihood
- `hmm/final/converged` - Statut de convergence (0/1)
- `hmm/final/n_iterations` - Nombre d'itérations EM
- `hmm/final/kmeans_inertia` - Inertie K-Means
- `hmm/final/log_likelihood` - Log-likelihood final
- `hmm/final/transmat_entropy` - Entropie de la matrice de transition
- `hmm/final/transmat_diag_avg` - Moyenne de la diagonale (persistance)
- `hmm/final/transition_penalty` - Pénalité de transition appliquée
- `hmm/state_{i}/annual_return_pct` - Return annuel par état
- `hmm/state_{i}/distribution_pct` - Distribution en pourcentage par état

**Fréquence** : 
- Par itération EM pour `log_likelihood`
- Par segment WFO pour les métriques finales

**Contexte** : Entraînement du HMM (Hidden Markov Model) pour la détection de régimes.

---

### 7. **run_full_wfo.py (Evaluation)**
**Fichier** : `scripts/run_full_wfo.py` (lignes 997-1293)

**Métriques loggées** :

**Évaluation Ensemble** :
- `eval_ensemble/sharpe` - Ratio de Sharpe
- `eval_ensemble/pnl_pct` - PnL en pourcentage
- `eval_ensemble/max_drawdown` - Max drawdown
- `eval_ensemble/avg_agreement` - Accord moyen entre modèles
- `eval_ensemble/avg_std` - Écart-type moyen des prédictions
- `eval_ensemble/alpha` - Alpha (retour ajusté au risque)

**Évaluation Standard** :
- `eval/sharpe` - Ratio de Sharpe
- `eval/pnl_pct` - PnL en pourcentage
- `eval/max_drawdown` - Max drawdown
- `eval/total_trades` - Nombre total de trades
- `eval/circuit_breakers` - Nombre de circuit breakers déclenchés
- `eval/final_nav` - NAV final

**Fréquence** : Par segment WFO (segment_id comme step)

**Contexte** : Évaluation des modèles pendant le Walk-Forward Optimization.

---

## 📝 Métriques SB3 Standard

Stable-Baselines3 log automatiquement ces métriques (sans callback personnalisé) :

- `rollout/ep_rew_mean` - Reward moyen par épisode
- `rollout/ep_len_mean` - Longueur moyenne d'épisode
- `train/actor_loss` - Loss de l'actor
- `train/critic_loss` - Loss du critic
- `train/ent_coef` - Coefficient d'entropie
- `train/ent_coef_loss` - Loss du coefficient d'entropie
- `train/learning_rate` - Taux d'apprentissage
- `train/n_updates` - Nombre de mises à jour
- `train/policy_gradient_loss` - Loss du gradient de politique
- `train/value_loss` - Loss de la valeur
- `time/fps` - FPS (peut être 0 avec BatchCryptoEnv, d'où `time/fps_live`)

---

## 🎯 Utilisation dans le Code

### Création des Callbacks

Les callbacks sont créés dans `src/training/train_agent.py` via `create_callbacks()` :

```python
callbacks = [
    UnifiedMetricsCallback(log_freq=config.log_freq, verbose=config.verbose),  # ⭐ NOUVEAU
    ThreePhaseCurriculumCallback(total_timesteps=config.total_timesteps),
    EvalCallbackWithNoiseControl(...),  # Pas de logging TensorBoard direct
    RotatingCheckpointCallback(...),    # Pas de logging TensorBoard
    OverfittingGuardCallbackV2(...),      # Si activé
    ModelEMACallback(...),               # Si activé
]
```

**Migration** : `UnifiedMetricsCallback` remplace `StepLoggingCallback` et `DetailTensorboardCallback`.

### Configuration TensorBoard

Les chemins de logs sont configurés dans :
- `src/config/training.py` : `tensorboard_log` (défaut: `"logs/tensorboard_tqc/"`)
- `src/config/base.py` : `tensorboard_log` (défaut: `"logs/tensorboard/"`)

### Visualisation

Pour visualiser les logs TensorBoard :

```bash
tensorboard --logdir logs/wfo --port 8081
```

---

## 📊 Résumé par Catégorie

| Catégorie | Callback | Métriques Principales |
|-----------|----------|----------------------|
| **Unifié** | UnifiedMetricsCallback ⭐ | Portfolio, Risk, Rewards, Strategy, Debug TQC |
| **Curriculum** | ThreePhaseCurriculumCallback | Phase, lambda, noise |
| **Overfitting** | OverfittingGuardCallbackV2 | 5 signaux de détection |
| **EMA** | ModelEMACallback | Différence poids |
| **HMM** | DataManager.fit_predict | Convergence, états, transitions |
| **MAE** | train_foundation.py | Loss, accuracy |
| **Evaluation** | run_full_wfo.py | Sharpe, PnL, drawdown |

---

## 🔍 Notes Importantes

1. **Architecture unifiée** : `UnifiedMetricsCallback` utilise uniquement `logger.record()`, garantissant la synchronisation avec SB3 et l'absence de doublons

2. **Fréquences** : 
   - `log_freq` contrôle la fréquence des métriques lourdes dans `UnifiedMetricsCallback`
   - `check_freq` contrôle la fréquence de `OverfittingGuardCallbackV2`

3. **Performance** : 
   - Métriques légères : Buffer à chaque step, loggées à `log_freq`
   - Métriques lourdes (gradients, Q-values) : Calculées uniquement à `log_freq` (évite 10-20% de ralentissement)
   - Les buffers sont limités (deque avec maxlen) pour éviter OOM

4. **WFO** : Les métriques HMM et d'évaluation utilisent `segment_id` comme step pour créer des courbes par segment.

5. **Monitoring TQC** : Le monitoring des Q-values est intégré dans `UnifiedMetricsCallback` pour le diagnostic Gamma (essentiel pour détecter si gamma est trop petit)

6. **Métriques épurées** : Seules les métriques vitales sont loggées (signal uniquement, pas de bruit). Les composantes individuelles de pénalités sont agrégées en `total_penalties`.

---

## 📝 Migration depuis l'Ancienne Architecture

**Callbacks supprimés** :
- ❌ `TensorBoardStepCallback` (utilisait SummaryWriter directement, créait des doublons)
- ❌ `StepLoggingCallback` (fonctionnalité fusionnée dans UnifiedMetricsCallback)
- ❌ `DetailTensorboardCallback` (fonctionnalité fusionnée dans UnifiedMetricsCallback)
- ❌ `CurriculumFeesCallback` (déjà obsolète, remplacé par ThreePhaseCurriculumCallback)

**Changements de namespaces** :
- `custom/nav` → `portfolio/nav`
- `custom/position` → `portfolio/position_pct`
- `custom/max_drawdown` → `risk/max_drawdown`
- `internal/reward/pnl_component` → `rewards/pnl_component`
- `internal/reward/churn_cost` + `internal/reward/smoothness` + `internal/reward/downside_risk` → `rewards/total_penalties` (agrégé)
- `grad/actor_norm` → `debug/grad_actor_norm`
- `grad/critic_norm` → `debug/grad_critic_norm`
- Nouveau : `debug/q_values_mean`, `debug/q_values_std` (monitoring TQC)

**Compatibilité** : La méthode `get_training_metrics()` est conservée dans `UnifiedMetricsCallback` pour compatibilité avec le code existant.

---

**Dernière mise à jour** : 2026-01-23 (Migration vers architecture unifiée)

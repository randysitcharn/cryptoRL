# Audit projet complet – Points à couvrir

**Date** : 2026-01-22  
**Objectif** : Checklist exhaustive des domaines à auditer pour une revue complète de CryptoRL.  
**Références** : Audits existants (`DATA_PIPELINE`, `AUDIT_MODELES_RL_RESULTATS`, `AUDIT_MORL`, etc.)

---

## 1. Vue d’ensemble

### 1.1 Déjà audité (à resynthétiser / suivi des correctifs)

| Domaine | Document | Score / statut | Correctifs |
|--------|----------|----------------|------------|
| **Data pipeline** | `DATA_PIPELINE_AUDIT_REPORT.md` | 7.5/10 | P0 corrigés (RobustScaler, purge 720h, embargo). Funding synthétique désactivé. |
| **Modèles RL** | `AUDIT_MODELES_RL_RESULTATS.md` + `MASTER_PLAN_AUDIT_MODELES_RL.md` | TQC 8/10, BatchCryptoEnv 8/10, Callbacks 7/10, Ensemble 8/10 | Spectral norm, incohérences config documentées. |
| **MORL** | `AUDIT_MORL.md` | Architecture validée | w_cost [0, 0.25, 0.5, 0.75, 1] en WFO. |
| **Observation noise** | `AUDIT_OBSERVATION_NOISE.md` | - | Dynamic noise implémenté. |
| **Corrections SOTA** | `AUDIT_CORRECTIONS_PROPOSEES.md` | Validé | Vol scaling train/eval, MORL, distributional shift. |

### 1.2 À auditer ou à ré-auditer

Les sections suivantes détaillent, par domaine, les **points à auditer** (nouveaux ou complémentaires aux audits existants).

---

## 2. Data & feature engineering

| # | Point | Priorité | Fichiers / zone | Statut audit |
|---|------|----------|------------------|--------------|
| D1 | **RobustScaler / purge / embargo** – Vérifier que les correctifs P0 sont bien utilisés partout (WFO, splitter, `run_full_wfo`) | P0 | `splitter.py`, `manager.py`, `run_full_wfo.py` | À vérifier |
| D2 | **Purge / embargo en WFO** – Cohérence `purge_window` (720), `embargo_window` (24), et respect des gaps entre train / eval / test | P0 | `run_full_wfo.py`, `splitter.py`, `constants.py` | Ouvert (P3.2) |
| D3 | **HMM** – Fit sur tout le segment vs expanding window (Lopez de Prado). Biais rétrospectif des régimes | P1 | `manager.py`, `run_full_wfo.py` | Nuancé (audit data) |
| D4 | **FFD** – Implémentation Lopez de Prado, recherche de `d` optimal, usage des features dérivées | P2 | `features.py` | Couvert (audit data) |
| D5 | **Garman-Klass vs Rogers-Satchell** – NaN, edge cases, remplacement éventuel | P1 | `features.py` | Recommandé (IMPROVEMENTS) |
| D6 | **Features utilisées** – Liste finale (sans funding synthétique), corrélations, redundancy | P2 | `loader.py`, `manager.py`, config features | Partiel |
| D7 | **Pipeline run_full_wfo** – `train_end_idx`, fit scaler sur train uniquement, pas de leakage | P0 | `run_full_wfo.py`, `DataManager` | À confirmer |
| D8 | **Qualité / stationnarité** – ADF, structure des séries, impact sur HMM et RL | P2 | `features.py`, tests stationnarité | Partiel |

---

## 3. Modèles (RL & foundation)

| # | Point | Priorité | Fichiers / zone | Statut audit |
|---|------|----------|------------------|--------------|
| M1 | **TQCDropoutPolicy** – Dropout actor/critic, spectral norm, compatibilité checkpoints | P1 | `tqc_dropout_policy.py` | Audité (P1.2) |
| M2 | **TQC config** – `n_quantiles`, `top_quantiles_to_drop`, `gamma`, `tau`, `lr`, gSDE vs noise classique | P1 | `training.py`, `train_agent.py` | Audité (P1.1) |
| M3 | **Incohérence config** – `training.py` vs overrides WFO (`run_full_wfo`), lr 3e-4 vs 1e-4, dropout 0.01 vs 0.1 | P1 | `training.py`, `run_full_wfo.py` | Ouvert (P2.1) |
| M4 | **FoundationFeatureExtractor (MAE → TQC)** – Frozen vs fine-tune, dimensionnement, normalisation | P1 | `rl_adapter.py`, `train_agent.py` | Partiel |
| M5 | **CryptoMAE** – Architecture, pretrain, usage des patches / masques | P2 | `foundation.py`, `train_foundation.py` | Partiel |
| M6 | **Stabilité numérique** – Gradients, spectral norm, `clipped_optimizer` si utilisé | P1 | `tqc_dropout_policy.py`, `clipped_optimizer.py` | Partiel |
| M7 | **Reproducibilité** – Seeds, `torch.manual_seed`, `np.random`, déterminisme GPU | P2 | `reproducibility.py`, `train_agent.py`, config | Tests existants |

---

## 4. Environnement & reward (BatchCryptoEnv, MORL)

| # | Point | Priorité | Fichiers / zone | Statut audit |
|---|------|----------|------------------|--------------|
| E1 | **Reward MORL** – `r_perf`, `w_cost`, `curriculum_lambda`, `MAX_PENALTY_SCALE`, balance smoothness / churn / downside | P0 | `batch_env.py` | Audité (MORL, corrections) |
| E2 | **Espace d’états** – Obs (features, position, w_cost), normalisation, fenêtre temporelle | P1 | `batch_env.py` | Audité (P1.3) |
| E3 | **Actions** – Discretisation, short/long, mapping [-1,1] ↔ position | P1 | `batch_env.py` | Couvert |
| E4 | **Vol scaling** – Mismatch train vs eval (max_leverage, etc.) identifié dans corrections | P0 | `batch_env.py`, eval | Correctif à valider |
| E5 | **Observation noise** – Dynamic noise, annealing, volatilité | P1 | `batch_env.py` | Audité |
| E6 | **Funding rate / coûts** – Shorts, coût de funding, integration dans reward | P2 | `batch_env.py` | Implémenté |
| E7 | **Terminaison d’épisode** – `done`, truncation, horizon effectif vs `gamma` | P2 | `batch_env.py` | Partiel |

---

## 5. Entraînement (train_agent, callbacks, WFO)

| # | Point | Priorité | Fichiers / zone | Statut audit |
|---|------|----------|------------------|--------------|
| T1 | **Curriculum 3 phases** – Churn, smooth, transitions, impact sur收敛 | P1 | `callbacks.py` | Audité (P1.5) |
| T2 | **OverfittingGuardV2** – Signaux (NAV, entropy, train/eval divergence), seuils, patience, actif ou non en WFO | P1 | `callbacks.py`, `OVERFITTING_GUARD_V2.md` | Audité, WFO nuance |
| T3 | **EvalCallback / eval split** – Données eval séparées, pas de leakage, `EVAL_DATA_SPLIT.md` | P1 | `train_agent.py`, `run_full_wfo.py` | Design doc |
| T4 | **Callbacks WFO** – Liste active (curriculum, overfitting guard, checkpoint, metrics), config par segment | P1 | `run_full_wfo.py`, `callbacks.py` | P3.2 |
| T5 | **Checkpoints** – Rotating, `tqc_foundation_*_steps.zip`, compatibilité `use_spectral_norm_*` | P2 | `callbacks.py`, `train_agent.py` | Partiel |
| T6 | **Buffer / batch** – `buffer_size`, `batch_size`, calcul dynamique, sample efficiency | P2 | `train_agent.py`, config | P1.1 |
| T7 | **Logging** – TensorBoard, métriques (NAV, reward, entropy), WFO segment logging | P2 | `callbacks.py`, `run_full_wfo.py` | Partiel |

---

## 6. Scripts & orchestration (WFO)

| # | Point | Priorité | Fichiers / zone | Statut audit |
|---|------|----------|------------------|--------------|
| S1 | **Pipeline WFO** – Ordre préprocessing → HMM → MAE → TQC → eval, isolation train/test | P0 | `run_full_wfo.py` | Audité (P3.2) |
| S2 | **Purge / embargo dans WFO** – Gaps réels entre fenêtres, pas de overlap | P0 | `run_full_wfo.py`, splitter | Ouvert |
| S3 | **Config centralisée** – Réduire magic numbers dans `run_full_wfo`, tout faire passer par `training.py` / `WFOConfig` | P1 | `run_full_wfo.py`, `training.py` | P2.1 |
| S4 | **Résultats / reproductibilité** – Sauvegarde config par segment, logs, `stop_reason` | P2 | `run_full_wfo.py` | Partiel |
| S5 | **Fail-over** – Comportement si OverfittingGuard arrête un segment (min completion, fallback) | P2 | `run_full_wfo.py`, `WFO_OVERFITTING_GUARD` | Design |
| S6 | **Ensemble dans WFO** – `--ensemble`, members, seeds, aggregation, parallel | P2 | `run_full_wfo.py`, `ensemble.py` | Design ENSEMBLE_RL |
| S7 | **Autres scripts** – `prepare_train_eval_split`, `analyze_tensorboard`, usage et maintenance | P3 | `scripts/` | Non audité |

---

## 7. Évaluation & backtest

| # | Point | Priorité | Fichiers / zone | Statut audit |
|---|------|----------|------------------|--------------|
| V1 | **Backtest** – Logique, fees, slippage, équité OOS | P1 | `evaluation/` (runner, backtest si présent) | À auditer |
| V2 | **Ensemble** – Agrégation (conservative, moyenne), OOD, multi-GPU | P1 | `ensemble.py` | Audité (P1.4) |
| V3 | **Métriques** – Sharpe, drawdown, PnL, alpha vs B&H, nombre de trades | P1 | `metrics.py`, evaluation | Partiel |
| V4 | **Train/eval split** – Fichiers `processed_data.parquet` vs `processed_data_eval.parquet`, purge | P1 | `EVAL_DATA_SPLIT.md`, loader | Design |
| V5 | **Stress testing / Monte Carlo** – Non implémenté, pertinence pour robustesse | P2 | - | IMPROVEMENTS |

---

## 8. Configuration & constantes

| # | Point | Priorité | Fichiers / zone | Statut audit |
|---|------|----------|------------------|--------------|
| C1 | **Constantes** – `DEFAULT_PURGE_WINDOW`, `MAX_LOOKBACK_WINDOW`, `DEFAULT_EMBARGO_WINDOW` | P1 | `constants.py`, `splitter.py` | Partiel |
| C2 | **WFOTrainingConfig / WFOConfig** – Redondance, source de vérité, override WFO | P1 | `training.py`, `run_full_wfo.py` | Ouvert |
| C3 | **Hyperparamètres** – Tableau défaut vs WFO (lr, dropout, batch, etc.) documenté | P2 | `training.py`, `MODELES_RL_DESIGN.md` | Partiel |

---

## 9. Tests

| # | Point | Priorité | Fichiers / zone | Statut audit |
|---|------|----------|------------------|--------------|
| X1 | **Data leakage** – Purge, embargo, scaler fit-on-train | P0 | `test_data_leakage.py`, `test_splitting.py` | Existants |
| X2 | **Callbacks** – Curriculum, OverfittingGuard (transitions, signaux) | P1 | `test_callbacks.py` | À compléter (IMPROVEMENTS) |
| X3 | **Env / reward** – Logique steps, reward, MORL | P1 | `test_env.py`, `test_reward.py`, `test_morl.py` | Existants |
| X4 | **Policy / dropout** – TQCDropoutPolicy | P1 | `test_dropout_policy.py` | Existant |
| X5 | **Ensemble** – Sanity + usage | P2 | `test_ensemble.py`, `test_ensemble_sanity.py` | Existants |
| X6 | **E2E WFO** – 2 segments, données synthétiques, leak-free | P1 | Nouveau | Recommandé (P3.2) |
| X7 | **Overfitting guard WFO** | P2 | `test_overfitting_guard_wfo.py` | Existant |
| X8 | **Coverage** – Gaps (evaluation, `run_full_wfo` intégration) | P2 | `tests/` | Non audité |

---

## 10. Infrastructure & opérationnel

| # | Point | Priorité | Fichiers / zone | Statut audit |
|---|------|----------|------------------|--------------|
| I1 | **CI** – `ci.yml`, tests, lint, déploiement | P2 | `.github/workflows/` | À auditer |
| I2 | **Hardware / GPU** – Détection, utilisation batch env, multi-GPU ensemble | P2 | `hardware.py`, `batch_env.py`, `ensemble.py` | Partiel |
| I3 | **Logs / artifacts** – Structure `logs/`, `results/`, nettoyage, rétention | P3 | `logs/`, `results/` | Non audité |
| I4 | **Requirements** – Versions, conflits, repro (CPU/GPU) | P2 | `requirements.txt` | Non audité |
| I5 | **Ruff** – Règles, conformité codebase | P3 | `ruff.toml` | Non audité |

---

## 11. Documentation & design

| # | Point | Priorité | Fichiers / zone | Statut audit |
|---|------|----------|------------------|--------------|
| O1 | **Alignement code ↔ design** – `DATA_PIPELINE_DESIGN`, `MODELES_RL_DESIGN`, `EVAL_DATA_SPLIT`, `MORL_DESIGN`, etc. | P1 | `docs/design/` | Partiel |
| O2 | **CONCEPTS_CLES / CONTEXTE** – À jour avec config, WFO, MORL | P2 | `CONTEXTE_PROJET.md`, `CONCEPTS_CLES.md` | Non audité |
| O3 | **IMPROVEMENTS / FICHIERS_OBSOLETES** – Cohérence avec l’existant | P3 | `IMPROVEMENTS.md`, `FICHIERS_OBSOLETES.md` | Non audité |
| O4 | **Audits** – Synthèse des findings, statut des correctifs, décisions non implémentées | P2 | `docs/audit/` | À faire |

---

## 12. Synthèse des priorités

### P0 – Bloquant (à traiter avant tout entraînement sérieux)

- **D1, D2, D7** : Data leakage (scaler, purge, embargo) et cohérence WFO.
- **E1, E4** : Reward MORL et vol scaling train/eval.

### P1 – Haute priorité (qualité, robustesse, alignement audits)

- **D3, D5, D6** : HMM, Rogers-Satchell, features.
- **M2, M3, M4, M6** : Config TQC, incohérence WFO, adapter MAE, stabilité.
- **E2, E3, E5** : Obs, actions, noise.
- **T1–T4** : Curriculum, OverfittingGuard, eval split, callbacks WFO.
- **S1–S3** : Pipeline WFO, purge/embargo, config centralisée.
- **V1–V4** : Backtest, ensemble, métriques, split eval.
- **C1, C2** : Constantes, config.
- **X2, X6** : Tests callbacks, E2E WFO.
- **O1** : Alignement design / code.

### P2 – Moyenne priorité

- **D4, D8, M5, M7, E6, E7** : FFD, stationnarité, MAE, repro, coûts, terminaison.
- **T5–T7, S4–S6, V5** : Checkpoints, logging, résultats WFO, fail-over, ensemble, stress testing.
- **C3, X5, X7, X8** : Config, tests ensemble/guard, coverage.
- **I1, I2, I4** : CI, hardware, requirements.
- **O2, O4** : Docs, synthèse audits.

### P3 – Basse priorité

- **S7, I3, I5, O3** : Autres scripts, logs, ruff, IMPROVEMENTS.

---

## 13. Ordre suggéré pour un audit complet

1. **Phase 0 – Blocage**  
   Revérifier D1, D2, D7, E1, E4 (data + reward) et les tests associés (X1).

2. **Phase 1 – Cœur RL & WFO**  
   M1–M4, E2–E5, T1–T4, S1–S3, C1–C2. Compléter X2, X6.

3. **Phase 2 – Évaluation & config**  
   V1–V4, C3, partie evaluation du code.

4. **Phase 3 – Intégration & infra**  
   S4–S6, X5, X7, X8, I1–I2, I4.

5. **Phase 4 – Documentation & synthèse**  
   O1–O4, mise à jour des audits et d’IMPROVEMENTS.

---

## 14. Références rapides

| Document | Usage |
|----------|--------|
| `DATA_PIPELINE_AUDIT_REPORT.md` | Data pipeline, P0/P1, correctifs |
| `MASTER_PLAN_AUDIT_MODELES_RL.md` | Structure des audits RL |
| `AUDIT_MODELES_RL_RESULTATS.md` | Résultats audits TQC, env, callbacks, ensemble, WFO |
| `AUDIT_MORL.md` | MORL, w_cost, front de Pareto |
| `AUDIT_OBSERVATION_NOISE.md` | Bruit d’observation |
| `AUDIT_CORRECTIONS_PROPOSEES.md` | Vol scaling, MORL, distributional shift |
| `IMPROVEMENTS.md` | P0–P4, suivi des tâches |
| `EVAL_DATA_SPLIT.md` | Split train/eval |
| `OVERFITTING_GUARD_V2.md` | Guard, signaux, WFO |
| `WFO_OVERFITTING_GUARD.md` | Intégration Guard dans WFO |

---

*Ce document sert de checklist pour un audit projet complet. Marquer chaque point audité et renvoyer vers les rapports détaillés (existants ou à créer).*

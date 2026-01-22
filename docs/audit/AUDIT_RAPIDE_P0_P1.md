# Audit rapide P0 / P1 – CryptoRL

**Date** : 2026-01-22  
**Périmètre** : Points P0 et P1 de `AUDIT_PROJET_COMPLET_POINTS.md`  
**Méthode** : Revue de code ciblée (data, env, WFO, config, tests)

---

## Résumé exécutif

| Catégorie | Verdict | Commentaire |
|-----------|---------|-------------|
| **P0 – Data leakage** | OK | Scaler fit-on-train, purge 720h, embargo 24h en place. WFO préprocess cohérent. |
| **P0 – Reward / vol scaling** | OK | MORL implémenté, `max_leverage` identique train/eval. Pas de distributional shift. |
| **P0 – Pipeline WFO** | OK | Ordre préprocessing → HMM → MAE → TQC → eval. Purge/embargo dans segments. |
| **P1 – Config** | OK | `WFOTrainingConfig` centralisé, plus de magic numbers WFO. |
| **P1 – Tests** | À corriger | `test_splitting` utilise `purge_window=50` (violation P0.2). WFO ne valide pas purge. |

**Conclusion** : Aucun blocant P0 restant. Actions P1 recommandées : valider purge dans WFO, corriger `test_splitting`, compléter tests callbacks / E2E WFO.

---

## 1. P0 – Data leakage (D1, D2, D7)

### D1 – RobustScaler / purge / embargo

- **`splitter.py`** : `validate_purge_window(purge)` impose `purge >= MAX_LOOKBACK_WINDOW` (720). `DEFAULT_PURGE_WINDOW = 720`, `DEFAULT_EMBARGO_WINDOW = 24`. Split train/val/test avec purge entre les blocs, embargo documenté.
- **`manager.py`** : `pipeline(..., train_end_idx)` : si `train_end_idx` fourni, `scaler.fit(df.iloc[:train_end_idx])` puis `transform` sur tout le dataset. Sinon, `UserWarning` legacy (fit sur tout).
- **`run_full_wfo.py`** : **N’utilise pas** `DataManager.pipeline()`. `preprocess_segment()` fait son propre scaling :
  - Fit `RobustScaler` sur `train_df` uniquement (l.362).
  - Transform train, eval, test.
- **Verdict** : OK. Correctifs P0 appliqués (fit-on-train, purge 720, embargo 24).

### D2 – Purge / embargo en WFO

- **`calculate_segments()`** (l.225–285) : structure `[train][PURGE][eval][PURGE][test][EMBARGO]`. `purge = config.purge_window` (720), `embargo = config.embargo_window` (24). Bornes `eval_start = train_end + purge`, `test_start = eval_end + purge`, step inclut l’embargo.
- **`preprocess_segment()`** : extrait `df_features` sur `[train_start, test_end]`, puis split en prenant les derniers `train_len`, `eval_len`, `test_len` lignes. Les lignes purge sont exclues des blocs train/eval/test.
- **Manque** : Aucun appel à `validate_purge_window` dans le WFO. Si `purge_window` était rendu configurable (CLI) et fixé &lt; 720, le leak ne serait pas détecté.
- **Verdict** : OK avec les valeurs par défaut. **P1** : appeler `validate_purge_window(config.purge_window)` dans `calculate_segments` (ou au chargement de `WFOConfig`) pour robustesse.

### D7 – Pipeline `run_full_wfo` / `train_end_idx`

- WFO ne passe pas par `DataManager.pipeline()` ni `train_end_idx`. Il utilise `preprocess_segment`, qui fit le scaler sur train uniquement.
- **Verdict** : OK. Pas de leakage via scaler dans le flux WFO.

---

## 2. P0 – Reward MORL et vol scaling (E1, E4)

### E1 – Reward MORL

- **`batch_env.py`** : `_compute_reward()` (l.400–481) :
  - `r_perf = log1p(returns) * SCALE`, `r_cost = -position_deltas * SCALE` (clippé).
  - `reward = r_perf + w_cost * r_cost * MAX_PENALTY_SCALE`. `w_cost` dans l’obs, sampling 20% / 60% / 20% (0, U[0,1], 1).
- Pas de `curriculum_lambda` dans la reward (utilisé seulement pour métriques / curriculum côté callbacks). Architecture MORL conforme au design.

### E4 – Vol scaling train vs eval

- **`train_agent.create_environments`** : train et eval envs créés avec `max_leverage=config.max_leverage` (depuis `TQCTrainingConfig`).
- **`run_full_wfo`** :
  - Train (l.842) : `max_leverage=self.config.training_config.max_leverage`.
  - Eval (l.1048) : idem, avec commentaire « MORL: Cohérence train/eval (fix Distributional Shift) ».
- **`WFOTrainingConfig`** : n’override pas `max_leverage` → 2.0 (hérité). Même valeur partout.
- **Verdict** : OK. Correction distributional shift (eval = 1.0 vs train = 2.0) bien appliquée.

---

## 3. P0 – Pipeline WFO et purge/embargo (S1, S2)

### S1 – Pipeline WFO

- Ordre réel : features globales → `preprocess_segment` (scaler fit-on-train) → `train_hmm` (fit sur train, predict train+eval+test) → MAE → TQC → évaluation OOS sur test.
- Isolation train / eval / test respectée ; pas de reuse de test pour l’entraînement.

### S2 – Purge / embargo dans les fenêtres

- Segments calculés avec purge 720 et embargo 24. Les blocs train/eval/test n’incluent pas les lignes purge. Gaps réels entre fenêtres.
- **Verdict** : OK.

---

## 4. P1 – Config (M3, S3, C1, C2)

### M3 / S3 – Incohérence config / magic numbers

- **`run_full_wfo`** : utilise `WFOTrainingConfig` (l.89, 588, 712). Hyperparamètres TQC (lr, dropout, batch, etc.) viennent de la config, plus de valeurs en dur pour le WFO.
- **`training.py`** : `WFOTrainingConfig` override explicites (lr 1e-4, `critic_dropout` 0.1, guard, etc.). Différences vs `TQCTrainingConfig` sont dans le code, pas en magic numbers.

### C1 / C2 – Constantes et WFOConfig

- `constants.py` : `DEFAULT_PURGE_WINDOW`, `MAX_LOOKBACK_WINDOW`, `DEFAULT_EMBARGO_WINDOW` définis et utilisés par le splitter.
- WFO : `purge_window` / `embargo_window` dans `WFOConfig`, pas encore branchés sur `validate_purge_window` ni sur les constantes (voir D2).

**Verdict** : Config centralisée, cohérente. P1 restant : lier WFO purge/embargo aux constantes et à la validation.

---

## 5. P1 – Tests (X1, X2, X6)

### X1 – Data leakage

- **`test_data_leakage`** : scaler fit-on-train, purge ≥ `MAX_INDICATOR_WINDOW`. Présents et alignés avec P0.
- **`test_splitting`** :
  - `test_split_sizes` appelle `split_data(..., purge_window=50)`. Or `validate_purge_window(50)` lève `ValueError` (50 &lt; 720). Le test **viole P0.2** et **échouera** avec le splitter actuel (si le fichier de données existe).
  - `test_chronological_order` et `test_no_empty_sets` utilisent `split_data()` sans override → `DEFAULT_PURGE_WINDOW` 720, cohérent.

**Action P1** : Corriger `test_splitting.test_split_sizes` pour utiliser `purge_window=720` (ou `DEFAULT_PURGE_WINDOW`) et adapter les asserts (pertes = 2×720).

### X2 – Callbacks

- **`test_callbacks`** : présent, couvre `ThreePhaseCurriculumCallback`, `OverfittingGuardCallbackV2`, `ModelEMACallback`. IMPROVEMENTS indique « à compléter » (transitions, signaux). À enrichir si besoin.

### X6 – E2E WFO

- Aucun test E2E WFO (2 segments, données synthétiques, leak-free). Reste une recommandation P1.

---

## 6. P1 – Autres points survolés

- **D5 (Garman-Klass)** : Toujours utilisé dans `features.py`. Rogers-Satchell non implémenté. P1 de l’IMPROVEMENTS.
- **V1 (Backtest)** : Pas de module `backtest` / `runner` dédié. Logique d’évaluation dans `run_full_wfo` et `ensemble`. À auditer plus en détail si on vise un backtest réutilisable.
- **T2–T4 (Callbacks WFO)** : OverfittingGuard, curriculum, EvalCallback documentés. WFO utilise bien la config guard. Pas de incohérence repérée.

---

## 7. Synthèse des actions recommandées

| Priorité | Action |
|----------|--------|
| **P1** | Appeler `validate_purge_window(config.purge_window)` dans le WFO (ex. `calculate_segments` ou au chargement de la config). |
| **P1** | Corriger `test_splitting.test_split_sizes` : `purge_window=720` (ou constante) et asserts sur les pertes (2×720). |
| **P1** | Ajouter un test E2E WFO (2 segments, données synthétiques, vérif leak-free). |
| **P2** | Documenter ou brancher purge/embargo WFO sur `constants` / `validate_purge_window`. |
| **P2** | Compléter `test_callbacks` (transitions curriculum, signaux guard) si nécessaire. |

---

## 8. Références

- `AUDIT_PROJET_COMPLET_POINTS.md` – grille P0/P1
- `DATA_PIPELINE_AUDIT_REPORT.md` – P0.1 / P0.2
- `AUDIT_CORRECTIONS_PROPOSEES.md` – vol scaling, MORL, distributional shift
- `IMPROVEMENTS.md` – tâches P0–P4

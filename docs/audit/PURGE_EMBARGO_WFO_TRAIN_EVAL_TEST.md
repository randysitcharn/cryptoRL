# Purge / Embargo vs Train–Eval–Test WFO – Sources de problèmes

**Date** : 2026-01-22  
**Périmètre** : `run_full_wfo.py`, `splitter.py`, constants, tests

---

## 1. Résumé

| Problème | Sévérité | Statut |
|----------|----------|--------|
| **Bug « take from end » dans `preprocess_segment`** | **P0** | À corriger |
| WFO n’appelle pas `validate_purge_window` | P1 | Corrigé (appel dans `calculate_segments`) |
| Splitter : `embargo` documenté mais non utilisé dans le split | P2 | Design |
| Step / embargo : chevauchement des segments WFO | P2 | Documenté |
| `test_splitting` : `purge_window=50` (violation P0.2) | P1 | Voir AUDIT_RAPIDE_P0_P1 |

---

## 2. Bug critique : split « take from end » dans `preprocess_segment`

### 2.1 Structure attendue (calculate_segments)

```
[train] [PURGE] [eval] [PURGE] [test] [EMBARGO]
```

- `train_start`, `train_end`
- `eval_start = train_end + purge`, `eval_end = eval_start + eval_rows`
- `test_start = eval_end + purge`, `test_end = test_start + test_rows`

Les lignes **purge** sont entre train↔eval et eval↔test ; elles ne doivent **pas** être dans train, eval ni test.

### 2.2 Ce que fait le code actuel (l.326–346)

1. `df_features = _df_features_global.loc[train_start : test_end]`  
   → Contient bien `[train | purge | eval | purge | test]` (avec purge).

2. **Split** :
   - `total_needed = train_len + eval_len + test_len` (sans purge).
   - « Take from END » :
     - `train_df = df_features.iloc[-total_needed : -(eval_len+test_len)]`
     - `eval_df = df_features.iloc[-(eval_len+test_len) : -test_len]`
     - `test_df = df_features.iloc[-test_len :]`

3. Effet :
   - On garde les **dernières** `total_needed` lignes (donc on **supprime** les premières `2 × purge`).
   - Ces premières lignes sont en **début de train**.
   - Le bloc « train » ainsi défini contient en réalité :
     - fin de train (sans les premiers `2×purge`),
     - **tout ou partie du premier purge**,
     - **éventuellement du début d’eval**.
   - Donc : **données purge (et possiblement eval) dans le bloc « train »** → **data leakage**.

### 2.3 Correction requise

Utiliser les **bornes exactes** du segment, en excluant les purge :

```python
purge = segment['purge_window']
train_df = df_features.iloc[0 : train_len]
eval_df  = df_features.iloc[train_len + purge : train_len + purge + eval_len]
test_df  = df_features.iloc[train_len + 2*purge + eval_len :]
```

On vérifie aussi que `len(df_features) >= train_len + 2*purge + eval_len + test_len` ; sinon, erreur claire ou fallback documenté (pas de split « from end »).

---

## 3. Purge / embargo : incohérences et manques

### 3.1 Splitter (`splitter.py`) vs WFO (`run_full_wfo`)

| | Splitter | WFO |
|-|----------|-----|
| Structure | `[train][PURGE][val][PURGE][test]` | `[train][PURGE][eval][PURGE][test][EMBARGO]` |
| Purge | Entre train↔val et val↔test | Entre train↔eval et eval↔test |
| Embargo | Paramètre présent, **non utilisé** dans les slices | Utilisé dans `segment_size` et step |

Dans le splitter, `embargo_window` est documenté et imprimé dans `_print_split_stats`, mais **aucun** `iloc` ne l’utilise. En WFO, l’embargo est pris en compte dans la taille du segment et le step.

### 3.2 WFO : pas de validation du purge

- `splitter.split_data` appelle `validate_purge_window(purge)`.
- En WFO, `calculate_segments` utilise `config.purge_window` **sans** appeler `validate_purge_window`.
- Si on rend `purge_window` configurable (ex. CLI) et qu’on met &lt; 720, **aucune erreur** n’est levée → risque de leakage.

**P1** : appeler `validate_purge_window(config.purge_window)` dans `calculate_segments` (ou à l’init du pipeline / chargement de la config).

### 3.3 Step vs segment size et embargo

- `segment_size = train + 2×purge + eval + test + embargo`.
- `step_rows = step_months × hours_per_month` (ex. 3 × 720 = 2160).
- Les segments se **chevauchent** : le segment suivant commence à `start + step_rows`, donc bien avant la fin du segment actuel. L’« embargo » est une **réserve** de 24h **à la fin** de chaque segment, pas un gap physique jusqu’au prochain train.

À documenter clairement : embargo = fenêtre réservée en fin de segment ; step = pas de roll forward (segments qui se chevauchent).

---

## 4. Train / Eval / Test : sémantique

- **Train** : apprentissage TQC (et HMM, MAE).  
- **Eval** : « in-train » evaluation (EvalCallback) ; en WFO, le design prévoit souvent de **désactiver** l’EvalCallback pour éviter tout usage de l’eval comme signal d’early stopping ou de sélection de modèle sur données « futures ».  
- **Test** : évaluation OOS finale, backtest.

Même si l’eval n’est pas utilisée pour early stopping en WFO, le **split** doit rester strict : train / eval / test selon les bornes du segment, **sans** inclure les purge. Le bug « take from end » casse cette propriété.

---

## 5. Actions recommandées

| Priorité | Action |
|----------|--------|
| **P0** | Corriger `preprocess_segment` : split par bornes exactes (train / eval / test), exclure les purge, pas de « take from end ». **Fait.** |
| P1 | Appeler `validate_purge_window(config.purge_window)` dans le WFO (ex. `calculate_segments` ou init). **Fait.** |
| P1 | Corriger `test_splitting` : `purge_window` ≥ 720 (voir AUDIT_RAPIDE_P0_P1). |
| P2 | Documenter ou aligner l’usage de l’embargo dans le splitter vs WFO. |
| P2 | Clarifier step / segment size / embargo (chevauchement, rôle de l’embargo). |

---

## 6. Références

- `AUDIT_RAPIDE_P0_P1.md` (D2, S2, tests)
- `DATA_PIPELINE_AUDIT_REPORT.md` (P0.2)
- `DATA_PIPELINE_DESIGN.md` (§ Purge & Embargo, WFO segment structure)

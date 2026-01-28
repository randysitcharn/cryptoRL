# Améliorations Futures - CryptoRL

> **Instructions d'utilisation**
> 
> Ce fichier liste les améliorations prévues pour le projet, organisées par priorité.
> 
> **Format des entrées :**
> - `[ ]` = À faire
> - `[x]` = Implémenté
> - `[~]` = En cours
> - `[R]` = Rejeté
> 
> **Priorités :**
> - **P0** : BLOQUANT ⛔ - Invalide les résultats, ne pas entraîner avant correction
> - **P1** : Haute priorité - Améliore significativement la qualité/robustesse
> - **P2** : Priorité moyenne - Amélioration incrémentale
> - **P3** : Basse priorité - Optimisations performance
> - **P4** : Recherche - Pistes à long terme
> 
> **Mise à jour :** Marquer les items comme `[x]` après implémentation et ajouter la date.

---

## P0 - BLOQUANTS ⛔ (Invalident les résultats)

> **⚠️ Ne lancer AUCUN entraînement avant correction des P0.** Tout résultat actuel est invalide.

### [x] Data Leakage - RobustScaler (fit sur tout dataset) ✅
**Fichier:** `src/data_engineering/manager.py:953`  
**Statut:** Implémenté (2026-01-22)  
**Référence:** Audit Data Pipeline P0 - CONFIRMÉ FATAL  
**Description:** Le `RobustScaler` fait un `.fit()` sur tout le dataset au lieu de train seulement. La médiane de t₀ est influencée par les prix de t₊₁₀₀₀.  
**Fix:** Fit uniquement sur `train_set`, `transform` le `test_set` avec statistiques figées du train.  
**Impact:** Test performance artificiellement gonflée. L'agent "voit le futur".  
**Implémentation:** Ajout du paramètre `train_end_idx` dans `DataManager.pipeline()` pour fit sur train uniquement. Warning en mode legacy.

---

### [x] Data Leakage - Purge Window insuffisant (50h vs 720h) ✅
**Fichier:** `src/data_engineering/splitter.py:28`  
**Statut:** Implémenté (2026-01-22)  
**Référence:** Audit Data Pipeline P0 - CONFIRMÉ MATHÉMATIQUEMENT  
**Description:** Purge window de 50h est inférieur à la fenêtre max des indicateurs (720h pour Z-Score). Les données entre T-720 et T-1 sont partagées entre train et validation.  
**Fix:** Augmenter à 720h (max indicator window).  
**Impact:** Rolling features leak future info. Début du validation set pollué par fin du training set.  
**Implémentation:** `DEFAULT_PURGE_WINDOW = 720` dans `constants.py`, utilisé dans `splitter.py` et `run_full_wfo.py`. Validation ajoutée pour garantir purge >= MAX_LOOKBACK_WINDOW.

---

### [x] Spectral Normalization (Stabilité Gradients) ✅
**Fichier:** `src/models/tqc_dropout_policy.py`  
**Statut:** Implémenté (2026-01-22)  
**Référence:** Audit SOTA #1  
**Description:** Wrapper les couches `nn.Linear` du Critic avec `torch.nn.utils.spectral_norm` pour éviter l'explosion des gradients en régimes volatiles.  
**Impact:** Stabilité critique en HFT/Crypto volatile.  
**Implémentation:** 
- Deux flags configurables: `use_spectral_norm_critic` et `use_spectral_norm_actor` (default: False)
- Appliqué uniquement aux couches cachées (pas la dernière couche output)
- Validation de compatibilité checkpoint ajoutée dans `train_agent.py`
- ⚠️ **WARNING:** Checkpoints ne sont PAS compatibles si les flags `use_spectral_norm_*` changent entre entraînements (incompatibilité `state_dict`)

---

### [x] Short Selling Support ✅
**Fichier:** `src/training/batch_env.py`  
**Statut:** Implémenté (2026-01-19)  
**Description:** Support complet du short selling avec mapping symétrique action=-1 → -100% short.  
**Impact:** L'agent peut profiter des marchés baissiers.

---

## P1 - Haute Priorité (Améliorations significatives)

### [ ] Tests Callbacks (test_callbacks.py)
**Fichier:** `tests/test_callbacks.py` (à créer)  
**Référence:** Audit Modèles RL - CRITIQUE  
**Description:** Créer tests pour `ThreePhaseCurriculumCallback` transitions et `OverfittingGuardV2` signal detection.  
**Impact:** Validation de la robustesse des callbacks critiques.

---

### [ ] Smooth Coef Tuning
**Fichier:** `src/training/callbacks.py`  
**Description:** Monitorer et ajuster la distribution de `w_cost` selon le nombre de trades par épisode.  
**Impact:** Balance entre réduction du churn et capacité à trader via MORL.

---

### [ ] Rogers-Satchell Volatility (Robustesse)
**Fichier:** `src/data_engineering/features.py`  
**Référence:** Audit Data Pipeline P2  
**Description:** Remplacer Garman-Klass par Rogers-Satchell (toujours >= 0, plus robuste). Garman-Klass peut produire des NaN.  
**Impact:** Élimine les NaN dans les features de volatilité, plus robuste aux edge cases.

---

### [ ] Curriculum Lambda Max Tuning
**Fichier:** `src/training/batch_env.py`  
**Description:** Rendre `curriculum_lambda_max` configurable (actuellement hardcodé à 0.4).  
**Impact:** Permet de tuner le ratio PnL/Penalties selon les résultats OOS.

---

### [ ] Métriques Ensemble Avancées (TensorBoard)
**Fichier:** `src/training/callbacks.py`  
**Description:** Logger les métriques ensemble (agreement, confidence, OOD score) dans TensorBoard pendant WFO.  
**Impact:** Visualisation de la santé de l'ensemble et détection des régimes OOD.

---

### [ ] Conservative Quantile Tuning (WFOConfig)
**Fichier:** `scripts/run_full_wfo.py`, `src/config/training.py`  
**Description:** Exposer `top_quantiles_to_drop_per_net` dans WFOConfig pour tuning selon profil (HFT vs Swing).  
**Impact:** Ajustement du conservatisme selon le timeframe sans modifier le code.

---

### [x] Embargo manquant (post-test gap) ✅
**Fichier:** `src/data_engineering/splitter.py`  
**Statut:** Implémenté (2026-01-22)  
**Référence:** Audit Data Pipeline P1-NEW  
**Description:** Ajouter un gap après le test set pour éviter contamination lors du retraining.  
**Impact:** Intégrité des données pour retraining.  
**Implémentation:** Paramètre `embargo_window` ajouté dans `TimeSeriesSplitter.split_data()` et `WFOConfig` (default 24h).

---

### [x] Training Multi-GPU Parallèle (Ensemble) ✅
**Fichier:** `src/evaluation/ensemble.py`  
**Statut:** Implémenté (2026-01-22)  
**Description:** Training parallèle des membres d'ensemble sur GPUs multiples via `torch.multiprocessing`.  
**Impact:** Réduction du temps de training (3 membres → 1/3 du temps avec 3 GPUs).

---

### [x] Funding Rate pour Shorts ✅
**Fichier:** `src/training/batch_env.py`  
**Statut:** Implémenté (2026-01-19)  
**Description:** Coût de funding pour positions short (style perpetual futures).  
**Impact:** Short selling réaliste avec coût de funding.

---

### [x] Funding Rate Synthétique Désactivé ✅
**Fichier:** `src/data_engineering/loader.py`  
**Statut:** Implémenté (2026-01-22)  
**Référence:** Audit Data Pipeline P1.2  
**Description:** Désactiver le funding rate synthétique par défaut (causes spurious correlations). `HMM_Funding` retiré de `HMM_FEATURES`.  
**Impact:** Évite les corrélations fallacieuses, améliore la généralisation.  
**Implémentation:** `use_synthetic_funding=False` par défaut, `HMM_Funding` retiré de la liste des features HMM.

---

### [x] Retry Logic pour Download (Tenacity) ✅
**Fichier:** `src/data_engineering/loader.py`  
**Statut:** Implémenté (2026-01-22)  
**Référence:** Audit Data Pipeline P1.3  
**Description:** Ajouter retry logic avec exponential backoff pour les téléchargements de données (résilience aux erreurs réseau).  
**Impact:** Robustesse face aux erreurs réseau temporaires.  
**Implémentation:** Décorateur `@retry` avec `tenacity` (3 tentatives, exponential backoff 4-10s).

---

### [x] Data Augmentation - Dynamic Noise ✅
**Fichier:** `src/training/batch_env.py`  
**Statut:** Implémenté (2026-01-19)  
**Description:** Bruit d'observation avec annealing temporel + adaptation à la volatilité.  
**Impact:** Meilleure généralisation, convergence plus stable.

---

## P2 - Priorité Moyenne (Améliorations incrémentales)

### [x] WFO In-Train Evaluation ✅
**Fichier:** `scripts/run_full_wfo.py`  
**Statut:** Implémenté (2026-01-19)  
**Description:** Split "eval" entre train et test pour évaluation in-train (EvalCallback) pendant WFO.  
**Impact:** Détection précoce de l'overfitting, meilleure sélection de modèle.

---

### [ ] SharedMemory pour Replay Buffer (OOM Protection)
**Fichier:** `src/training/replay_buffer.py` (nouveau)  
**Description:** Utiliser `multiprocessing.SharedMemory` pour partager le Replay Buffer entre membres d'ensemble.  
**Impact:** Réduit la mémoire de O(n_members × buffer_size) à O(buffer_size).

---

### [ ] Multi-Asset Support
**Fichier:** `src/training/batch_env.py`  
**Description:** Étendre BatchCryptoEnv pour gérer un portefeuille multi-assets (BTC + ETH).  
**Impact:** Permet la diversification et les stratégies de spread.

---

### [ ] Stress Testing / Monte Carlo
**Fichier:** `src/evaluation/stress_testing.py` (nouveau)  
**Description:** Évaluer la robustesse via simulation Monte Carlo et stress testing sur variations des données.  
**Impact:** Évaluation robuste en conditions adverses, détection d'overfitting.

---

### [ ] Data Augmentation - Magnitude Scaling
**Fichier:** `src/training/batch_env.py`  
**Description:** Multiplier les observations par un facteur aléatoire pour simuler différentes conditions de volatilité.  
**Impact:** Simule différentes conditions de volatilité, préserve la structure relative.

---

### [ ] Data Augmentation - Time Warping
**Fichier:** `src/training/batch_env.py`  
**Description:** Étirer/compresser temporellement certaines portions de la série temporelle.  
**Impact:** Crée de la variété structurelle pour les patterns de chartisme.  
**Note:** Complexe à implémenter, peut casser les relations temporelles.

---

### [x] A/B Testing: gSDE vs Actor Noise ✅
**Fichier:** `src/training/train_agent.py`, `src/config/training.py`  
**Statut:** Implémenté (2026-01-19)  
**Description:** Support pour deux approches d'exploration (gSDE et OrnsteinUhlenbeckActionNoise).  
**Impact:** Permet de tester quelle stratégie d'exploration fonctionne mieux.

---

## P3 - Basse Priorité / Optimisations

### [ ] Data Pipeline - FFD Vectorisé (Performance 10x-100x)
**Fichier:** `src/data_engineering/features.py`  
**Description:** Implémenter FFD via FFT ou Numba JIT au lieu de boucle Python naïve.  
**Impact:** Pipeline de données 10x-100x plus rapide.

---

### [ ] Data Pipeline - Cache d_optimal Persistant
**Fichier:** `src/data_engineering/features.py`  
**Description:** Persister le cache `d_optimal` sur disque avec invalidation par hash des données.  
**Impact:** Skip ADF test si données inchangées (économie ~30s par asset).

---

### [ ] Data Pipeline - Feature Computation Parallèle
**Fichier:** `src/data_engineering/features.py`  
**Description:** Paralléliser le calcul des features par asset avec `concurrent.futures`.  
**Impact:** Feature engineering 3-4x plus rapide sur machines multi-core.

---

## P4 - Recherche / Long Terme

### [ ] Chain of Inheritance pour Ensemble (Warm Start par Seed)
**Fichier:** `src/evaluation/ensemble.py`, `scripts/run_full_wfo.py`  
**Description:** Connecter les poids finaux de l'ensemble du segment N comme initialisation pour le segment N+1.  
**Impact:** Convergence plus rapide, continuité des représentations apprises.

---

### [ ] Ensemble RL v1.4 - Gaps Critiques SOTA
**Fichier:** `src/evaluation/ensemble.py`  
**Description:** 7 gaps identifiés pour passer de 8/10 à 9+/10 : Composite Risk formel, BMA, Ensemble hétérogène, Meta-contrôleur, VAE OOD, Transformers temporels.  
**Référence:** `docs/design/ENSEMBLE_RL_DESIGN.md` v1.3 + Audit SOTA 2026-01-22  
**Impact:** Transformation vers framework bayésien, composite-risk, avec OOD structuré.

---

### [ ] HMM Relative Artifacts + A/B Testing
**Fichier:** `src/data_engineering/features.py`  
**Description:** Passer à des artifacts relatifs et implémenter A/B testing pour valider l'apport du HMM.  
**Impact:** Validation objective de l'apport du HMM.

---

### [ ] 3 HMM Timeframes
**Fichier:** `src/data_engineering/features.py`  
**Description:** Entraîner plusieurs HMM sur différents timeframes pour capturer les régimes à plusieurs échelles.  
**Impact:** Capture des régimes de marché à court, moyen et long terme.

---

### [ ] Constrained RL Formalisé (CMDP)
**Fichier:** Nouveau module  
**Description:** Formaliser les contraintes de trading via Constrained MDP avec Lagrangiens appris.  
**Impact:** Garanties théoriques sur le respect des contraintes, plus rigoureux que MORL heuristique.

---

### [ ] GP Diffusion Policy (GPDP)
**Fichier:** Nouveau module de recherche  
**Description:** Régularisation de la politique via Gaussian Process Regression.  
**Référence:** "Overcoming Overfitting in RL via Gaussian Process Diffusion Policy" (arXiv:2506.13111)  
**Impact:** Potentiellement meilleure généralisation, mais effort de recherche significatif.

---

### [ ] Data Augmentation - Synthetic Episode Generation
**Fichier:** Nouveau module  
**Description:** Générer des épisodes synthétiques avec modèles génératifs (GANs, Diffusion Models).  
**Impact:** Haute valeur si bien fait, mais effort très élevé.

---

## 💡 Insights / Best Practices

### Monitoring des Quantiles (Spread) - "Killer Feature" Anti-Overfitting

**Contexte:** Détection d'overfitting en Reinforcement Learning pour trading

**Problème:** 
- En apprentissage supervisé, on regarde la Loss Validation
- En RL, la "Reward OOS" est très bruitée et peu fiable pour détecter l'overfitting

**Solution - Le Spread des Quantiles comme "Canari dans la Mine":**

Le **Spread des Quantiles** (écart entre quantiles min/max du Critic TQC) est un indicateur critique pour détecter l'overfitting :

- **Signal d'overfitting:** Si le spread se réduit (le modèle devient sûr de lui) alors que la performance OOS stagne ou baisse, c'est la définition exacte de l'overfitting (arrogance du modèle)
- **Signal de bonne généralisation:** Un spread stable ou qui augmente avec une performance OOS croissante indique une bonne généralisation

**Implémentation recommandée:**
- Logger le spread des quantiles dans TensorBoard pendant l'entraînement
- Monitorer la divergence entre spread (confiance du modèle) et performance OOS
- Alerter si spread ↓ + performance OOS → ou ↓ (signe d'overfitting)

**Référence:** Section 4.1 - Monitoring des Quantiles (Spread)

---

## Propositions REJETÉES

### [R] Feature-Specific Noise
**Raison:** Complexité de maintenance trop élevée pour gain marginal.  
**Alternative:** Reporter après validation des techniques approuvées.

---

### [R] SNI (Selective Noise Injection)
**Raison:** Changement architectural trop profond, hors scope.  
**Alternative:** Créer ticket de recherche pour évaluation future.

---

## Techniques à ÉVITER

| Technique | Pourquoi l'éviter |
|-----------|-------------------|
| **Flip temporel** | Le temps a une direction. Un pattern inversé temporellement devient complètement différent. |
| **Shuffling des features** | Les colonnes ont une sémantique fixe. Le modèle apprend que colonne 0 = prix. |
| **Mixup/CutMix** | Mélanger deux contextes de marché crée une chimère irréaliste (mi-bull mi-bear). |
| **Bruit trop fort (>5%)** | Détruit le signal. Le modèle apprend à ignorer les observations. |

---

*Dernière mise à jour: 2026-01-22*

---

## ✅ Corrections Data Pipeline Complétées (2026-01-22)

Toutes les corrections P0 et améliorations P1 du plan `data_pipeline_p0_fixes` ont été implémentées :

- ✅ **P0.1** - Fix scaler leakage (train_end_idx dans DataManager.pipeline)
- ✅ **P0.2** - Fix purge window (50h → 720h)
- ✅ **P0.3** - Constantes centralisées (MAX_LOOKBACK_WINDOW, DEFAULT_PURGE_WINDOW, DEFAULT_EMBARGO_WINDOW)
- ✅ **Tests** - test_data_leakage.py créé avec tests de non-fuite
- ✅ **P1.1** - Embargo window ajouté
- ✅ **P1.2** - Funding rate synthétique désactivé
- ✅ **P1.3** - Retry logic avec tenacity

**Référence:** `docs/audit/DATA_PIPELINE_AUDIT_REPORT.md` et plan `data_pipeline_p0_fixes_de0acb6b.plan.md`

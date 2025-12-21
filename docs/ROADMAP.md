# CryptoRL - Roadmap de Développement

## Vue d'ensemble

Ce document décrit les prochaines étapes de développement du projet CryptoRL, structuré en 4 phases principales.

---

## Phase 1 : Universal Data & Regimes (Les Yeux)

### Technologies Clés
- **Multi-Asset** : BTC, ETH, SPX, DXY, Funding Rates
- **Fracdiff (d≈0.4)** : Différenciation fractionnaire
- **GMM-HMM** : Gaussian Mixture Hidden Markov Model

### Objectif & Justification SOTA
**Contexte & Mémoire**
- Fracdiff : Garde la mémoire long terme tout en étant stationnaire
- GMM-HMM : Gère les "Fat Tails" (crashs) et donne la probabilité du régime (Bull/Bear) en temps réel

### Instructions d'Implémentation
```
Implémente FFD avec fenêtre fixe pour préserver la mémoire.
Utilise hmmlearn.GMMHMM (3 états, 2 mixtures) pour sortir un vecteur
de probabilités predict_proba.
Normalise avec RobustScaler.
```

### Tâches
- [x] Implémenter FFD (Fractional Differentiation) avec fenêtre fixe
- [x] Configurer pipeline multi-asset (BTC, ETH, SPX, DXY, NASDAQ)
- [x] Intégrer GMM-HMM avec hmmlearn (3 états, 2 mixtures)
- [x] Créer fonction `predict_proba` pour probabilités de régime
- [x] Normalisation avec RobustScaler

---

## Phase 2 : Foundation Model (Le Cerveau)

### Technologies Clés
- **Pre-Norm Transformer** : Encoder-only
- **Masked Auto-Encoder (MAE)** : Self-Supervised Learning
- **AdamW + Scheduler**

### Objectif & Justification SOTA
**Compréhension Physique**
Avant de trader, l'IA doit comprendre la dynamique des prix. Le Pre-training accélère la convergence RL et réduit le risque de sur-apprentissage (Overfitting).

### Instructions d'Implémentation
```
Crée une classe PretrainTrainer.
Masque 15% des bougies (Fracdiff + LogRet).
Entraîne le Transformer à reconstruire les trous (MSE Loss).
Sauvegarde les poids de l'encodeur pour le RL.
```

### Tâches
- [x] Créer classe `CryptoMAE` (Masked Auto-Encoder)
- [x] Implémenter architecture Pre-Norm Transformer (Encoder-only)
- [x] Configurer masquage aléatoire (15% des bougies)
- [x] Implémenter MSE Loss pour reconstruction
- [x] Configurer AdamW avec Learning Rate Scheduler
- [x] Pipeline de sauvegarde des poids pré-entraînés (`weights/pretrained_encoder.pth`)

---

## Phase 3 : Single Agent TQC (La Stratégie v1)

### Technologies Clés
- **TQC (Truncated Quantile Critics)** : Distributional RL
- **FoundationFeatureExtractor** : Encoder pré-entraîné gelé
- **DSR v3** : Differential Sharpe Ratio avec tanh squashing

### Objectif & Justification SOTA
**Stabilité & Baseline**
Un agent unique stable avant de complexifier. Le TQC coupe l'optimisme (FOMO) et le DSR v3 avec tanh garantit des gradients stables.

### Instructions d'Implémentation
```
Charge les poids pré-entraînés (encoder gelé).
Utilise sb3_contrib pour TQC avec top_quantiles_dropped=2.
Reward: DSR avec tanh squashing pour stabilité numérique.
```

### Tâches
- [x] Charger poids pré-entraînés du Transformer
- [x] Configurer TQC avec sb3_contrib (`top_quantiles_dropped=2`)
- [x] Implémenter DSR v3 avec tanh squashing
- [x] Training stable sur 500k steps
- [ ] Évaluation sur données de test (out-of-sample)

---

## Phase 4 : Adversarial Validation (L'Audit)

### Technologies Clés
- **CPCV** : Combinatorial Purged Cross-Validation
- **WFO** : Walk-Forward Optimization
- **Deflated Sharpe Ratio**

### Objectif & Justification SOTA
**Preuve Mathématique**
Vérifie que la performance n'est pas due à la chance. Simule des milliers de découpages temporels pour torturer la stratégie.

### Instructions d'Implémentation
```
Implémente la méthode de Lopez de Prado.
Génère 100 chemins de Backtest avec des permutations de blocs Train/Test.
Calcule la probabilité de sur-apprentissage.
```

### Tâches
- [ ] Implémenter CPCV (Combinatorial Purged Cross-Validation)
- [ ] Configurer Walk-Forward Optimization
- [ ] Calculer Deflated Sharpe Ratio
- [ ] Générer 100 chemins de backtest
- [ ] Calculer probabilité de sur-apprentissage (PBO)
- [ ] Rapport de validation finale

---

## Phase 5 : Ensemble Strategy v2 (Future Version)

### Technologies Clés
- **Mixture of Experts (MoE)** : 3 sous-agents + Gating Network
- **gSDE** : Exploration d'état dépendante
- **Regime-Conditioned Gating** : Pondération selon GMM-HMM

### Objectif & Justification SOTA
**Robustesse & Adaptabilité**
Au lieu d'un seul agent, 3 experts (Sniper, Swing, Risk-Manager) votent. Le Gating Network pondère dynamiquement selon le régime de marché détecté.

### Instructions d'Implémentation
```
Architecture MoE : Le Gating Network pondère les actions des 3 experts
selon le régime détecté par GMM-HMM.
Chaque expert est un TQC spécialisé.
Intégrer gSDE pour exploration fluide.
```

### Tâches
- [ ] Implémenter Expert 1 : Sniper (trading haute fréquence)
- [ ] Implémenter Expert 2 : Swing (positions moyennes)
- [ ] Implémenter Expert 3 : Risk-Manager (gestion du risque)
- [ ] Créer Gating Network pour pondération dynamique
- [ ] Intégrer régimes GMM-HMM dans le Gating
- [ ] Intégrer gSDE pour exploration

---

## Dépendances Principales

```
# requirements.txt additions
hmmlearn>=0.3.0
torch>=2.0.0
sb3-contrib>=2.0.0
stable-baselines3>=2.0.0
scikit-learn>=1.3.0
```

---

## Architecture Globale

```
┌─────────────────────────────────────────────────────────────┐
│                    Phase 1: Les Yeux                        │
│  [Multi-Asset Data] → [Fracdiff] → [GMM-HMM] → [Régimes]   │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   Phase 2: Le Cerveau                       │
│  [MAE Pre-training] → [Transformer Encoder] → [Features]   │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                Phase 3: La Stratégie v1                     │
│  [Encoder Gelé] → [TQC + DSR v3] → [Actions]               │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    Phase 4: L'Audit                         │
│  [CPCV] → [WFO] → [Deflated Sharpe] → [Validation Finale]  │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              Phase 5: La Stratégie v2 (Future)              │
│  [Sniper] ─┐                                                │
│  [Swing]  ─┼─→ [Gating Network] → [TQC + gSDE] → [Actions] │
│  [Risk]   ─┘                                                │
└─────────────────────────────────────────────────────────────┘
```

---

## Références

- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*
- Hoseinzade & Haratizadeh (2019). *CNNpred: CNN-based stock market prediction*
- Kuznetsov et al. (2020). *Controlling Overestimation Bias with Truncated Mixture of Continuous Distributional Quantile Critics*

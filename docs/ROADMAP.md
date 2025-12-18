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
- [ ] Implémenter FFD (Fractional Differentiation) avec fenêtre fixe
- [ ] Configurer pipeline multi-asset (BTC, ETH, SPX, DXY, Funding Rates)
- [ ] Intégrer GMM-HMM avec hmmlearn (3 états, 2 mixtures)
- [ ] Créer fonction `predict_proba` pour probabilités de régime
- [ ] Normalisation avec RobustScaler

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
- [ ] Créer classe `PretrainTrainer`
- [ ] Implémenter architecture Pre-Norm Transformer (Encoder-only)
- [ ] Configurer masquage aléatoire (15% des bougies)
- [ ] Implémenter MSE Loss pour reconstruction
- [ ] Configurer AdamW avec Learning Rate Scheduler
- [ ] Pipeline de sauvegarde des poids pré-entraînés

---

## Phase 3 : Ensemble Strategy (La Stratégie)

### Technologies Clés
- **TQC (Truncated Quantile Critics)** : Distributional RL
- **Mixture of Experts (MoE)** : 3 sous-agents + Gating Network
- **gSDE** : Exploration d'état dépendante

### Objectif & Justification SOTA
**Robustesse & Adaptabilité**
Au lieu d'un seul agent, 3 experts (Sniper, Swing, Risk-Manager) votent. Le TQC coupe l'optimisme (FOMO) et le gSDE assure une exécution fluide.

### Instructions d'Implémentation
```
Charge les poids pré-entraînés.
Architecture MoE : Le Gating Network pondère les actions des 3 experts
selon le régime détecté.
Utilise sb3_contrib pour TQC avec top_quantiles_dropped=2.
```

### Tâches
- [ ] Charger poids pré-entraînés du Transformer
- [ ] Implémenter Expert 1 : Sniper (trading haute fréquence)
- [ ] Implémenter Expert 2 : Swing (positions moyennes)
- [ ] Implémenter Expert 3 : Risk-Manager (gestion du risque)
- [ ] Créer Gating Network pour pondération dynamique
- [ ] Configurer TQC avec sb3_contrib (`top_quantiles_dropped=2`)
- [ ] Intégrer gSDE pour exploration

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
│                  Phase 3: La Stratégie                      │
│  [Sniper] ─┐                                                │
│  [Swing]  ─┼─→ [Gating Network] → [TQC + gSDE] → [Actions] │
│  [Risk]   ─┘                                                │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    Phase 4: L'Audit                         │
│  [CPCV] → [WFO] → [Deflated Sharpe] → [Validation Finale]  │
└─────────────────────────────────────────────────────────────┘
```

---

## Références

- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*
- Hoseinzade & Haratizadeh (2019). *CNNpred: CNN-based stock market prediction*
- Kuznetsov et al. (2020). *Controlling Overestimation Bias with Truncated Mixture of Continuous Distributional Quantile Critics*

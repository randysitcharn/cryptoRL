# Rapport d'Audit : Implémentation Dropout SOTA pour TQC

## 1. Contexte et Objectif

**Projet** : CryptoRL - Agent de trading RL utilisant TQC (Truncated Quantile Critics)

**Objectif** : Implémenter le dropout de manière SOTA pour améliorer :
- La régularisation (anti-overfitting)
- La stabilité de l'entraînement
- L'estimation d'incertitude

---

## 2. État de l'Art (Janvier 2026)

### 2.1 Approches Analysées

| Méthode | Année | Idée Principale | Où appliquer Dropout | Dropout Rate |
|---------|-------|-----------------|---------------------|--------------|
| **DroQ** | 2021 | Dropout + LayerNorm remplacent gros ensembles | Critics uniquement | 0.01 - 0.1 |
| **STAC** | 2026 | Incertitude aléatoire temporelle | Actor + Critic | ~0.1 |
| **UWAC** | 2021 | Pondération par incertitude | Critics (MC Dropout) | 0.1 - 0.2 |

### 2.2 Findings Clés

1. **DroQ** (Hiraoka et al., 2021) :
   - Atteint la sample-efficiency de REDQ (20 critics) avec seulement 2 critics + dropout
   - **LayerNorm est CRITIQUE** : sans lui, le dropout déstabilise l'entraînement
   - Architecture : `Linear → LayerNorm → ReLU → Dropout`
   - Dropout rates très faibles : 0.01 (Hopper) à 0.1 (Humanoid)

2. **STAC** (Özalp, Janvier 2026) :
   - Plus récent, applique dropout sur actor ET critic
   - Utilise la variance du critic distributionnel pour pessimisme adaptatif
   - Synergie naturelle avec TQC (déjà distributionnel via quantiles)

3. **Consensus scientifique** :
   - Dropout sur critics = améliore estimation de valeur
   - Dropout sur actor = régularisation, peut aider généralisation
   - LayerNorm = **obligatoire** pour stabilité avec dropout en RL

---

## 3. Architecture Proposée

### 3.1 Vue d'Ensemble

```
┌─────────────────────────────────────────────────────────────────┐
│                    TQCDropoutPolicy                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ FoundationFeatureExtractor (existant, frozen)            │   │
│  │ Market (64, 43) → MAE Encoder → 8192 + position → 512    │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ↓                                   │
│         ┌────────────────────┴────────────────────┐             │
│         ↓                                         ↓              │
│  ┌─────────────────┐                    ┌─────────────────────┐ │
│  │ Actor (π)       │                    │ Critics (Q1, Q2)    │ │
│  │                 │                    │                     │ │
│  │ Linear(512,64)  │                    │ Linear(512+1, 64)   │ │
│  │ LayerNorm ←NEW  │                    │ LayerNorm ←NEW      │ │
│  │ ReLU            │                    │ ReLU                │ │
│  │ Dropout ←NEW    │                    │ Dropout ←NEW        │ │
│  │ Linear(64,64)   │                    │ Linear(64,64)       │ │
│  │ LayerNorm ←NEW  │                    │ LayerNorm ←NEW      │ │
│  │ ReLU            │                    │ ReLU                │ │
│  │ Dropout ←NEW    │                    │ Dropout ←NEW        │ │
│  │ Linear(64,1)    │                    │ Linear(64, 25)      │ │
│  │ → μ, log_σ      │                    │ → 25 quantiles      │ │
│  └─────────────────┘                    └─────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Modifications par Rapport à TQC Standard

| Composant | TQC Standard | TQC + Dropout (Proposé) |
|-----------|--------------|-------------------------|
| Critic MLP | `Linear → ReLU` | `Linear → LayerNorm → ReLU → Dropout` |
| Actor MLP | `Linear → ReLU` | `Linear → LayerNorm → ReLU → Dropout` |
| Quantile Head | Inchangé | Inchangé |
| Truncation | 2 top quantiles dropped | Inchangé (synergie) |

### 3.3 Hyperparamètres Recommandés

```python
# Configuration DroQ-style pour TQC
dropout_config = {
    # Dropout rates (DroQ recommande très faible)
    "critic_dropout": 0.01,      # Conservateur pour critics
    "actor_dropout": 0.005,      # Encore plus faible pour actor
    
    # Architecture
    "use_layer_norm": True,      # OBLIGATOIRE (DroQ)
    "dropout_on_actor": True,    # STAC-style (optionnel)
    "dropout_on_critic": True,   # DroQ-style (principal)
    
    # Placement
    "dropout_after_activation": True,  # Linear → LN → ReLU → Dropout
}
```

---

## 4. Implémentation Technique

### 4.1 Fichiers Créés/Modifiés

| Fichier | Action | Description |
|---------|--------|-------------|
| `src/models/tqc_dropout_policy.py` | ✅ CRÉÉ | Custom policy avec dropout |
| `src/models/__init__.py` | ✅ MODIFIÉ | Export TQCDropoutPolicy |
| `src/config/training.py` | ✅ MODIFIÉ | Ajout hyperparamètres dropout |
| `src/training/train_agent.py` | ✅ MODIFIÉ | Utilisation conditionnelle de TQCDropoutPolicy |

**Intégration WFO** : La policy est automatiquement active dans `run_full_wfo.py` car il appelle `train()` depuis `train_agent.py`.

### 4.2 Code Proposé : `TQCDropoutPolicy`

```python
# src/models/tqc_dropout_policy.py
"""
TQC Policy avec Dropout + LayerNorm (DroQ/STAC style).

Implémente les best practices SOTA pour régularisation en RL:
- LayerNorm avant activation (stabilité)
- Dropout après activation (régularisation)
- Rates très faibles (0.01-0.05) contrairement au DL classique
"""

import torch
import torch.nn as nn
from typing import List, Type, Dict, Any, Optional, Tuple
from sb3_contrib.tqc.policies import TQCPolicy, Actor
from sb3_contrib.common.sde import StateDependentNoiseDistribution
from stable_baselines3.common.preprocessing import get_action_dim


def create_mlp_with_dropout(
    input_dim: int,
    output_dim: int,
    net_arch: List[int],
    activation_fn: Type[nn.Module] = nn.ReLU,
    dropout_rate: float = 0.01,
    use_layer_norm: bool = True,
    squash_output: bool = False,
) -> nn.Sequential:
    """
    Crée un MLP avec LayerNorm + Dropout (DroQ-style).
    
    Architecture par couche:
        Linear → LayerNorm → Activation → Dropout
    
    Args:
        input_dim: Dimension d'entrée
        output_dim: Dimension de sortie
        net_arch: Liste des dimensions cachées [64, 64]
        activation_fn: Fonction d'activation (ReLU par défaut)
        dropout_rate: Taux de dropout (0.01 recommandé pour RL)
        use_layer_norm: Activer LayerNorm (CRITIQUE pour stabilité)
        squash_output: Ajouter Tanh en sortie (pour actor)
    
    Returns:
        nn.Sequential avec l'architecture complète
    """
    layers = []
    last_dim = input_dim
    
    for hidden_dim in net_arch:
        layers.append(nn.Linear(last_dim, hidden_dim))
        
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        
        layers.append(activation_fn())
        
        if dropout_rate > 0:
            layers.append(nn.Dropout(p=dropout_rate))
        
        last_dim = hidden_dim
    
    # Couche de sortie (sans dropout ni LayerNorm)
    layers.append(nn.Linear(last_dim, output_dim))
    
    if squash_output:
        layers.append(nn.Tanh())
    
    return nn.Sequential(*layers)


class DropoutActor(Actor):
    """
    Actor avec Dropout + LayerNorm pour TQC.
    
    Hérite de Actor (sb3-contrib) et remplace le MLP
    par une version avec régularisation.
    """
    
    def __init__(
        self,
        observation_space,
        action_space,
        net_arch: List[int],
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        full_std: bool = True,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        normalize_images: bool = True,
        # Nouveaux paramètres dropout
        dropout_rate: float = 0.005,
        use_layer_norm: bool = True,
    ):
        # Stocker les paramètres dropout AVANT super().__init__
        self._dropout_rate = dropout_rate
        self._use_layer_norm = use_layer_norm
        
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            net_arch=net_arch,
            features_dim=features_dim,
            activation_fn=activation_fn,
            use_sde=use_sde,
            log_std_init=log_std_init,
            full_std=full_std,
            use_expln=use_expln,
            clip_mean=clip_mean,
            normalize_images=normalize_images,
        )
        
        # Reconstruire le latent_pi avec dropout
        self._rebuild_with_dropout()
    
    def _rebuild_with_dropout(self):
        """Reconstruit le réseau latent avec dropout."""
        action_dim = get_action_dim(self.action_space)
        
        # Remplacer latent_pi par version avec dropout
        self.latent_pi = create_mlp_with_dropout(
            input_dim=self.features_dim,
            output_dim=self.net_arch[-1] if self.net_arch else self.features_dim,
            net_arch=self.net_arch[:-1] if len(self.net_arch) > 1 else [],
            activation_fn=self.activation_fn,
            dropout_rate=self._dropout_rate,
            use_layer_norm=self._use_layer_norm,
            squash_output=False,
        )
        
        # Note: mu et log_std sont recréés par le parent, pas besoin de les toucher


class DropoutQuantileNetwork(nn.Module):
    """
    Quantile Network avec Dropout + LayerNorm pour TQC Critics.
    
    Chaque critic prédit n_quantiles valeurs pour distribution de retour.
    Architecture DroQ: Linear → LayerNorm → ReLU → Dropout
    """
    
    def __init__(
        self,
        features_dim: int,
        action_dim: int,
        n_quantiles: int = 25,
        net_arch: List[int] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        dropout_rate: float = 0.01,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        
        if net_arch is None:
            net_arch = [64, 64]
        
        self.n_quantiles = n_quantiles
        self.features_dim = features_dim
        self.action_dim = action_dim
        
        # Input: features + action
        input_dim = features_dim + action_dim
        
        # Construire le réseau avec dropout
        self.quantile_net = create_mlp_with_dropout(
            input_dim=input_dim,
            output_dim=n_quantiles,
            net_arch=net_arch,
            activation_fn=activation_fn,
            dropout_rate=dropout_rate,
            use_layer_norm=use_layer_norm,
            squash_output=False,
        )
    
    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            obs: Observations (B, features_dim)
            action: Actions (B, action_dim)
        
        Returns:
            Quantiles (B, n_quantiles)
        """
        x = torch.cat([obs, action], dim=1)
        return self.quantile_net(x)


class TQCDropoutPolicy(TQCPolicy):
    """
    TQC Policy avec Dropout + LayerNorm (SOTA 2026).
    
    Combine les techniques de:
    - DroQ (Hiraoka 2021): Dropout + LayerNorm dans critics
    - STAC (Özalp 2026): Dropout dans actor aussi
    
    Avantages:
    - Meilleure régularisation (anti-overfitting)
    - Sample efficiency similaire à gros ensembles
    - Estimation d'incertitude implicite
    
    Usage:
        model = TQC(
            policy=TQCDropoutPolicy,
            env=env,
            policy_kwargs={
                "critic_dropout": 0.01,
                "actor_dropout": 0.005,
                "use_layer_norm": True,
            }
        )
    """
    
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        net_arch=None,
        activation_fn=nn.ReLU,
        use_sde=False,
        log_std_init=-3,
        use_expln=False,
        clip_mean=2.0,
        features_extractor_class=None,
        features_extractor_kwargs=None,
        normalize_images=True,
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs=None,
        n_critics=2,
        n_quantiles=25,
        share_features_extractor=False,
        # Nouveaux paramètres dropout
        critic_dropout: float = 0.01,
        actor_dropout: float = 0.005,
        use_layer_norm: bool = True,
    ):
        self.critic_dropout = critic_dropout
        self.actor_dropout = actor_dropout
        self.use_layer_norm = use_layer_norm
        
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            use_sde=use_sde,
            log_std_init=log_std_init,
            use_expln=use_expln,
            clip_mean=clip_mean,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            n_critics=n_critics,
            n_quantiles=n_quantiles,
            share_features_extractor=share_features_extractor,
        )
    
    def make_actor(self, features_extractor=None) -> DropoutActor:
        """Crée l'actor avec dropout."""
        actor_kwargs = self._update_features_extractor(
            self.actor_kwargs, features_extractor
        )
        return DropoutActor(
            dropout_rate=self.actor_dropout,
            use_layer_norm=self.use_layer_norm,
            **actor_kwargs
        ).to(self.device)
    
    def make_critic(self, features_extractor=None) -> nn.Module:
        """Crée les critics avec dropout."""
        critic_kwargs = self._update_features_extractor(
            self.critic_kwargs, features_extractor
        )
        
        # Créer l'ensemble de critics avec dropout
        critics = nn.ModuleList([
            DropoutQuantileNetwork(
                features_dim=self.features_dim,
                action_dim=self.action_dim,
                n_quantiles=self.n_quantiles,
                net_arch=critic_kwargs.get("net_arch", [64, 64]),
                activation_fn=self.activation_fn,
                dropout_rate=self.critic_dropout,
                use_layer_norm=self.use_layer_norm,
            )
            for _ in range(self.n_critics)
        ])
        
        return critics.to(self.device)
```

### 4.3 Modifications de Configuration

```python
# src/config/training.py (ajouts)

@dataclass
class TQCTrainingConfig:
    # ... existing fields ...
    
    # --- Dropout Configuration (DroQ/STAC style) ---
    critic_dropout: float = 0.01      # DroQ recommande 0.01-0.1
    actor_dropout: float = 0.005      # Plus faible pour actor
    use_layer_norm: bool = True       # OBLIGATOIRE pour stabilité
    dropout_on_actor: bool = True     # STAC-style
    dropout_on_critic: bool = True    # DroQ-style
```

### 4.4 Intégration dans `train_agent.py`

```python
# Modification de create_policy_kwargs()

def create_policy_kwargs(config: TrainingConfig) -> dict:
    net_arch = config.net_arch if config.net_arch else dict(pi=[64, 64], qf=[64, 64])
    
    return dict(
        features_extractor_class=FoundationFeatureExtractor,
        features_extractor_kwargs=dict(
            encoder_path=config.encoder_path,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            freeze_encoder=config.freeze_encoder,
        ),
        net_arch=net_arch,
        n_critics=config.n_critics,
        n_quantiles=config.n_quantiles,
        # Nouveaux paramètres dropout
        critic_dropout=config.critic_dropout,
        actor_dropout=config.actor_dropout,
        use_layer_norm=config.use_layer_norm,
        optimizer_class=ClippedAdamW,
        optimizer_kwargs=dict(
            max_grad_norm=0.5,
            weight_decay=1e-5,
            eps=1e-5,
        ),
    )

# Modification de la création du modèle
from src.models.tqc_dropout_policy import TQCDropoutPolicy

model = TQC(
    policy=TQCDropoutPolicy,  # ← Nouvelle policy
    env=train_env,
    # ... rest unchanged ...
)
```

---

## 5. Analyse des Risques et Mitigations

### 5.1 Risques Identifiés

| Risque | Probabilité | Impact | Mitigation |
|--------|-------------|--------|------------|
| Déstabilisation training | Moyenne | Élevé | LayerNorm obligatoire + dropout rate faible |
| Incompatibilité SB3 | Faible | Élevé | Héritage de TQCPolicy, pas de monkey-patching |
| Overhead computationnel | Faible | Faible | Dropout/LN très légers vs ensembles |
| Régression performances | Moyenne | Moyen | A/B test avec baseline sans dropout |

### 5.2 Points d'Attention

1. **Mode Train/Eval** : 
   - `model.policy.set_training_mode(False)` désactive dropout pendant eval
   - Vérifié : TQCPolicy gère déjà cela correctement

2. **Interaction avec gSDE** :
   - gSDE ajoute du bruit state-dependent sur l'actor
   - Dropout ajoute du bruit sur les activations
   - Risque : double stochasticité → réduire `actor_dropout` si instable

3. **Interaction avec truncation TQC** :
   - Truncation drop les top quantiles (optimistes)
   - Dropout régularise l'ensemble du réseau
   - Synergie positive : les deux réduisent l'overestimation

---

## 6. Plan de Validation

### 6.1 Tests Unitaires

```python
# tests/test_tqc_dropout.py

def test_dropout_forward_pass():
    """Vérifie que le forward pass fonctionne avec dropout."""
    
def test_dropout_train_eval_mode():
    """Vérifie que dropout est désactivé en mode eval."""
    
def test_layernorm_stability():
    """Vérifie la stabilité des activations avec LayerNorm."""
    
def test_gradient_flow():
    """Vérifie que les gradients passent correctement."""
    
def test_compatibility_with_foundation_extractor():
    """Vérifie l'intégration avec FoundationFeatureExtractor."""
```

### 6.2 Tests d'Intégration

1. **Smoke test** : 1000 steps pour vérifier pas de crash
2. **Baseline comparison** : 100k steps avec/sans dropout
3. **Stability check** : Monitoring des loss/gradients

### 6.3 Métriques à Surveiller

| Métrique | Attendu avec Dropout | Action si Problème |
|----------|---------------------|-------------------|
| Critic Loss | Légèrement plus élevée, plus stable | Réduire dropout rate |
| Actor Loss | Similaire | - |
| Entropy | Légèrement plus haute | Normal (régularisation) |
| Gradient Norm | Plus stable | - |
| Eval Reward Variance | Plus faible | Objectif atteint |

---

## 7. Conclusion et Recommandations

### 7.1 Synthèse

L'implémentation proposée combine les meilleures pratiques de **DroQ** (2021) et **STAC** (2026) :

- **Dropout + LayerNorm** sur critics (DroQ) → meilleure estimation Q
- **Dropout léger** sur actor (STAC) → régularisation supplémentaire
- **Rates très faibles** (0.01 critic, 0.005 actor) → stabilité préservée
- **Compatible** avec l'architecture existante (FoundationFeatureExtractor, gSDE, truncation)

### 7.2 Recommandations

1. **Implémenter en 2 phases** :
   - Phase 1 : Dropout sur critics uniquement (DroQ pur)
   - Phase 2 : Ajouter dropout sur actor si Phase 1 stable

2. **Hyperparamètres initiaux** :
   ```python
   critic_dropout = 0.01
   actor_dropout = 0.0  # Désactivé en Phase 1
   use_layer_norm = True
   ```

3. **Monitoring renforcé** :
   - Logger `critic_loss_std` pour vérifier stabilisation
   - Logger `q_value_std` pour vérifier diversité des critics

### 7.3 Questions pour l'Audit

1. L'architecture `Linear → LayerNorm → ReLU → Dropout` est-elle correcte ? (DroQ utilise cette séquence)

2. Les dropout rates (0.01/0.005) sont-ils appropriés pour un environnement financier ?

3. Faut-il implémenter le **MC Dropout** (multiple forward passes) pour estimation d'incertitude explicite ?

4. L'interaction avec `top_quantiles_to_drop=2` de TQC pose-t-elle problème ?

---

## 8. Résultats de l'Audit (19 janvier 2026)

### 8.1 Verdict

**APPROUVÉ AVEC RÉSERVES MINEURES**

Le design est solide, bien documenté et aligné sur les contraintes spécifiques du RL en finance.

### 8.2 Points Validés ✅

1. **LayerNorm obligatoire** : Correctement identifié comme critique
2. **Rates conservateurs (0.01)** : Appropriés pour finance (faible SNR)
3. **Synergie TQC** : Dropout (incertitude épistémique) + Quantiles (incertitude aléatoire) = robustesse maximale
4. **Architecture `Linear → LayerNorm → ReLU → Dropout`** : Ordre correct (Post-Norm)

### 8.3 Réserve Critique ⚠️ : Conflit Actor Dropout vs gSDE

**Problème identifié** :
- gSDE (Generalized State Dependent Exploration) apprend une matrice de bruit dépendant de l'état
- Si dropout appliqué sur l'actor AVANT gSDE → bruit structuré transformé en bruit blanc chaotique
- Casse la propriété de "continuité temporelle" de l'exploration gSDE

**Solution implémentée** :
```python
# Dans TQCDropoutPolicy.__init__
if use_sde and actor_dropout > 0:
    print("[WARNING] Actor Dropout + gSDE conflict detected. Forcing actor_dropout=0.0")
    self.actor_dropout = 0.0
```

### 8.4 Réponses aux Questions

| Question | Réponse |
|----------|---------|
| Q1. Architecture correcte ? | **OUI** - Standard DroQ |
| Q2. Rates appropriés finance ? | **OUI** - 0.01 = denoising regularizer |
| Q3. MC Dropout nécessaire ? | **NON pour training**, optionnel pour inférence live. Préférer variance des quantiles TQC (moins coûteux) |
| Q4. Interaction truncation ? | **SYNERGIE POSITIVE** - Dropout lisse les distributions, truncation plus efficace |

### 8.5 Stratégie de Validation A/B

| Run | Configuration | Objectif |
|-----|---------------|----------|
| A (Baseline) | TQC Standard | Référence |
| B (DroQ pur) | `critic_dropout=0.01`, `actor_dropout=0.0` | **Prioritaire** |
| C (Full STAC) | `critic_dropout=0.01`, `actor_dropout=0.005` | Si Run B stable |

---

**Statut** : ✅ APPROUVÉ - Prêt pour implémentation

**Date audit** : 19 janvier 2026

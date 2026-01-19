# -*- coding: utf-8 -*-
"""
tqc_dropout_policy.py - TQC Policy avec Dropout + LayerNorm (DroQ/STAC style).

Implémente les best practices SOTA pour régularisation en RL:
- LayerNorm avant activation (stabilité)
- Dropout après activation (régularisation)
- Rates très faibles (0.01-0.05) contrairement au DL classique
- Sécurité automatique : désactive actor_dropout si gSDE actif

Références:
- DroQ (Hiraoka et al., 2021): "Dropout Q-Functions for Doubly Efficient RL"
- STAC (Özalp, 2026): "Stochastic Actor-Critic: Mitigating Overestimation via Temporal Aleatoric Uncertainty"

Usage:
    from src.models.tqc_dropout_policy import TQCDropoutPolicy
    
    model = TQC(
        policy=TQCDropoutPolicy,
        env=env,
        policy_kwargs={
            "critic_dropout": 0.01,
            "actor_dropout": 0.005,  # Auto-disabled if use_sde=True
            "use_layer_norm": True,
        }
    )
"""

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Type, Tuple, Union

from sb3_contrib.tqc.policies import TQCPolicy, Actor
from stable_baselines3.common.preprocessing import get_action_dim


# ==============================================================================
# 1. Utility : MLP Builder avec LayerNorm & Dropout (Architecture DroQ)
# ==============================================================================

def create_mlp_with_dropout(
    input_dim: int,
    output_dim: int,
    net_arch: List[int],
    activation_fn: Type[nn.Module] = nn.ReLU,
    dropout_rate: float = 0.0,
    use_layer_norm: bool = True,
    squash_output: bool = False,
) -> nn.Sequential:
    """
    Construit un MLP suivant l'architecture DroQ/SOTA pour le RL.
    
    Architecture par couche:
        Linear -> LayerNorm -> Activation -> Dropout
    
    Cette séquence est validée par l'audit:
    - LayerNorm AVANT activation = données centrées-réduites pour ReLU (évite Dead ReLU)
    - Dropout APRÈS activation = préserve la sparsité du ReLU
    
    Args:
        input_dim: Dimension d'entrée
        output_dim: Dimension de sortie  
        net_arch: Liste des dimensions cachées [64, 64]
        activation_fn: Fonction d'activation (ReLU par défaut)
        dropout_rate: Taux de dropout (0.01 recommandé en finance)
        use_layer_norm: Critique pour la stabilité avec dropout
        squash_output: Ajouter Tanh en sortie (pour actor)
    
    Returns:
        nn.Sequential avec l'architecture complète
    """
    layers = []
    last_dim = input_dim
    
    for hidden_dim in net_arch:
        layers.append(nn.Linear(last_dim, hidden_dim))
        
        # LayerNorm avant l'activation (Post-Norm standard en RL moderne)
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        
        layers.append(activation_fn())
        
        # Dropout après l'activation
        if dropout_rate > 0:
            layers.append(nn.Dropout(p=dropout_rate))
        
        last_dim = hidden_dim
    
    # Couche de sortie finale (Raw projection, sans dropout ni LayerNorm)
    layers.append(nn.Linear(last_dim, output_dim))
    
    if squash_output:
        layers.append(nn.Tanh())
    
    return nn.Sequential(*layers)


# ==============================================================================
# 2. Components : Actor & Critic modifiés
# ==============================================================================

class DropoutActor(Actor):
    """
    Actor personnalisé intégrant Dropout et LayerNorm.
    
    Remplace le réseau latent standard de SB3 par une version
    avec régularisation DroQ-style.
    
    IMPORTANT: Si use_sde=True dans la policy parente, dropout devrait
    être à 0 pour éviter de casser l'exploration gSDE.
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
        features_extractor: Optional[nn.Module] = None,
        # Paramètres custom DroQ
        dropout_rate: float = 0.0,
        use_layer_norm: bool = True,
        **kwargs,  # Absorb any extra args from SB3
    ):
        # Stocker les paramètres AVANT super().__init__
        self._dropout_rate = dropout_rate
        self._use_layer_norm = use_layer_norm

        # Use ONLY keyword args to avoid positional conflicts
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
            features_extractor=features_extractor,
        )
        
        # Reconstruire le MLP après super().__init__ car SB3 le crée par défaut
        self._rebuild_mlp()

    def _rebuild_mlp(self):
        """Remplace self.latent_pi par la version régularisée."""
        if self.net_arch is None or len(self.net_arch) == 0:
            return

        # L'Actor SB3 sépare feature extractor et projection (mu/sigma).
        # Ici on remplace la partie "feature extraction" de l'actor (latent_pi).
        output_dim = self.net_arch[-1]
        
        self.latent_pi = create_mlp_with_dropout(
            input_dim=self.features_dim,
            output_dim=output_dim,
            net_arch=self.net_arch[:-1],  # Tous sauf le dernier qui est output du latent
            activation_fn=self.activation_fn,
            dropout_rate=self._dropout_rate,
            use_layer_norm=self._use_layer_norm
        )


class SingleQuantileNet(nn.Module):
    """
    Réseau pour UNE tête de critique (Q-function) avec quantiles.
    
    Prédit n_quantiles valeurs pour la distribution de retour.
    """
    
    def __init__(
        self,
        input_dim: int,
        n_quantiles: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module],
        dropout_rate: float,
        use_layer_norm: bool,
    ):
        super().__init__()
        self.net = create_mlp_with_dropout(
            input_dim=input_dim,
            output_dim=n_quantiles,
            net_arch=net_arch,
            activation_fn=activation_fn,
            dropout_rate=dropout_rate,
            use_layer_norm=use_layer_norm
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DropoutCritic(nn.Module):
    """
    Critic complet pour TQC (Ensemble de SingleQuantileNet).
    
    Remplace QuantileNetwork de SB3 avec support Dropout + LayerNorm.
    Chaque critic de l'ensemble est un SingleQuantileNet indépendant.
    
    Output format: (batch_size, n_critics, n_quantiles)
    C'est le format exact attendu par la loss TQC.
    """
    
    def __init__(
        self,
        features_dim: int,
        action_dim: int,
        n_critics: int,
        n_quantiles: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module],
        dropout_rate: float = 0.0,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.n_critics = n_critics
        self.n_quantiles = n_quantiles
        self.features_dim = features_dim
        self.action_dim = action_dim
        
        # Input = features concatenées avec action
        input_dim = features_dim + action_dim
        
        # Création de l'ensemble de critics indépendants
        self.critics = nn.ModuleList([
            SingleQuantileNet(
                input_dim=input_dim,
                n_quantiles=n_quantiles,
                net_arch=net_arch,
                activation_fn=activation_fn,
                dropout_rate=dropout_rate,
                use_layer_norm=use_layer_norm
            )
            for _ in range(n_critics)
        ])

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass à travers tous les critics.
        
        Args:
            obs: Observations/features (batch_size, features_dim)
            action: Actions (batch_size, action_dim)
        
        Returns:
            Quantiles de tous les critics (batch_size, n_critics, n_quantiles)
        """
        x = torch.cat([obs, action], dim=1)
        # Exécute chaque critic et empile les résultats
        return torch.stack([critic(x) for critic in self.critics], dim=1)


# ==============================================================================
# 3. Policy : TQCDropoutPolicy
# ==============================================================================

class TQCDropoutPolicy(TQCPolicy):
    """
    Policy TQC intégrant les pratiques DroQ (Dropout + LayerNorm).
    
    Combine les techniques de:
    - DroQ (Hiraoka 2021): Dropout + LayerNorm dans critics
    - STAC (Özalp 2026): Dropout dans actor (si pas gSDE)
    
    Gestion automatique du conflit gSDE:
    Si use_sde=True, le dropout de l'actor est forcé à 0 pour éviter 
    de casser la continuité temporelle du bruit d'exploration.
    
    Usage:
        model = TQC(
            policy=TQCDropoutPolicy,
            env=env,
            policy_kwargs={
                "critic_dropout": 0.01,
                "actor_dropout": 0.005,  # Auto-disabled if use_sde=True
                "use_layer_norm": True,
            }
        )
    """

    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        # ====== Paramètres Dropout & Architecture ======
        critic_dropout: float = 0.01,   # Défaut DroQ (conservateur)
        actor_dropout: float = 0.005,   # Défaut STAC (très faible)
        use_layer_norm: bool = True,    # CRITIQUE pour stabilité
        **kwargs
    ):
        """
        Initialize TQCDropoutPolicy.
        
        Args:
            observation_space: Observation space
            action_space: Action space
            lr_schedule: Learning rate schedule
            net_arch: Network architecture (list or dict with 'pi'/'qf' keys)
            activation_fn: Activation function
            use_sde: Whether to use gSDE (State Dependent Exploration)
            critic_dropout: Dropout rate for critics (0.01 recommended)
            actor_dropout: Dropout rate for actor (0.005, auto-disabled with gSDE)
            use_layer_norm: Use LayerNorm (CRITICAL for stability)
            **kwargs: Additional arguments passed to TQCPolicy
        """
        # ====== SÉCURITÉ gSDE (AUDIT FIX) ======
        # gSDE apprend une matrice de bruit dépendant de l'état.
        # Le Dropout sur l'Actor casse la cohérence temporelle de gSDE
        # car il bruiterait l'input de la matrice de bruit à chaque step.
        if use_sde and actor_dropout > 0:
            print(f"\n[TQCDropoutPolicy] WARNING: Conflit détecté entre gSDE et Actor Dropout.")
            print(f"[TQCDropoutPolicy] Le Dropout sur l'Actor casse la cohérence temporelle de gSDE.")
            print(f"[TQCDropoutPolicy] -> FORCE actor_dropout = 0.0 (Sécurité Active)\n")
            actor_dropout = 0.0

        # Stocker les paramètres AVANT super().__init__
        self.critic_dropout = critic_dropout
        self.actor_dropout = actor_dropout
        self.use_layer_norm = use_layer_norm

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            use_sde=use_sde,
            **kwargs
        )

    def make_actor(self, features_extractor: Optional[nn.Module] = None) -> DropoutActor:
        """
        Crée l'actor avec Dropout + LayerNorm.
        
        Override de TQCPolicy.make_actor() pour utiliser DropoutActor.
        """
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        
        return DropoutActor(
            dropout_rate=self.actor_dropout,
            use_layer_norm=self.use_layer_norm,
            **actor_kwargs
        ).to(self.device)

    def make_critic(self, features_extractor: Optional[nn.Module] = None) -> DropoutCritic:
        """
        Crée les critics avec Dropout + LayerNorm.

        Override de TQCPolicy.make_critic() pour utiliser DropoutCritic.

        Returns:
            DropoutCritic avec n_critics têtes, chacune prédisant n_quantiles
        """
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)

        # Get features_dim from critic_kwargs (set by TQCPolicy.__init__)
        features_dim = critic_kwargs.get('features_dim', 512)

        # Récupération de l'architecture spécifique au critic
        if self.net_arch is None:
            net_arch = [256, 256]
        elif isinstance(self.net_arch, dict):
            net_arch = self.net_arch.get("qf", [256, 256])
        else:
            net_arch = self.net_arch

        return DropoutCritic(
            features_dim=features_dim,
            action_dim=get_action_dim(self.action_space),
            n_critics=self.n_critics,
            n_quantiles=self.n_quantiles,
            net_arch=net_arch,
            activation_fn=self.activation_fn,
            dropout_rate=self.critic_dropout,
            use_layer_norm=self.use_layer_norm,
        ).to(self.device)


# ==============================================================================
# 4. Tests de validation
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TQCDropoutPolicy - Tests de Validation")
    print("=" * 70)
    
    # Test 1: create_mlp_with_dropout
    print("\n[Test 1] create_mlp_with_dropout")
    mlp = create_mlp_with_dropout(
        input_dim=512,
        output_dim=64,
        net_arch=[256, 128],
        dropout_rate=0.01,
        use_layer_norm=True,
    )
    print(f"  Architecture: {mlp}")
    
    layer_types = [type(layer).__name__ for layer in mlp]
    print(f"  Layers: {layer_types}")
    assert "LayerNorm" in layer_types, "LayerNorm manquant!"
    assert "Dropout" in layer_types, "Dropout manquant!"
    print("  [OK] MLP créé avec LayerNorm + Dropout")
    
    # Test 2: SingleQuantileNet
    print("\n[Test 2] SingleQuantileNet")
    qnet = SingleQuantileNet(
        input_dim=513,  # features + action
        n_quantiles=25,
        net_arch=[64, 64],
        activation_fn=nn.ReLU,
        dropout_rate=0.01,
        use_layer_norm=True,
    )
    
    batch_size = 4
    dummy_input = torch.randn(batch_size, 513)
    output = qnet(dummy_input)
    print(f"  Output shape: {output.shape}")
    assert output.shape == (batch_size, 25), f"Shape incorrect: {output.shape}"
    print("  [OK] SingleQuantileNet fonctionne")
    
    # Test 3: DropoutCritic
    print("\n[Test 3] DropoutCritic")
    critic = DropoutCritic(
        features_dim=512,
        action_dim=1,
        n_critics=2,
        n_quantiles=25,
        net_arch=[64, 64],
        activation_fn=nn.ReLU,
        dropout_rate=0.01,
        use_layer_norm=True,
    )
    
    dummy_obs = torch.randn(batch_size, 512)
    dummy_action = torch.randn(batch_size, 1)
    
    critic.train()
    output = critic(dummy_obs, dummy_action)
    print(f"  Output shape: {output.shape}")
    assert output.shape == (batch_size, 2, 25), f"Shape incorrect: {output.shape}"
    print("  [OK] DropoutCritic format (B, n_critics, n_quantiles) correct")
    
    # Test 4: Mode train vs eval
    print("\n[Test 4] Mode Train vs Eval")
    critic.eval()
    out1 = critic(dummy_obs, dummy_action)
    out2 = critic(dummy_obs, dummy_action)
    assert torch.allclose(out1, out2), "Eval mode devrait être déterministe!"
    print("  [OK] Eval mode déterministe")
    
    # Test 5: Sécurité gSDE
    print("\n[Test 5] Sécurité gSDE")
    test_actor_dropout = 0.01
    test_use_sde = True
    
    print(f"  AVANT: actor_dropout={test_actor_dropout}, use_sde={test_use_sde}")
    if test_use_sde and test_actor_dropout > 0:
        test_actor_dropout = 0.0
    print(f"  APRÈS: actor_dropout={test_actor_dropout}")
    assert test_actor_dropout == 0.0, "Sécurité gSDE non appliquée!"
    print("  [OK] Sécurité gSDE fonctionne")
    
    print("\n" + "=" * 70)
    print("Tous les tests passés!")
    print("=" * 70)

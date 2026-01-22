# -*- coding: utf-8 -*-
"""
ensemble.py - Ensemble RL Policy Aggregation for TQC.

Implements SOTA ensemble techniques for robust trading:
- Multi-seed aggregation
- Confidence-weighted voting via TQC quantile spread
- Agreement-based action filtering
- OOD detection and conservative fallback

References:
- Ensemble RL through Classifier Models (arXiv:2502.17518)
- DroQ (Hiraoka 2021) - implicit ensemble via dropout
- TQC (Kuznetsov 2020) - distributional RL with quantiles

Author: CryptoRL Team
Date: 2026-01-22
"""

import os
import json
import numpy as np
import torch
from typing import List, Optional, Tuple, Dict, Any, Literal, Union, TYPE_CHECKING
from pathlib import Path
from dataclasses import dataclass, field, asdict

if TYPE_CHECKING:
    from src.config import TQCTrainingConfig

from sb3_contrib import TQC
from stable_baselines3.common.vec_env import VecEnv


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class EnsembleConfig:
    """Configuration for Ensemble RL training and inference."""

    # === Ensemble Composition ===
    n_members: int = 3
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456])

    # === Aggregation ===
    aggregation: Literal['mean', 'median', 'confidence', 'conservative', 'pessimistic_bound'] = 'confidence'
    # Note: 'vote' déprécié (perd l'amplitude en trading continu) - utiliser 'median'
    confidence_epsilon: float = 1e-6  # Prevent division by zero

    # === Softmax Temperature (Audit Gemini) ===
    # Plus τ est petit, plus les modèles incertains sont "tués"
    softmax_temperature: float = 1.0  # τ=1.0 standard, τ=0.5 agressif

    # === Spread Calibration (Audit Gemini) ===
    # Normalise le spread par EMA pour éviter la timidité en haute volatilité
    calibrate_spread: bool = True
    spread_ema_alpha: float = 0.01  # EMA decay pour moyenne mobile du spread

    # === Pessimistic Scaling (Audit SOTA v1.2) ===
    # Réduit la position quand le désaccord entre membres est fort
    apply_pessimistic_scaling: bool = True  # Appliqué après toute méthode d'agrégation
    risk_aversion: float = 1.0  # k factor: 1.0=standard, 2.0=très conservateur, 0.5=agressif
    min_scaling: float = 0.1  # Ne jamais réduire en dessous de 10% de la position

    # === OOD Detection (Audit SOTA v1.3) ===
    # Détection Out-of-Distribution pour éviter les pertes catastrophiques
    enable_ood_detection: bool = True
    ood_threshold: float = 2.5  # Z-score seuil pour déclarer OOD
    ood_warning_threshold: float = 1.5  # Z-score pour réduction préventive
    fallback_action: float = 0.0  # Action en mode OOD (0 = Hold)
    fallback_leverage_scale: float = 0.25  # Réduire à 25% si proche OOD
    ood_history_window: int = 500  # Fenêtre pour statistiques spread

    # === Agreement Filtering ===
    min_agreement: float = 0.0  # Min agreement to act (0 = always act)
    disagreement_action: float = 0.0  # Action when disagreement (0 = hold)

    # === Training ===
    parallel_gpus: List[int] = field(default_factory=lambda: [0, 1])
    shared_encoder: bool = True
    shared_replay_buffer: bool = False  # Mitigation OOM (Audit Gemini)

    # === Forced Diversity (Audit Gemini) ===
    # Varier légèrement les hyperparamètres entre membres
    use_diverse_hyperparams: bool = True
    gamma_range: Tuple[float, float] = (0.94, 0.96)  # [0.94, 0.95, 0.96]
    lr_range: Tuple[float, float] = (5e-5, 2e-4)     # [5e-5, 1e-4, 2e-4]

    # === Inference ===
    use_ema_weights: bool = True  # Use EMA weights for inference
    deterministic: bool = True  # Disable exploration noise

    # === Memory Management & I/O Performance ===
    # IMPORTANT: Lazy loading recommandé pour éviter saturation RAM
    # 3 modèles TQC .zip = ~30MB chacun en RAM + GPU tensors
    preload_models: bool = False  # False = lazy loading (charge à la demande)
    unload_after_predict: bool = False  # True = libère GPU après chaque predict (lent mais économe)
    device: str = 'cuda'

    def to_json(self, path: str):
        """Save config to JSON."""
        # Convert tuples to lists for JSON serialization
        data = asdict(self)
        data['gamma_range'] = list(data['gamma_range'])
        data['lr_range'] = list(data['lr_range'])
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> 'EnsembleConfig':
        """Load config from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)
        # Convert lists back to tuples
        if 'gamma_range' in data:
            data['gamma_range'] = tuple(data['gamma_range'])
        if 'lr_range' in data:
            data['lr_range'] = tuple(data['lr_range'])
        return cls(**data)


# =============================================================================
# Ensemble Policy
# =============================================================================

class EnsemblePolicy:
    """
    Ensemble of TQC policies for robust trading decisions.

    Aggregates predictions from multiple TQC models trained with different
    seeds. Uses TQC's quantile spread for confidence-weighted aggregation.

    Features:
    - Multiple aggregation methods (mean, median, confidence, conservative, pessimistic_bound)
    - Confidence estimation via quantile spread (TQC-native)
    - Agreement-based action filtering
    - OOD detection with conservative fallback
    - GPU memory management

    Example:
        >>> ensemble = EnsemblePolicy(
        ...     model_paths=['tqc_0.zip', 'tqc_1.zip', 'tqc_2.zip'],
        ...     config=EnsembleConfig(aggregation='confidence')
        ... )
        >>> action, info = ensemble.predict(obs)
        >>> print(f"Action: {action}, Confidence: {info['ensemble_confidence']}")
    """

    def __init__(
        self,
        model_paths: List[str],
        config: Optional[EnsembleConfig] = None,
        verbose: int = 0,
    ):
        """
        Initialize the ensemble policy.

        Args:
            model_paths: List of paths to TQC model .zip files.
            config: Ensemble configuration. Uses defaults if None.
            verbose: Verbosity level (0=silent, 1=info, 2=debug).
        """
        self.config = config or EnsembleConfig()
        self.verbose = verbose
        self.device = torch.device(
            self.config.device if torch.cuda.is_available() else 'cpu'
        )

        # Validate paths
        self.model_paths = []
        for path in model_paths:
            if not Path(path).exists():
                raise FileNotFoundError(f"Model not found: {path}")
            self.model_paths.append(path)

        # Load models
        self.models: List[TQC] = []
        if self.config.preload_models:
            self._load_models()

        # OOD detection state
        self._spread_history: List[float] = []
        self._spread_ema: Optional[float] = None

        if self.verbose > 0:
            print(f"[Ensemble] Initialized with {len(self.model_paths)} models")
            print(f"           Aggregation: {self.config.aggregation}")
            print(f"           Device: {self.device}")
            print(f"           OOD Detection: {self.config.enable_ood_detection}")

    def _load_models(self):
        """Load all models into memory."""
        self.models = []
        for i, path in enumerate(self.model_paths):
            if self.verbose > 1:
                print(f"  Loading model {i+1}/{len(self.model_paths)}: {path}")

            model = TQC.load(path, device=self.device)
            model.policy.set_training_mode(False)  # Disable dropout
            self.models.append(model)

        if self.verbose > 0:
            print(f"[Ensemble] Loaded {len(self.models)} models")

    def _ensure_loaded(self):
        """Lazy loading: ensure models are loaded before prediction."""
        if not self.models:
            self._load_models()

    def compute_ood_score(self, obs: Union[Dict[str, np.ndarray], np.ndarray]) -> float:
        """
        Detect if the observation is out-of-distribution.

        Uses spread anomaly as OOD proxy: if average spread across models
        is abnormally high, the observation may be from an unknown regime.

        Args:
            obs: Observation.

        Returns:
            Z-score of current spread vs historical spread.
        """
        self._ensure_loaded()

        # Compute spread for each model
        spreads = [self._get_quantile_spread(m, obs).mean() for m in self.models]
        avg_spread = np.mean(spreads)

        # Update history
        self._spread_history.append(avg_spread)

        # Keep history bounded
        if len(self._spread_history) > self.config.ood_history_window:
            self._spread_history = self._spread_history[-self.config.ood_history_window:]

        # Compute z-score
        if len(self._spread_history) > 100:
            mean_spread = np.mean(self._spread_history[-100:])
            std_spread = np.std(self._spread_history[-100:])
            z_score = (avg_spread - mean_spread) / (std_spread + 1e-6)
            return float(z_score)

        return 0.0

    def predict_with_safety(
        self,
        obs: Union[Dict[str, np.ndarray], np.ndarray],
        deterministic: Optional[bool] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Predict action with OOD safety checks.

        If OOD score exceeds threshold, falls back to conservative action.

        Args:
            obs: Observation.
            deterministic: Override config.deterministic if provided.

        Returns:
            Tuple of (action, info_dict) with OOD status.
        """
        if not self.config.enable_ood_detection:
            return self.predict(obs, deterministic)

        ood_score = self.compute_ood_score(obs)

        if ood_score > self.config.ood_threshold:
            # Mode Survie : réduire drastiquement l'exposition
            if self.verbose > 0:
                print(f"[Ensemble] OOD FALLBACK triggered (z={ood_score:.2f})")

            action = np.array([[self.config.fallback_action]])
            return action, {
                'mode': 'OOD_FALLBACK',
                'ood_score': ood_score,
                'action_override': True,
                'ensemble_std': 0.0,
                'ensemble_confidence': 0.0,
                'ensemble_agreement': 1.0,
                'n_models': len(self.models),
            }

        # Mode normal
        action, info = self.predict(obs, deterministic)

        # Scaling additionnel si proche du seuil OOD
        if ood_score > self.config.ood_warning_threshold:
            safety_scale = self.config.fallback_leverage_scale
            action = action * safety_scale
            info['ood_scaling'] = safety_scale
            info['mode'] = 'OOD_WARNING'
        else:
            info['mode'] = 'NORMAL'

        info['ood_score'] = ood_score
        return action, info

    def predict(
        self,
        obs: Union[Dict[str, np.ndarray], np.ndarray],
        deterministic: Optional[bool] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Predict action using ensemble aggregation.

        Args:
            obs: Observation (dict with 'market', 'position', 'w_cost' or flat array).
            deterministic: Override config.deterministic if provided.

        Returns:
            Tuple of (action, info_dict) where info contains:
                - ensemble_std: Standard deviation across members (disagreement)
                - ensemble_confidence: Aggregated confidence (higher = more certain)
                - member_actions: Individual actions from each member
                - member_confidences: Individual confidences (if confidence aggregation)
        """
        self._ensure_loaded()

        deterministic = deterministic if deterministic is not None else self.config.deterministic
        n_models = len(self.models)

        # Collect predictions from all models
        actions = []
        quantile_spreads = []

        for model in self.models:
            action, _ = model.predict(obs, deterministic=deterministic)
            actions.append(action)

            # Get quantile spread for confidence weighting
            if self.config.aggregation in ['confidence', 'pessimistic_bound']:
                spread = self._get_quantile_spread(model, obs)
                quantile_spreads.append(spread)

        # Stack actions: (n_models, batch_size, action_dim)
        actions_stack = np.stack(actions, axis=0)

        # Compute aggregation
        final_action, weights = self._aggregate(actions_stack, quantile_spreads)

        # Apply pessimistic scaling if enabled
        if self.config.apply_pessimistic_scaling:
            action_std = np.std(actions_stack, axis=0)
            scaling_factor = np.clip(
                1.0 - (self.config.risk_aversion * action_std),
                self.config.min_scaling,
                1.0
            )
            final_action = final_action * scaling_factor

        # Check agreement (optional filtering)
        agreement = self._compute_agreement(actions_stack)
        if self.config.min_agreement > 0 and agreement < self.config.min_agreement:
            final_action = np.full_like(final_action, self.config.disagreement_action)

        # Build info dict
        info = {
            'ensemble_std': float(np.std(actions_stack, axis=0).mean()),
            'ensemble_confidence': float(1.0 / (np.std(actions_stack, axis=0).mean() + 1e-6)),
            'ensemble_agreement': float(agreement),
            'member_actions': actions_stack.tolist(),
            'n_models': n_models,
        }

        if weights is not None:
            info['member_weights'] = weights.tolist()
            info['mean_weight'] = float(weights.mean())
            info['weight_std'] = float(weights.std())

        if self.config.apply_pessimistic_scaling:
            info['pessimistic_scaling'] = float(scaling_factor.mean())

        return final_action, info

    def _aggregate(
        self,
        actions: np.ndarray,
        quantile_spreads: List[np.ndarray],
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Aggregate actions from ensemble members.

        Args:
            actions: Stacked actions (n_models, batch_size, action_dim).
            quantile_spreads: List of quantile spreads per model.

        Returns:
            Tuple of (final_action, weights) where weights is None for some methods.
        """
        method = self.config.aggregation

        if method == 'mean':
            return np.mean(actions, axis=0), None

        elif method == 'median':
            return np.median(actions, axis=0), None

        elif method == 'confidence':
            # === Softmax Temperature Weighting (Audit Gemini) ===
            spreads = np.stack(quantile_spreads, axis=0)  # (n_models, batch_size)

            # Calibrate spread by EMA if enabled (prevents timidity in high vol)
            if self.config.calibrate_spread and self._spread_ema is not None:
                spreads_norm = spreads / (self._spread_ema + self.config.confidence_epsilon)
            else:
                spreads_norm = spreads
                # Update EMA for next call
                if self.config.calibrate_spread:
                    current_mean = spreads.mean()
                    if self._spread_ema is None:
                        self._spread_ema = current_mean
                    else:
                        alpha = self.config.spread_ema_alpha
                        self._spread_ema = alpha * current_mean + (1 - alpha) * self._spread_ema

            # Softmax with temperature: exp(-spread/τ) / Σ(exp(-spread/τ))
            # Lower spread = higher confidence = higher weight
            tau = self.config.softmax_temperature
            log_weights = -spreads_norm / tau
            # Numerical stability: subtract max before exp
            log_weights = log_weights - log_weights.max(axis=0, keepdims=True)
            weights = np.exp(log_weights)
            weights = weights / weights.sum(axis=0, keepdims=True)  # Normalize

            # Weighted average: (n_models, batch, 1) * (n_models, batch, action_dim)
            final_action = np.sum(
                actions * weights[:, :, np.newaxis],
                axis=0
            )
            return final_action, weights

        elif method == 'pessimistic_bound':
            # Pessimistic Bound: reduce position when disagreement is strong
            mean_action = np.mean(actions, axis=0)
            std_action = np.std(actions, axis=0)

            # k = facteur d'aversion au risque
            k = self.config.risk_aversion
            scaling_factor = np.clip(1.0 - (k * std_action), 0.0, 1.0)
            final_action = mean_action * scaling_factor

            return final_action, None

        elif method == 'conservative':
            # Select action closest to 0 (most risk-averse)
            abs_actions = np.abs(actions)
            # For each batch element, find model with smallest |action|
            min_idx = np.argmin(abs_actions.mean(axis=-1), axis=0)  # (batch,)

            final_action = np.zeros_like(actions[0])
            for i, idx in enumerate(min_idx):
                final_action[i] = actions[idx, i]

            return final_action, None

        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    def _get_quantile_spread(
        self,
        model: TQC,
        obs: Union[Dict[str, np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """
        Extract quantile spread from TQC critic for confidence estimation.

        The spread (max - min quantile) indicates uncertainty:
        - Low spread = high confidence
        - High spread = low confidence

        Args:
            model: TQC model.
            obs: Observation.

        Returns:
            Spread per batch element (batch_size,).
        """
        with torch.no_grad():
            # Convert obs to tensor dict
            if isinstance(obs, dict):
                obs_tensor = {
                    k: torch.tensor(v, device=self.device, dtype=torch.float32)
                    for k, v in obs.items()
                }
            else:
                obs_tensor = torch.tensor(obs, device=self.device, dtype=torch.float32)

            # Ensure batch dimension
            if isinstance(obs_tensor, dict):
                for k in obs_tensor:
                    if obs_tensor[k].ndim == 1:
                        obs_tensor[k] = obs_tensor[k].unsqueeze(0)
            else:
                if obs_tensor.ndim == 1:
                    obs_tensor = obs_tensor.unsqueeze(0)

            try:
                # Get features from extractor
                features = model.policy.extract_features(
                    obs_tensor,
                    model.policy.features_extractor
                )

                # Get action from actor (mean for stochastic)
                action_dist = model.policy.actor(features)
                if hasattr(action_dist, 'mean'):
                    action = action_dist.mean
                else:
                    action = action_dist

                # Get quantile values from all critics
                # TQC stores critics as quantile_critics in critic module
                all_quantiles = []
                critic = model.critic

                # Handle different TQC implementations
                if hasattr(critic, 'quantile_critics'):
                    # sb3-contrib structure
                    for qf in critic.quantile_critics:
                        q_values = qf(features, action)
                        all_quantiles.append(q_values)
                else:
                    # Try direct access (qf0, qf1, ...)
                    n_critics = getattr(model.critic, 'n_critics', 2)
                    for i in range(n_critics):
                        qf = getattr(critic, f'qf{i}', None)
                        if qf is not None:
                            q_values = qf(features, action)
                            all_quantiles.append(q_values)

                if not all_quantiles:
                    # Fallback: return uniform spread
                    batch_size = features.shape[0] if hasattr(features, 'shape') else 1
                    return np.ones(batch_size) * 0.1

                # Concatenate all quantiles
                all_q = torch.cat(all_quantiles, dim=-1)  # (batch, total_quantiles)

                # Compute spread (max - min)
                spread = (all_q.max(dim=-1)[0] - all_q.min(dim=-1)[0]).cpu().numpy()

                return spread

            except Exception as e:
                if self.verbose > 1:
                    print(f"[Ensemble] Warning: Could not compute quantile spread: {e}")
                # Fallback
                return np.ones(1) * 0.1

    def _compute_agreement(self, actions: np.ndarray) -> float:
        """
        Compute agreement ratio across ensemble members.

        Agreement is measured as the inverse of action variance.
        High agreement = all models predict similar actions.

        Args:
            actions: Stacked actions (n_models, batch_size, action_dim).

        Returns:
            Agreement ratio in [0, 1] where 1 = perfect agreement.
        """
        # Standard deviation across models
        std = np.std(actions, axis=0).mean()

        # Convert to agreement (inversely proportional to std)
        # Max reasonable std is ~1.0 (action range is [-1, 1])
        agreement = np.clip(1.0 - std, 0.0, 1.0)

        return agreement

    def get_individual_predictions(
        self,
        obs: Union[Dict[str, np.ndarray], np.ndarray],
        deterministic: bool = True,
    ) -> List[Tuple[np.ndarray, Dict]]:
        """
        Get predictions from each ensemble member individually.

        Useful for ablation studies and debugging.

        Args:
            obs: Observation.
            deterministic: Use deterministic policy.

        Returns:
            List of (action, info) tuples, one per member.
        """
        self._ensure_loaded()

        results = []
        for i, model in enumerate(self.models):
            action, _ = model.predict(obs, deterministic=deterministic)
            spread = self._get_quantile_spread(model, obs)

            info = {
                'member_idx': i,
                'seed': self.config.seeds[i] if i < len(self.config.seeds) else None,
                'quantile_spread': float(spread.mean()),
                'confidence': float(1.0 / (spread.mean() + 1e-6)),
            }
            results.append((action, info))

        return results

    def close(self):
        """Release GPU memory."""
        for model in self.models:
            del model
        self.models = []

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if self.verbose > 0:
            print("[Ensemble] Models unloaded, GPU memory released")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# =============================================================================
# Ensemble Trainer
# =============================================================================

class EnsembleTrainer:
    """
    Trains multiple TQC models for ensemble.

    Supports:
    - Sequential training (single GPU)
    - Parallel training (multi-GPU)
    - Forced diversity via gamma/LR variation

    Example:
        >>> trainer = EnsembleTrainer(
        ...     base_config=tqc_config,
        ...     ensemble_config=EnsembleConfig(n_members=3)
        ... )
        >>> model_paths = trainer.train(output_dir='weights/ensemble/')
    """

    def __init__(
        self,
        base_config: 'TQCTrainingConfig',
        ensemble_config: Optional[EnsembleConfig] = None,
        verbose: int = 1,
    ):
        """
        Initialize ensemble trainer.

        Args:
            base_config: Base TQCTrainingConfig for all members.
            ensemble_config: Ensemble configuration.
            verbose: Verbosity level.
        """
        self.base_config = base_config
        self.config = ensemble_config or EnsembleConfig()
        self.verbose = verbose

        # Validate
        if self.config.n_members > len(self.config.seeds):
            raise ValueError(
                f"Need {self.config.n_members} seeds, got {len(self.config.seeds)}"
            )

    def train_sequential(self, output_dir: str) -> List[str]:
        """
        Train ensemble members sequentially (single GPU).

        Args:
            output_dir: Directory to save models.

        Returns:
            List of paths to trained models.
        """
        import copy
        import gc
        from src.training.train_agent import train
        from src.config import SEED

        os.makedirs(output_dir, exist_ok=True)
        model_paths = []
        all_metrics = []

        for i in range(self.config.n_members):
            seed = self.config.seeds[i]

            if self.verbose > 0:
                print(f"\n{'='*60}")
                print(f"Training Ensemble Member {i+1}/{self.config.n_members}")
                print(f"Seed: {seed}")
                print(f"{'='*60}")

            # Clone and modify config
            config = copy.deepcopy(self.base_config)
            config.seed = seed
            config.save_path = os.path.join(output_dir, f"tqc_seed_{seed}.zip")
            config.checkpoint_dir = os.path.join(output_dir, f"checkpoints_seed_{seed}/")
            config.name = f"ensemble_seed_{seed}"

            # === Forced Diversity (Audit Gemini) ===
            # Varier légèrement gamma et LR entre membres pour diversité structurelle
            if self.config.use_diverse_hyperparams:
                gamma_min, gamma_max = self.config.gamma_range
                lr_min, lr_max = self.config.lr_range
                n = self.config.n_members

                # Linspace pour distribution uniforme
                config.gamma = gamma_min + (gamma_max - gamma_min) * i / max(n - 1, 1)
                config.learning_rate = lr_min + (lr_max - lr_min) * i / max(n - 1, 1)

                if self.verbose > 0:
                    print(f"  [Diversity] gamma={config.gamma:.4f}, lr={config.learning_rate:.2e}")

            # Ensure directories exist
            os.makedirs(config.checkpoint_dir, exist_ok=True)

            # Train
            model, metrics = train(config, use_batch_env=True)

            model_paths.append(config.save_path)
            all_metrics.append(metrics)

            # Cleanup
            del model
            torch.cuda.empty_cache()
            gc.collect()

        # Save ensemble config
        self.config.to_json(os.path.join(output_dir, "ensemble_config.json"))

        # Save aggregated metrics
        self._save_ensemble_metrics(output_dir, all_metrics)

        return model_paths

    def train_parallel(self, output_dir: str) -> List[str]:
        """
        Train ensemble members in parallel (multi-GPU).

        Uses torch.multiprocessing to train on separate GPUs.

        Args:
            output_dir: Directory to save models.

        Returns:
            List of paths to trained models.
        """
        import copy
        import torch.multiprocessing as mp

        os.makedirs(output_dir, exist_ok=True)

        n_gpus = len(self.config.parallel_gpus)
        n_members = self.config.n_members

        if self.verbose > 0:
            print(f"\n{'='*60}")
            print(f"Parallel Ensemble Training")
            print(f"Members: {n_members}, GPUs: {n_gpus}")
            print(f"{'='*60}")

        # Prepare configs for each member
        configs = []
        for i in range(n_members):
            seed = self.config.seeds[i]
            gpu_id = self.config.parallel_gpus[i % n_gpus]

            config = copy.deepcopy(self.base_config)
            config.seed = seed
            config.save_path = os.path.join(output_dir, f"tqc_seed_{seed}.zip")
            config.checkpoint_dir = os.path.join(output_dir, f"checkpoints_seed_{seed}/")
            config.name = f"ensemble_seed_{seed}"
            config.device = f"cuda:{gpu_id}"

            # === Forced Diversity ===
            if self.config.use_diverse_hyperparams:
                gamma_min, gamma_max = self.config.gamma_range
                lr_min, lr_max = self.config.lr_range
                n = self.config.n_members

                config.gamma = gamma_min + (gamma_max - gamma_min) * i / max(n - 1, 1)
                config.learning_rate = lr_min + (lr_max - lr_min) * i / max(n - 1, 1)

            os.makedirs(config.checkpoint_dir, exist_ok=True)
            configs.append((config, gpu_id))

        # Train in batches of n_gpus
        model_paths = []

        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # Already set

        for batch_start in range(0, n_members, n_gpus):
            batch_end = min(batch_start + n_gpus, n_members)
            batch_configs = configs[batch_start:batch_end]

            if self.verbose > 0:
                print(f"\nBatch {batch_start//n_gpus + 1}: Members {batch_start+1}-{batch_end}")

            # Spawn processes
            processes = []

            for config, gpu_id in batch_configs:
                p = mp.Process(
                    target=self._train_member,
                    args=(config, gpu_id)
                )
                p.start()
                processes.append(p)

            # Wait for all to finish
            for p in processes:
                p.join()

            # Collect paths
            for config, _ in batch_configs:
                model_paths.append(config.save_path)

        # Save ensemble config
        self.config.to_json(os.path.join(output_dir, "ensemble_config.json"))

        return model_paths

    @staticmethod
    def _train_member(config: 'TQCTrainingConfig', gpu_id: int):
        """Worker function for parallel training."""
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        from src.training.train_agent import train

        model, metrics = train(config, use_batch_env=True)
        del model

    def _save_ensemble_metrics(self, output_dir: str, all_metrics: List[Dict]):
        """Save aggregated ensemble training metrics."""
        import pandas as pd

        df = pd.DataFrame(all_metrics)
        df['seed'] = self.config.seeds[:len(all_metrics)]
        df.to_csv(os.path.join(output_dir, "ensemble_training_metrics.csv"), index=False)

        # Summary
        summary = {
            'n_members': len(all_metrics),
            'seeds': self.config.seeds[:len(all_metrics)],
            'avg_action_saturation': float(df['action_saturation'].mean()) if 'action_saturation' in df else None,
            'avg_entropy': float(df['avg_entropy'].mean()) if 'avg_entropy' in df else None,
            'avg_critic_loss': float(df['avg_critic_loss'].mean()) if 'avg_critic_loss' in df else None,
        }

        with open(os.path.join(output_dir, "ensemble_summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)


# =============================================================================
# Utility Functions
# =============================================================================

def load_ensemble(
    ensemble_dir: str,
    config_path: Optional[str] = None,
    device: str = 'cuda',
    verbose: int = 0,
) -> EnsemblePolicy:
    """
    Load a trained ensemble from directory.

    Args:
        ensemble_dir: Directory containing ensemble models.
        config_path: Path to config JSON. If None, looks for ensemble_config.json.
        device: PyTorch device.
        verbose: Verbosity level.

    Returns:
        Loaded EnsemblePolicy.
    """
    # Load config
    if config_path is None:
        config_path = os.path.join(ensemble_dir, "ensemble_config.json")

    if os.path.exists(config_path):
        config = EnsembleConfig.from_json(config_path)
    else:
        config = EnsembleConfig()

    config.device = device

    # Find model files
    model_paths = []
    for seed in config.seeds[:config.n_members]:
        # Try EMA weights first if configured
        if config.use_ema_weights:
            ema_path = os.path.join(ensemble_dir, f"tqc_seed_{seed}_ema.zip")
            if os.path.exists(ema_path):
                model_paths.append(ema_path)
                continue

        # Fallback to regular weights
        path = os.path.join(ensemble_dir, f"tqc_seed_{seed}.zip")
        if os.path.exists(path):
            model_paths.append(path)

    if not model_paths:
        raise FileNotFoundError(f"No ensemble models found in {ensemble_dir}")

    return EnsemblePolicy(model_paths, config=config, verbose=verbose)


def compare_single_vs_ensemble(
    single_model_path: str,
    ensemble_dir: str,
    test_env: VecEnv,
    n_episodes: int = 10,
) -> Dict[str, Any]:
    """
    Compare single model vs ensemble on test environment.

    Args:
        single_model_path: Path to single TQC model.
        ensemble_dir: Directory containing ensemble.
        test_env: Test environment.
        n_episodes: Number of evaluation episodes.

    Returns:
        Comparison metrics dict.
    """
    from stable_baselines3.common.evaluation import evaluate_policy

    # Evaluate single
    single_model = TQC.load(single_model_path)
    single_mean, single_std = evaluate_policy(
        single_model, test_env, n_eval_episodes=n_episodes
    )
    del single_model

    # Evaluate ensemble
    ensemble = load_ensemble(ensemble_dir)

    # Manual evaluation (EnsemblePolicy doesn't have evaluate_policy)
    ensemble_rewards = []
    for _ in range(n_episodes):
        obs = test_env.reset()
        done = False
        episode_reward = 0

        while not done:
            action, _ = ensemble.predict(obs)
            obs, reward, done, info = test_env.step(action)
            episode_reward += reward[0]

        ensemble_rewards.append(episode_reward)

    ensemble.close()

    ensemble_mean = np.mean(ensemble_rewards)
    ensemble_std = np.std(ensemble_rewards)

    return {
        'single_mean': single_mean,
        'single_std': single_std,
        'ensemble_mean': ensemble_mean,
        'ensemble_std': ensemble_std,
        'improvement_mean': ensemble_mean - single_mean,
        'improvement_pct': (ensemble_mean - single_mean) / abs(single_mean) * 100 if single_mean != 0 else 0,
        'variance_reduction': (single_std - ensemble_std) / single_std * 100 if single_std > 0 else 0,
    }

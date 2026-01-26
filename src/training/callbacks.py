# -*- coding: utf-8 -*-
"""
callbacks.py - Consolidated callbacks for SB3 training.

Provides all training-related callbacks:
- UnifiedMetricsCallback: Unified metrics logging with standardized namespaces
- ThreePhaseCurriculumCallback: Curriculum learning for MORL architecture
- OverfittingGuardCallback: Early stopping for overfitting detection
- ModelEMACallback: Exponential Moving Average for policy weights
"""

import os
import time
import numpy as np
import torch
from collections import deque, defaultdict
from typing import TYPE_CHECKING, Optional
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.common.vec_env import VecEnv

if TYPE_CHECKING:
    from multiprocessing.sharedctypes import Synchronized
    from typing import TYPE_CHECKING as _TYPE_CHECKING
else:
    _TYPE_CHECKING = False


# ============================================================================
# Utility Functions
# ============================================================================

def get_next_run_dir(base_dir: str, prefix: str = "run") -> str:
    """
    Find the next available run number.

    Args:
        base_dir: Base directory for logs.
        prefix: Prefix for folders (default: "run").

    Returns:
        Path to the next run folder (e.g., base_dir/run_3).
    """
    os.makedirs(base_dir, exist_ok=True)

    existing = []
    for name in os.listdir(base_dir):
        if name.startswith(f"{prefix}_"):
            try:
                num = int(name.split("_")[1])
                existing.append(num)
            except (IndexError, ValueError):
                pass

    next_num = max(existing, default=0) + 1
    return os.path.join(base_dir, f"{prefix}_{next_num}")


def get_underlying_batch_env(env):
    """
    Unwrap récursif pour trouver BatchCryptoEnv sous les wrappers SB3.

    SB3 peut wrapper les VecEnv dans VecMonitor, VecNormalize, etc.
    Ces wrappers ne forwardent pas correctement set_attr pour les envs GPU.

    Args:
        env: L'environnement (potentiellement wrappé)

    Returns:
        L'instance BatchCryptoEnv sous-jacente, ou l'env original si non trouvé
    """
    depth = 0
    while depth < 20:
        # Cible atteinte (méthode spécifique BatchCryptoEnv)
        # Utiliser set_w_cost_target() qui est unique à BatchCryptoEnv (MORL)
        # ou set_progress() pour curriculum_lambda (ancien système)
        if hasattr(env, 'set_w_cost_target') or hasattr(env, 'set_progress'):
            return env
        # Wrapper VecEnv (ex: VecMonitor, VecNormalize)
        elif hasattr(env, 'venv'):
            env = env.venv
        # Wrapper Gym standard
        elif hasattr(env, 'env'):
            env = env.env
        else:
            break
        depth += 1
    return env


# ============================================================================
# Checkpoint Callbacks
# ============================================================================

class RotatingCheckpointCallback(CheckpointCallback):
    """
    Checkpoint callback that keeps the last N checkpoints to prevent data loss.
    Refactored 2026-01-16 to fix aggressive deletion bug.
    """
    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "rl_model", verbose: int = 0, keep_last: int = 2):
        super().__init__(save_freq, save_path, name_prefix, verbose)
        self.keep_last = keep_last
        self.saved_checkpoints = []  # List to track paths

    def _on_step(self) -> bool:
        # Call parent to save
        result = super()._on_step()

        # Logic matches CheckpointCallback naming convention
        if self.n_calls % self.save_freq == 0:
            step = self.num_timesteps
            current_path = os.path.join(self.save_path, f"{self.name_prefix}_{step}_steps.zip")

            # Track new file if it exists
            if os.path.exists(current_path):
                self.saved_checkpoints.append(current_path)

                # Prune oldest if we exceed limit
                while len(self.saved_checkpoints) > self.keep_last:
                    to_remove = self.saved_checkpoints.pop(0)
                    try:
                        if os.path.exists(to_remove):
                            os.remove(to_remove)
                            if self.verbose > 0:
                                print(f"  [Disk Opt] Pruned old checkpoint: {os.path.basename(to_remove)}")
                    except OSError as e:
                        print(f"  [Warning] Failed to prune {to_remove}: {e}")
        return result


# ============================================================================
# TensorBoard Callbacks
# ============================================================================

class UnifiedMetricsCallback(BaseCallback):
    """
    Unified callback for all environment and training metrics logging.
    
    Replaces TensorBoardStepCallback, StepLoggingCallback, and DetailTensorboardCallback.
    Uses only logger.record() for consistency with SB3's internal logging.
    
    Architecture optimisée:
    - Métriques légères (chaque step): Buffer pour lissage
    - Métriques lourdes (log_freq): Gradients, Q-values, get_global_metrics()
    
    Namespaces standardisés (épurés - signal uniquement):
    - portfolio/: nav, position_pct
    - risk/: max_drawdown
    - rewards/: pnl_component, total_penalties (agrégé)
    - strategy/: churn_ratio
    - debug/: q_values_mean, q_values_std, grad_actor_norm, grad_critic_norm
    """
    
    def __init__(self, log_freq: int = 100, verbose: int = 0):
        """
        Args:
            log_freq: Logging frequency (métriques lourdes uniquement à cette fréquence).
            verbose: Verbosity level (console logging si > 0).
        """
        super().__init__(verbose)
        self.log_freq = log_freq
        
        # Buffer pour métriques légères (lissage)
        self.metrics_buffer = defaultdict(list)
        
        # Episode accumulators
        self.episode_churn_penalties = []
        self.episode_log_returns = []
        
        # FPS tracking (pour console logging)
        self.last_time = None
        self.last_step = 0
        
        # Diagnostic metrics (pour get_training_metrics)
        self.all_actions = deque(maxlen=100_000)
        self.entropy_values = deque(maxlen=100_000)
        self.critic_losses = deque(maxlen=100_000)
        self.actor_losses = deque(maxlen=100_000)
        self.churn_ratios = deque(maxlen=100_000)
        self.actor_grad_norms = deque(maxlen=100_000)
        self.critic_grad_norms = deque(maxlen=100_000)
    
    def _init_callback(self) -> None:
        """Initialize FPS tracking at callback start."""
        self.last_time = time.time()
        self.last_step = 0
    
    def _on_step(self) -> bool:
        """Log metrics at each step (light metrics buffered, heavy metrics at log_freq)."""
        should_log = (self.n_calls % self.log_freq == 0)

        # 0. Collect raw data for diagnostics (every step)
        self._collect_actions_and_entropy()

        # 1. Collect light metrics (every step, buffered) - from infos if available
        self._collect_light_metrics()

        # 2. Log heavy metrics (only at log_freq)
        if should_log:
            # Log buffered metrics first
            self._log_buffered_metrics()
            # Then log global metrics (which may override buffered ones with GPU values)
            self._log_global_metrics()
            # Log expensive operations
            self._log_gradients()
            self._log_tqc_stats()
            # Dump to TensorBoard
            self.logger.dump(self.num_timesteps)

        # 3. Episode end logging (event-driven)
        self._handle_episode_end()

        # 4. Console logging (if verbose and at log_freq)
        if should_log and self.verbose > 0:
            self._log_console()

        return True

    def _collect_actions_and_entropy(self):
        """
        Collect actions and entropy coefficient for diagnostics.

        Called every step to build statistics for:
        - Action saturation (% of actions near ±1)
        - Average entropy coefficient (exploration level)
        """
        # Collect actions
        if 'actions' in self.locals and self.locals['actions'] is not None:
            actions = self.locals['actions']
            if isinstance(actions, torch.Tensor):
                actions = actions.cpu().numpy()
            # Store raw actions (not abs) for proper saturation calculation
            self.all_actions.extend(actions.flatten().tolist())

        # Collect entropy coefficient (alpha) for TQC/SAC
        try:
            if hasattr(self.model, 'ent_coef_tensor'):
                # TQC/SAC with learnable alpha
                ent_coef = self.model.ent_coef_tensor
                if isinstance(ent_coef, torch.Tensor):
                    ent_coef = ent_coef.item()
                self.entropy_values.append(float(ent_coef))
            elif hasattr(self.model, 'ent_coef'):
                # Static entropy coefficient
                ent_coef = self.model.ent_coef
                if isinstance(ent_coef, torch.Tensor):
                    ent_coef = ent_coef.item()
                elif isinstance(ent_coef, str):
                    # "auto" case - try to get from log_ent_coef
                    if hasattr(self.model, 'log_ent_coef'):
                        ent_coef = torch.exp(self.model.log_ent_coef).item()
                    else:
                        return  # Can't get value
                self.entropy_values.append(float(ent_coef))
        except Exception:
            pass  # Silently fail if entropy not accessible
    
    def _collect_light_metrics(self):
        """Collect light metrics from infos dict (fast, no GPU call)."""
        infos = self.locals.get("infos", [{}])
        if infos and len(infos) > 0:
            info = infos[0]
            if isinstance(info, dict):
                # Portfolio metrics
                if "portfolio_value" in info:
                    self.metrics_buffer["portfolio/nav"].append(info["portfolio_value"])
                if "position_pct" in info:
                    self.metrics_buffer["portfolio/position_pct"].append(info["position_pct"])
                
                # Risk metrics
                if "max_drawdown" in info:
                    self.metrics_buffer["risk/max_drawdown"].append(info["max_drawdown"] * 100)  # Convert to %
    
    def _log_buffered_metrics(self):
        """Log buffered metrics (mean for smoothing)."""
        for key, values in self.metrics_buffer.items():
            if values:
                self.logger.record_mean(key, np.mean(values))
        self.metrics_buffer.clear()
    
    def _log_global_metrics(self):
        """Log heavy GPU metrics from get_global_metrics() (only at log_freq)."""
        try:
            # Unwrap to find BatchCryptoEnv under SB3 wrappers
            real_env = get_underlying_batch_env(self.model.env)
            
            if real_env is not None and hasattr(real_env, "get_global_metrics"):
                metrics = real_env.get_global_metrics()
                
                # Portfolio (vitales)
                if "portfolio_value" in metrics:
                    self.logger.record("portfolio/nav", metrics["portfolio_value"])
                if "position_pct" in metrics:
                    self.logger.record("portfolio/position_pct", metrics["position_pct"])
                
                # Risk (vitales)
                if "max_drawdown" in metrics:
                    self.logger.record("risk/max_drawdown", metrics["max_drawdown"] * 100)
                
                # Rewards (agrégées)
                if "reward/pnl_component" in metrics:
                    self.logger.record("rewards/pnl_component", metrics["reward/pnl_component"])
                    self.episode_log_returns.append(metrics["reward/pnl_component"])
                
                # Agréger les pénalités en total_penalties (suppression des composantes individuelles)
                total_penalties = 0.0
                if "reward/churn_cost" in metrics:
                    total_penalties += metrics["reward/churn_cost"]
                    self.episode_churn_penalties.append(metrics["reward/churn_cost"])
                if "reward/smoothness" in metrics:
                    total_penalties += metrics["reward/smoothness"]
                if "reward/downside_risk" in metrics:
                    total_penalties += metrics["reward/downside_risk"]
                
                # Log total_penalties (agrégé, pas les composantes individuelles)
                self.logger.record("rewards/total_penalties", total_penalties)
        except Exception as e:
            if self.verbose > 0:
                print(f"[UnifiedMetricsCallback] Error in _log_global_metrics: {e}")
    
    def _log_gradients(self):
        """Log gradient norms (only at log_freq, expensive operation)."""
        try:
            # Actor gradients
            if hasattr(self.model, 'policy') and hasattr(self.model.policy, 'actor'):
                actor_grad_norm = self._compute_grad_norm(self.model.policy.actor)
                if actor_grad_norm is not None and actor_grad_norm > 0:
                    self.logger.record("debug/grad_actor_norm", actor_grad_norm)
                    self.actor_grad_norms.append(actor_grad_norm)
            
            # Critic gradients
            if hasattr(self.model, 'policy') and hasattr(self.model.policy, 'critic'):
                critic_grad_norm = self._compute_grad_norm(self.model.policy.critic)
                if critic_grad_norm is not None and critic_grad_norm > 0:
                    self.logger.record("debug/grad_critic_norm", critic_grad_norm)
                    self.critic_grad_norms.append(critic_grad_norm)
        except Exception as e:
            if self.verbose > 0:
                print(f"[UnifiedMetricsCallback] Error in _log_gradients: {e}")
    
    def _log_tqc_stats(self):
        """Log TQC Q-values statistics (only at log_freq, for Gamma diagnosis)."""
        try:
            # Access TQC critic
            if not hasattr(self.model, 'policy') or not hasattr(self.model.policy, 'critic'):
                return
            
            critic = self.model.policy.critic
            
            # Get a sample observation and action for Q-value computation
            # SB3 stores observations in 'new_obs' or 'observations', actions in 'actions'
            obs = self.locals.get('new_obs') or self.locals.get('observations')
            actions = self.locals.get('actions')
            
            if obs is None or actions is None:
                return
            
            # Get device from model (handle different SB3 versions)
            device = None
            if hasattr(self.model, 'device'):
                device = self.model.device
            elif hasattr(self.model, 'policy') and hasattr(self.model.policy, 'device'):
                device = self.model.policy.device
            elif hasattr(self.model.policy, 'parameters'):
                device = next(self.model.policy.parameters()).device
            else:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Convert to tensor if needed
            if isinstance(obs, np.ndarray):
                obs_tensor = torch.tensor(obs, device=device, dtype=torch.float32)
            elif isinstance(obs, dict):
                obs_tensor = {
                    k: torch.tensor(v, device=device, dtype=torch.float32) 
                    if isinstance(v, np.ndarray) else v
                    for k, v in obs.items()
                }
            else:
                obs_tensor = obs
            
            if isinstance(actions, np.ndarray):
                actions_tensor = torch.tensor(actions, device=device, dtype=torch.float32)
            else:
                actions_tensor = actions
            
            # Extract features
            if hasattr(self.model.policy, 'extract_features'):
                features = self.model.policy.extract_features(
                    obs_tensor,
                    self.model.policy.features_extractor
                )
            else:
                # Fallback: use observation directly
                features = obs_tensor if isinstance(obs_tensor, torch.Tensor) else obs_tensor['observation']
            
            # Get Q-values from critic
            with torch.no_grad():
                # Handle DropoutCritic (has critics ModuleList)
                if hasattr(critic, 'critics'):
                    # DropoutCritic structure
                    all_q_values = []
                    for qf in critic.critics:
                        q_vals = qf(features, actions_tensor)  # (batch, n_quantiles)
                        all_q_values.append(q_vals)
                    q_values = torch.stack(all_q_values, dim=1)  # (batch, n_critics, n_quantiles)
                elif hasattr(critic, 'quantile_critics'):
                    # Standard TQC structure
                    all_q_values = []
                    for qf in critic.quantile_critics:
                        q_vals = qf(features, actions_tensor)
                        all_q_values.append(q_vals)
                    q_values = torch.stack(all_q_values, dim=1)
                else:
                    # Direct critic call
                    q_values = critic(features, actions_tensor)
                
                # Flatten to get all Q-values
                q_flat = q_values.flatten().cpu().numpy()
                
                # Log mean and std only (no min/max to avoid outliers)
                self.logger.record("debug/q_values_mean", float(np.mean(q_flat)))
                self.logger.record("debug/q_values_std", float(np.std(q_flat)))
        except Exception as e:
            # Gracefully handle if Q-values are not accessible
            if self.verbose > 0:
                pass  # Silent fail for Q-values (may not be accessible in all contexts)
    
    def _handle_episode_end(self):
        """Handle episode end metrics (churn ratio)."""
        if self.locals.get("infos"):
            info = self.locals["infos"][0]
            if info and "episode" in info:
                if self.episode_churn_penalties and self.episode_log_returns:
                    total_churn = sum(self.episode_churn_penalties)
                    total_log_ret = sum(self.episode_log_returns)
                    
                    if abs(total_log_ret) > 1e-8:
                        ratio = abs(total_churn / total_log_ret)
                        self.logger.record("strategy/churn_ratio", ratio)
                        self.churn_ratios.append(ratio)
                
                # Reset episode accumulators
                self.episode_churn_penalties = []
                self.episode_log_returns = []
    
    def _log_console(self):
        """Console logging (if verbose)."""
        # Try to get metrics from get_global_metrics first (most accurate)
        nav = 0
        pos = 0
        max_dd = 0
        
        try:
            real_env = get_underlying_batch_env(self.model.env)
            if real_env is not None and hasattr(real_env, "get_global_metrics"):
                metrics = real_env.get_global_metrics()
                nav = metrics.get("portfolio_value", 0)
                pos = metrics.get("position_pct", 0)
                max_dd = metrics.get("max_drawdown", 0) * 100
        except Exception:
            # Fallback to buffer or infos
            if self.metrics_buffer.get("portfolio/nav"):
                nav = np.mean(self.metrics_buffer["portfolio/nav"])
            if self.metrics_buffer.get("portfolio/position_pct"):
                pos = np.mean(self.metrics_buffer["portfolio/position_pct"])
            if self.metrics_buffer.get("risk/max_drawdown"):
                max_dd = np.mean(self.metrics_buffer["risk/max_drawdown"])
        
        # Calculate FPS manually (fixes FPS=0 bug with BatchCryptoEnv)
        current_time = time.time()
        if self.last_time is not None:
            dt = current_time - self.last_time
            if dt > 0:
                fps = (self.num_timesteps - self.last_step) / dt
            else:
                fps = 0
        else:
            fps = 0
        self.last_time = current_time
        self.last_step = self.num_timesteps
        
        # Get mean reward from episode info
        mean_reward = 0
        if self.locals.get("infos"):
            for info in self.locals["infos"]:
                if info and "episode" in info:
                    mean_reward = info["episode"].get("r", 0)
                    break
        
        print(f"Step {self.num_timesteps:>7} | "
              f"Reward: {mean_reward:>8.2f} | "
              f"NAV: {nav:>10.2f} | "
              f"Pos: {pos:>+5.2f} | "
              f"DD: {max_dd:>5.1f}% | "
              f"FPS: {fps:>7.0f}")
    
    def _compute_grad_norm(self, model) -> Optional[float]:
        """Compute the L2 norm of gradients for a model."""
        total_norm = 0.0
        n_params = 0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
                n_params += 1
        if n_params == 0:
            return None
        return total_norm ** 0.5
    
    def get_training_metrics(self) -> dict:
        """
        Return diagnostic metrics at end of training.

        Metrics:
            action_saturation: Ratio of actions with |a| > 0.95 (policy collapse indicator)
            avg_entropy: Mean entropy coefficient (exploration level)
            action_mean: Mean action value (position bias)
            action_std: Action standard deviation (exploration diversity)
        """
        # Default metrics if no data collected
        default_metrics = {
            "action_saturation": 0.0,
            "avg_entropy": 0.0,
            "action_mean": 0.0,
            "action_std": 0.0,
            "avg_critic_loss": 0.0,
            "avg_actor_loss": 0.0,
            "avg_churn_ratio": 0.0,
            "avg_actor_grad_norm": 0.0,
            "avg_critic_grad_norm": 0.0,
        }

        if not self.all_actions:
            return default_metrics

        # Convert to numpy for calculations
        actions_array = np.array(list(self.all_actions))

        # Action saturation: ratio of actions near ±1 (|a| > 0.95)
        saturation_threshold = 0.95
        saturated_ratio = float(np.mean(np.abs(actions_array) > saturation_threshold))

        # Action statistics
        action_mean = float(np.mean(actions_array))
        action_std = float(np.std(actions_array))

        return {
            "action_saturation": saturated_ratio,
            "avg_entropy": float(np.mean(self.entropy_values)) if self.entropy_values else 0.0,
            "action_mean": action_mean,
            "action_std": action_std,
            "avg_critic_loss": float(np.mean(self.critic_losses)) if self.critic_losses else 0.0,
            "avg_actor_loss": float(np.mean(self.actor_losses)) if self.actor_losses else 0.0,
            "avg_churn_ratio": float(np.mean(self.churn_ratios)) if self.churn_ratios else 0.0,
            "avg_actor_grad_norm": float(np.mean(self.actor_grad_norms)) if self.actor_grad_norms else 0.0,
            "avg_critic_grad_norm": float(np.mean(self.critic_grad_norms)) if self.critic_grad_norms else 0.0,
        }


# OBSOLETE: StepLoggingCallback removed - replaced by UnifiedMetricsCallback


# OBSOLETE: DetailTensorboardCallback removed - replaced by UnifiedMetricsCallback
# Note: get_training_metrics() method preserved in UnifiedMetricsCallback for compatibility


# ============================================================================
# Curriculum Learning Callback
# ============================================================================

# OBSOLETE: CurriculumFeesCallback removed - replaced by ThreePhaseCurriculumCallback


# ═══════════════════════════════════════════════════════════════════════
# OBSOLETE: ThreePhaseCurriculumCallback - Replaced by MORLCurriculumCallback
# ═══════════════════════════════════════════════════════════════════════
# This callback managed curriculum_lambda (0.0 → 0.4) for the OLD reward function.
# The new MORL architecture uses w_cost in observation instead.
# Use MORLCurriculumCallback for progressive w_cost modulation.
# ═══════════════════════════════════════════════════════════════════════
class ThreePhaseCurriculumCallback(BaseCallback):
    """
    Three-Phase Curriculum Learning with IPC Fault Tolerance.
    Refactored 2026-01-16 to fix multiprocessing crashes.
    
    ⚠️ OBSOLETE: This callback manages curriculum_lambda for the OLD reward function.
    It has been replaced by MORLCurriculumCallback which modulates w_cost for MORL.
    
    This callback manages curriculum_lambda for the environment, controlling
    the gradual introduction of cost penalties during training (MORL architecture).
    """
    # Modified by CryptoRL: curriculum extended to 75% of training
    PHASES = [
        {'end_progress': 0.15},
        {'end_progress': 0.75},
        {'end_progress': 1.0},
    ]

    def __init__(self, total_timesteps: int, verbose: int = 0):
        import warnings
        warnings.warn(
            "ThreePhaseCurriculumCallback is OBSOLETE. Use MORLCurriculumCallback instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self._phase = 1

    def _on_step(self) -> bool:
        progress = self.num_timesteps / self.total_timesteps

        # Locate Phase
        phase_cfg = self.PHASES[-1]
        for i, p in enumerate(self.PHASES):
            if progress <= p['end_progress']:
                phase_cfg = p
                self._phase = i + 1
                break

        # ALWAYS update env (for BatchCryptoEnv curriculum_lambda)
        self._update_envs()

        self.logger.record("curriculum/phase", self._phase)
        self.logger.record("curriculum/progress", progress)

        # Log curriculum_lambda from env if available
        real_env = get_underlying_batch_env(self.model.env)
        if hasattr(real_env, 'curriculum_lambda'):
            self.logger.record("curriculum/lambda", real_env.curriculum_lambda)
        
        # Log Dynamic Noise effective scale (Audit 2026-01-19)
        if hasattr(real_env, '_last_noise_scale'):
            self.logger.record("observation_noise/effective_scale", real_env._last_noise_scale)

        return True

    def _update_envs(self):
        """Update curriculum progress on all environments (unwrap for BatchCryptoEnv)."""
        # Unwrap pour atteindre BatchCryptoEnv sous les wrappers SB3
        real_env = get_underlying_batch_env(self.model.env)

        # Calculate progress for curriculum lambda
        progress = self.num_timesteps / self.total_timesteps

        # Sync progress for curriculum_lambda (AAAI 2024 Curriculum Learning)
        if hasattr(real_env, 'set_progress'):
            real_env.set_progress(progress)

    def _on_training_start(self) -> None:
        if self.verbose > 0:
            print(f"\n[3-Phase Curriculum] Configuration (MORL architecture):")
            for i, phase in enumerate(self.PHASES):
                pct = int(phase['end_progress'] * 100)
                print(f"  Phase {i+1} (0-{pct}%): curriculum_lambda={0.4 * min(1.0, max(0, (phase['end_progress'] - 0.15) / 0.60)):.2f}")


# ============================================================================
# MORL Curriculum Callback (Replaces ThreePhaseCurriculumCallback)
# ============================================================================

class MORLCurriculumCallback(BaseCallback):
    """
    MORL Curriculum Learning: Progressive w_cost modulation.
    
    Gradually increases w_cost (cost preference parameter) during training to
    introduce cost constraints progressively. This allows the agent to first
    learn to maximize performance (w_cost ≈ 0) then gradually learn to balance
    performance and costs (w_cost → end_cost).
    
    Architecture:
        - Linear ramp: w_cost goes from start_cost to end_cost over progress_ratio
        - Plateau: w_cost stays at end_cost for the remaining training
    
    The environment samples w_cost around the target value to maintain exploration.
    
    Args:
        start_cost: Initial w_cost value (default: 0.0 = pure performance)
        end_cost: Final w_cost value (default: 0.1 = balanced)
        progress_ratio: Ratio where ramp ends (default: 0.5 = ramp on first half)
        total_timesteps: Total training timesteps for progress calculation
        verbose: Verbosity level
    """
    
    def __init__(
        self,
        start_cost: float = 0.0,
        end_cost: float = 0.0,  # DISABLED: No cost penalty (was 0.1)
        progress_ratio: float = 0.5,
        total_timesteps: int = 30_000_000,
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.start_cost = max(0.0, min(1.0, start_cost))
        self.end_cost = max(0.0, min(1.0, end_cost))
        self.progress_ratio = max(0.0, min(1.0, progress_ratio))
        self.total_timesteps = total_timesteps
        
        if self.start_cost > self.end_cost:
            raise ValueError(f"start_cost ({start_cost}) must be <= end_cost ({end_cost})")
    
    def _on_step(self) -> bool:
        """Update w_cost target based on training progress."""
        # 1. Calculate progress (0.0 to 1.0)
        progress = self.num_timesteps / self.total_timesteps
        
        # 2. Calculate w_cost target (linear ramp then plateau)
        if progress < self.progress_ratio:
            # Linear ramp phase
            alpha = progress / self.progress_ratio
            current_w = self.start_cost + alpha * (self.end_cost - self.start_cost)
        else:
            # Plateau phase
            current_w = self.end_cost
        
        # 3. Apply to environment via dedicated method
        # CRITICAL: Only apply to training environment, never to eval environment
        # self.model.env is the training environment (passed to model.learn())
        # The eval environment is separate and managed by EvalCallback
        real_env = get_underlying_batch_env(self.model.env)
        if real_env is not None and hasattr(real_env, 'set_w_cost_target'):
            real_env.set_w_cost_target(current_w)
        else:
            if self.verbose > 0 and self.n_calls % 1000 == 0:
                print(f"[MORLCurriculumCallback] Warning: BatchCryptoEnv.set_w_cost_target() not found")
        
        # 4. Log to TensorBoard
        self.logger.record("curriculum/w_cost_target", current_w)
        self.logger.record("curriculum/w_cost_progress", progress)
        
        return True
    
    def _on_training_start(self) -> None:
        """
        Initialize curriculum target BEFORE first rollout begins.

        This fixes a timing bug where the first episodes would use the default
        biased distribution (20% w=0, 20% w=1, 60% uniform) instead of the
        curriculum's start_cost value.
        """
        # 1. Set initial curriculum target (also applies to currently running episodes)
        real_env = get_underlying_batch_env(self.model.env)
        if real_env is not None and hasattr(real_env, 'set_w_cost_target'):
            real_env.set_w_cost_target(self.start_cost)
            if self.verbose > 0:
                print(f"[MORL Curriculum] Initial w_cost set to {self.start_cost:.3f}")
        else:
            if self.verbose > 0:
                print(f"[MORL Curriculum] Warning: Could not set initial w_cost_target")

        # 2. Log initial state to TensorBoard (step 0)
        self.logger.record("curriculum/w_cost_target", self.start_cost)
        self.logger.record("curriculum/w_cost_progress", 0.0)

        # 3. Print configuration summary
        if self.verbose > 0:
            print(f"\n[MORL Curriculum] Progressive w_cost modulation:")
            print(f"  Start w_cost: {self.start_cost:.3f} (pure performance)")
            print(f"  End w_cost: {self.end_cost:.3f} (balanced)")
            print(f"  Ramp duration: {self.progress_ratio*100:.0f}% of training")
            print(f"  Plateau duration: {(1-self.progress_ratio)*100:.0f}% of training")
            print(f"  Total timesteps: {self.total_timesteps:,}")


# ============================================================================
# Overfitting Guard Callback
# ============================================================================

class OverfittingGuardCallback(BaseCallback):
    """
    Early stopping if training shows signs of overfitting.

    Triggers abort if NAV exceeds threshold (e.g., 5x initial = +400%).
    Such returns are unrealistic and indicate memorization of training data.
    """

    def __init__(
        self,
        nav_threshold: float = 5.0,  # 5x = +400%
        initial_nav: float = 10_000.0,
        check_freq: int = 25_600,
        verbose: int = 1
    ):
        """
        Args:
            nav_threshold: Multiplier of initial NAV to trigger stop (5.0 = +400%).
            initial_nav: Starting portfolio value.
            check_freq: How often to check (in timesteps).
            verbose: Verbosity level.
        """
        super().__init__(verbose)
        self.nav_threshold = nav_threshold
        self.initial_nav = initial_nav
        self.check_freq = check_freq
        self.max_nav_seen = initial_nav

    def _on_step(self) -> bool:
        if self.num_timesteps % self.check_freq != 0:
            return True

        # Get current NAV from env
        env = self.training_env
        if hasattr(env, 'get_global_metrics'):
            metrics = env.get_global_metrics()
            current_nav = metrics.get("portfolio_value", self.initial_nav)
            self.max_nav_seen = max(self.max_nav_seen, current_nav)

            # Check threshold
            if self.max_nav_seen > self.initial_nav * self.nav_threshold:
                ratio = self.max_nav_seen / self.initial_nav
                print("\n" + "=" * 60)
                print("  EARLY STOPPING: Potential Overfitting Detected!")
                print("=" * 60)
                print(f"  Max NAV seen: {self.max_nav_seen:,.0f}")
                print(f"  Ratio vs initial: {ratio:.1f}x (+{(ratio-1)*100:.0f}%)")
                print(f"  Threshold: {self.nav_threshold}x")
                print("  Such returns are unrealistic - likely memorization.")
                print("=" * 60 + "\n")
                return False  # Stop training

        return True


# ============================================================================
# Evaluation Callback with Observation Noise Management
# ============================================================================

class EvalCallbackWithNoiseControl(EvalCallback):
    """
    Wrapper around EvalCallback that automatically disables observation noise
    in BatchCryptoEnv during evaluation and re-enables it after.

    This ensures that evaluation metrics are not affected by observation noise,
    which should only be active during training for regularization.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the callback with noise control.

        All arguments are passed to EvalCallback.
        """
        super().__init__(*args, **kwargs)
        self._training_env_has_noise = False
        self._eval_env_has_noise = False
        self._last_eval_step = -1

    def _on_step(self) -> bool:
        """
        Override to manage observation noise before/after evaluation.
        """
        # Check if evaluation will occur (EvalCallback logic)
        will_eval = (self.eval_freq > 0 and 
                     self.n_calls % self.eval_freq == 0 and 
                     self.n_calls != self._last_eval_step)
        
        if will_eval:
            # Before evaluation: disable noise in training env if it's BatchCryptoEnv
            train_env = self._get_batch_env(self.training_env)
            if train_env is not None and hasattr(train_env, 'set_training_mode'):
                self._training_env_has_noise = train_env.training
                train_env.set_training_mode(False)
                if self.verbose > 0:
                    print(f"  [Noise Control] Disabled observation noise in training env for evaluation")

            # Check eval environment (should already be False, but ensure it)
            eval_env = self._get_batch_env(self.eval_env)
            if eval_env is not None and hasattr(eval_env, 'set_training_mode'):
                self._eval_env_has_noise = eval_env.training
                eval_env.set_training_mode(False)
                if self.verbose > 0:
                    print(f"  [Noise Control] Disabled observation noise in eval env")

            # 1. Sauvegarder les poids d'entraînement actuels (VITAL)
            training_params = [p.clone() for p in self.model.policy.parameters()]
            
            # 2. Charger les poids EMA pour l'évaluation (si disponible)
            # Access callbacks via self.callback (CallbackList parent) instead of model._callbacks
            ema_callback = None
            if hasattr(self, 'callback'):
                # Import here to avoid forward reference issue
                from src.training.callbacks import ModelEMACallback
                # Try different ways to access callbacks list (SB3 version compatibility)
                callbacks_list = None
                if hasattr(self.callback, 'callbacks'):
                    callbacks_list = self.callback.callbacks
                elif hasattr(self.callback, '_callbacks'):
                    callbacks_list = self.callback._callbacks
                elif isinstance(self.callback, list):
                    callbacks_list = self.callback
                
                if callbacks_list is not None:
                    for callback in callbacks_list:
                        if isinstance(callback, ModelEMACallback):
                            ema_callback = callback
                            ema_callback.load_ema_weights()
                            break

        # Call parent evaluation (this will trigger evaluation if needed)
        try:
            result = super()._on_step()
        finally:
            # After evaluation: RESTAURER les poids d'entraînement (CRITIQUE)
            # Sans cette restauration, l'entraînement continuerait avec les poids EMA
            # (moyenne retardée), ce qui ralentirait massivement l'apprentissage
            if will_eval and ema_callback is not None:
                with torch.no_grad():
                    for param, train_param in zip(self.model.policy.parameters(), training_params):
                        param.data.copy_(train_param.data)
                if self.verbose > 0:
                    print("[EvalCallbackWithNoiseControl] Restored training weights after evaluation")

        # After evaluation: re-enable noise in training env
        if will_eval:
            train_env = self._get_batch_env(self.training_env)
            if train_env is not None and hasattr(train_env, 'set_training_mode'):
                train_env.set_training_mode(self._training_env_has_noise)
                if self.verbose > 0 and self._training_env_has_noise:
                    print(f"  [Noise Control] Re-enabled observation noise in training env")

            # Restore eval env state (should stay False, but restore original)
            eval_env = self._get_batch_env(self.eval_env)
            if eval_env is not None and hasattr(eval_env, 'set_training_mode'):
                eval_env.set_training_mode(self._eval_env_has_noise)
            
            self._last_eval_step = self.n_calls

        return result

    def _get_batch_env(self, env):
        """
        Unwrap environment to find BatchCryptoEnv instance.

        Args:
            env: Environment (potentially wrapped).

        Returns:
            BatchCryptoEnv instance if found, None otherwise.
        """
        return get_underlying_batch_env(env)


# ============================================================================
# Overfitting Guard Callback V2 (SOTA Multi-Signal Detection)
# ============================================================================

class OverfittingGuardCallbackV2(BaseCallback):
    """
    SOTA Overfitting Detection for RL Trading.

    Version 2.3 - Production Release (Post-Audit):
    - Signal 2: Weight Stagnation (replaces Gradient Variance - not accessible in SB3)
    - Signal 3: Train/Eval divergence via ep_info_buffer + EvalCallback (NOT logger)
    - Signal 5: Raw rewards + CV (handles VecNormalize bias)

    Combines 5 independent detection signals:
    1. NAV threshold - Unrealistic returns detection
    2. Weight stagnation (GRADSTOP adapted) - Convergence/collapse detection
    3. Train/Eval divergence - Classic overfitting signal (via buffers)
    4. Action saturation - Policy collapse detection
    5. Reward variance - Memorization detection

    Decision Logic:
    - Stop if any signal reaches 'patience' consecutive violations
    - Stop if 2+ signals are active simultaneously

    References:
    [1] GRADSTOP (arXiv:2508.19028) - adapted for SB3 constraints
    [2] FineFT (arXiv:2512.23773) - action saturation
    [3] Sparse-Reg (arXiv:2506.17155) - reward variance
    [4] Walk-Forward (arXiv:2512.12924) - train/eval divergence

    Audit Fixes:
    - v2.2: Memory leak fix (deque), raw rewards via infos
    - v2.3: "Logger Trap" fix - reads ep_info_buffer + EvalCallback.last_mean_reward
    """

    def __init__(
        self,
        # === Signal 1: NAV Threshold ===
        nav_threshold: float = 5.0,
        initial_nav: float = 10_000.0,

        # === Signal 2: Weight Stagnation (v2.1) ===
        weight_delta_threshold: float = 1e-7,
        cv_threshold: float = 0.01,

        # === Signal 3: Train/Eval Divergence (v2.3: via buffers) ===
        divergence_threshold: float = 0.5,
        eval_callback: Optional[EvalCallback] = None,

        # === Signal 4: Action Saturation ===
        action_saturation_threshold: float = 0.95,
        saturation_ratio_limit: float = 0.8,

        # === Signal 5: Reward Variance ===
        reward_variance_threshold: float = 1e-4,
        reward_window: int = 1000,

        # === Decision Logic ===
        check_freq: int = 10_000,
        patience: int = 3,

        # === Logging ===
        verbose: int = 1
    ):
        """
        Initialize OverfittingGuardCallbackV2.

        Args:
            nav_threshold: NAV multiplier to trigger stop (5.0 = +400%)
            initial_nav: Starting portfolio value
            weight_delta_threshold: Min weight change to consider "learning"
            cv_threshold: Coefficient of Variation threshold for stagnation
            divergence_threshold: Train/Eval reward ratio to trigger (0.5 = 50% better)
            eval_callback: EvalCallback instance for Signal 3 (optional but recommended)
            action_saturation_threshold: |action| above this is "saturated"
            saturation_ratio_limit: Fraction of saturated actions to trigger
            reward_variance_threshold: Min variance to consider "adaptive"
            reward_window: Window size for reward statistics
            check_freq: How often to check signals (in timesteps)
            patience: Consecutive violations before stopping
            verbose: Verbosity level
        """
        super().__init__(verbose)

        # Signal 1
        self.nav_threshold = nav_threshold
        self.initial_nav = initial_nav

        # Signal 2
        self.weight_delta_threshold = weight_delta_threshold
        self.cv_threshold = cv_threshold

        # Signal 3 (v2.3: uses EvalCallback directly, not logger)
        self.divergence_threshold = divergence_threshold
        self.eval_callback = eval_callback

        # Signal 4
        self.action_saturation_threshold = action_saturation_threshold
        self.saturation_ratio_limit = saturation_ratio_limit

        # Signal 5
        self.reward_variance_threshold = reward_variance_threshold
        self.reward_window = reward_window

        # Decision
        self.check_freq = check_freq
        self.patience = patience

        # Internal state
        self.violation_counts = {
            'nav': 0,
            'weight': 0,
            'divergence': 0,
            'saturation': 0,
            'variance': 0
        }
        self.max_nav_seen = initial_nav
        self.last_params = None

        # v2.2 FIX: Use deque with maxlen to prevent memory leak
        # Without this, lists grow unbounded (1M steps = crash)
        self.actions_history: deque = deque(maxlen=reward_window)
        self.rewards_history: deque = deque(maxlen=reward_window)

        # Metrics for logging
        self._last_weight_cv = 0.0
        self._last_weight_delta = 0.0
        self._last_divergence = 0.0
        self._last_saturation_ratio = 0.0
        self._last_reward_variance = 0.0
        self._last_reward_cv = 0.0

    def _on_step(self) -> bool:
        # 1. Collect data (every step, low overhead)
        self._collect_step_data()

        # 2. Evaluate signals (periodically)
        if self.num_timesteps % self.check_freq != 0:
            return True

        violations = []

        # Signal 1: NAV Threshold
        if nav_violation := self._check_nav_threshold():
            violations.append(nav_violation)
            self.violation_counts['nav'] += 1
        else:
            self.violation_counts['nav'] = 0

        # Signal 2: Weight Stagnation (v2.1)
        if weight_violation := self._check_weight_stagnation():
            violations.append(weight_violation)
            self.violation_counts['weight'] += 1
        else:
            self.violation_counts['weight'] = 0

        # Signal 3: Train/Eval Divergence (v2.1: via logs)
        if div_violation := self._check_train_eval_divergence():
            violations.append(div_violation)
            self.violation_counts['divergence'] += 1
        else:
            self.violation_counts['divergence'] = 0

        # Signal 4: Action Saturation
        if sat_violation := self._check_action_saturation():
            violations.append(sat_violation)
            self.violation_counts['saturation'] += 1
        else:
            self.violation_counts['saturation'] = 0

        # Signal 5: Reward Variance
        if var_violation := self._check_reward_variance():
            violations.append(var_violation)
            self.violation_counts['variance'] += 1
        else:
            self.violation_counts['variance'] = 0

        # Log metrics to TensorBoard
        self._log_metrics(violations)

        # Decision
        should_stop = self._decide_stop(violations)

        if should_stop:
            self._print_report(violations)
            return False

        return True

    def _collect_step_data(self):
        """
        Collect data for analysis (low overhead).

        v2.2 FIX: Uses deque with maxlen, no manual truncation needed.
        v2.2 FIX: Attempts to get raw rewards from infos if VecNormalize is used.
        """
        # Actions - take absolute value for saturation check
        if 'actions' in self.locals and self.locals['actions'] is not None:
            actions = self.locals['actions']
            # deque.extend handles maxlen automatically
            self.actions_history.extend(np.abs(actions).flatten())

        # Rewards - try to get RAW rewards (before VecNormalize)
        # Priority: infos['raw_reward'] > infos['original_reward'] > self.locals['rewards']
        raw_rewards = None

        # Attempt 1: Check infos for raw/original reward (custom wrapper or VecNormalize)
        if 'infos' in self.locals and self.locals['infos'] is not None:
            infos = self.locals['infos']
            for info in infos:
                if info is not None:
                    # Some wrappers store raw reward in infos
                    if 'raw_reward' in info:
                        raw_rewards = [i.get('raw_reward', 0) for i in infos if i]
                        break
                    elif 'original_reward' in info:
                        raw_rewards = [i.get('original_reward', 0) for i in infos if i]
                        break

        # Attempt 2: Fallback to self.locals['rewards']
        # Note: Under VecNormalize, these are normalized (variance ~1)
        # Signal 5 may be less effective in this case
        if raw_rewards is None and 'rewards' in self.locals and self.locals['rewards'] is not None:
            raw_rewards = self.locals['rewards'].flatten()

        if raw_rewards is not None:
            self.rewards_history.extend(raw_rewards)

    def _check_nav_threshold(self) -> Optional[str]:
        """Signal 1: Detect unrealistic returns."""
        env = self.training_env
        if hasattr(env, 'get_global_metrics'):
            metrics = env.get_global_metrics()
            current_nav = metrics.get("portfolio_value", self.initial_nav)
            self.max_nav_seen = max(self.max_nav_seen, current_nav)

            if self.max_nav_seen > self.initial_nav * self.nav_threshold:
                ratio = self.max_nav_seen / self.initial_nav
                return f"NAV {ratio:.1f}x (>{self.nav_threshold}x)"
        return None

    def _check_weight_stagnation(self) -> Optional[str]:
        """
        Signal 2: GRADSTOP proxy - Monitor if network weights stop evolving.

        If weights don't change between rollouts, gradients were null/ineffective.

        Note v2.1: Replaces gradient variance check because gradients are not
        accessible in _on_step (collection phase ≠ optimization phase in SB3).
        """
        import torch

        try:
            # Snapshot current weights
            current_params = torch.nn.utils.parameters_to_vector(
                self.model.policy.parameters()
            ).detach().cpu().numpy()

            if self.last_params is not None:
                # Compute delta
                delta = np.abs(current_params - self.last_params)
                mean_delta = np.mean(delta)

                # Coefficient of variation
                if mean_delta > 1e-12:
                    cv = np.std(delta) / mean_delta
                else:
                    cv = 0.0  # Total stagnation

                # Store for logging
                self._last_weight_cv = cv
                self._last_weight_delta = mean_delta

                # Violation if CV low AND mean delta low
                if cv < self.cv_threshold and mean_delta < self.weight_delta_threshold:
                    self.last_params = current_params
                    return f"Weight stagnation (CV={cv:.4f}, Δ={mean_delta:.2e})"

            self.last_params = current_params

        except Exception:
            pass  # Graceful degradation if policy not accessible

        return None

    def _check_train_eval_divergence(self) -> Optional[str]:
        """
        Signal 3: Detect train >> eval gap via SB3 buffers.

        v2.3 FIX ("Logger Trap"):
        - DO NOT use logger.name_to_value (flushed after dump())
        - Train reward: Read from self.model.ep_info_buffer (source)
        - Eval reward: Read from eval_callback.last_mean_reward (source)

        Note v2.2: Be aware of temporal lag!
        - ep_info_buffer is a rolling window (typically 100 episodes)
        - eval/mean_reward is an instantaneous snapshot
        - This is mitigated by 'patience' but signal has inertia
        """
        # v2.3: Disabled if no EvalCallback linked
        if self.eval_callback is None:
            return None

        try:
            # === TRAIN REWARD: Read from ep_info_buffer (SB3 internal buffer) ===
            # This is where SB3 stores episode info for computing ep_rew_mean
            if not hasattr(self.model, 'ep_info_buffer') or len(self.model.ep_info_buffer) == 0:
                return None  # Not enough data yet

            train_mean = np.mean([ep_info['r'] for ep_info in self.model.ep_info_buffer])

            # === EVAL REWARD: Read directly from EvalCallback ===
            eval_mean = self.eval_callback.last_mean_reward

            # Edge case: Eval hasn't run yet (initialized to -inf)
            if eval_mean == -np.inf:
                return None

            # Avoid division by zero
            if abs(eval_mean) < 1e-8:
                return None

            divergence = (train_mean - eval_mean) / (abs(eval_mean) + 1e-9)

            # Store for logging
            self._last_divergence = divergence

            if divergence > self.divergence_threshold:
                return f"Train/Eval divergence {divergence:.1%} (Train={train_mean:.1f}, Eval={eval_mean:.1f})"

        except (AttributeError, KeyError, TypeError):
            pass  # Buffer not available or unexpected structure

        return None

    def _check_action_saturation(self) -> Optional[str]:
        """
        Signal 4: Detect policy collapse via action saturation.

        If agent always outputs |action| ≈ 1, it's a sign of degenerate policy.
        """
        if len(self.actions_history) < self.reward_window:
            return None

        # deque is already bounded, convert to array for numpy ops
        recent = np.array(self.actions_history)
        saturated = np.sum(recent > self.action_saturation_threshold)
        ratio = saturated / len(recent)

        # Store for logging
        self._last_saturation_ratio = ratio

        if ratio > self.saturation_ratio_limit:
            return f"Action saturation {ratio:.0%} (>{self.saturation_ratio_limit:.0%})"

        return None

    def _check_reward_variance(self) -> Optional[str]:
        """
        Signal 5: Detect memorization via reward variance collapse.

        Note v2.1: Uses raw rewards to avoid VecNormalize bias.
        Note v2.2: Attempts to get raw rewards from infos first.
                   If VecNormalize is used and raw_reward not in infos,
                   this signal may be less effective (variance ~1).
        """
        if len(self.rewards_history) < self.reward_window:
            return None

        # deque is already bounded, convert to array for numpy ops
        recent = np.array(self.rewards_history)
        variance = np.var(recent)
        mean = np.mean(np.abs(recent))

        # Store for logging
        self._last_reward_variance = variance

        # Use CV if rewards are in narrow range
        if mean > 1e-8:
            cv = np.std(recent) / mean
            self._last_reward_cv = cv

            # CV < 1% = rewards quasi-constant
            if cv < 0.01 and variance < self.reward_variance_threshold:
                return f"Reward variance collapse (var={variance:.2e}, CV={cv:.4f})"
        elif variance < self.reward_variance_threshold:
            return f"Reward variance collapse ({variance:.2e})"

        return None

    def _decide_stop(self, active_violations: list) -> bool:
        """Multi-criteria decision logic."""
        # Criterion 1: Patience exhausted on any signal
        for count in self.violation_counts.values():
            if count >= self.patience:
                return True

        # Criterion 2: 2+ signals active simultaneously
        if len(active_violations) >= 2:
            return True

        return False

    def _log_metrics(self, violations: list):
        """Log all overfitting metrics to TensorBoard."""
        # Signal 1
        self.logger.record("overfit/max_nav_ratio", self.max_nav_seen / self.initial_nav)

        # Signal 2
        self.logger.record("overfit/weight_delta", self._last_weight_delta)
        self.logger.record("overfit/weight_cv", self._last_weight_cv)

        # Signal 3
        self.logger.record("overfit/train_eval_divergence", self._last_divergence)

        # Signal 4
        self.logger.record("overfit/action_saturation", self._last_saturation_ratio)

        # Signal 5
        self.logger.record("overfit/reward_variance", self._last_reward_variance)
        self.logger.record("overfit/reward_cv", self._last_reward_cv)

        # Violation counts
        for name, count in self.violation_counts.items():
            self.logger.record(f"overfit/violations_{name}", count)

        # Active signals
        self.logger.record("overfit/active_signals", len(violations))

    def _print_report(self, violations: list):
        """Print detailed overfitting report."""
        print("\n" + "=" * 70)
        print("  EARLY STOPPING: Overfitting Signals Detected!")
        print("=" * 70)
        print(f"\n  Step: {self.num_timesteps:,}")
        print(f"\n  Active Violations:")
        for v in violations:
            print(f"    - {v}")
        print(f"\n  Violation History (patience={self.patience}):")
        for name, count in self.violation_counts.items():
            status = "TRIGGERED" if count >= self.patience else f"{count}/{self.patience}"
            print(f"    {name}: {status}")
        print("\n" + "=" * 70 + "\n")

    def _on_training_start(self) -> None:
        if self.verbose > 0:
            print(f"\n[Overfitting Guard V2.3] SOTA Multi-Signal Detection:")
            print(f"  Signal 1 - NAV threshold: {self.nav_threshold}x")
            print(f"  Signal 2 - Weight stagnation: Δ<{self.weight_delta_threshold:.0e}, CV<{self.cv_threshold}")
            eval_status = "ENABLED (via EvalCallback)" if self.eval_callback else "DISABLED (no EvalCallback)"
            print(f"  Signal 3 - Train/Eval divergence: >{self.divergence_threshold:.0%} [{eval_status}]")
            print(f"  Signal 4 - Action saturation: {self.saturation_ratio_limit:.0%} @ |a|>{self.action_saturation_threshold}")
            print(f"  Signal 5 - Reward variance: <{self.reward_variance_threshold:.0e}")
            print(f"  Decision: patience={self.patience}, check_freq={self.check_freq:,}")


# ============================================================================
# Entropy Floor Callback (Prevents Entropy Collapse)
# ============================================================================

class EntropyFloorCallback(BaseCallback):
    """
    Empêche ent_coef de descendre sous un seuil minimum.
    
    Résout le problème d'entropy collapse dans SAC/TQC où l'auto-tuning
    réduit l'entropie trop agressivement, causant une policy déterministe.
    
    Args:
        min_ent_coef: Valeur minimale de ent_coef (default: 0.01)
        check_freq: Fréquence de vérification en steps (default: 1000)
        verbose: Niveau de verbosité (default: 1)
    """
    
    def __init__(self, min_ent_coef: float = 0.01, check_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.min_ent_coef = min_ent_coef
        self.check_freq = check_freq
        self.floor_count = 0
        self._last_ent_coef = None
    
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq != 0:
            return True
        
        # SAC/TQC stocke log(ent_coef) pour la stabilité numérique
        if not hasattr(self.model, 'log_ent_coef'):
            return True
        
        current_log = self.model.log_ent_coef.item()
        current_ent = np.exp(current_log)
        self._last_ent_coef = current_ent
        
        # Appliquer le floor si nécessaire
        if current_ent < self.min_ent_coef:
            new_log = np.log(self.min_ent_coef)
            with torch.no_grad():
                self.model.log_ent_coef.fill_(new_log)
            
            self.floor_count += 1
            if self.verbose > 0:
                print(f"[EntropyFloor] ent_coef {current_ent:.4f} → {self.min_ent_coef} (floor #{self.floor_count})")
        
        # Log pour TensorBoard
        if self.logger is not None:
            self.logger.record("entropy/ent_coef_raw", current_ent)
            self.logger.record("entropy/floor_applied_count", self.floor_count)
            self.logger.record("entropy/min_ent_coef", self.min_ent_coef)
        
        return True


# ============================================================================
# Model EMA Callback (Polyak Averaging)
# ============================================================================

class ModelEMACallback(BaseCallback):
    """
    Maintient une copie 'Shadow' (EMA) du modèle pour éviter l'overfitting.
    
    Utilise stable_baselines3.common.utils.polyak_update pour la mise à jour.
    Les poids EMA sont stockés séparément et peuvent être chargés pour l'évaluation.
    
    Formula: θ_ema = τ * θ + (1-τ) * θ_ema where τ = 1 - decay.
    
    References:
    - Polyak & Juditsky (1992) - Stochastic Approximation
    - Lillicrap et al. (2015) - DDPG (τ=0.001)
    - TQC uses τ=0.005 by default (decay=0.995)
    
    Args:
        decay: EMA decay factor (0.995 = slow, 0.99 = medium, 0.95 = fast)
               Corresponds to τ = 1 - decay (0.005, 0.01, 0.05)
        save_path: Optional path to save EMA model at end of training
        verbose: Verbosity level
    """
    def __init__(self, decay: float = 0.995, save_path: str = None, verbose: int = 0):
        super().__init__(verbose)
        self.decay = decay
        self.tau = 1.0 - decay  # Convert decay to tau for polyak_update
        self.save_path = save_path
        self.ema_params = None  # List of cloned parameter tensors
        self.param_shapes = None  # Track parameter shapes for validation
        
    def _on_training_start(self) -> None:
        """Initialize EMA weights from current policy weights."""
        if not hasattr(self.model, 'policy') or self.model.policy is None:
            if self.verbose > 0:
                print("[ModelEMACallback] Warning: Policy not available yet")
            return
        
        policy = self.model.policy
        if not hasattr(policy, 'actor'):
            if self.verbose > 0:
                print("[ModelEMACallback] Warning: No actor found, skipping EMA")
            return
        
        # Clone and detach parameters (on correct device)
        # Filter to only trainable parameters (same filter used in _on_step)
        policy_params = [param for param in policy.parameters() if param.requires_grad]
        self.ema_params = [
            param.clone().detach().to(param.device)
            for param in policy_params
        ]
        
        # Store parameter shapes for validation
        self.param_shapes = [tuple(p.shape) for p in policy_params]
        
        if self.verbose > 0:
            n_params = sum(p.numel() for p in self.ema_params)
            print(f"[ModelEMACallback] Initialized EMA with decay={self.decay} (τ={self.tau:.4f})")
            print(f"  Parameters: {n_params:,}")
        
    def _on_step(self) -> bool:
        """Update EMA weights using SB3's polyak_update."""
        if self.ema_params is None:
            return True
        
        if not hasattr(self.model, 'policy') or self.model.policy is None:
            return True
        
        # Get current parameters with same filter as initialization
        policy_params = [param for param in self.model.policy.parameters() if param.requires_grad]
        
        # Validate shapes match (safety check for model architecture changes)
        if len(policy_params) != len(self.ema_params):
            if self.verbose > 0:
                print(f"[ModelEMACallback] Warning: Parameter count mismatch "
                      f"({len(policy_params)} vs {len(self.ema_params)}). Reinitializing EMA.")
            self._on_training_start()  # Reinitialize
            return True
        
        # Validate shapes match
        current_shapes = [tuple(p.shape) for p in policy_params]
        if current_shapes != self.param_shapes:
            if self.verbose > 0:
                print(f"[ModelEMACallback] Warning: Parameter shape mismatch. Reinitializing EMA.")
            self._on_training_start()  # Reinitialize
            return True
        
        # Use SB3's native polyak_update function with filtered parameters
        with torch.no_grad():
            polyak_update(
                params=policy_params,
                target_params=self.ema_params,
                tau=self.tau
            )
        
        # Optional logging (every 10k steps to reduce overhead)
        if self.num_timesteps % 10_000 == 0:
            total_diff = sum(
                (p.data - ema_p.data).norm().item()
                for p, ema_p in zip(
                    policy_params,
                    self.ema_params
                )
            )
            self.logger.record("ema/weight_diff_l2", total_diff)
        
        return True

    def load_ema_weights(self) -> None:
        """
        Charge les poids EMA dans le policy (pour évaluation).
        
        Appeler cette méthode avant l'évaluation pour utiliser les poids EMA
        au lieu des poids actuels.
        """
        if self.ema_params is None:
            if self.verbose > 0:
                print("[ModelEMACallback] Warning: EMA not initialized, cannot load")
            return
        
        if not hasattr(self.model, 'policy') or self.model.policy is None:
            return
        
        # Get parameters with same filter as initialization
        policy_params = [param for param in self.model.policy.parameters() if param.requires_grad]
        
        # Validate shapes match
        if len(policy_params) != len(self.ema_params):
            if self.verbose > 0:
                print(f"[ModelEMACallback] Warning: Cannot load EMA - parameter count mismatch")
            return
        
        with torch.no_grad():
            for param, ema_param in zip(policy_params, self.ema_params):
                param.data.copy_(ema_param.data)
        
        if self.verbose > 0:
            print("[ModelEMACallback] Loaded EMA weights into policy")

    def _on_training_end(self) -> None:
        """Sauvegarde le modèle EMA à la fin de l'entraînement."""
        if self.save_path is None or self.ema_params is None:
            return
        
        if not hasattr(self.model, 'policy') or self.model.policy is None:
            return
        
        if self.verbose > 0:
            print(f"[ModelEMACallback] Saving EMA model with decay {self.decay}...")
        
        # Get parameters with same filter as initialization
        policy_params = [param for param in self.model.policy.parameters() if param.requires_grad]
        
        # Validate shapes match
        if len(policy_params) != len(self.ema_params):
            if self.verbose > 0:
                print(f"[ModelEMACallback] Warning: Cannot save EMA - parameter count mismatch")
            return
        
        # 1. Sauvegarde des poids actuels (overfittés ?)
        original_params = [p.clone() for p in policy_params]
        
        # 2. Chargement des poids EMA (using filtered parameters)
        with torch.no_grad():
            for param, ema_param in zip(policy_params, self.ema_params):
                param.data.copy_(ema_param.data)
        
        # 3. Save
        os.makedirs(self.save_path, exist_ok=True)
        ema_path = os.path.join(self.save_path, "best_model_ema.zip")
        self.model.save(ema_path)
        
        if self.verbose > 0:
            print(f"  Saved EMA model to {ema_path}")
        
        # 4. Restauration des poids originaux (si training continue)
        with torch.no_grad():
            for param, orig_param in zip(policy_params, original_params):
                param.data.copy_(orig_param.data)

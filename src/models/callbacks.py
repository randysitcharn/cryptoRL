# -*- coding: utf-8 -*-
"""
callbacks.py - Custom callbacks pour l'entrainement SB3.
"""

import os
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.callbacks import BaseCallback


def get_next_run_dir(base_dir: str, prefix: str = "run") -> str:
    """
    Trouve le prochain numéro de run disponible.

    Args:
        base_dir: Répertoire de base pour les logs.
        prefix: Préfixe pour les dossiers (default: "run").

    Returns:
        Chemin vers le prochain dossier de run (ex: base_dir/run_3).
    """
    os.makedirs(base_dir, exist_ok=True)

    # Trouve tous les dossiers existants avec le préfixe
    existing = []
    for name in os.listdir(base_dir):
        if name.startswith(f"{prefix}_"):
            try:
                num = int(name.split("_")[1])
                existing.append(num)
            except (IndexError, ValueError):
                pass

    # Prochain numéro
    next_num = max(existing, default=0) + 1
    return os.path.join(base_dir, f"{prefix}_{next_num}")


class TensorBoardStepCallback(BaseCallback):
    """
    Callback qui log toutes les métriques pertinentes à chaque step.

    Utilise SummaryWriter directement pour éviter les conflits avec
    le logger interne de SB3.

    Les runs sont automatiquement numérotés (run_1, run_2, etc.).

    Métriques loggées:
    - rollout/reward: Reward instantané
    - env/portfolio_value: Valeur du portfolio (NAV)
    - env/price: Prix actuel de l'actif
    - env/max_drawdown: Max drawdown depuis le début (%)
    - train/actor_loss: Loss de l'acteur
    - train/critic_loss: Loss du critique
    - train/ent_coef: Coefficient d'entropie
    - train/ent_coef_loss: Loss du coefficient d'entropie
    """

    def __init__(self, log_dir: str = None, run_name: str = None, log_freq: int = 1, verbose: int = 0):
        """
        Args:
            log_dir: Répertoire de base pour les logs TensorBoard.
            run_name: Nom du run (optionnel). Si None, numérotation auto.
            log_freq: Fréquence de logging (1 = chaque step).
            verbose: Niveau de verbosité.
        """
        super().__init__(verbose)
        self.log_dir = log_dir
        self.run_name = run_name
        self.log_freq = log_freq
        self.writer = None
        self.run_path = None

    def _on_training_start(self) -> None:
        """Initialise le SummaryWriter au début du training."""
        # Détermine le répertoire de base
        if self.log_dir is None:
            base_dir = self.logger.dir if hasattr(self.logger, 'dir') else "./logs/tensorboard_steps/"
        else:
            base_dir = self.log_dir

        # Détermine le chemin du run
        if self.run_name is not None:
            self.run_path = os.path.join(base_dir, self.run_name)
        else:
            self.run_path = get_next_run_dir(base_dir, prefix="run")

        os.makedirs(self.run_path, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.run_path)

        if self.verbose > 0:
            print(f"[TensorBoardStepCallback] Logging to {self.run_path}")

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq != 0:
            return True

        if self.writer is None:
            return True

        step = self.num_timesteps

        try:
            # Log rewards
            if 'rewards' in self.locals:
                rewards = self.locals['rewards']
                if rewards is not None and len(rewards) > 0:
                    self.writer.add_scalar("rollout/reward", float(rewards[0]), step)

            # Log environment info
            if 'infos' in self.locals:
                infos = self.locals['infos']
                if infos is not None and len(infos) > 0:
                    info = infos[0]
                    if info is not None and isinstance(info, dict):

                        if 'portfolio_value' in info:
                            self.writer.add_scalar("env/portfolio_value", info['portfolio_value'], step)

                        if 'price' in info:
                            self.writer.add_scalar("env/price", info['price'], step)

                        if 'max_drawdown' in info:
                            self.writer.add_scalar("env/max_drawdown", info['max_drawdown'] * 100, step)

            # Log training metrics (losses, entropy)
            if hasattr(self.model, 'logger') and hasattr(self.model.logger, 'name_to_value'):
                logger_values = self.model.logger.name_to_value

                if 'train/actor_loss' in logger_values:
                    self.writer.add_scalar("train/actor_loss", logger_values['train/actor_loss'], step)

                if 'train/critic_loss' in logger_values:
                    self.writer.add_scalar("train/critic_loss", logger_values['train/critic_loss'], step)

                if 'train/ent_coef' in logger_values:
                    self.writer.add_scalar("train/ent_coef", logger_values['train/ent_coef'], step)

                if 'train/ent_coef_loss' in logger_values:
                    self.writer.add_scalar("train/ent_coef_loss", logger_values['train/ent_coef_loss'], step)

        except Exception as e:
            if self.verbose > 0:
                print(f"[TensorBoardStepCallback] Error logging: {e}")

        return True

    def _on_training_end(self) -> None:
        """Ferme le SummaryWriter à la fin du training."""
        if self.writer is not None:
            self.writer.close()
            if self.verbose > 0:
                print("[TensorBoardStepCallback] Writer closed")

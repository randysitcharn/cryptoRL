# -*- coding: utf-8 -*-
"""
test_ensemble_sanity.py - Sanity tests for Ensemble RL integration.

Quick validation tests to run BEFORE launching a full WFO training.
These tests verify imports, config defaults, and API availability.

Usage:
    pytest tests/test_ensemble_sanity.py -v
    python tests/test_ensemble_sanity.py  # Direct execution
"""

import pytest
from src.evaluation.ensemble import EnsembleConfig, EnsemblePolicy, EnsembleTrainer
from src.config import TQCTrainingConfig


def test_ensemble_config_defaults():
    """Vérifie que la config charge les valeurs par défaut SOTA."""
    cfg = EnsembleConfig()
    # Vérifications v1.3
    assert cfg.n_members == 3
    assert cfg.aggregation == 'confidence'
    assert cfg.softmax_temperature == 1.0
    assert cfg.enable_ood_detection is True  # Critique pour la sécurité
    assert cfg.gamma_range == (0.94, 0.96)   # Diversité forcée


def test_ensemble_trainer_init():
    """Vérifie que le Trainer s'initialise sans erreur."""
    base_cfg = TQCTrainingConfig()
    ens_cfg = EnsembleConfig(n_members=2)

    # On force le séquentiel pour le test pour éviter de spawn des process
    ens_cfg.parallel_gpus = [0]

    trainer = EnsembleTrainer(base_config=base_cfg, ensemble_config=ens_cfg, verbose=0)
    assert trainer.config.n_members == 2
    # Vérifier que les configs de training sont prêtes
    assert len(trainer.config.seeds) >= 2


def test_policy_safety_api():
    """Vérifie que l'API de sécurité est présente sur la classe."""
    assert hasattr(EnsemblePolicy, 'predict_with_safety')
    assert hasattr(EnsemblePolicy, 'compute_ood_score')


if __name__ == "__main__":
    # Permet de lancer le test directement avec python
    test_ensemble_config_defaults()
    test_ensemble_trainer_init()
    test_policy_safety_api()
    print("✅ Tous les tests de sanity sont passés.")

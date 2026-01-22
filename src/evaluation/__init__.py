# -*- coding: utf-8 -*-
"""
evaluation - Evaluation and ensemble modules for CryptoRL.

Contains:
- EnsemblePolicy: Multi-model ensemble for robust trading
- EnsembleTrainer: Training orchestrator for ensemble members
- EnsembleConfig: Configuration dataclass
"""

from src.evaluation.ensemble import (
    EnsembleConfig,
    EnsemblePolicy,
    EnsembleTrainer,
    load_ensemble,
    compare_single_vs_ensemble,
)

__all__ = [
    "EnsembleConfig",
    "EnsemblePolicy",
    "EnsembleTrainer",
    "load_ensemble",
    "compare_single_vs_ensemble",
]

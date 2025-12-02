"""
models - Module pour les agents RL.

Contient la factory SAC et les architectures neuronales custom.
"""

from src.models.agent import create_sac_agent

__all__ = ['create_sac_agent']

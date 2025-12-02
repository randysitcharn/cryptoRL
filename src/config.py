"""
config.py - Configuration globale du projet.

Définit les constantes globales (SEED) et détecte automatiquement
le device optimal (CUDA, MPS, ou CPU).
"""

import torch

# Global Configuration
SEED = 42

# Device Detection
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

DEVICE = get_device()

print(f"Global Configuration Loaded. Device selected: {DEVICE}")

"""
reproducibility.py - Utilitaires pour la reproductibilité.

Fournit seed_everything() pour initialiser tous les générateurs
de nombres aléatoires (Python, NumPy, PyTorch) avec une graine fixe.
"""

import random
import os
import numpy as np
import torch

def seed_everything(seed: int):
    """
    Sets the seed for generating random numbers to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    # Ensure deterministic behavior for CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Note: MPS (Metal Performance Shaders) currently doesn't strictly support
    # the same level of determinism flags as CUDA, but manual_seed helps.

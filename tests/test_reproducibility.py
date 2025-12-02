"""
test_reproducibility.py - Tests de reproductibilité.

Vérifie que seed_everything() produit des résultats identiques
lors d'exécutions successives avec la même graine.
"""

import torch
import sys
import os

# Add the project root to the python path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.reproducibility import seed_everything
from src.config import SEED

def test_reproducibility():
    print(f"Testing reproducibility with SEED={SEED}...")

    # First run
    seed_everything(SEED)
    tensor1 = torch.randn(3, 3)

    # Second run
    seed_everything(SEED)
    tensor2 = torch.randn(3, 3)

    # Check if they are identical
    if torch.equal(tensor1, tensor2):
        print("SUCCESS: Tensors are identical.")
        print(f"Tensor 1:\n{tensor1}")
        print(f"Tensor 2:\n{tensor2}")
    else:
        print("FAILURE: Tensors are different.")
        print(f"Tensor 1:\n{tensor1}")
        print(f"Tensor 2:\n{tensor2}")
        sys.exit(1)

if __name__ == "__main__":
    test_reproducibility()

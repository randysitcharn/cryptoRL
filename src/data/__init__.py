"""
data - Module pour le chargement et préparation des données.

Contient:
- CryptoDataset: Dataset PyTorch avec fenêtres glissantes
- create_dataloader: Factory pour DataLoader
"""

from src.data.dataset import CryptoDataset, create_dataloader

__all__ = ['CryptoDataset', 'create_dataloader']

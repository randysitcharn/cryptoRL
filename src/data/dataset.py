"""
dataset.py - Dataset PyTorch pour séries temporelles financières.

Fournit:
- CryptoDataset: Fenêtres glissantes sur données Parquet
- Gestion automatique des colonnes (exclusion des prix bruts)
- Support train/val/test split
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Tuple


class CryptoDataset(Dataset):
    """
    Dataset avec fenêtres glissantes pour séries temporelles financières.

    Charge les données depuis un fichier Parquet et crée des fenêtres
    glissantes de taille fixe pour l'entraînement.

    Attributes:
        data: Array numpy des features (n_samples, n_features)
        seq_len: Longueur des fenêtres
        n_features: Nombre de features
        feature_cols: Liste des noms de colonnes utilisées
    """

    # Colonnes à exclure (prix bruts non-scalés)
    EXCLUDE_COLS = [
        'BTC_Close', 'ETH_Close', 'SPX_Close', 'DXY_Close', 'NASDAQ_Close',
        'BTC_Open', 'BTC_High', 'BTC_Low', 'BTC_Volume',
        'ETH_Open', 'ETH_High', 'ETH_Low',
        'SPX_Open', 'SPX_High', 'SPX_Low',
        'DXY_Open', 'DXY_High', 'DXY_Low',
        'NASDAQ_Open', 'NASDAQ_High', 'NASDAQ_Low',
    ]

    def __init__(
        self,
        parquet_path: str = "data/processed_data.parquet",
        seq_len: int = 64,
        feature_cols: Optional[List[str]] = None,
        start_idx: Optional[int] = None,
        end_idx: Optional[int] = None
    ):
        """
        Initialise le dataset.

        Args:
            parquet_path: Chemin vers le fichier Parquet.
            seq_len: Longueur des fenêtres glissantes.
            feature_cols: Liste des colonnes à utiliser (si None, auto-détection).
            start_idx: Index de début (pour train/val/test split).
            end_idx: Index de fin (pour train/val/test split).
        """
        self.seq_len = seq_len
        self.parquet_path = parquet_path

        # Charger les données
        df = pd.read_parquet(parquet_path)

        # Appliquer le slicing si spécifié
        if start_idx is not None or end_idx is not None:
            df = df.iloc[start_idx:end_idx]

        # Déterminer les colonnes à utiliser
        if feature_cols is None:
            # Auto-détection: exclure les prix bruts et garder les features scalées
            feature_cols = [
                col for col in df.columns
                if col not in self.EXCLUDE_COLS
                and df[col].dtype in ['float64', 'float32']
            ]

        self.feature_cols = feature_cols
        self.n_features = len(feature_cols)

        # Extraire les données comme array numpy
        self.data = df[feature_cols].values.astype(np.float32)

        # Vérifier qu'il n'y a pas de NaN
        if np.isnan(self.data).any():
            nan_count = np.isnan(self.data).sum()
            print(f"[WARNING] Dataset contains {nan_count} NaN values. Replacing with 0.")
            self.data = np.nan_to_num(self.data, nan=0.0)

        # Stocker les métadonnées
        self.index = df.index[seq_len - 1:]  # Index des derniers éléments de chaque fenêtre

        print(f"[CryptoDataset] Loaded {len(self)} windows")
        print(f"  Shape: ({len(self.data)}, {self.n_features})")
        print(f"  Window size: {seq_len}")
        print(f"  Features: {self.n_features}")

    def __len__(self) -> int:
        """Nombre de fenêtres disponibles."""
        return len(self.data) - self.seq_len + 1

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Retourne une fenêtre glissante.

        Args:
            idx: Index de la fenêtre.

        Returns:
            Tensor de shape (seq_len, n_features).
        """
        window = self.data[idx:idx + self.seq_len]
        return torch.from_numpy(window)

    def get_feature_names(self) -> List[str]:
        """Retourne la liste des noms de features."""
        return self.feature_cols

    @staticmethod
    def create_splits(
        parquet_path: str = "data/processed_data.parquet",
        seq_len: int = 64,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        feature_cols: Optional[List[str]] = None
    ) -> Tuple['CryptoDataset', 'CryptoDataset', 'CryptoDataset']:
        """
        Crée les splits train/val/test.

        Args:
            parquet_path: Chemin vers le fichier Parquet.
            seq_len: Longueur des fenêtres.
            train_ratio: Ratio pour l'entraînement.
            val_ratio: Ratio pour la validation.
            feature_cols: Colonnes à utiliser.

        Returns:
            Tuple (train_dataset, val_dataset, test_dataset).
        """
        # Charger pour obtenir la taille
        df = pd.read_parquet(parquet_path)
        n_samples = len(df)

        # Calculer les indices de split
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))

        print(f"[Dataset Splits]")
        print(f"  Total samples: {n_samples}")
        print(f"  Train: 0 to {train_end} ({train_end} samples)")
        print(f"  Val: {train_end} to {val_end} ({val_end - train_end} samples)")
        print(f"  Test: {val_end} to {n_samples} ({n_samples - val_end} samples)")

        # Créer les datasets
        train_ds = CryptoDataset(
            parquet_path=parquet_path,
            seq_len=seq_len,
            feature_cols=feature_cols,
            start_idx=0,
            end_idx=train_end
        )

        val_ds = CryptoDataset(
            parquet_path=parquet_path,
            seq_len=seq_len,
            feature_cols=train_ds.feature_cols,  # Même colonnes que train
            start_idx=train_end,
            end_idx=val_end
        )

        test_ds = CryptoDataset(
            parquet_path=parquet_path,
            seq_len=seq_len,
            feature_cols=train_ds.feature_cols,  # Même colonnes que train
            start_idx=val_end,
            end_idx=None
        )

        return train_ds, val_ds, test_ds


def create_dataloader(
    dataset: CryptoDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True
) -> DataLoader:
    """
    Crée un DataLoader à partir d'un CryptoDataset.

    Args:
        dataset: Instance de CryptoDataset.
        batch_size: Taille des batches.
        shuffle: Mélanger les données.
        num_workers: Nombre de workers pour le chargement.
        pin_memory: Utiliser pin_memory pour GPU.

    Returns:
        DataLoader configuré.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Éviter les batches incomplets
    )


if __name__ == "__main__":
    # Test du dataset
    print("=" * 60)
    print("CRYPTO DATASET TEST")
    print("=" * 60)

    # Créer le dataset
    dataset = CryptoDataset(seq_len=64)

    print(f"\nDataset length: {len(dataset)}")
    print(f"Number of features: {dataset.n_features}")
    print(f"Feature names: {dataset.feature_cols[:5]}...")

    # Test d'un batch
    sample = dataset[0]
    print(f"\nSample shape: {sample.shape}")
    print(f"Sample dtype: {sample.dtype}")

    # Test du DataLoader
    loader = create_dataloader(dataset, batch_size=32)
    batch = next(iter(loader))
    print(f"\nBatch shape: {batch.shape}")

    # Test des splits
    print("\n" + "=" * 60)
    print("TESTING TRAIN/VAL/TEST SPLITS")
    print("=" * 60)

    train_ds, val_ds, test_ds = CryptoDataset.create_splits(seq_len=64)
    print(f"\nTrain: {len(train_ds)} windows")
    print(f"Val: {len(val_ds)} windows")
    print(f"Test: {len(test_ds)} windows")

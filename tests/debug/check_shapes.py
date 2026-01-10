"""
check_shapes.py - Script de debug pour vérifier les shapes du modèle MAE.

Ce script:
1. Charge un mini-batch de données réelles
2. Passe dans le modèle CryptoMAE
3. Vérifie que Output shape == Input shape
"""

import sys
import os

# Ajouter le chemin racine pour les imports
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import torch
from torch.utils.data import DataLoader

from src.data.dataset import CryptoDataset
from src.models.foundation import CryptoMAE


def main():
    print("=" * 70)
    print("MAE SHAPE VERIFICATION")
    print("=" * 70)

    # Configuration
    BATCH_SIZE = 32
    SEQ_LEN = 64
    MASK_RATIO = 0.15

    # =========================================================================
    # 1. Charger les données
    # =========================================================================
    print("\n[1] Loading dataset...")

    dataset = CryptoDataset(
        parquet_path=os.path.join(ROOT_DIR, "data/processed_data.parquet"),
        seq_len=SEQ_LEN
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )

    print(f"  Dataset size: {len(dataset)} windows")
    print(f"  Features: {dataset.n_features}")
    print(f"  Batch size: {BATCH_SIZE}")

    # =========================================================================
    # 2. Créer le modèle
    # =========================================================================
    print("\n[2] Creating model...")

    model = CryptoMAE(
        input_dim=dataset.n_features,
        d_model=128,
        n_heads=4,
        n_layers=2,
        dropout=0.1
    )

    # Compter les paramètres
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  Model: CryptoMAE")
    print(f"  d_model: 128")
    print(f"  n_heads: 4")
    print(f"  n_layers: 2")
    print(f"  Total parameters: {n_params:,}")
    print(f"  Trainable parameters: {n_trainable:,}")

    # =========================================================================
    # 3. Forward pass
    # =========================================================================
    print("\n[3] Running forward pass...")

    # Récupérer un batch
    batch = next(iter(loader))

    # Mode évaluation (pas de dropout)
    model.eval()

    with torch.no_grad():
        pred, target, mask = model(batch, mask_ratio=MASK_RATIO)

    # =========================================================================
    # 4. Afficher les shapes
    # =========================================================================
    print("\n[4] Shape verification:")
    print(f"  Input Shape:  {tuple(batch.shape)}")
    print(f"  Output Shape: {tuple(pred.shape)}")
    print(f"  Target Shape: {tuple(target.shape)}")
    print(f"  Mask Shape:   {tuple(mask.shape)}")

    # Informations supplémentaires
    n_masked = mask.sum().item()
    n_total = mask.numel()
    actual_ratio = n_masked / n_total

    print(f"\n  Masked tokens: {n_masked} / {n_total} ({actual_ratio:.1%})")
    print(f"  Expected ratio: {MASK_RATIO:.1%}")

    # =========================================================================
    # 5. Vérification
    # =========================================================================
    print("\n[5] Verification:")

    # Check 1: Output shape == Input shape
    shape_match = pred.shape == batch.shape
    print(f"  Output shape == Input shape: {'PASS' if shape_match else 'FAIL'}")

    # Check 2: Target shape coherent
    expected_target_shape = (n_masked, dataset.n_features)
    target_match = target.shape == torch.Size(expected_target_shape)
    print(f"  Target shape correct: {'PASS' if target_match else 'FAIL'}")

    # Check 3: Mask shape coherent
    expected_mask_shape = (BATCH_SIZE, SEQ_LEN)
    mask_match = mask.shape == torch.Size(expected_mask_shape)
    print(f"  Mask shape correct: {'PASS' if mask_match else 'FAIL'}")

    # Check 4: Pas de NaN
    no_nan_pred = not torch.isnan(pred).any()
    no_nan_target = not torch.isnan(target).any()
    print(f"  No NaN in predictions: {'PASS' if no_nan_pred else 'FAIL'}")
    print(f"  No NaN in targets: {'PASS' if no_nan_target else 'FAIL'}")

    # =========================================================================
    # 6. Test de la loss
    # =========================================================================
    print("\n[6] Loss computation test:")

    loss = model.get_reconstruction_loss(pred, target, mask)
    print(f"  Reconstruction loss: {loss.item():.6f}")

    # =========================================================================
    # Résultat final
    # =========================================================================
    all_passed = all([
        shape_match,
        target_match,
        mask_match,
        no_nan_pred,
        no_nan_target
    ])

    print("\n" + "=" * 70)
    if all_passed:
        print("ALL CHECKS PASSED!")
    else:
        print("SOME CHECKS FAILED!")
    print("=" * 70)

    # Assert pour CI/CD
    assert shape_match, f"Output shape {pred.shape} != Input shape {batch.shape}"
    assert target_match, f"Target shape {target.shape} != Expected {expected_target_shape}"
    assert mask_match, f"Mask shape {mask.shape} != Expected {expected_mask_shape}"

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

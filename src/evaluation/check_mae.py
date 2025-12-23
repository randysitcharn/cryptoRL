# -*- coding: utf-8 -*-
"""
check_mae.py - MAE (Masked Autoencoder) Quality Evaluation.

Evaluates the reconstruction quality of the pretrained CryptoMAE model:
- Computes MSE and R² Score on validation data
- Generates visualization of original vs reconstructed sequences

Usage:
    python -m src.evaluation.check_mae
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, Subset

from src.models.foundation import CryptoMAE
from src.data.dataset import CryptoDataset


# ============================================================================
# Configuration
# ============================================================================

class MAECheckConfig:
    """Configuration for MAE evaluation."""

    # Paths
    data_path: str = "data/processed_data.parquet"
    model_path: str = "weights/best_foundation_full.pth"
    output_path: str = "mae_reconstruction.png"

    # Model architecture (must match pretrained)
    input_dim: int = 42
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2

    # Evaluation
    seq_len: int = 64
    mask_ratio: float = 0.15  # Same as training
    batch_size: int = 64
    train_ratio: float = 0.8

    # Visualization
    n_examples: int = 4
    feature_to_plot: str = "HMM_Momentum"  # Most volatile feature


# ============================================================================
# Model Loading
# ============================================================================

def load_model(config: MAECheckConfig) -> CryptoMAE:
    """
    Load pretrained CryptoMAE model.

    Args:
        config: Configuration object.

    Returns:
        Loaded CryptoMAE model in eval mode.
    """
    print(f"[1/4] Loading model from {config.model_path}...")

    model = CryptoMAE(
        input_dim=config.input_dim,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
    )

    checkpoint = torch.load(config.model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Print checkpoint info
    epoch = checkpoint.get('epoch', 'N/A')
    val_loss = checkpoint.get('val_loss', 'N/A')
    print(f"      Loaded checkpoint from epoch {epoch}")
    if isinstance(val_loss, float):
        print(f"      Validation loss at checkpoint: {val_loss:.6f}")

    return model


# ============================================================================
# Data Loading
# ============================================================================

def load_validation_data(config: MAECheckConfig) -> tuple:
    """
    Load validation dataset.

    Args:
        config: Configuration object.

    Returns:
        Tuple of (val_loader, val_dataset, feature_names).
    """
    print(f"[2/4] Loading validation data...")

    # Load full dataset to get feature names
    full_dataset = CryptoDataset(
        parquet_path=config.data_path,
        seq_len=config.seq_len,
    )

    feature_names = full_dataset.get_feature_names()
    n_total = len(full_dataset)

    # Create validation subset (last 20%)
    val_start = int(n_total * config.train_ratio)
    val_indices = list(range(val_start, n_total))
    val_dataset = Subset(full_dataset, val_indices)

    print(f"      Total windows: {n_total}")
    print(f"      Validation windows: {len(val_dataset)}")
    print(f"      Features: {len(feature_names)}")

    # Create DataLoader
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )

    return val_loader, full_dataset, feature_names


# ============================================================================
# Metrics Computation
# ============================================================================

def compute_metrics(
    model: CryptoMAE,
    val_loader: DataLoader,
    mask_ratio: float = 0.15
) -> dict:
    """
    Compute MSE and R² Score on validation data.

    Args:
        model: Pretrained CryptoMAE model.
        val_loader: Validation DataLoader.
        mask_ratio: Masking ratio to use.

    Returns:
        Dictionary with metrics.
    """
    print(f"[3/4] Computing metrics (mask_ratio={mask_ratio})...")

    all_targets = []
    all_preds = []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            # Forward pass with masking
            pred, target, mask = model(batch, mask_ratio=mask_ratio)

            # Get predictions at masked positions
            pred_masked = pred[mask]

            # Compute batch loss
            batch_loss = torch.nn.functional.mse_loss(pred_masked, target)
            total_loss += batch_loss.item()
            n_batches += 1

            # Collect for R² computation
            all_targets.append(target.numpy())
            all_preds.append(pred_masked.numpy())

    # Concatenate all results
    all_targets = np.concatenate(all_targets, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)

    # Compute metrics
    mse = total_loss / n_batches

    # R² Score (flatten for sklearn)
    r2 = r2_score(all_targets.flatten(), all_preds.flatten())

    # Per-feature R²
    n_features = all_targets.shape[1]
    feature_r2 = []
    for i in range(n_features):
        feat_r2 = r2_score(all_targets[:, i], all_preds[:, i])
        feature_r2.append(feat_r2)

    print(f"      MSE (masked positions): {mse:.6f}")
    print(f"      R² Score (overall):     {r2:.4f}")
    print(f"      R² Score range:         [{min(feature_r2):.4f}, {max(feature_r2):.4f}]")

    return {
        'mse': mse,
        'r2': r2,
        'feature_r2': feature_r2,
        'n_samples': len(all_targets),
    }


# ============================================================================
# Visualization
# ============================================================================

def visualize_reconstruction(
    model: CryptoMAE,
    dataset: CryptoDataset,
    feature_names: list,
    config: MAECheckConfig,
) -> None:
    """
    Generate visualization of original vs reconstructed sequences.

    Args:
        model: Pretrained CryptoMAE model.
        dataset: Full dataset.
        feature_names: List of feature names.
        config: Configuration object.
    """
    print(f"[4/4] Generating visualization...")

    # Find feature index
    try:
        feature_idx = feature_names.index(config.feature_to_plot)
    except ValueError:
        print(f"      Warning: '{config.feature_to_plot}' not found, using first feature")
        feature_idx = 0
        config.feature_to_plot = feature_names[0]

    print(f"      Feature: {config.feature_to_plot} (index {feature_idx})")

    # Get validation samples (evenly spaced)
    n_total = len(dataset)
    val_start = int(n_total * config.train_ratio)
    val_size = n_total - val_start

    # Select n_examples evenly spaced samples from validation set
    sample_indices = [
        val_start + int(i * val_size / config.n_examples)
        for i in range(config.n_examples)
    ]

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    with torch.no_grad():
        for i, idx in enumerate(sample_indices):
            ax = axes[i]

            # Get single sample
            x = dataset[idx].unsqueeze(0)  # (1, 64, 35)

            # Forward pass
            pred, target, mask = model(x, mask_ratio=config.mask_ratio)

            # Extract data for plotting
            x_np = x[0, :, feature_idx].numpy()  # Original
            pred_np = pred[0, :, feature_idx].numpy()  # Reconstructed
            mask_np = mask[0].numpy()  # Mask (True = masked)

            # Time axis
            t = np.arange(config.seq_len)

            # Plot original (blue, solid with markers)
            ax.plot(t, x_np, 'b-o', linewidth=1.0, markersize=3, label='Original', alpha=0.8)

            # Plot reconstruction (red, dashed with markers)
            ax.plot(t, pred_np, 'r--x', linewidth=1.0, markersize=3, label='Reconstructed', alpha=0.8)

            # Highlight masked regions (gray)
            mask_starts = []
            mask_ends = []
            in_mask = False

            for j in range(len(mask_np)):
                if mask_np[j] and not in_mask:
                    mask_starts.append(j)
                    in_mask = True
                elif not mask_np[j] and in_mask:
                    mask_ends.append(j)
                    in_mask = False

            if in_mask:
                mask_ends.append(len(mask_np))

            for start, end in zip(mask_starts, mask_ends):
                ax.axvspan(start - 0.5, end - 0.5, alpha=0.2, color='gray')

            # Styling
            ax.set_xlabel('Time Step')
            ax.set_ylabel(config.feature_to_plot)
            ax.set_title(f'Sample {i+1} (idx={idx}) - {int(mask_np.sum())} masked tokens')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)

    # Global title
    fig.suptitle(
        f'MAE Reconstruction Quality\n'
        f'Feature: {config.feature_to_plot} | Mask Ratio: {config.mask_ratio:.0%}',
        fontsize=14,
        fontweight='bold'
    )

    plt.tight_layout()

    # Save figure
    plt.savefig(config.output_path, dpi=150, bbox_inches='tight')
    print(f"      Saved: {config.output_path}")

    plt.close()


# ============================================================================
# Main
# ============================================================================

def run_mae_check(config: MAECheckConfig = None) -> dict:
    """
    Run complete MAE quality evaluation.

    Args:
        config: Configuration object.

    Returns:
        Dictionary with all metrics.
    """
    if config is None:
        config = MAECheckConfig()

    print("=" * 70)
    print("MAE QUALITY EVALUATION")
    print("=" * 70)

    # 1. Load model
    model = load_model(config)

    # 2. Load validation data
    val_loader, dataset, feature_names = load_validation_data(config)

    # 3. Compute metrics
    metrics = compute_metrics(model, val_loader, config.mask_ratio)

    # 4. Generate visualization
    visualize_reconstruction(model, dataset, feature_names, config)

    # Summary
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    print(f"\n  Metrics Summary:")
    print(f"    MSE:  {metrics['mse']:.6f}")
    print(f"    R²:   {metrics['r2']:.4f}")
    print(f"    N:    {metrics['n_samples']:,} masked tokens")

    # Interpretation
    print(f"\n  Interpretation:")
    if metrics['r2'] > 0.9:
        print("    [EXCELLENT] R² > 0.9 - MAE captures patterns very well")
    elif metrics['r2'] > 0.7:
        print("    [GOOD] R² > 0.7 - MAE captures most patterns")
    elif metrics['r2'] > 0.5:
        print("    [MODERATE] R² > 0.5 - MAE captures some patterns")
    else:
        print("    [POOR] R² < 0.5 - MAE struggles to reconstruct")

    print(f"\n  Output: {config.output_path}")
    print("=" * 70)

    return metrics


if __name__ == "__main__":
    metrics = run_mae_check()

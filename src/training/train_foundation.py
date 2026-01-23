"""
train_foundation.py - Script d'entraînement pour le Masked Auto-Encoder (MAE).

Ce script:
1. Charge les données avec split chronologique (80/20)
2. Entraîne le CryptoMAE avec reconstruction masquée
3. Implémente Early Stopping (patience=5)
4. Sauvegarde le modèle complet et l'encodeur seul

Usage:
    python src/training/train_foundation.py
"""

import os
import sys
import time
from datetime import datetime

# Ajouter le chemin racine pour les imports
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_DIR)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Optional

from src.data.dataset import CryptoDataset
from src.models.foundation import CryptoMAE


# =============================================================================
# Configuration (imported from centralized config module)
# =============================================================================

from src.config import FoundationTrainingConfig as TrainingConfig, get_device


# =============================================================================
# Device Detection (verbose version for training output)
# =============================================================================

def get_device_verbose() -> torch.device:
    """Détecte automatiquement le meilleur device disponible avec log."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Device] Using CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[Device] Using Apple MPS")
    else:
        device = torch.device("cpu")
        print("[Device] Using CPU")
    return device


# =============================================================================
# Data Preparation
# =============================================================================

def prepare_data(config: TrainingConfig, supervised: bool = True):
    """
    Prépare les données avec split chronologique.

    Args:
        config: Configuration d'entraînement.
        supervised: Si True, retourne (features, target) pour supervised learning.

    Returns:
        train_loader, val_loader, n_features
    """
    print("\n[Data] Loading dataset...")

    # Charger le dataset complet
    dataset = CryptoDataset(
        parquet_path=os.path.join(ROOT_DIR, config.data_path),
        seq_len=config.seq_len,
        return_targets=supervised
    )

    n_samples = len(dataset)
    n_features = dataset.n_features

    # Split chronologique (PAS de mélange futur/passé)
    train_size = int(config.train_ratio * n_samples)
    val_size = n_samples - train_size

    print(f"[Data] Total samples: {n_samples}")
    print(f"[Data] Train: {train_size} ({config.train_ratio:.0%})")
    print(f"[Data] Val: {val_size} ({1 - config.train_ratio:.0%})")

    # Créer les subsets
    train_dataset = Subset(dataset, range(0, train_size))
    val_dataset = Subset(dataset, range(train_size, n_samples))

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,  # OK de mélanger DANS le train set
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False
    )

    print(f"[Data] Train batches: {len(train_loader)}")
    print(f"[Data] Val batches: {len(val_loader)}")

    return train_loader, val_loader, n_features


# =============================================================================
# Training Functions
# =============================================================================

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    config: TrainingConfig,
    supervised: bool = True,
    aux_loss_weight: float = 2.0,
    limit: Optional[int] = None
) -> dict:
    """
    Entraîne le modèle pour une époque.

    Args:
        model: Le modèle CryptoMAE.
        loader: DataLoader d'entraînement.
        optimizer: Optimiseur.
        scaler: GradScaler pour mixed precision.
        device: Device (cuda/cpu).
        config: Configuration d'entraînement.
        supervised: Si True, utilise les targets de direction.
        aux_loss_weight: Poids de la loss auxiliaire (défaut: 2.0 pour forcer l'apprentissage directionnel).

    Returns:
        Dict avec 'total', 'recon', 'aux' losses moyennes.
    """
    model.train()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_aux_loss = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc="Training", leave=False)

    for batch_idx, batch in enumerate(pbar):
        # Limiter le nombre de batches si spécifié (pour tests)
        if limit is not None and batch_idx >= limit:
            break
        # Dépack le batch selon le mode
        if supervised:
            x, target_direction = batch
            x = x.to(device)
            target_direction = target_direction.to(device)
            # S'assurer que target_direction a shape (batch, 1)
            if target_direction.dim() == 1:
                target_direction = target_direction.unsqueeze(1)
        else:
            x = batch.to(device)
            target_direction = None

        optimizer.zero_grad()

        # Mixed precision forward
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
            pred, target_recon, mask, pred_logits = model(x, mask_ratio=config.mask_ratio)

            # 1. Reconstruction Loss (tokens masqués uniquement)
            pred_masked = pred[mask]
            recon_loss = nn.functional.mse_loss(pred_masked, target_recon)

            # 2. Auxiliary Prediction Loss (direction du marché)
            if supervised and target_direction is not None:
                # Utiliser pos_weight pour pénaliser les Faux Positifs (modèle trop optimiste)
                # pos_weight < 1.0 pénalise la classe positive (Hausse)
                # On utilise 0.8 pour sous-pondérer légèrement la classe majoritaire
                pos_weight = torch.tensor([0.8], device=device)
                aux_loss = nn.functional.binary_cross_entropy_with_logits(
                    pred_logits.view(-1), target_direction.view(-1).float(),
                    pos_weight=pos_weight
                )
                # Augmentation drastique du poids de la supervision pour forcer l'apprentissage directionnel
                # La recon_loss est déjà faible (~0.03), on veut que la prédiction domine le gradient
                loss = recon_loss + aux_loss_weight * aux_loss
            else:
                aux_loss = torch.tensor(0.0, device=device)
                loss = recon_loss

        # Backward avec scaling
        if device.type == 'cuda':
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_aux_loss += aux_loss.item()
        n_batches += 1

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'recon': f'{recon_loss.item():.4f}',
            'aux': f'{aux_loss.item():.4f}'
        })

    return {
        'total': total_loss / n_batches,
        'recon': total_recon_loss / n_batches,
        'aux': total_aux_loss / n_batches
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: TrainingConfig,
    supervised: bool = True,
    aux_loss_weight: float = 2.0
) -> dict:
    """
    Valide le modèle.

    Args:
        model: Le modèle CryptoMAE.
        loader: DataLoader de validation.
        device: Device (cuda/cpu).
        config: Configuration d'entraînement.
        supervised: Si True, utilise les targets de direction.
        aux_loss_weight: Poids de la loss auxiliaire.

    Returns:
        Dict avec 'total', 'recon', 'aux', 'accuracy' métriques.
    """
    model.eval()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_aux_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    n_batches = 0

    for batch in loader:
        # Dépack le batch selon le mode
        if supervised:
            x, target_direction = batch
            x = x.to(device)
            target_direction = target_direction.to(device)
            # S'assurer que target_direction a shape (batch, 1)
            if target_direction.dim() == 1:
                target_direction = target_direction.unsqueeze(1)
        else:
            x = batch.to(device)
            target_direction = None

        with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
            pred, target_recon, mask, pred_logits = model(x, mask_ratio=config.mask_ratio)

            # 1. Reconstruction Loss
            pred_masked = pred[mask]
            recon_loss = nn.functional.mse_loss(pred_masked, target_recon)

            # 2. Auxiliary Prediction Loss
            if supervised and target_direction is not None:
                # Utiliser pos_weight pour pénaliser les Faux Positifs (modèle trop optimiste)
                pos_weight = torch.tensor([0.8], device=device)
                aux_loss = nn.functional.binary_cross_entropy_with_logits(
                    pred_logits.view(-1), target_direction.view(-1).float(),
                    pos_weight=pos_weight
                )
                loss = recon_loss + aux_loss_weight * aux_loss

                # Accuracy de prédiction de direction
                pred_direction = (torch.sigmoid(pred_logits) > 0.5).float()
                correct_predictions += (pred_direction == target_direction).sum().item()
                total_predictions += target_direction.numel()
            else:
                aux_loss = torch.tensor(0.0, device=device)
                loss = recon_loss

        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_aux_loss += aux_loss.item()
        n_batches += 1

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0

    return {
        'total': total_loss / n_batches,
        'recon': total_recon_loss / n_batches,
        'aux': total_aux_loss / n_batches,
        'accuracy': accuracy
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    config: TrainingConfig
):
    """Sauvegarde le checkpoint complet et l'encodeur seul."""

    # Créer le dossier si nécessaire
    os.makedirs(config.weights_dir, exist_ok=True)

    # 1. Checkpoint complet (pour reprendre l'entraînement)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'config': {
            'd_model': config.d_model,
            'n_heads': config.n_heads,
            'n_layers': config.n_layers,
            'dropout': config.dropout,
            'seq_len': config.seq_len,
        }
    }
    torch.save(checkpoint, os.path.join(ROOT_DIR, config.checkpoint_path))
    print(f"  [Checkpoint] Saved: {config.checkpoint_path}")

    # 2. CRUCIAL: Encodeur seul (pour l'agent RL Phase 3)
    # Inclut: embedding + pos_encoder + encoder
    encoder_state = {
        'embedding': model.embedding.state_dict(),
        'pos_encoder': model.pos_encoder.state_dict(),
        'encoder': model.encoder.state_dict(),
        'd_model': config.d_model,
    }
    torch.save(encoder_state, os.path.join(ROOT_DIR, config.encoder_path))
    print(f"  [Encoder] Saved: {config.encoder_path}")


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str,
    device: torch.device
) -> tuple[int, float]:
    """
    Charge un checkpoint et retourne l'état d'entraînement.

    Returns:
        (epoch, val_loss) du checkpoint.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['val_loss']


# =============================================================================
# Main Training Loop
# =============================================================================

def train(
    config: TrainingConfig = None,
    from_scratch: bool = False,
    extra_epochs: int = None,
    supervised: bool = True,
    aux_loss_weight: float = 2.0,
    limit: Optional[int] = None
):
    """
    Boucle d'entraînement principale.

    Args:
        config: Configuration (utilise défauts si None).
        from_scratch: Force un entraînement depuis zéro.
        extra_epochs: Nombre d'epochs supplémentaires (mode reprise).
        supervised: Si True, entraîne avec loss auxiliaire de direction.
        aux_loss_weight: Poids de la loss auxiliaire (défaut: 2.0 pour forcer l'apprentissage directionnel).
    """
    if config is None:
        config = TrainingConfig()

    print("=" * 70)
    print("FOUNDATION MODEL TRAINING - CryptoMAE")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Device
    device = get_device()

    # Data
    train_loader, val_loader, n_features = prepare_data(config, supervised=supervised)

    # Model
    print("\n[Model] Creating CryptoMAE...")
    model = CryptoMAE(
        input_dim=n_features,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        dropout=config.dropout
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] Parameters: {n_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay
    )

    # Mixed Precision Scaler
    scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))

    # ==========================================================================
    # Resume Logic
    # ==========================================================================

    checkpoint_file = os.path.join(ROOT_DIR, config.checkpoint_path)
    start_epoch = 1
    best_val_loss = float('inf')
    patience_counter = 0

    if not from_scratch and os.path.exists(checkpoint_file):
        # Charger le checkpoint
        loaded_epoch, loaded_val_loss = load_checkpoint(
            model, optimizer, checkpoint_file, device
        )
        best_val_loss = loaded_val_loss

        if extra_epochs is not None:
            # Mode: epochs supplémentaires
            start_epoch = loaded_epoch + 1
            end_epoch = loaded_epoch + extra_epochs
            print(f"\n[Resume] Loaded checkpoint from epoch {loaded_epoch} (val_loss: {loaded_val_loss:.4f})")
            print(f"[Resume] Training for {extra_epochs} additional epochs ({start_epoch} -> {end_epoch})")
        else:
            # Mode: continuer jusqu'à config.epochs
            start_epoch = loaded_epoch + 1
            end_epoch = config.epochs
            if start_epoch > end_epoch:
                print(f"\n[Resume] Already trained {loaded_epoch} epochs (target: {config.epochs})")
                print("[Resume] Nothing to do. Use --extra-epochs or increase --epochs.")
                return model, best_val_loss
            print(f"\n[Resume] Loaded checkpoint from epoch {loaded_epoch} (val_loss: {loaded_val_loss:.4f})")
            print(f"[Resume] Continuing to epoch {end_epoch} ({start_epoch} -> {end_epoch})")
    else:
        # Entraînement depuis zéro
        end_epoch = config.epochs
        if from_scratch and os.path.exists(checkpoint_file):
            print("\n[Training] Starting from scratch (--from-scratch flag)")
        else:
            print("\n[Training] Starting fresh (no checkpoint found)")

    start_time = time.time()

    print(f"\n[Training] Configuration:")
    print(f"  Epochs: {start_epoch} -> {end_epoch}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.lr}")
    print(f"  Mask ratio: {config.mask_ratio}")
    print(f"  Early stopping patience: {config.patience}")
    print(f"  Mode: {'Supervised (Task-Aware)' if supervised else 'Unsupervised (Reconstruction only)'}")
    if supervised:
        print(f"  Aux loss weight: {aux_loss_weight}")
    print()

    # ==========================================================================
    # TensorBoard Setup
    # ==========================================================================

    writer = None
    if config.tensorboard_log:
        run_name = config.run_name or f"MAE_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        log_dir = os.path.join(ROOT_DIR, config.tensorboard_log, run_name)
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        print(f"[TensorBoard] Logging to: {log_dir}")

    # ==========================================================================
    # Epoch Loop
    # ==========================================================================

    for epoch in range(start_epoch, end_epoch + 1):
        epoch_start = time.time()

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scaler, device, config,
            supervised=supervised, aux_loss_weight=aux_loss_weight, limit=limit
        )

        # Validate
        val_metrics = validate(
            model, val_loader, device, config,
            supervised=supervised, aux_loss_weight=aux_loss_weight
        )

        epoch_time = time.time() - epoch_start

        # Print progress
        if supervised:
            print(f"Epoch {epoch:02d}/{end_epoch} | "
                  f"Train: {train_metrics['total']:.4f} (R:{train_metrics['recon']:.4f} A:{train_metrics['aux']:.4f}) | "
                  f"Val: {val_metrics['total']:.4f} (Acc:{val_metrics['accuracy']:.1%}) | "
                  f"Time: {epoch_time:.1f}s")
        else:
            print(f"Epoch {epoch:02d}/{end_epoch} | "
                  f"Train Loss: {train_metrics['total']:.4f} | "
                  f"Val Loss: {val_metrics['total']:.4f} | "
                  f"Time: {epoch_time:.1f}s")

        # TensorBoard Logging
        if writer:
            writer.add_scalar("loss/train_total", train_metrics['total'], epoch)
            writer.add_scalar("loss/train_recon", train_metrics['recon'], epoch)
            writer.add_scalar("loss/val_total", val_metrics['total'], epoch)
            writer.add_scalar("loss/val_recon", val_metrics['recon'], epoch)
            writer.add_scalar("loss/best_val", best_val_loss, epoch)
            writer.add_scalar("time/epoch_seconds", epoch_time, epoch)
            if supervised:
                writer.add_scalar("loss/train_aux", train_metrics['aux'], epoch)
                writer.add_scalar("loss/val_aux", val_metrics['aux'], epoch)
                writer.add_scalar("accuracy/val_direction", val_metrics['accuracy'], epoch)

        # Early Stopping Check (based on total loss)
        val_loss = val_metrics['total']
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # Save best model
            save_checkpoint(model, optimizer, epoch, val_loss, config)
        else:
            patience_counter += 1
            print(f"  [Early Stopping] No improvement ({patience_counter}/{config.patience})")

            if patience_counter >= config.patience:
                print(f"\n[Early Stopping] Triggered at epoch {epoch}")
                break

    # ==========================================================================
    # Training Complete
    # ==========================================================================

    total_time = time.time() - start_time

    # Close TensorBoard writer
    if writer:
        writer.add_hparams(
            {"lr": config.lr, "batch_size": config.batch_size, "mask_ratio": config.mask_ratio},
            {"hparam/best_val_loss": best_val_loss, "hparam/epochs": end_epoch}
        )
        writer.close()

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Total time: {total_time / 60:.1f} minutes")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Checkpoint: {config.checkpoint_path}")
    print(f"Encoder weights: {config.encoder_path}")
    print("=" * 70)

    return model, best_val_loss


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    # Parse arguments si besoin
    import argparse

    parser = argparse.ArgumentParser(description="Train CryptoMAE Foundation Model")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    parser.add_argument("--from-scratch", action="store_true",
                        help="Force training from scratch, ignoring checkpoints")
    parser.add_argument("--extra-epochs", type=int, default=None,
                        help="Additional epochs to train (resume mode)")
    parser.add_argument("--unsupervised", action="store_true",
                        help="Train without auxiliary prediction loss (reconstruction only)")
    parser.add_argument("--aux-weight", type=float, default=2.0,
                        help="Weight for auxiliary prediction loss (default: 2.0)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of batches per epoch (for testing)")

    args = parser.parse_args()

    # Create config
    config = TrainingConfig()
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.lr = args.lr
    config.patience = args.patience

    # Train
    train(
        config,
        from_scratch=args.from_scratch,
        extra_epochs=args.extra_epochs,
        supervised=not args.unsupervised,
        aux_loss_weight=args.aux_weight,
        limit=args.limit
    )

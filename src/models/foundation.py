"""
foundation.py - Masked Auto-Encoder (MAE) pour séries temporelles financières.

Architecture:
- Embedding: Projection linéaire (input_dim -> d_model)
- Positional Encoding: Sinusoidal standard
- Encoder: TransformerEncoder (Pre-LN pour stabilité)
- Decoder: Projection inverse (d_model -> input_dim)

Référence: He et al. (2022) - Masked Autoencoders Are Scalable Vision Learners
Adapté pour les séries temporelles financières.
"""

import math
import torch
import torch.nn as nn
from typing import Tuple, Optional

from src.config.constants import MAE_D_MODEL, MAE_N_HEADS, MAE_N_LAYERS, MAE_DROPOUT


class SinusoidalPositionalEncoding(nn.Module):
    """
    Positional Encoding sinusoidal standard (Vaswani et al., 2017).

    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Args:
            d_model: Dimension du modèle.
            max_len: Longueur maximale de séquence.
            dropout: Taux de dropout.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Créer la matrice de positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Shape: (1, max_len, d_model) pour broadcasting avec batch
        pe = pe.unsqueeze(0)

        # Enregistrer comme buffer (non-trainable)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor de shape (batch, seq_len, d_model)

        Returns:
            Tensor avec positional encoding ajouté.
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class CryptoMAE(nn.Module):
    """
    Masked Auto-Encoder pour séries temporelles financières.

    Architecture:
    1. Linear Embedding (input_dim -> d_model)
    2. Sinusoidal Positional Encoding
    3. TransformerEncoder (n_layers, n_heads, norm_first=True)
    4. Linear Decoder (d_model -> input_dim)

    Le masquage aléatoire force le modèle à apprendre des représentations
    contextuelles des séries temporelles.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = MAE_D_MODEL,
        n_heads: int = MAE_N_HEADS,
        n_layers: int = MAE_N_LAYERS,
        dim_feedforward: Optional[int] = None,
        dropout: float = MAE_DROPOUT,
        max_len: int = 512
    ):
        """
        Args:
            input_dim: Nombre de features en entrée.
            d_model: Dimension du transformer.
            n_heads: Nombre de têtes d'attention.
            n_layers: Nombre de couches encoder.
            dim_feedforward: Dimension du FFN (défaut: 4 * d_model).
            dropout: Taux de dropout.
            max_len: Longueur maximale de séquence.
        """
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model

        # Feedforward dimension (standard: 4x d_model)
        if dim_feedforward is None:
            dim_feedforward = d_model * 4

        # 1. Embedding: Linear projection
        self.embedding = nn.Linear(input_dim, d_model)

        # 2. Positional Encoding: Sinusoidal
        self.pos_encoder = SinusoidalPositionalEncoding(
            d_model=d_model,
            max_len=max_len,
            dropout=dropout
        )

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-LN pour meilleure stabilité
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

        # 4. Decoder Head: Linear projection inverse
        self.decoder = nn.Linear(d_model, input_dim)

        # 5. Prediction Head: Classification de direction (auxiliaire)
        # Prend la représentation poolée et prédit la direction du marché
        self.prediction_head = nn.Linear(d_model, 1)

        # Learnable mask token (optionnel, utilisé à la place de 0)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, input_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        # Initialisation des poids
        self._init_weights()

    def _init_weights(self):
        """Initialisation Xavier/Glorot pour les couches linéaires."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _create_mask(
        self,
        x: torch.Tensor,
        mask_ratio: float
    ) -> torch.Tensor:
        """
        Crée un masque aléatoire pour le MAE.

        Args:
            x: Tensor de shape (batch, seq_len, input_dim)
            mask_ratio: Fraction de tokens à masquer.

        Returns:
            Boolean mask de shape (batch, seq_len).
            True = position masquée.
        """
        batch_size, seq_len, _ = x.shape

        # Nombre de tokens à masquer par séquence
        n_mask = int(seq_len * mask_ratio)

        # Générer des indices aléatoires pour le masquage
        # On utilise torch.rand pour avoir des valeurs différentes par batch
        noise = torch.rand(batch_size, seq_len, device=x.device)

        # Trier et prendre les n_mask plus petits indices
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Créer le masque: True pour les n_mask premiers indices
        mask = torch.zeros(batch_size, seq_len, device=x.device, dtype=torch.bool)
        mask.scatter_(1, ids_shuffle[:, :n_mask], True)

        return mask

    def forward(
        self,
        x: torch.Tensor,
        mask_ratio: float = 0.15
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass avec masquage et prédiction de direction.

        Args:
            x: Input tensor de shape (batch, seq_len, input_dim)
            mask_ratio: Fraction de tokens à masquer (0.0 à 1.0)

        Returns:
            pred: Prédictions de reconstruction (batch, seq_len, input_dim)
            target: Valeurs originales aux positions masquées (n_masked, input_dim)
            mask: Boolean mask (batch, seq_len), True = masqué
            pred_logits: Logits de prédiction de direction (batch, 1)
        """
        batch_size, seq_len, _ = x.shape

        # 1. Créer le masque aléatoire
        mask = self._create_mask(x, mask_ratio)

        # 2. Appliquer le masque (remplacer par mask_token)
        x_masked = x.clone()

        # Expand mask_token pour correspondre aux positions masquées
        mask_tokens = self.mask_token.expand(batch_size, seq_len, -1)

        # Remplacer les positions masquées
        x_masked = torch.where(
            mask.unsqueeze(-1).expand_as(x),
            mask_tokens,
            x_masked
        )

        # 3. Embedding
        x_emb = self.embedding(x_masked)

        # 4. Positional Encoding
        x_emb = self.pos_encoder(x_emb)

        # 5. Transformer Encoder
        encoded = self.encoder(x_emb)  # Shape: (batch, seq_len, d_model)

        # 6. Decoder (reconstruction)
        pred = self.decoder(encoded)

        # 7. Extraire les targets (valeurs originales aux positions masquées)
        target = x[mask]  # Shape: (n_masked_total, input_dim)

        # 8. Prediction Head (direction auxiliaire)
        # Global Average Pooling sur la dimension séquence
        latent_pooled = encoded.mean(dim=1)  # Shape: (batch, d_model)
        pred_logits = self.prediction_head(latent_pooled)  # Shape: (batch, 1)

        return pred, target, mask, pred_logits

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode sans masquage (pour l'inférence/extraction de features).

        Args:
            x: Input tensor de shape (batch, seq_len, input_dim)

        Returns:
            Encoded representations de shape (batch, seq_len, d_model)
        """
        x_emb = self.embedding(x)
        x_emb = self.pos_encoder(x_emb)
        encoded = self.encoder(x_emb)
        return encoded

    def get_reconstruction_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Calcule la MSE loss uniquement sur les positions masquées.

        Args:
            pred: Prédictions (batch, seq_len, input_dim)
            target: Valeurs originales masquées (n_masked, input_dim)
            mask: Boolean mask (batch, seq_len)

        Returns:
            Scalar loss.
        """
        # Extraire les prédictions aux positions masquées
        pred_masked = pred[mask]  # Shape: (n_masked_total, input_dim)

        # MSE sur les positions masquées uniquement
        loss = nn.functional.mse_loss(pred_masked, target)

        return loss


if __name__ == "__main__":
    # Test rapide
    batch_size = 4
    seq_len = 64
    input_dim = 35

    model = CryptoMAE(input_dim=input_dim)
    x = torch.randn(batch_size, seq_len, input_dim)

    pred, target, mask, pred_logits = model(x, mask_ratio=0.15)

    print(f"Input shape:       {x.shape}")
    print(f"Recon pred shape:  {pred.shape}")
    print(f"Target shape:      {target.shape}")
    print(f"Mask shape:        {mask.shape}")
    print(f"Pred logits shape: {pred_logits.shape}")
    print(f"Masked tokens:     {mask.sum().item()}")

    # Vérifier les shapes
    assert pred.shape == x.shape, "Reconstruction shape mismatch!"
    assert pred_logits.shape == (batch_size, 1), "Prediction logits shape mismatch!"
    print("\n[OK] Shape verification passed!")

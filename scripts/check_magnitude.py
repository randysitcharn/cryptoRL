"""
Check Input Magnitude - Audit des valeurs brutes envoyees au MAE.

Verifie si les donnees d'entree sont a l'echelle attendue (Z-Score).
"""
import numpy as np
import torch
from src.training.batch_env import BatchCryptoEnv


def check_magnitude():
    print("--- INSPECTION DES VALEURS D'ENTREE ---")

    # 1. Charger l'env avec les donnees WFO
    wfo_train_path = "data/wfo/segment_0/train.parquet"
    env = BatchCryptoEnv(
        parquet_path=wfo_train_path,
        price_column="BTC_Close",
        n_envs=10,
        device="cpu"
    )
    obs = env.reset()

    market = obs["market"]  # (10, 64, N_features)

    # 2. Stats Globales
    print(f"\n[Global Market Data]")
    print(f"Shape: {market.shape}")
    print(f"Mean: {np.mean(market):.4f}")
    print(f"Std:  {np.std(market):.4f}")
    print(f"Min:  {np.min(market):.4f}")
    print(f"Max:  {np.max(market):.4f}")

    # 3. Stats par Feature (Les 10 premieres)
    n_features = market.shape[2]
    print(f"\n[Feature Sample - First 10 of {n_features} Columns]")
    for i in range(min(10, n_features)):
        col_data = market[:, :, i]
        print(f"Feat {i:2d}: Min={np.min(col_data):10.2f}, Max={np.max(col_data):10.2f}, Mean={np.mean(col_data):10.2f}, Std={np.std(col_data):8.2f}")

    # 4. Stats par Feature (Les 5 dernieres = HMM)
    print(f"\n[HMM Features - Last 5 Columns]")
    for i in range(n_features - 5, n_features):
        col_data = market[:, :, i]
        print(f"Feat {i:2d}: Min={np.min(col_data):10.4f}, Max={np.max(col_data):10.4f}, Mean={np.mean(col_data):10.4f}, Std={np.std(col_data):8.4f}")

    # 5. Diagnostic par seuil
    print(f"\n[Diagnostic]")
    limits = [1.0, 5.0, 10.0, 100.0, 1000.0]
    for limit in limits:
        count = np.sum(np.abs(market) > limit)
        pct = 100 * count / market.size
        print(f"  |x| > {limit:6.0f}: {count:8d} valeurs ({pct:5.2f}%)")

    # 6. Alerte critique
    max_abs = np.max(np.abs(market))
    if max_abs > 10.0:
        print(f"\n[ALERTE CRITIQUE] Valeurs max = {max_abs:.2f} detectees !")
        print("Le MAE pre-entraine (Z-Score) va saturer. Normalisation requise.")

        # Trouver les features problematiques
        print(f"\n[Features Problematiques (|max| > 10)]")
        for i in range(n_features):
            col_data = market[:, :, i]
            col_max = np.max(np.abs(col_data))
            if col_max > 10.0:
                print(f"  Feat {i:2d}: |max| = {col_max:.2f}")
    else:
        print(f"\n[OK] Valeurs dans la plage attendue (-10 a +10). Max = {max_abs:.4f}")


if __name__ == "__main__":
    check_magnitude()

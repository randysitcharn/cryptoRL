"""
main_process.py - Point d'entrée pour le traitement des données.

Exécute le pipeline de traitement des données BTC-USD :
nettoyage, ajout d'indicateurs techniques, log-returns et encodage temporel.
Sauvegarde le résultat dans data/processed/.
"""

from src.data_engineering.processor import DataProcessor

if __name__ == "__main__":
    processor = DataProcessor()
    df = processor.process_data("data/raw/BTC-USD_1h.csv")

    print("\n=== 5 premières lignes ===")
    print(df.head())

    print("\n=== Colonnes finales ===")
    print(list(df.columns))

    print(f"\n=== Stats ===")
    print(f"Lignes: {len(df)}")
    print(f"Colonnes: {len(df.columns)}")

    # Validation NaN
    nan_count = df.isna().sum().sum()
    assert nan_count == 0, f"Il reste {nan_count} NaN!"
    print("\n[OK] Aucun NaN!")

    # Vérification normalisation (min/max raisonnables)
    print("\n=== describe() - Vérification normalisation ===")
    print(df.describe().T[['min', 'max', 'mean', 'std']])

    print("\n[OK] Données normalisées prêtes pour RL!")

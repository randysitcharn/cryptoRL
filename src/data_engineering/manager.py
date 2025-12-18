"""
manager.py - Orchestrateur de pipeline de données pour Trading RL.

Contient:
- RegimeDetector: Détection de régimes de marché via GMM-HMM (4 états)
- DataManager: Pipeline complet Download -> Features -> Regimes -> Scale -> Save

Référence: Hamilton (1989) - Regime Switching Models
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans
from hmmlearn.hmm import GMMHMM

from src.data_engineering.loader import MultiAssetDownloader
from src.data_engineering.features import FeatureEngineer


class RegimeDetector:
    """
    Détecteur de régimes de marché via GMM-HMM avec K-Means Warm Start.

    Approche SOTA:
    1. Features dédiées au HMM (lissées sur 168h = 1 semaine):
       - HMM_Trend: Moyenne glissante des Log-Returns
       - HMM_Vol: Volatilité Parkinson rolling
       - HMM_Momentum: RSI 14 normalisé [0, 1]

    2. Initialisation K-Means pour garantir des clusters séparés

    3. Mapping force brute basé sur means_[Trend]:
       - BULL: état avec Trend la plus élevée
       - BEAR: état avec Trend la plus faible
       - RANGE: état intermédiaire
    """

    # Features dédiées au HMM (calculées en interne)
    HMM_FEATURES = ['HMM_Trend', 'HMM_Vol', 'HMM_Momentum']

    # Fenêtre de lissage (1 semaine en heures)
    SMOOTHING_WINDOW = 168

    def __init__(
        self,
        n_components: int = 3,
        n_mix: int = 2,
        n_iter: int = 200,
        random_state: int = 42
    ):
        """
        Initialise le détecteur de régimes.

        Args:
            n_components: Nombre d'états cachés (3: Bull, Bear, Range).
            n_mix: Nombre de composantes du mélange gaussien.
            n_iter: Nombre d'itérations pour l'algorithme EM.
            random_state: Graine pour reproductibilité.
        """
        self.n_components = n_components
        self.n_mix = n_mix
        self.n_iter = n_iter
        self.random_state = random_state

        # Modèle HMM
        self.hmm: Optional[GMMHMM] = None

        # K-Means pour warm start
        self.kmeans: Optional[KMeans] = None

        # Scaler pour normaliser les features avant HMM
        self.scaler = StandardScaler()

        # Mapping des états HMM vers noms de régimes
        self.state_mapping: Dict[int, str] = {}
        self.regime_names = ['Bear', 'Range', 'Bull']

        # Statistiques par état (pour le mapping)
        self.state_stats: Dict[int, Dict] = {}

    def _compute_hmm_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule les features dédiées au HMM.

        Ces features sont lissées sur 168h pour capter les tendances
        de fond et ignorer le bruit horaire.

        Args:
            df: DataFrame avec BTC_LogRet, BTC_Parkinson, BTC_Close.

        Returns:
            DataFrame avec colonnes HMM_Trend, HMM_Vol, HMM_Momentum ajoutées.
        """
        df_result = df.copy()

        # 1. Trend_Feature: Moyenne glissante 168h des Log-Returns
        df_result['HMM_Trend'] = df['BTC_LogRet'].rolling(
            window=self.SMOOTHING_WINDOW,
            min_periods=self.SMOOTHING_WINDOW
        ).mean()

        # 2. Vol_Feature: Volatilité Parkinson rolling 168h
        df_result['HMM_Vol'] = df['BTC_Parkinson'].rolling(
            window=self.SMOOTHING_WINDOW,
            min_periods=self.SMOOTHING_WINDOW
        ).mean()

        # 3. Momentum_Feature: RSI 14 normalisé [0, 1]
        delta = df['BTC_Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14, min_periods=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()

        # Éviter division par zéro
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        df_result['HMM_Momentum'] = rsi / 100  # Normaliser [0, 1]

        print(f"  Computed HMM features (window={self.SMOOTHING_WINDOW}h)")

        return df_result

    def _initialize_hmm_with_kmeans(self, features_scaled: np.ndarray) -> None:
        """
        Initialise le HMM avec les centres K-Means (warm start).

        Cette technique garantit que le HMM démarre avec des clusters
        physiquement séparés au lieu d'une initialisation aléatoire.

        Args:
            features_scaled: Features scalées (n_samples, n_features).
        """
        # Fit K-Means
        self.kmeans = KMeans(
            n_clusters=self.n_components,
            random_state=self.random_state,
            n_init=10
        )
        self.kmeans.fit(features_scaled)

        print(f"  K-Means fitted (inertia={self.kmeans.inertia_:.2f})")

        # Créer le HMM avec init_params sans 'm' (means)
        # On va injecter les means manuellement
        self.hmm = GMMHMM(
            n_components=self.n_components,
            n_mix=self.n_mix,
            covariance_type='full',
            n_iter=self.n_iter,
            random_state=self.random_state,
            init_params='stc'  # startprob, transmat, covars (pas means)
        )

        # Initialiser le HMM pour créer les attributs
        # On fait un fit partiel juste pour initialiser la structure
        self.hmm._init(features_scaled, None)

        # Injecter les centres K-Means dans means_
        # means_ shape: (n_components, n_mix, n_features)
        kmeans_centers = self.kmeans.cluster_centers_

        # Répliquer les centres pour chaque mixture component
        # Avec un léger bruit pour différencier les mixtures
        np.random.seed(self.random_state)
        for i in range(self.n_components):
            for j in range(self.n_mix):
                noise = np.random.normal(0, 0.1, size=kmeans_centers.shape[1])
                self.hmm.means_[i, j, :] = kmeans_centers[i] + noise

        print(f"  HMM initialized with K-Means centers")

    def _map_states_to_regimes(self) -> Dict[int, str]:
        """
        Mappe les états HMM vers des régimes basé sur means_[Trend].

        Logique force brute:
        - BULL: état avec HMM_Trend (colonne 0) la plus élevée
        - BEAR: état avec HMM_Trend la plus faible
        - RANGE: état intermédiaire

        Returns:
            Dict mapping état -> nom de régime.
        """
        # means_ shape: (n_components, n_mix, n_features)
        # HMM_Trend est la colonne 0
        # Moyenne sur les mixtures pour chaque état
        trend_means = self.hmm.means_[:, :, 0].mean(axis=1)

        # Trier les états par Trend (du plus bas au plus haut)
        sorted_states = np.argsort(trend_means)

        mapping = {
            int(sorted_states[0]): 'Bear',   # Trend la plus faible
            int(sorted_states[1]): 'Range',  # Trend intermédiaire
            int(sorted_states[2]): 'Bull'    # Trend la plus élevée
        }

        # Calculer les statistiques pour l'affichage
        self.state_stats = {}
        for state in range(self.n_components):
            self.state_stats[state] = {
                'trend_mean': trend_means[state],
                'vol_mean': self.hmm.means_[state, :, 1].mean(),
                'momentum_mean': self.hmm.means_[state, :, 2].mean()
            }

        return mapping

    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Entraîne le HMM et prédit les probabilités de régime.

        Pipeline:
        1. Calcul des features HMM dédiées (168h smoothing)
        2. K-Means warm start
        3. Fit HMM
        4. Mapping force brute basé sur means_
        5. Création des colonnes Prob_Bear, Prob_Range, Prob_Bull

        Args:
            df: DataFrame avec BTC_LogRet, BTC_Parkinson, BTC_Close.

        Returns:
            DataFrame avec colonnes HMM et Prob_* ajoutées.
        """
        print("\n[RegimeDetector] Fitting GMM-HMM (3 states) with K-Means warm start...")

        # 1. Calculer les features HMM dédiées
        df_result = self._compute_hmm_features(df)

        # 2. Extraire les features et identifier les lignes valides
        features_raw = df_result[self.HMM_FEATURES].values
        valid_mask = ~np.isnan(features_raw).any(axis=1)
        features_valid = features_raw[valid_mask]

        if len(features_valid) < 100:
            raise ValueError(f"Not enough valid samples for HMM: {len(features_valid)}")

        print(f"  Valid samples: {len(features_valid)}")

        # 3. Scaler les features
        features_scaled = self.scaler.fit_transform(features_valid)

        # 4. K-Means warm start
        self._initialize_hmm_with_kmeans(features_scaled)

        # 5. Fit HMM (avec warm start)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.hmm.fit(features_scaled)

        print(f"  HMM converged: {self.hmm.monitor_.converged}")

        # 6. Mapping force brute basé sur means_
        self.state_mapping = self._map_states_to_regimes()

        print("  State mapping (based on HMM means_):")
        for state, name in sorted(self.state_mapping.items()):
            stats = self.state_stats[state]
            print(f"    State {state} -> {name}: "
                  f"trend={stats['trend_mean']:.6f}, "
                  f"vol={stats['vol_mean']:.6f}, "
                  f"momentum={stats['momentum_mean']:.3f}")

        # 7. Prédire les probabilités
        proba = self.hmm.predict_proba(features_scaled)

        # 8. Créer les colonnes de probabilités avec noms sémantiques
        col_names = ['Prob_Bear', 'Prob_Range', 'Prob_Bull']

        # Initialiser les colonnes avec NaN
        for col in col_names:
            df_result[col] = np.nan

        # Mapper les probabilités brutes vers les noms sémantiques
        valid_indices = df_result.index[valid_mask]
        for state, regime_name in self.state_mapping.items():
            col_name = f'Prob_{regime_name}'
            if col_name in col_names:
                df_result.loc[valid_indices, col_name] = proba[:, state]

        print(f"  Added columns: {', '.join(col_names)}")

        return df_result

    def get_dominant_regime(self, df: pd.DataFrame) -> pd.Series:
        """
        Retourne le régime dominant (argmax des probabilités).

        Args:
            df: DataFrame avec colonnes Prob_Bear, Prob_Range, Prob_Bull.

        Returns:
            Series avec le nom du régime dominant.
        """
        prob_cols = ['Prob_Bear', 'Prob_Range', 'Prob_Bull']
        probs = df[prob_cols].values

        # Argmax sur les colonnes ordonnées: 0=Bear, 1=Range, 2=Bull
        dominant_idx = np.argmax(probs, axis=1)
        regime_map = {0: 'Bear', 1: 'Range', 2: 'Bull'}
        dominant_regimes = pd.Series(
            [regime_map[idx] for idx in dominant_idx],
            index=df.index
        )

        return dominant_regimes


class DataManager:
    """
    Orchestrateur de pipeline de données pour Trading RL.

    Exécute le pipeline complet:
    1. Download multi-asset data
    2. Feature engineering (FFD, volatilité, Z-Score)
    3. Regime detection (GMM-HMM)
    4. Scaling global (RobustScaler)
    5. Export en Parquet
    """

    def __init__(
        self,
        data_dir: str = "data",
        ffd_window: int = 100,
        vol_window: int = 24,
        zscore_window: int = 720
    ):
        """
        Initialise le DataManager.

        Args:
            data_dir: Répertoire de données.
            ffd_window: Fenêtre pour FFD.
            vol_window: Fenêtre pour volatilité.
            zscore_window: Fenêtre pour Z-Score.
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        # Composants du pipeline
        self.downloader = MultiAssetDownloader()
        self.feature_engineer = FeatureEngineer(
            ffd_window=ffd_window,
            vol_window=vol_window,
            zscore_window=zscore_window
        )
        self.regime_detector = RegimeDetector()

        # Scaler global
        self.scaler = RobustScaler()
        self.scaler_path: Optional[str] = None

        # Colonnes à exclure du scaling (prix bruts, etc.)
        self.exclude_from_scaling = [
            'BTC_Close', 'ETH_Close', 'SPX_Close', 'DXY_Close', 'NASDAQ_Close',
            'BTC_Open', 'BTC_High', 'BTC_Low',
            'ETH_Open', 'ETH_High', 'ETH_Low',
            'SPX_Open', 'SPX_High', 'SPX_Low',
            'DXY_Open', 'DXY_High', 'DXY_Low',
            'NASDAQ_Open', 'NASDAQ_High', 'NASDAQ_Low',
            'BTC_Volume'
        ]

    def pipeline(
        self,
        save_path: Optional[str] = None,
        scaler_path: Optional[str] = None,
        use_cached_data: bool = True
    ) -> pd.DataFrame:
        """
        Exécute le pipeline complet de préparation des données.

        Args:
            save_path: Chemin de sauvegarde du parquet (défaut: data/processed_data.parquet).
            scaler_path: Chemin de sauvegarde du scaler (défaut: data/scaler.pkl).
            use_cached_data: Si True, utilise les données CSV existantes.

        Returns:
            DataFrame final prêt pour l'entraînement.
        """
        save_path = save_path or os.path.join(self.data_dir, "processed_data.parquet")
        scaler_path = scaler_path or os.path.join(self.data_dir, "scaler.pkl")
        self.scaler_path = scaler_path

        print("=" * 60)
        print("DATA PIPELINE - Starting...")
        print("=" * 60)

        # =====================================================================
        # ÉTAPE 1: Download / Load Data
        # =====================================================================
        print("\n[1/6] Loading multi-asset data...")

        csv_path = os.path.join(self.data_dir, "processed/multi_asset.csv")

        if use_cached_data and os.path.exists(csv_path):
            print(f"  Using cached data: {csv_path}")
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        else:
            print("  Downloading fresh data...")
            df = self.downloader.download_multi_asset()

        print(f"  Shape: {df.shape}")

        # =====================================================================
        # ÉTAPE 2: Feature Engineering
        # =====================================================================
        print("\n[2/6] Applying feature engineering...")
        df = self.feature_engineer.engineer_features(df)
        print(f"  Shape after features: {df.shape}")

        # =====================================================================
        # ÉTAPE 3: Regime Detection
        # =====================================================================
        print("\n[3/6] Detecting market regimes...")
        df = self.regime_detector.fit_predict(df)
        print(f"  Shape after regimes: {df.shape}")

        # =====================================================================
        # ÉTAPE 4: Final Cleanup
        # =====================================================================
        print("\n[4/6] Final cleanup (dropna)...")
        initial_rows = len(df)
        df = df.dropna()
        dropped = initial_rows - len(df)
        print(f"  Dropped {dropped} rows with NaN ({len(df)} remaining)")

        # =====================================================================
        # ÉTAPE 5: Global Scaling
        # =====================================================================
        print("\n[5/6] Applying RobustScaler...")

        # Identifier les colonnes à scaler
        cols_to_scale = [
            col for col in df.columns
            if col not in self.exclude_from_scaling
            and df[col].dtype in ['float64', 'float32', 'int64', 'int32']
        ]

        print(f"  Scaling {len(cols_to_scale)} columns")

        # Appliquer le scaler
        df[cols_to_scale] = self.scaler.fit_transform(df[cols_to_scale])

        # Sauvegarder le scaler
        with open(scaler_path, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'columns': cols_to_scale
            }, f)
        print(f"  Scaler saved to: {scaler_path}")

        # =====================================================================
        # ÉTAPE 6: Export
        # =====================================================================
        print("\n[6/6] Exporting to Parquet...")

        # Créer le dossier si nécessaire
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        df.to_parquet(save_path, engine='pyarrow')
        print(f"  Saved to: {save_path}")

        # =====================================================================
        # Résumé
        # =====================================================================
        print("\n" + "=" * 60)
        print("DATA PIPELINE - Complete!")
        print("=" * 60)
        print(f"Final shape: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        print(f"Output: {save_path}")
        print(f"Scaler: {scaler_path}")

        # Afficher les statistiques de régimes
        print("\n--- Regime Statistics ---")
        dominant = self.regime_detector.get_dominant_regime(df)
        for regime in ['Bear', 'Range', 'Bull']:
            count = (dominant == regime).sum()
            pct = 100 * count / len(dominant)
            print(f"  {regime}: {count} ({pct:.1f}%)")

        print("=" * 60)

        return df

    def load_processed_data(
        self,
        path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Charge les données processées depuis le fichier Parquet.

        Args:
            path: Chemin du fichier (défaut: data/processed_data.parquet).

        Returns:
            DataFrame des données processées.
        """
        path = path or os.path.join(self.data_dir, "processed_data.parquet")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Processed data not found: {path}")

        df = pd.read_parquet(path)
        print(f"Loaded processed data: {df.shape}")

        return df

    def load_scaler(
        self,
        path: Optional[str] = None
    ) -> Tuple[RobustScaler, list]:
        """
        Charge le scaler sauvegardé.

        Args:
            path: Chemin du fichier (défaut: data/scaler.pkl).

        Returns:
            Tuple (scaler, columns).
        """
        path = path or os.path.join(self.data_dir, "scaler.pkl")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Scaler not found: {path}")

        with open(path, 'rb') as f:
            data = pickle.load(f)

        return data['scaler'], data['columns']


if __name__ == "__main__":
    # Test du pipeline complet
    manager = DataManager()
    df = manager.pipeline()

    print("\n--- Sample of final data ---")
    print(df.head())

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
        n_components: int = 4,
        n_mix: int = 2,
        n_iter: int = 200,
        random_state: int = 42
    ):
        """
        Initialise le détecteur de régimes.

        Args:
            n_components: Nombre d'états cachés (4: Crash, Downtrend, Range, Uptrend).
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

        # Statistiques par état (pour debug)
        self.state_stats: Dict[int, Dict] = {}

        # Mapping stable: indices triés par mean_return (Bear < Range < Bull)
        # Permet d'éviter le Label Switching entre réentraînements
        self.sorted_indices: Optional[np.ndarray] = None

        # Flag pour savoir si le modèle est entraîné
        self._is_fitted = False

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

    def _compute_state_stats(self) -> None:
        """
        Calcule les statistiques par état pour debug/analyse.
        """
        # means_ shape: (n_components, n_mix, n_features)
        trend_means = self.hmm.means_[:, :, 0].mean(axis=1)

        self.state_stats = {}
        for state in range(self.n_components):
            self.state_stats[state] = {
                'trend_mean': trend_means[state],
                'vol_mean': self.hmm.means_[state, :, 1].mean(),
                'momentum_mean': self.hmm.means_[state, :, 2].mean()
            }

    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Entraîne le HMM et prédit les probabilités de régime.

        Pipeline:
        1. Calcul des features HMM dédiées (168h smoothing)
        2. K-Means warm start
        3. Fit HMM
        4. Smart Sorting: trier les états par mean_return (du pire au meilleur)
        5. Création des colonnes Prob_0, Prob_1, Prob_2, Prob_3

        Args:
            df: DataFrame avec BTC_LogRet, BTC_Parkinson, BTC_Close.

        Returns:
            DataFrame avec colonnes HMM et Prob_* ajoutées.
        """
        print(f"\n[RegimeDetector] Fitting GMM-HMM ({self.n_components} states) with K-Means warm start...")

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

        self._is_fitted = True
        print(f"  HMM converged: {self.hmm.monitor_.converged}")

        # 6. Compute stats for debug
        self._compute_state_stats()

        # 7. Prédire les probabilités brutes
        proba = self.hmm.predict_proba(features_scaled)

        # 8. Smart Sorting: calculer mean_return par état et trier
        #    Évite le Label Switching entre réentraînements
        #    Utilise les vrais log returns depuis BTC_Close (non scalés)
        btc_close = df_result.loc[df_result.index[valid_mask], 'BTC_Close'].values
        real_log_returns = np.zeros(len(btc_close))
        real_log_returns[1:] = np.log(btc_close[1:] / btc_close[:-1])

        state_returns = []
        for state in range(self.n_components):
            # État dominant pour chaque sample
            dominant = proba.argmax(axis=1)
            state_mask = dominant == state
            if state_mask.sum() > 0:
                mean_ret = real_log_returns[state_mask].mean()
            else:
                mean_ret = 0.0
            state_returns.append((state, mean_ret))

        # Trier par mean_return (du pire au meilleur)
        # Prob_0 = Crash, Prob_1 = Downtrend, Prob_2 = Range, Prob_3 = Uptrend
        state_returns.sort(key=lambda x: x[1])
        self.sorted_indices = np.array([s[0] for s in state_returns])

        print("  Smart Sorting (by real BTC log return, worst to best):")
        regime_labels = ['Crash', 'Downtrend', 'Range', 'Uptrend'][:self.n_components]
        for i, (state, mean_ret) in enumerate(state_returns):
            label = regime_labels[i] if i < len(regime_labels) else f"State{i}"
            annual_ret = mean_ret * 24 * 365 * 100  # Annualized %
            print(f"    Prob_{i} ({label}): HMM_State {state} (mean_ret={mean_ret*100:.4f}%/h, {annual_ret:+.1f}%/yr)")

        # 9. Créer les colonnes Prob_0, Prob_1, ..., Prob_N (triées)
        col_names = [f'Prob_{i}' for i in range(self.n_components)]

        for col in col_names:
            df_result[col] = np.nan

        valid_indices = df_result.index[valid_mask]
        for i in range(self.n_components):
            original_state = self.sorted_indices[i]
            df_result.loc[valid_indices, f'Prob_{i}'] = proba[:, original_state]

        print(f"  Added columns: {', '.join(col_names)}")

        return df_result

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prédit les probabilités de régime avec un HMM déjà entraîné.

        Utilisé pour le test set (évite data leakage).
        Utilise sorted_indices pour garantir un mapping stable.

        Args:
            df: DataFrame avec BTC_LogRet, BTC_Parkinson, BTC_Close.

        Returns:
            DataFrame avec colonnes Prob_0, Prob_1, ..., Prob_N ajoutées.
        """
        if not self._is_fitted:
            raise RuntimeError("HMM not fitted. Call fit_predict() first or load() a saved model.")

        if self.sorted_indices is None:
            raise RuntimeError("sorted_indices not set. Model may be from old version.")

        print("\n[RegimeDetector] Predicting with fitted HMM...")

        # 1. Calculer les features HMM dédiées
        df_result = self._compute_hmm_features(df)

        # 2. Extraire les features et identifier les lignes valides
        features_raw = df_result[self.HMM_FEATURES].values
        valid_mask = ~np.isnan(features_raw).any(axis=1)
        features_valid = features_raw[valid_mask]

        print(f"  Valid samples: {len(features_valid)}")

        # 3. Scaler les features (utilise le scaler déjà fitté)
        features_scaled = self.scaler.transform(features_valid)

        # 4. Prédire les probabilités brutes
        proba = self.hmm.predict_proba(features_scaled)

        # 5. Créer les colonnes Prob_0, Prob_1, ..., Prob_N (triées par Smart Sorting)
        col_names = [f'Prob_{i}' for i in range(self.n_components)]

        for col in col_names:
            df_result[col] = np.nan

        valid_indices = df_result.index[valid_mask]
        for i in range(self.n_components):
            original_state = self.sorted_indices[i]
            df_result.loc[valid_indices, f'Prob_{i}'] = proba[:, original_state]

        print(f"  Added columns: {', '.join(col_names)}")

        return df_result

    def save(self, path: str) -> None:
        """
        Sauvegarde le HMM entraîné avec le mapping stable.

        Args:
            path: Chemin du fichier pickle.
        """
        if not self._is_fitted:
            raise RuntimeError("HMM not fitted. Nothing to save.")

        data = {
            'hmm': self.hmm,
            'scaler': self.scaler,
            'kmeans': self.kmeans,
            'n_components': self.n_components,
            'n_mix': self.n_mix,
            'state_stats': self.state_stats,
            'sorted_indices': self.sorted_indices,  # Smart Sorting mapping
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"[RegimeDetector] Saved to {path}")
        print(f"  sorted_indices: {self.sorted_indices} (Bear, Range, Bull)")

    @classmethod
    def load(cls, path: str) -> 'RegimeDetector':
        """
        Charge un HMM sauvegardé avec son mapping stable.

        Args:
            path: Chemin du fichier pickle.

        Returns:
            Instance de RegimeDetector avec HMM chargé.
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)

        detector = cls(
            n_components=data['n_components'],
            n_mix=data['n_mix']
        )
        detector.hmm = data['hmm']
        detector.scaler = data['scaler']
        detector.kmeans = data['kmeans']
        detector.state_stats = data['state_stats']
        detector.sorted_indices = data.get('sorted_indices')  # Smart Sorting mapping
        detector._is_fitted = True

        print(f"[RegimeDetector] Loaded from {path}")
        if detector.sorted_indices is not None:
            print(f"  sorted_indices: {detector.sorted_indices} (Bear, Range, Bull)")
        return detector

    def get_dominant_regime(self, df: pd.DataFrame) -> pd.Series:
        """
        Retourne le régime dominant (argmax des probabilités).

        Args:
            df: DataFrame avec colonnes Prob_0, Prob_1, ..., Prob_N.

        Returns:
            Series avec l'index du régime dominant (0=Crash, 1=Downtrend, ...).
        """
        prob_cols = [f'Prob_{i}' for i in range(self.n_components)]
        probs = df[prob_cols].values

        # Argmax sur les colonnes
        dominant_idx = np.argmax(probs, axis=1)
        dominant_regimes = pd.Series(dominant_idx, index=df.index)

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

        # Colonnes à exclure du scaling (prix bruts, volumes bruts, et log-returns)
        self.exclude_from_scaling = [
            # Prix OHLC bruts
            'BTC_Close', 'ETH_Close', 'SPX_Close', 'DXY_Close', 'NASDAQ_Close',
            'BTC_Open', 'BTC_High', 'BTC_Low',
            'ETH_Open', 'ETH_High', 'ETH_Low',
            'SPX_Open', 'SPX_High', 'SPX_Low',
            'DXY_Open', 'DXY_High', 'DXY_Low',
            'NASDAQ_Open', 'NASDAQ_High', 'NASDAQ_Low',
            # Volumes bruts (VolRel sera scalé)
            'BTC_Volume', 'ETH_Volume', 'SPX_Volume', 'DXY_Volume', 'NASDAQ_Volume',
            # Log-returns (déjà clippés à +/- 20%, pas besoin de scaler)
            'BTC_LogRet', 'ETH_LogRet', 'SPX_LogRet', 'DXY_LogRet', 'NASDAQ_LogRet',
            # Probabilités HMM (déjà dans [0, 1], triées par Smart Sorting)
            'Prob_0', 'Prob_1', 'Prob_2', 'Prob_3',
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
        regime_labels = ['Crash', 'Downtrend', 'Range', 'Uptrend']
        for i in range(self.regime_detector.n_components):
            count = (dominant == i).sum()
            pct = 100 * count / len(dominant)
            label = regime_labels[i] if i < len(regime_labels) else f"State{i}"
            print(f"  Prob_{i} ({label}): {count} ({pct:.1f}%)")

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

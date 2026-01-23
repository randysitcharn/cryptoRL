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

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

from src.data_engineering.loader import MultiAssetDownloader
from src.data_engineering.features import FeatureEngineer
from src.data_engineering.historical_downloader import HistoricalDownloader
from src.data_engineering.processor import DataProcessor


class RegimeDetector:
    """
    Détecteur de régimes de marché via GMM-HMM avec K-Means Warm Start.

    Approche SOTA avec Leading Indicators + Post-Training State Sorting:
    1. Features dédiées au HMM (4 features):
       - HMM_Trend: Z-Score glissant des Log-Returns (normalisé pour éviter d'être écrasé par les autres features)
       - HMM_Vol: Volatilité Parkinson rolling 168h
       - HMM_RSI_14: RSI 14 depuis FeatureEngineer (via TA-Lib)
       - HMM_ADX_14: ADX 14 depuis FeatureEngineer (via TA-Lib)

    2. Initialisation K-Means pour garantir des clusters séparés

    3. Post-Training State Sorting par Trend croissant:
       - Après l'entraînement, les états sont triés par Trend (rendement) croissant
       - État 0 = Trend le plus bas (baissier)
       - État N = Trend le plus haut (haussier)
       - Les matrices internes du HMM sont réorganisées pour garantir un ordre stable
       - Résout le problème de "Label Switching" (Seed Robustness 30% -> 100%)
    """

    # Features dédiées au HMM (calculées en interne)
    # Features HMM stables (4 features orthogonales)
    # Configuration validée: génère 4 états distincts avec scaling interne
    HMM_FEATURES = [
        'HMM_Vol',      # Volatilité (log Parkinson)
        'HMM_RSI_14',   # RSI brut [0-100], scalé en interne
        'HMM_ADX_14',   # ADX brut [0-100], scalé en interne (force du trend)
        'HMM_Trend',    # Rolling mean log-returns
    ]

    # Archétypes en Z-Scores (unités standardisées après RobustScaler)
    # Utilisés pour aligner les états HMM de manière absolue via Hungarian Algorithm
    # Résout le problème de "Semantic Drift" entre segments WFO
    # NOTE: Valeurs en sigmas (écarts-types), pas en pourcentages bruts
    STATE_ARCHETYPES = {
        0: {'name': 'Crash',     'mean_ret': -2.5, 'mean_vol':  2.0},  # Crash violent, vol extrême
        1: {'name': 'Downtrend', 'mean_ret': -0.8, 'mean_vol':  0.5},  # Légèrement baissier
        2: {'name': 'Range',     'mean_ret':  0.0, 'mean_vol': -1.0},  # Neutre, vol faible
        3: {'name': 'Uptrend',   'mean_ret':  1.0, 'mean_vol':  0.5},  # Haussier, vol moyenne
    }

    # Fenêtre de lissage (24h pour plus de réactivité - fix HMM convergence)
    SMOOTHING_WINDOW = 24

    def __init__(
        self,
        n_components: int = 4,
        n_mix: int = 2,
        n_iter: int = 200,
        random_state: int = 42,
        transition_penalty: float = 0.1
    ):
        """
        Initialise le détecteur de régimes.

        Args:
            n_components: Nombre d'états cachés (4: Crash, Downtrend, Range, Uptrend).
            n_mix: Nombre de composantes du mélange gaussien.
            n_iter: Nombre d'itérations pour l'algorithme EM.
            random_state: Graine pour reproductibilité.
            transition_penalty: Pénalité pour transitions (0-1). Plus élevé = plus sticky.
                               Inspiré des Statistical Jump Models (Shu et al., 2024).
                               Default 0.1 = +10% probabilité de rester dans le même état.
        """
        self.n_components = n_components
        self.n_mix = n_mix
        self.n_iter = n_iter
        self.random_state = random_state
        self.transition_penalty = transition_penalty

        # Modèle HMM
        self.hmm: Optional[GMMHMM] = None

        # K-Means pour warm start
        self.kmeans: Optional[KMeans] = None

        # Scaler pour normaliser les features avant HMM
        # RobustScaler utilise médiane/IQR au lieu de moyenne/std, plus robuste aux outliers
        # Crucial pour RSI/MACD/ADX qui ont des échelles différentes
        self.scaler = RobustScaler()

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

        Ces features sont lissées sur SMOOTHING_WINDOW (24h) pour capter les tendances.
        NOTE: Les features sont retournées en valeurs BRUTES - le scaling est fait
        dans fit_predict() avec un RobustScaler interne avant passage au HMM.

        Args:
            df: DataFrame avec BTC_LogRet, BTC_Parkinson, BTC_Fracdiff.

        Returns:
            DataFrame avec colonnes HMM_Trend, HMM_Vol, HMM_Momentum ajoutées.
        """
        df_result = df.copy()

        # 1. Trend_Feature: Z-Score glissant des Log-Returns (normalisé pour éviter d'être écrasé)
        # Z-Score = (val - mean) / std sur fenêtre glissante
        # Cela normalise l'amplitude de HMM_Trend pour qu'elle soit comparable aux autres features HMM
        rolling_mean = df['BTC_LogRet'].rolling(
            window=self.SMOOTHING_WINDOW,
            min_periods=self.SMOOTHING_WINDOW
        ).mean()
        rolling_std = df['BTC_LogRet'].rolling(
            window=self.SMOOTHING_WINDOW,
            min_periods=self.SMOOTHING_WINDOW
        ).std()
        df_result['HMM_Trend'] = (df['BTC_LogRet'] - rolling_mean) / (rolling_std + 1e-8)  # +1e-8 pour éviter div/0

        # 2. Vol_Feature: Volatilité Parkinson rolling 168h
        # Apply Log transform for more Gaussian distribution (HMM-friendly)
        # Log transform reduces the impact of level shifts and makes KPSS more likely to pass
        hmm_vol_raw = df['BTC_Parkinson'].rolling(
            window=self.SMOOTHING_WINDOW,
            min_periods=self.SMOOTHING_WINDOW
        ).mean()
        df_result['HMM_Vol'] = np.log(hmm_vol_raw + 1e-6)

        # 3. Momentum Features depuis FeatureEngineer (via TA-Lib)
        # NOTE: Ces features sont DÉJÀ scalées par RobustScaler dans preprocess_segment
        # Ne PAS diviser par 100 - elles sont déjà en format mean≈0, std≈1, clip[-5,5]
        if 'BTC_RSI_14' in df.columns:
            df_result['HMM_RSI_14'] = df['BTC_RSI_14']  # Already scaled by RobustScaler
        else:
            print("  [WARNING] BTC_RSI_14 not found, using 0.0 (neutral scaled value)")
            df_result['HMM_RSI_14'] = 0.0  # Scaled neutral = 0

        # MACD Histogram: already scaled by RobustScaler
        if 'BTC_MACD_Hist' in df.columns:
            df_result['HMM_MACD_Hist'] = df['BTC_MACD_Hist']  # Already scaled
        else:
            print("  [WARNING] BTC_MACD_Hist not found, using 0.0 (neutral)")
            df_result['HMM_MACD_Hist'] = 0.0

        # ADX 14: already scaled by RobustScaler
        if 'BTC_ADX_14' in df.columns:
            df_result['HMM_ADX_14'] = df['BTC_ADX_14']  # Already scaled
        else:
            print("  [WARNING] BTC_ADX_14 not found, using 0.0 (neutral)")
            df_result['HMM_ADX_14'] = 0.0

        # 4. Funding Rate - REMOVED (see audit P1.2)
        # Synthetic funding rates cause spurious correlations and are not used as features.
        # The environment uses a fixed funding_rate=0.0001 for short position costs.
        # HMM_Funding has been removed from HMM_FEATURES to avoid learning artificial patterns.

        # 5. Risk-On/Off via SPX vs DXY
        # SPX monte + DXY baisse = Risk-On (bon pour crypto)
        # SPX baisse + DXY monte = Risk-Off (mauvais pour crypto)
        spx_ret = df.get('SPX_LogRet', pd.Series(0, index=df.index))
        dxy_ret = df.get('DXY_LogRet', pd.Series(0, index=df.index))
        risk_signal = spx_ret - dxy_ret  # Positif = Risk-On
        df_result['HMM_RiskOnOff'] = risk_signal.rolling(
            window=self.SMOOTHING_WINDOW, min_periods=self.SMOOTHING_WINDOW
        ).mean()

        # 6. Vol Ratio (short-term / long-term) - early warning
        # Ratio > 1 = volatilité en accélération = régime change imminent
        vol_short = df['BTC_Parkinson'].rolling(window=24, min_periods=24).mean()
        vol_long = df['BTC_Parkinson'].rolling(window=168, min_periods=168).mean()
        df_result['HMM_VolRatio'] = vol_short / (vol_long + 1e-10)

        # Clip HMM features pour stabilité numérique (outliers extrêmes uniquement)
        # NOTE: Le scaling interne du RegimeDetector (fit_predict) normalise ces features
        # avant de les passer au HMM, donc on ne clippe que les outliers vraiment extrêmes
        # HMM_Trend est maintenant un Z-Score (normalisé), donc on clippe à ±5 (5 écarts-types)
        df_result['HMM_Trend'] = df_result['HMM_Trend'].clip(-5.0, 5.0)  # Z-Score: ±5 std max (outliers)
        df_result['HMM_Vol'] = df_result['HMM_Vol'].clip(-12, 0)  # Log-vol bounds (élargi)
        # RSI, MACD, ADX: valeurs BRUTES ici, scalées dans fit_predict() - pas de clip agressif
        # RSI: 0-100, ADX: 0-100, MACD: variable selon prix
        df_result['HMM_RSI_14'] = df_result['HMM_RSI_14'].clip(0, 100)  # Bornes naturelles RSI
        df_result['HMM_MACD_Hist'] = df_result['HMM_MACD_Hist'].clip(-2000, 2000)  # Outliers extrêmes
        df_result['HMM_ADX_14'] = df_result['HMM_ADX_14'].clip(0, 100)  # Bornes naturelles ADX
        df_result['HMM_RiskOnOff'] = df_result['HMM_RiskOnOff'].clip(-0.05, 0.05)  # ±5% (élargi)
        df_result['HMM_VolRatio'] = df_result['HMM_VolRatio'].clip(0.1, 10.0)  # Ratio [0.1x, 10x]

        n_features = len(self.HMM_FEATURES)
        print(f"  Computed HMM features (window={self.SMOOTHING_WINDOW}h, {n_features} features: {', '.join(self.HMM_FEATURES)})")

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
        # covariance_type='diag' est plus stable numériquement que 'full'
        self.hmm = GMMHMM(
            n_components=self.n_components,
            n_mix=self.n_mix,
            covariance_type='diag',  # Diagonal = stable, avoids non-symmetric issues
            n_iter=self.n_iter,
            random_state=self.random_state,
            init_params='stc',  # startprob, transmat, covars (pas means)
            min_covar=1e-3,  # Régularisation supplémentaire
        )

        # Initialiser le HMM pour créer les attributs
        # On fait un fit partiel juste pour initialiser la structure
        self.hmm._init(features_scaled, None)

        # Injecter les centres K-Means dans means_
        # means_ shape: (n_components, n_mix, n_features)
        kmeans_centers = self.kmeans.cluster_centers_
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

    def _apply_transition_penalty(self) -> None:
        """
        Applique une pénalité de transition pour améliorer la persistence des régimes.

        Inspiré des Statistical Jump Models (Shu et al., 2024) qui montrent que
        la persistence explicite améliore le Sharpe ratio et réduit le drawdown.

        Formule: A_sticky = A * (1 - penalty) + I * penalty

        Effet: Augmente la probabilité de rester dans le même état (diagonale).
        Avec penalty=0.1: diagonale +10%, off-diagonale -10% proportionnellement.
        """
        if self.transition_penalty <= 0:
            return

        n = self.n_components
        penalty = min(self.transition_penalty, 0.5)  # Cap à 50% pour éviter HMM dégénéré

        # Récupérer la matrice de transition originale
        A = self.hmm.transmat_.copy()

        # Appliquer la pénalité: plus de poids sur la diagonale
        I = np.eye(n)
        A_sticky = A * (1.0 - penalty) + I * penalty

        # Renormaliser les lignes pour que chaque ligne somme à 1
        A_sticky = A_sticky / A_sticky.sum(axis=1, keepdims=True)

        # Remplacer dans le HMM
        self.hmm.transmat_ = A_sticky

        # Log le changement
        diag_before = np.diag(A).mean()
        diag_after = np.diag(A_sticky).mean()
        print(f"  Transition Penalty applied (penalty={penalty:.2f}): "
              f"diag_avg {diag_before:.3f} -> {diag_after:.3f}")

    def _sort_states_by_trend(self) -> np.ndarray:
        """
        Trie les états HMM par Trend croissant (rendement croissant).

        Après l'entraînement du HMM, réorganise les états pour garantir un ordre stable:
        - État 0 = Trend le plus bas (baissier)
        - État N = Trend le plus haut (haussier)

        Cette méthode réorganise les matrices internes du HMM (startprob_, transmat_, means_, covars_)
        pour garantir que l'ordre des états est toujours cohérent entre différents entraînements.

        Returns:
            np.ndarray: Permutation [old_state_idx -> new_state_idx]
                       Exemple: [2, 0, 3, 1] signifie que l'ancien état 2 devient le nouvel état 0.
        """
        if not self._is_fitted or self.hmm is None:
            raise RuntimeError("HMM must be fitted before sorting states")

        # Trouver l'index de la feature Trend dans HMM_FEATURES
        try:
            trend_idx = self.HMM_FEATURES.index('HMM_Trend')
        except ValueError:
            # Fallback: utiliser le dernier élément si 'HMM_Trend' n'est pas trouvé
            trend_idx = len(self.HMM_FEATURES) - 1
            print(f"  [WARNING] 'HMM_Trend' not found in HMM_FEATURES, using index {trend_idx}")

        # Extraire les moyennes de Trend pour chaque état
        # means_ shape: (n_components, n_mix, n_features)
        # On prend la moyenne des mixtures pour chaque état
        trend_means = np.zeros(self.n_components)
        for state in range(self.n_components):
            trend_means[state] = self.hmm.means_[state, :, trend_idx].mean()

        # Trouver l'ordre de tri (du plus baissier au plus haussier)
        sorted_indices = np.argsort(trend_means)

        print(f"  [HMM State Sorting] Trend means (before sort): {trend_means}")
        print(f"  [HMM State Sorting] Sorted order: {sorted_indices}")

        # Réorganiser les matrices internes du HMM
        # Note: sorted_indices[i] = ancien état qui devient le nouvel état i
        # Pour réorganiser, on utilise l'inverse: on veut que new_state[i] = old_state[sorted_indices[i]]

        # 1. startprob_ (probabilités initiales)
        self.hmm.startprob_ = self.hmm.startprob_[sorted_indices]

        # 2. transmat_ (matrice de transition)
        # Réorganiser les lignes ET les colonnes
        self.hmm.transmat_ = self.hmm.transmat_[sorted_indices][:, sorted_indices]

        # 3. means_ (moyennes des mixtures)
        self.hmm.means_ = self.hmm.means_[sorted_indices]

        # 4. covars_ (covariances des mixtures)
        self.hmm.covars_ = self.hmm.covars_[sorted_indices]

        # 5. weights_ (poids des mixtures dans GMMHMM)
        if hasattr(self.hmm, 'weights_'):
            self.hmm.weights_ = self.hmm.weights_[sorted_indices]

        # Après réorganisation des matrices internes, les prédictions du HMM
        # retourneront déjà les états dans le bon ordre (trié par Trend croissant).
        # On retourne l'identité [0, 1, 2, 3] pour indiquer qu'il n'y a plus besoin de mapping.
        print(f"  [HMM State Sorting] States reordered by Trend (ascending): "
              f"Old state order was {sorted_indices.tolist()}, now using identity mapping")

        # Retourner l'identité car les matrices sont maintenant dans le bon ordre
        # Les prédictions seront directement dans l'ordre trié (Trend croissant)
        return np.arange(self.n_components)

    def align_to_archetypes(self) -> np.ndarray:
        """
        Aligne les états HMM trouvés vers les archétypes fixes via Hungarian Algorithm.

        IMPORTANT: Les HMM means sont en espace scalé (z-scores).
        On utilise inverse_transform pour revenir en espace brut avant comparaison.

        Résout le problème de "Semantic Drift" entre segments WFO:
        - Prob_0 signifie TOUJOURS "situation type Crash" (-5%/h, haute vol)
        - Prob_3 signifie TOUJOURS "situation type Uptrend" (+0.15%/h)

        Returns:
            np.ndarray: Mapping [archetype_idx -> hmm_state_idx]
        """
        from scipy.optimize import linear_sum_assignment
        from scipy.spatial.distance import cdist

        if not self._is_fitted:
            raise RuntimeError("HMM not fitted")

        n = self.n_components
        n_features = len(self.HMM_FEATURES)

        # 1. Extraire les means scalés de chaque état HMM
        # means_ shape: (n_states, n_mix, n_features)
        # On prend la moyenne des mixtures
        scaled_means = np.zeros((n, n_features))
        for state in range(n):
            scaled_means[state, :] = self.hmm.means_[state, :, :].mean(axis=0)

        # 2. INVERSE TRANSFORM: convertir de z-scores vers valeurs brutes
        # Utilise le scaler déjà fitté sur les données d'entraînement
        raw_means = self.scaler.inverse_transform(scaled_means)

        # 3. Extraire [trend, vol] pour chaque état (features 0 et 1)
        current_features = raw_means[:, :2]  # [HMM_Trend, HMM_Vol]

        # 4. Construire la matrice des archétypes (valeurs brutes)
        archetype_features = np.array([
            [self.STATE_ARCHETYPES[i]['mean_ret'], self.STATE_ARCHETYPES[i]['mean_vol']]
            for i in range(n)
        ])

        # 5. Calculer la matrice de distances
        # Pondérer vol plus fort car c'est plus discriminant
        weights = np.array([1.0, 2.0])  # [ret_weight, vol_weight]
        weighted_current = current_features * weights
        weighted_archetypes = archetype_features * weights

        cost_matrix = cdist(weighted_archetypes, weighted_current, metric='euclidean')

        # 6. Check for NaN/Inf in cost_matrix (can happen when HMM converges to 1 state)
        if not np.isfinite(cost_matrix).all():
            print("  [WARNING] Cost matrix contains NaN/Inf - using default identity mapping")
            # Replace NaN/Inf with large values to allow Hungarian to proceed
            cost_matrix = np.nan_to_num(cost_matrix, nan=1e10, posinf=1e10, neginf=1e10)

        # 7. Hungarian Algorithm pour appariement optimal
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Debug: afficher le mapping trouvé
        print("  Archetype Alignment (raw space, inverse transformed):")
        for i in range(n):
            matched_state = col_ind[i]
            dist = cost_matrix[i, matched_state]
            print(f"    {self.STATE_ARCHETYPES[i]['name']:10s} -> State {matched_state} "
                  f"(dist={dist:.6f}, trend={current_features[matched_state, 0]:.6f}, "
                  f"vol={current_features[matched_state, 1]:.5f})")

        return col_ind

    def _validate_hmm_quality(
        self,
        proba: np.ndarray,
        returns: np.ndarray,
        min_proportion: float = 0.05
    ) -> Dict:
        """
        Valide que les états HMM sont bien distincts et utilisés.

        Args:
            proba: Probabilités HMM (n_samples, n_components).
            returns: Log-returns réels pour calculer mean_ret par état.
            min_proportion: Proportion minimum par état (défaut 5%).

        Returns:
            Dict avec métriques de qualité:
            - n_active_states: Nombre d'états avec proportion > min_proportion
            - state_proportions: Liste des proportions par état
            - state_mean_returns: Liste des mean_ret par état
            - is_valid: True si au moins 3 états actifs
            - separation_score: Écart-type des mean_returns (plus élevé = meilleur)
        """
        dominant = proba.argmax(axis=1)
        n_samples = len(dominant)

        state_proportions = []
        state_mean_returns = []
        n_active = 0

        for state in range(self.n_components):
            mask = dominant == state
            proportion = mask.sum() / n_samples

            if mask.sum() > 0:
                mean_ret = returns[mask].mean()
            else:
                mean_ret = 0.0

            state_proportions.append(proportion)
            state_mean_returns.append(mean_ret)

            if proportion >= min_proportion:
                n_active += 1

        # Score de séparation: écart-type des mean_returns
        separation = np.std(state_mean_returns) if len(state_mean_returns) > 1 else 0.0

        quality = {
            'n_active_states': n_active,
            'state_proportions': state_proportions,
            'state_mean_returns': state_mean_returns,
            'is_valid': n_active >= 3,  # Au moins 3 états distincts
            'separation_score': separation
        }

        return quality

    def fit_predict(
        self,
        df: pd.DataFrame,
        tensorboard_log: Optional[str] = None,
        run_name: Optional[str] = None,
        segment_id: int = 0
    ) -> pd.DataFrame:
        """
        Entraîne le HMM et prédit les probabilités de régime.

        Pipeline:
        1. Calcul des features HMM dédiées (168h smoothing)
        2. K-Means warm start
        3. Fit HMM
        4. Post-Training State Sorting: trie les états par Trend croissant (rendement croissant)
        5. Création des colonnes Prob_0 (Trend le plus bas), Prob_1, ..., Prob_N (Trend le plus haut)

        Les états sont réorganisés dans les matrices internes du HMM pour garantir un ordre stable
        entre différents entraînements, résolvant le problème de "Label Switching" (Seed Robustness).

        Args:
            df: DataFrame avec BTC_LogRet, BTC_Parkinson, BTC_Fracdiff.
            tensorboard_log: Chemin pour logs TensorBoard (optional).
            run_name: Nom du run TensorBoard (optional).
            segment_id: ID du segment WFO pour TensorBoard step (default 0).

        Returns:
            DataFrame avec colonnes HMM et Prob_* ajoutées.
            Prob_0 = Trend le plus bas (baissier), Prob_N = Trend le plus haut (haussier).
        """
        print(f"\n[RegimeDetector] Fitting GMM-HMM ({self.n_components} states) with K-Means warm start...")

        # TensorBoard setup
        writer = None
        if tensorboard_log and HAS_TENSORBOARD:
            from datetime import datetime
            run_name = run_name or f"HMM_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            log_dir = os.path.join(tensorboard_log, run_name)
            os.makedirs(log_dir, exist_ok=True)
            writer = SummaryWriter(log_dir=log_dir)
            print(f"  [TensorBoard] Logging to: {log_dir}")

        # 1. Calculer les features HMM dédiées
        df_result = self._compute_hmm_features(df)

        # 2. Extraire les features et identifier les lignes valides
        features_raw = df_result[self.HMM_FEATURES].values
        valid_mask = np.isfinite(features_raw).all(axis=1)  # Attrape NaN ET Inf
        features_valid = features_raw[valid_mask]

        if len(features_valid) < 100:
            raise ValueError(f"Not enough valid samples for HMM: {len(features_valid)}")

        print(f"  Valid samples: {len(features_valid)}")

        # 3. Scaler les features - Utilise DataProcessor unifié (config spécifique pour HMM)
        # Note: HMM utilise min_iqr=1.0 (plus élevé) car les features HMM sont déjà partiellement normalisées
        hmm_processor = DataProcessor(config={'min_iqr': 1.0, 'clip_range': (-5, 5)})
        
        # Convertir en DataFrame pour DataProcessor
        features_df = pd.DataFrame(features_valid, columns=self.HMM_FEATURES)
        hmm_processor.fit(features_df)
        features_scaled_df = hmm_processor.transform(features_df)
        features_scaled = features_scaled_df.values
        
        # Récupérer le scaler pour compatibilité (utilisé dans align_to_archetypes)
        self.scaler = hmm_processor.get_scaler()

        # Pré-calculer les log-returns pour validation de qualité
        btc_close = df_result.loc[df_result.index[valid_mask], 'BTC_Close'].values
        real_log_returns = np.zeros(len(btc_close))
        real_log_returns[1:] = np.log(btc_close[1:] / btc_close[:-1])

        # 4. K-Means warm start + Fit HMM avec retry si qualité insuffisante
        MAX_RETRIES = 3
        best_quality = None
        best_hmm = None
        best_proba = None

        for attempt in range(MAX_RETRIES):
            # K-Means avec random_state différent à chaque tentative
            current_random_state = self.random_state + attempt * 17

            self.kmeans = KMeans(
                n_clusters=self.n_components,
                random_state=current_random_state,
                n_init=10
            )
            self.kmeans.fit(features_scaled)

            if attempt == 0:
                print(f"  K-Means fitted (inertia={self.kmeans.inertia_:.2f})")

            # Créer et initialiser le HMM
            self.hmm = GMMHMM(
                n_components=self.n_components,
                n_mix=self.n_mix,
                covariance_type='diag',
                n_iter=self.n_iter,
                random_state=current_random_state,
                init_params='stc',
                min_covar=1e-3,
            )
            self.hmm._init(features_scaled, None)

            # Injecter les centres K-Means dans means_
            kmeans_centers = self.kmeans.cluster_centers_
            np.random.seed(current_random_state)
            for i in range(self.n_components):
                for j in range(self.n_mix):
                    noise = np.random.normal(0, 0.1, size=kmeans_centers.shape[1])
                    self.hmm.means_[i, j, :] = kmeans_centers[i] + noise

            # Fit HMM
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.hmm.fit(features_scaled)

            # Fix NaN in startprob_
            if np.any(np.isnan(self.hmm.startprob_)):
                self.hmm.startprob_ = np.ones(self.n_components) / self.n_components

            # Prédire et valider
            try:
                proba = self.hmm.predict_proba(features_scaled)
            except ValueError:
                proba = np.ones((len(features_scaled), self.n_components)) / self.n_components

            quality = self._validate_hmm_quality(proba, real_log_returns)

            # Log la qualité
            if attempt == 0 or quality['n_active_states'] > (best_quality['n_active_states'] if best_quality else 0):
                best_quality = quality
                best_hmm = self.hmm
                best_proba = proba

            print(f"  [Attempt {attempt+1}/{MAX_RETRIES}] n_active_states={quality['n_active_states']}, "
                  f"separation={quality['separation_score']*100:.4f}%, valid={quality['is_valid']}")

            if quality['is_valid']:
                break

        # Utiliser le meilleur résultat
        self.hmm = best_hmm
        proba = best_proba

        if not best_quality['is_valid']:
            print(f"  [WARNING] HMM quality insufficient after {MAX_RETRIES} attempts. "
                  f"Best: {best_quality['n_active_states']} active states.")

        self._is_fitted = True
        print(f"  HMM converged: {self.hmm.monitor_.converged}")

        # 5. Apply transition penalty for better regime persistence (Sticky HMM)
        #    Inspired by Statistical Jump Models (Shu et al., 2024)
        self._apply_transition_penalty()

        # Log HMM convergence to TensorBoard
        if writer:
            history = self.hmm.monitor_.history
            print(f"  EM iterations: {len(history)}")

            # 1. Log likelihood par itération EM
            for i, score in enumerate(history):
                writer.add_scalar("hmm/log_likelihood", score, i)

            # 2. Amélioration par itération (delta)
            if len(history) > 1:
                for i in range(1, len(history)):
                    delta = history[i] - history[i-1]
                    writer.add_scalar("hmm/log_likelihood_delta", delta, i)

            # 3. Métriques finales groupées (use segment_id for WFO curves)
            writer.add_scalar("hmm/final/converged", float(self.hmm.monitor_.converged), segment_id)
            writer.add_scalar("hmm/final/n_iterations", len(history), segment_id)
            writer.add_scalar("hmm/final/kmeans_inertia", self.kmeans.inertia_, segment_id)
            writer.add_scalar("hmm/final/log_likelihood", history[-1] if history else 0, segment_id)

            # 4. Transition matrix entropy (régularité des transitions)
            transmat = self.hmm.transmat_
            entropy = -np.sum(transmat * np.log(transmat + 1e-10)) / self.n_components
            writer.add_scalar("hmm/final/transmat_entropy", entropy, segment_id)

            # 5. Transition penalty metrics (Sticky HMM)
            diag_avg = np.diag(transmat).mean()
            writer.add_scalar("hmm/final/transmat_diag_avg", diag_avg, segment_id)
            writer.add_scalar("hmm/final/transition_penalty", self.transition_penalty, segment_id)

        # 6. Compute stats for debug
        self._compute_state_stats()

        # 7. Post-Training State Sorting by Trend (rendement croissant)
        #    Trie les états HMM par Trend croissant pour garantir un ordre stable entre entraînements.
        #    État 0 = Trend le plus bas (baissier), État N = Trend le plus haut (haussier)
        #    Réorganise les matrices internes du HMM (startprob_, transmat_, means_, covars_)
        #    pour garantir la cohérence de l'ordre des états.
        self.sorted_indices = self._sort_states_by_trend()

        # Recalculer proba après réorganisation (les matrices sont maintenant dans le bon ordre)
        try:
            proba = self.hmm.predict_proba(features_scaled)
        except ValueError:
            proba = np.ones((len(features_scaled), self.n_components)) / self.n_components

        # Calculer les mean_returns réels pour TensorBoard (info seulement)
        state_returns = []
        dominant = proba.argmax(axis=1)
        for i in range(self.n_components):
            # Après réorganisation, l'état i correspond déjà au nouvel état i (trié)
            state_mask = dominant == i
            if state_mask.sum() > 0:
                mean_ret = real_log_returns[state_mask].mean()
            else:
                mean_ret = 0.0
            state_returns.append((i, mean_ret))

        print(f"  Final state order (sorted by Trend ascending): States 0..{self.n_components-1}")

        # 9. Créer les colonnes Prob_0, Prob_1, ..., Prob_N (déjà triées après réorganisation)
        col_names = [f'Prob_{i}' for i in range(self.n_components)]

        for col in col_names:
            df_result[col] = np.nan

        valid_indices = df_result.index[valid_mask]
        # Après réorganisation des matrices, proba est déjà dans le bon ordre
        for i in range(self.n_components):
            df_result.loc[valid_indices, f'Prob_{i}'] = proba[:, i]

        print(f"  Added columns: {', '.join(col_names)}")

        # Log final state distributions to TensorBoard
        if writer:
            dominant = proba.argmax(axis=1)
            for i, (state, mean_ret) in enumerate(state_returns):
                annual_ret = mean_ret * 24 * 365 * 100
                # Après réorganisation, l'état i correspond déjà au nouvel état i (trié)
                state_pct = (dominant == i).sum() / len(dominant) * 100
                writer.add_scalar(f"hmm/state_{i}/annual_return_pct", annual_ret, segment_id)
                writer.add_scalar(f"hmm/state_{i}/distribution_pct", state_pct, segment_id)

            writer.close()

        # 10. Extraire les Belief States (probabilités filtrées) pour alimenter le TQC
        # NOTE: Pour l'entraînement, on utilise use_forward_only=False (accepte le lissage intra-segment)
        # Pour le live/test, utiliser get_belief_states_df() avec use_forward_only=True
        print("\n[RegimeDetector] Extracting belief states for TQC...")
        df_result = self.get_belief_states_df(df_result, use_forward_only=False)
        print("  [NOTE] Belief states extracted with Forward-Backward (acceptable for training)")
        print("  [NOTE] For live/test inference, use get_belief_states_df(use_forward_only=True)")

        return df_result

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prédit les probabilités de régime avec un HMM déjà entraîné.

        Utilisé pour le test set (évite data leakage).
        Les états sont triés par Trend croissant (rendement croissant) après fit_predict(),
        donc les prédictions sont déjà dans le bon ordre.

        Args:
            df: DataFrame avec BTC_LogRet, BTC_Parkinson, BTC_Close.

        Returns:
            DataFrame avec colonnes Prob_0, Prob_1, ..., Prob_N ajoutées.
            Prob_0 = Trend le plus bas (baissier), Prob_N = Trend le plus haut (haussier).
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
        valid_mask = np.isfinite(features_raw).all(axis=1)  # Attrape NaN ET Inf
        features_valid = features_raw[valid_mask]

        print(f"  Valid samples: {len(features_valid)}")

        # 3. Scaler les features (utilise le scaler déjà fitté via DataProcessor)
        # Convertir en DataFrame pour compatibilité avec DataProcessor
        features_df = pd.DataFrame(features_valid, columns=self.HMM_FEATURES)
        
        # Créer un DataProcessor temporaire avec le scaler déjà fitté
        # (pour utiliser transform() qui gère le clipping)
        temp_processor = DataProcessor(config={'min_iqr': 1.0, 'clip_range': (-5, 5)})
        temp_processor.scaler = self.scaler  # Utiliser le scaler déjà fitté
        features_scaled_df = temp_processor.transform(features_df, columns=self.HMM_FEATURES)
        features_scaled = features_scaled_df.values

        # 4. Prédire les probabilités brutes
        try:
            proba = self.hmm.predict_proba(features_scaled)
        except ValueError as e:
            print(f"  [WARNING] HMM predict_proba failed: {e}")
            print(f"  [FALLBACK] Using uniform probabilities")
            proba = np.ones((len(features_scaled), self.n_components)) / self.n_components

        # 5. Créer les colonnes Prob_0, Prob_1, ..., Prob_N (triées par Trend croissant)
        # Après réorganisation des matrices dans fit_predict(), les prédictions sont déjà dans le bon ordre.
        # sorted_indices est l'identité [0,1,2,3] après réorganisation, donc on utilise directement proba[:, i]
        col_names = [f'Prob_{i}' for i in range(self.n_components)]

        for col in col_names:
            df_result[col] = np.nan

        valid_indices = df_result.index[valid_mask]
        # Matrices réorganisées: proba est déjà dans le bon ordre (Trend croissant)
        # Pour compatibilité avec anciens modèles, on vérifie si sorted_indices est l'identité
        if np.array_equal(self.sorted_indices, np.arange(self.n_components)):
            for i in range(self.n_components):
                df_result.loc[valid_indices, f'Prob_{i}'] = proba[:, i]
        else:
            # Ancien modèle (avant réorganisation): utiliser le mapping
            for i in range(self.n_components):
                original_state = self.sorted_indices[i]
                df_result.loc[valid_indices, f'Prob_{i}'] = proba[:, original_state]

        print(f"  Added columns: {', '.join(col_names)}")

        # 6. Extraire les Belief States (probabilités filtrées) pour alimenter le TQC
        # Pour le test/live, on utilise Forward-Only pour éviter le look-ahead bias
        print("\n[RegimeDetector] Extracting belief states (forward-only, no look-ahead)...")
        df_result = self.get_belief_states_df(df_result, use_forward_only=True)

        return df_result

    def get_belief_states(
        self,
        features_scaled: np.ndarray,
        use_forward_only: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extrait les probabilités filtrées (Belief States) des états HMM.
        
        IMPORTANT: Cette méthode calcule P(s_t | y_{1:t}) (Forward-Only) pour éviter
        le Look-Ahead Bias. Contrairement à `predict_proba()` qui utilise Forward-Backward
        et inclut l'information future, cette méthode est adaptée au trading en temps réel.
        
        Args:
            features_scaled: Features HMM scalées (n_samples, n_features).
                             Doit être déjà scalé avec le même scaler que l'entraînement.
            use_forward_only: Si True, utilise l'algorithme Forward uniquement (pas de look-ahead).
                             Si False, utilise predict_proba() standard (plus rapide mais avec look-ahead).
        
        Returns:
            Tuple (probs, entropy):
            - probs: Array (n_samples, n_components) avec probabilités filtrées P(s_t | y_{1:t})
            - entropy: Array (n_samples,) avec entropie de Shannon normalisée [0, 1]
                      (0.0 = certitude absolue, 1.0 = confusion totale)
        
        Note:
            Pour le backtesting WFO, on peut accepter le lissage intra-segment d'entraînement,
            mais pour l'inférence live/test, il faut utiliser use_forward_only=True.
        """
        if not self._is_fitted or self.hmm is None:
            raise RuntimeError("HMM must be fitted before extracting belief states")
        
        if use_forward_only:
            # Algorithme Forward-Only (pas de look-ahead bias)
            probs = self._forward_filter(features_scaled)
        else:
            # Utilise Forward-Backward (plus rapide mais avec look-ahead)
            # Acceptable pour l'entraînement si on accepte le lissage intra-segment
            probs = self.hmm.predict_proba(features_scaled)
        
        # Calcul de l'entropie de Shannon (incertitude du régime)
        # H = -sum_i P(s_t=i) * log(P(s_t=i))
        # Plus l'entropie est élevée, plus l'incertitude est grande
        from scipy.stats import entropy
        entropy_vals = entropy(probs.T, base=2)  # base=2 pour bits
        
        # Normalisation: 0.0 = certitude absolue, 1.0 = confusion totale
        # L'entropie max pour n_components états est log2(n_components)
        max_entropy = np.log2(self.n_components)  # log2(4) ≈ 1.386 pour 4 états
        normalized_entropy = entropy_vals / max_entropy
        
        return probs, normalized_entropy
    
    def _forward_filter(self, X: np.ndarray) -> np.ndarray:
        """
        Implémente l'algorithme Forward pour calculer P(s_t | y_{1:t}).
        
        Algorithme Forward:
        - alpha[t, i] = P(y_1:t, s_t=i | model)
        - P(s_t=i | y_{1:t}) = alpha[t, i] / sum_j alpha[t, j]
        
        Args:
            X: Features scalées (n_samples, n_features)
        
        Returns:
            Array (n_samples, n_components) avec probabilités filtrées
        """
        n_samples, n_features = X.shape
        n_states = self.n_components
        
        # Initialisation: alpha[0, i] = startprob[i] * P(y_0 | s_0=i)
        alpha = np.zeros((n_samples, n_states))
        
        # Émission probabilities pour t=0
        log_emissions = self._compute_log_emissions(X[0:1])  # (1, n_states)
        log_emissions = log_emissions[0]  # (n_states,)
        
        # Initialisation avec normalisation log-space pour stabilité numérique
        log_alpha_prev = np.log(self.hmm.startprob_ + 1e-10) + log_emissions
        # Normalisation log-space: log(sum(exp(log_alpha))) = logsumexp
        log_alpha_prev = log_alpha_prev - self._logsumexp(log_alpha_prev)
        alpha[0] = np.exp(log_alpha_prev)
        
        # Récursion Forward pour t=1 à T-1
        for t in range(1, n_samples):
            # Émission probabilities pour t
            log_emissions = self._compute_log_emissions(X[t:t+1])[0]  # (n_states,)
            
            # Forward recursion: alpha[t, i] = sum_j alpha[t-1, j] * transmat[j, i] * emission[i, y_t]
            # En log-space pour stabilité: log_alpha[t, i] = logsumexp_j(log_alpha[t-1, j] + log_trans[j, i] + log_emission[i])
            log_alpha_t = np.zeros(n_states)
            for i in range(n_states):
                # log_alpha[t-1, j] + log_trans[j, i]
                log_transitions = np.log(self.hmm.transmat_[:, i] + 1e-10)  # (n_states,)
                log_terms = log_alpha_prev + log_transitions
                # logsumexp pour sommer sur j
                log_alpha_t[i] = self._logsumexp(log_terms) + log_emissions[i]
            
            # Normalisation
            log_alpha_t = log_alpha_t - self._logsumexp(log_alpha_t)
            alpha[t] = np.exp(log_alpha_t)
            log_alpha_prev = log_alpha_t.copy()
        
        return alpha
    
    def _compute_log_emissions(self, X: np.ndarray) -> np.ndarray:
        """
        Calcule les log-probabilités d'émission P(y_t | s_t=i) pour un GMMHMM.
        
        Pour GMMHMM: P(y_t | s_t=i) = sum_mix w[i, mix] * N(y_t | mean[i, mix], cov[i, mix])
        
        Args:
            X: Features (n_samples, n_features)
        
        Returns:
            Array (n_samples, n_states) avec log P(y_t | s_t=i)
        """
        from scipy.stats import multivariate_normal
        
        n_samples, n_features = X.shape
        n_states = self.n_components
        n_mix = self.n_mix
        
        log_emissions = np.zeros((n_samples, n_states))
        
        for state in range(n_states):
            # Pour chaque mixture dans l'état
            log_mixture_probs = np.zeros((n_samples, n_mix))
            
            for mix in range(n_mix):
                mean = self.hmm.means_[state, mix, :]  # (n_features,)
                cov = self.hmm.covars_[state, mix, :]  # (n_features,) pour diag
                
                # Convertir en matrice de covariance diagonale
                cov_matrix = np.diag(cov)
                
                # Log-probabilité de la mixture
                try:
                    log_pdf = multivariate_normal.logpdf(X, mean=mean, cov=cov_matrix)
                except Exception:
                    # Fallback si problème numérique
                    log_pdf = np.full(n_samples, -np.inf)
                
                # Poids de la mixture
                if hasattr(self.hmm, 'weights_'):
                    weight = self.hmm.weights_[state, mix]
                else:
                    weight = 1.0 / n_mix  # Uniforme si pas de weights
                
                log_mixture_probs[:, mix] = np.log(weight + 1e-10) + log_pdf
            
            # Logsumexp sur les mixtures: log(sum_mix w_mix * N(...))
            log_emissions[:, state] = self._logsumexp(log_mixture_probs, axis=1)
        
        return log_emissions
    
    def _logsumexp(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """
        Calcul stable de log(sum(exp(x))) pour éviter l'overflow.
        
        Formule: logsumexp(x) = max(x) + log(sum(exp(x - max(x))))
        
        Args:
            x: Array à sommer
            axis: Axe pour la sommation
        
        Returns:
            log(sum(exp(x))) le long de l'axe spécifié
        """
        x_max = np.max(x, axis=axis, keepdims=True)
        x_shifted = x - x_max
        log_sum = np.log(np.sum(np.exp(x_shifted), axis=axis, keepdims=True) + 1e-10)
        result = x_max + log_sum
        
        # Retirer keepdims si nécessaire
        if axis == -1 and x.ndim > 1:
            result = result.squeeze(axis=axis)
        elif axis is not None:
            result = result.squeeze(axis=axis)
        
        return result

    def get_belief_states_df(
        self,
        df: pd.DataFrame,
        use_forward_only: bool = True
    ) -> pd.DataFrame:
        """
        Extrait les probabilités filtrées (Belief States) depuis un DataFrame.
        
        Cette méthode calcule les probabilités filtrées P(s_t | y_{1:t}) et les ajoute
        au DataFrame sous forme de colonnes HMM_Prob_0, HMM_Prob_1, ..., HMM_Prob_N
        et HMM_Entropy.
        
        Args:
            df: DataFrame avec les features nécessaires (BTC_LogRet, BTC_Parkinson, etc.)
            use_forward_only: Si True, utilise Forward-Only (pas de look-ahead).
                            Si False, utilise predict_proba() standard (plus rapide).
        
        Returns:
            DataFrame avec colonnes HMM_Prob_* et HMM_Entropy ajoutées.
        """
        if not self._is_fitted:
            raise RuntimeError("HMM must be fitted before extracting belief states")
        
        print(f"\n[RegimeDetector] Extracting belief states (forward_only={use_forward_only})...")
        
        # 1. Calculer les features HMM dédiées
        df_result = df.copy()
        df_result = self._compute_hmm_features(df_result)
        
        # 2. Extraire les features et identifier les lignes valides
        features_raw = df_result[self.HMM_FEATURES].values
        valid_mask = np.isfinite(features_raw).all(axis=1)
        features_valid = features_raw[valid_mask]
        
        print(f"  Valid samples: {len(features_valid)}")
        
        # 3. Scaler les features (utilise le scaler déjà fitté)
        from src.data_engineering.processor import DataProcessor
        temp_processor = DataProcessor(config={'min_iqr': 1.0, 'clip_range': (-5, 5)})
        temp_processor.scaler = self.scaler
        features_df = pd.DataFrame(features_valid, columns=self.HMM_FEATURES)
        features_scaled_df = temp_processor.transform(features_df, columns=self.HMM_FEATURES)
        features_scaled = features_scaled_df.values
        
        # 4. Calculer les belief states (probabilités filtrées)
        probs, entropy_vals = self.get_belief_states(features_scaled, use_forward_only=use_forward_only)
        
        # 5. Créer les colonnes HMM_Prob_* et HMM_Entropy
        col_names = [f'HMM_Prob_{i}' for i in range(self.n_components)]
        
        # Initialiser avec NaN
        for col in col_names + ['HMM_Entropy']:
            df_result[col] = np.nan
        
        # Remplir les valeurs valides
        valid_indices = df_result.index[valid_mask]
        for i in range(self.n_components):
            df_result.loc[valid_indices, f'HMM_Prob_{i}'] = probs[:, i]
        
        df_result.loc[valid_indices, 'HMM_Entropy'] = entropy_vals
        
        print(f"  Added columns: {', '.join(col_names)}, HMM_Entropy")
        
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
            'sorted_indices': self.sorted_indices,  # Archetype Alignment mapping
            'transition_penalty': self.transition_penalty,  # Sticky HMM param
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
            n_mix=data['n_mix'],
            transition_penalty=data.get('transition_penalty', 0.0)  # Backwards compat
        )
        detector.hmm = data['hmm']
        detector.scaler = data['scaler']
        detector.kmeans = data['kmeans']
        detector.state_stats = data['state_stats']
        detector.sorted_indices = data.get('sorted_indices')  # Archetype Alignment mapping
        detector._is_fitted = True

        print(f"[RegimeDetector] Loaded from {path}")
        if detector.sorted_indices is not None:
            print(f"  sorted_indices: {detector.sorted_indices} (Crash, Downtrend, Range, Uptrend)")
        print(f"  transition_penalty: {detector.transition_penalty}")
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
        zscore_window: int = 720,
        polygon_api_key: Optional[str] = None
    ):
        """
        Initialise le DataManager.

        Args:
            data_dir: Répertoire de données.
            ffd_window: Fenêtre pour FFD.
            vol_window: Fenêtre pour volatilité.
            zscore_window: Fenêtre pour Z-Score.
            polygon_api_key: Clé API Polygon.io pour téléchargement historique (optionnel).
        """
        self.data_dir = data_dir
        self.polygon_api_key = polygon_api_key
        os.makedirs(data_dir, exist_ok=True)

        # Composants du pipeline
        self.downloader = MultiAssetDownloader()  # Fallback Yahoo Finance
        self.historical_downloader = None  # Lazy init si besoin
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
            # Belief States HMM (probabilités filtrées Forward-Only, déjà dans [0, 1])
            'HMM_Prob_0', 'HMM_Prob_1', 'HMM_Prob_2', 'HMM_Prob_3',
            # Entropie HMM (incertitude du régime, déjà normalisée)
            'HMM_Entropy',
        ]

    def pipeline(
        self,
        save_path: Optional[str] = None,
        scaler_path: Optional[str] = None,
        use_cached_data: bool = True,
        train_end_idx: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Exécute le pipeline complet de préparation des données.

        Priorité des sources de données:
        1. raw_historical/multi_asset_historical.csv (8 ans, Polygon/Binance)
        2. Téléchargement via HistoricalDownloader si polygon_api_key fourni
        3. Fallback: processed/multi_asset.csv ou Yahoo Finance (730 jours)

        Args:
            save_path: Chemin de sauvegarde du parquet (défaut: data/processed_data.parquet).
            scaler_path: Chemin de sauvegarde du scaler (défaut: data/scaler.pkl).
            use_cached_data: Si True, utilise les données CSV existantes.
            train_end_idx: Index de fin du train set pour fit scaler on train only.
                          Si None, utilise fit_transform sur tout le dataset (legacy mode avec warning).
                          IMPORTANT: Pour éviter le data leakage, toujours spécifier ce paramètre
                          en production. Voir audit DATA_PIPELINE_AUDIT_REPORT.md P0.1.

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

        # Priorité: raw_historical (8 ans) > HistoricalDownloader > Yahoo (730j)
        historical_path = os.path.join(self.data_dir, "raw_historical/multi_asset_historical.csv")
        legacy_path = os.path.join(self.data_dir, "processed/multi_asset.csv")

        if os.path.exists(historical_path):
            # Données historiques disponibles (8 ans)
            print(f"  Using historical data: {historical_path}")
            df = pd.read_csv(historical_path, index_col=0, parse_dates=True)
        elif self.polygon_api_key:
            # Télécharger via HistoricalDownloader
            print("  Downloading historical data (Polygon/Binance)...")
            if self.historical_downloader is None:
                self.historical_downloader = HistoricalDownloader(
                    polygon_api_key=self.polygon_api_key,
                    output_dir=os.path.join(self.data_dir, "raw_historical")
                )
            df = self.historical_downloader.download_all(start_date="2017-08-01")
        elif use_cached_data and os.path.exists(legacy_path):
            # Fallback: Yahoo Finance cache (730 jours)
            print(f"  Using legacy data: {legacy_path}")
            df = pd.read_csv(legacy_path, index_col=0, parse_dates=True)
        else:
            # Télécharger via Yahoo Finance
            print("  Downloading from Yahoo Finance (730 days)...")
            df = self.downloader.download_multi_asset()

        print(f"  Shape: {df.shape}")

        # =====================================================================
        # ÉTAPE 1.5: Remove Synthetic Funding Rate (Audit P1.2)
        # =====================================================================
        # Synthetic funding rates cause spurious correlations and should not be used
        # as features. The environment uses a fixed funding_rate=0.0001 for short costs.
        # See audit DATA_PIPELINE_AUDIT_REPORT.md P1.2 for details.
        if 'Funding_Rate' in df.columns:
            print("\n[1.5/6] Removing Funding_Rate column (synthetic data - audit P1.2)...")
            print("  [INFO] Funding rate removed to avoid spurious correlations")
            print("  [INFO] Environment uses fixed funding_rate=0.0001 for short position costs")
            df = df.drop(columns=['Funding_Rate'])
            print(f"  Shape after removal: {df.shape}")

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
        # ÉTAPE 5: Global Scaling (Leak-Free) - Utilise DataProcessor unifié
        # =====================================================================
        print("\n[5/6] Applying RobustScaler (via DataProcessor)...")

        # Identifier les colonnes à scaler
        cols_to_scale = [
            col for col in df.columns
            if col not in self.exclude_from_scaling
            and df[col].dtype in ['float64', 'float32', 'int64', 'int32']
        ]

        print(f"  Scaling {len(cols_to_scale)} columns")

        # Utiliser DataProcessor unifié (configuration standardisée)
        processor = DataProcessor(config={'min_iqr': 1e-2, 'clip_range': (-5, 5)})

        # Appliquer le scaler - LEAK-FREE MODE
        if train_end_idx is not None:
            # Fit on train only, transform all (prevents data leakage)
            print(f"  [LEAK-FREE] Fitting scaler on train only (first {train_end_idx} rows)")
            processor.fit(df.iloc[:train_end_idx][cols_to_scale], columns=cols_to_scale)
            df[cols_to_scale] = processor.transform(df[cols_to_scale], columns=cols_to_scale)
        else:
            # Legacy mode - fit_transform on full dataset (DATA LEAKAGE WARNING)
            warnings.warn(
                "Scaler fit on full dataset - this causes data leakage! "
                "Use train_end_idx parameter for production. See audit P0.1.",
                UserWarning
            )
            print("  [WARNING] Legacy mode: fit_transform on full dataset (data leakage risk)")
            processor.fit(df[cols_to_scale], columns=cols_to_scale)
            df[cols_to_scale] = processor.transform(df[cols_to_scale], columns=cols_to_scale)

        # Récupérer le scaler pour sauvegarde (compatibilité)
        self.scaler = processor.get_scaler()

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

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


class RegimeDetector:
    """
    Détecteur de régimes de marché via GMM-HMM avec K-Means Warm Start.

    Approche SOTA avec Leading Indicators + Archetype Alignment:
    1. Features dédiées au HMM (6 features):
       - HMM_Trend: Moyenne glissante 168h des Log-Returns
       - HMM_Vol: Volatilité Parkinson rolling 168h
       - HMM_Momentum: RSI 14 normalisé [0, 1]
       - HMM_Funding: Funding Rate 24h (leading indicator)
       - HMM_RiskOnOff: SPX - DXY (risk-on/off signal)
       - HMM_VolRatio: Vol court/long terme (early warning)

    2. Initialisation K-Means pour garantir des clusters séparés

    3. Archetype Alignment via Hungarian Algorithm:
       - Prob_0 = Crash (-5%/h, 4% vol) - ABSOLU
       - Prob_1 = Downtrend (-0.1%/h, 1.5% vol)
       - Prob_2 = Range (0%, 0.5% vol)
       - Prob_3 = Uptrend (+0.15%/h, 2% vol)
       - Résout le problème de "Semantic Drift" entre segments WFO
    """

    # Features dédiées au HMM (calculées en interne)
    # Quick Wins: +Funding, +RiskOnOff, +VolRatio pour anticipation régimes
    HMM_FEATURES = [
        'HMM_Trend', 'HMM_Vol', 'HMM_Momentum',
        'HMM_Funding',      # Leading indicator (monte avant pumps)
        'HMM_RiskOnOff',    # SPX - DXY (risk-on/off signal)
        'HMM_VolRatio',     # Vol court/long (early warning)
    ]

    # Archétypes fixes basés sur connaissance métier BTC (hourly data)
    # Utilisés pour aligner les états HMM de manière absolue via Hungarian Algorithm
    # Résout le problème de "Semantic Drift" entre segments WFO
    STATE_ARCHETYPES = {
        0: {'name': 'Crash',     'mean_ret': -0.0050, 'mean_vol': 0.040},  # -5%/h, 4% vol
        1: {'name': 'Downtrend', 'mean_ret': -0.0010, 'mean_vol': 0.015},  # -0.1%/h, 1.5% vol
        2: {'name': 'Range',     'mean_ret':  0.0000, 'mean_vol': 0.005},  # Plat, 0.5% vol
        3: {'name': 'Uptrend',   'mean_ret':  0.0015, 'mean_vol': 0.020},  # +0.15%/h, 2% vol
    }

    # Fenêtre de lissage (1 semaine en heures)
    SMOOTHING_WINDOW = 168

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
            df: DataFrame avec BTC_LogRet, BTC_Parkinson, BTC_Fracdiff.

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

        # 3. Momentum_Feature: RSI 14 sur BTC_LogRet (centré autour de 0)
        # Note: BTC_Fracdiff après log() est toujours positif, donc RSI ne fonctionne pas
        # On utilise BTC_LogRet qui oscille naturellement autour de 0
        logret = df['BTC_LogRet']
        gain = logret.where(logret > 0, 0).rolling(window=14, min_periods=14).mean()
        loss = (-logret.where(logret < 0, 0)).rolling(window=14, min_periods=14).mean()

        # RSI borne naturellement [0, 100] → idéal pour Gaussian HMM
        # Ajouter epsilon pour éviter division par zéro
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        df_result['HMM_Momentum'] = rsi / 100  # Normaliser [0, 1]

        # 4. Funding Rate (leading indicator - monte avant pumps)
        # Smooth sur 24h car c'est déjà un indicateur synthétique
        if 'Funding_Rate' in df.columns:
            df_result['HMM_Funding'] = df['Funding_Rate'].rolling(
                window=24, min_periods=24
            ).mean()
        else:
            df_result['HMM_Funding'] = 0.0

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

        # Clip HMM features pour stabilité numérique
        df_result['HMM_Trend'] = df_result['HMM_Trend'].clip(-0.05, 0.05)  # ±5%/h max
        df_result['HMM_Vol'] = df_result['HMM_Vol'].clip(0, 0.2)  # Vol max 20%/h
        df_result['HMM_Momentum'] = df_result['HMM_Momentum'].clip(0, 1)  # RSI strict [0,1]
        df_result['HMM_Funding'] = df_result['HMM_Funding'].clip(-0.005, 0.005)  # ±0.5%
        df_result['HMM_RiskOnOff'] = df_result['HMM_RiskOnOff'].clip(-0.02, 0.02)  # ±2%
        df_result['HMM_VolRatio'] = df_result['HMM_VolRatio'].clip(0.2, 5.0)  # Ratio [0.2x, 5x]

        print(f"  Computed HMM features (window={self.SMOOTHING_WINDOW}h, 6 features: Trend/Vol/Mom/Funding/RiskOnOff/VolRatio)")

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

        # 6. Hungarian Algorithm pour appariement optimal
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
        4. Archetype Alignment: aligner états sur archétypes fixes via Hungarian Algorithm
        5. Création des colonnes Prob_0 (Crash), Prob_1 (Downtrend), Prob_2 (Range), Prob_3 (Uptrend)

        Args:
            df: DataFrame avec BTC_LogRet, BTC_Parkinson, BTC_Fracdiff.
            tensorboard_log: Chemin pour logs TensorBoard (optional).
            run_name: Nom du run TensorBoard (optional).
            segment_id: ID du segment WFO pour TensorBoard step (default 0).

        Returns:
            DataFrame avec colonnes HMM et Prob_* ajoutées.
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

        # 3. Scaler les features
        features_scaled = self.scaler.fit_transform(features_valid)

        # Clip scaled features to [-5, 5] for numerical stability
        features_scaled = np.clip(features_scaled, -5, 5)

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

            # Injecter les centres K-Means
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

        # 7. Archetype-Based Alignment (remplace Smart Sorting)
        #    Utilise Hungarian Algorithm pour aligner les états sur des archétypes fixes
        #    Résout le problème de "Semantic Drift" entre segments WFO
        #    Prob_0 = Crash, Prob_1 = Downtrend, Prob_2 = Range, Prob_3 = Uptrend (ABSOLU)
        self.sorted_indices = self.align_to_archetypes()

        # Calculer les mean_returns réels pour TensorBoard (info seulement)
        state_returns = []
        dominant = proba.argmax(axis=1)
        for i in range(self.n_components):
            hmm_state = self.sorted_indices[i]
            state_mask = dominant == hmm_state
            if state_mask.sum() > 0:
                mean_ret = real_log_returns[state_mask].mean()
            else:
                mean_ret = 0.0
            state_returns.append((hmm_state, mean_ret))

        print(f"  Final mapping: Prob_i -> HMM_State {self.sorted_indices.tolist()}")

        # 9. Créer les colonnes Prob_0, Prob_1, ..., Prob_N (triées)
        col_names = [f'Prob_{i}' for i in range(self.n_components)]

        for col in col_names:
            df_result[col] = np.nan

        valid_indices = df_result.index[valid_mask]
        for i in range(self.n_components):
            original_state = self.sorted_indices[i]
            df_result.loc[valid_indices, f'Prob_{i}'] = proba[:, original_state]

        print(f"  Added columns: {', '.join(col_names)}")

        # Log final state distributions to TensorBoard
        if writer:
            dominant = proba.argmax(axis=1)
            for i, (state, mean_ret) in enumerate(state_returns):
                annual_ret = mean_ret * 24 * 365 * 100
                state_pct = (dominant == self.sorted_indices[i]).sum() / len(dominant) * 100
                writer.add_scalar(f"hmm/state_{i}/annual_return_pct", annual_ret, segment_id)
                writer.add_scalar(f"hmm/state_{i}/distribution_pct", state_pct, segment_id)

            writer.close()

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
        valid_mask = np.isfinite(features_raw).all(axis=1)  # Attrape NaN ET Inf
        features_valid = features_raw[valid_mask]

        print(f"  Valid samples: {len(features_valid)}")

        # 3. Scaler les features (utilise le scaler déjà fitté)
        features_scaled = self.scaler.transform(features_valid)

        # Clip scaled features to [-5, 5] for numerical stability
        features_scaled = np.clip(features_scaled, -5, 5)

        # 4. Prédire les probabilités brutes
        try:
            proba = self.hmm.predict_proba(features_scaled)
        except ValueError as e:
            print(f"  [WARNING] HMM predict_proba failed: {e}")
            print(f"  [FALLBACK] Using uniform probabilities")
            proba = np.ones((len(features_scaled), self.n_components)) / self.n_components

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
        ]

    def pipeline(
        self,
        save_path: Optional[str] = None,
        scaler_path: Optional[str] = None,
        use_cached_data: bool = True
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

"""
features.py - Feature Engineering avancé pour Trading RL.

Classe FeatureEngineer pour générer des features d'entrée pour le Transformer:
- Fractional Differentiation (FFD) - Résout le dilemme Stationnarité vs Mémoire
- Parkinson & Garman-Klass Volatility - Volatilité SOTA basée sur OHLC
- Rolling Z-Score - Normalisation pour comparaison cross-asset
- Log-Returns - Rendements logarithmiques classiques

Référence: Lopez de Prado (2018) - Advances in Financial Machine Learning
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from statsmodels.tsa.stattools import adfuller


class FeatureEngineer:
    """
    Feature engineering avancé pour séries temporelles financières.

    Implémente la Différenciation Fractionnaire (FFD) pour obtenir des séries
    stationnaires tout en préservant un maximum de mémoire historique.
    """

    # Actifs pour lesquels calculer les features
    ASSETS = ['BTC', 'ETH', 'SPX', 'DXY', 'NASDAQ']

    # Actifs pour Fracdiff (Close uniquement)
    FRACDIFF_ASSETS = ['BTC', 'ETH', 'SPX', 'DXY', 'NASDAQ']

    # Actifs pour Volume relatif
    VOLUME_ASSETS = ['BTC', 'ETH', 'SPX', 'DXY', 'NASDAQ']

    def __init__(
        self,
        ffd_window: int = 100,
        d_range: Tuple[float, float, float] = (0.0, 1.0, 0.05),
        vol_window: int = 24,
        zscore_window: int = 720
    ):
        """
        Initialise le FeatureEngineer.

        Args:
            ffd_window: Taille de la fenêtre FFD (Fixed-Width Window).
            d_range: Tuple (start, stop, step) pour la recherche du d optimal.
            vol_window: Fenêtre pour les indicateurs de volatilité (heures).
            zscore_window: Fenêtre pour le Z-Score (720h = 30 jours).
        """
        self.ffd_window = ffd_window
        self.d_range = d_range
        self.vol_window = vol_window
        self.zscore_window = zscore_window

        # Stocke les valeurs d optimales trouvées
        self.optimal_d: Dict[str, float] = {}
        self.adf_results: Dict[str, Dict] = {}

    # =========================================================================
    # FRACTIONAL DIFFERENTIATION (FFD)
    # =========================================================================

    # =========================================================================
    # DATA SANITIZATION
    # =========================================================================

    def _sanitize_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Replace 0 prices with NaN and forward-fill.

        Prevents division by zero in log-return calculations that cause
        extreme values like -35 (which crashed the RL training).

        Args:
            df: DataFrame with price columns.

        Returns:
            DataFrame with sanitized prices.
        """
        print("\n[Sanitize] Cleaning price data (0 -> NaN -> ffill)...")

        price_cols = []
        for asset in self.ASSETS:
            price_cols.extend([
                f"{asset}_Close", f"{asset}_Open",
                f"{asset}_High", f"{asset}_Low"
            ])

        zeros_replaced = 0
        for col in price_cols:
            if col in df.columns:
                # Count zeros before replacing
                n_zeros = (df[col] == 0).sum()
                zeros_replaced += n_zeros

                # Replace 0 with NaN
                df[col] = df[col].replace(0, np.nan)

                # Forward fill to maintain continuity
                df[col] = df[col].ffill()

                # Backward fill for any remaining NaN at the start
                df[col] = df[col].bfill()

        print(f"  Replaced {zeros_replaced} zero values across price columns")

        return df

    def _validate_features(self, df: pd.DataFrame) -> None:
        """
        Check for extreme values that indicate data corruption.

        Prints warnings for any feature with |value| > 10.

        Args:
            df: DataFrame to validate.
        """
        print("\n[Validate] Checking for extreme values...")

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        warnings_count = 0

        for col in numeric_cols:
            max_val = df[col].max()
            min_val = df[col].min()

            if abs(max_val) > 10 or abs(min_val) > 10:
                print(f"  [WARNING] {col}: min={min_val:.2f}, max={max_val:.2f}")
                warnings_count += 1

        if warnings_count == 0:
            print("  All features within normal range [-10, 10]")
        else:
            print(f"  Found {warnings_count} features with extreme values!")

    # =========================================================================
    # FRACTIONAL DIFFERENTIATION (FFD)
    # =========================================================================

    def _get_weights_ffd(self, d: float, threshold: float = 1e-5) -> np.ndarray:
        """
        Calcule les poids pour la Différenciation Fractionnaire (FFD).

        Les poids suivent la formule récursive:
            w_k = -w_{k-1} * (d - k + 1) / k

        Args:
            d: Ordre de différenciation (0 < d < 1).
            threshold: Seuil pour tronquer les poids négligeables.

        Returns:
            Array numpy des poids FFD.
        """
        weights = [1.0]
        k = 1

        while True:
            w_k = -weights[-1] * (d - k + 1) / k
            if abs(w_k) < threshold:
                break
            weights.append(w_k)
            k += 1

            # Limite de sécurité pour éviter boucle infinie
            if k > self.ffd_window:
                break

        return np.array(weights[::-1])  # Inverser pour convolution

    def _ffd(self, series: pd.Series, d: float) -> pd.Series:
        """
        Applique la Différenciation Fractionnaire (Fixed-Width Window).

        Utilise une fenêtre fixe pour éviter le look-ahead bias.

        Args:
            series: Série temporelle à différencier.
            d: Ordre de différenciation.

        Returns:
            Série différenciée fractionnellement.
        """
        weights = self._get_weights_ffd(d)
        width = len(weights)

        # Appliquer la convolution (dot product avec fenêtre glissante)
        result = pd.Series(index=series.index, dtype=np.float64)

        for i in range(width - 1, len(series)):
            window = series.iloc[i - width + 1:i + 1].values
            if len(window) == width:
                result.iloc[i] = np.dot(weights, window)

        return result

    def find_min_d(
        self,
        series: pd.Series,
        pvalue_threshold: float = 0.05,
        min_d_floor: float = 0.30
    ) -> Tuple[float, float]:
        """
        Trouve le plus petit d qui rend la série stationnaire.

        Itère sur d à partir de min_d_floor et teste la stationnarité via ADF.
        L'objectif est de préserver un maximum de mémoire historique tout en
        garantissant une transformation minimale pour éviter les faux positifs.

        Args:
            series: Série temporelle à analyser.
            pvalue_threshold: Seuil de p-value pour ADF (défaut: 0.05).
            min_d_floor: Valeur minimale de d pour éviter les faux positifs ADF (défaut: 0.30).

        Returns:
            Tuple (d_optimal, p_value).
        """
        d_start, d_stop, d_step = self.d_range

        # Commencer au maximum entre d_start et min_d_floor
        effective_start = max(d_start, min_d_floor)

        for d in np.arange(effective_start, d_stop + d_step, d_step):
            diff_series = self._ffd(series, d).dropna()

            if len(diff_series) < 100:
                continue

            try:
                adf_result = adfuller(diff_series, maxlag=1, regression='c')
                p_value = adf_result[1]

                if p_value < pvalue_threshold:
                    return round(d, 2), p_value

            except Exception:
                continue

        # Si aucun d trouvé, retourner d=1.0 (différenciation complète)
        return 1.0, 0.0

    def add_fracdiff(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ajoute les colonnes de Différenciation Fractionnaire.

        Pour chaque actif, trouve le d optimal et applique FFD au Close.

        Args:
            df: DataFrame avec colonnes {ASSET}_Close.

        Returns:
            DataFrame avec colonnes {ASSET}_Fracdiff ajoutées.
        """
        print("\n[FFD] Calculating Fractional Differentiation...")

        for asset in self.FRACDIFF_ASSETS:
            close_col = f"{asset}_Close"

            if close_col not in df.columns:
                print(f"[WARNING] {close_col} not found, skipping.")
                continue

            series = df[close_col].dropna()

            # Trouver le d optimal
            d_opt, p_value = self.find_min_d(series)
            self.optimal_d[asset] = d_opt
            self.adf_results[asset] = {'d': d_opt, 'p_value': p_value}

            print(f"  {asset}: d_optimal = {d_opt:.2f}, ADF p-value = {p_value:.4f}")

            # Appliquer FFD avec le d optimal
            df[f"{asset}_Fracdiff"] = self._ffd(df[close_col], d_opt)

        return df

    # =========================================================================
    # VOLATILITÉ SOTA
    # =========================================================================

    def add_parkinson_volatility(
        self,
        df: pd.DataFrame,
        window: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Ajoute la volatilité de Parkinson (High-Low based).

        Formule: σ_P = sqrt(1 / (4 * ln(2)) * (ln(High/Low))^2)

        Plus précise que la std dev du Close car utilise l'information intraday.

        Args:
            df: DataFrame avec colonnes {ASSET}_High et {ASSET}_Low.
            window: Fenêtre de rolling (défaut: self.vol_window).

        Returns:
            DataFrame avec colonnes {ASSET}_Parkinson ajoutées.
        """
        window = window or self.vol_window
        print(f"\n[Parkinson] Calculating volatility (window={window}h)...")

        for asset in self.ASSETS:
            high_col = f"{asset}_High"
            low_col = f"{asset}_Low"

            if high_col not in df.columns or low_col not in df.columns:
                continue

            # Parkinson volatility formula
            log_hl = np.log(df[high_col] / df[low_col])
            parkinson = np.sqrt((1 / (4 * np.log(2))) * (log_hl ** 2))

            # Rolling mean pour lisser
            df[f"{asset}_Parkinson"] = parkinson.rolling(window=window).mean()

        return df

    def add_garman_klass_volatility(
        self,
        df: pd.DataFrame,
        window: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Ajoute la volatilité de Garman-Klass (OHLC-based).

        Formule: σ_GK = sqrt(0.5 * (ln(H/L))^2 - (2*ln(2) - 1) * (ln(C/O))^2)

        Plus précise que Parkinson car utilise aussi Open et Close.

        Args:
            df: DataFrame avec colonnes {ASSET}_Open/High/Low/Close.
            window: Fenêtre de rolling (défaut: self.vol_window).

        Returns:
            DataFrame avec colonnes {ASSET}_GK ajoutées.
        """
        window = window or self.vol_window
        print(f"\n[Garman-Klass] Calculating volatility (window={window}h)...")

        for asset in self.ASSETS:
            o_col = f"{asset}_Open"
            h_col = f"{asset}_High"
            l_col = f"{asset}_Low"
            c_col = f"{asset}_Close"

            if not all(col in df.columns for col in [o_col, h_col, l_col, c_col]):
                continue

            # Garman-Klass formula
            log_hl = np.log(df[h_col] / df[l_col])
            log_co = np.log(df[c_col] / df[o_col])

            gk = np.sqrt(0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2))

            # Gérer les valeurs négatives sous la racine (remplacer par NaN)
            gk = gk.replace([np.inf, -np.inf], np.nan)

            # Rolling mean pour lisser
            df[f"{asset}_GK"] = gk.rolling(window=window).mean()

        return df

    def add_rolling_zscore(
        self,
        df: pd.DataFrame,
        window: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Ajoute le Z-Score rolling pour chaque actif.

        Formule: Z = (Price - RollingMean) / RollingStd

        Crucial pour que le Transformer compare les variations relatives
        entre actifs à faible volatilité (SPX) et haute volatilité (BTC).

        Args:
            df: DataFrame avec colonnes {ASSET}_Close.
            window: Fenêtre de rolling (défaut: 720h = 30 jours).

        Returns:
            DataFrame avec colonnes {ASSET}_ZScore ajoutées.
        """
        window = window or self.zscore_window
        print(f"\n[Z-Score] Calculating rolling Z-Score (window={window}h)...")

        for asset in self.ASSETS:
            close_col = f"{asset}_Close"

            if close_col not in df.columns:
                continue

            rolling_mean = df[close_col].rolling(window=window).mean()
            rolling_std = df[close_col].rolling(window=window).std()

            # Éviter division par zéro
            df[f"{asset}_ZScore"] = (df[close_col] - rolling_mean) / (rolling_std + 1e-8)

        return df

    # =========================================================================
    # LOG-RETURNS
    # =========================================================================

    def add_log_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ajoute les rendements logarithmiques (1h).

        Formule: r = ln(Close_t / Close_{t-1})

        Args:
            df: DataFrame avec colonnes {ASSET}_Close.

        Returns:
            DataFrame avec colonnes {ASSET}_LogRet ajoutées.
        """
        print("\n[Log-Returns] Calculating 1h log returns...")

        for asset in self.ASSETS:
            close_col = f"{asset}_Close"

            if close_col not in df.columns:
                continue

            log_ret = np.log(df[close_col] / df[close_col].shift(1))

            # Remplacer inf par 0
            log_ret = log_ret.replace([np.inf, -np.inf], 0)

            # Hard clip: +/- 20% max per hour (prevents data corruption explosions)
            # A 20% hourly move is already extreme; anything beyond is data error
            log_ret = np.clip(log_ret, -0.20, 0.20)

            df[f"{asset}_LogRet"] = log_ret

        return df

    # =========================================================================
    # VOLUME FEATURES
    # =========================================================================

    def add_volume_features(
        self,
        df: pd.DataFrame,
        zscore_window: int = 336  # 14 jours = 336 heures
    ) -> pd.DataFrame:
        """
        Ajoute les features de volume pour chaque actif.

        Features créées:
        - {ASSET}_Vol_LogRet: Log-return du volume (momentum)
        - {ASSET}_Vol_ZScore: Z-Score glissant 14j (détection d'anomalies)

        Gestion des volumes manquants (DXY, SPX):
        - Si pas de volume valide, force les deux features à 0.0
        - Cela agit comme "zero padding" pour le réseau de neurones

        Args:
            df: DataFrame avec colonnes {ASSET}_Volume.
            zscore_window: Fenêtre pour Z-Score (défaut: 336h = 14 jours).

        Returns:
            DataFrame avec colonnes Vol_LogRet et Vol_ZScore ajoutées.
        """
        print(f"\n[Volume] Calculating Vol_LogRet and Vol_ZScore (window={zscore_window}h)...")

        for asset in self.VOLUME_ASSETS:
            vol_col = f"{asset}_Volume"

            if vol_col not in df.columns:
                print(f"  [WARNING] {vol_col} not found, setting to 0.")
                df[f"{asset}_Vol_LogRet"] = 0.0
                df[f"{asset}_Vol_ZScore"] = 0.0
                continue

            volume = df[vol_col].copy()

            # Vérifier si le volume est valide (non-zéro)
            has_valid_volume = volume.sum() > 0 and not volume.isna().all()

            if not has_valid_volume:
                # Pas de volume valide -> Zero Padding
                print(f"  {asset}: No valid volume -> Zero Padding")
                df[f"{asset}_Vol_LogRet"] = 0.0
                df[f"{asset}_Vol_ZScore"] = 0.0
                continue

            # ===== Vol_LogRet (Momentum) =====
            # Remplacer 0 par 1 avant le log pour éviter -inf
            volume_safe = volume.replace(0, 1)
            vol_logret = np.log(volume_safe / volume_safe.shift(1))
            vol_logret = vol_logret.replace([np.inf, -np.inf], 0.0)
            vol_logret = vol_logret.fillna(0.0)
            df[f"{asset}_Vol_LogRet"] = vol_logret

            # ===== Vol_ZScore (Anomalie) =====
            rolling_mean = volume.rolling(window=zscore_window, min_periods=1).mean()
            rolling_std = volume.rolling(window=zscore_window, min_periods=1).std()

            # Éviter division par zéro
            vol_zscore = (volume - rolling_mean) / (rolling_std + 1e-8)
            vol_zscore = vol_zscore.replace([np.inf, -np.inf], 0.0)
            vol_zscore = vol_zscore.fillna(0.0)
            df[f"{asset}_Vol_ZScore"] = vol_zscore

            print(f"  {asset}: Vol_LogRet and Vol_ZScore computed")

        # Vérification de contrôle
        print("\n  [VERIFICATION]")
        if 'BTC_Vol_ZScore' in df.columns:
            btc_sample = df['BTC_Vol_ZScore'].iloc[400:405].values
            print(f"    BTC_Vol_ZScore (sample): {btc_sample}")
        if 'DXY_Vol_ZScore' in df.columns:
            dxy_sample = df['DXY_Vol_ZScore'].iloc[400:405].values
            print(f"    DXY_Vol_ZScore (sample): {dxy_sample}")

        return df

    # =========================================================================
    # PIPELINE
    # =========================================================================

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Supprime les lignes avec NaN (début du dataset).

        Les calculs de rolling windows et FFD créent des NaN au début.

        Args:
            df: DataFrame à nettoyer.

        Returns:
            DataFrame sans NaN.
        """
        initial_rows = len(df)
        df = df.dropna()
        dropped = initial_rows - len(df)

        print(f"\n[Clean] Dropped {dropped} rows with NaN ({len(df)} remaining)")

        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pipeline complet de feature engineering.

        Ordre d'exécution:
        0. Sanitize prices (0 -> NaN -> ffill)
        1. Log-Returns (with hard clip +/- 20%)
        2. Volume Relatif (log)
        3. Parkinson Volatility
        4. Garman-Klass Volatility
        5. Rolling Z-Score
        6. Fractional Differentiation (FFD)
        7. Clean (drop NaN)
        8. Validate (check for extreme values)

        Args:
            df: DataFrame multi-actifs brut.

        Returns:
            DataFrame enrichi avec toutes les features.
        """
        print("=" * 60)
        print("FEATURE ENGINEERING - Starting...")
        print("=" * 60)

        # 0. Sanitize prices (prevents log(0) = -inf explosions)
        df = self._sanitize_prices(df)

        # 1. Log-Returns (with hard clip +/- 20%)
        df = self.add_log_returns(df)

        # 2. Volume Relatif (log)
        df = self.add_volume_features(df)

        # 3. Parkinson Volatility
        df = self.add_parkinson_volatility(df)

        # 4. Garman-Klass Volatility
        df = self.add_garman_klass_volatility(df)

        # 5. Rolling Z-Score
        df = self.add_rolling_zscore(df)

        # 6. Fractional Differentiation (le plus coûteux)
        df = self.add_fracdiff(df)

        # 7. Clean NaN
        df = self.clean(df)

        # 8. Validate features (check for extreme values)
        self._validate_features(df)

        print("\n" + "=" * 60)
        print("FEATURE ENGINEERING - Complete!")
        print(f"Final shape: {df.shape[0]} rows, {df.shape[1]} columns")
        print("=" * 60)

        return df

    def get_stationarity_report(self) -> str:
        """
        Génère un rapport textuel des résultats de stationnarité.

        Returns:
            String formaté du rapport.
        """
        report = "\n" + "=" * 40 + "\n"
        report += "STATIONARITY REPORT\n"
        report += "=" * 40 + "\n"

        for asset, results in self.adf_results.items():
            d = results['d']
            p = results['p_value']
            status = "Stationary" if p < 0.05 else "Non-Stationary"
            report += f"{asset}: d={d:.2f}, p-value={p:.4f} ({status})\n"

        report += "=" * 40 + "\n"

        return report


if __name__ == "__main__":
    # Test rapide
    print("Loading data...")
    df = pd.read_csv("data/processed/multi_asset.csv", index_col=0, parse_dates=True)

    print(f"Input shape: {df.shape}")

    engineer = FeatureEngineer()
    df_features = engineer.engineer_features(df)

    print("\n--- New Columns ---")
    new_cols = [c for c in df_features.columns if c not in df.columns]
    print(new_cols)

    print(engineer.get_stationarity_report())

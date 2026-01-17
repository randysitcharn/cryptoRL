# -*- coding: utf-8 -*-
"""
analyze_tensorboard.py - Analyse statistique des logs TensorBoard.

Analyse de manière statistique les logs TensorBoard d'un run d'entraînement,
en calculant des métriques descriptives, des tendances, et en détectant des patterns/anomalies.

Usage:
    python scripts/analyze_tensorboard.py --log_dir logs/tensorboard/run_1
    python scripts/analyze_tensorboard.py --log_dir logs/wfo/segment_0 --output report.txt
"""

import os
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict
from tensorboard.backend.event_processing import event_accumulator

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[WARNING] scipy non disponible, skewness/kurtosis seront ignorés")

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False
    print("[WARNING] tabulate non disponible, utilisation du formatage manuel")


# ============================================================================
# Chargement des données TensorBoard
# ============================================================================

def find_all_event_files(log_dir: str) -> List[str]:
    """Trouve tous les fichiers event TensorBoard dans un dossier."""
    event_files = []

    for root, dirs, files in os.walk(log_dir):
        for f in files:
            if f.startswith('events.out.tfevents'):
                full_path = os.path.join(root, f)
                event_files.append((full_path, os.path.getmtime(full_path)))

    if not event_files:
        raise FileNotFoundError(f"Aucun fichier TensorBoard trouvé dans {log_dir}")

    # Tri par date (plus récent en premier)
    event_files.sort(key=lambda x: x[1], reverse=True)
    return [f[0] for f in event_files]


def load_tensorboard_run(log_dir: str, run_name: Optional[str] = None) -> Dict[str, pd.Series]:
    """
    Charge toutes les métriques scalaires d'un run TensorBoard.

    Args:
        log_dir: Dossier de base contenant les logs TensorBoard
        run_name: Nom du run spécifique (optionnel). Si None, prend le plus récent.

    Returns:
        Dictionnaire {metric_name: pd.Series} où chaque Series a un index 'step' et des valeurs
    """
    # Si run_name est spécifié, utiliser ce dossier
    if run_name:
        target_dir = os.path.join(log_dir, run_name)
        if not os.path.exists(target_dir):
            raise FileNotFoundError(f"Run '{run_name}' non trouvé dans {log_dir}")
    else:
        # Chercher tous les sous-dossiers et prendre le plus récent
        if os.path.isdir(log_dir):
            subdirs = [d for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))]
            if subdirs:
                # Trouver le plus récent en regardant les fichiers event
                most_recent = None
                most_recent_time = 0
                for subdir in subdirs:
                    subdir_path = os.path.join(log_dir, subdir)
                    event_files = find_all_event_files(subdir_path)
                    if event_files:
                        mtime = os.path.getmtime(event_files[0])
                        if mtime > most_recent_time:
                            most_recent_time = mtime
                            most_recent = subdir
                if most_recent:
                    target_dir = os.path.join(log_dir, most_recent)
                    print(f"[INFO] Utilisation du run le plus récent: {most_recent}")
                else:
                    target_dir = log_dir
            else:
                target_dir = log_dir
        else:
            target_dir = log_dir

    try:
        event_files = find_all_event_files(target_dir)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Aucun fichier TensorBoard trouvé dans {target_dir}\n"
            f"Vérifiez que le dossier contient des fichiers 'events.out.tfevents.*'"
        ) from e

    print(f"[INFO] {len(event_files)} fichier(s) TensorBoard trouvé(s)")

    # Charger toutes les métriques scalaires
    all_metrics = defaultdict(lambda: {'steps': [], 'values': []})
    all_tags = set()

    for event_file in event_files:
        ea = event_accumulator.EventAccumulator(event_file)
        ea.Reload()

        available_tags = ea.Tags().get('scalars', [])
        all_tags.update(available_tags)

        for tag in available_tags:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]

            all_metrics[tag]['steps'].extend(steps)
            all_metrics[tag]['values'].extend(values)

    print(f"[INFO] {len(all_tags)} métrique(s) trouvée(s): {sorted(all_tags)}")

    # Convertir en Series pandas
    result = {}
    for metric_name, data in all_metrics.items():
        if not data['steps']:
            continue

        # Trier et dédupliquer (prendre la dernière valeur pour chaque step)
        step_value = {}
        for s, v in zip(data['steps'], data['values']):
            step_value[s] = v

        sorted_steps = sorted(step_value.keys())
        sorted_values = [step_value[s] for s in sorted_steps]

        # Créer une Series avec step comme index
        series = pd.Series(sorted_values, index=sorted_steps, name=metric_name)
        result[metric_name] = series

        print(f"[OK] {metric_name}: {len(series)} points (steps {series.index.min()} -> {series.index.max()})")

    if not result:
        raise ValueError(
            "Aucune métrique trouvée dans les logs TensorBoard.\n"
            "Les fichiers event ont été trouvés mais ne contiennent pas de métriques scalaires."
        )

    return result


# ============================================================================
# Statistiques descriptives
# ============================================================================

def compute_descriptive_stats(metric_series: pd.Series) -> Dict[str, float]:
    """
    Calcule les statistiques descriptives d'une série de métriques.

    Args:
        metric_series: pd.Series avec index=step et values=metric values

    Returns:
        Dictionnaire avec toutes les statistiques
    """
    if len(metric_series) == 0:
        return {}

    values = metric_series.values
    steps = metric_series.index.values

    stats_dict = {
        'count': len(metric_series),
        'min_step': int(steps.min()),
        'max_step': int(steps.max()),
        'step_range': int(steps.max() - steps.min()),
        'mean': float(np.mean(values)),
        'median': float(np.median(values)),
        'std': float(np.std(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'q25': float(np.percentile(values, 25)),
        'q50': float(np.percentile(values, 50)),
        'q75': float(np.percentile(values, 75)),
        'q90': float(np.percentile(values, 90)),
        'q95': float(np.percentile(values, 95)),
        'q99': float(np.percentile(values, 99)),
    }

    # Coefficient de variation
    if stats_dict['mean'] != 0:
        stats_dict['cv'] = stats_dict['std'] / abs(stats_dict['mean'])
    else:
        stats_dict['cv'] = float('inf') if stats_dict['std'] > 0 else 0.0

    # Skewness et kurtosis (si scipy disponible)
    if HAS_SCIPY and len(values) > 2:
        stats_dict['skewness'] = float(stats.skew(values))
        stats_dict['kurtosis'] = float(stats.kurtosis(values))
    else:
        stats_dict['skewness'] = None
        stats_dict['kurtosis'] = None

    return stats_dict


# ============================================================================
# Analyse de tendances
# ============================================================================

def analyze_trends(metric_series: pd.Series) -> Dict[str, Any]:
    """
    Analyse les tendances d'une série de métriques.

    Args:
        metric_series: pd.Series avec index=step et values=metric values

    Returns:
        Dictionnaire avec analyse de tendance
    """
    if len(metric_series) < 2:
        return {}

    steps = metric_series.index.values.astype(float)
    values = metric_series.values.astype(float)

    # Régression linéaire: y = mx + b
    n = len(steps)
    sum_x = np.sum(steps)
    sum_y = np.sum(values)
    sum_xy = np.sum(steps * values)
    sum_x2 = np.sum(steps ** 2)
    sum_y2 = np.sum(values ** 2)

    denominator = n * sum_x2 - sum_x ** 2
    if denominator == 0:
        slope = 0.0
        intercept = np.mean(values)
        r2 = 0.0
    else:
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        intercept = (sum_y - slope * sum_x) / n

        # R²
        y_pred = slope * steps + intercept
        ss_res = np.sum((values - y_pred) ** 2)
        ss_tot = np.sum((values - np.mean(values)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Déterminer la tendance
    if abs(slope) < 1e-10:
        trend = 'stable'
    elif slope > 0:
        trend = 'croissante'
    else:
        trend = 'décroissante'

    # Moyennes mobiles sur fenêtres (début, milieu, fin)
    n_points = len(metric_series)
    window_size = max(1, n_points // 10)  # 10% de la série

    if n_points >= window_size:
        start_window = metric_series.iloc[:window_size].mean()
        end_window = metric_series.iloc[-window_size:].mean()
        mid_start = n_points // 2 - window_size // 2
        mid_end = mid_start + window_size
        mid_window = metric_series.iloc[mid_start:mid_end].mean() if mid_end <= n_points else None
    else:
        start_window = metric_series.mean()
        end_window = metric_series.mean()
        mid_window = None

    # Détection de plateaux (variation < seuil sur N points)
    plateau_detected = False
    plateau_threshold = np.std(values) * 0.1  # 10% de l'écart-type
    plateau_window = max(1, n_points // 20)  # 5% de la série

    if n_points >= plateau_window:
        for i in range(len(values) - plateau_window + 1):
            window_values = values[i:i + plateau_window]
            if np.std(window_values) < plateau_threshold:
                plateau_detected = True
                break

    return {
        'slope': float(slope),
        'intercept': float(intercept),
        'r2': float(r2),
        'trend': trend,
        'start_mean': float(start_window),
        'end_mean': float(end_window),
        'mid_mean': float(mid_window) if mid_window is not None else None,
        'plateau_detected': plateau_detected,
    }


# ============================================================================
# Détection d'anomalies
# ============================================================================

def detect_anomalies(metric_series: pd.Series, iqr_factor: float = 1.5, std_threshold: float = 3.0) -> Dict[str, Any]:
    """
    Détecte les anomalies dans une série de métriques.

    Args:
        metric_series: pd.Series avec index=step et values=metric values
        iqr_factor: Facteur IQR pour détection d'outliers (défaut: 1.5)
        std_threshold: Seuil en écarts-types pour discontinuités (défaut: 3.0)

    Returns:
        Dictionnaire avec anomalies détectées
    """
    if len(metric_series) < 3:
        return {'outliers': [], 'discontinuities': [], 'spikes': []}

    values = metric_series.values
    steps = metric_series.index.values

    # Outliers via IQR
    q25 = np.percentile(values, 25)
    q75 = np.percentile(values, 75)
    iqr = q75 - q25
    lower_bound = q25 - iqr_factor * iqr
    upper_bound = q75 + iqr_factor * iqr

    outliers = []
    for step, value in zip(steps, values):
        if value < lower_bound or value > upper_bound:
            outliers.append({'step': int(step), 'value': float(value)})

    # Discontinuités (sauts > N écarts-types)
    std = np.std(values)
    mean = np.mean(values)
    discontinuities = []

    for i in range(1, len(values)):
        diff = abs(values[i] - values[i-1])
        if diff > std_threshold * std:
            discontinuities.append({
                'step': int(steps[i]),
                'value': float(values[i]),
                'prev_value': float(values[i-1]),
                'jump': float(diff),
            })

    # Détection de spikes/creux anormaux (variation locale > 2 std)
    spikes = []
    if len(values) >= 3:
        local_std = np.std(np.diff(values))
        for i in range(1, len(values) - 1):
            # Variation par rapport aux voisins
            local_change = abs(values[i] - (values[i-1] + values[i+1]) / 2)
            if local_change > 2 * local_std:
                spikes.append({
                    'step': int(steps[i]),
                    'value': float(values[i]),
                    'local_change': float(local_change),
                })

    return {
        'outliers': outliers,
        'discontinuities': discontinuities,
        'spikes': spikes,
        'n_outliers': len(outliers),
        'n_discontinuities': len(discontinuities),
        'n_spikes': len(spikes),
    }


# ============================================================================
# Analyse par phases
# ============================================================================

def analyze_phases(metric_series: pd.Series, n_phases: int = 3) -> Dict[str, Dict]:
    """
    Analyse une série en la divisant en phases et compare les statistiques.

    Args:
        metric_series: pd.Series avec index=step et values=metric values
        n_phases: Nombre de phases (défaut: 3)

    Returns:
        Dictionnaire avec statistiques par phase
    """
    if len(metric_series) < n_phases:
        return {}

    n_points = len(metric_series)
    phase_size = n_points // n_phases

    phases = {}
    phase_names = ['début', 'milieu', 'fin'] if n_phases == 3 else [f'phase_{i+1}' for i in range(n_phases)]

    for i in range(n_phases):
        start_idx = i * phase_size
        if i == n_phases - 1:
            end_idx = n_points  # Dernière phase prend le reste
        else:
            end_idx = (i + 1) * phase_size

        phase_series = metric_series.iloc[start_idx:end_idx]
        phase_stats = compute_descriptive_stats(phase_series)

        phases[phase_names[i]] = {
            'start_step': int(phase_series.index.min()),
            'end_step': int(phase_series.index.max()),
            'n_points': len(phase_series),
            'mean': phase_stats['mean'],
            'std': phase_stats['std'],
            'min': phase_stats['min'],
            'max': phase_stats['max'],
        }

    # Comparaison entre phases
    if n_phases >= 2:
        first_mean = phases[phase_names[0]]['mean']
        last_mean = phases[phase_names[-1]]['mean']
        if first_mean != 0:
            improvement_pct = ((last_mean - first_mean) / abs(first_mean)) * 100
        else:
            improvement_pct = float('inf') if last_mean != 0 else 0.0

        phases['comparison'] = {
            'first_phase': phase_names[0],
            'last_phase': phase_names[-1],
            'first_mean': first_mean,
            'last_mean': last_mean,
            'improvement_pct': improvement_pct,
            'direction': 'amélioration' if improvement_pct > 0 else 'dégradation' if improvement_pct < 0 else 'stable',
        }

    return phases


# ============================================================================
# Génération de rapport
# ============================================================================

def format_number(value: float, precision: int = 4) -> str:
    """Formate un nombre avec la précision appropriée."""
    if value is None:
        return "N/A"
    if abs(value) < 1e-3 or abs(value) > 1e6:
        return f"{value:.4e}"
    return f"{value:.{precision}f}"


def group_metrics_by_category(metrics_dict: Dict[str, pd.Series]) -> Dict[str, List[str]]:
    """Groupe les métriques par catégorie (train/, rollout/, env/, etc.)."""
    categories = defaultdict(list)
    for metric_name in metrics_dict.keys():
        if '/' in metric_name:
            category = metric_name.split('/')[0]
        else:
            category = 'other'
        categories[category].append(metric_name)

    return dict(categories)


def generate_report(metrics_dict: Dict[str, pd.Series], output_file: Optional[str] = None) -> str:
    """
    Génère un rapport statistique complet pour toutes les métriques.

    Args:
        metrics_dict: Dictionnaire {metric_name: pd.Series}
        output_file: Fichier de sortie optionnel

    Returns:
        Rapport sous forme de string
    """
    lines = []
    lines.append("=" * 80)
    lines.append("ANALYSE STATISTIQUE DES LOGS TENSORBOARD")
    lines.append("=" * 80)
    lines.append("")

    # Grouper par catégorie
    categories = group_metrics_by_category(metrics_dict)
    category_order = ['train', 'rollout', 'env', 'rewards', 'custom', 'loss', 'eval', 'other']

    for category in category_order:
        if category not in categories:
            continue

        metric_names = sorted(categories[category])
        if not metric_names:
            continue

        lines.append("")
        lines.append("=" * 80)
        lines.append(f"CATÉGORIE: {category.upper()}")
        lines.append("=" * 80)
        lines.append("")

        for metric_name in metric_names:
            if metric_name not in metrics_dict:
                continue

            series = metrics_dict[metric_name]
            lines.append(f"\n--- {metric_name} ---")
            lines.append("")

            # Statistiques descriptives
            stats_dict = compute_descriptive_stats(series)
            if stats_dict:
                if HAS_TABULATE:
                    stats_table = [
                        ['Count', stats_dict['count']],
                        ['Plage steps', f"{stats_dict['min_step']} -> {stats_dict['max_step']} ({stats_dict['step_range']})"],
                        ['Moyenne', format_number(stats_dict['mean'])],
                        ['Médiane', format_number(stats_dict['median'])],
                        ['Écart-type', format_number(stats_dict['std'])],
                        ['Min', format_number(stats_dict['min'])],
                        ['Max', format_number(stats_dict['max'])],
                        ['Q25', format_number(stats_dict['q25'])],
                        ['Q75', format_number(stats_dict['q75'])],
                        ['Q90', format_number(stats_dict['q90'])],
                        ['Q95', format_number(stats_dict['q95'])],
                        ['Q99', format_number(stats_dict['q99'])],
                        ['CV', format_number(stats_dict['cv'])],
                    ]
                    if stats_dict['skewness'] is not None:
                        stats_table.append(['Skewness', format_number(stats_dict['skewness'])])
                        stats_table.append(['Kurtosis', format_number(stats_dict['kurtosis'])])
                    lines.append(tabulate(stats_table, headers=['Statistique', 'Valeur'], tablefmt='grid'))
                else:
                    lines.append(f"  Count: {stats_dict['count']}")
                    lines.append(f"  Plage steps: {stats_dict['min_step']} -> {stats_dict['max_step']} ({stats_dict['step_range']})")
                    lines.append(f"  Moyenne: {format_number(stats_dict['mean'])}")
                    lines.append(f"  Médiane: {format_number(stats_dict['median'])}")
                    lines.append(f"  Écart-type: {format_number(stats_dict['std'])}")
                    lines.append(f"  Min: {format_number(stats_dict['min'])} | Max: {format_number(stats_dict['max'])}")
                    lines.append(f"  Q25: {format_number(stats_dict['q25'])} | Q75: {format_number(stats_dict['q75'])}")
                    lines.append(f"  Q90: {format_number(stats_dict['q90'])} | Q95: {format_number(stats_dict['q95'])} | Q99: {format_number(stats_dict['q99'])}")
                    lines.append(f"  CV: {format_number(stats_dict['cv'])}")
                    if stats_dict['skewness'] is not None:
                        lines.append(f"  Skewness: {format_number(stats_dict['skewness'])} | Kurtosis: {format_number(stats_dict['kurtosis'])}")

            # Analyse de tendance
            trends = analyze_trends(series)
            if trends:
                lines.append("")
                lines.append("  TENDANCE:")
                arrow = "↑" if trends['trend'] == 'croissante' else "↓" if trends['trend'] == 'décroissante' else "→"
                lines.append(f"    Direction: {trends['trend'].upper()} {arrow}")
                lines.append(f"    Pente: {format_number(trends['slope'])}")
                lines.append(f"    R²: {format_number(trends['r2'], 6)}")
                lines.append(f"    Début (moyenne): {format_number(trends['start_mean'])}")
                if trends['mid_mean'] is not None:
                    lines.append(f"    Milieu (moyenne): {format_number(trends['mid_mean'])}")
                lines.append(f"    Fin (moyenne): {format_number(trends['end_mean'])}")
                if trends['plateau_detected']:
                    lines.append(f"    ⚠ Plateau détecté")

            # Anomalies
            anomalies = detect_anomalies(series)
            if anomalies['n_outliers'] > 0 or anomalies['n_discontinuities'] > 0 or anomalies['n_spikes'] > 0:
                lines.append("")
                lines.append("  ANOMALIES DÉTECTÉES:")
                if anomalies['n_outliers'] > 0:
                    lines.append(f"    Outliers (IQR): {anomalies['n_outliers']}")
                    # Afficher les 5 premiers
                    for outlier in anomalies['outliers'][:5]:
                        lines.append(f"      Step {outlier['step']}: {format_number(outlier['value'])}")
                    if len(anomalies['outliers']) > 5:
                        lines.append(f"      ... et {len(anomalies['outliers']) - 5} autres")
                if anomalies['n_discontinuities'] > 0:
                    lines.append(f"    Discontinuités: {anomalies['n_discontinuities']}")
                    for disc in anomalies['discontinuities'][:3]:
                        lines.append(f"      Step {disc['step']}: saut de {format_number(disc['jump'])}")
                if anomalies['n_spikes'] > 0:
                    lines.append(f"    Spikes: {anomalies['n_spikes']}")

            # Analyse par phases
            phases = analyze_phases(series)
            if phases and 'comparison' in phases:
                lines.append("")
                lines.append("  ANALYSE PAR PHASES:")
                comp = phases['comparison']
                lines.append(f"    {comp['first_phase']}: moyenne = {format_number(comp['first_mean'])}")
                lines.append(f"    {comp['last_phase']}: moyenne = {format_number(comp['last_mean'])}")
                lines.append(f"    Évolution: {comp['improvement_pct']:+.2f}% ({comp['direction']})")

    lines.append("")
    lines.append("=" * 80)
    lines.append("FIN DE L'ANALYSE")
    lines.append("=" * 80)

    report = "\n".join(lines)

    # Sauvegarder si demandé
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n[OK] Rapport sauvegardé dans: {output_file}")

    return report


# ============================================================================
# Fonction principale
# ============================================================================

def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(
        description="Analyse statistique des logs TensorBoard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python scripts/analyze_tensorboard.py --log_dir logs/tensorboard/run_1
  python scripts/analyze_tensorboard.py --log_dir logs/wfo/segment_0 --output report.txt
  python scripts/analyze_tensorboard.py --log_dir logs/tensorboard --run_name run_5
        """
    )

    parser.add_argument(
        '--log_dir',
        type=str,
        required=True,
        help='Dossier contenant les logs TensorBoard'
    )

    parser.add_argument(
        '--run_name',
        type=str,
        default=None,
        help='Nom du run spécifique (optionnel). Si non spécifié, utilise le plus récent.'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Fichier de sortie pour sauvegarder le rapport (optionnel)'
    )

    args = parser.parse_args()

    # Vérifier que le dossier existe
    if not os.path.exists(args.log_dir):
        print(f"[ERREUR] Le dossier '{args.log_dir}' n'existe pas.")
        return 1

    try:
        # Charger les métriques
        print(f"\n[INFO] Chargement des métriques depuis: {args.log_dir}")
        if args.run_name:
            print(f"[INFO] Run spécifique: {args.run_name}")
        metrics_dict = load_tensorboard_run(args.log_dir, args.run_name)

        # Générer le rapport
        print(f"\n[INFO] Génération du rapport statistique...")
        report = generate_report(metrics_dict, args.output)

        # Afficher le rapport
        print("\n" + report)

        return 0

    except FileNotFoundError as e:
        print(f"\n[ERREUR] {e}")
        print("\nConseil: Vérifiez que le dossier contient des logs TensorBoard valides.")
        print("Les logs TensorBoard sont généralement générés lors de l'entraînement.")
        return 1
    except Exception as e:
        print(f"\n[ERREUR] {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    # Ajouter le chemin racine du projet
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)

    exit(main())

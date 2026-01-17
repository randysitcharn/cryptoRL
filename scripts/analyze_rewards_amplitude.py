# -*- coding: utf-8 -*-
"""
analyze_rewards_amplitude.py - Analyse de l'amplitude des rewards et pénalités.

Analyse l'amplitude (min, max, std, range) des rewards et pénalités tout au long
de l'entraînement pour comprendre leur évolution et leur stabilité.

Usage:
    python scripts/analyze_rewards_amplitude.py --log_dir logs/tensorboard/run_1
    python scripts/analyze_rewards_amplitude.py --log_dir logs/wfo/segment_0 --output report_rewards.txt
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Backend non-interactif pour serveurs
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from tensorboard.backend.event_processing import event_accumulator

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

    print(f"[INFO] {len(all_tags)} métrique(s) trouvée(s)")

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

    return result


# ============================================================================
# Filtrage des métriques de rewards
# ============================================================================

def filter_reward_metrics(metrics_dict: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
    """
    Filtre uniquement les métriques liées aux rewards et pénalités.

    Args:
        metrics_dict: Dictionnaire de toutes les métriques

    Returns:
        Dictionnaire filtré avec uniquement les métriques rewards/pénalités
    """
    reward_keys = [
        # Patterns pour différents noms possibles
        'rewards/', 'reward/', 'internal/reward/',
        'rollout/ep_rew', 'rollout/reward',
        'churn/', 'internal/churn/',
    ]

    filtered = {}
    for key, series in metrics_dict.items():
        # Garder toutes les métriques qui contiennent "reward", "penalty", "churn"
        key_lower = key.lower()
        if any(pattern.lower() in key_lower for pattern in reward_keys) or \
           'reward' in key_lower or 'penalty' in key_lower or 'churn' in key_lower:
            filtered[key] = series

    # Normaliser les noms si possible (trouver les variantes)
    normalized = {}
    key_mapping = {}

    # Chercher les variantes (internal/reward/pnl_component vs rewards/pnl)
    for key in filtered.keys():
        base_name = None
        if 'pnl' in key.lower() or 'log_return' in key.lower():
            base_name = 'pnl'
        elif 'churn' in key.lower():
            base_name = 'churn_penalty'
        elif 'smooth' in key.lower():
            base_name = 'smoothness_penalty'
        elif 'downside' in key.lower() or 'vol' in key.lower():
            base_name = 'downside_penalty'
        elif 'upside' in key.lower() or 'bonus' in key.lower():
            base_name = 'upside_bonus'
        elif 'total' in key.lower() or 'raw' in key.lower():
            base_name = 'total_reward'
        elif 'scaled' in key.lower():
            base_name = 'scaled_reward'
        elif 'ep_rew' in key.lower():
            base_name = 'episode_reward'

        if base_name:
            if base_name not in normalized:
                normalized[base_name] = filtered[key]
                key_mapping[base_name] = key
            else:
                # Si déjà présent, prendre celui avec le plus de points
                if len(filtered[key]) > len(normalized[base_name]):
                    normalized[base_name] = filtered[key]
                    key_mapping[base_name] = key

    # Ajouter les métriques non normalisées
    for key, series in filtered.items():
        if key not in key_mapping.values():
            normalized[key] = series

    return normalized


# ============================================================================
# Analyse de l'amplitude
# ============================================================================

def compute_amplitude_stats(series: pd.Series, window_size: int = 100) -> Dict[str, pd.Series]:
    """
    Calcule les statistiques d'amplitude glissantes pour une série.

    Args:
        series: pd.Series avec index=step et values=metric values
        window_size: Taille de la fenêtre glissante

    Returns:
        Dictionnaire avec différentes statistiques d'amplitude
    """
    if len(series) == 0:
        return {}

    values = series.values
    steps = series.index.values

    # Statistiques globales
    global_min = np.min(values)
    global_max = np.max(values)
    global_amplitude = global_max - global_min
    global_std = np.std(values)
    global_mean = np.mean(values)

    # Statistiques glissantes (rolling window)
    df = pd.DataFrame({'value': values}, index=steps)
    rolling = df['value'].rolling(window=min(window_size, len(df)), center=True, min_periods=1)

    rolling_min = rolling.min()
    rolling_max = rolling.max()
    rolling_amplitude = rolling_max - rolling_min
    rolling_std = rolling.std()
    rolling_mean = rolling.mean()

    # Amplitude relative (en pourcentage de la moyenne)
    rolling_amplitude_pct = (rolling_amplitude / np.abs(rolling_mean + 1e-10)) * 100

    # Coefficient de variation glissant
    rolling_cv = rolling_std / (np.abs(rolling_mean) + 1e-10)

    return {
        'global_min': global_min,
        'global_max': global_max,
        'global_amplitude': global_amplitude,
        'global_std': global_std,
        'global_mean': global_mean,
        'rolling_min': pd.Series(rolling_min.values, index=steps),
        'rolling_max': pd.Series(rolling_max.values, index=steps),
        'rolling_amplitude': pd.Series(rolling_amplitude.values, index=steps),
        'rolling_std': pd.Series(rolling_std.values, index=steps),
        'rolling_mean': pd.Series(rolling_mean.values, index=steps),
        'rolling_amplitude_pct': pd.Series(rolling_amplitude_pct.values, index=steps),
        'rolling_cv': pd.Series(rolling_cv.values, index=steps),
    }


def analyze_amplitude_evolution(series: pd.Series, n_segments: int = 5) -> Dict[str, Any]:
    """
    Analyse l'évolution de l'amplitude au cours du training (par segments).

    Args:
        series: pd.Series avec index=step et values=metric values
        n_segments: Nombre de segments pour l'analyse

    Returns:
        Dictionnaire avec analyse par segments
    """
    if len(series) < n_segments:
        return {}

    n_points = len(series)
    segment_size = n_points // n_segments

    segments = {}
    segment_names = [f'segment_{i+1}' for i in range(n_segments)]

    for i in range(n_segments):
        start_idx = i * segment_size
        if i == n_segments - 1:
            end_idx = n_points
        else:
            end_idx = (i + 1) * segment_size

        segment_series = series.iloc[start_idx:end_idx]
        segment_values = segment_series.values

        segment_min = np.min(segment_values)
        segment_max = np.max(segment_values)
        segment_amplitude = segment_max - segment_min
        segment_std = np.std(segment_values)
        segment_mean = np.mean(segment_values)

        segments[segment_names[i]] = {
            'start_step': int(segment_series.index.min()),
            'end_step': int(segment_series.index.max()),
            'n_points': len(segment_series),
            'min': float(segment_min),
            'max': float(segment_max),
            'amplitude': float(segment_amplitude),
            'std': float(segment_std),
            'mean': float(segment_mean),
            'cv': float(segment_std / (np.abs(segment_mean) + 1e-10)),
        }

    # Évolution de l'amplitude entre segments
    amplitudes = [segments[name]['amplitude'] for name in segment_names]
    stds = [segments[name]['std'] for name in segment_names]

    segments['evolution'] = {
        'amplitude_trend': 'croissante' if amplitudes[-1] > amplitudes[0] else \
                          'décroissante' if amplitudes[-1] < amplitudes[0] else 'stable',
        'amplitude_change_pct': float(((amplitudes[-1] - amplitudes[0]) / (amplitudes[0] + 1e-10)) * 100),
        'std_trend': 'croissante' if stds[-1] > stds[0] else \
                    'décroissante' if stds[-1] < stds[0] else 'stable',
        'std_change_pct': float(((stds[-1] - stds[0]) / (stds[0] + 1e-10)) * 100),
    }

    return segments


# ============================================================================
# Visualisation
# ============================================================================

def plot_amplitude_analysis(
    metrics_dict: Dict[str, pd.Series],
    output_dir: Optional[str] = None,
    figure_size: Tuple[int, int] = (16, 10)
) -> None:
    """
    Crée des graphiques d'analyse de l'amplitude.

    Args:
        metrics_dict: Dictionnaire de métriques rewards/pénalités
        output_dir: Dossier pour sauvegarder les figures
        figure_size: Taille des figures
    """
    if not metrics_dict:
        print("[WARNING] Aucune métrique à visualiser")
        return

    n_metrics = len(metrics_dict)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols

    # Figure 1: Évolution des valeurs brutes avec amplitude glissante
    fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=figure_size, squeeze=False)
    axes1 = axes1.flatten()

    # Figure 2: Évolution de l'amplitude glissante
    fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=figure_size, squeeze=False)
    axes2 = axes2.flatten()

    for idx, (metric_name, series) in enumerate(sorted(metrics_dict.items())):
        if idx >= len(axes1):
            break

        steps = series.index.values
        values = series.values

        # Calculer les statistiques d'amplitude
        amp_stats = compute_amplitude_stats(series)

        # Graphique 1: Valeurs brutes avec enveloppe d'amplitude
        ax1 = axes1[idx]
        ax1.plot(steps, values, alpha=0.5, color='blue', label='Valeur', linewidth=0.5)
        if 'rolling_mean' in amp_stats:
            ax1.plot(steps, amp_stats['rolling_mean'].values, 'g-', label='Moyenne glissante', linewidth=1.5)
            ax1.fill_between(
                steps,
                amp_stats['rolling_min'].values,
                amp_stats['rolling_max'].values,
                alpha=0.2, color='orange', label='Amplitude glissante'
            )
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Valeur')
        ax1.set_title(f'{metric_name}\nAmplitude globale: {amp_stats.get("global_amplitude", 0):.4f}')
        ax1.legend(loc='best', fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Graphique 2: Évolution de l'amplitude glissante
        ax2 = axes2[idx]
        if 'rolling_amplitude' in amp_stats:
            ax2.plot(steps, amp_stats['rolling_amplitude'].values, 'r-', label='Amplitude', linewidth=1.5)
            ax2.plot(steps, amp_stats['rolling_std'].values, 'b--', label='Écart-type', linewidth=1.5)
            ax2.axhline(y=amp_stats['global_amplitude'], color='gray', linestyle=':', label='Amplitude globale')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Amplitude / Std')
        ax2.set_title(f'{metric_name}\nÉvolution de l\'amplitude')
        ax2.legend(loc='best', fontsize=8)
        ax2.grid(True, alpha=0.3)

    # Cacher les axes vides
    for idx in range(len(metrics_dict), len(axes1)):
        axes1[idx].set_visible(False)
        axes2[idx].set_visible(False)

    plt.tight_layout()

    # Sauvegarder
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig1.savefig(os.path.join(output_dir, 'rewards_amplitude_raw.png'), dpi=150, bbox_inches='tight')
        fig2.savefig(os.path.join(output_dir, 'rewards_amplitude_evolution.png'), dpi=150, bbox_inches='tight')
        print(f"[OK] Graphiques sauvegardés dans {output_dir}")
    else:
        # Sauvegarder dans le dossier courant
        fig1.savefig('rewards_amplitude_raw.png', dpi=150, bbox_inches='tight')
        fig2.savefig('rewards_amplitude_evolution.png', dpi=150, bbox_inches='tight')
        print("[OK] Graphiques sauvegardés: rewards_amplitude_raw.png, rewards_amplitude_evolution.png")

    plt.close(fig1)
    plt.close(fig2)


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


def generate_amplitude_report(
    metrics_dict: Dict[str, pd.Series],
    output_file: Optional[str] = None
) -> str:
    """
    Génère un rapport d'analyse de l'amplitude.

    Args:
        metrics_dict: Dictionnaire de métriques rewards/pénalités
        output_file: Fichier de sortie optionnel

    Returns:
        Rapport sous forme de string
    """
    lines = []
    lines.append("=" * 80)
    lines.append("ANALYSE DE L'AMPLITUDE DES REWARDS ET PÉNALITÉS")
    lines.append("=" * 80)
    lines.append("")

    if not metrics_dict:
        lines.append("[ERREUR] Aucune métrique de reward/pénalité trouvée.")
        report = "\n".join(lines)
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
        return report

    # Statistiques globales pour toutes les métriques
    lines.append("─" * 80)
    lines.append("RÉSUMÉ GLOBAL")
    lines.append("─" * 80)
    lines.append("")

    summary_table = []
    for metric_name in sorted(metrics_dict.keys()):
        series = metrics_dict[metric_name]
        amp_stats = compute_amplitude_stats(series)

        summary_table.append([
            metric_name,
            len(series),
            format_number(amp_stats['global_mean']),
            format_number(amp_stats['global_std']),
            format_number(amp_stats['global_min']),
            format_number(amp_stats['global_max']),
            format_number(amp_stats['global_amplitude']),
            format_number(amp_stats['global_std'] / (abs(amp_stats['global_mean']) + 1e-10)),
        ])

    if HAS_TABULATE:
        lines.append(tabulate(
            summary_table,
            headers=['Métrique', 'N points', 'Moyenne', 'Std', 'Min', 'Max', 'Amplitude', 'CV'],
            tablefmt='grid'
        ))
    else:
        lines.append(f"{'Métrique':<30} {'N':>8} {'Moyenne':>12} {'Std':>12} {'Min':>12} {'Max':>12} {'Amplitude':>12} {'CV':>10}")
        lines.append("-" * 120)
        for row in summary_table:
            lines.append(f"{row[0]:<30} {row[1]:>8} {row[2]:>12} {row[3]:>12} {row[4]:>12} {row[5]:>12} {row[6]:>12} {row[7]:>10}")

    lines.append("")

    # Analyse détaillée par métrique
    for metric_name in sorted(metrics_dict.keys()):
        series = metrics_dict[metric_name]
        lines.append("")
        lines.append("─" * 80)
        lines.append(f"ANALYSE DÉTAILLÉE: {metric_name}")
        lines.append("─" * 80)
        lines.append("")

        # Statistiques d'amplitude
        amp_stats = compute_amplitude_stats(series)
        lines.append("Statistiques globales:")
        lines.append(f"  Moyenne: {format_number(amp_stats['global_mean'])}")
        lines.append(f"  Écart-type: {format_number(amp_stats['global_std'])}")
        lines.append(f"  Min: {format_number(amp_stats['global_min'])}")
        lines.append(f"  Max: {format_number(amp_stats['global_max'])}")
        lines.append(f"  Amplitude (Max - Min): {format_number(amp_stats['global_amplitude'])}")
        lines.append(f"  Coefficient de variation: {format_number(amp_stats['global_std'] / (abs(amp_stats['global_mean']) + 1e-10))}")

        # Analyse par segments
        segments_analysis = analyze_amplitude_evolution(series, n_segments=5)
        if segments_analysis and 'evolution' in segments_analysis:
            lines.append("")
            lines.append("Évolution par segments (5 segments égaux):")
            for seg_name in [f'segment_{i+1}' for i in range(5)]:
                if seg_name in segments_analysis:
                    seg = segments_analysis[seg_name]
                    lines.append(f"  {seg_name}:")
                    lines.append(f"    Steps: {seg['start_step']} -> {seg['end_step']} ({seg['n_points']} points)")
                    lines.append(f"    Amplitude: {format_number(seg['amplitude'])} (Min: {format_number(seg['min'])}, Max: {format_number(seg['max'])})")
                    lines.append(f"    Std: {format_number(seg['std'])}, CV: {format_number(seg['cv'])}")

            lines.append("")
            lines.append("Tendance d'évolution:")
            evo = segments_analysis['evolution']
            lines.append(f"  Amplitude: {evo['amplitude_trend']} ({evo['amplitude_change_pct']:+.2f}%)")
            lines.append(f"  Écart-type: {evo['std_trend']} ({evo['std_change_pct']:+.2f}%)")

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
        description="Analyse de l'amplitude des rewards et pénalités",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python scripts/analyze_rewards_amplitude.py --log_dir logs/tensorboard/run_1
  python scripts/analyze_rewards_amplitude.py --log_dir logs/wfo/segment_0 --output report_rewards.txt --plots_dir plots/
  python scripts/analyze_rewards_amplitude.py --log_dir logs/tensorboard --run_name run_5 --output report.txt
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

    parser.add_argument(
        '--plots_dir',
        type=str,
        default=None,
        help='Dossier pour sauvegarder les graphiques (optionnel). Par défaut, sauvegarde dans le dossier courant.'
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
        all_metrics = load_tensorboard_run(args.log_dir, args.run_name)

        # Filtrer les métriques de rewards/pénalités
        print(f"\n[INFO] Filtrage des métriques de rewards/pénalités...")
        reward_metrics = filter_reward_metrics(all_metrics)

        if not reward_metrics:
            print("[ERREUR] Aucune métrique de reward/pénalité trouvée dans les logs.")
            print(f"Métriques disponibles: {sorted(all_metrics.keys())}")
            return 1

        print(f"[OK] {len(reward_metrics)} métrique(s) de reward/pénalité trouvée(s):")
        for metric_name in sorted(reward_metrics.keys()):
            series = reward_metrics[metric_name]
            print(f"  - {metric_name}: {len(series)} points (steps {series.index.min()} -> {series.index.max()})")

        # Générer le rapport
        print(f"\n[INFO] Génération du rapport d'analyse...")
        report = generate_amplitude_report(reward_metrics, args.output)

        # Créer les graphiques
        print(f"\n[INFO] Génération des graphiques...")
        plot_amplitude_analysis(reward_metrics, args.plots_dir)

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

# -*- coding: utf-8 -*-
"""
export_metrics.py - Script d'analyse des metriques d'entrainement.

Extrait les metriques TensorBoard, resample a 5000 steps,
et calcule les indicateurs de stabilite.
"""

import os
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator


def find_all_event_files(log_dir: str) -> list:
    """Trouve tous les fichiers event TensorBoard dans un dossier."""
    event_files = []

    for root, dirs, files in os.walk(log_dir):
        for f in files:
            if f.startswith('events.out.tfevents'):
                full_path = os.path.join(root, f)
                event_files.append((full_path, os.path.getmtime(full_path)))

    if not event_files:
        raise FileNotFoundError(f"Aucun fichier TensorBoard trouve dans {log_dir}")

    # Tri par date (plus recent en premier)
    event_files.sort(key=lambda x: x[1], reverse=True)
    return [f[0] for f in event_files]


def load_tensorboard_metrics(log_dir: str) -> pd.DataFrame:
    """
    Charge les metriques depuis TOUS les fichiers TensorBoard.

    Returns:
        DataFrame avec colonnes: step, critic_loss, actor_loss, ep_rew_mean
    """
    event_files = find_all_event_files(log_dir)
    print(f"[INFO] {len(event_files)} fichiers TensorBoard trouves")

    # Mapping des noms de metriques
    metric_mapping = {
        'train/critic_loss': 'critic_loss',
        'train/actor_loss': 'actor_loss',
        'rollout/reward': 'ep_rew_mean',
        'rollout/ep_rew_mean': 'ep_rew_mean',
    }

    # Aggreger les donnees de tous les fichiers
    data = {}
    all_tags = set()

    for event_file in event_files:
        print(f"[INFO] Lecture de: {os.path.basename(os.path.dirname(event_file))}/{os.path.basename(event_file)[:40]}...")

        ea = event_accumulator.EventAccumulator(event_file)
        ea.Reload()

        available_tags = ea.Tags().get('scalars', [])
        all_tags.update(available_tags)

        for tb_tag, col_name in metric_mapping.items():
            if tb_tag in available_tags:
                events = ea.Scalars(tb_tag)
                steps = [e.step for e in events]
                values = [e.value for e in events]

                if col_name not in data:
                    data[col_name] = {'steps': [], 'values': []}

                data[col_name]['steps'].extend(steps)
                data[col_name]['values'].extend(values)

    print(f"[INFO] Tags disponibles: {list(all_tags)}")

    if not data:
        raise ValueError("Aucune metrique trouvee dans les logs TensorBoard")

    # Afficher stats par metrique
    for col_name, metric_data in data.items():
        print(f"[OK] {col_name}: {len(metric_data['steps'])} points (steps {min(metric_data['steps'])} -> {max(metric_data['steps'])})")

    # Fusionner toutes les metriques sur une grille commune
    all_steps = set()
    for metric_data in data.values():
        all_steps.update(metric_data['steps'])
    all_steps = sorted(all_steps)

    # Creer le DataFrame avec interpolation
    df = pd.DataFrame({'step': all_steps})

    for col_name, metric_data in data.items():
        # Trier et dedupliquer les points (prendre la derniere valeur pour chaque step)
        step_value = {}
        for s, v in zip(metric_data['steps'], metric_data['values']):
            step_value[s] = v

        sorted_steps = sorted(step_value.keys())
        sorted_values = [step_value[s] for s in sorted_steps]

        steps = np.array(sorted_steps)
        values = np.array(sorted_values)

        # Interpolation lineaire sur la grille commune
        interpolated = np.interp(all_steps, steps, values)
        df[col_name] = interpolated

    return df


def resample_to_5k(df: pd.DataFrame, step_interval: int = 5000) -> pd.DataFrame:
    """
    Resample le DataFrame pour avoir exactement une ligne tous les N steps.

    Args:
        df: DataFrame avec colonne 'step'
        step_interval: Intervalle entre les points (defaut: 5000)

    Returns:
        DataFrame resampled
    """
    min_step = 0
    max_step = int(df['step'].max())

    # Grille reguliere
    new_steps = np.arange(min_step, max_step + 1, step_interval)

    result = pd.DataFrame({'step': new_steps})

    # Interpoler chaque colonne
    for col in df.columns:
        if col != 'step':
            result[col] = np.interp(
                new_steps,
                df['step'].values,
                df[col].values
            )

    return result


def analyze_stability(df: pd.DataFrame) -> dict:
    """
    Analyse la stabilite de l'entrainement.

    Calcule:
    - Variation moyenne de critic_loss sur les 5 derniers points
    - Pente de ep_rew_mean (tendance globale)

    Returns:
        dict avec les indicateurs de stabilite
    """
    results = {}

    # Variation critic_loss (5 derniers points)
    if 'critic_loss' in df.columns and len(df) >= 5:
        last_5 = df['critic_loss'].tail(5).values
        variations = np.abs(np.diff(last_5))
        results['critic_loss_mean_variation'] = np.mean(variations)
        results['critic_loss_is_stable'] = results['critic_loss_mean_variation'] < 1.0

    # Pente de ep_rew_mean (regression lineaire)
    if 'ep_rew_mean' in df.columns and len(df) >= 2:
        x = df['step'].values
        y = df['ep_rew_mean'].values

        # Regression lineaire simple: y = mx + b
        n = len(x)
        slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)
        results['ep_rew_mean_slope'] = slope
        results['ep_rew_mean_trend'] = 'up' if slope > 0 else 'down'

    return results


def main():
    """Fonction principale."""
    print("=" * 60)
    print("CryptoRL - Export Metrics Analysis")
    print("=" * 60)

    # Configuration
    LOG_DIR = "logs/demo/tensorboard_steps"
    OUTPUT_CSV = "logs/training_metrics_5k.csv"
    TARGET_POINTS = 20  # Nombre de points cible dans le resample

    # 1. Charger les metriques TensorBoard
    print(f"\n[INFO] Recherche des logs dans: {LOG_DIR}")
    df_raw = load_tensorboard_metrics(LOG_DIR)
    print(f"[OK] {len(df_raw)} points charges")

    # 2. Calculer l'intervalle optimal pour avoir environ TARGET_POINTS
    max_step = int(df_raw['step'].max())
    step_interval = max(1, max_step // TARGET_POINTS)

    # Arrondir a un multiple de 5 pour la lisibilite
    if step_interval > 10:
        step_interval = (step_interval // 5) * 5
    if step_interval < 1:
        step_interval = 1

    print(f"\n[INFO] Resampling a {step_interval} steps (max_step={max_step})...")
    df_resampled = resample_to_5k(df_raw, step_interval=step_interval)
    print(f"[OK] {len(df_resampled)} points apres resampling")

    # 3. Sauvegarder CSV
    df_resampled.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[OK] CSV sauvegarde: {OUTPUT_CSV}")

    # 4. Afficher les 10 dernieres lignes
    print("\n" + "=" * 60)
    print("=== 10 DERNIERES LIGNES ===")
    print("=" * 60)
    print(df_resampled.tail(10).to_string(index=False))

    # 5. Analyse de stabilite
    print("\n" + "=" * 60)
    print("=== ANALYSE DE STABILITE ===")
    print("=" * 60)

    stability = analyze_stability(df_resampled)

    if 'critic_loss_mean_variation' in stability:
        status = "STABLE" if stability['critic_loss_is_stable'] else "INSTABLE"
        print(f"Variation moyenne critic_loss (5 derniers): {stability['critic_loss_mean_variation']:.4f} [{status}]")

    if 'ep_rew_mean_slope' in stability:
        arrow = "↑" if stability['ep_rew_mean_trend'] == 'up' else "↓"
        print(f"Pente ep_rew_mean: {stability['ep_rew_mean_slope']:+.6f} {arrow} ({stability['ep_rew_mean_trend'].upper()})")

    print("\n" + "=" * 60)
    print("Analyse terminee.")
    print("=" * 60)

    return df_resampled, stability


if __name__ == "__main__":
    import sys
    # Ajouter le chemin racine du projet
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(project_root)

    df, stability = main()

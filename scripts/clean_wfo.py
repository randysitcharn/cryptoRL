#!/usr/bin/env python3
"""
clean_wfo.py - Nettoyage des artefacts générés par le WFO (Walk-Forward Optimization).

Supprime :
- Logs et runs TensorBoard (logs/wfo/, logs/vol_debug.txt, etc.)
- Données WFO (data/wfo/, segment train/eval/test parquet)
- Modèles et artefacts (models/wfo/, scaler, HMM)
- Poids encodeur, MAE, TQC (weights/wfo/, encoder.pth, tqc.zip, checkpoints, etc.)
- Résultats (results/wfo_results.csv, results/plots/)

Usage:
    python scripts/clean_wfo.py                    # Tout nettoyer (avec confirmation)
    python scripts/clean_wfo.py --yes              # Sans confirmation
    python scripts/clean_wfo.py --dry-run          # Afficher sans supprimer
    python scripts/clean_wfo.py --logs-only        # Logs + TensorBoard uniquement
    python scripts/clean_wfo.py --weights-only     # Poids (encoder, TQC, checkpoints) uniquement

Chemins par defaut alignes sur WFOConfig (run_full_wfo.py) :
  data/wfo, models/wfo, weights/wfo, results/wfo_results.csv, results/plots.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Defaults alignés avec WFOConfig (run_full_wfo.py)
DEFAULT_OUTPUT_DIR = "data/wfo"
DEFAULT_MODELS_DIR = "models/wfo"
DEFAULT_WEIGHTS_DIR = "weights/wfo"
DEFAULT_RESULTS_PATH = "results/wfo_results.csv"
DEFAULT_LOGS_WFO = "logs/wfo"
DEFAULT_LOGS_DIR = "logs"
DEFAULT_PLOTS_DIR = "results/plots"
GITKEEP = ".gitkeep"


def _resolve(path: str | Path, base: Path = ROOT) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = base / p
    return p.resolve()


def _dir_size(path: Path) -> int:
    """Taille récursive d'un répertoire en octets."""
    if not path.exists() or not path.is_dir():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _fmt_size(n: int) -> str:
    if n >= 1024**3:
        return f"{n / 1024**3:.1f} GB"
    if n >= 1024**2:
        return f"{n / 1024**2:.1f} MB"
    if n >= 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n} B"


def _rmtree_if_exists(path: Path, dry_run: bool) -> int:
    """Supprime un répertoire récursivement. Retourne la taille libérée."""
    if not path.exists() or not path.is_dir():
        return 0
    size = _dir_size(path)
    if dry_run:
        print(f"  [DRY-RUN] Would remove: {path} ({_fmt_size(size)})")
        return size
    shutil.rmtree(path)
    print(f"  Removed: {path} ({_fmt_size(size)})")
    return size


def _rmfile_if_exists(path: Path, dry_run: bool) -> int:
    if not path.exists() or not path.is_file():
        return 0
    size = path.stat().st_size
    if dry_run:
        print(f"  [DRY-RUN] Would remove: {path} ({_fmt_size(size)})")
        return size
    path.unlink()
    print(f"  Removed: {path} ({_fmt_size(size)})")
    return size


def _clean_logs(base: Path, dry_run: bool) -> int:
    """Logs + TensorBoard WFO uniquement. Préserve logs/.gitkeep et les autres logs (audit, etc.)."""
    freed = 0
    # logs/wfo (HMM, MAE, segment_*, eval) — générés par WFO
    wfo = _resolve(DEFAULT_LOGS_WFO, base)
    freed += _rmtree_if_exists(wfo, dry_run)
    # vol_debug.txt (batch_env pendant WFO)
    vol_debug = _resolve(DEFAULT_LOGS_DIR, base) / "vol_debug.txt"
    freed += _rmfile_if_exists(vol_debug, dry_run)
    return freed


def _clean_data_wfo(base: Path, output_dir: str, dry_run: bool) -> int:
    """data/wfo et sous-dossiers segment_*."""
    d = _resolve(output_dir, base)
    return _rmtree_if_exists(d, dry_run)


def _clean_models_wfo(base: Path, models_dir: str, dry_run: bool) -> int:
    """models/wfo (HMM, scaler, etc.)."""
    d = _resolve(models_dir, base)
    return _rmtree_if_exists(d, dry_run)


def _clean_weights_wfo(base: Path, weights_dir: str, dry_run: bool) -> int:
    """weights/wfo (encoder.pth, mae_full.pth, tqc*, checkpoints, ensemble)."""
    d = _resolve(weights_dir, base)
    return _rmtree_if_exists(d, dry_run)


def _clean_results(base: Path, results_path: str, plots_dir: str, dry_run: bool) -> int:
    """results/wfo_results.csv et results/plots/ (segment_*_report.png)."""
    freed = 0
    freed += _rmfile_if_exists(_resolve(results_path, base), dry_run)
    pd = _resolve(plots_dir, base)
    if pd.exists() and pd.is_dir():
        for f in pd.glob("segment_*_*report.png"):
            freed += _rmfile_if_exists(f, dry_run)
        # Supprimer results/plots si vide
        if pd.exists() and not any(pd.iterdir()):
            if not dry_run:
                try:
                    pd.rmdir()
                    print(f"  Removed (empty): {pd}")
                except OSError:
                    pass
            else:
                print(f"  [DRY-RUN] Would remove (empty): {pd}")
    return freed


def run(
    base: Path = ROOT,
    *,
    logs: bool = True,
    data: bool = True,
    models: bool = True,
    weights: bool = True,
    results: bool = True,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    models_dir: str = DEFAULT_MODELS_DIR,
    weights_dir: str = DEFAULT_WEIGHTS_DIR,
    results_path: str = DEFAULT_RESULTS_PATH,
    plots_dir: str = DEFAULT_PLOTS_DIR,
    dry_run: bool = False,
) -> int:
    total = 0
    if logs:
        print("\n[Logs & TensorBoard]")
        total += _clean_logs(base, dry_run)
    if data:
        print("\n[Data WFO]")
        total += _clean_data_wfo(base, output_dir, dry_run)
    if models:
        print("\n[Models WFO]")
        total += _clean_models_wfo(base, models_dir, dry_run)
    if weights:
        print("\n[Weights WFO (encoder, MAE, TQC, checkpoints)]")
        total += _clean_weights_wfo(base, weights_dir, dry_run)
    if results:
        print("\n[Results]")
        total += _clean_results(base, results_path, plots_dir, dry_run)
    return total


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Nettoyer les artefacts générés par le WFO (logs, TensorBoard, weights, encoder, data)."
    )
    ap.add_argument("--yes", "-y", action="store_true", help="Pas de confirmation")
    ap.add_argument("--dry-run", action="store_true", help="Afficher les suppressions sans exécuter")
    ap.add_argument("--logs-only", action="store_true", help="Nettoyer uniquement logs + TensorBoard")
    ap.add_argument("--weights-only", action="store_true", help="Nettoyer uniquement weights/wfo")
    ap.add_argument("--data-only", action="store_true", help="Nettoyer uniquement data/wfo")
    ap.add_argument("--models-only", action="store_true", help="Nettoyer uniquement models/wfo")
    ap.add_argument("--results-only", action="store_true", help="Nettoyer uniquement results (CSV + plots)")
    ap.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="data/wfo path")
    ap.add_argument("--models-dir", type=str, default=DEFAULT_MODELS_DIR, help="models/wfo path")
    ap.add_argument("--weights-dir", type=str, default=DEFAULT_WEIGHTS_DIR, help="weights/wfo path")
    ap.add_argument("--results-path", type=str, default=DEFAULT_RESULTS_PATH, help="results CSV path")
    ap.add_argument("--plots-dir", type=str, default=DEFAULT_PLOTS_DIR, help="results/plots path")
    ap.add_argument("--base", type=str, default=None, help="Répertoire racine du projet (default: parent de scripts/)")
    args = ap.parse_args()

    base = Path(args.base).resolve() if args.base else ROOT

    logs = data = models = weights = results = True
    if args.logs_only:
        data = models = weights = results = False
    elif args.weights_only:
        logs = data = models = results = False
    elif args.data_only:
        logs = models = weights = results = False
    elif args.models_only:
        logs = data = weights = results = False
    elif args.results_only:
        logs = data = models = weights = False

    print("clean_wfo: nettoyage des artefacts WFO")
    print("=" * 50)
    print(f"  Base: {base}")
    if args.dry_run:
        print("  Mode: DRY-RUN (aucune suppression)")
    for k, v in [
        ("Logs/TensorBoard", logs),
        ("Data WFO", data),
        ("Models WFO", models),
        ("Weights WFO", weights),
        ("Results", results),
    ]:
        if v:
            print(f"  {k}: oui")

    if not args.yes and not args.dry_run:
        try:
            r = input("\nContinuer ? [y/N] ").strip().lower()
        except EOFError:
            r = "n"
        if r not in ("y", "yes"):
            print("Annulé.")
            sys.exit(0)

    total = run(
        base,
        logs=logs,
        data=data,
        models=models,
        weights=weights,
        results=results,
        output_dir=args.output_dir,
        models_dir=args.models_dir,
        weights_dir=args.weights_dir,
        results_path=args.results_path,
        plots_dir=args.plots_dir,
        dry_run=args.dry_run,
    )
    print("\n" + "=" * 50)
    if args.dry_run:
        print(f"  [DRY-RUN] Equivalent ~{_fmt_size(total)} a supprimer")
    else:
        print(f"  Libere: ~{_fmt_size(total)}")
    print("Termine.")


if __name__ == "__main__":
    main()

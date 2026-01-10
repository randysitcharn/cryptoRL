#!/usr/bin/env python3
"""
plot_wfo_equity.py - Visualisation de l'equity curve WFO combinée.

Crée un graphique montrant:
1. L'equity curve cumulée sur les 13 segments
2. Le Sharpe ratio par segment
3. Les métriques globales

Usage:
    python scripts/plot_wfo_equity.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Données du rapport WFO (13 segments complétés)
SEGMENTS = [
    {"id": 0, "pnl": -23.2, "sharpe": -6.71, "max_dd": 25.9, "trades": 12, "period": "2018 Q1"},
    {"id": 1, "pnl": -8.7, "sharpe": -1.47, "max_dd": 15.7, "trades": 52, "period": "2018 Q2"},
    {"id": 2, "pnl": 12.0, "sharpe": 0.37, "max_dd": 12.7, "trades": 216, "period": "2018 Q3"},
    {"id": 3, "pnl": -7.8, "sharpe": -2.15, "max_dd": 21.4, "trades": 68, "period": "2018 Q4"},
    {"id": 4, "pnl": -14.6, "sharpe": -3.63, "max_dd": 18.2, "trades": 25, "period": "2019 Q1"},
    {"id": 5, "pnl": -10.4, "sharpe": -2.19, "max_dd": 23.6, "trades": 146, "period": "2019 Q2"},
    {"id": 6, "pnl": 38.5, "sharpe": 1.34, "max_dd": 16.2, "trades": 227, "period": "2019 Q3"},
    {"id": 7, "pnl": 10.9, "sharpe": 0.41, "max_dd": 16.4, "trades": 31, "period": "2019 Q4"},
    {"id": 8, "pnl": 58.5, "sharpe": 2.84, "max_dd": 13.9, "trades": 29, "period": "2020 Q1"},
    {"id": 9, "pnl": 138.0, "sharpe": 2.57, "max_dd": 21.1, "trades": 23, "period": "2020 Q2"},
    {"id": 10, "pnl": -19.7, "sharpe": -3.55, "max_dd": 31.8, "trades": 63, "period": "2020 Q3"},
    {"id": 11, "pnl": 11.2, "sharpe": -0.47, "max_dd": 17.3, "trades": 99, "period": "2020 Q4"},
    {"id": 12, "pnl": -15.8, "sharpe": -3.12, "max_dd": 19.1, "trades": 204, "period": "2021 Q1"},
]


def calculate_equity_curve(segments: list, initial_capital: float = 100.0) -> list:
    """Calcule l'equity curve cumulée."""
    equity = [initial_capital]
    for seg in segments:
        new_value = equity[-1] * (1 + seg["pnl"] / 100)
        equity.append(new_value)
    return equity


def calculate_max_drawdown(equity: list) -> float:
    """Calcule le drawdown maximum."""
    peak = equity[0]
    max_dd = 0
    for value in equity:
        if value > peak:
            peak = value
        dd = (peak - value) / peak * 100
        if dd > max_dd:
            max_dd = dd
    return max_dd


def plot_wfo_equity(segments: list, output_path: str = "results/wfo_equity_curve.png"):
    """Crée le graphique de l'equity curve WFO."""

    # Calculs
    equity = calculate_equity_curve(segments)
    max_dd = calculate_max_drawdown(equity)
    final_return = (equity[-1] / equity[0] - 1) * 100
    positive_segments = sum(1 for s in segments if s["pnl"] > 0)
    avg_sharpe = np.mean([s["sharpe"] for s in segments])

    # Figure avec 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle("WFO Performance Analysis - 13 Segments (2018-2021)", fontsize=14, fontweight='bold')

    # 1. Equity Curve
    ax1 = axes[0]
    x_labels = ["Start"] + [s["period"] for s in segments]
    ax1.plot(range(len(equity)), equity, 'b-', linewidth=2, marker='o', markersize=6)
    ax1.axhline(100, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
    ax1.fill_between(range(len(equity)), 100, equity,
                     where=[e >= 100 for e in equity], alpha=0.3, color='green')
    ax1.fill_between(range(len(equity)), 100, equity,
                     where=[e < 100 for e in equity], alpha=0.3, color='red')
    ax1.set_xticks(range(len(equity)))
    ax1.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
    ax1.set_ylabel('Portfolio Value ($)', fontsize=10)
    ax1.set_title(f'Equity Curve | Return: {final_return:+.1f}% | Max DD: {max_dd:.1f}%', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')

    # Annotations pour les extremes
    max_idx = equity.index(max(equity))
    min_idx = equity.index(min(equity))
    ax1.annotate(f'Peak: ${max(equity):.0f}', xy=(max_idx, max(equity)),
                 xytext=(max_idx, max(equity)+20), fontsize=9,
                 arrowprops=dict(arrowstyle='->', color='green'))

    # 2. Sharpe Ratio par segment
    ax2 = axes[1]
    colors = ['green' if s["sharpe"] > 0 else 'red' for s in segments]
    bars = ax2.bar([s["id"] for s in segments], [s["sharpe"] for s in segments], color=colors, edgecolor='black', linewidth=0.5)
    ax2.axhline(0, color='black', linewidth=1)
    ax2.axhline(avg_sharpe, color='blue', linestyle='--', alpha=0.7, label=f'Avg: {avg_sharpe:.2f}')
    ax2.set_xlabel('Segment', fontsize=10)
    ax2.set_ylabel('Sharpe Ratio', fontsize=10)
    ax2.set_title(f'Sharpe Ratio par Segment | Positifs: {positive_segments}/13 ({positive_segments/13*100:.0f}%)', fontsize=11)
    ax2.set_xticks([s["id"] for s in segments])
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend(loc='upper right')

    # 3. PnL par segment
    ax3 = axes[2]
    colors_pnl = ['green' if s["pnl"] > 0 else 'red' for s in segments]
    ax3.bar([s["id"] for s in segments], [s["pnl"] for s in segments], color=colors_pnl, edgecolor='black', linewidth=0.5)
    ax3.axhline(0, color='black', linewidth=1)
    avg_pnl = np.mean([s["pnl"] for s in segments])
    ax3.axhline(avg_pnl, color='blue', linestyle='--', alpha=0.7, label=f'Avg: {avg_pnl:+.1f}%')
    ax3.set_xlabel('Segment', fontsize=10)
    ax3.set_ylabel('PnL (%)', fontsize=10)
    ax3.set_title(f'PnL par Segment | Avg: {avg_pnl:+.1f}%', fontsize=11)
    ax3.set_xticks([s["id"] for s in segments])
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.legend(loc='upper right')

    plt.tight_layout()

    # Sauvegarder
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Saved: {output_path}")

    return fig


def print_summary(segments: list):
    """Affiche un résumé des métriques."""
    equity = calculate_equity_curve(segments)
    max_dd = calculate_max_drawdown(equity)

    print("\n" + "=" * 60)
    print("WFO SUMMARY - 13 Segments")
    print("=" * 60)

    print(f"\n{'Metric':<25} {'Value':>15}")
    print("-" * 40)
    print(f"{'Initial Capital':<25} {'$100.00':>15}")
    print(f"{'Final Capital':<25} {f'${equity[-1]:.2f}':>15}")
    print(f"{'Total Return':<25} {f'{(equity[-1]/equity[0]-1)*100:+.1f}%':>15}")
    print(f"{'Max Drawdown':<25} {f'{max_dd:.1f}%':>15}")
    avg_sharpe = np.mean([s["sharpe"] for s in segments])
    avg_pnl = np.mean([s["pnl"] for s in segments])
    pos_segs = sum(1 for s in segments if s["pnl"] > 0)
    total_trades = sum(s["trades"] for s in segments)
    print(f"{'Avg Sharpe':<25} {avg_sharpe:>15.2f}")
    print(f"{'Avg PnL/Segment':<25} {avg_pnl:>+14.1f}%")
    print(f"{'Positive Segments':<25} {pos_segs}/13")
    print(f"{'Total Trades':<25} {total_trades:>15}")

    print("\n" + "-" * 60)
    print("Best Segments:")
    best = sorted(segments, key=lambda x: x["sharpe"], reverse=True)[:3]
    for s in best:
        print(f"  Seg {s['id']:2d} ({s['period']}): Sharpe {s['sharpe']:+.2f}, PnL {s['pnl']:+.1f}%")

    print("\nWorst Segments:")
    worst = sorted(segments, key=lambda x: x["sharpe"])[:3]
    for s in worst:
        print(f"  Seg {s['id']:2d} ({s['period']}): Sharpe {s['sharpe']:+.2f}, PnL {s['pnl']:+.1f}%")

    print("=" * 60)


if __name__ == "__main__":
    print_summary(SEGMENTS)
    fig = plot_wfo_equity(SEGMENTS)
    plt.show()

# -*- coding: utf-8 -*-
"""Tests for get_robust_trend_stats (Binning + SNR) in audit_pipeline."""

import json
import numpy as np
import pytest

# Import from audit_pipeline (script)
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_pipeline import (
    get_robust_trend_stats,
    to_sparkline,
    _serialize_results,
)


def test_robust_trend_converging():
    """Loss-like descending series -> CONVERGING (signal dominant over noise)."""
    x = np.linspace(10.0, 2.0, 120)
    s = get_robust_trend_stats(x.tolist(), n_bins=50)
    assert "history_50" in s
    assert "status" in s
    assert "snr" in s
    assert "emoji" in s
    assert s["status"] == "CONVERGING"
    assert s["emoji"] == "✅"
    assert len(s["history_50"]) == 50


def test_robust_trend_stable_noisy():
    """Noisy plateau -> STABLE (Noisy)."""
    rng = np.random.RandomState(123)
    y = rng.randn(100) * 0.5 + 5.0
    s = get_robust_trend_stats(y.tolist(), n_bins=50)
    assert s["status"] == "STABLE (Noisy)"
    assert s["emoji"] == "〰️"
    assert s["snr"] < 2.0


def test_robust_trend_insufficient_data():
    """Too few points -> INSUFFICIENT_DATA."""
    z = [1.0, 2.0, 3.0] * 10
    s = get_robust_trend_stats(z, n_bins=50)
    assert s["status"] == "INSUFFICIENT_DATA"
    assert "history_50" in s
    assert len(s["history_50"]) == 30
    assert "uncertainty_50" not in s
    assert "snr" not in s


def test_robust_trend_with_nan():
    """Series with NaN -> nanmean/nanstd used, no crash."""
    w = np.linspace(1.0, 5.0, 120)
    w[10:14] = np.nan
    s = get_robust_trend_stats(w.tolist(), n_bins=50)
    assert "history_50" in s
    assert s["start"] >= 0
    assert s["end"] >= 0
    assert s["peak"] >= 0


def test_sparkline_from_history_50():
    """Sparkline can be built from history_50."""
    x = np.linspace(1, 10, 60)
    s = get_robust_trend_stats(x.tolist(), n_bins=50)
    sl = to_sparkline(s["history_50"])
    assert isinstance(sl, str)
    assert len(sl) == 50


def test_serialize_results_nan_in_metrics():
    """metrics.json serialization: NaN -> null."""
    s = get_robust_trend_stats(np.linspace(1, 5, 80).tolist(), n_bins=50)
    mock = {
        "convergence": {
            "metrics": {
                "critic_loss": {
                    "history_50": s["history_50"],
                    "start": s["start"],
                    "end": float("nan"),
                    "snr": s["snr"],
                }
            }
        }
    }
    out = _serialize_results(mock)
    js = json.dumps(out)
    assert "history_50" in js
    assert "null" in js

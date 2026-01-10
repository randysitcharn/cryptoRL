# -*- coding: utf-8 -*-
"""
metrics.py - Financial performance metrics for evaluation and optimization.

Shared utility functions for calculating KPIs across the codebase.
"""

import numpy as np


def calculate_sharpe_ratio(
    returns: np.ndarray,
    periods_per_year: int = 252 * 24,
    risk_free_rate: float = 0.0
) -> float:
    """
    Calculate annualized Sharpe Ratio.

    Args:
        returns: Array of log returns.
        periods_per_year: Number of periods per year (default: hourly data = 252*24).
        risk_free_rate: Annual risk-free rate (default: 0.0).

    Returns:
        Annualized Sharpe Ratio. Returns -100.0 if non-finite.
    """
    if len(returns) < 2 or np.std(returns) == 0:
        return 0.0

    # Adjust for risk-free rate (convert annual to per-period)
    rf_per_period = risk_free_rate / periods_per_year
    excess_returns = returns - rf_per_period

    sharpe = float(np.mean(excess_returns) / np.std(returns) * np.sqrt(periods_per_year))
    return sharpe if np.isfinite(sharpe) else -100.0


def calculate_sortino_ratio(
    returns: np.ndarray,
    periods_per_year: int = 252 * 24,
    risk_free_rate: float = 0.0
) -> float:
    """
    Calculate annualized Sortino Ratio.

    Like Sharpe but only considers downside volatility.

    Args:
        returns: Array of log returns.
        periods_per_year: Number of periods per year.
        risk_free_rate: Annual risk-free rate.

    Returns:
        Annualized Sortino Ratio.
    """
    if len(returns) < 2:
        return 0.0

    rf_per_period = risk_free_rate / periods_per_year
    excess_returns = returns - rf_per_period

    # Downside deviation: std of negative returns only
    negative_returns = returns[returns < 0]
    if len(negative_returns) < 2:
        return 0.0

    downside_std = np.std(negative_returns)
    if downside_std == 0:
        return 0.0

    sortino = float(np.mean(excess_returns) / downside_std * np.sqrt(periods_per_year))
    return sortino if np.isfinite(sortino) else -100.0


def calculate_max_drawdown(nav_series: np.ndarray) -> float:
    """
    Calculate maximum drawdown.

    Args:
        nav_series: Array of NAV (Net Asset Value) values.

    Returns:
        Maximum drawdown as a positive decimal (e.g., 0.15 = 15%).
    """
    if len(nav_series) < 2:
        return 0.0

    peak = np.maximum.accumulate(nav_series)
    drawdown = (peak - nav_series) / peak
    return float(np.max(drawdown))


def calculate_total_return(nav_series: np.ndarray) -> float:
    """
    Calculate total return.

    Args:
        nav_series: Array of NAV values.

    Returns:
        Total return as a decimal (e.g., 0.25 = 25%).
    """
    if len(nav_series) < 2 or nav_series[0] == 0:
        return 0.0
    return float((nav_series[-1] - nav_series[0]) / nav_series[0])


def calculate_buy_hold_return(prices: np.ndarray) -> float:
    """
    Calculate Buy & Hold return.

    Args:
        prices: Array of prices.

    Returns:
        Buy & Hold return as a decimal (e.g., 0.25 = 25%).
    """
    if len(prices) < 2 or prices[0] == 0:
        return 0.0
    return float((prices[-1] - prices[0]) / prices[0])


def calculate_win_rate(returns: np.ndarray) -> float:
    """
    Calculate win rate (percentage of positive returns).

    Args:
        returns: Array of returns.

    Returns:
        Win rate as a decimal (e.g., 0.55 = 55%).
    """
    if len(returns) == 0:
        return 0.0
    return float(np.sum(returns > 0) / len(returns))


def calculate_profit_factor(returns: np.ndarray) -> float:
    """
    Calculate profit factor (gross profit / gross loss).

    Args:
        returns: Array of returns.

    Returns:
        Profit factor. Returns inf if no losses.
    """
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())

    if losses == 0:
        return float('inf') if gains > 0 else 0.0
    return float(gains / losses)

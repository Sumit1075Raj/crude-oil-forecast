"""
utils/helpers.py
─────────────────
Shared utility functions used across the project.
"""

import numpy as np
import pandas as pd
from typing import Union


def annualised_volatility(prices: Union[pd.Series, np.ndarray], window: int = 30) -> float:
    """Rolling annualised volatility (%) of the last `window` trading days."""
    if isinstance(prices, np.ndarray):
        prices = pd.Series(prices)
    log_ret = np.log(prices / prices.shift(1)).dropna()
    return float(log_ret.tail(window).std() * np.sqrt(252) * 100)


def brent_to_inr_per_litre(brent_usd: float, usd_inr: float,
                             refinery_margin: float = 0.15,
                             barrel_litres: float = 158.987) -> float:
    """
    Approximate Brent crude price in INR per litre at the refinery gate.
    (Excludes taxes, duties, dealer margins — for indicative purposes only.)
    """
    inr_per_barrel  = brent_usd * usd_inr
    inr_per_litre   = inr_per_barrel / barrel_litres
    return inr_per_litre * (1 + refinery_margin)


def format_price_change(current: float, previous: float) -> str:
    """Returns formatted string like '+2.34 (+1.82%)' or '-1.10 (-0.87%)'."""
    diff = current - previous
    pct  = (diff / previous) * 100 if previous != 0 else 0
    sign = "+" if diff >= 0 else ""
    return f"{sign}{diff:.2f} ({sign}{pct:.2f}%)"


def next_trading_day(date: pd.Timestamp) -> pd.Timestamp:
    """Returns the next business day after `date`."""
    return date + pd.offsets.BDay(1)


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """Division that returns `default` when b is zero."""
    return a / b if b != 0 else default


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error."""
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric MAPE — more balanced than standard MAPE."""
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2 + 1e-9
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)

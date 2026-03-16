"""
scripts/data_ingestion.py
─────────────────────────
Responsible for fetching ALL raw data needed by the forecasting pipeline.

Data Sources:
  1. Brent/WTI crude oil prices          → yfinance
  2. USD/INR exchange rate               → yfinance
  3. Macroeconomic indicators            → FRED API (fallback: synthetic)
  4. Oil supply/demand imbalance proxy   → EIA API  (fallback: synthetic)
  5. News sentiment on oil markets       → NewsAPI  (fallback: TextBlob on headlines)

Each function returns a pandas DataFrame with a DatetimeIndex named 'date'.
"""

import sys
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from textblob import TextBlob

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    FRED_API_KEY, EIA_API_KEY, NEWS_API_KEY,
    START_DATE, BRENT_TICKER, WTI_TICKER, DATA_RAW
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. CRUDE OIL PRICES (Brent + WTI via yfinance)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_oil_prices(start: str = START_DATE) -> pd.DataFrame:
    """
    Download daily Brent and WTI crude oil prices from Yahoo Finance.

    Returns
    -------
    DataFrame with columns: [open, high, low, close, volume, wti_close]
    Index: DatetimeIndex named 'date'
    """
    log.info("Fetching Brent crude prices from Yahoo Finance …")
    try:
        brent = yf.download(BRENT_TICKER, start=start, progress=False, auto_adjust=True)
        wti   = yf.download(WTI_TICKER,   start=start, progress=False, auto_adjust=True)

        if brent.empty:
            raise ValueError("Brent data empty — using synthetic fallback")

        # Flatten MultiIndex columns if present
        if isinstance(brent.columns, pd.MultiIndex):
            brent.columns = [c[0].lower() for c in brent.columns]
        else:
            brent.columns = [c.lower() for c in brent.columns]

        if isinstance(wti.columns, pd.MultiIndex):
            wti.columns = [c[0].lower() for c in wti.columns]
        else:
            wti.columns = [c.lower() for c in wti.columns]

        brent.index.name = "date"
        brent["wti_close"] = wti["close"].reindex(brent.index).ffill()
        brent = brent[["open", "high", "low", "close", "volume", "wti_close"]].copy()
        brent = brent.dropna(subset=["close"])
        log.info(f"  ✓ Brent data: {len(brent)} rows, {brent.index[0].date()} → {brent.index[-1].date()}")
        return brent

    except Exception as e:
        log.warning(f"  ✗ yfinance failed ({e}). Generating synthetic oil price data.")
        return _synthetic_oil_prices(start)


def _synthetic_oil_prices(start: str) -> pd.DataFrame:
    """Generates realistic-looking crude oil price data for offline/demo use."""
    np.random.seed(42)
    idx   = pd.bdate_range(start=start, end=datetime.today())
    n     = len(idx)
    # Geometric Brownian Motion
    daily_ret  = np.random.normal(0.0002, 0.02, n)
    price      = 60.0 * np.exp(np.cumsum(daily_ret))
    price      = np.clip(price, 20, 150)
    df = pd.DataFrame({
        "open":      price * (1 + np.random.uniform(-0.005, 0.005, n)),
        "high":      price * (1 + np.random.uniform(0.005, 0.02,  n)),
        "low":       price * (1 - np.random.uniform(0.005, 0.02,  n)),
        "close":     price,
        "volume":    np.random.randint(200_000, 600_000, n).astype(float),
        "wti_close": price * np.random.uniform(0.90, 0.98, n),
    }, index=idx)
    df.index.name = "date"
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. USD/INR EXCHANGE RATE
# ─────────────────────────────────────────────────────────────────────────────

def fetch_usd_inr(start: str = START_DATE) -> pd.Series:
    """
    Fetch USD→INR exchange rate. India imports ~85% of crude; the rupee
    directly amplifies or dampens the domestic price impact.
    """
    log.info("Fetching USD/INR exchange rate …")
    try:
        fx = yf.download("INR=X", start=start, progress=False, auto_adjust=True)
        if isinstance(fx.columns, pd.MultiIndex):
            fx.columns = [c[0].lower() for c in fx.columns]
        else:
            fx.columns = [c.lower() for c in fx.columns]
        fx.index.name = "date"
        series = fx["close"].rename("usd_inr").dropna()
        log.info(f"  ✓ USD/INR: {len(series)} rows")
        return series
    except Exception as e:
        log.warning(f"  ✗ USD/INR fetch failed ({e}). Using synthetic.")
        idx = pd.bdate_range(start=start, end=datetime.today())
        vals = 70 + np.cumsum(np.random.normal(0, 0.05, len(idx)))
        return pd.Series(vals, index=idx, name="usd_inr")


# ─────────────────────────────────────────────────────────────────────────────
# 3. MACROECONOMIC INDICATORS — FRED API
# ─────────────────────────────────────────────────────────────────────────────

FRED_SERIES = {
    "us_cpi":          "CPIAUCSL",   # US Consumer Price Index
    "us_interest_rate": "FEDFUNDS",  # Federal Funds Rate
    "us_gdp_growth":   "A191RL1Q225SBEA",  # US Real GDP Growth
    "dxy":             "DTWEXBGS",   # US Dollar Index
    "us_10yr_yield":   "DGS10",      # 10-Year Treasury Yield
}


def fetch_fred_macro(start: str = START_DATE) -> pd.DataFrame:
    """
    Pull macroeconomic series from FRED. These influence global oil demand.
    Falls back to synthetic data if API key is missing.
    """
    if not FRED_API_KEY:
        log.warning("  ✗ FRED_API_KEY not set. Using synthetic macro data.")
        return _synthetic_macro(start)

    log.info("Fetching macroeconomic indicators from FRED …")
    frames = {}
    base_url = "https://api.stlouisfed.org/fred/series/observations"

    for name, series_id in FRED_SERIES.items():
        try:
            params = {
                "series_id":       series_id,
                "api_key":         FRED_API_KEY,
                "file_type":       "json",
                "observation_start": start,
                "frequency":       "d",    # request daily (FRED auto-fills)
            }
            r = requests.get(base_url, params=params, timeout=15)
            r.raise_for_status()
            obs = r.json()["observations"]
            s = pd.Series(
                {o["date"]: float(o["value"]) for o in obs if o["value"] != "."},
                name=name
            )
            s.index = pd.to_datetime(s.index)
            s.index.name = "date"
            frames[name] = s
            log.info(f"  ✓ {name}: {len(s)} rows")
            time.sleep(0.2)   # be polite to FRED
        except Exception as e:
            log.warning(f"  ✗ FRED {series_id} failed: {e}")

    if not frames:
        return _synthetic_macro(start)

    macro = pd.DataFrame(frames)
    macro = macro.resample("B").last().ffill().bfill()
    return macro


def _synthetic_macro(start: str) -> pd.DataFrame:
    np.random.seed(7)
    idx = pd.bdate_range(start=start, end=datetime.today())
    n   = len(idx)
    return pd.DataFrame({
        "us_cpi":           200 + np.cumsum(np.random.normal(0.02, 0.1, n)),
        "us_interest_rate": np.clip(2 + np.cumsum(np.random.normal(0, 0.01, n)), 0, 10),
        "us_gdp_growth":    np.random.normal(2.5, 1.0, n),
        "dxy":              100 + np.cumsum(np.random.normal(0, 0.1, n)),
        "us_10yr_yield":    np.clip(3 + np.cumsum(np.random.normal(0, 0.02, n)), 0.5, 8),
    }, index=idx)


# ─────────────────────────────────────────────────────────────────────────────
# 4. NEWS SENTIMENT
# ─────────────────────────────────────────────────────────────────────────────

OIL_HEADLINES_FALLBACK = [
    "OPEC+ agrees to extend oil production cuts through next quarter",
    "Global oil demand surges as economic recovery gains momentum",
    "US shale production hits record high, pressuring oil prices",
    "Middle East tensions spark crude oil price spike",
    "China's industrial output growth boosts oil consumption forecasts",
    "IEA warns of global oil oversupply in 2024",
    "India's crude oil imports rise as fuel demand increases",
    "Federal Reserve rate hike strengthens dollar, weighs on oil",
]


def fetch_news_sentiment(lookback_days: int = 30) -> pd.Series:
    """
    Fetch oil-related news and compute daily sentiment scores using TextBlob.
    Uses NewsAPI if key is present; otherwise falls back to synthetic scores.

    Returns: pd.Series with DatetimeIndex named 'date', values in [-1, 1]
    """
    log.info("Fetching news sentiment …")
    end_date   = datetime.today()
    start_date = end_date - timedelta(days=lookback_days)

    if NEWS_API_KEY:
        try:
            from newsapi import NewsApiClient
            api  = NewsApiClient(api_key=NEWS_API_KEY)
            arts = api.get_everything(
                q          ="crude oil OR Brent crude OR OPEC OR oil price",
                language   ="en",
                from_param =start_date.strftime("%Y-%m-%d"),
                to         =end_date.strftime("%Y-%m-%d"),
                sort_by    ="publishedAt",
                page_size  =100,
            )
            articles = arts.get("articles", [])
            records  = []
            for a in articles:
                dt    = pd.to_datetime(a["publishedAt"]).normalize()
                text  = f"{a.get('title','')} {a.get('description','')}"
                score = TextBlob(text).sentiment.polarity
                records.append((dt, score))
            if records:
                df = pd.DataFrame(records, columns=["date", "sentiment"])
                df = df.set_index("date").resample("B").mean().fillna(0)
                log.info(f"  ✓ News sentiment: {len(df)} days")
                return df["sentiment"].rename("sentiment_score")
        except Exception as e:
            log.warning(f"  ✗ NewsAPI failed: {e}")

    # Fallback: synthetic sentiment based on a random walk
    log.warning("  Using synthetic sentiment scores.")
    idx   = pd.bdate_range(start=start_date, end=end_date)
    scores = np.clip(np.cumsum(np.random.normal(0, 0.05, len(idx))), -1, 1)
    return pd.Series(scores, index=idx, name="sentiment_score")


# ─────────────────────────────────────────────────────────────────────────────
# 4b. EIA OIL SUPPLY / DEMAND DATA
# ─────────────────────────────────────────────────────────────────────────────

def fetch_eia_data(start: str = START_DATE) -> pd.DataFrame:
    """
    Fetch US crude oil supply/demand indicators from the EIA v2 API.

    Series used:
      PET.WCRSTUS1.W  — US crude oil stocks (weekly, thousand barrels)
      PET.WCRFPUS2.W  — US crude oil field production (weekly, thousand bbl/day)

    These act as global supply proxies — when US stocks build, it signals
    oversupply and typically pressures Brent prices lower.
    """
    if not EIA_API_KEY:
        log.warning("  ✗ EIA_API_KEY not set. Skipping EIA data.")
        return pd.DataFrame()

    log.info("Fetching EIA crude oil supply/demand data …")

    eia_series = {
        "us_crude_stocks":      "PET.WCRSTUS1.W",
        "us_crude_production":  "PET.WCRFPUS2.W",
    }

    frames = {}
    base   = "https://api.eia.gov/v2/seriesid/{series}"

    for name, series_id in eia_series.items():
        try:
            url    = base.format(series=series_id)
            params = {"api_key": EIA_API_KEY, "frequency": "weekly",
                      "start": start, "sort[0][column]": "period",
                      "sort[0][direction]": "asc", "length": 5000}
            r = requests.get(url, params=params, timeout=20)
            r.raise_for_status()
            data = r.json()
            rows = data.get("response", {}).get("data", [])
            if not rows:
                log.warning(f"  ✗ EIA {series_id}: no data returned")
                continue
            s = pd.Series(
                {row["period"]: float(row["value"]) for row in rows
                 if row.get("value") not in (None, "")},
                name=name,
            )
            s.index = pd.to_datetime(s.index)
            s.index.name = "date"
            frames[name] = s
            log.info(f"  ✓ EIA {name}: {len(s)} rows")
            time.sleep(0.3)
        except Exception as e:
            log.warning(f"  ✗ EIA {series_id} failed: {e}")

    if not frames:
        return pd.DataFrame()

    eia_df = pd.DataFrame(frames)
    eia_df = eia_df.resample("B").last().ffill().bfill()
    # Derived feature: week-on-week stock change (positive = building glut)
    if "us_crude_stocks" in eia_df.columns:
        eia_df["us_stocks_change"] = eia_df["us_crude_stocks"].diff()
    return eia_df


# ─────────────────────────────────────────────────────────────────────────────
# 5. MASTER DATA INGESTION
# ─────────────────────────────────────────────────────────────────────────────

def ingest_all(start: str = START_DATE, save: bool = True) -> pd.DataFrame:
    """
    Orchestrates all data fetching and merges everything into one
    aligned daily DataFrame. Missing values are forward-filled then
    backward-filled (standard practice for financial time series).

    Parameters
    ----------
    start : str   ISO date string, default from config
    save  : bool  If True, save raw CSV to data/raw/

    Returns
    -------
    Merged DataFrame with DatetimeIndex 'date'
    """
    oil   = fetch_oil_prices(start)
    fx    = fetch_usd_inr(start)
    macro = fetch_fred_macro(start)
    eia   = fetch_eia_data(start)
    sent  = fetch_news_sentiment(lookback_days=365)

    # Merge on date index — outer join then forward-fill
    df = oil.copy()
    df = df.join(fx,    how="left")
    df = df.join(macro, how="left")
    if not eia.empty:
        df = df.join(eia, how="left")
    df = df.join(sent,  how="left")

    # Align all to business day frequency
    df = df.resample("B").last()
    df = df.ffill().bfill()

    # Fill sentiment with 0 for older dates (no news data)
    if "sentiment_score" in df.columns:
        df["sentiment_score"] = df["sentiment_score"].fillna(0.0)

    log.info(f"\n{'─'*50}")
    log.info(f"  Final dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    log.info(f"  Date range   : {df.index[0].date()} → {df.index[-1].date()}")
    log.info(f"  Columns      : {list(df.columns)}")
    log.info(f"{'─'*50}\n")

    if save:
        path = DATA_RAW / "raw_merged.csv"
        df.to_csv(path)
        log.info(f"  Saved raw data → {path}")

    return df


if __name__ == "__main__":
    df = ingest_all()
    print(df.tail())

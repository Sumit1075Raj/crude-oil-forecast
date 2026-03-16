"""
config.py — Central configuration for all modules.

All hyperparameters, paths, and API settings live here so that
any student can tweak the system without hunting through files.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent
DATA_RAW        = BASE_DIR / "data" / "raw"
DATA_PROCESSED  = BASE_DIR / "data" / "processed"
MODELS_DIR      = BASE_DIR / "models"
NOTEBOOKS_DIR   = BASE_DIR / "notebooks"

# Create directories if they don't exist
for d in [DATA_RAW, DATA_PROCESSED, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── API Keys (loaded from .env) ───────────────────────────────────────────────
FRED_API_KEY  = os.getenv("FRED_API_KEY", "")
EIA_API_KEY   = os.getenv("EIA_API_KEY", "")
NEWS_API_KEY  = os.getenv("NEWS_API_KEY", "")

# ── Data settings ─────────────────────────────────────────────────────────────
START_DATE      = "2000-01-01"          # Historical data start
TARGET_COLUMN   = "close"               # Column we predict
BRENT_TICKER    = "BZ=F"               # Brent crude Yahoo Finance ticker
WTI_TICKER      = "CL=F"               # WTI crude Yahoo Finance ticker

# ── Feature Engineering ───────────────────────────────────────────────────────
LAG_PERIODS         = [1, 3, 7, 14, 30]
ROLLING_WINDOWS     = [7, 14, 30]

# ── Model Hyperparameters ─────────────────────────────────────────────────────
LOOKBACK_WINDOW     = int(os.getenv("LOOKBACK_WINDOW", 60))    # LSTM sequence length
FORECAST_HORIZON    = int(os.getenv("FORECAST_HORIZON", 30))   # Days to forecast
TRAIN_TEST_SPLIT    = float(os.getenv("TRAIN_TEST_SPLIT", 0.85))
RANDOM_STATE        = int(os.getenv("RANDOM_STATE", 42))

# LSTM architecture
LSTM_UNITS          = [128, 64]
LSTM_DROPOUT        = 0.2
LSTM_EPOCHS         = 50
LSTM_BATCH_SIZE     = 32
LSTM_LEARNING_RATE  = 0.001

# Random Forest
RF_N_ESTIMATORS     = 300
RF_MAX_DEPTH        = 15
RF_MIN_SAMPLES_LEAF = 2

# XGBoost
XGB_N_ESTIMATORS    = 300
XGB_MAX_DEPTH       = 6
XGB_LEARNING_RATE   = 0.05
XGB_SUBSAMPLE       = 0.8

# Ensemble weights: [LSTM, RF, XGB]
ENSEMBLE_WEIGHTS    = [0.45, 0.25, 0.30]

# ── Dashboard settings ────────────────────────────────────────────────────────
REFRESH_INTERVAL_SECONDS = 300         # Auto-refresh every 5 minutes
DASHBOARD_TITLE          = "🛢️ CrudeEdge — India Oil Price Intelligence"

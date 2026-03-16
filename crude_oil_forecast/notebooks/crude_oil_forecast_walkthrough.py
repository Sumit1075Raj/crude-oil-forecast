"""
notebooks/crude_oil_forecast_walkthrough.py
─────────────────────────────────────────────
Complete project walkthrough notebook (Jupytext format).
Convert to .ipynb with:   jupytext --to notebook notebooks/crude_oil_forecast_walkthrough.py

Or open directly in VS Code with the Jupytext extension.
"""

# %% [markdown]
# # 🛢️ CrudeEdge — India Crude Oil Price Forecasting System
# ## Complete Walkthrough: Data → Features → Hybrid Model → Dashboard
#
# **Author:** BTech Major Project  
# **Objective:** Predict Brent crude oil prices affecting India using a hybrid
# LSTM + Random Forest + XGBoost ensemble, and forecast the next 30 days.
#
# ---
# ### Architecture Overview
# ```
# Raw Data (yfinance + FRED + NewsAPI)
#      ↓
# Feature Engineering (60+ features)
#      ↓
# ┌─────────┬──────────────────┬─────────┐
# │  LSTM   │  Random Forest   │ XGBoost │
# │ (45%)   │     (25%)        │  (30%)  │
# └─────────┴──────────────────┴─────────┘
#      ↓ Weighted Ensemble
# Price Prediction + Direction Prediction
#      ↓
# Streamlit Dashboard
# ```

# %% [markdown]
# ## 1. Setup & Imports

# %%
import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "..")   # Add project root to path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta

# Project modules
from config import *

print("✅ All imports successful")
print(f"📂 Project root: {BASE_DIR}")
print(f"🎯 Target column: {TARGET_COLUMN}")
print(f"📅 Data from: {START_DATE}")

# %% [markdown]
# ## 2. Data Ingestion
#
# We pull data from multiple sources and merge them into one aligned DataFrame.
# If API keys are not set, synthetic fallback data is used automatically.

# %%
from scripts.data_ingestion import (
    fetch_oil_prices, fetch_usd_inr, fetch_fred_macro,
    fetch_news_sentiment, ingest_all
)

# Fetch a small sample for illustration
print("🔽 Fetching Brent crude oil prices …")
oil = fetch_oil_prices(start="2015-01-01")
print(oil.tail())

# %%
# Full merged dataset
print("\n🔽 Fetching all data sources and merging …")
raw_df = ingest_all(start="2010-01-01", save=True)
print(f"\nShape: {raw_df.shape}")
print(raw_df.describe().round(2))

# %% [markdown]
# ## 3. Exploratory Data Analysis

# %%
fig, axes = plt.subplots(3, 1, figsize=(16, 12), facecolor="#0d1117")
colors     = ["#00d4ff", "#39d353", "#e3b341"]

for ax, col, color in zip(axes,
                           ["close", "volume", "usd_inr"],
                           colors):
    if col in raw_df.columns:
        ax.plot(raw_df.index, raw_df[col], color=color, linewidth=0.8)
        ax.set_facecolor("#0d1117")
        ax.set_title(col.upper(), color=color, fontsize=12)
        ax.tick_params(colors="#8b949e")
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(color="#21262d", alpha=0.5)

plt.suptitle("Raw Data Overview", color="#c9d1d9", fontsize=14, y=1.01)
plt.tight_layout()
plt.show()

# %%
# Distribution of daily returns
returns = raw_df["close"].pct_change().dropna()
fig, ax = plt.subplots(figsize=(10, 5), facecolor="#0d1117")
ax.set_facecolor("#0d1117")
ax.hist(returns, bins=100, color="#00d4ff", alpha=0.7, edgecolor="none")
ax.axvline(returns.mean(), color="#e3b341", linestyle="--", label=f"Mean: {returns.mean():.4f}")
ax.axvline(returns.std(),  color="#f85149", linestyle="--", label=f"Std:  {returns.std():.4f}")
ax.set_title("Daily Return Distribution (Fat Tails = Oil Market Typical)", color="#c9d1d9")
ax.tick_params(colors="#8b949e")
ax.legend(facecolor="#161b22", edgecolor="#21262d")
plt.tight_layout()
plt.show()

print(f"Skewness: {returns.skew():.3f}")
print(f"Kurtosis: {returns.kurt():.3f}  (> 3 = fat tails)")

# %% [markdown]
# ## 4. Feature Engineering
#
# We create 60+ features from the raw price series:
# - **Lag features**: what was the price N days ago?
# - **Rolling stats**: moving averages, standard deviation, skewness
# - **Technical indicators**: RSI, MACD, Bollinger Bands
# - **Momentum**: Rate of Change (ROC), price momentum
# - **India-specific**: Brent price in INR (Brent USD × USD/INR)

# %%
from features.feature_engineering import (
    build_features, get_feature_columns, split_dataset,
    scale_features, scale_target, build_sequences,
    run_feature_pipeline
)

feat_df   = build_features(raw_df)
feat_cols = get_feature_columns(feat_df)

print(f"Total features: {len(feat_cols)}")
print(f"\nSample features:")
for f in feat_cols[:20]:
    print(f"  • {f}")
print("  ...")

# %%
# Correlation heatmap of key features vs target
corr_cols = feat_cols[:20] + ["log_return"]
corr      = feat_df[corr_cols].corr()["log_return"].drop("log_return").sort_values()

fig, ax = plt.subplots(figsize=(10, 8), facecolor="#0d1117")
ax.set_facecolor("#0d1117")
colors   = ["#f85149" if v < 0 else "#39d353" for v in corr.values]
ax.barh(corr.index, corr.values, color=colors, alpha=0.8)
ax.set_title("Feature Correlation with Log Return (Target)", color="#c9d1d9")
ax.axvline(0, color="#8b949e", linewidth=0.8)
ax.tick_params(colors="#8b949e")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Train/Test Split & Scaling
#
# ⚠️ **Critical Rule**: Never shuffle time-series data.
# The train set = past, the test set = future.

# %%
pipeline = run_feature_pipeline(raw_df, split_ratio=TRAIN_TEST_SPLIT)

print("✅ Pipeline complete")
print(f"  Train rows:         {len(pipeline['train'])}")
print(f"  Test rows:          {len(pipeline['test'])}")
print(f"  Feature count:      {len(pipeline['feat_cols'])}")
print(f"  LSTM train shape:   {pipeline['X_lstm_train'].shape}")
print(f"  LSTM test shape:    {pipeline['X_lstm_test'].shape}")
print(f"\n  Train period: {pipeline['train'].index[0].date()} → {pipeline['train'].index[-1].date()}")
print(f"  Test  period: {pipeline['test'].index[0].date()}  → {pipeline['test'].index[-1].date()}")

# %% [markdown]
# ## 6. Model Training
#
# ### 6a. LSTM Neural Network
# The LSTM sees sequences of 60 trading days and learns long-range dependencies.

# %%
from scripts.train_models import (
    build_lstm, train_lstm, train_random_forest, train_xgboost,
    train_direction_models, ensemble_predict_log_returns, log_returns_to_prices
)

# Build and summarise the architecture
model_summary = build_lstm(input_shape=(LOOKBACK_WINDOW, len(pipeline["feat_cols"])))
model_summary.summary()

# %%
# NOTE: Uncomment the following to actually train.
# Training takes ~10 minutes on CPU, ~2 minutes on GPU.

# trained = {}
# val_split = int(len(pipeline["X_lstm_train"]) * 0.9)
# lstm, history = train_lstm(
#     pipeline["X_lstm_train"][:val_split],
#     pipeline["y_lstm_train"][:val_split],
#     pipeline["X_lstm_train"][val_split:],
#     pipeline["y_lstm_train"][val_split:],
# )
# trained["lstm"] = lstm

# %% [markdown]
# ### 6b. Random Forest Regressor

# %%
# rf = train_random_forest(
#     pipeline["X_train"][LOOKBACK_WINDOW:],
#     pipeline["y_train_log"][LOOKBACK_WINDOW:],
# )
# trained["rf"] = rf

# %% [markdown]
# ### 6c. XGBoost Regressor

# %%
# xgb = train_xgboost(
#     pipeline["X_train"][LOOKBACK_WINDOW:],
#     pipeline["y_train_log"][LOOKBACK_WINDOW:],
#     pipeline["X_test"][:int(len(pipeline["X_test"])*0.5)],
#     pipeline["y_test_log"][:int(len(pipeline["y_test_log"])*0.5)],
# )
# trained["xgb"] = xgb

# %% [markdown]
# ### 6d. Run Full Training (One Command)

# %%
# To train ALL models at once, run from terminal:
#   python run_pipeline.py
#
# Or in fast mode (skip LSTM):
#   python run_pipeline.py --fast

# %% [markdown]
# ## 7. Evaluation
#
# We load pre-trained models from disk and evaluate them.

# %%
import joblib, os

def load_if_exists(path):
    return joblib.load(path) if Path(path).exists() else None

rf_model  = load_if_exists(MODELS_DIR / "random_forest.pkl")
xgb_model = load_if_exists(MODELS_DIR / "xgboost.pkl")
feat_sc   = load_if_exists(MODELS_DIR / "feat_scaler.pkl")
log_sc    = load_if_exists(MODELS_DIR / "log_scaler.pkl")

available = [n for n, m in [("RF", rf_model), ("XGB", xgb_model),
                              ("FeatScaler", feat_sc), ("LogScaler", log_sc)]
             if m is not None]
print("Loaded:", available or "No pre-trained models found — run run_pipeline.py first")

# %%
# Evaluation demo (requires trained models)
if rf_model and feat_sc and log_sc:
    from scripts.evaluate import regression_metrics, directional_accuracy
    from features.feature_engineering import get_feature_columns

    feat_cols = get_feature_columns(pipeline["df"])
    lk        = LOOKBACK_WINDOW

    X_test    = feat_sc.transform(pipeline["test"][feat_cols])
    y_test_log_raw = pipeline["test"]["log_return"].values

    # RF predictions
    rf_log_scaled  = rf_model.predict(X_test[lk:])
    last_price     = float(pipeline["test"]["close"].iloc[lk - 1])
    rf_prices      = log_returns_to_prices(rf_log_scaled, last_price, log_sc)
    actual_prices  = pipeline["y_test_price"][lk:]

    m = regression_metrics(actual_prices[:len(rf_prices)], rf_prices, "Random Forest")
    da = directional_accuracy(actual_prices[:len(rf_prices)], rf_prices)
    print(f"\nDirectional Accuracy: {da:.4f} ({da*100:.2f}%)")

# %% [markdown]
# ## 8. Visualisations

# %%
if rf_model and feat_sc and log_sc:
    fig, ax = plt.subplots(figsize=(16, 6), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    n   = min(200, len(actual_prices), len(rf_prices))
    idx = pipeline["test"].index[lk:lk+n]

    ax.plot(idx, actual_prices[:n], color="#c9d1d9", linewidth=1.5, label="Actual")
    ax.plot(idx, rf_prices[:n],     color="#00d4ff", linewidth=1.2, linestyle="--",
            alpha=0.9, label="Random Forest")

    ax.set_title("Actual vs Predicted Brent Crude Oil Price", color="#c9d1d9", fontsize=13)
    ax.set_ylabel("Price (USD/bbl)", color="#8b949e")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0f"))
    ax.tick_params(colors="#8b949e")
    ax.legend(facecolor="#161b22", edgecolor="#21262d")
    ax.grid(color="#21262d", alpha=0.5)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 9. Feature Importance Analysis

# %%
if xgb_model:
    imp        = xgb_model.feature_importances_
    feat_cols2 = get_feature_columns(pipeline["df"])
    top        = pd.Series(imp, index=feat_cols2).nlargest(20).sort_values()

    fig, ax = plt.subplots(figsize=(10, 8), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    colors = ["#e3b341" if i >= 15 else "#00d4ff" for i in range(len(top))]
    ax.barh(top.index, top.values, color=colors, alpha=0.85)
    ax.set_title("XGBoost Feature Importance — Top 20", color="#c9d1d9", fontsize=13)
    ax.tick_params(colors="#8b949e")
    ax.grid(color="#21262d", alpha=0.5, axis="x")
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 10. 30-Day Forecast

# %%
# Load LSTM if available
lstm_model = None
try:
    import tensorflow as tf
    lstm_path = MODELS_DIR / "lstm_final.keras"
    if lstm_path.exists():
        lstm_model = tf.keras.models.load_model(str(lstm_path))
        print("✅ LSTM loaded")
    else:
        print("⚠️  LSTM not found — train first with run_pipeline.py")
except ImportError:
    print("⚠️  TensorFlow not installed")

# %%
if lstm_model and rf_model and xgb_model and feat_sc and log_sc:
    from scripts.train_models import forecast_30_days

    feat_cols3 = get_feature_columns(pipeline["df"])
    X_all      = feat_sc.transform(pipeline["df"][feat_cols3])
    last_seq   = X_all[-LOOKBACK_WINDOW:]
    last_flat  = X_all[-1]
    last_price = float(pipeline["df"]["close"].iloc[-1])

    forecast = forecast_30_days(
        lstm_model, rf_model, xgb_model,
        last_seq, last_flat, last_price, log_sc
    )

    last_date   = pipeline["df"].index[-1]
    fore_dates  = pd.bdate_range(
        start=last_date + timedelta(days=1), periods=FORECAST_HORIZON
    )

    print("30-Day Forecast:")
    for d, p in zip(fore_dates[:10], forecast[:10]):
        change = ((p - last_price) / last_price) * 100
        print(f"  {d.date()}  ${p:.2f}  ({change:+.2f}%)")
    print("  …")

# %% [markdown]
# ## 11. Launch Dashboard
#
# ```bash
# streamlit run dashboard/app.py
# ```
#
# The dashboard provides:
# - Live price ticker (refreshes every 5 min)
# - Predicted vs actual price chart
# - 30-day forecast with confidence bands
# - Feature importance heatmap
# - Model comparison metrics
# - Retrain button

# %% [markdown]
# ---
# ## Summary
#
# | Component | Details |
# |---|---|
# | Data Sources | yfinance (Brent/WTI/USD-INR), FRED API, NewsAPI |
# | Features | 60+ engineered features (lags, rolling, RSI, MACD, BB) |
# | LSTM | 128→64 units, Dropout 0.2, Huber loss, log-return target |
# | Random Forest | 300 trees, OOB validation, feature importance |
# | XGBoost | 300 rounds, early stopping, L1+L2 regularisation |
# | Ensemble | Weighted avg: LSTM 45% + RF 25% + XGB 30% |
# | Direction Model | RF + XGB classifiers (UP/DOWN) |
# | Dashboard | Streamlit + Plotly, dark industrial theme |
#
# ### How to Run the Full System
#
# ```bash
# # 1. Install dependencies
# pip install -r requirements.txt
#
# # 2. Add API keys (optional)
# cp .env.example .env
# # Edit .env with your FRED/EIA/NewsAPI keys
#
# # 3. Run full pipeline (train + evaluate)
# python run_pipeline.py
#
# # 4. Launch dashboard
# streamlit run dashboard/app.py
# ```

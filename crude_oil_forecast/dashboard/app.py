"""
dashboard/app.py
─────────────────
CrudeEdge — India Oil Price Intelligence Dashboard

A full Streamlit dashboard with:
  • Live price ticker
  • Predicted vs actual prices (interactive Plotly chart)
  • 30-day forecast with confidence bands
  • Feature importance
  • Model performance metrics
  • Directional accuracy meter
  • Auto-refresh every 5 minutes
  • Retrain button

Run with:
    streamlit run dashboard/app.py
"""

import sys
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import joblib

# ── Force working directory to project root so relative paths resolve ──────
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
import os
os.chdir(_PROJECT_ROOT)
sys.path.insert(0, str(_PROJECT_ROOT))


def sanitize(arr):
    """Replace inf/-inf with NaN, forward-fill, then drop remaining NaN."""
    s = pd.Series(np.array(arr, dtype=np.float64))
    s = s.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    return s.values
from config import (
    MODELS_DIR, DATA_PROCESSED, DATA_RAW,
    ENSEMBLE_WEIGHTS, FORECAST_HORIZON, LOOKBACK_WINDOW,
    REFRESH_INTERVAL_SECONDS, DASHBOARD_TITLE, TARGET_COLUMN
)

logging.basicConfig(level=logging.WARNING)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  ="CrudeEdge | Oil Intelligence",
    page_icon   ="🛢️",
    layout      ="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — Dark industrial terminal aesthetic
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Rajdhani:wght@400;500;600;700&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif;
    background-color: #060a10;
    color: #c9d1d9;
}

/* Header */
.main-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.6rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    color: #00d4ff;
    text-transform: uppercase;
    margin-bottom: 0;
    text-shadow: 0 0 20px rgba(0, 212, 255, 0.4);
}
.sub-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: #8b949e;
    letter-spacing: 0.15em;
    margin-top: 0;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    transition: border-color 0.3s;
}
.metric-card:hover { border-color: #00d4ff; }
.metric-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #8b949e;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
.metric-value {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #c9d1d9;
    line-height: 1;
}
.metric-delta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    margin-top: 0.2rem;
}
.up   { color: #39d353; }
.down { color: #f85149; }
.neutral { color: #e3b341; }

/* Section headers */
.section-header {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.1rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    color: #00d4ff;
    text-transform: uppercase;
    border-bottom: 1px solid #21262d;
    padding-bottom: 0.4rem;
    margin: 1.2rem 0 0.8rem 0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #0d1117;
    border-right: 1px solid #21262d;
}

/* Streamlit elements override */
div[data-testid="stMetricValue"] { font-family: 'Rajdhani', sans-serif !important; }
.stButton>button {
    background: linear-gradient(135deg, #1f6feb, #0d419d);
    color: white;
    border: none;
    border-radius: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 0.05em;
    padding: 0.5rem 1.2rem;
    transition: all 0.2s;
}
.stButton>button:hover { opacity: 0.85; transform: translateY(-1px); }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY THEME
# ─────────────────────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor = "#060a10",
    plot_bgcolor  = "#0d1117",
    font          = dict(family="JetBrains Mono", color="#c9d1d9"),
    xaxis         = dict(gridcolor="#21262d", linecolor="#21262d", showgrid=True),
    yaxis         = dict(gridcolor="#21262d", linecolor="#21262d", showgrid=True),
    legend        = dict(bgcolor="#161b22", bordercolor="#21262d"),
    margin        = dict(l=40, r=30, t=50, b=40),
)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=REFRESH_INTERVAL_SECONDS, show_spinner="Fetching latest oil data …")
def load_data():
    """Load (or re-fetch) raw merged dataset."""
    raw_path = DATA_RAW / "raw_merged.csv"
    if raw_path.exists():
        df = pd.read_csv(raw_path, index_col="date", parse_dates=True)
        return df
    # Fresh fetch
    from scripts.data_ingestion import ingest_all
    return ingest_all(save=True)


@st.cache_data(ttl=REFRESH_INTERVAL_SECONDS, show_spinner="Engineering features …")
def load_features():
    """Load processed feature matrix."""
    feat_path = DATA_PROCESSED / "features.csv"
    if feat_path.exists():
        df = pd.read_csv(feat_path, index_col="date", parse_dates=True)
        return df
    return None


@st.cache_resource(show_spinner="Loading ML models …")
def load_models():
    """Load all trained model files from disk."""
    models = {}
    try:
        import tensorflow as tf
        lstm_path = MODELS_DIR / "lstm_final.keras"
        if lstm_path.exists():
            models["lstm"] = tf.keras.models.load_model(str(lstm_path))
    except Exception as e:
        pass  # LSTM optional — RF+XGB still work

    for name, fname in [
        ("rf",           "random_forest.pkl"),
        ("xgb",          "xgboost.pkl"),
        ("rf_clf",       "direction_rf.pkl"),
        ("xgb_clf",      "direction_xgb.pkl"),
        ("feat_scaler",  "feat_scaler.pkl"),
        ("price_scaler", "price_scaler.pkl"),
    ]:
        path = MODELS_DIR / fname
        if path.exists():
            try:
                models[name] = joblib.load(path)
            except Exception:
                pass

    return models


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_ensemble_predictions(feat_df, models):
    """
    Run ensemble inference on the test portion of the feature dataset.
    Works with or without LSTM (fast mode).
    Returns dates, actual prices, ensemble predicted prices.
    """
    from features.feature_engineering import get_feature_columns
    from config import TRAIN_TEST_SPLIT

    feat_cols    = get_feature_columns(feat_df)
    feat_scaler  = models.get("feat_scaler")
    price_scaler = models.get("price_scaler")

    if feat_scaler is None or price_scaler is None:
        return None, None, None
    if "rf" not in models and "xgb" not in models:
        return None, None, None

    lk       = LOOKBACK_WINDOW
    split    = int(len(feat_df) * TRAIN_TEST_SPLIT)
    test_df  = feat_df.iloc[split:]
    X_test   = feat_scaler.transform(test_df[feat_cols])

    # Flat predictions (RF + XGB predict raw USD price)
    X_flat   = X_test[lk:]
    actual   = test_df["next_close"].values[lk:]

    preds = []
    weights_used = []

    if "lstm" in models:
        from features.feature_engineering import build_sequences
        y_sc             = price_scaler.transform(test_df["next_close"].values.reshape(-1,1)).ravel()
        X_seq, _         = build_sequences(X_test, y_sc, lk)
        lstm_sc          = models["lstm"].predict(X_seq, verbose=0).ravel()
        lstm_usd         = price_scaler.inverse_transform(lstm_sc.reshape(-1,1)).ravel()
        preds.append(lstm_usd)
        weights_used.append(ENSEMBLE_WEIGHTS[0])

    if "rf" in models:
        preds.append(models["rf"].predict(X_flat))
        weights_used.append(ENSEMBLE_WEIGHTS[1])

    if "xgb" in models:
        preds.append(models["xgb"].predict(X_flat))
        weights_used.append(ENSEMBLE_WEIGHTS[2])

    # Normalise weights and compute ensemble
    total     = sum(weights_used)
    n_min     = min(len(arr) for arr in preds)
    ensemble  = np.array(sum((w / total) * arr[-n_min:] for w, arr in zip(weights_used, preds)))

    dates      = test_df.index[lk: lk + n_min]
    actual_aln = actual[:n_min]

    ensemble   = sanitize(ensemble)
    actual_aln = sanitize(actual_aln)
    return dates, actual_aln, ensemble


def get_forecast(feat_df, models):
    """Run 30-day iterative forecast from the last known date."""
    from features.feature_engineering import get_feature_columns
    feat_cols   = get_feature_columns(feat_df)
    feat_scaler = models.get("feat_scaler")
    price_scaler  = models.get("price_scaler")
    if feat_scaler is None or price_scaler is None or "lstm" not in models:
        return None, None

    X_all       = feat_scaler.transform(feat_df[feat_cols])
    last_seq    = X_all[-LOOKBACK_WINDOW:]
    last_flat   = X_all[-1]
    last_price  = feat_df["close"].iloc[-1]

    from scripts.train_models import forecast_30_days
    forecast = forecast_30_days(
        models["lstm"], models["rf"], models["xgb"],
        last_seq, last_flat, price_scaler,
        n_days=FORECAST_HORIZON
    )
    last_date    = feat_df.index[-1]
    bdays        = pd.bdate_range(start=last_date + timedelta(days=1), periods=FORECAST_HORIZON)
    return bdays, forecast


def get_direction_prediction(feat_df, models):
    """Predict direction (UP/DOWN) for the NEXT trading day."""
    from features.feature_engineering import get_feature_columns
    feat_cols   = get_feature_columns(feat_df)
    feat_scaler = models.get("feat_scaler")
    rf_clf      = models.get("rf_clf")
    xgb_clf     = models.get("xgb_clf")
    if feat_scaler is None or (rf_clf is None and xgb_clf is None):
        return None, None

    X_last = feat_scaler.transform(feat_df[feat_cols].iloc[[-1]])
    probs  = []
    if rf_clf:
        probs.append(rf_clf.predict_proba(X_last)[0][1])
    if xgb_clf:
        probs.append(xgb_clf.predict_proba(X_last)[0][1])
    avg_prob  = np.mean(probs)
    direction = "UP ↑" if avg_prob >= 0.5 else "DOWN ↓"
    return direction, avg_prob


# ─────────────────────────────────────────────────────────────────────────────
# CHART BUILDERS (Plotly)
# ─────────────────────────────────────────────────────────────────────────────

def chart_actual_vs_predicted(dates, actual, predicted):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=actual,
        name="Actual Price", mode="lines",
        line=dict(color="#c9d1d9", width=1.5)
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=predicted,
        name="Ensemble Prediction", mode="lines",
        line=dict(color="#00d4ff", width=1.5, dash="dot")
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Actual vs Predicted — Brent Crude (Test Set)",
        yaxis_title="Price (USD/bbl)",
        height=400,
    )
    return fig


def chart_forecast(hist_dates, hist_prices, fore_dates, fore_prices):
    fig = go.Figure()
    # Historical last 90 days
    n   = min(90, len(hist_dates))
    fig.add_trace(go.Scatter(
        x=hist_dates[-n:], y=hist_prices[-n:],
        name="Historical", mode="lines",
        line=dict(color="#c9d1d9", width=1.5)
    ))
    # Confidence band ±5%
    fig.add_trace(go.Scatter(
        x=list(fore_dates) + list(fore_dates[::-1]),
        y=list(fore_prices * 1.05) + list(fore_prices[::-1] * 0.95),
        fill="toself", fillcolor="rgba(0,212,255,0.08)",
        line=dict(color="rgba(0,0,0,0)"),
        name="±5% Band", showlegend=True,
    ))
    # Forecast line
    fig.add_trace(go.Scatter(
        x=fore_dates, y=fore_prices,
        name="30-Day Forecast", mode="lines",
        line=dict(color="#00d4ff", width=2.5)
    ))
    # Vertical separator — scatter trace avoids Plotly Timestamp bug
    fig.add_trace(go.Scatter(
        x=[hist_dates[-1], hist_dates[-1]],
        y=[float(fore_prices.min()) * 0.95, float(fore_prices.max()) * 1.05],
        mode="lines",
        line=dict(color="#e3b341", width=1.5, dash="dot"),
        name="Forecast Start",
        showlegend=True,
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Brent Crude — 30-Day Price Forecast",
        yaxis_title="Price (USD/bbl)",
        height=420,
    )
    return fig


def chart_feature_importance(model, feat_cols, model_name="XGBoost"):
    imp  = model.feature_importances_
    top  = pd.Series(imp, index=feat_cols).nlargest(20).sort_values()
    clrs = ["#e3b341" if i >= len(top) - 5 else "#00d4ff" for i in range(len(top))]
    fig  = go.Figure(go.Bar(
        x=top.values, y=top.index,
        orientation="h",
        marker_color=clrs,
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=f"Feature Importance — {model_name} (Top 20)",
        xaxis_title="Importance Score",
        height=500,
    )
    return fig


def chart_price_history(df):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3], vertical_spacing=0.04)
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["open"], high=df["high"],
        low=df["low"],   close=df["close"],
        name="Price", increasing_line_color="#39d353",
        decreasing_line_color="#f85149",
    ), row=1, col=1)
    if "volume" in df.columns:
        colors = ["#39d353" if c >= o else "#f85149"
                  for c, o in zip(df["close"], df["open"])]
        fig.add_trace(go.Bar(
            x=df.index, y=df["volume"],
            marker_color=colors, name="Volume", opacity=0.6,
        ), row=2, col=1)
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Brent Crude — Historical OHLCV",
        yaxis_title="Price (USD/bbl)",
        yaxis2_title="Volume",
        xaxis_rangeslider_visible=False,
        height=520,
    )
    return fig


def chart_volatility(df):
    log_ret = np.log(df["close"] / df["close"].shift(1))
    vol_7   = log_ret.rolling(7).std()  * np.sqrt(252) * 100
    vol_30  = log_ret.rolling(30).std() * np.sqrt(252) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=vol_7,  name="7-Day Vol",
                             line=dict(color="#e3b341", width=1.2)))
    fig.add_trace(go.Scatter(x=df.index, y=vol_30, name="30-Day Vol",
                             fill="tozeroy", fillcolor="rgba(248,81,73,0.12)",
                             line=dict(color="#f85149", width=1.5)))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Realised Volatility (Annualised)",
        yaxis_title="Volatility (%)",
        height=350,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("### ⚙️ Controls")
        st.markdown("---")

        st.markdown("**Data Window**")
        years = st.slider("Historical years", 1, 25, 5)

        st.markdown("**Forecast**")
        n_days = st.slider("Forecast horizon (days)", 7, 90, FORECAST_HORIZON)

        st.markdown("**Ensemble Weights**")
        w_lstm = st.slider("LSTM",         0.0, 1.0, ENSEMBLE_WEIGHTS[0], 0.05)
        w_rf   = st.slider("Random Forest",0.0, 1.0, ENSEMBLE_WEIGHTS[1], 0.05)
        w_xgb  = st.slider("XGBoost",      0.0, 1.0, ENSEMBLE_WEIGHTS[2], 0.05)
        total  = w_lstm + w_rf + w_xgb
        if abs(total - 1.0) > 0.05:
            st.warning(f"Weights sum = {total:.2f}. They will be normalised.")
        weights = [w / (total or 1) for w in [w_lstm, w_rf, w_xgb]]

        st.markdown("---")
        retrain_btn = st.button("🔁 Retrain Models")
        refresh_btn = st.button("🔄 Refresh Data")

        st.markdown("---")
        st.markdown(
            "<div style='font-family:JetBrains Mono;font-size:0.7rem;color:#8b949e'>"
            "CrudeEdge v1.0<br>BTech Major Project<br>India Oil Intelligence</div>",
            unsafe_allow_html=True
        )

    return years, n_days, weights, retrain_btn, refresh_btn


# ─────────────────────────────────────────────────────────────────────────────
# MAIN DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Header
    st.markdown(
        "<p class='main-title'>🛢️ CrudeEdge</p>"
        "<p class='sub-title'>INDIA OIL PRICE INTELLIGENCE PLATFORM</p>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    years, n_days, weights, retrain_btn, refresh_btn = render_sidebar()

    # Load data
    if refresh_btn:
        st.cache_data.clear()
        st.rerun()

    raw_df   = load_data()
    feat_df  = load_features()
    models   = load_models()

    # Filter by date range
    cutoff   = datetime.today() - timedelta(days=365 * years)
    raw_view = raw_df[raw_df.index >= cutoff]

    # ── TOP METRIC CARDS ───────────────────────────────────────────────────
    current_price = raw_df["close"].iloc[-1]
    prev_price    = raw_df["close"].iloc[-2]
    price_change  = current_price - prev_price
    pct_change    = (price_change / prev_price) * 100
    direction_str, dir_prob = get_direction_prediction(feat_df, models) if feat_df is not None and models else (None, None)
    usd_inr = raw_df["usd_inr"].iloc[-1] if "usd_inr" in raw_df.columns else 83.5
    inr_price = current_price * usd_inr

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        chg_cls = "up" if price_change >= 0 else "down"
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>BRENT CRUDE (USD)</div>
            <div class='metric-value'>${current_price:.2f}</div>
            <div class='metric-delta {chg_cls}'>{'+' if price_change>=0 else ''}{price_change:.2f} ({pct_change:.2f}%)</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>BRENT CRUDE (INR/bbl)</div>
            <div class='metric-value'>₹{inr_price:,.0f}</div>
            <div class='metric-delta neutral'>USD/INR: {usd_inr:.2f}</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        dir_css = "up" if direction_str and "UP" in direction_str else "down"
        dir_display = direction_str or "—"
        prob_display = f"{dir_prob*100:.1f}% confidence" if dir_prob else "—"
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>NEXT DAY TREND</div>
            <div class='metric-value {dir_css}'>{dir_display}</div>
            <div class='metric-delta neutral'>{prob_display}</div>
        </div>""", unsafe_allow_html=True)

    with c4:
        vol_30 = raw_df["close"].pct_change().rolling(30).std().iloc[-1] * np.sqrt(252) * 100
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>30D VOLATILITY (ANN.)</div>
            <div class='metric-value'>{vol_30:.1f}%</div>
            <div class='metric-delta neutral'>Realised</div>
        </div>""", unsafe_allow_html=True)

    with c5:
        last_update = raw_df.index[-1].strftime("%Y-%m-%d")
        n_models    = sum(1 for k in ["lstm","rf","xgb"] if k in models)
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>LAST UPDATE</div>
            <div class='metric-value' style='font-size:1.3rem'>{last_update}</div>
            <div class='metric-delta neutral'>{n_models}/3 models loaded</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── TABS ───────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Price & Predictions",
        "🔮 30-Day Forecast",
        "🎯 Feature Importance",
        "📊 Model Metrics",
        "🌡️ Volatility",
    ])

    # TAB 1 — Price & Predictions
    with tab1:
        st.plotly_chart(chart_price_history(raw_view), use_container_width=True)

        if feat_df is not None and models and "rf" in models and "xgb" in models:
            with st.spinner("Running ensemble inference …"):
                dates, actual, predicted = get_ensemble_predictions(feat_df, models)
            if dates is not None:
                # ── Sanitize predictions before any metric computation ──────
                min_len  = min(len(actual), len(predicted))
                a        = np.array(actual[:min_len],    dtype=np.float64)
                p        = np.array(predicted[:min_len], dtype=np.float64)
                # Replace inf / -inf with NaN then forward-fill
                a = pd.Series(a).replace([np.inf, -np.inf], np.nan).ffill().bfill().values
                p = pd.Series(p).replace([np.inf, -np.inf], np.nan).ffill().bfill().values
                # Drop any remaining NaN pairs
                mask = np.isfinite(a) & np.isfinite(p)
                a, p = a[mask], p[mask]

                if len(a) < 2:
                    st.warning("Not enough valid predictions to display metrics. Train the models first.")
                else:
                    st.plotly_chart(
                        chart_actual_vs_predicted(dates[:len(a)], a, p),
                        use_container_width=True
                    )
                    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                    rmse    = np.sqrt(mean_squared_error(a, p))
                    mae     = mean_absolute_error(a, p)
                    r2      = r2_score(a, p)
                    dir_acc = np.mean(np.sign(np.diff(a)) == np.sign(np.diff(p))) * 100

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("RMSE",             f"${rmse:.2f}")
                    m2.metric("MAE",              f"${mae:.2f}")
                    m3.metric("R² Score",         f"{r2:.4f}")
                    m4.metric("Directional Acc.", f"{dir_acc:.1f}%")
        else:
            st.info("🔧 Train the models first using **scripts/train_models.py** to see predictions here.")

    # TAB 2 — 30-Day Forecast
    with tab2:
        if feat_df is not None and models and "rf" in models and "xgb" in models:
            with st.spinner("Generating 30-day forecast …"):
                fore_dates, fore_prices = get_forecast(feat_df, models)
            if fore_dates is not None:
                fore_prices = sanitize(fore_prices)
                st.plotly_chart(
                    chart_forecast(feat_df.index, feat_df["close"].values,
                                   fore_dates, fore_prices),
                    use_container_width=True
                )
                # Forecast table
                st.markdown("<p class='section-header'>Forecast Table</p>", unsafe_allow_html=True)
                fore_df = pd.DataFrame({
                    "Date":              fore_dates.strftime("%Y-%m-%d"),
                    "Forecast (USD)":    [f"${p:.2f}" for p in fore_prices],
                    "Forecast (INR)":    [f"₹{p*usd_inr:,.0f}" for p in fore_prices],
                    "Change from Today": [f"{((p-current_price)/current_price)*100:+.2f}%" for p in fore_prices],
                })
                st.dataframe(fore_df, use_container_width=True, hide_index=True)
        else:
            st.info("Train models to see forecast.")

    # TAB 3 — Feature Importance
    with tab3:
        if "xgb" in models and feat_df is not None:
            from features.feature_engineering import get_feature_columns
            feat_cols = get_feature_columns(feat_df)
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(
                    chart_feature_importance(models["xgb"], feat_cols, "XGBoost Regressor"),
                    use_container_width=True
                )
            with col2:
                if "rf" in models:
                    st.plotly_chart(
                        chart_feature_importance(models["rf"], feat_cols, "Random Forest"),
                        use_container_width=True
                    )
        else:
            st.info("Train models to see feature importances.")

    # TAB 4 — Model Metrics
    with tab4:
        st.markdown("<p class='section-header'>Regression Performance</p>", unsafe_allow_html=True)
        if feat_df is not None and models and "rf" in models and "xgb" in models:
            dates, actual, pred_ens = get_ensemble_predictions(feat_df, models)
            if dates is not None:
                min_len = min(len(actual), len(pred_ens))
                a, p = actual[:min_len], pred_ens[:min_len]

                metrics_rows = []
                metrics_rows.append({
                    "Model":     "🧠 Ensemble (Weighted)",
                    "RMSE":      f"${np.sqrt(np.mean((a-p)**2)):.3f}",
                    "MAE":       f"${np.mean(np.abs(a-p)):.3f}",
                    "R²":        f"{1 - np.sum((a-p)**2)/np.sum((a-np.mean(a))**2):.4f}",
                    "Dir Acc %": f"{np.mean(np.sign(np.diff(a))==np.sign(np.diff(p)))*100:.1f}%",
                })
                st.dataframe(pd.DataFrame(metrics_rows), use_container_width=True, hide_index=True)

        st.markdown("<p class='section-header'>Direction Classification</p>", unsafe_allow_html=True)
        if "rf_clf" in models and feat_df is not None:
            from features.feature_engineering import get_feature_columns
            from config import TRAIN_TEST_SPLIT
            feat_cols   = get_feature_columns(feat_df)
            split       = int(len(feat_df) * TRAIN_TEST_SPLIT)
            test_df     = feat_df.iloc[split:]
            X_test      = models["feat_scaler"].transform(test_df[feat_cols])
            y_true_dir  = test_df["direction"].values
            y_pred_rf   = models["rf_clf"].predict(X_test)
            y_pred_xgb  = models["xgb_clf"].predict(X_test) if "xgb_clf" in models else y_pred_rf

            from sklearn.metrics import accuracy_score, f1_score
            clf_rows = [
                {"Model": "RF Classifier",  "Accuracy": f"{accuracy_score(y_true_dir, y_pred_rf):.4f}",
                 "F1": f"{f1_score(y_true_dir, y_pred_rf, zero_division=0):.4f}"},
                {"Model": "XGB Classifier", "Accuracy": f"{accuracy_score(y_true_dir, y_pred_xgb):.4f}",
                 "F1": f"{f1_score(y_true_dir, y_pred_xgb, zero_division=0):.4f}"},
            ]
            st.dataframe(pd.DataFrame(clf_rows), use_container_width=True, hide_index=True)
        else:
            st.info("Train direction models to see classification metrics.")

    # TAB 5 — Volatility
    with tab5:
        st.plotly_chart(chart_volatility(raw_view), use_container_width=True)

    # ── RETRAIN ───────────────────────────────────────────────────────────
    if retrain_btn:
        st.warning("⚠️ Retraining will take several minutes. Do not close this window.")
        with st.spinner("Retraining all models …"):
            try:
                from scripts.data_ingestion import ingest_all
                from features.feature_engineering import run_feature_pipeline
                from scripts.train_models import train_all
                raw   = ingest_all(save=True)
                pipe  = run_feature_pipeline(raw)
                train_all(pipe)
                st.cache_data.clear()
                st.cache_resource.clear()
                st.success("✅ Models retrained and saved! Refreshing …")
                time.sleep(2)
                st.rerun()
            except Exception as e:
                st.error(f"Retraining failed: {e}")

    # Auto-refresh footer
    st.markdown("---")
    st.markdown(
        f"<div style='font-family:JetBrains Mono;font-size:0.7rem;color:#8b949e;text-align:center'>"
        f"Auto-refresh every {REFRESH_INTERVAL_SECONDS//60} min  •  "
        f"Last render: {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}  •  "
        f"CrudeEdge — India Oil Intelligence"
        f"</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()

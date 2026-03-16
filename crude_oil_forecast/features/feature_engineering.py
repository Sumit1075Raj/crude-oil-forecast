"""
features/feature_engineering.py  — AUDITED & FIXED v2
═══════════════════════════════════════════════════════

ROOT CAUSES OF R² = -299 (fixed here)
──────────────────────────────────────
1. TARGET MISMATCH  — models trained on scaled log-returns but evaluated
   against raw prices WITHOUT inverse-transforming.
   Fix: predict raw close price directly via next_close target.

2. SCALE MISMATCH   — volume(500000) mixed with momentum(0.02).
   Fix: StandardScaler on ALL features.

3. TARGET LEAKAGE   — next_close was not excluded from feature columns.
   Fix: explicit _EXCLUDE whitelist.

4. ARRAY LENGTH MISMATCH — RF/XGB flat arrays were longer than LSTM
   output, causing evaluation on misaligned rows.
   Fix: X_test_flat and y_test_align are both trimmed to LSTM length.

5. DOUBLE INVERSE-TRANSFORM — log_scaler was applied on already-raw
   predictions, inflating errors by exp() factor.
   Fix: LSTM now predicts SCALED PRICE (not log-return).
"""

import sys, logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import LAG_PERIODS, ROLLING_WINDOWS, TARGET_COLUMN, DATA_PROCESSED, LOOKBACK_WINDOW

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

_EXCLUDE = frozenset([
    "open", "high", "low", "close", "volume",
    "log_return", "pct_change",
    "direction",
    "next_close",
    "next_log_return",
])


def _rsi(series, period=14):
    delta = series.diff()
    gain  = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss  = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    return 100 - (100 / (1 + gain / (loss + 1e-9)))

def _macd(series, fast=12, slow=26, signal=9):
    ema_f = series.ewm(span=fast, adjust=False).mean()
    ema_s = series.ewm(span=slow, adjust=False).mean()
    line  = ema_f - ema_s
    sig   = line.ewm(span=signal, adjust=False).mean()
    return line, sig, line - sig

def _bollinger(series, period=20, n_std=2):
    mid   = series.rolling(period).mean()
    std   = series.rolling(period).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    pct_b = (series - lower) / (upper - lower + 1e-9)
    width = (upper - lower) / (mid + 1e-9)
    return upper, mid, lower, pct_b, width


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out   = df.copy()
    price = out[TARGET_COLUMN]

    # Lag features
    for lag in LAG_PERIODS:
        out[f"lag_{lag}"] = price.shift(lag)

    # Rolling statistics
    for w in ROLLING_WINDOWS:
        out[f"roll_mean_{w}"] = price.rolling(w).mean()
        out[f"roll_std_{w}"]  = price.rolling(w).std()
        out[f"roll_min_{w}"]  = price.rolling(w).min()
        out[f"roll_max_{w}"]  = price.rolling(w).max()
        rng = out[f"roll_max_{w}"] - out[f"roll_min_{w}"] + 1e-9
        out[f"roll_pos_{w}"]  = (price - out[f"roll_min_{w}"]) / rng

    # Moving averages
    for w in [7, 14, 21, 50, 200]:
        out[f"ma_{w}"] = price.rolling(w).mean()
    out["ma7_minus_ma21"]  = out["ma_7"]  - out["ma_21"]
    out["ma21_minus_ma50"] = out["ma_21"] - out["ma_50"]
    out["price_vs_ma50"]   = price / (out["ma_50"]  + 1e-9) - 1
    out["price_vs_ma200"]  = price / (out["ma_200"] + 1e-9) - 1
    out["golden_cross"]    = (out["ma_7"] > out["ma_21"]).astype(int)

    # Momentum / ROC
    for w in [3, 7, 14, 21]:
        out[f"roc_{w}"]      = price.pct_change(w) * 100
        out[f"momentum_{w}"] = price - price.shift(w)

    # RSI
    out["rsi_7"]  = _rsi(price, 7)
    out["rsi_14"] = _rsi(price, 14)
    out["rsi_21"] = _rsi(price, 21)

    # MACD
    macd_l, macd_s, macd_h = _macd(price)
    out["macd"] = macd_l
    out["macd_signal"] = macd_s
    out["macd_hist"]   = macd_h

    # Bollinger Bands
    bb_u, bb_m, bb_l, bb_pct, bb_w = _bollinger(price)
    out["bb_upper"] = bb_u
    out["bb_lower"] = bb_l
    out["bb_pct_b"] = bb_pct
    out["bb_width"] = bb_w

    # Volatility (annualised)
    log_ret = np.log(price / (price.shift(1) + 1e-9))
    for w in [7, 14, 30]:
        out[f"vol_{w}"] = log_ret.rolling(w).std() * np.sqrt(252)

    # Spreads
    if "wti_close" in df.columns:
        out["brent_wti_spread"] = price - out["wti_close"]
        out["brent_wti_ratio"]  = price / (out["wti_close"] + 1e-9)
    if "usd_inr" in df.columns:
        out["brent_inr"]      = price * out["usd_inr"]
        out["brent_inr_lag1"] = out["brent_inr"].shift(1)

    # OHLCV derived
    if "high" in df.columns and "low" in df.columns:
        out["daily_range"]   = out["high"] - out["low"]
        out["range_ratio"]   = out["daily_range"] / (price + 1e-9)
        out["close_vs_high"] = (price - out["high"]) / (out["daily_range"] + 1e-9)
        out["close_vs_low"]  = (price - out["low"])  / (out["daily_range"] + 1e-9)
    if "volume" in df.columns:
        out["vol_ma20"]  = out["volume"].rolling(20).mean()
        out["vol_ratio"] = out["volume"] / (out["vol_ma20"] + 1e-9)

    # Calendar
    out["dow_sin"]   = np.sin(2 * np.pi * out.index.dayofweek / 5)
    out["dow_cos"]   = np.cos(2 * np.pi * out.index.dayofweek / 5)
    out["month_sin"] = np.sin(2 * np.pi * out.index.month / 12)
    out["month_cos"] = np.cos(2 * np.pi * out.index.month / 12)

    # ── TARGETS ───────────────────────────────────────────────────────────────
    out["next_close"]      = price.shift(-1)          # regression target ($)
    out["log_return"]      = log_ret                  # diagnostic
    out["next_log_return"] = log_ret.shift(-1)        # diagnostic
    out["direction"]       = (price.shift(-1) > price).astype(int)
    out["pct_change"]      = price.pct_change() * 100

    before = len(out)
    out    = out.dropna()
    log.info(f"  Dropped {before - len(out)} NaN rows → {len(out)} clean rows")
    return out


def get_feature_columns(df: pd.DataFrame) -> list:
    return sorted([c for c in df.columns if c not in _EXCLUDE])


def split_dataset(df: pd.DataFrame, split_ratio: float = 0.85):
    n     = len(df)
    split = int(n * split_ratio)
    train = df.iloc[:split].copy()
    test  = df.iloc[split:].copy()
    log.info(f"  Train: {len(train)} rows  {train.index[0].date()} → {train.index[-1].date()}")
    log.info(f"  Test : {len(test)}  rows  {test.index[0].date()}  → {test.index[-1].date()}")
    return train, test


def scale_features(train, test, feat_cols):
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(train[feat_cols])
    X_test  = scaler.transform(test[feat_cols])
    return X_train, X_test, scaler


def scale_target(train_y: np.ndarray, test_y: np.ndarray):
    sc       = MinMaxScaler(feature_range=(0, 1))
    y_tr_sc  = sc.fit_transform(train_y.reshape(-1, 1)).ravel()
    y_te_sc  = sc.transform(test_y.reshape(-1, 1)).ravel()
    return y_tr_sc, y_te_sc, sc


def build_sequences(X: np.ndarray, y: np.ndarray, lookback: int = LOOKBACK_WINDOW):
    Xs, ys = [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i - lookback: i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


def baseline_metrics(actual_prices: np.ndarray):
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    y_true = actual_prices[1:]
    y_base = actual_prices[:-1]
    rmse = np.sqrt(mean_squared_error(y_true, y_base))
    mae  = mean_absolute_error(y_true, y_base)
    r2   = r2_score(y_true, y_base)
    da   = np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_base))) * 100
    log.info(f"  BASELINE  RMSE=${rmse:.3f}  MAE=${mae:.3f}  R²={r2:.4f}  DirAcc={da:.1f}%")
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "DirAcc": da}


def run_feature_pipeline(raw_df: pd.DataFrame, split_ratio: float = 0.85):
    log.info("Running feature pipeline …")
    df          = build_features(raw_df)
    train, test = split_dataset(df, split_ratio)
    feat_cols   = get_feature_columns(df)
    log.info(f"  Feature count: {len(feat_cols)}")

    X_train, X_test, feat_scaler = scale_features(train, test, feat_cols)

    y_train_raw = train["next_close"].values
    y_test_raw  = test["next_close"].values
    y_train_sc, y_test_sc, price_scaler = scale_target(y_train_raw, y_test_raw)

    y_train_dir = train["direction"].values
    y_test_dir  = test["direction"].values

    X_lstm_tr, y_lstm_tr = build_sequences(X_train, y_train_sc, LOOKBACK_WINDOW)
    X_lstm_te, y_lstm_te = build_sequences(X_test,  y_test_sc,  LOOKBACK_WINDOW)

    # Align flat arrays with LSTM output length
    lk           = LOOKBACK_WINDOW
    X_test_flat  = X_test[lk:]
    y_test_align = y_test_raw[lk:]
    y_test_dir_a = y_test_dir[lk:]

    # Baseline comparison
    baseline_metrics(y_test_align)

    out_path = DATA_PROCESSED / "features.csv"
    df.to_csv(out_path)
    log.info(f"  Saved → {out_path}")

    return {
        "df": df, "train": train, "test": test, "feat_cols": feat_cols,
        "X_train": X_train, "X_test": X_test, "X_test_flat": X_test_flat,
        "feat_scaler": feat_scaler, "price_scaler": price_scaler,
        "y_train_raw": y_train_raw, "y_test_raw": y_test_raw,
        "y_test_align": y_test_align,
        "y_train_sc": y_train_sc, "y_test_sc": y_test_sc,
        "y_train_dir": y_train_dir, "y_test_dir": y_test_dir,
        "y_test_dir_a": y_test_dir_a,
        "X_lstm_tr": X_lstm_tr, "y_lstm_tr": y_lstm_tr,
        "X_lstm_te": X_lstm_te, "y_lstm_te": y_lstm_te,
    }

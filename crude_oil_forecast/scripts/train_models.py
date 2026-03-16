"""
scripts/train_models.py — AUDITED & FIXED v2
═════════════════════════════════════════════

Key fixes vs v1
───────────────
• LSTM predicts SCALED PRICE → inverse_transform → USD  (not log-return)
• RF / XGB predict RAW USD price directly  (no inverse needed)
• Ensemble averages USD predictions  (all in same space)
• forecast_30_days reconstructs price from SCALED predictions only
• All evaluation arrays are length-aligned before metric computation
"""

import sys, logging, warnings
import numpy as np
import joblib
from pathlib import Path
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    MODELS_DIR, RANDOM_STATE,
    LSTM_UNITS, LSTM_DROPOUT, LSTM_EPOCHS, LSTM_BATCH_SIZE, LSTM_LEARNING_RATE,
    RF_N_ESTIMATORS, RF_MAX_DEPTH, RF_MIN_SAMPLES_LEAF,
    XGB_N_ESTIMATORS, XGB_MAX_DEPTH, XGB_LEARNING_RATE, XGB_SUBSAMPLE,
    ENSEMBLE_WEIGHTS, LOOKBACK_WINDOW, FORECAST_HORIZON,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. LSTM  — predicts SCALED next_close price
# ─────────────────────────────────────────────────────────────────────────────

def build_lstm(input_shape: tuple):
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam

    model = Sequential([
        LSTM(LSTM_UNITS[0], return_sequences=True, input_shape=input_shape),
        Dropout(LSTM_DROPOUT),
        BatchNormalization(),
        LSTM(LSTM_UNITS[1], return_sequences=False),
        Dropout(LSTM_DROPOUT),
        BatchNormalization(),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid"),   # sigmoid → output in [0,1] matching MinMaxScaler
    ], name="BrentLSTM")

    model.compile(
        optimizer=Adam(learning_rate=LSTM_LEARNING_RATE),
        loss="mse",
        metrics=["mae"]
    )
    return model


def train_lstm(X_train, y_train, X_val=None, y_val=None):
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

    log.info("─── Training LSTM ─────────────────────────────────────")
    model = build_lstm((X_train.shape[1], X_train.shape[2]))
    model.summary(print_fn=lambda x: log.info(f"  {x}"))

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1),
        ModelCheckpoint(str(MODELS_DIR / "lstm_best.keras"), save_best_only=True, verbose=0),
    ]
    val_data = (X_val, y_val) if X_val is not None else None
    history  = model.fit(
        X_train, y_train,
        epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH_SIZE,
        validation_data=val_data, callbacks=callbacks, verbose=1,
    )
    model.save(str(MODELS_DIR / "lstm_final.keras"))
    log.info(f"  ✓ LSTM saved")
    return model, history


# ─────────────────────────────────────────────────────────────────────────────
# 2. RANDOM FOREST  — predicts RAW USD price
# ─────────────────────────────────────────────────────────────────────────────

def train_random_forest(X_train, y_train):
    from sklearn.ensemble import RandomForestRegressor
    log.info("─── Training Random Forest ─────────────────────────────")
    rf = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        n_jobs=-1, random_state=RANDOM_STATE, oob_score=True,
    )
    rf.fit(X_train, y_train)
    log.info(f"  OOB R² = {rf.oob_score_:.4f}")
    joblib.dump(rf, MODELS_DIR / "random_forest.pkl", compress=3)
    log.info(f"  ✓ RF saved")
    return rf


# ─────────────────────────────────────────────────────────────────────────────
# 3. XGBOOST  — predicts RAW USD price
# ─────────────────────────────────────────────────────────────────────────────

def train_xgboost(X_train, y_train, X_val=None, y_val=None):
    import xgboost as xgb
    log.info("─── Training XGBoost ───────────────────────────────────")
    eval_set = [(X_train, y_train)]
    if X_val is not None:
        eval_set.append((X_val, y_val))
    model = xgb.XGBRegressor(
        n_estimators=XGB_N_ESTIMATORS, max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LEARNING_RATE, subsample=XGB_SUBSAMPLE,
        colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        random_state=RANDOM_STATE, tree_method="hist", eval_metric="rmse",
        early_stopping_rounds=20 if X_val is not None else None, verbosity=1,
    )
    model.fit(X_train, y_train, eval_set=eval_set, verbose=50)
    joblib.dump(model, MODELS_DIR / "xgboost.pkl", compress=3)
    log.info(f"  ✓ XGBoost saved")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 4. DIRECTION CLASSIFIERS
# ─────────────────────────────────────────────────────────────────────────────

def train_direction_models(X_train, y_train_dir, X_val=None, y_val_dir=None):
    from sklearn.ensemble import RandomForestClassifier
    import xgboost as xgb

    log.info("─── Training Direction Classifiers ─────────────────────")
    rf_clf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH,
        n_jobs=-1, random_state=RANDOM_STATE, class_weight="balanced",
    )
    rf_clf.fit(X_train, y_train_dir)
    joblib.dump(rf_clf, MODELS_DIR / "direction_rf.pkl", compress=3)

    eval_set = [(X_train, y_train_dir)]
    if X_val is not None:
        eval_set.append((X_val, y_val_dir))
    xgb_clf = xgb.XGBClassifier(
        n_estimators=XGB_N_ESTIMATORS, max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LEARNING_RATE, subsample=XGB_SUBSAMPLE,
        colsample_bytree=0.8, use_label_encoder=False, eval_metric="logloss",
        random_state=RANDOM_STATE, tree_method="hist",
        scale_pos_weight=(y_train_dir == 0).sum() / max((y_train_dir == 1).sum(), 1),
    )
    xgb_clf.fit(X_train, y_train_dir, eval_set=eval_set, verbose=50)
    joblib.dump(xgb_clf, MODELS_DIR / "direction_xgb.pkl", compress=3)
    log.info(f"  ✓ Direction classifiers saved")
    return rf_clf, xgb_clf


# ─────────────────────────────────────────────────────────────────────────────
# 5. ENSEMBLE PREDICTION  — all models output USD price
# ─────────────────────────────────────────────────────────────────────────────

def ensemble_predict(lstm_model, rf_model, xgb_model,
                     X_seq, X_flat, price_scaler,
                     weights=ENSEMBLE_WEIGHTS):
    """
    Returns ensemble predicted prices in USD.

    LSTM: predicts scaled price → inverse_transform → USD
    RF  : predicts raw USD directly
    XGB : predicts raw USD directly
    """
    # LSTM
    lstm_sc  = lstm_model.predict(X_seq, verbose=0).ravel()
    lstm_usd = price_scaler.inverse_transform(lstm_sc.reshape(-1, 1)).ravel()

    # RF / XGB (flat features, length-aligned)
    n        = len(lstm_usd)
    rf_usd   = rf_model.predict(X_flat[-n:])
    xgb_usd  = xgb_model.predict(X_flat[-n:])

    w       = [w / sum(weights) for w in weights]   # normalise weights
    ensemble = w[0] * lstm_usd + w[1] * rf_usd + w[2] * xgb_usd
    return ensemble, lstm_usd, rf_usd, xgb_usd


# ─────────────────────────────────────────────────────────────────────────────
# 6. 30-DAY FORECAST
# ─────────────────────────────────────────────────────────────────────────────

def forecast_30_days(lstm_model, rf_model, xgb_model,
                     last_sequence, last_flat, price_scaler,
                     n_days=FORECAST_HORIZON, weights=ENSEMBLE_WEIGHTS):
    """
    Iterative forecast. At each step:
      1. Predict SCALED price with LSTM → inverse_transform → USD
      2. Predict USD directly with RF + XGB
      3. Ensemble average
      4. Roll window forward
    """
    log.info(f"  Generating {n_days}-day forecast …")
    seq   = last_sequence.copy()
    flat  = last_flat.copy()
    w     = [wt / sum(weights) for wt in weights]
    prices = []

    for _ in range(n_days):
        lstm_sc  = lstm_model.predict(seq[np.newaxis], verbose=0).ravel()[0]
        lstm_usd = price_scaler.inverse_transform([[lstm_sc]])[0, 0]
        rf_usd   = rf_model.predict(flat[np.newaxis])[0]
        xgb_usd  = xgb_model.predict(flat[np.newaxis])[0]
        pred     = w[0] * lstm_usd + w[1] * rf_usd + w[2] * xgb_usd
        prices.append(pred)
        # Roll sequence window
        new_row = flat.copy()
        seq     = np.vstack([seq[1:], new_row])
        flat    = new_row

    return np.array(prices)


# ─────────────────────────────────────────────────────────────────────────────
# 7. MASTER TRAINING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def train_all(pipeline_data: dict):
    lk = LOOKBACK_WINDOW

    X_lstm_tr   = pipeline_data["X_lstm_tr"]
    y_lstm_tr   = pipeline_data["y_lstm_tr"]
    X_lstm_te   = pipeline_data["X_lstm_te"]
    y_lstm_te   = pipeline_data["y_lstm_te"]

    # Flat train arrays (skip first lk rows to align with LSTM)
    X_tr_flat   = pipeline_data["X_train"][lk:]
    y_tr_raw    = pipeline_data["y_train_raw"][lk:]
    y_tr_dir    = pipeline_data["y_train_dir"][lk:]

    # Validation split (last 10% of train)
    vs          = int(len(X_lstm_tr) * 0.9)
    X_lv, y_lv = X_lstm_tr[vs:], y_lstm_tr[vs:]
    X_lt, y_lt = X_lstm_tr[:vs], y_lstm_tr[:vs]

    vf          = int(len(X_tr_flat) * 0.9)
    X_fv        = X_tr_flat[vf:]
    y_fv        = y_tr_raw[vf:]

    lstm, hist  = train_lstm(X_lt, y_lt, X_lv, y_lv)
    rf          = train_random_forest(X_tr_flat, y_tr_raw)
    xgb         = train_xgboost(X_tr_flat, y_tr_raw, X_fv, y_fv)

    # Direction classifiers
    rf_clf, xgb_clf = train_direction_models(
        X_tr_flat, y_tr_dir,
        pipeline_data["X_test_flat"], pipeline_data["y_test_dir_a"],
    )

    # Save scalers
    joblib.dump(pipeline_data["feat_scaler"],  MODELS_DIR / "feat_scaler.pkl")
    joblib.dump(pipeline_data["price_scaler"], MODELS_DIR / "price_scaler.pkl")
    log.info("  ✓ Scalers saved")

    return {
        "lstm": lstm, "rf": rf, "xgb": xgb,
        "rf_clf": rf_clf, "xgb_clf": xgb_clf,
        "history": hist,
        "price_scaler": pipeline_data["price_scaler"],
        "feat_scaler":  pipeline_data["feat_scaler"],
    }


if __name__ == "__main__":
    from scripts.data_ingestion import ingest_all
    from features.feature_engineering import run_feature_pipeline
    raw  = ingest_all(save=True)
    pipe = run_feature_pipeline(raw)
    train_all(pipe)

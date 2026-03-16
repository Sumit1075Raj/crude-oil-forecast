"""
run_pipeline.py — AUDITED & FIXED v2
══════════════════════════════════════
Step 1 → Ingest data
Step 2 → Feature engineering  (target = next_close USD price)
Step 3 → Train LSTM + RF + XGB + Direction classifiers
Step 4 → Evaluate all models + save charts
Step 5 → (Optional) launch dashboard

Usage:
    python run_pipeline.py           # Full pipeline
    python run_pipeline.py --fast    # Skip LSTM
    python run_pipeline.py --dash    # Launch dashboard after training
"""

import sys, time, argparse, logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

BANNER = """
╔══════════════════════════════════════════════════════╗
║   CrudeEdge — India Oil Price Intelligence System    ║
║   Hybrid LSTM + Random Forest + XGBoost Forecasting  ║
╚══════════════════════════════════════════════════════╝
"""

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fast",  action="store_true")
    p.add_argument("--dash",  action="store_true")
    p.add_argument("--start", default="2010-01-01")
    return p.parse_args()


def step(n, title):
    log.info(f"\n{'═'*55}\n  STEP {n}: {title}\n{'═'*55}")


def main():
    print(BANNER)
    args = parse_args()
    t0   = time.time()

    # ── STEP 1 ────────────────────────────────────────────────────────────────
    step(1, "DATA INGESTION")
    from scripts.data_ingestion import ingest_all
    raw_df = ingest_all(start=args.start, save=True)

    # ── STEP 2 ────────────────────────────────────────────────────────────────
    step(2, "FEATURE ENGINEERING")
    from features.feature_engineering import run_feature_pipeline
    from config import TRAIN_TEST_SPLIT, LOOKBACK_WINDOW, FORECAST_HORIZON, ENSEMBLE_WEIGHTS
    pipe = run_feature_pipeline(raw_df, split_ratio=TRAIN_TEST_SPLIT)

    log.info(f"  Features        : {len(pipe['feat_cols'])}")
    log.info(f"  LSTM train shape: {pipe['X_lstm_tr'].shape}")
    log.info(f"  LSTM test  shape: {pipe['X_lstm_te'].shape}")
    log.info(f"  y_test_align len: {len(pipe['y_test_align'])}")

    # ── STEP 3 ────────────────────────────────────────────────────────────────
    step(3, "MODEL TRAINING")
    from scripts.train_models import (
        train_lstm, train_random_forest, train_xgboost,
        train_direction_models, ensemble_predict, forecast_30_days,
    )
    from config import MODELS_DIR
    import joblib

    lk = LOOKBACK_WINDOW

    if args.fast:
        log.info("  ⚡ FAST mode — skipping LSTM")
        X_tr = pipe["X_train"][lk:]; y_tr = pipe["y_train_raw"][lk:]
        X_fv = pipe["X_test_flat"][:int(len(pipe["X_test_flat"])*0.5)]
        y_fv = pipe["y_test_align"][:int(len(pipe["y_test_align"])*0.5)]
        rf   = train_random_forest(X_tr, y_tr)
        xgb  = train_xgboost(X_tr, y_tr, X_fv, y_fv)
        rf_clf, xgb_clf = train_direction_models(
            X_tr, pipe["y_train_dir"][lk:],
            pipe["X_test_flat"], pipe["y_test_dir_a"],
        )
        joblib.dump(pipe["feat_scaler"],  MODELS_DIR / "feat_scaler.pkl")
        joblib.dump(pipe["price_scaler"], MODELS_DIR / "price_scaler.pkl")
        trained = {"rf": rf, "xgb": xgb, "rf_clf": rf_clf, "xgb_clf": xgb_clf}
        lstm_available = False
    else:
        from scripts.train_models import train_all
        trained = train_all(pipe)
        lstm_available = "lstm" in trained

    # ── STEP 4 ────────────────────────────────────────────────────────────────
    step(4, "EVALUATION & VISUALISATION")
    from scripts.evaluate import (
        regression_metrics, directional_accuracy, classification_metrics,
        plot_actual_vs_predicted, plot_30_day_forecast,
        plot_feature_importance, plot_confusion_matrix,
        plot_volatility, plot_model_comparison,
    )

    y_actual    = pipe["y_test_align"]          # actual prices (USD)
    X_flat_test = pipe["X_test_flat"]           # aligned flat features
    price_sc    = pipe["price_scaler"]

    pred_dict   = {}
    reg_metrics = []

    # RF predictions (already in USD)
    rf_preds = trained["rf"].predict(X_flat_test)
    pred_dict["Random Forest"] = rf_preds
    m = regression_metrics(y_actual, rf_preds, "Random Forest")
    m["DirAcc(%)"] = directional_accuracy(y_actual, rf_preds)
    reg_metrics.append(m)

    # XGB predictions
    xgb_preds = trained["xgb"].predict(X_flat_test)
    pred_dict["XGBoost"] = xgb_preds
    m = regression_metrics(y_actual, xgb_preds, "XGBoost")
    m["DirAcc(%)"] = directional_accuracy(y_actual, xgb_preds)
    reg_metrics.append(m)

    # LSTM predictions (inverse_transform scaled → USD)
    if lstm_available:
        lstm_sc_preds = trained["lstm"].predict(pipe["X_lstm_te"], verbose=0).ravel()
        lstm_preds    = price_sc.inverse_transform(lstm_sc_preds.reshape(-1,1)).ravel()
        pred_dict["LSTM"] = lstm_preds
        m = regression_metrics(y_actual, lstm_preds, "LSTM")
        m["DirAcc(%)"] = directional_accuracy(y_actual, lstm_preds)
        reg_metrics.append(m)

        # Weighted ensemble
        n_min  = min(len(rf_preds), len(xgb_preds), len(lstm_preds))
        w      = [wt / sum(ENSEMBLE_WEIGHTS) for wt in ENSEMBLE_WEIGHTS]
        ens    = (w[0] * lstm_preds[-n_min:] +
                  w[1] * rf_preds[-n_min:]   +
                  w[2] * xgb_preds[-n_min:])
    else:
        n_min  = min(len(rf_preds), len(xgb_preds))
        ens    = 0.5 * rf_preds[-n_min:] + 0.5 * xgb_preds[-n_min:]

    pred_dict["Ensemble"] = ens
    m = regression_metrics(y_actual[-n_min:], ens, "Ensemble")
    m["DirAcc(%)"] = directional_accuracy(y_actual[-n_min:], ens)
    reg_metrics.append(m)

    # Plots
    dates_test = pipe["test"].index[lk:]
    plot_actual_vs_predicted(dates_test, y_actual, pred_dict)
    plot_volatility(pipe["df"].index, pipe["df"]["close"].values)
    plot_model_comparison(reg_metrics)
    plot_feature_importance(trained["xgb"],    pipe["feat_cols"], "XGBoost")
    plot_feature_importance(trained["rf"],     pipe["feat_cols"], "RandomForest")

    # Direction evaluation
    y_dir_true = pipe["y_test_dir_a"]
    rf_dir     = trained["rf_clf"].predict(X_flat_test)
    xgb_dir    = trained["xgb_clf"].predict(X_flat_test)
    cm_rf      = classification_metrics(y_dir_true, rf_dir,  "Direction RF")
    cm_xgb     = classification_metrics(y_dir_true, xgb_dir, "Direction XGB")
    plot_confusion_matrix(cm_rf["CM"],  "Direction_RF")
    plot_confusion_matrix(cm_xgb["CM"], "Direction_XGB")

    # 30-day forecast
    if lstm_available:
        feat_cols = pipe["feat_cols"]
        X_all     = pipe["feat_scaler"].transform(pipe["df"][feat_cols])
        fore      = forecast_30_days(
            trained["lstm"], trained["rf"], trained["xgb"],
            X_all[-lk:], X_all[-1], price_sc,
        )
        last_date  = pipe["df"].index[-1]
        fore_dates = pd.bdate_range(start=last_date + timedelta(days=1),
                                    periods=FORECAST_HORIZON)
        plot_30_day_forecast(pipe["df"].index, pipe["df"]["close"].values,
                             fore_dates, fore)

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    log.info(f"\n{'═'*55}\n  PIPELINE COMPLETE in {elapsed:.1f}s\n{'═'*55}")
    log.info("\n  Regression Results (USD price space):")
    for m in reg_metrics:
        log.info(f"    {m['Model']:22s}  RMSE=${m['RMSE']:.2f}  "
                 f"MAE=${m['MAE']:.2f}  R²={m['R²']:.4f}  "
                 f"DirAcc={m.get('DirAcc(%)',0):.1f}%")
    log.info(f"\n  Direction Models:")
    log.info(f"    RF  — Acc={cm_rf['Accuracy']:.4f}  F1={cm_rf['F1']:.4f}")
    log.info(f"    XGB — Acc={cm_xgb['Accuracy']:.4f}  F1={cm_xgb['F1']:.4f}")
    from config import DATA_PROCESSED
    log.info(f"\n  Charts → {DATA_PROCESSED}/")
    log.info(f"  Run dashboard: streamlit run dashboard/app.py")

    if args.dash:
        import subprocess
        subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard/app.py"])


if __name__ == "__main__":
    main()

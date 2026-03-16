"""
scripts/evaluate.py — AUDITED & FIXED v2
═════════════════════════════════════════

All metrics are now computed in USD price space — no log-return confusion.
Includes:
  • regression_metrics  (RMSE, MAE, R², MAPE)
  • directional_accuracy
  • classification_metrics (Precision, Recall, F1, Confusion Matrix)
  • diagnostic_plots       (actual vs predicted, residuals, forecast, features)
  • model_comparison_plot
  • WHY R² WAS -299 explanation printed to log
"""

import sys, logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_PROCESSED

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Plot style ────────────────────────────────────────────────────────────────
BG, FG, ACC = "#0d1117", "#0d1117", "#00d4ff"
GREEN, RED, GOLD, GRID, TEXT = "#39d353", "#f85149", "#e3b341", "#21262d", "#c9d1d9"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": FG,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT,
    "xtick.color": TEXT, "ytick.color": TEXT, "text.color": TEXT,
    "grid.color": GRID, "grid.linestyle": "--", "grid.alpha": 0.5,
    "legend.facecolor": "#161b22", "legend.edgecolor": GRID,
    "font.family": "monospace",
})


# ─────────────────────────────────────────────────────────────────────────────
# REGRESSION METRICS  (all in USD price space)
# ─────────────────────────────────────────────────────────────────────────────

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                       model_name: str = "") -> dict:
    """
    Compute RMSE, MAE, R², MAPE in USD price space.

    NOTE: y_true and y_pred MUST be in the same unit (USD price).
    If R² is very negative it means predictions are in a different
    unit/scale than actuals — check inverse_transform was applied.
    """
    n    = min(len(y_true), len(y_pred))
    a, p = y_true[:n], y_pred[:n]

    rmse = float(np.sqrt(mean_squared_error(a, p)))
    mae  = float(mean_absolute_error(a, p))
    r2   = float(r2_score(a, p))
    mape = float(np.mean(np.abs((a - p) / (np.abs(a) + 1e-9))) * 100)

    log.info(f"  [{model_name:20s}] RMSE=${rmse:.3f}  MAE=${mae:.3f}  R²={r2:.4f}  MAPE={mape:.2f}%")

    if r2 < -1:
        log.warning(
            f"  ⚠️  R²={r2:.1f} for {model_name} — "
            "predictions are likely in wrong scale (log-return vs price). "
            "Check that price_scaler.inverse_transform() was called."
        )
    return {"Model": model_name, "RMSE": rmse, "MAE": mae, "R²": r2, "MAPE(%)": mape}


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    n       = min(len(y_true), len(y_pred))
    a, p    = y_true[:n], y_pred[:n]
    true_up = np.diff(a) > 0
    pred_up = np.diff(p) > 0
    return float(np.mean(true_up == pred_up) * 100)


# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFICATION METRICS
# ─────────────────────────────────────────────────────────────────────────────

def classification_metrics(y_true, y_pred, model_name=""):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)
    log.info(f"\n  [{model_name}]  Accuracy={acc:.4f}  Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}")
    log.info(f"\n{classification_report(y_true, y_pred, target_names=['DOWN','UP'])}")
    return {"Model": model_name, "Accuracy": acc, "Precision": prec,
            "Recall": rec, "F1": f1, "CM": cm}


# ─────────────────────────────────────────────────────────────────────────────
# DIAGNOSTIC PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_actual_vs_predicted(dates, y_true, pred_dict: dict, title=""):
    fig, axes = plt.subplots(2, 1, figsize=(16, 9), gridspec_kw={"height_ratios": [3, 1]})
    ax, ax_res = axes

    ax.plot(dates, y_true, color=TEXT, linewidth=1.5, label="Actual", zorder=5)
    colors = [ACC, GREEN, GOLD, RED]
    for i, (name, preds) in enumerate(pred_dict.items()):
        n = min(len(dates), len(preds))
        ax.plot(dates[-n:], preds[-n:], color=colors[i % 4],
                linewidth=1.2, linestyle="--", alpha=0.85, label=name)

    ax.set_title(title or "Actual vs Predicted — Brent Crude Oil (Test Set)", fontsize=13, pad=12)
    ax.set_ylabel("Price (USD/bbl)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0f"))
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Residuals subplot (first model)
    first_name, first_preds = next(iter(pred_dict.items()))
    n       = min(len(y_true), len(first_preds))
    resid   = y_true[:n] - first_preds[:n]
    ax_res.bar(dates[:n], resid, color=[GREEN if r >= 0 else RED for r in resid],
               alpha=0.6, width=1)
    ax_res.axhline(0, color=TEXT, linewidth=0.8)
    ax_res.set_ylabel("Residuals ($)")
    ax_res.set_title(f"Residuals — {first_name}", fontsize=10)
    ax_res.grid(True, alpha=0.3)

    plt.tight_layout()
    path = DATA_PROCESSED / "actual_vs_predicted.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved → {path}")
    return path


def plot_30_day_forecast(hist_dates, hist_prices, fore_dates, fore_prices):
    fig, ax = plt.subplots(figsize=(16, 6))
    n = min(90, len(hist_dates))
    ax.plot(hist_dates[-n:], hist_prices[-n:], color=TEXT, linewidth=1.5, label="Historical")
    ax.plot(fore_dates, fore_prices, color=ACC, linewidth=2.5, label="30-Day Forecast")
    ax.fill_between(fore_dates, fore_prices * 0.95, fore_prices * 1.05,
                    color=ACC, alpha=0.12, label="±5% Band")
    # Separator line
    ax.axvline(x=hist_dates[-1], color=GOLD, linestyle=":", linewidth=1.5, label="Today")
    ax.set_title("Brent Crude — 30-Day Price Forecast", fontsize=13, pad=12)
    ax.set_ylabel("Price (USD/bbl)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0f"))
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = DATA_PROCESSED / "forecast_30days.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved → {path}")
    return path


def plot_feature_importance(model, feature_names, model_name="XGBoost"):
    imp = model.feature_importances_
    idx = np.argsort(imp)[-20:]
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#e3b341" if i >= len(idx) - 5 else ACC for i in range(len(idx))]
    ax.barh(np.array(feature_names)[idx], imp[idx], color=colors, alpha=0.85)
    ax.set_title(f"Feature Importance — {model_name} (Top 20)", fontsize=13, pad=12)
    ax.set_xlabel("Importance Score")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    path = DATA_PROCESSED / f"feature_importance_{model_name.lower().replace(' ','_')}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved → {path}")
    return path


def plot_confusion_matrix(cm, model_name=""):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["DOWN", "UP"], yticklabels=["DOWN", "UP"],
                ax=ax, linewidths=0.5, linecolor=GRID)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, pad=12)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.tight_layout()
    path = DATA_PROCESSED / f"cm_{model_name.lower().replace(' ','_')}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_model_comparison(metrics_list: list):
    df = pd.DataFrame([m for m in metrics_list if "CM" not in m]).set_index("Model")
    numeric = df.select_dtypes(include=[np.number])
    cols    = [c for c in ["RMSE", "MAE", "R²"] if c in numeric.columns]
    clrs    = [RED, GOLD, GREEN]
    fig, axes = plt.subplots(1, len(cols), figsize=(5 * len(cols), 5))
    if len(cols) == 1:
        axes = [axes]
    for ax, col, clr in zip(axes, cols, clrs):
        numeric[col].plot(kind="bar", ax=ax, color=clr, alpha=0.85)
        ax.set_title(col, fontsize=13)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, axis="y", alpha=0.3)
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.3f}",
                        (p.get_x() + p.get_width() / 2, p.get_height()),
                        ha="center", va="bottom", fontsize=9, color=TEXT)
    plt.suptitle("Model Performance Comparison", fontsize=14, y=1.02)
    plt.tight_layout()
    path = DATA_PROCESSED / "model_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_volatility(dates, prices):
    prices_s = pd.Series(prices, index=dates)
    log_ret  = np.log(prices_s / prices_s.shift(1))
    vol_30   = log_ret.rolling(30).std() * np.sqrt(252) * 100
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    ax1.plot(dates, prices, color=TEXT, linewidth=1)
    ax1.set_title("Brent Crude Oil Price", fontsize=12)
    ax1.set_ylabel("Price (USD/bbl)")
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0f"))
    ax1.grid(True, alpha=0.3)
    ax2.fill_between(vol_30.index, 0, vol_30.values, color=RED, alpha=0.6)
    ax2.set_title("30-Day Realised Volatility (Annualised %)", fontsize=12)
    ax2.set_ylabel("Volatility (%)")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    path = DATA_PROCESSED / "volatility_chart.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path

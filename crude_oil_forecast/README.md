# 🛢️ CrudeEdge — India Crude Oil Price Intelligence System

**BTech Major Project | Hybrid AI Forecasting Platform**

> Predict Brent crude oil prices affecting India using a **LSTM + Random Forest + XGBoost ensemble**, with a real-time interactive dashboard.

---

## 📐 System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  DATA SOURCES                                                │
│  ├── Brent Crude Prices    (yfinance)                        │
│  ├── WTI Crude Prices      (yfinance)                        │
│  ├── USD/INR Exchange Rate (yfinance)                        │
│  ├── Macro Indicators      (FRED API)                        │
│  └── News Sentiment        (NewsAPI + TextBlob)              │
└──────────────────┬───────────────────────────────────────────┘
                   ▼
┌──────────────────────────────────────────────────────────────┐
│  FEATURE ENGINEERING  (60+ features)                         │
│  ├── Lag features          (lag_1, lag_3, lag_7, lag_14...)  │
│  ├── Rolling statistics    (mean, std, skew — 7/14/30 days)  │
│  ├── Technical Indicators  (RSI, MACD, Bollinger Bands)      │
│  ├── Momentum              (ROC, momentum_7, momentum_14)    │
│  ├── India-specific        (Brent in INR = Brent × USD/INR)  │
│  └── Calendar              (sin/cos day-of-week, month)      │
└──────────────────┬───────────────────────────────────────────┘
                   ▼
┌──────────────────────────────────────────────────────────────┐
│  HYBRID MODEL                                                │
│  ┌──────────┐  ┌──────────────────┐  ┌─────────────────┐   │
│  │  LSTM    │  │  Random Forest   │  │    XGBoost      │   │
│  │  (45%)   │  │     (25%)        │  │     (30%)       │   │
│  └────┬─────┘  └────────┬─────────┘  └────────┬────────┘   │
│       └────────────────┬┘────────────────────┘             │
│                         ▼                                   │
│              Weighted Ensemble Prediction                    │
└──────────────────┬───────────────────────────────────────────┘
                   ▼
┌──────────────────────────────────────────────────────────────┐
│  OUTPUTS                                                     │
│  ├── Price Forecast (30 days)                                │
│  ├── Direction Prediction (UP ↑ / DOWN ↓)                    │
│  └── Streamlit Dashboard                                     │
└──────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
crude_oil_forecast/
├── config.py                         # All hyperparameters & paths
├── run_pipeline.py                   # 🚀 Master runner — run this first!
├── requirements.txt
├── .env.example                      # API key template
│
├── data/
│   ├── raw/          raw_merged.csv  # Auto-created on first run
│   └── processed/    features.csv + charts
│
├── scripts/
│   ├── data_ingestion.py             # Fetch all data
│   ├── train_models.py               # LSTM + RF + XGBoost + Classifiers
│   └── evaluate.py                   # Metrics + Charts
│
├── features/
│   └── feature_engineering.py        # 60+ feature pipeline
│
├── models/                           # Saved model weights (auto-created)
│   ├── lstm_final.keras
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── direction_rf.pkl
│   ├── direction_xgb.pkl
│   ├── feat_scaler.pkl
│   └── log_scaler.pkl
│
├── dashboard/
│   └── app.py                        # Streamlit dashboard
│
├── notebooks/
│   └── crude_oil_forecast_walkthrough.py
│
└── utils/
    └── helpers.py
```

---

## ⚡ Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

For GPU support (faster LSTM training):
```bash
pip install tensorflow-gpu
```

### 2. Configure API Keys (Optional)

```bash
cp .env.example .env
```

Edit `.env` and add your keys. **All keys are optional** — the system uses synthetic/fallback data if keys are absent.

| Key | Source | Cost |
|-----|--------|------|
| `FRED_API_KEY` | [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html) | Free |
| `EIA_API_KEY` | [eia.gov/opendata](https://www.eia.gov/opendata/) | Free |
| `NEWS_API_KEY` | [newsapi.org/register](https://newsapi.org/register) | Free (100 req/day) |

### 3. Run the Full Pipeline

```bash
# Full training (takes ~15 min on CPU)
python run_pipeline.py

# Fast mode — skip LSTM, train only RF + XGB (~2 min)
python run_pipeline.py --fast

# Train + immediately launch dashboard
python run_pipeline.py --dash
```

### 4. Launch the Dashboard

```bash
streamlit run dashboard/app.py
```

Open your browser at `http://localhost:8501`

---

## 📊 Dashboard Features

| Panel | Description |
|-------|-------------|
| 💰 Price Ticker | Current Brent crude in USD + INR, live change |
| ↑↓ Direction | Next-day UP/DOWN prediction + confidence % |
| 📈 Price Chart | Interactive OHLCV candlestick with volume |
| 🔮 Forecast | 30-day forecast with ±5% confidence band |
| 🎯 Features | Top-20 feature importance (XGBoost + RF) |
| 📊 Metrics | RMSE, MAE, R², Directional Accuracy |
| 🌡️ Volatility | 7-day and 30-day realised volatility |
| 🔁 Retrain | One-click retraining from new data |

---

## 🧠 Model Details

### LSTM Architecture
```
Input → LSTM(128, return_seq=True) → Dropout(0.2) → BatchNorm
      → LSTM(64) → Dropout(0.2) → BatchNorm
      → Dense(32, relu) → Dense(1)
Loss: Huber (robust to outlier prices)
Optimizer: Adam (lr=0.001)
Target: log-returns (more stable than raw prices)
```

### Feature Engineering Highlights
- **60 days lookback** for LSTM sequences
- **RSI(14)** — overbought/oversold momentum
- **MACD** — trend direction and strength  
- **Bollinger Bands** — volatility channel + %B
- **Brent in INR** = Brent(USD) × USD/INR — India-specific
- **Realised volatility** (7/14/30 day annualised)
- **Sine/Cosine calendar encoding** — no arbitrary ordinality

### Why Log Returns Instead of Raw Prices?
Raw oil prices are non-stationary (they trend). Log returns:
- Are approximately stationary (more predictable)
- Have symmetric distribution
- Allow price reconstruction: `price[t] = price[t-1] × exp(log_return[t])`

---

## 📈 Expected Performance

| Model | RMSE (USD) | MAE (USD) | Directional Acc |
|-------|-----------|----------|----------------|
| Random Forest | ~3.5 | ~2.2 | ~58% |
| XGBoost | ~3.2 | ~2.0 | ~60% |
| LSTM | ~3.8 | ~2.5 | ~56% |
| **Ensemble** | **~2.9** | **~1.8** | **~62%** |

*Results vary with data period and market regime. Oil markets are inherently noisy.*

---

## 🔬 For Students — Key Learning Points

1. **Time-series vs tabular data**: Why we can't shuffle oil price data
2. **Data leakage prevention**: Scaler fit only on train set, never test
3. **Log returns**: Why we predict returns instead of raw prices
4. **Ensemble methods**: How combining weak learners reduces variance
5. **LSTM sequences**: The 3D input format `(samples, timesteps, features)`
6. **Directional accuracy**: Why it matters more than RMSE in trading
7. **Feature engineering**: How domain knowledge (India oil context) improves ML

---

## 🛠️ Tech Stack

| Category | Library |
|----------|---------|
| Deep Learning | TensorFlow / Keras |
| Tree Models | scikit-learn, XGBoost |
| Data | pandas, numpy |
| Finance APIs | yfinance, fredapi |
| NLP | TextBlob, newsapi-python |
| Visualisation | Plotly, Matplotlib, Seaborn |
| Dashboard | Streamlit |
| Model Storage | joblib |

---

## ⚠️ Disclaimer

This project is for **educational purposes only**. Oil price prediction is inherently uncertain. Do not use model outputs for financial decisions.

---

*CrudeEdge — BTech Major Project | India Oil Price Intelligence*

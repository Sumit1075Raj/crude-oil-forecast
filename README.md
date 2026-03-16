# 🛢️ CrudeEdge — Hybrid AI System for Crude Oil Price Forecasting

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-green)
![Deep Learning](https://img.shields.io/badge/Deep-Learning-red)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

CrudeEdge is a **hybrid AI-powered oil price forecasting system** designed to analyze and predict crude oil price movements affecting India.
The project combines **deep learning and ensemble machine learning models (LSTM, Random Forest, and XGBoost)** to capture both **time-series trends and nonlinear relationships** in oil market data.

The system integrates **real-world financial and macroeconomic datasets**, performs **advanced feature engineering**, and uses a **weighted ensemble strategy** to generate more reliable predictions.

---

# 🚀 Key Features

* Hybrid AI model combining **LSTM + Random Forest + XGBoost**
* Forecast **future crude oil prices**
* Predict **price movement direction (UP ↑ / DOWN ↓)**
* Feature engineering with **technical indicators and macroeconomic variables**
* **Weighted ensemble learning** for improved prediction accuracy
* **Interactive Streamlit dashboard** for visualization
* Model evaluation using:

  * RMSE
  * MAE
  * R² Score
  * Directional Accuracy

---

# 🧠 Hybrid Model Architecture

The system integrates **three different models** to leverage their individual strengths.

### LSTM (Long Short-Term Memory)

* Captures sequential patterns in time-series data
* Learns long-term dependencies in oil price trends

### Random Forest

* Handles nonlinear feature relationships
* Reduces overfitting through bagging

### XGBoost

* Gradient boosting model optimized for structured data
* Provides strong predictive performance

### Ensemble Strategy

Final predictions are generated using a **weighted combination of the three models**:

Final Prediction =
(0.45 × LSTM) + (0.25 × Random Forest) + (0.30 × XGBoost)

---

# 📊 System Workflow

1. Data Collection from multiple sources (oil prices, macro indicators, exchange rates)
2. Data Cleaning and Preprocessing
3. Feature Engineering (technical indicators, lag features, rolling statistics)
4. Model Training
5. Ensemble Prediction
6. Visualization and Dashboard Deployment

---

# 📁 Project Structure

```
CrudeEdge
│
├── data
│   ├── raw
│   └── processed
│
├── scripts
│   ├── data_ingestion.py
│   ├── train_models.py
│   └── evaluate.py
│
├── features
│   └── feature_engineering.py
│
├── models
│   └── saved model files
│
├── dashboard
│   └── app.py
│
├── notebooks
│
├── run_pipeline.py
├── requirements.txt
└── README.md
```

---

# ⚡ Installation

Clone the repository:

```
git clone https://github.com/yourusername/crudeedge.git
cd crudeedge
```

Install dependencies:

```
pip install -r requirements.txt
```

---

# ▶️ Run the Project

Run the full pipeline:

```
python run_pipeline.py
```

Launch the dashboard:

```
streamlit run dashboard/app.py
```

---

# 📈 Dashboard Features

* Live crude oil price tracker
* Interactive price charts
* 30-day forecast visualization
* Direction prediction with confidence score
* Feature importance visualization

---

# 🛠️ Tech Stack

| Category        | Tools                 |
| --------------- | --------------------- |
| Programming     | Python                |
| ML Libraries    | Scikit-learn, XGBoost |
| Deep Learning   | TensorFlow / Keras    |
| Data Processing | Pandas, NumPy         |
| Visualization   | Plotly, Matplotlib    |
| Dashboard       | Streamlit             |

---

# 🎯 Project Objective

The goal of this project is to demonstrate how **hybrid AI models combining deep learning and ensemble methods** can improve prediction accuracy in complex financial markets like crude oil trading.

---

# ⚠️ Disclaimer

This project is created for **educational and research purposes only**.
Oil price forecasting is inherently uncertain and should not be used for financial decisions.

---

# 👨‍💻 Author

**Sumit Raj**
B.Tech Computer Science Engineering
Machine Learning & Data Science Enthusiast

#  AeroSense — Automated AQI Prediction & Early Warning System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/TensorFlow%2FKeras-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/Hopsworks-1B8EF2?style=for-the-badge&logo=databricks&logoColor=white"/>
  <img src="https://img.shields.io/badge/SHAP-8A2BE2?style=for-the-badge&logo=ai&logoColor=white"/>
  <img src="https://img.shields.io/badge/GitHub_Actions-2088FF?style=for-the-badge&logo=githubactions&logoColor=white"/>
  <img src="https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge"/>
</p>

<p align="center">
  A production-ready MLOps system that delivers continuous <b>3-day (24h/48h/72h) Air Quality Index forecasts</b> and proactive hazardous-pollution alerts for major Pakistani cities — powered by <b>FastAPI, Hopsworks, SHAP & Streamlit</b>.
</p>

---

##  Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Setup & Installation](#-setup--installation)
- [Configuration](#-configuration)
- [How It Works](#-how-it-works)
- [API Endpoints](#-api-endpoints)
- [Automation & CI/CD](#-automation--cicd)
- [Roadmap](#-roadmap)
- [Author](#-author)

---

##  Overview

**AeroSense** is an automated, production-grade Machine Learning system that forecasts Air Quality Index (AQI) for major Pakistani metropolises — **Lahore, Karachi, and Islamabad** — at three forecast horizons: **+24h, +48h, and +72h**.

Traditional air-quality platforms are reactive, reporting pollution levels only after harmful air has already accumulated. AeroSense instead operationalizes a fully **decoupled MLOps architecture** that continuously ingests meteorological and particulate data, engineers physics-informed features, benchmarks statistical, ensemble, and deep learning models, explains every prediction using **SHAP**, and serves forecasts through a **FastAPI backend** and an **IQAir-inspired Streamlit dashboard**.

---

##  Features

| Feature | Description |
|:---|:---|
|  Multi-Source Ingestion | Pulls live pollutant data (AQICN), live weather (OpenWeather), and historical climate data (Open-Meteo) |
|  Physics-Informed Features | Lag features, rolling statistics, cyclical time encodings, and domain-derived ratios (e.g. PM2.5/PM10) |
|  Multi-Horizon Forecasting | Predicts AQI at +24h, +48h, and +72h with strict time-series leakage prevention |
|  Model Benchmarking | Compares Persistence Baseline, Ridge Regression, Random Forest, and LSTM models |
|  Explainable AI (XAI) | SHAP-based global and local explanations translated into plain-language drivers |
|  Early Warning Alerts | Automated hazard-tier alerts (Moderate → Emergency) with health guidance |
|  Decoupled MLOps | Hopsworks Feature Store + Model Registry with scheduled GitHub Actions pipelines |
|  REST API | Full FastAPI service for forecasts, alerts, and model leaderboard access |
|  Interactive Dashboard | Streamlit web app with live AQI cards, forecast curves, and SHAP driver visuals |

---

##  Tech Stack

| Tool | Purpose |
|:---|:---|
| Python | Core language |
| FastAPI | REST API backend (forecasts, alerts, model metadata) |
| Streamlit + Altair | Interactive dashboard & forecast visualizations |
| Hopsworks Feature Store | Online/offline feature storage & leakage-free training |
| Hopsworks Model Registry | Model artifact versioning & deployment |
| scikit-learn | Ridge Regression & Random Forest models |
| TensorFlow / Keras | LSTM sequence model for multi-step forecasting |
| SHAP | Model explainability (global & local feature attribution) |
| AQICN / OpenWeather / Open-Meteo | Live and historical data sources |
| GitHub Actions | Scheduled feature, training, and inference pipelines |

---

##  Project Structure

```
pearls-aqi-predictor-main/
│
├── api/
│   └── main.py                     # FastAPI application & routes
│
├── ml/
│   ├── clients/                    # API clients (AQICN, OpenWeather, Open-Meteo)
│   ├── features/
│   │   └── engineering.py          # Feature engineering (lags, rolling stats, cyclical time)
│   ├── storage/                    # Hopsworks Feature Store integration
│   ├── training/
│   │   └── models.py               # Model definitions (Ridge, RF, LSTM)
│   ├── pipelines/
│   │   ├── training_pipeline.py    # Model training & registration
│   │   ├── inference_pipeline.py   # +24h/+48h/+72h prediction pipeline
│   │   ├── train_lstm.py           # LSTM training routine
│   │   ├── explain.py              # SHAP explainability pipeline
│   │   └── alerts.py               # Early warning alert classifier
│   ├── analysis/
│   │   └── eda.py                  # Exploratory data analysis
│   └── tools/                      # Utility & smoke-test scripts
│
├── streamlit_app/
│   └── app.py                      # Streamlit dashboard application
│
├── data/                           # Cached / local data directory
│
├── .github/
│   └── workflows/
│       ├── feature-inference.yml   # Hourly ingestion + inference pipeline
│       └── training.yml            # Daily model training pipeline
│
├── cities.json                     # Target city configuration
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

---

##  Setup & Installation

**1. Clone the repository**

```bash
git clone https://github.com/Sadia446/pearls-aqi-predictor.git
cd pearls-aqi-predictor-main
```

**2. Create a virtual environment** (recommended)

```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Configure environment variables**

Create a `.env` file in the project root with your API keys:

```bash
AQICN_API_KEY=your_aqicn_key
OPENWEATHER_API_KEY=your_openweather_key
HOPSWORKS_API_KEY=your_hopsworks_key
```

**5. Run the feature & training pipelines**

```bash
python ml/pipelines/training_pipeline.py
python ml/pipelines/inference_pipeline.py
```

**6. Start the API**

```bash
uvicorn api.main:app --reload
```

The API will be available at `http://localhost:8000` (interactive docs at `/docs`).

**7. Launch the dashboard**

```bash
streamlit run streamlit_app/app.py
```

---

##  Configuration

Core settings are defined via `cities.json` and environment variables — no separate config file is required:

- **Target Cities**: Lahore, Karachi, Islamabad (extendable via `cities.json`)
- **Forecast Horizons**: +24h, +48h, +72h
- **Feature Store**: Hopsworks (online & offline tables)
- **Model Registry**: Hopsworks Model Registry
- **Deployment Rule**: A lighter model is preferred if its RMSE is within 2% of the top-performing model

---

##  API Endpoints

### **GET /health**
```bash
curl http://localhost:8000/health
```
Returns liveness status, feature store status, and timestamp of the latest observations.

### **GET /cities**
```bash
curl http://localhost:8000/cities
```
Returns covered cities with current observed AQI, PM2.5, and temperature.

### **GET /current/{city_id}**
```bash
curl http://localhost:8000/current/lahore
```
Returns detailed real-time atmospheric measurements for a given city.

### **GET /forecast/{city_id}**
```bash
curl http://localhost:8000/forecast/lahore
```
Returns the 3-day forecast breakdown (+24h, +48h, +72h) along with top SHAP drivers.

### **GET /alerts**
```bash
curl http://localhost:8000/alerts
```
Returns active public health alerts for cities projected to exceed safe AQI thresholds.

### **GET /models**
```bash
curl http://localhost:8000/models
```
Returns the model benchmark leaderboard and active model registry metadata.

---

##  Automation & CI/CD

All pipelines run automatically via **GitHub Actions**:

| Workflow | Schedule | Purpose |
|:---|:---|:---|
| `feature-inference.yml` | Hourly | Ingests live data → updates feature store → generates predictions → writes alerts |
| `training.yml` | Daily @ 00:30 UTC | Retrains & benchmarks models → registers top artifact → refreshes SHAP explanations |
| `ci.yml` / `benchmark.yml` | On push/PR | Code quality checks, unit tests, and performance baseline verification |


---

##  Author

**Sadia Noreen**
*Software Engineering Graduate | AI & ML Enthusiast*

---

<p align="center">If you found this helpful, consider giving it a star!⭐</p>

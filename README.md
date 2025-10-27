# 🌍 AQI Prediction Dashboard

A real-time **Air Quality Index (AQI) prediction and monitoring dashboard** for **Sukkur, Pakistan**, built using **Streamlit**, **Hopsworks Feature Store**, **Machine Learning**, and the **OpenWeather API**.

## 🎬 Demo Video

[![AQI Dashboard Demo](<img width="1350" height="595" alt="Image" src="https://github.com/user-attachments/assets/39dce7de-1807-4cae-9c47-e7d2853ec85d" />)](https://github.com/user-attachments/assets/5597f057-5c51-4335-81cd-d52c1ba955c0)

> 🎥 Click the thumbnail above to watch the full demo showcasing real-time AQI tracking, pollutant visualization, and 3-day forecasting for Sukkur, Pakistan.

---

## 📘 Overview

This dashboard provides **live air quality data**, **3-day AQI forecasts**, and **pollutant breakdowns** (PM2.5, PM10, NO₂, O₃, SO₂, CO) using the **US EPA AQI standard**.  
It combines real-time data ingestion, ML-based forecasting, and web visualization for a complete MLOps demonstration.

---

## 🚀 Features

- **Live AQI & Pollutant Levels**
- **3-Day AQI Forecasts** using ML models
- **Interactive Historical Charts**
- **Automated Data Updates & Model Retraining**
- **Responsive Streamlit Dashboard**

---

## 🧰 Technologies

- **Frontend:** Streamlit, Plotly  
- **Backend:** Python, pandas, numpy, scikit-learn  
- **Data & Model Management:** Hopsworks Feature Store  
- **APIs:** OpenWeather API  
- **Automation:** GitHub Actions (CI/CD)

---

## ⚙️ Installation

```bash
# Clone repository
git clone https://github.com/yourusername/10_Pearls_AQI_Predict_Bot.git
cd 10_Pearls_AQI_Predict_Bot

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Mac/Linux
.venv\Scripts\activate   # Windows

# Install dependencies
pip install -r dashboard/requirements.txt
```

Create a `.env` file in the project root with:
```
HOPSWORKS_API_KEY=your_hopsworks_api_key
HOPSWORKS_PROJECT_NAME=your_project_name
OPENWEATHER_API_KEY=your_openweather_api_key
```

---

## ▶️ Usage

```bash
# Train model (if needed)
python src/train_model_fast.py

# Launch dashboard
streamlit run dashboard/streamlit_app.py
```
Then open [http://localhost:8501](http://localhost:8501).

---

## 🪪 License

Built and maintained by [Soyam Kapoor].
> Developed as part of the **10 Pearls AQI Prediction project** — for educational and demonstration purposes.

---




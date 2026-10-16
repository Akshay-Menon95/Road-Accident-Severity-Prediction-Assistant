# 🚗 AI Road Safety Navigation & Accident Severity Prediction Assistant

An intelligent, machine-learning-powered road safety co-pilot and accident risk assessment dashboard. This application combines route navigation, real-time weather telemetry, accident blackspot proximity scanning, and predictive machine learning models to identify high-risk travel corridors and assess accident severity.

---

## 🌟 Key Features

- **🗺️ Intelligent Route Navigation & Mapping**
  - Interactive route mapping powered by **Folium** and **OSRM (Open Source Routing Machine)**.
  - Geocoding support across India using **Nominatim / GeoPy** for effortless source-destination planning.
- **⚠️ Active Accident Blackspot Detection**
  - Proximity scanning engine using geodesic distance calculations.
  - Detects and visualizes accident-prone blackspots along the calculated driving route.
- **🌦️ Live Weather & Road Surface Telemetry**
  - Real-time weather diagnostics via **Open-Meteo API** (zero API keys required).
  - Automatically translates precipitation, visibility, and weather codes into live road condition assessments (Dry, Wet, Foggy).
- **🧠 Machine Learning Severity Classifier**
  - Trained **Decision Tree Classifier** predicting accident severity (**Minor**, **Serious**, or **Fatal**).
  - Evaluates critical risk factors: time of day, weather conditions, road surface conditions, vehicle type, and number of vehicles involved.
- **🛡️ Actionable Driver Advisories**
  - Generates context-aware driving advisories, speed warnings, and hazard mitigation alerts.

---

## 📂 Repository Structure

```text
Road-Accident-Severity-Prediction-Assistant/
├── app.py                      # Main Streamlit web dashboard & interactive co-pilot
├── navigation_engine.py        # Routing, geocoding, weather integration, & hazard scanner
├── blackspots.py               # Geocoded accident blackspots database
├── train.py                    # Model training, validation, & serialization pipeline
├── fetch_kerala_real_data.py   # Historical weather & accident data ingestion script
├── requirements.txt            # Project dependencies
├── .gitignore                  # Git ignore rules for virtual environments & cache
├── data/
│   └── kerala_accident_data.csv# Accident & transit dataset
└── models/
    ├── decision_tree_model.joblib # Serialized Decision Tree model
    ├── feature_columns.joblib     # Feature schema metadata
    └── label_encoders.joblib      # Categorical encoders
```

---

## 🚀 Quickstart Guide

### 1. Clone the Repository

```bash
git clone https://github.com/Akshay-Menon95/Road-Accident-Severity-Prediction-Assistant.git
cd Road-Accident-Severity-Prediction-Assistant
```

### 2. Set Up a Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**On Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application

```bash
streamlit run app.py
```

Open your browser and navigate to `http://localhost:8501`.

---

## 🔬 Model Training & Data Pipeline

Pre-trained models are already provided in the [`models/`](models/) directory. If you wish to retrain or update the model with new data:

```bash
python train.py
```

To fetch real historical Kerala weather & accident transit data:
```bash
python fetch_kerala_real_data.py
```

---

## 🛠️ Tech Stack

- **Frontend & App Framework**: [Streamlit](https://streamlit.io/), [Folium](https://python-visualization.github.io/folium/), [streamlit-folium](https://github.com/randyzwitch/streamlit-folium)
- **Machine Learning**: [Scikit-learn](https://scikit-learn.org/), [Joblib](https://joblib.readthedocs.io/)
- **Data Engineering**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Geospatial & Navigation**: [GeoPy](https://geopy.readthedocs.io/), [Polyline](https://github.com/frederick/polyline), OSRM API
- **Weather Telemetry**: [Open-Meteo API](https://open-meteo.com/)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

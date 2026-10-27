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
  - Evaluates risk factors recorded in the crash data: time of day, weather, road surface, vehicle type, and number of vehicles involved. Live weather is translated into the exact labels the model was trained on (`label_maps.py`).
  - See [Model performance](#-model-performance) for honest accuracy numbers and limitations.
- **🛡️ Actionable Driver Advisories**
  - Context-aware advisories for speed (limit drops from 70 to 50 km/h in rain, fog or snow), slippery roads, low visibility, and blackspots you are in or approaching (5 km look-ahead).
  - Vehicle speed drives these advisories only; the crash dataset has no speed field, so speed is not a model input.

---

## 📂 Repository Structure

```text
Road-Accident-Severity-Prediction-Assistant/
├── app.py                      # Main Streamlit web dashboard & interactive co-pilot
├── navigation_engine.py        # Routing, geocoding, weather integration, & hazard scanner
├── label_maps.py               # Shared vehicle / weather / road labels (training = live inputs)
├── risk_model.py               # Loads artifacts and predicts severity from dashboard inputs
├── ui_components.py            # Styling & HTML building blocks for the dashboard
├── blackspots.py               # Geocoded accident blackspots database
├── train.py                    # Model training, validation, & serialization pipeline
├── tests/test_pipeline.py      # Pytest suite for mappings, model artifacts, hazards, advisories
├── requirements.txt            # Project dependencies (requirements-dev.txt adds pytest)
├── data/
│   └── RTA_Dataset.csv         # Road Traffic Accident dataset (Addis Ababa, Ethiopia)
└── models/
    ├── decision_tree_model.joblib # Serialized Decision Tree model
    ├── feature_columns.joblib     # Feature schema metadata
    ├── label_encoders.joblib      # Categorical encoders
    └── metrics.json               # Held-out evaluation written by train.py
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

Pre-trained models are already provided in the [`models/`](models/) directory. If you wish to retrain (for example after upgrading scikit-learn, since the `.joblib` files are tied to the version that wrote them):

```bash
python train.py
```

`train.py` maps the raw dataset onto the shared labels in `label_maps.py`, makes a stratified 80/20 split, tunes the tree depth and leaf size by 5-fold cross-validation on macro-F1 (with balanced class weights), and writes the held-out results to `models/metrics.json`.

Run the tests with:
```bash
pip install -r requirements-dev.txt
pytest
```

---

## 📊 Model performance

Evaluated on a stratified held-out set of 2,274 crashes from the RTA dataset (11,366 usable rows):

| Metric | Value |
|---|---|
| Accuracy | 44.6% |
| Balanced accuracy | 45.9% |
| Macro F1 | 0.305 |
| Fatal recall / Serious recall / Minor recall | 62% / 29% / 47% |

Please read these numbers carefully:

- **Accuracy is deliberately low.** 84% of crashes are "Minor", so always predicting "Minor" scores 84% accuracy but is useless for road safety. The model is class-weighted to flag Serious and Fatal crashes, which costs raw accuracy and produces many false alarms (Fatal precision is 2%, because only about 1% of crashes are fatal).
- **The signal is weak.** Hour, weather, road surface, vehicle type and vehicle count explain little about severity. Treat the output as a relative risk indicator, not a calibrated probability or a prediction for a specific trip.
- **Different country.** The data comes from Addis Ababa, Ethiopia, while the navigation and blackspots cover India.
- **Earlier versions reported ~90% accuracy** because vehicle speed was synthesised from the severity label itself (target leakage). That feature has been removed.

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

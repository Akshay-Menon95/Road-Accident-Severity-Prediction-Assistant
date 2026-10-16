# app.py
import json
import os
import streamlit as st
import folium
from streamlit_folium import st_folium
import joblib
import numpy as np
import pandas as pd
from navigation_engine import (
    get_driving_route,
    get_live_weather,
    scan_hazards,
    generate_advisory,
    geocode_location
)
import ui_components as ui

# Attempt to load pan-India blackspots; fallback to Kerala blackspots if needed
try:
    from navigation_engine import NATIONAL_BLACKSPOTS as ACTIVE_BLACKSPOTS
except ImportError:
    try:
        from blackspots import KERALA_BLACKSPOTS as ACTIVE_BLACKSPOTS
    except ImportError:
        ACTIVE_BLACKSPOTS = []

st.set_page_config(
    page_title="AI Road Safety Navigation & Live Risk Co-Pilot",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(ui.GLOBAL_CSS, unsafe_allow_html=True)


def html(markup):
    st.markdown(markup, unsafe_allow_html=True)


# 1. Load Trained Machine Learning Model & Preprocessing Encoders
@st.cache_resource
def load_ml():
    model = joblib.load("models/decision_tree_model.joblib")
    encoders = joblib.load("models/label_encoders.joblib")
    features = joblib.load("models/feature_columns.joblib")
    return model, encoders, features

try:
    model, encoders, feature_columns = load_ml()
except Exception as e:
    html(ui.error_state("Model artifacts not found", f"Run 'python train.py' first. Error: {e}"))
    st.stop()

# 2. Cached API Functions
@st.cache_data(show_spinner=False)
def get_cached_route(start_name, end_name):
    start_coords = geocode_location(start_name)
    end_coords = geocode_location(end_name)

    if not start_coords:
        return None, f"Could not find coordinates for: '{start_name}'. Try adding state/city name."
    if not end_coords:
        return None, f"Could not find coordinates for: '{end_name}'. Try adding state/city name."

    try:
        route = get_driving_route(start_coords[0], start_coords[1], end_coords[0], end_coords[1])
        return route, "Success"
    except Exception as e:
        return None, f"Routing engine error: {e}"

@st.cache_data(show_spinner=False)
def get_cached_weather(lat, lon):
    return get_live_weather(lat, lon)

def safe_encode(encoder, value, default_val=0):
    """Safely encodes categorical values without throwing ValueError on unseen labels."""
    if encoder is None:
        return default_val
    try:
        if value in encoder.classes_:
            return int(encoder.transform([value])[0])
        return default_val
    except Exception:
        return default_val

# Live-weather labels from navigation_engine -> equivalents in the RTA training data.
WEATHER_ALIASES = {"Clear": "Normal", "Rainy": "Raining", "Foggy": "Fog or mist"}
ROAD_ALIASES = {"Wet": "Wet or damp"}

def model_label(encoder, value, aliases):
    """Returns the label the encoder was trained on for a live-telemetry value."""
    classes = getattr(encoder, "classes_", [])
    if value in classes:
        return value
    return aliases.get(value, value)

METRICS_PATH = "models/metrics.json"
HISTORY_PATH = "models/metrics_history.csv"

@st.cache_data(show_spinner=False)
def load_metrics(mtime):
    """Evaluation results written by train.py / ml_eval.py (mtime busts the cache after a retrain)."""
    with open(METRICS_PATH) as f:
        return json.load(f)

@st.cache_data(show_spinner=False)
def load_history(mtime):
    return pd.read_csv(HISTORY_PATH)

def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(a))

@st.cache_data(show_spinner=False)
def route_profile(waypoints, total_km, blackspots):
    """Cumulative distance (km) at each waypoint, plus the blackspots the route passes through."""
    pts = np.asarray(waypoints, dtype=float)
    seg = haversine_km(pts[:-1, 0], pts[:-1, 1], pts[1:, 0], pts[1:, 1])
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] > 0:
        cum = cum * (total_km / cum[-1])  # align with the routing engine's reported distance

    on_route = []
    for spot in blackspots:
        dist_m = haversine_km(pts[:, 0], pts[:, 1], spot["lat"], spot["lon"]) * 1000
        inside = np.flatnonzero(dist_m <= spot.get("radius_meters", 1000))
        if inside.size:
            on_route.append({
                **spot,
                "km": float(cum[int(np.argmin(dist_m))]),
                "enter_km": float(cum[inside[0]]),
                "exit_km": float(cum[inside[-1]]),
            })
    on_route.sort(key=lambda s: s["km"])
    return cum, on_route

# 3. Sidebar: Trip Configuration & Driver Profile
with st.sidebar:
    html(ui.side_brand())

    html(ui.side_label("Route"))
    start_loc = st.text_input("Start location", value="Palakkad", placeholder="Anywhere in India")
    end_loc = st.text_input("Destination", value="Thrissur", placeholder="City, landmark or address")

    html(ui.side_label("Vehicle"))
    vehicle_classes = (
        list(encoders["Vehicle_Type"].classes_)
        if "Vehicle_Type" in encoders and hasattr(encoders["Vehicle_Type"], "classes_")
        else ["Car", "Two-Wheeler", "Truck", "Bus", "Auto-Rickshaw"]
    )
    vehicle_type = st.selectbox("Vehicle type", vehicle_classes, format_func=ui.pretty_vehicle)
    vehicle_speed = st.slider("Current speed (km/h)", min_value=10, max_value=150, value=65, step=1)

    html(ui.side_label("Scenario"))
    time_hour = st.slider("Time of day (hour)", min_value=0, max_value=23, value=15)
    clock, day_phase = ui.hour_label(time_hour)
    html(f'<div class="side-hint"><b>{clock}</b> · {day_phase}</div>')
    num_vehicles = st.slider("Vehicles involved (collision scenario)", min_value=1, max_value=6, value=2)

    html('<div class="footnote">Predictions are statistical estimates from historical data '
         'and are not a substitute for attentive driving.</div>')

# 4. Route Computation
with st.spinner("Calculating route geometry and hazard intersections..."):
    route_data, status_msg = get_cached_route(start_loc, end_loc)

if not route_data:
    html(ui.hero(start_loc, end_loc))
    html(ui.error_state("We couldn't plan that route", status_msg))
    st.stop()

waypoints = route_data["waypoints"]
distance_km = route_data["distance_km"]
cum_km, route_hazards = route_profile(waypoints, distance_km, ACTIVE_BLACKSPOTS)

html(ui.hero(start_loc, end_loc, distance_km))
kpi_slot = st.container()  # filled once the prediction is available

col_map, col_hud = st.columns([1.55, 1], gap="large")

with col_map:
    html(ui.panel_title("🛰️ Live navigation", "Drag to simulate vehicle movement"))
    progress = st.slider("Trip progress", 0, 100, 0, format="%d%%", label_visibility="collapsed")

step_idx = int(min(np.searchsorted(cum_km, progress / 100 * cum_km[-1]), len(waypoints) - 1))
current_loc = waypoints[step_idx]
covered_km = float(cum_km[step_idx])

# 5. Real-Time Telemetry Retrieval
weather = get_cached_weather(round(current_loc[0], 2), round(current_loc[1], 2))
hazard = scan_hazards(current_loc[0], current_loc[1])

# 6. Adaptive Model Feature Alignment
# Supports both Kerala models and raw RTA/Kaggle models without schema collisions
weather_label = model_label(encoders.get("Weather_Condition"), weather.get("Weather_Condition", "Clear"), WEATHER_ALIASES)
road_label = model_label(encoders.get("Road_Condition"), weather.get("Road_Condition", "Dry"), ROAD_ALIASES)
candidate_row = {
    "Vehicle_Speed": vehicle_speed,
    "Time_Hour": time_hour,
    "Number_of_Vehicles": num_vehicles,
    "District": safe_encode(encoders.get("District"), "Palakkad"),
    "State": safe_encode(encoders.get("State"), "Kerala"),
    "Road_Type": safe_encode(encoders.get("Road_Type"), "National Highway"),
    "Traffic_Condition": safe_encode(encoders.get("Traffic_Condition"), "Medium"),
    "Weather_Condition": safe_encode(encoders.get("Weather_Condition"), weather_label),
    "Road_Condition": safe_encode(encoders.get("Road_Condition"), road_label),
    "Vehicle_Type": safe_encode(encoders.get("Vehicle_Type"), vehicle_type),
    "Precipitation_mm": weather.get("Precipitation_mm", 0.0),
    "Wind_Speed_kmh": weather.get("Wind_Speed_kmh", 10.0)
}

# Construct dataframe adhering strictly to feature_columns trained in train.py
input_dict = {col: candidate_row.get(col, 0) for col in feature_columns}
input_df = pd.DataFrame([input_dict])[feature_columns]

pred_idx = model.predict(input_df)[0]
severity = encoders["Severity"].inverse_transform([pred_idx])[0]
probabilities = model.predict_proba(input_df)[0]
prob_by_class = {str(c): float(p) for c, p in zip(encoders["Severity"].classes_, probabilities)}

# 7. KPI Strip
weather_icon = {"Rainy": "🌧️", "Foggy": "🌫️"}.get(weather.get("Weather_Condition"), "☀️")
upcoming = [s for s in route_hazards if s["exit_km"] >= covered_km]
k1, k2, k3, k4 = kpi_slot.columns(4)
with k1:
    html(ui.kpi("🛡️ Risk level", severity, f"{prob_by_class.get(severity, 0) * 100:.0f}% model confidence",
                accent=ui.severity_color(severity), text_value=True))
with k2:
    html(ui.kpi("🛣️ Remaining", f"{max(distance_km - covered_km, 0):.1f} km",
                f"{covered_km:.1f} of {distance_km:.1f} km covered"))
with k3:
    html(ui.kpi(f"{weather_icon} Weather", weather.get("Weather_Condition", "Clear"),
                f"{weather.get('Road_Condition', 'Dry')} road · {weather.get('Precipitation_mm', 0):.1f} mm · "
                f"{weather.get('Wind_Speed_kmh', 0):.0f} km/h wind",
                accent="#38BDF8", text_value=True))
with k4:
    html(ui.kpi("⚠️ Blackspots ahead", len(upcoming),
                f"{len(route_hazards)} on this route" if route_hazards else "Route clear of known clusters",
                accent="#F59E0B" if upcoming else "#22C55E"))
kpi_slot.markdown('<div style="height:18px"></div>', unsafe_allow_html=True)

# 8. Map & Co-Pilot HUD
with col_map:
    html(f'<div class="progress-meta"><span><b>{progress}%</b> complete</span>'
         f'<span>{covered_km:.1f} km / {distance_km:.1f} km</span></div>')

    m = folium.Map(location=current_loc, zoom_start=11, tiles=None, control_scale=True)
    m.get_root().header.add_child(folium.Element(ui.MAP_CSS))

    # Multi-Layer Selection (Night, Street, Satellite, Terrain)
    # CARTO's dark basemap now requires an API key, so use Esri's key-free dark canvas instead
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ",
        name="Night Navigation",
        max_zoom=16,
    ).add_to(m)
    folium.TileLayer("OpenStreetMap", name="Standard Street View").add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Satellite View"
    ).add_to(m)
    folium.TileLayer(
        tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        attr="OpenTopoMap",
        name="Topographic / Terrain View"
    ).add_to(m)

    # Render Accident Blackspots
    for spot in ACTIVE_BLACKSPOTS:
        color = ui.severity_color(spot.get("historical_severity"))
        folium.Circle(
            location=[spot["lat"], spot["lon"]],
            radius=spot.get("radius_meters", 1000),
            color=color,
            weight=1.5,
            dash_array="6 6",
            fill=True,
            fill_color=color,
            fill_opacity=0.2,
            tooltip=f"<b>{ui.esc(spot['name'])}</b><br>{spot.get('historical_severity', 'Risk')} zone · "
                    f"{ui.esc(spot.get('primary_hazard', ''))}"
        ).add_to(m)

    # Route: glow underlay, remaining path, travelled path
    folium.PolyLine(waypoints, color="#22D3EE", weight=12, opacity=0.15).add_to(m)
    folium.PolyLine(waypoints, color="#22D3EE", weight=4, opacity=0.55, tooltip="Planned route").add_to(m)
    if step_idx > 0:
        folium.PolyLine(waypoints[:step_idx + 1], color="#A5F3FC", weight=5, opacity=0.95,
                        tooltip="Travelled").add_to(m)

    folium.Marker(
        waypoints[0], tooltip=f"Start · {ui.esc(start_loc)}",
        icon=folium.DivIcon(html='<div class="pin pin-start"><span>A</span></div>',
                            icon_size=(30, 30), icon_anchor=(15, 30))
    ).add_to(m)
    folium.Marker(
        waypoints[-1], tooltip=f"Destination · {ui.esc(end_loc)}",
        icon=folium.DivIcon(html='<div class="pin pin-end"><span>B</span></div>',
                            icon_size=(30, 30), icon_anchor=(15, 30))
    ).add_to(m)
    folium.Marker(
        current_loc,
        tooltip=f"Vehicle · {vehicle_speed} km/h",
        icon=folium.DivIcon(html='<div class="vehicle"></div>', icon_size=(18, 18), icon_anchor=(9, 9))
    ).add_to(m)

    m.fit_bounds([[min(p[0] for p in waypoints), min(p[1] for p in waypoints)],
                  [max(p[0] for p in waypoints), max(p[1] for p in waypoints)]], padding=(30, 30))
    folium.LayerControl(position="topright").add_to(m)
    st_folium(m, height=560, use_container_width=True, returned_objects=[])

with col_hud:
    html(ui.panel_title("🛡️ Safety co-pilot", f"{clock} · {ui.pretty_vehicle(vehicle_type)}"))
    html(ui.risk_card(severity, prob_by_class))

    html(ui.panel_title("📢 Real-time advisories"))
    advisories = generate_advisory(severity, vehicle_speed, weather, hazard)
    html("".join(ui.advisory_card(a) for a in advisories))

# 9. Route Intelligence
html('<div style="height:10px"></div>')
col_tl, col_inputs = st.columns([1.55, 1], gap="large")

with col_tl:
    html(ui.panel_title("🚧 Hazards along this route", f"{len(route_hazards)} known blackspot(s)"))
    html(f'<div class="card">{ui.hazard_timeline(route_hazards, covered_km)}</div>')

with col_inputs:
    html(ui.panel_title("🧠 Model inputs", "What the classifier sees"))
    html(f"""<div class="card">{ui.inputs_grid([
        ("Time of day", f"{clock} · {day_phase.split(' ', 1)[-1]}"),
        ("Vehicle", ui.pretty_vehicle(vehicle_type)),
        ("Weather", weather_label),
        ("Road surface", road_label),
        ("Vehicles involved", num_vehicles),
        ("Position", f"{current_loc[0]:.4f}, {current_loc[1]:.4f}"),
    ])}<div class="footnote">Decision Tree classifier trained on historical road traffic accident records.
    Weather is sampled live at the vehicle's current position.</div></div>""")

# 10. Model Performance: confusion matrices & accuracy tracking
html('<div style="height:14px"></div>')
html(ui.panel_title("📊 Model performance", "Held-out test results · logged on every retrain"))

if not os.path.exists(METRICS_PATH):
    html(ui.note("No evaluation results yet. Run <b>python ml_eval.py</b> (or <b>python train.py</b>) to score the model "
                 "and generate <b>models/metrics.json</b>."))
else:
    perf = load_metrics(os.path.getmtime(METRICS_PATH))
    overall, labels, per_class = perf["overall"], perf["labels"], perf["per_class"]
    fatal = per_class.get("Fatal", {})

    p1, p2, p3, p4 = st.columns(4)
    with p1:
        html(ui.kpi("🎯 Accuracy", f"{overall['accuracy'] * 100:.1f}%",
                    f"majority-class baseline {overall['majority_baseline'] * 100:.1f}%"))
    with p2:
        html(ui.kpi("⚖️ Balanced accuracy", f"{overall['balanced_accuracy'] * 100:.1f}%",
                    "mean recall across classes", accent="#818CF8"))
    with p3:
        html(ui.kpi("🧪 Macro F1", f"{overall['macro_f1'] * 100:.1f}%",
                    "precision/recall, classes weighted equally", accent="#38BDF8"))
    with p4:
        html(ui.kpi("🚨 Fatal recall", f"{fatal.get('recall', 0) * 100:.1f}%",
                    f"precision {fatal.get('precision', 0) * 100:.1f}% · {fatal.get('support', 0)} fatal cases",
                    accent="#EF4444"))
    html('<div style="height:14px"></div>')

    col_cm, col_stats = st.columns([1.3, 1], gap="large")
    with col_cm:
        mode = st.radio("Matrix view", ui.CM_MODES, horizontal=True, label_visibility="collapsed")
        matrix_html = ui.confusion_matrix(labels, perf["confusion_matrix"], perf["confusion_matrix_recall"],
                                          perf["confusion_matrix_precision"], mode)
        html('<div class="card">' + matrix_html + '</div>')

    with col_stats:
        cv = perf["cross_validation"]
        cv_acc = cv["accuracy"]
        stats = (
            ui.stat_line("Train accuracy", "{:.1f}%".format(overall["train_accuracy"] * 100))
            + ui.stat_line("{}-fold CV accuracy".format(cv["folds"]),
                           "{:.1f}% ± {:.1f}".format(cv_acc["mean"] * 100, cv_acc["std"] * 100))
            + ui.stat_line("Cohen’s kappa", "{:.3f}".format(overall["cohen_kappa"]))
            + ui.stat_line("Test cases", "{:,} of {:,}".format(perf["dataset"]["test"], perf["dataset"]["rows"]))
        )
        html('<div class="card">' + ui.per_class_table(labels, per_class) + stats + '</div>')

    if overall["accuracy"] < overall["majority_baseline"]:
        top_class = max(perf["dataset"]["test_class_counts"], key=perf["dataset"]["test_class_counts"].get)
        html(ui.note(f"<b>Read accuracy with care.</b> {overall['majority_baseline'] * 100:.0f}% of test cases are "
                     f"<b>{ui.esc(top_class)}</b>, so always predicting it would score higher than this model. "
                     "The model is class-balanced to flag rarer, more severe outcomes, which trades overall accuracy "
                     "for recall: check the matrix and per-class recall/precision rather than accuracy alone."))

    html('<div style="height:10px"></div>')
    html(ui.panel_title("📈 Accuracy over time", "One row per retrain"))
    if os.path.exists(HISTORY_PATH):
        history = load_history(os.path.getmtime(HISTORY_PATH))
        if len(history) > 1:
            trend = history[["accuracy", "balanced_accuracy", "macro_f1"]].copy()
            trend.index = pd.RangeIndex(1, len(trend) + 1, name="Run #")
            st.line_chart(trend, height=220)
        else:
            html(ui.note("Only one evaluation is logged so far. Retrain with <b>python train.py</b> after changing the "
                         "model or data and the trend appears here."))
        with st.expander("Evaluation log"):
            st.dataframe(history.iloc[::-1], use_container_width=True, hide_index=True)
    model_info = perf["model"]
    html('<div class="footnote">Evaluated on {} · {} {} · scikit-learn {} · generated {}</div>'.format(
        ui.esc(perf["evaluated_on"]), ui.esc(model_info["type"]), ui.esc(json.dumps(model_info["params"])),
        ui.esc(model_info["sklearn_version"]), ui.esc(perf["generated_at"])))

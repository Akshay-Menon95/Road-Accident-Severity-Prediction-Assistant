# app.py
import streamlit as st
import folium
from streamlit_folium import st_folium
import joblib
import numpy as np
import pandas as pd
from navigation_engine import get_driving_route, get_live_weather, scan_hazards, generate_advisory, geocode_location, reverse_geocode
import ui_components as ui

try:
    from navigation_engine import NATIONAL_BLACKSPOTS as ACTIVE_BLACKSPOTS
except ImportError:
    ACTIVE_BLACKSPOTS = []

st.set_page_config(page_title="AI Road Safety Navigation", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")
st.markdown(ui.GLOBAL_CSS, unsafe_allow_html=True)
def html(markup): st.markdown(markup, unsafe_allow_html=True)

# ==========================================
# FIX: PROCESS MAP CLICKS BEFORE UI RENDERS
# ==========================================
if "start_input" not in st.session_state:
    st.session_state.start_input = ""
if "end_input" not in st.session_state:
    st.session_state.end_input = ""
if "last_click" not in st.session_state:
    st.session_state.last_click = None

# Scan session state for clicks from either the default map or active route map
for m_key in ["default_map", "active_map"]:
    if m_key in st.session_state and st.session_state[m_key]:
        clicked = st.session_state[m_key].get("last_clicked")
        if clicked:
            lat, lon = clicked["lat"], clicked["lng"]
            click_str = f"{lat:.5f},{lon:.5f}"
            
            # If it's a new click, reverse geocode and update the destination input
            if st.session_state.last_click != click_str:
                st.session_state.last_click = click_str
                st.session_state.end_input = reverse_geocode(lat, lon)


# ==========================================
# MODEL & API CACHING
# ==========================================
@st.cache_resource
def load_ml():
    return joblib.load("models/decision_tree_model.joblib"), joblib.load("models/label_encoders.joblib"), joblib.load("models/feature_columns.joblib")

try: 
    model, encoders, feature_columns = load_ml()
except Exception as e:
    html(ui.error_state("Model artifacts not found", f"Run 'python train.py' first. Error: {e}"))
    st.stop()

@st.cache_data(show_spinner=False)
def get_cached_route(start_name, end_name):
    start_coords = geocode_location(start_name)
    end_coords = geocode_location(end_name)
    if not start_coords or not end_coords: return None, "Could not locate places. Type name or click map."
    try: return get_driving_route(start_coords[0], start_coords[1], end_coords[0], end_coords[1]), "Success"
    except Exception as e: return None, f"Routing engine error: {e}"

@st.cache_data(show_spinner=False)
def get_cached_weather(lat, lon): return get_live_weather(lat, lon)

def safe_encode(encoder, value, default=0):
    try: return int(encoder.transform([value])[0]) if value in encoder.classes_ else default
    except Exception: return default

def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(a))


# ==========================================
# 1. UI SIDEBAR
# ==========================================
with st.sidebar:
    html(ui.side_brand())
    html(ui.side_label("Route (Search or Click Map)"))
    
    # State-managed location inputs (Tied to session_state)
    start_loc = st.text_input("Start location", key="start_input", placeholder="Enter your starting point")
    end_loc = st.text_input("Destination", key="end_input", placeholder="Enter destination or click the map")

    html(ui.side_label("Vehicle & Telemetry"))
    
    raw_classes = list(encoders["Vehicle_Type"].classes_)
    ui_mapping = {}
    for c in raw_classes:
        c_lower = c.lower()
        if "bicycle" in c_lower or "ridden horse" in c_lower: ui_mapping["Bicycle / Non-Motorized"] = c
        elif "motorcycle" in c_lower: ui_mapping["Motorcycle / Scooter"] = c
        elif "bajaj" in c_lower: ui_mapping["Auto-Rickshaw"] = c
        elif "automobile" in c_lower or "stationwagen" in c_lower or "taxi" in c_lower: ui_mapping["Car / Taxi"] = c
        elif "pick up" in c_lower: ui_mapping["Pick up / Light Truck"] = c
        elif "12 seats" in c_lower: ui_mapping["Minibus (12 seats)"] = c
        elif "13?45" in c_lower or "13-45" in c_lower: ui_mapping["Bus (13-45 seats)"] = c
        elif "> 45" in c_lower: ui_mapping["Bus (> 45 seats)"] = c
        elif "11?40" in c_lower or "turbo" in c_lower: ui_mapping["Truck (Short wheelbase)"] = c
        elif "41?100" in c_lower or "long" in c_lower: ui_mapping["Truck (Long wheelbase / Heavy)"] = c
        else: ui_mapping["Special / Other"] = c

    preferred_order = [
        "Bicycle / Non-Motorized", "Motorcycle / Scooter", "Auto-Rickshaw", 
        "Car / Taxi", "Pick up / Light Truck", "Minibus (12 seats)", 
        "Bus (13-45 seats)", "Bus (> 45 seats)", "Truck (Short wheelbase)", 
        "Truck (Long wheelbase / Heavy)", "Special / Other"
    ]
    
    ordered_ui_choices = [k for k in preferred_order if k in ui_mapping]
    for k in ui_mapping.keys():
        if k not in ordered_ui_choices: ordered_ui_choices.append(k)

    ui_choice = st.selectbox("Vehicle type", ordered_ui_choices)
    vehicle_type = ui_mapping[ui_choice]
    
    vehicle_speed = st.slider("Current speed (km/h)", min_value=10, max_value=160, value=65)
    time_hour = st.slider("Time of day (hour)", 0, 23, 14)
    num_vehicles = st.slider("Vehicles involved in potential crash", 1, 6, 2)


# ==========================================
# 2. DEFAULT STARTUP VIEW (No Inputs)
# ==========================================
if not start_loc or not end_loc:
    col_map, col_hud = st.columns([1.55, 1], gap="large")
    with col_map:
        html(ui.panel_title("🛰️ Live navigation", "Click anywhere to set destination"))
        
        m_def = folium.Map(location=[10.7867, 76.6548], zoom_start=8, tiles=None) # Centered near Kerala by default
        folium.TileLayer(
            tiles="https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}",
            attr="Google",
            name="Google Street View",
            control=True
        ).add_to(m_def)
        folium.LayerControl(position="topright").add_to(m_def)
        
        # Render the map WITH A KEY so the click processor at the top can see it
        st_folium(m_def, height=560, use_container_width=True, key="default_map")
        
    with col_hud:
        st.title("🗺️ AI Road Safety Navigation")
        st.info("👈 Enter a Start Location, then **click anywhere on the map** to instantly plot your destination.")
    st.stop()


# ==========================================
# 3. ROUTE COMPUTATION
# ==========================================
with st.spinner("Calculating dynamic hazards and route geometry..."):
    route_data, status_msg = get_cached_route(start_loc, end_loc)

if not route_data:
    html(ui.error_state("Route Error", status_msg))
    st.stop()

waypoints, distance_km = route_data["waypoints"], route_data["distance_km"]

pts = np.array(waypoints)
seg = haversine_km(pts[:-1, 0], pts[:-1, 1], pts[1:, 0], pts[1:, 1])
cum_km = np.concatenate([[0.0], np.cumsum(seg)])
if cum_km[-1] > 0: cum_km = cum_km * (distance_km / cum_km[-1]) 


# ==========================================
# 4. MAIN INTERFACE & PREDICTION
# ==========================================
col_map, col_hud = st.columns([1.55, 1], gap="large")

with col_map:
    html(ui.panel_title("🛰️ Live navigation", "Drag slider or click map to change destination"))
    covered_km = st.slider("Trip Progress", 0.0, float(distance_km), 0.0, format="%.1f km", label_visibility="collapsed")

step_idx = int(np.argmin(np.abs(cum_km - covered_km)))
current_loc = waypoints[step_idx]

weather = get_cached_weather(round(current_loc[0], 2), round(current_loc[1], 2))
hazard = scan_hazards(current_loc[0], current_loc[1], waypoints)

candidate_row = {
    "Vehicle_Speed": vehicle_speed,
    "Time_Hour": time_hour,
    "Number_of_Vehicles": num_vehicles,
    "Weather_Condition": safe_encode(encoders.get("Weather_Condition"), weather.get("Weather_Condition")),
    "Road_Condition": safe_encode(encoders.get("Road_Condition"), weather.get("Road_Condition")),
    "Vehicle_Type": safe_encode(encoders.get("Vehicle_Type"), vehicle_type)
}

input_df = pd.DataFrame([{col: candidate_row.get(col, 0) for col in feature_columns}])[feature_columns]
severity = encoders["Severity"].inverse_transform([model.predict(input_df)[0]])[0]
probabilities = model.predict_proba(input_df)[0]
prob_by_class = {str(c): float(p) for c, p in zip(encoders["Severity"].classes_, probabilities)}

with col_map:
    html(f'<div class="progress-meta"><span><b>{covered_km:.1f} km</b> covered</span><span><b>{distance_km - covered_km:.1f} km</b> remaining</span></div>')

    m = folium.Map(location=current_loc, zoom_start=14, tiles=None)
    
    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
        attr="Google",
        name="Google Satellite",
        control=True
    ).add_to(m)
    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}",
        attr="Google",
        name="Google Street View",
        control=True
    ).add_to(m)

    folium.PolyLine(waypoints, color="#22D3EE", weight=6).add_to(m)
    
    folium.Marker(waypoints[0], tooltip="Start Location", icon=folium.Icon(color="green", icon="play", prefix="fa")).add_to(m)
    folium.Marker(waypoints[-1], tooltip="Destination", icon=folium.Icon(color="purple", icon="flag", prefix="fa")).add_to(m)
    folium.Marker(current_loc, icon=folium.Icon(color="red", icon="car", prefix="fa")).add_to(m)
    
    if hazard:
        folium.Circle(
            location=[hazard["lat"], hazard["lon"]], radius=hazard.get("radius_meters", 400),
            color="red", fill=True, fill_opacity=0.4, tooltip=hazard["name"]
        ).add_to(m)

    if covered_km == 0:
        m.fit_bounds([waypoints[0], waypoints[-1]])

    folium.LayerControl(position="topright").add_to(m)
    
    # Render Active Map WITH KEY
    st_folium(m, height=560, use_container_width=True, key="active_map")

with col_hud:
    html(ui.panel_title("🛡️ Safety co-pilot", f"{vehicle_speed} km/h"))
    html(ui.risk_card(severity, prob_by_class))
    
    html(ui.panel_title("📢 Advisories"))
    for adv in generate_advisory(severity, vehicle_speed, weather, hazard):
        st.info(adv)
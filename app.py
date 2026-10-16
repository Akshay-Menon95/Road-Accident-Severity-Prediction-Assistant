# app.py
import streamlit as st
import folium
from streamlit_folium import st_folium
import joblib
import pandas as pd
from navigation_engine import (
    get_driving_route, 
    get_live_weather, 
    scan_hazards, 
    generate_advisory, 
    geocode_location
)

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
    page_icon="🗺️", 
    layout="wide"
)

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
    st.error(f"Failed to load model artifacts. Ensure 'train.py' has been executed. Error: {e}")
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

# 3. Main Header & Controls
st.title("🗺️ AI Road Safety Navigation & Live Risk Co-Pilot")
st.markdown("Dynamic route safety analysis powered by **Real Historical Accident Telemetry**, **Live Meteorological APIs**, and **Decision Tree Classification**.")

# 4. Sidebar: Trip Configuration & Driver Profile
st.sidebar.header("📍 Route Configuration")
start_loc = st.sidebar.text_input("Start Location (Anywhere in India)", value="Palakkad")
end_loc = st.sidebar.text_input("Destination", value="Thrissur")

st.sidebar.divider()
st.sidebar.subheader("🚗 Vehicle & Telemetry Inputs")

vehicle_classes = (
    list(encoders["Vehicle_Type"].classes_) 
    if "Vehicle_Type" in encoders and hasattr(encoders["Vehicle_Type"], "classes_") 
    else ["Car", "Two-Wheeler", "Truck", "Bus", "Auto-Rickshaw"]
)
vehicle_type = st.sidebar.selectbox("Vehicle Type", vehicle_classes)
vehicle_speed = st.sidebar.slider("Current Vehicle Speed (km/h)", min_value=10, max_value=150, value=65, step=1)
time_hour = st.sidebar.slider("Time of Day (Hour: 0 = 12 AM, 23 = 11 PM)", min_value=0, max_value=23, value=15)
num_vehicles = st.sidebar.slider("Vehicles Involved (Collision Scenario)", min_value=1, max_value=6, value=2)

# 5. Route Computation
with st.spinner("Calculating route geometry, elevation, and hazard intersections..."):
    route_data, status_msg = get_cached_route(start_loc, end_loc)

if not route_data:
    st.error(f"Navigation Error: {status_msg}")
    st.stop()

waypoints = route_data["waypoints"]
step_idx = st.slider("🚗 Simulate Vehicle Movement along Route", 0, len(waypoints) - 1, 0)
current_loc = waypoints[step_idx]

# 6. Real-Time Telemetry Retrieval
weather = get_cached_weather(round(current_loc[0], 2), round(current_loc[1], 2))
hazard = scan_hazards(current_loc[0], current_loc[1])

# 7. Adaptive Model Feature Alignment
# Supports both Kerala models and raw RTA/Kaggle models without schema collisions
candidate_row = {
    "Vehicle_Speed": vehicle_speed,
    "Time_Hour": time_hour,
    "Number_of_Vehicles": num_vehicles,
    "District": safe_encode(encoders.get("District"), "Palakkad"),
    "State": safe_encode(encoders.get("State"), "Kerala"),
    "Road_Type": safe_encode(encoders.get("Road_Type"), "National Highway"),
    "Traffic_Condition": safe_encode(encoders.get("Traffic_Condition"), "Medium"),
    "Weather_Condition": safe_encode(encoders.get("Weather_Condition"), weather.get("Weather_Condition", "Clear")),
    "Road_Condition": safe_encode(encoders.get("Road_Condition"), weather.get("Road_Condition", "Dry")),
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

# 8. User Interface Layout
col_map, col_hud = st.columns([1.3, 0.9], gap="medium")

with col_map:
    st.subheader("Interactive Navigation View")
    
    # Base Folium Map centered on current vehicle GPS
    m = folium.Map(location=current_loc, zoom_start=12, tiles=None)
    
    # Multi-Layer Selection (Street, Satellite, Terrain)
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
    
    # Route Line & Dynamic Vehicle Marker
    folium.PolyLine(waypoints, color="#2A81CB", weight=5, opacity=0.8, tooltip="Planned Route").add_to(m)
    folium.Marker(
        current_loc, 
        tooltip=f"Current Vehicle Position ({vehicle_speed} km/h)", 
        icon=folium.Icon(color="green", icon="car", prefix="fa")
    ).add_to(m)
    
    # Render Accident Blackspots
    for spot in ACTIVE_BLACKSPOTS:
        is_fatal = spot.get("historical_severity") == "Fatal"
        folium.Circle(
            location=[spot["lat"], spot["lon"]], 
            radius=spot.get("radius_meters", 1000),
            color="#D9534F" if is_fatal else "#F0AD4E", 
            fill=True,
            fill_opacity=0.35,
            tooltip=f"Blackspot: {spot['name']} ({spot.get('historical_severity', 'Risk')} Zone)"
        ).add_to(m)
        
    folium.LayerControl(position="topright").add_to(m)
    st_folium(m, width="100%", height=520, returned_objects=[])

with col_hud:
    st.subheader("🛡️ Safety Co-Pilot HUD")
    
    # Visual Severity Banners
    if severity == "Fatal":
        st.error("### 🚨 RISK LEVEL: FATAL")
        st.markdown("**Critical Hazard:** High probability of fatal trauma. Emergency mitigation required.")
    elif severity == "Serious":
        st.warning("### ⚠️ RISK LEVEL: SERIOUS")
        st.markdown("**Elevated Risk:** Probable hospitalization or severe structural compromise.")
    else:
        st.success("### 🟢 RISK LEVEL: MINOR")
        st.markdown("**Low Risk:** Expected damage limited to property or minor abrasions.")

    # Live Environmental Telemetry
    m1, m2 = st.columns(2)
    m1.metric("Trip Distance", f"{route_data['distance_km']} km")
    m2.metric("Target Weather", weather.get("Weather_Condition", "Clear"), f"{weather.get('Road_Condition', 'Dry')} Road")

    st.markdown("#### 📢 Real-Time Co-Pilot Advisories:")
    advisories = generate_advisory(severity, vehicle_speed, weather, hazard)
    for adv in advisories:
        st.info(adv)
        
    if hazard:
        st.error(f"📍 **Approaching Cluster:** {hazard['name']} (Distance: {hazard.get('distance_m', 0)}m). Risk: {hazard.get('primary_hazard', 'Historical Accident Spot')}")

    st.divider()
    st.markdown("##### Prediction Confidence Breakdown:")
    prob_df = pd.DataFrame({
        "Severity Class": encoders["Severity"].classes_,
        "Confidence": [f"{p * 100:.1f}%" for p in probabilities]
    })
    st.dataframe(prob_df, use_container_width=True, hide_index=True)
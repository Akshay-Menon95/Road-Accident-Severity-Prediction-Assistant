# app.py
import streamlit as st
import folium
from streamlit_folium import st_folium
import numpy as np
from navigation_engine import (get_driving_route, get_live_weather, scan_hazards, scan_route_hazards,
                               route_cumulative_km, generate_advisory, geocode_location, reverse_geocode)
from label_maps import VEHICLE_GROUPS, light_for_hour
from risk_model import load_artifacts, predict_severity
import ui_components as ui

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
    return load_artifacts()

try: 
    model, encoders, feature_columns = load_ml()
except Exception as e:
    html(ui.error_state("Model artifacts not found", f"Run 'python train.py' first. Error: {e}"))
    st.stop()

class RouteLookupError(Exception): pass

@st.cache_data(show_spinner=False)
def _compute_route(start_name, end_name):
    # Raising (rather than returning an error tuple) keeps st.cache_data from caching a
    # transient geocoder/router outage for the rest of the session.
    start_coords = geocode_location(start_name)
    end_coords = geocode_location(end_name)
    if not start_coords or not end_coords:
        raise RouteLookupError("Could not locate places. Type name or click map.")
    return get_driving_route(start_coords[0], start_coords[1], end_coords[0], end_coords[1])

def get_cached_route(start_name, end_name):
    try: return _compute_route(start_name, end_name), "Success"
    except RouteLookupError as e: return None, str(e)
    except Exception as e: return None, f"Routing engine error: {e}"

@st.cache_data(show_spinner=False, ttl=600)
def get_cached_weather(lat, lon): return get_live_weather(lat, lon)


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
    
    known_vehicles = set(encoders["Vehicle_Type"].classes_)
    vehicle_choices = [v for v in VEHICLE_GROUPS if v in known_vehicles]
    vehicle_choices += sorted(known_vehicles - set(vehicle_choices))
    vehicle_type = st.selectbox("Vehicle type", vehicle_choices, index=vehicle_choices.index("Car / Taxi") if "Car / Taxi" in vehicle_choices else 0)

    vehicle_speed = st.slider("Current speed (km/h)", min_value=10, max_value=160, value=65,
                              help="Drives the driving advisories only. The crash dataset has no speed field, so speed does not feed the severity model.")
    time_hour = st.slider("Time of day (hour)", 0, 23, 14)
    num_vehicles = st.slider("Vehicles involved in potential crash", 1, 6, 2)


# ==========================================
# 2. DEFAULT STARTUP VIEW (No Inputs)
# ==========================================
if not start_loc or not end_loc:
    col_map, col_hud = st.columns([1.55, 1], gap="large")
    with col_map:
        html(ui.panel_title("🛰️ Live navigation", "Click anywhere to set destination"))
        
        m_def = folium.Map(location=[10.7867, 76.6548], zoom_start=8, tiles="OpenStreetMap") # Centered near Kerala by default
        
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

if distance_km <= 0 or len(waypoints) < 2:
    html(ui.error_state("Route Error", "Start and destination resolve to the same place. Pick a different destination."))
    st.stop()

cum_km = route_cumulative_km(waypoints, distance_km)
route_hazards = scan_route_hazards(waypoints, cum_km)


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
hazard = scan_hazards(current_loc[0], current_loc[1], waypoints, cum_km)

try:
    severity, prob_by_class = predict_severity(model, encoders, feature_columns, {
        "Time_Hour": time_hour,
        "Number_of_Vehicles": num_vehicles,
        "Weather_Condition": weather["Weather_Condition"],
        "Road_Condition": weather["Road_Condition"],
        "Vehicle_Type": vehicle_type,
        "Light_Condition": light_for_hour(time_hour),
    })
except (KeyError, ValueError) as e:
    html(ui.error_state("Model input mismatch", str(e)))
    st.stop()

with col_map:
    html(f'<div class="progress-meta"><span><b>{covered_km:.1f} km</b> covered</span><span><b>{distance_km - covered_km:.1f} km</b> remaining</span></div>')

    m = folium.Map(location=current_loc, zoom_start=14, tiles=None)
    
    folium.TileLayer(tiles="OpenStreetMap", name="Street map", control=True).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri, Maxar, Earthstar Geographics",
        name="Satellite",
        control=True
    ).add_to(m)

    folium.PolyLine(waypoints, color="#22D3EE", weight=6).add_to(m)
    
    folium.Marker(waypoints[0], tooltip="Start Location", icon=folium.Icon(color="green", icon="play", prefix="fa")).add_to(m)
    folium.Marker(waypoints[-1], tooltip="Destination", icon=folium.Icon(color="purple", icon="flag", prefix="fa")).add_to(m)
    folium.Marker(current_loc, icon=folium.Icon(color="red", icon="car", prefix="fa")).add_to(m)
    
    for spot in route_hazards:
        active = hazard is not None and hazard["name"] == spot["name"]
        folium.Circle(
            location=[spot["lat"], spot["lon"]], radius=spot.get("radius_meters", 400),
            color="red" if active else "orange", fill=True, fill_opacity=0.4 if active else 0.2,
            tooltip=spot["name"]
        ).add_to(m)

    if covered_km == 0:
        m.fit_bounds([waypoints[0], waypoints[-1]])

    folium.LayerControl(position="topright").add_to(m)
    
    # Render Active Map WITH KEY
    st_folium(m, height=560, use_container_width=True, key="active_map")

with col_hud:
    html(ui.panel_title("🛡️ Safety co-pilot", f"{vehicle_speed} km/h"))
    html(ui.risk_card(severity, prob_by_class))
    st.caption("Trained on ~11k Ethiopian crash records (RTA dataset). Scores are relative risk, not calibrated probabilities.")
    
    html(ui.panel_title("📢 Advisories"))
    if not weather.get("live", True):
        st.warning("Live weather unavailable - assuming normal weather and a dry road.")
    for adv in generate_advisory(severity, vehicle_speed, weather, hazard):
        html(ui.advisory_card(adv))

    html(ui.panel_title("🚧 Route blackspots", f"{len(route_hazards)} on route"))
    html(ui.hazard_timeline(route_hazards, covered_km))
# navigation_engine.py
import requests
import polyline
import re
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from blackspots import NATIONAL_BLACKSPOTS

def geocode_location(place_name):
    """Converts text place names or raw coordinate strings into GPS coordinates."""
    place_str = str(place_name).strip()
    
    # 1. SMART EXTRACTION: Grabs coordinates if format is "Name (lat, lon)" from map clicks
    paren_match = re.search(r"\(\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*\)", place_str)
    if paren_match:
        return (float(paren_match.group(1)), float(paren_match.group(2)))
        
    # 2. Extract coordinates if format is purely "lat, lon"
    coord_match = re.match(r"^\s*(-?\d+(\.\d+)?)\s*,\s*(-?\d+(\.\d+)?)\s*$", place_str)
    if coord_match:
        return (float(coord_match.group(1)), float(coord_match.group(3)))

    # 3. Standard text geocoding
    geolocator = Nominatim(user_agent="india_safety_navigator")
    try:
        location = geolocator.geocode(f"{place_str}, India")
        if location:
            return (location.latitude, location.longitude)
    except Exception as e:
        print(f"Geocoding error: {e}")
    return None

def reverse_geocode(lat, lon):
    """Converts map click coordinates into a readable place name."""
    geolocator = Nominatim(user_agent="india_safety_navigator")
    try:
        loc = geolocator.reverse((lat, lon), exactly_one=True, timeout=5)
        if loc:
            addr = loc.raw.get('address', {})
            name = addr.get('city', addr.get('town', addr.get('village', addr.get('suburb', addr.get('county', '')))))
            if name:
                return f"{name} ({lat:.5f}, {lon:.5f})"
            return loc.address.split(',')[0]
    except Exception:
        pass
    return f"{lat:.5f}, {lon:.5f}"

def get_live_weather(lat, lon):
    """Fetches real-time weather from Open-Meteo."""
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=precipitation,weather_code,wind_speed_10m"
    try:
        res = requests.get(url, timeout=5).json().get("current", {})
        precip = float(res.get("precipitation", 0.0))
        code = int(res.get("weather_code", 0))
        wind = float(res.get("wind_speed_10m", 0.0))
        
        if code in [45, 48]:
            w_cond, r_cond = "Foggy", "Wet"
        elif precip > 0.0 or code in [51, 53, 55, 61, 63, 65, 80, 81, 82]:
            w_cond, r_cond = "Rainy", "Wet"
        else:
            w_cond, r_cond = "Clear", "Dry"
            
        return {"Weather_Condition": w_cond, "Road_Condition": r_cond, "Precipitation_mm": precip, "Wind_Speed_kmh": wind}
    except:
        return {"Weather_Condition": "Clear", "Road_Condition": "Dry", "Precipitation_mm": 0.0, "Wind_Speed_kmh": 10.0}

def get_driving_route(start_lat, start_lon, end_lat, end_lon):
    """Calculates route using the free OSRM API."""
    url = f"https://router.project-osrm.org/route/v1/driving/{start_lon},{start_lat};{end_lon},{end_lat}?overview=full&geometries=polyline"
    res = requests.get(url, timeout=10).json()
    route = res["routes"][0]
    return {
        "distance_km": round(route["distance"] / 1000.0, 2),
        "waypoints": polyline.decode(route["geometry"])
    }

def scan_hazards(current_lat, current_lon):
    """Checks if the vehicle is inside a known blackspot radius."""
    for spot in NATIONAL_BLACKSPOTS:
        dist_m = geodesic((current_lat, current_lon), (spot["lat"], spot["lon"])).meters
        if dist_m <= spot["radius_meters"]:
            return {**spot, "distance_m": round(dist_m, 0)}
    return None

def generate_advisory(severity, speed, weather, hazard):
    """Creates ADAS-style text alerts based on the predictions."""
    advisories = []
    if speed > 70:
        advisories.append("⚠️ **Speed Alert:** High speed detected. Reduce speed below 60 km/h to maintain safe braking distance.")
    if weather["Road_Condition"] == "Wet":
        advisories.append("🌧️ **Slippery Surface:** Wet roads detected. Risk of hydroplaning. Increase following distance.")
    if hazard:
        advisories.append(f"🚨 **Hazard Zone:** You are in '{hazard['name']}'. Risk: {hazard['primary_hazard']}.")
        if severity == "Fatal":
            advisories.append("🛑 **CRITICAL RISK:** High fatality probability ahead. Consider immediate speed reduction or route diversion.")
    if not advisories:
        advisories.append("✅ **Clear:** Standard driving conditions. Maintain lane discipline.")
    return advisories
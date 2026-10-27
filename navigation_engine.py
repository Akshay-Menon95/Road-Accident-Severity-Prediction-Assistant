# navigation_engine.py
import re

import numpy as np
import polyline
import requests
from geopy.exc import GeopyError
from geopy.geocoders import Nominatim

from blackspots import NATIONAL_BLACKSPOTS
from label_maps import (ADVERSE_ROAD, ADVERSE_WEATHER, ROAD_DRY, ROAD_WET, WEATHER_FOG,
                        WEATHER_NORMAL, classify_weather)

SPEED_ALERT_KMH = 70
SPEED_ALERT_ADVERSE_KMH = 50
LOOKAHEAD_KM = 5.0
EARTH_RADIUS_KM = 6371.0088

_COORD_PAIR = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*$")
_COORD_IN_PARENS = re.compile(r"\(\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*\)")


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km; accepts scalars or numpy arrays."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def _valid_coords(lat, lon):
    return -90 <= lat <= 90 and -180 <= lon <= 180


def geocode_location(place_name):
    """Converts text place names or raw coordinate strings into GPS coordinates.

    Returns None when nothing matches. Geocoding service failures propagate as
    GeopyError so callers do not mistake an outage for an unknown place.
    """
    place_str = str(place_name).strip()

    # "Name (lat, lon)" as written by reverse_geocode for map clicks, or plain "lat, lon".
    match = _COORD_IN_PARENS.search(place_str) or _COORD_PAIR.match(place_str)
    if match:
        lat, lon = float(match.group(1)), float(match.group(2))
        if _valid_coords(lat, lon):
            return (lat, lon)

    location = Nominatim(user_agent="india_safety_navigator").geocode(f"{place_str}, India", timeout=10)
    if location:
        return (location.latitude, location.longitude)
    return None


def reverse_geocode(lat, lon):
    """Converts map click coordinates into a readable name that still carries the coordinates."""
    coords = f"{lat:.5f}, {lon:.5f}"
    try:
        loc = Nominatim(user_agent="india_safety_navigator").reverse((lat, lon), exactly_one=True, timeout=5)
    except GeopyError:
        return coords
    if loc:
        addr = loc.raw.get('address', {})
        name = (addr.get('city') or addr.get('town') or addr.get('village')
                or addr.get('suburb') or addr.get('county') or loc.address.split(',')[0])
        if name:
            return f"{name} ({coords})"
    return coords


def get_live_weather(lat, lon):
    """Fetches real-time weather from Open-Meteo, expressed in the model's training labels.

    `live` is False when the API could not be read and the conservative-looking default
    (normal weather, dry road) was substituted, so the UI can say so.
    """
    try:
        res = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": lat, "longitude": lon, "current": "precipitation,weather_code,wind_speed_10m"},
            timeout=5,
        )
        res.raise_for_status()
        current = res.json()["current"]
        precip = float(current.get("precipitation") or 0.0)
        code = int(current.get("weather_code") or 0)
        wind = float(current.get("wind_speed_10m") or 0.0)
    except (requests.RequestException, KeyError, ValueError, TypeError):
        return {"Weather_Condition": WEATHER_NORMAL, "Road_Condition": ROAD_DRY,
                "Precipitation_mm": 0.0, "Wind_Speed_kmh": 0.0, "live": False}

    weather, road = classify_weather(code, precip, wind)
    return {"Weather_Condition": weather, "Road_Condition": road,
            "Precipitation_mm": precip, "Wind_Speed_kmh": wind, "live": True}


def get_driving_route(start_lat, start_lon, end_lat, end_lon):
    """Calculates route using the free OSRM API."""
    url = f"https://router.project-osrm.org/route/v1/driving/{start_lon},{start_lat};{end_lon},{end_lat}"
    res = requests.get(url, params={"overview": "full", "geometries": "polyline"}, timeout=10)
    res.raise_for_status()
    body = res.json()
    if body.get("code") != "Ok" or not body.get("routes"):
        raise RuntimeError(body.get("message") or f"OSRM returned '{body.get('code')}' - no drivable route found")
    route = body["routes"][0]
    return {
        "distance_km": round(route["distance"] / 1000.0, 2),
        "waypoints": polyline.decode(route["geometry"]),
    }


def route_cumulative_km(waypoints, distance_km=None):
    """Distance along the route at each waypoint, rescaled to the router's reported length."""
    pts = np.asarray(waypoints, dtype=float)
    seg = haversine_km(pts[:-1, 0], pts[:-1, 1], pts[1:, 0], pts[1:, 1])
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if distance_km and cum[-1] > 0:
        cum = cum * (distance_km / cum[-1])
    return cum


def scan_hazards(current_lat, current_lon, waypoints=None, cum_km=None, lookahead_km=LOOKAHEAD_KM):
    """Finds the most relevant blackspot for the vehicle's position.

    Returns a blackspot dict plus `status` ("inside" or "ahead"), `distance_m` to its centre
    and, when ahead, `ahead_km` along the route until it is entered. Spots the vehicle is
    already inside win; otherwise the nearest one within `lookahead_km` of upcoming route.
    Returns None when the road ahead is clear.
    """
    inside = []
    for spot in NATIONAL_BLACKSPOTS:
        dist_m = float(haversine_km(current_lat, current_lon, spot["lat"], spot["lon"])) * 1000.0
        if dist_m <= spot["radius_meters"]:
            inside.append({**spot, "distance_m": round(dist_m), "status": "inside"})
    if inside:
        return min(inside, key=lambda s: s["distance_m"])

    if waypoints is None or len(waypoints) < 2:
        return None
    pts = np.asarray(waypoints, dtype=float)
    cum = route_cumulative_km(pts) if cum_km is None else np.asarray(cum_km, dtype=float)
    here = int(np.argmin(haversine_km(current_lat, current_lon, pts[:, 0], pts[:, 1])))
    ahead = np.flatnonzero((np.arange(len(pts)) >= here) & (cum - cum[here] <= lookahead_km))

    best = None
    for spot in NATIONAL_BLACKSPOTS:
        d_m = haversine_km(pts[ahead, 0], pts[ahead, 1], spot["lat"], spot["lon"]) * 1000.0
        hits = ahead[d_m <= spot["radius_meters"]]
        if len(hits) == 0:
            continue
        ahead_km = float(cum[hits[0]] - cum[here])
        if best is None or ahead_km < best["ahead_km"]:
            centre_m = float(haversine_km(current_lat, current_lon, spot["lat"], spot["lon"])) * 1000.0
            best = {**spot, "distance_m": round(centre_m), "status": "ahead", "ahead_km": round(ahead_km, 1)}
    return best


def scan_route_hazards(waypoints, cum_km=None):
    """Lists every blackspot the whole route passes through, ordered along the route.

    Each entry carries the keys ui_components.hazard_timeline expects:
    `km` (closest approach), `enter_km` and `exit_km`.
    """
    pts = np.asarray(waypoints, dtype=float)
    cum = route_cumulative_km(pts) if cum_km is None else np.asarray(cum_km, dtype=float)
    found = []
    for spot in NATIONAL_BLACKSPOTS:
        d_m = haversine_km(pts[:, 0], pts[:, 1], spot["lat"], spot["lon"]) * 1000.0
        hits = np.flatnonzero(d_m <= spot["radius_meters"])
        if len(hits) == 0:
            continue
        found.append({**spot, "km": float(cum[int(np.argmin(d_m))]),
                      "enter_km": float(cum[hits[0]]), "exit_km": float(cum[hits[-1]])})
    return sorted(found, key=lambda s: s["enter_km"])


def generate_advisory(severity, speed, weather, hazard):
    """Creates ADAS-style text alerts based on the predictions."""
    advisories = []
    adverse = (weather["Road_Condition"] in ADVERSE_ROAD or weather["Weather_Condition"] in ADVERSE_WEATHER)
    limit = SPEED_ALERT_ADVERSE_KMH if adverse else SPEED_ALERT_KMH

    if speed > limit:
        reason = "in these conditions" if adverse else "for a safe braking distance"
        advisories.append(f"⚠️ **Speed Alert:** {speed} km/h is too fast {reason}. Reduce speed to {limit} km/h or below.")
    if weather["Road_Condition"] == ROAD_WET:
        advisories.append("🌧️ **Slippery Surface:** Wet roads detected. Risk of hydroplaning. Increase following distance.")
    if weather["Weather_Condition"] == WEATHER_FOG:
        advisories.append("🌫️ **Low Visibility:** Fog or mist detected. Use low-beam headlights and slow down.")

    if hazard:
        if hazard.get("status") == "ahead":
            advisories.append(f"🚧 **Hazard Ahead:** '{hazard['name']}' in {hazard['ahead_km']:.1f} km. Risk: {hazard['primary_hazard']}.")
        else:
            advisories.append(f"🚨 **Hazard Zone:** You are in '{hazard['name']}'. Risk: {hazard['primary_hazard']}.")
            if severity == "Fatal" or (hazard.get("historical_severity") == "Fatal" and speed > limit):
                advisories.append("🛑 **CRITICAL RISK:** High fatality risk here. Reduce speed immediately or consider a diversion.")
    if not advisories:
        advisories.append("✅ **Clear:** Standard driving conditions. Maintain lane discipline.")
    return advisories

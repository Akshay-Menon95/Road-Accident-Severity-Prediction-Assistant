# label_maps.py
"""Single source of truth for the labels the severity model is trained on.

Training (train.py), live inputs (navigation_engine.py) and the dashboard (app.py) all
translate their raw values through this module, so a label can never silently fall
outside the encoder vocabulary.
"""
import pandas as pd

# --- Vehicles -------------------------------------------------------------------------
BICYCLE = "Bicycle / Non-Motorized"
MOTORCYCLE = "Motorcycle / Scooter"
AUTO_RICKSHAW = "Auto-Rickshaw"
CAR = "Car / Taxi"
PICKUP = "Pick up / Light Truck"
MINIBUS = "Minibus (12 seats)"
BUS_MID = "Bus (13-45 seats)"
BUS_LARGE = "Bus (> 45 seats)"
TRUCK_SHORT = "Truck (Short wheelbase)"
TRUCK_LONG = "Truck (Long wheelbase / Heavy)"
OTHER_VEHICLE = "Special / Other"

# Display order for the dashboard dropdown.
VEHICLE_GROUPS = [BICYCLE, MOTORCYCLE, AUTO_RICKSHAW, CAR, PICKUP, MINIBUS,
                  BUS_MID, BUS_LARGE, TRUCK_SHORT, TRUCK_LONG, OTHER_VEHICLE]

# Raw `Type_of_vehicle` values in data/RTA_Dataset.csv (lower-cased; the dataset's
# en-dashes were corrupted to "?", which _normalise_vehicle turns into "-").
_RAW_VEHICLE_TO_GROUP = {
    "bicycle": BICYCLE,
    "ridden horse": BICYCLE,
    "motorcycle": MOTORCYCLE,
    "bajaj": AUTO_RICKSHAW,
    "automobile": CAR,
    "stationwagen": CAR,
    "taxi": CAR,
    "pick up upto 10q": PICKUP,
    "public (12 seats)": MINIBUS,
    "public (13-45 seats)": BUS_MID,
    "public (> 45 seats)": BUS_LARGE,
    "lorry (11-40q)": TRUCK_SHORT,
    "turbo": TRUCK_SHORT,
    "lorry (41-100q)": TRUCK_LONG,
    "long lorry": TRUCK_LONG,
    "special vehicle": OTHER_VEHICLE,
    "other": OTHER_VEHICLE,
}


def _normalise_vehicle(raw):
    return str(raw).strip().lower().replace("?", "-").replace("–", "-")


def map_vehicle(raw):
    """Maps a raw dataset vehicle name to a model class.

    Missing values return None (the caller drops them rather than guessing). A value
    that is present but unknown raises, so dataset drift is loud instead of being
    quietly filed under some default class.
    """
    if pd.isna(raw):
        return None
    try:
        return _RAW_VEHICLE_TO_GROUP[_normalise_vehicle(raw)]
    except KeyError:
        raise ValueError(f"Unmapped vehicle type in dataset: {raw!r}") from None


# --- Weather & road surface -----------------------------------------------------------
# Labels as they appear in the dataset's Weather_conditions / Road_surface_conditions.
WEATHER_NORMAL = "Normal"
WEATHER_CLOUDY = "Cloudy"
WEATHER_RAIN = "Raining"
WEATHER_RAIN_WIND = "Raining and Windy"
WEATHER_WINDY = "Windy"
WEATHER_FOG = "Fog or mist"
WEATHER_SNOW = "Snow"

ROAD_DRY = "Dry"
ROAD_WET = "Wet or damp"
ROAD_SNOW = "Snow"

WINDY_KMH = 30.0
_FOG_CODES = {45, 48}
_SNOW_CODES = {71, 73, 75, 77, 85, 86}
_RAIN_CODES = {51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82, 95, 96, 99}
_CLOUDY_CODES = {2, 3}


def classify_weather(weather_code, precipitation_mm, wind_kmh):
    """Translates Open-Meteo readings into (weather, road surface) training labels."""
    windy = wind_kmh >= WINDY_KMH
    if weather_code in _FOG_CODES:
        return WEATHER_FOG, ROAD_WET
    if weather_code in _SNOW_CODES:
        return WEATHER_SNOW, ROAD_SNOW
    if precipitation_mm > 0.0 or weather_code in _RAIN_CODES:
        return (WEATHER_RAIN_WIND if windy else WEATHER_RAIN), ROAD_WET
    if windy:
        return WEATHER_WINDY, ROAD_DRY
    if weather_code in _CLOUDY_CODES:
        return WEATHER_CLOUDY, ROAD_DRY
    return WEATHER_NORMAL, ROAD_DRY


# --- Lighting --------------------------------------------------------------------------
LIGHT_DAY = "Daylight"
LIGHT_DARK = "Darkness - lights lit"


def light_for_hour(hour):
    """Lighting label for an hour of day; in the dataset 06:00-17:59 is almost all daylight."""
    return LIGHT_DAY if 6 <= hour < 18 else LIGHT_DARK


# Conditions that call for a lower speed limit in advisories.
ADVERSE_WEATHER = {WEATHER_RAIN, WEATHER_RAIN_WIND, WEATHER_FOG, WEATHER_SNOW}
ADVERSE_ROAD = {ROAD_WET, ROAD_SNOW, "Flood over 3cm. deep"}

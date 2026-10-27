import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import label_maps  # noqa: E402
import navigation_engine as nav  # noqa: E402
import risk_model  # noqa: E402
import train  # noqa: E402

RAW = pd.read_csv(ROOT / "data" / "RTA_Dataset.csv")


# --- label mapping --------------------------------------------------------------------
def test_every_dataset_vehicle_is_mapped_or_missing():
    for raw in RAW["Type_of_vehicle"].dropna().unique():
        assert label_maps.map_vehicle(raw) in label_maps.VEHICLE_GROUPS


def test_vehicle_mapping_is_not_collapsed():
    mapped = RAW["Type_of_vehicle"].map(label_maps.map_vehicle)
    assert (mapped == label_maps.MOTORCYCLE).sum() == (RAW["Type_of_vehicle"] == "Motorcycle").sum()
    assert label_maps.map_vehicle("Lorry (11?40Q)") == label_maps.TRUCK_SHORT
    assert label_maps.map_vehicle("Stationwagen") == label_maps.CAR
    assert label_maps.map_vehicle(float("nan")) is None


def test_unknown_vehicle_raises():
    with pytest.raises(ValueError):
        label_maps.map_vehicle("Hovercraft")


@pytest.mark.parametrize("code,precip,wind,expected", [
    (0, 0.0, 5.0, ("Normal", "Dry")),
    (3, 0.0, 5.0, ("Cloudy", "Dry")),
    (0, 0.0, 40.0, ("Windy", "Dry")),
    (61, 1.2, 5.0, ("Raining", "Wet or damp")),
    (0, 0.4, 5.0, ("Raining", "Wet or damp")),
    (63, 2.0, 35.0, ("Raining and Windy", "Wet or damp")),
    (45, 0.0, 5.0, ("Fog or mist", "Wet or damp")),
    (73, 0.0, 5.0, ("Snow", "Snow")),
])
def test_classify_weather(code, precip, wind, expected):
    assert label_maps.classify_weather(code, precip, wind) == expected


# --- training pipeline & artifacts -----------------------------------------------------
def test_prepare_dataset_has_only_real_features():
    df = train.prepare_dataset(RAW)
    assert "Vehicle_Speed" not in df.columns
    assert set(df["Severity"]) == {"Minor", "Serious", "Fatal"}
    assert df["Time_Hour"].between(0, 23).all()
    assert df["Vehicle_Type"].notna().all()


@pytest.fixture(scope="module")
def artifacts():
    return risk_model.load_artifacts()


def test_artifacts_match_training_features(artifacts):
    _, _, feature_columns = artifacts
    assert feature_columns == train.FEATURES


def test_live_labels_are_known_to_the_encoders(artifacts):
    _, encoders, _ = artifacts
    weather = set(encoders["Weather_Condition"].classes_)
    road = set(encoders["Road_Condition"].classes_)
    for code, precip, wind in [(0, 0, 0), (3, 0, 0), (0, 0, 40), (61, 1, 0), (63, 2, 35), (45, 0, 0), (73, 0, 0)]:
        w, r = label_maps.classify_weather(code, precip, wind)
        assert w in weather and r in road
    assert set(label_maps.VEHICLE_GROUPS) <= set(encoders["Vehicle_Type"].classes_)
    light = set(encoders["Light_Condition"].classes_)
    assert {label_maps.light_for_hour(h) for h in range(24)} <= light


def test_predict_severity_for_every_vehicle(artifacts):
    model, encoders, features = artifacts
    for vehicle in label_maps.VEHICLE_GROUPS:
        severity, probs = risk_model.predict_severity(model, encoders, features, {
            "Time_Hour": 23, "Number_of_Vehicles": 2, "Weather_Condition": "Raining",
            "Road_Condition": "Wet or damp", "Vehicle_Type": vehicle,
            "Light_Condition": "Darkness - lights lit"})
        assert severity in {"Minor", "Serious", "Fatal"}
        assert sum(probs.values()) == pytest.approx(1.0)


def test_inputs_actually_reach_the_model(artifacts):
    model, encoders, features = artifacts
    base = {"Time_Hour": 14, "Number_of_Vehicles": 2, "Weather_Condition": "Normal",
            "Road_Condition": "Dry", "Vehicle_Type": "Car / Taxi",
            "Light_Condition": "Daylight"}
    _, p0 = risk_model.predict_severity(model, encoders, features, base)
    changed = [
        {**base, "Road_Condition": "Wet or damp"}, {**base, "Weather_Condition": "Raining"},
        {**base, "Vehicle_Type": "Motorcycle / Scooter"}, {**base, "Number_of_Vehicles": 5},
        {**base, "Time_Hour": 2}, {**base, "Light_Condition": "Darkness - lights lit"},
    ]
    assert any(risk_model.predict_severity(model, encoders, features, c)[1] != p0 for c in changed)


def test_unknown_label_and_missing_input_are_rejected(artifacts):
    model, encoders, features = artifacts
    base = {"Time_Hour": 14, "Number_of_Vehicles": 2, "Weather_Condition": "Normal",
            "Road_Condition": "Dry", "Vehicle_Type": "Car / Taxi",
            "Light_Condition": "Daylight"}
    with pytest.raises(ValueError):
        risk_model.predict_severity(model, encoders, features, {**base, "Weather_Condition": "Rainy"})
    with pytest.raises(KeyError):
        risk_model.predict_severity(model, encoders, features, {k: v for k, v in base.items() if k != "Time_Hour"})


def test_light_for_hour():
    assert label_maps.light_for_hour(12) == "Daylight"
    assert label_maps.light_for_hour(2) == label_maps.light_for_hour(21) == "Darkness - lights lit"


def test_grouped_split_keeps_each_crash_on_one_side():
    df = train.prepare_dataset(RAW)
    y = df["Severity"].astype("category").cat.codes.to_numpy()
    groups = df["Crash_Group"].to_numpy()
    for train_idx, test_idx in train.grouped_folds(y, groups):
        assert not set(groups[train_idx]) & set(groups[test_idx])


def test_prior_correction_flags_rare_classes_more_often():
    import numpy as np
    from severity_model import PriorCorrectedClassifier

    class Fixed:
        classes_ = np.array(["Fatal", "Minor", "Serious"])
        def predict_proba(self, X):
            return np.array([[0.05, 0.80, 0.15]])

    priors = [0.01, 0.85, 0.14]
    assert PriorCorrectedClassifier(Fixed(), priors, 0.0).predict(None)[0] == "Minor"
    corrected = PriorCorrectedClassifier(Fixed(), priors, 1.0)
    assert corrected.predict(None)[0] == "Fatal"
    assert corrected.predict_proba(None).sum() == pytest.approx(1.0)


def test_held_out_metrics_beat_the_old_pipeline():
    import json
    metrics = json.loads((ROOT / "models" / "metrics.json").read_text())
    assert "crash-grouped" in metrics["split"]
    assert metrics["accuracy"] > 0.60 and metrics["macro_f1"] > 0.36


# --- navigation engine -----------------------------------------------------------------
KUTHIRAN = next(s for s in nav.NATIONAL_BLACKSPOTS if s["name"].startswith("Kuthiran"))


def straight_route(lat, lon, n=200, step=0.001):
    """A due-east test route (about 110 m between points) passing through the given point."""
    return [(lat, lon + (i - n // 2) * step) for i in range(n)]


def test_scan_hazards_accepts_waypoints_and_reports_inside():
    route = straight_route(KUTHIRAN["lat"], KUTHIRAN["lon"])
    hazard = nav.scan_hazards(KUTHIRAN["lat"], KUTHIRAN["lon"], route)
    assert hazard["name"] == KUTHIRAN["name"] and hazard["status"] == "inside"


def test_scan_hazards_looks_ahead_but_not_behind():
    route = straight_route(KUTHIRAN["lat"], KUTHIRAN["lon"])
    before = nav.scan_hazards(*route[60], route)  # ~4.4 km west of the spot, heading east
    assert before and before["status"] == "ahead" and 0 < before["ahead_km"] <= nav.LOOKAHEAD_KM
    after = nav.scan_hazards(*route[150], route)  # ~5.5 km past it
    assert after is None


def test_scan_hazards_clear_and_two_argument_use():
    assert nav.scan_hazards(12.9716, 77.5946) is None  # Bengaluru, no waypoints


def test_scan_route_hazards_matches_timeline_contract():
    route = straight_route(KUTHIRAN["lat"], KUTHIRAN["lon"])
    found = nav.scan_route_hazards(route)
    assert [s["name"] for s in found] == [KUTHIRAN["name"]]
    spot = found[0]
    assert spot["enter_km"] <= spot["km"] <= spot["exit_km"]


def test_advisories_use_model_vocabulary():
    wet = {"Weather_Condition": "Raining", "Road_Condition": "Wet or damp"}
    dry = {"Weather_Condition": "Normal", "Road_Condition": "Dry"}
    fog = {"Weather_Condition": "Fog or mist", "Road_Condition": "Wet or damp"}
    assert any("Slippery" in a for a in nav.generate_advisory("Minor", 30, wet, None))
    assert any("Low Visibility" in a for a in nav.generate_advisory("Minor", 30, fog, None))
    assert any("Clear" in a for a in nav.generate_advisory("Minor", 30, dry, None))
    # Speed limit is lower in adverse conditions and the message names the limit it enforces.
    assert not any("Speed Alert" in a for a in nav.generate_advisory("Minor", 60, dry, None))
    alert = [a for a in nav.generate_advisory("Minor", 60, wet, None) if "Speed Alert" in a]
    assert alert and "50 km/h" in alert[0]


def test_geocode_parses_coordinates_without_network():
    assert nav.geocode_location("Thrissur (10.52000, 76.21000)") == (10.52, 76.21)
    assert nav.geocode_location("10.5, 76.2") == (10.5, 76.2)

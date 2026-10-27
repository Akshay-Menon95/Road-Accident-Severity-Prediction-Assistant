# evaluate_scenarios.py

# ==========================================
# TEST SCENARIOS (EVALUATION INPUTS)
# ==========================================
evaluation_scenarios = [
    {"speed_kmh": 65, "time_hour": 14, "weather": "Clear", "vehicle": "Car / Taxi", "location": "Standard Highway"},
    {"speed_kmh": 105, "time_hour": 23, "weather": "Wet", "vehicle": "Motorcycle", "location": "Standard Highway"},
    {"speed_kmh": 110, "time_hour": 8, "weather": "Clear", "vehicle": "Car / Taxi", "location": "High-Density Zone"},
    {"speed_kmh": 45, "time_hour": 16, "weather": "Clear", "vehicle": "HMV Lorry", "location": "Walayar–Kanjikode Blackspot"},
    {"speed_kmh": 130, "time_hour": 12, "weather": "Dry", "vehicle": "Car / Taxi", "location": "Standard Highway"},
    {"speed_kmh": 50, "time_hour": 2, "weather": "Foggy", "vehicle": "Motorcycle", "location": "Rural Road"},
    {"speed_kmh": 85, "time_hour": 19, "weather": "Rainy", "vehicle": "Bus", "location": "Mundur–Kongad Junction (SH 26)"},
    {"speed_kmh": 140, "time_hour": 3, "weather": "Clear", "vehicle": "Car / Taxi", "location": "Kuthiran Tunnel Approach"},
    {"speed_kmh": 30, "time_hour": 9, "weather": "Wet", "vehicle": "Auto Rickshaw", "location": "City Center"},
    {"speed_kmh": 115, "time_hour": 18, "weather": "Clear", "vehicle": "Motorcycle", "location": "Standard Highway"},
    {"speed_kmh": 75, "time_hour": 14, "weather": "Heavy Rain", "vehicle": "HMV Lorry", "location": "Lakkidi / Mangalam Highway"},
    {"speed_kmh": 60, "time_hour": 22, "weather": "Clear", "vehicle": "Pedestrian/Bicycle", "location": "Rural Road"}
]

# ==========================================
# GROUND TRUTHS & EXPECTED SYSTEM BEHAVIOR
# ==========================================
ground_truths = [
    {"expected_severity": "Minor", "expected_advisory": "Standard driving conditions. Maintain lane discipline."},
    {"expected_severity": "Fatal", "expected_advisory": "Severe risk: High speed in low visibility/wet conditions."},
    {"expected_severity": "Serious", "expected_advisory": "Peak traffic hours. High risk of multi-vehicle collision."},
    {"expected_severity": "Serious", "expected_advisory": "Proximity Alert: High-speed freight cross-traffic zone."},
    {"expected_severity": "Fatal", "expected_advisory": "Critical speed threshold exceeded for daytime limits."},
    {"expected_severity": "Serious", "expected_advisory": "Poor nighttime visibility. Reduce speed immediately."},
    {"expected_severity": "Serious", "expected_advisory": "Active Blackspot: Narrow carriageway with high two-wheeler density."},
    {"expected_severity": "Fatal", "expected_advisory": "Extreme risk: Over-speeding at tunnel approach during night hours."},
    {"expected_severity": "Minor", "expected_advisory": "Safe speed for wet conditions. Beware of slippery roads."},
    {"expected_severity": "Serious", "expected_advisory": "Peak traffic hours. High speed for two-wheelers increases severe injury risk."},
    {"expected_severity": "Serious", "expected_advisory": "Heavy vehicle in poor weather conditions. Braking distance significantly increased."},
    {"expected_severity": "Minor", "expected_advisory": "Nighttime rural road. Ensure reflective gear is visible."}
]

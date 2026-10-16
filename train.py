# train.py
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

os.makedirs("models", exist_ok=True)

print("📡 Fetching real historical accident dataset from GitHub...")
url = "https://raw.githubusercontent.com/avikumart/Road-Traffic-Severity-Classification-Project/main/RTA%20Dataset.csv"

try:
    df = pd.read_csv(url)
    print(f"✅ Successfully loaded {len(df)} real accident records.")
except Exception as e:
    print(f"Failed to fetch data: {e}")
    exit()

# 1. Clean and Map Columns to our App's API
df.rename(columns={
    'Weather_conditions': 'Weather_Condition',
    'Road_surface_conditions': 'Road_Condition',
    'Type_of_vehicle': 'Vehicle_Type',
    'Number_of_vehicles_involved': 'Number_of_Vehicles'
}, inplace=True)

# Process Time into Hour format safely (Fixes the dateutil warning)
df['Time_Hour'] = df['Time'].astype(str).str.split(':').str[0]
df['Time_Hour'] = pd.to_numeric(df['Time_Hour'], errors='coerce').fillna(12).astype(int)

# Map the target variable to our standard Severity labels
severity_mapping = {
    "Slight Injury": "Minor", 
    "Serious Injury": "Serious", 
    "Fatal injury": "Fatal"
}
df['Severity'] = df['Accident_severity'].map(severity_mapping)

features = ['Time_Hour', 'Weather_Condition', 'Road_Condition', 'Vehicle_Type', 'Number_of_Vehicles']
df.dropna(subset=features + ['Severity'], inplace=True)

X = df[features].copy()
y = df['Severity'].copy()

# 2. Encode Categorical Data (Fixes the strict dtype error)
encoders = {}
for col in ['Weather_Condition', 'Road_Condition', 'Vehicle_Type']:
    le = LabelEncoder()
    # Direct assignment circumvents strict string dtype constraints
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le

target_le = LabelEncoder()
y_encoded = target_le.fit_transform(y)
encoders['Severity'] = target_le

# 3. Train Model
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

clf = DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"🎯 Model trained on REAL Data. Accuracy: {acc * 100:.2f}%")

# 4. Save Artifacts
joblib.dump(clf, "models/decision_tree_model.joblib")
joblib.dump(encoders, "models/label_encoders.joblib")
joblib.dump(features, "models/feature_columns.joblib")
print("💾 Model artifacts saved to 'models/' folder.")
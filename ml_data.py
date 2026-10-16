# ml_data.py
"""Dataset loading and preprocessing shared by train.py and ml_eval.py.

Keeping this in one place guarantees the evaluation reproduces the exact
train/test split that the saved model was trained on.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_URL = "https://raw.githubusercontent.com/avikumart/Road-Traffic-Severity-Classification-Project/main/RTA%20Dataset.csv"

FEATURES = ['Time_Hour', 'Weather_Condition', 'Road_Condition', 'Vehicle_Type', 'Number_of_Vehicles']
CATEGORICAL = ['Weather_Condition', 'Road_Condition', 'Vehicle_Type']
SEVERITY_MAPPING = {
    "Slight Injury": "Minor",
    "Serious Injury": "Serious",
    "Fatal injury": "Fatal",
}
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_raw(url=DATA_URL):
    return pd.read_csv(url)


def prepare(df):
    """Maps the raw RTA columns to the app's schema. Returns (X, y) with un-encoded categoricals."""
    df = df.rename(columns={
        'Weather_conditions': 'Weather_Condition',
        'Road_surface_conditions': 'Road_Condition',
        'Type_of_vehicle': 'Vehicle_Type',
        'Number_of_vehicles_involved': 'Number_of_Vehicles'
    })

    # Process Time into Hour format safely
    df['Time_Hour'] = df['Time'].astype(str).str.split(':').str[0]
    df['Time_Hour'] = pd.to_numeric(df['Time_Hour'], errors='coerce').fillna(12).astype(int)

    df['Severity'] = df['Accident_severity'].map(SEVERITY_MAPPING)
    df = df.dropna(subset=FEATURES + ['Severity'])
    return df[FEATURES].copy(), df['Severity'].copy()


def encode(X, y, encoders=None):
    """Label-encodes features and target. Fits new encoders unless `encoders` is given."""
    X = X.copy()
    fit = encoders is None
    encoders = {} if fit else encoders
    for col in CATEGORICAL:
        if fit:
            encoders[col] = LabelEncoder().fit(X[col].astype(str))
        X[col] = encoders[col].transform(X[col].astype(str))
    if fit:
        encoders['Severity'] = LabelEncoder().fit(y)
    return X, encoders['Severity'].transform(y), encoders


def split(X, y_encoded):
    return train_test_split(X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded)

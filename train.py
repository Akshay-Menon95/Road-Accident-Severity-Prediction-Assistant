# train.py
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from label_maps import map_vehicle

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "RTA_Dataset.csv"
MODEL_DIR = ROOT / "models"
RAW_URL = "https://raw.githubusercontent.com/avikumart/Road-Traffic-Severity-Classification-Project/main/RTA%20Dataset.csv"

SEED = 42
# Only features that exist in the dataset AND can be known at prediction time.
# Vehicle speed is deliberately absent: the dataset has no speed column.
FEATURES = ['Time_Hour', 'Weather_Condition', 'Road_Condition', 'Vehicle_Type', 'Number_of_Vehicles']
CATEGORICAL = ['Weather_Condition', 'Road_Condition', 'Vehicle_Type']
SEVERITY_MAP = {"Slight Injury": "Minor", "Serious Injury": "Serious", "Fatal injury": "Fatal"}


def load_raw(data_path=DATA_PATH):
    """Reads the RTA dataset, downloading it first if the local copy is missing or truncated."""
    if data_path.exists():
        try:
            df = pd.read_csv(data_path)
            if len(df) > 1000:
                return df
        except Exception:
            pass
    print("Downloading raw CSV...")
    df = pd.read_csv(RAW_URL)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(data_path, index=False)
    return df


def prepare_dataset(raw):
    """Renames, maps and filters the raw RTA table down to FEATURES + Severity."""
    df = raw.rename(columns={
        'Weather_conditions': 'Weather_Condition',
        'Road_surface_conditions': 'Road_Condition',
        'Number_of_vehicles_involved': 'Number_of_Vehicles',
    })
    df['Time_Hour'] = pd.to_numeric(df['Time'].astype(str).str.split(':').str[0], errors='coerce')
    df['Severity'] = df['Accident_severity'].map(SEVERITY_MAP)
    df['Vehicle_Type'] = df['Type_of_vehicle'].map(map_vehicle)

    before = len(df)
    df = df.dropna(subset=FEATURES + ['Severity'])
    df = df[df['Time_Hour'].between(0, 23)].copy()
    df['Time_Hour'] = df['Time_Hour'].astype(int)
    print(f"Rows kept: {len(df)} of {before} (dropped rows missing vehicle type, hour or severity)")
    return df


def evaluate(clf, X_test, y_test, labels):
    pred = clf.predict(X_test)
    majority = float(np.max(np.bincount(y_test)) / len(y_test))
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "majority_class_baseline_accuracy": majority,
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average='macro')),
        "labels": list(labels),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "per_class": classification_report(y_test, pred, target_names=labels,
                                           output_dict=True, zero_division=0),
    }


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    df = prepare_dataset(load_raw())

    X = df[FEATURES].copy()
    encoders = {}
    for col in CATEGORICAL:
        encoders[col] = LabelEncoder().fit(X[col].astype(str))
        X[col] = encoders[col].transform(X[col].astype(str))
    encoders['Severity'] = LabelEncoder().fit(df['Severity'])
    y = encoders['Severity'].transform(df['Severity'])
    labels = list(encoders['Severity'].classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)

    # Fatal is ~1% of rows, so classes are re-weighted and tuning optimises macro-F1;
    # plain accuracy is won by always predicting "Minor".
    search = GridSearchCV(
        DecisionTreeClassifier(class_weight='balanced', random_state=SEED),
        {'max_depth': [3, 4, 5, 6, 8], 'min_samples_leaf': [20, 50, 100]},
        scoring='f1_macro',
        cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
    ).fit(X_train, y_train)
    clf = search.best_estimator_

    metrics = evaluate(clf, X_test, y_test, labels)
    metrics.update({
        "rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "best_params": search.best_params_,
        "cv_macro_f1_on_train": float(search.best_score_),
    })

    print(f"Best params: {search.best_params_}")
    print(f"Held-out test set ({len(X_test)} rows, stratified):")
    print(f"  accuracy          {metrics['accuracy'] * 100:.2f}%  "
          f"(always predicting the majority class: {metrics['majority_class_baseline_accuracy'] * 100:.2f}%)")
    print(f"  balanced accuracy {metrics['balanced_accuracy'] * 100:.2f}%")
    print(f"  macro F1          {metrics['macro_f1']:.3f}")
    print(f"  labels {labels}\n{np.array(metrics['confusion_matrix'])}")
    print("Note: class weighting trades raw accuracy for recall on Serious/Fatal, and the "
          "dataset has weak signal, so treat outputs as relative risk, not calibrated probabilities.")

    joblib.dump(clf, MODEL_DIR / "decision_tree_model.joblib")
    joblib.dump(encoders, MODEL_DIR / "label_encoders.joblib")
    joblib.dump(FEATURES, MODEL_DIR / "feature_columns.joblib")
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

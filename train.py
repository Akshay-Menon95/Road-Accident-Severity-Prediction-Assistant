# train.py
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

from label_maps import map_vehicle
from severity_model import PriorCorrectedClassifier

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "RTA_Dataset.csv"
MODEL_DIR = ROOT / "models"
RAW_URL = "https://raw.githubusercontent.com/avikumart/Road-Traffic-Severity-Classification-Project/main/RTA%20Dataset.csv"

SEED = 42
# Only features that exist in the dataset AND can be known at prediction time.
# Vehicle speed is deliberately absent: the dataset has no speed column.
FEATURES = ['Time_Hour', 'Weather_Condition', 'Road_Condition', 'Vehicle_Type',
            'Number_of_Vehicles', 'Light_Condition']
CATEGORICAL = ['Weather_Condition', 'Road_Condition', 'Vehicle_Type', 'Light_Condition']
SEVERITY_MAP = {"Slight Injury": "Minor", "Serious Injury": "Serious", "Fatal injury": "Fatal"}

FOREST_TREES = 300
LEAF_GRID = [25, 50, 100]
EXPONENT_GRID = np.round(np.arange(0.0, 1.21, 0.1), 1)


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
    """Renames, maps and filters the raw RTA table down to FEATURES + Severity + Crash_Group."""
    df = raw.rename(columns={
        'Weather_conditions': 'Weather_Condition',
        'Road_surface_conditions': 'Road_Condition',
        'Number_of_vehicles_involved': 'Number_of_Vehicles',
        'Light_conditions': 'Light_Condition',
    })
    df['Time_Hour'] = pd.to_numeric(df['Time'].astype(str).str.split(':').str[0], errors='coerce')
    df['Severity'] = df['Accident_severity'].map(SEVERITY_MAP)
    df['Vehicle_Type'] = df['Type_of_vehicle'].map(map_vehicle)
    # The file has one row per vehicle/casualty, so rows of one crash share a timestamp
    # and weekday. Rows with the same group must never straddle the train/test split.
    df['Crash_Group'] = df['Time'].astype(str) + '|' + df['Day_of_week'].astype(str)

    before = len(df)
    df = df.dropna(subset=FEATURES + ['Severity'])
    df = df[df['Time_Hour'].between(0, 23)].copy()
    df['Time_Hour'] = df['Time_Hour'].astype(int)
    print(f"Rows kept: {len(df)} of {before} (dropped rows missing vehicle type, hour or severity)")
    return df


def grouped_folds(y, groups, n_splits=5):
    """Stratified folds in which every crash group lands entirely in one fold."""
    return list(StratifiedGroupKFold(n_splits, shuffle=True, random_state=SEED).split(y, y, groups))


def make_forest(leaf):
    return RandomForestClassifier(FOREST_TREES, min_samples_leaf=leaf, n_jobs=-1, random_state=SEED)


def out_of_fold_probabilities(X, y, groups, leaf):
    """Class probabilities for every row from a model that never saw that row's crash."""
    proba = np.zeros((len(y), len(np.unique(y))))
    for train_idx, val_idx in grouped_folds(y, groups):
        forest = make_forest(leaf).fit(X.iloc[train_idx], y[train_idx])
        proba[val_idx] = forest.predict_proba(X.iloc[val_idx])
    return proba


def tuning_score(y, pred):
    """Macro-F1 balances precision and recall per class, balanced accuracy rewards flagging
    the rare Fatal/Serious classes; averaging them avoids settings that never flag Fatal."""
    return 0.5 * (f1_score(y, pred, average='macro') + balanced_accuracy_score(y, pred))


def tune(X, y, groups, priors):
    """Picks forest leaf size and prior-correction exponent by crash-grouped CV."""
    best = None
    for leaf in LEAF_GRID:
        proba = out_of_fold_probabilities(X, y, groups, leaf)
        for exponent in EXPONENT_GRID:
            pred = (proba / priors ** exponent).argmax(axis=1)
            score = tuning_score(y, pred)
            if best is None or score > best['score']:
                best = {'score': float(score), 'leaf': leaf, 'exponent': float(exponent)}
    return best


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
    groups = df['Crash_Group'].to_numpy()

    # Hold out one grouped fold (~20% of crashes) as the final test set.
    train_idx, test_idx = grouped_folds(y, groups)[0]
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    assert not set(groups[train_idx]) & set(groups[test_idx]), "crash rows leaked across the split"

    # Plain accuracy is won by always predicting "Minor" (~84%), so the decision point is
    # tuned on out-of-fold predictions instead of being left at argmax.
    priors = np.bincount(y_train) / len(y_train)
    best = tune(X_train, y_train, groups[train_idx], priors)
    forest = make_forest(best['leaf']).fit(X_train, y_train)
    clf = PriorCorrectedClassifier(forest, priors, best['exponent'])

    metrics = evaluate(clf, X_test, y_test, labels)
    metrics.update({
        "rows": int(len(df)),
        "crashes": int(df['Crash_Group'].nunique()),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "split": "crash-grouped (no crash appears in both train and test)",
        "best_params": {"min_samples_leaf": best['leaf'], "prior_exponent": best['exponent']},
        "cv_tuning_score_on_train": best['score'],
    })

    print(f"Best params: leaf={best['leaf']}, prior exponent={best['exponent']}")
    print(f"Held-out test set ({len(X_test)} rows, crash-grouped, stratified):")
    print(f"  accuracy          {metrics['accuracy'] * 100:.2f}%  "
          f"(always predicting the majority class: {metrics['majority_class_baseline_accuracy'] * 100:.2f}%)")
    print(f"  balanced accuracy {metrics['balanced_accuracy'] * 100:.2f}%")
    print(f"  macro F1          {metrics['macro_f1']:.3f}")
    print(f"  labels {labels}\n{np.array(metrics['confusion_matrix'])}")
    print("Note: the dataset has weak signal and outputs are prior-corrected, so treat them as "
          "relative risk, not calibrated probabilities.")

    joblib.dump(clf, MODEL_DIR / "severity_model.joblib")
    joblib.dump(encoders, MODEL_DIR / "label_encoders.joblib")
    joblib.dump(FEATURES, MODEL_DIR / "feature_columns.joblib")
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

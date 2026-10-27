# risk_model.py
"""Loading the trained artifacts and turning dashboard inputs into a severity prediction."""
from pathlib import Path

import joblib
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent / "models"


def load_artifacts(model_dir=MODEL_DIR):
    # joblib unpickles, so only load artifacts produced by this repo's own train.py.
    model_dir = Path(model_dir)
    return (
        joblib.load(model_dir / "decision_tree_model.joblib"),
        joblib.load(model_dir / "label_encoders.joblib"),
        joblib.load(model_dir / "feature_columns.joblib"),
    )


def encode_value(encoders, column, value):
    """Encodes a categorical label; unknown labels raise instead of defaulting to class 0."""
    encoder = encoders[column]
    if value not in encoder.classes_:
        raise ValueError(
            f"{value!r} is not a known {column} label. Known: {list(encoder.classes_)}. "
            "Re-run 'python train.py' if the model artifacts are out of date."
        )
    return int(encoder.transform([value])[0])


def predict_severity(model, encoders, feature_columns, inputs):
    """Returns (severity label, {severity label: probability}) for one set of inputs.

    `inputs` maps every feature name to a raw value: categorical features as their
    string label, numeric features as numbers.
    """
    missing = [c for c in feature_columns if c not in inputs]
    if missing:
        raise KeyError(f"Missing model inputs: {missing}")
    row = {c: (encode_value(encoders, c, inputs[c]) if c in encoders else inputs[c])
           for c in feature_columns}
    frame = pd.DataFrame([row])[feature_columns]
    labels = encoders["Severity"].classes_
    probabilities = model.predict_proba(frame)[0]
    probs = {str(label): float(p) for label, p in zip(labels, probabilities)}
    return max(probs, key=probs.get), probs

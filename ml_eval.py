# ml_eval.py
"""Model evaluation: confusion matrices and accuracy metrics, recorded to models/.

Run `python ml_eval.py` to score the saved model on the held-out test split, or
let train.py call `evaluate_and_record` after each (re)training. Every run
overwrites models/metrics.json (shown in the app's "Model performance" section)
and appends a row to models/metrics_history.csv so accuracy can be tracked
across retrains.
"""
import csv
import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np
import sklearn
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix,
    f1_score, matthews_corrcoef, precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

import ml_data

MODELS_DIR = "models"
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")
HISTORY_PATH = os.path.join(MODELS_DIR, "metrics_history.csv")
DISPLAY_ORDER = ["Minor", "Serious", "Fatal"]  # least -> most severe
CV_FOLDS = 5

HISTORY_FIELDS = [
    "timestamp", "source", "n_test", "accuracy", "balanced_accuracy", "macro_f1",
    "fatal_recall", "fatal_precision", "cv_accuracy",
]


def _safe_div(a, b):
    return a / b if b else 0.0


def compute_metrics(model, target_encoder, X_train, y_train, X_test, y_test, X_all, y_all, source):
    classes = list(target_encoder.classes_)
    labels = [c for c in DISPLAY_ORDER if c in classes] + [c for c in classes if c not in DISPLAY_ORDER]
    label_idx = [classes.index(c) for c in labels]

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=label_idx)
    row_sums, col_sums = cm.sum(axis=1), cm.sum(axis=0)
    cm_recall = [[_safe_div(v, row_sums[i]) for v in row] for i, row in enumerate(cm)]
    cm_precision = [[_safe_div(v, col_sums[j]) for j, v in enumerate(row)] for row in cm]

    prec, rec, f1, support = precision_recall_fscore_support(y_test, y_pred, labels=label_idx, zero_division=0)
    per_class = {
        lab: {"precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i]), "support": int(support[i])}
        for i, lab in enumerate(labels)
    }

    # Cross-validation on the full dataset: a steadier estimate than one 80/20 split.
    cv = cross_validate(
        clone(model), X_all, y_all,
        cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=ml_data.RANDOM_STATE),
        scoring={"accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy", "macro_f1": "f1_macro"},
    )
    cv_summary = {
        name: {"mean": float(cv[f"test_{name}"].mean()), "std": float(cv[f"test_{name}"].std())}
        for name in ("accuracy", "balanced_accuracy", "macro_f1")
    }

    test_counts = np.bincount(y_test, minlength=len(classes))
    params = model.get_params()
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": source,
        "evaluated_on": f"held-out {int(ml_data.TEST_SIZE * 100)}% test split (random_state={ml_data.RANDOM_STATE}, stratified)",
        "model": {
            "type": type(model).__name__,
            "params": {k: params[k] for k in ("max_depth", "class_weight", "random_state") if k in params},
            "sklearn_version": sklearn.__version__,
        },
        "dataset": {
            "rows": int(len(y_all)),
            "train": int(len(y_train)),
            "test": int(len(y_test)),
            "test_class_counts": {classes[i]: int(test_counts[i]) for i in label_idx},
        },
        "labels": labels,
        "confusion_matrix": cm.astype(int).tolist(),
        "confusion_matrix_recall": cm_recall,
        "confusion_matrix_precision": cm_precision,
        "overall": {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
            "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
            "cohen_kappa": float(cohen_kappa_score(y_test, y_pred)),
            "mcc": float(matthews_corrcoef(y_test, y_pred)),
            "majority_baseline": float(test_counts.max() / test_counts.sum()),
            "train_accuracy": float(accuracy_score(y_train, model.predict(X_train))),
        },
        "per_class": per_class,
        "cross_validation": {"folds": CV_FOLDS, **cv_summary},
    }


def _history_row(m):
    fatal = m["per_class"].get("Fatal", {})
    return {
        "timestamp": m["generated_at"],
        "source": m["source"],
        "n_test": m["dataset"]["test"],
        "accuracy": round(m["overall"]["accuracy"], 4),
        "balanced_accuracy": round(m["overall"]["balanced_accuracy"], 4),
        "macro_f1": round(m["overall"]["macro_f1"], 4),
        "fatal_recall": round(fatal.get("recall", 0.0), 4),
        "fatal_precision": round(fatal.get("precision", 0.0), 4),
        "cv_accuracy": round(m["cross_validation"]["accuracy"]["mean"], 4),
    }


def record(metrics, models_dir=MODELS_DIR):
    """Writes metrics.json and appends to the history (skipping a run identical to the last one)."""
    os.makedirs(models_dir, exist_ok=True)
    with open(os.path.join(models_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    history_path = os.path.join(models_dir, "metrics_history.csv")
    row = _history_row(metrics)
    rows = []
    if os.path.exists(history_path):
        with open(history_path, newline="") as f:
            rows = list(csv.DictReader(f))
    compare = [k for k in HISTORY_FIELDS if k not in ("timestamp", "source")]
    if rows and all(str(rows[-1][k]) == str(row[k]) for k in compare):
        return False
    rows.append(row)
    with open(history_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return True


def evaluate_and_record(model, encoders, X_train, y_train, X_test, y_test, X_all, y_all, source):
    metrics = compute_metrics(model, encoders["Severity"], X_train, y_train, X_test, y_test, X_all, y_all, source)
    appended = record(metrics)
    return metrics, appended


def print_report(m):
    o, labels = m["overall"], m["labels"]
    print(f"\nEvaluated on {m['evaluated_on']}: {m['dataset']['test']} of {m['dataset']['rows']} rows")
    print(f"  Accuracy            {o['accuracy']:.3f}   (always-predict-majority baseline: {o['majority_baseline']:.3f})")
    print(f"  Balanced accuracy   {o['balanced_accuracy']:.3f}")
    print(f"  Macro F1            {o['macro_f1']:.3f}")
    print(f"  Train accuracy      {o['train_accuracy']:.3f}")
    cv = m["cross_validation"]
    print(f"  {cv['folds']}-fold CV accuracy  {cv['accuracy']['mean']:.3f} +/- {cv['accuracy']['std']:.3f}")
    width = max(len(l) for l in labels) + 2
    print("\nConfusion matrix (rows = actual, columns = predicted)")
    print(" " * width + "".join(f"{l:>{width}}" for l in labels))
    for lab, row in zip(labels, m["confusion_matrix"]):
        print(f"{lab:<{width}}" + "".join(f"{v:>{width}}" for v in row))
    print("\nPer class")
    for lab in labels:
        c = m["per_class"][lab]
        print(f"  {lab:<8} precision {c['precision']:.3f}  recall {c['recall']:.3f}  f1 {c['f1']:.3f}  support {c['support']}")


def main():
    # joblib/pickle: these artifacts are the repo's own files written by train.py (same as app.py loads); never point this at untrusted files.
    model = joblib.load(os.path.join(MODELS_DIR, "decision_tree_model.joblib"))
    encoders = joblib.load(os.path.join(MODELS_DIR, "label_encoders.joblib"))

    print("Fetching dataset...")
    X_raw, y_raw = ml_data.prepare(ml_data.load_raw())
    X, y, _ = ml_data.encode(X_raw, y_raw, encoders=encoders)
    X_train, X_test, y_train, y_test = ml_data.split(X, y)

    metrics, appended = evaluate_and_record(model, encoders, X_train, y_train, X_test, y_test, X, y, "ml_eval.py")
    print_report(metrics)
    print(f"\nSaved {METRICS_PATH}" + (f" and appended to {HISTORY_PATH}" if appended else " (history unchanged)"))


if __name__ == "__main__":
    main()

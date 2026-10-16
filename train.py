# train.py
import os
import sys
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

import ml_data
import ml_eval

os.makedirs("models", exist_ok=True)

print("📡 Fetching real historical accident dataset from GitHub...")
try:
    df = ml_data.load_raw()
    print(f"✅ Successfully loaded {len(df)} real accident records.")
except Exception as e:
    print(f"Failed to fetch data: {e}")
    sys.exit(1)

# 1. Clean, map columns to our App's API, and encode categorical data
X_raw, y_raw = ml_data.prepare(df)
X, y_encoded, encoders = ml_data.encode(X_raw, y_raw)

# 2. Train Model
X_train, X_test, y_train, y_test = ml_data.split(X, y_encoded)

clf = DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=42)
clf.fit(X_train, y_train)

acc = accuracy_score(y_test, clf.predict(X_test))
print(f"🎯 Model trained on REAL Data. Accuracy: {acc * 100:.2f}%")

# 3. Save Artifacts
joblib.dump(clf, "models/decision_tree_model.joblib")
joblib.dump(encoders, "models/label_encoders.joblib")
joblib.dump(ml_data.FEATURES, "models/feature_columns.joblib")
print("💾 Model artifacts saved to 'models/' folder.")

# 4. Evaluate: confusion matrices + accuracy metrics, tracked across runs
metrics, appended = ml_eval.evaluate_and_record(clf, encoders, X_train, y_train, X_test, y_test, X, y_encoded, "train.py")
ml_eval.print_report(metrics)
print(f"\n📊 Metrics saved to {ml_eval.METRICS_PATH}" + (f" and logged in {ml_eval.HISTORY_PATH}" if appended else " (history unchanged)"))

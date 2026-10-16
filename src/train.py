"""Train a reproducible demo classifier and save candidate artifacts."""
import json
import os
from pathlib import Path

import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", "artifacts"))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
N_ESTIMATORS = int(os.getenv("N_ESTIMATORS", "150"))


def train() -> dict:
    """Load demo data, train candidate, evaluate on a held-out test set."""
    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data
    y = dataset.target

    if X.empty or X.isnull().all().any():
        raise ValueError("Input data is empty or has an entirely-null feature.")
    if y.isnull().any():
        raise ValueError("Target labels contain null values.")
    if X.columns.duplicated().any():
        raise ValueError("Feature names must be unique.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        )),
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "features": list(X.columns),
        "model_type": "RandomForestClassifier",
        "random_state": RANDOM_STATE,
        "n_estimators": N_ESTIMATORS,
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ARTIFACT_DIR / "model.joblib")
    (ARTIFACT_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (ARTIFACT_DIR / "feature_names.json").write_text(
        json.dumps(list(X.columns), indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    train()

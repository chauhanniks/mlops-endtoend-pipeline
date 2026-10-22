"""Train a reproducible demo classifier and save candidate artifacts."""
import hashlib
import json
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", "artifacts"))
DATA_PATH = Path(os.getenv("DATA_PATH", "data/breast_cancer.csv"))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
N_ESTIMATORS = int(os.getenv("N_ESTIMATORS", "150"))


def train() -> dict:
    """Load demo data, train candidate, evaluate on a held-out test set."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Training data not found: {DATA_PATH}. Run `dvc repro` to prepare it."
        )

    dataset = pd.read_csv(DATA_PATH)
    if "target" not in dataset.columns:
        raise ValueError("Input data must contain a 'target' column.")
    y = dataset.pop("target")
    X = dataset

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
    input_example = X_train.head(5)
    signature = infer_signature(input_example, model.predict(input_example))

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "features": list(X.columns),
        "model_type": "RandomForestClassifier",
        "random_state": RANDOM_STATE,
        "n_estimators": N_ESTIMATORS,
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = ARTIFACT_DIR / "model.joblib"
    metrics_path = ARTIFACT_DIR / "metrics.json"
    features_path = ARTIFACT_DIR / "feature_names.json"
    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    features_path.write_text(json.dumps(list(X.columns), indent=2), encoding="utf-8")

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "mlops-demo"))
    with mlflow.start_run(run_name="random-forest-candidate"):
        mlflow.log_params({
            "random_state": RANDOM_STATE,
            "n_estimators": N_ESTIMATORS,
            "model_type": metrics["model_type"],
            "dataset_sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
        })
        mlflow.log_metrics({
            key: metrics[key] for key in ("accuracy", "precision", "recall", "f1_score")
        })
        mlflow.sklearn.log_model(
            model,
            name="model",
            signature=signature,
            input_example=input_example,
            serialization_format="skops",
            skops_trusted_types=["numpy.dtype", "sklearn.tree._tree.Tree"],
        )
        mlflow.log_artifact(str(metrics_path))
        mlflow.log_artifact(str(features_path))

    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    train()

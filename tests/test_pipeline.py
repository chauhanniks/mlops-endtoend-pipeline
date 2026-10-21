import json

import pytest
from mlflow.tracking import MlflowClient

from src.evaluate import evaluate
from src.prepare_data import prepare_data
from src.train import train


def test_training_creates_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr("src.train.ARTIFACT_DIR", tmp_path)
    data_path = prepare_data(tmp_path / "breast_cancer.csv")
    monkeypatch.setattr("src.train.DATA_PATH", data_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file:{tmp_path / 'mlruns'}")
    metrics = train()
    assert (tmp_path / "model.joblib").exists()
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "feature_names.json").exists()
    assert 0.0 <= metrics["f1_score"] <= 1.0

    tracking_uri = f"file:{tmp_path / 'mlruns'}"
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name("mlops-demo")
    assert experiment is not None
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    assert runs[0].data.metrics["f1_score"] == metrics["f1_score"]
    assert client.list_artifacts(runs[0].info.run_id, "model")


def test_quality_gate_passes(tmp_path):
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps({"f1_score": 0.95}))
    result = evaluate(str(metrics_path), min_f1=0.90)
    assert result["f1_score"] == 0.95


def test_quality_gate_fails(tmp_path):
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps({"f1_score": 0.50}))
    with pytest.raises(SystemExit):
        evaluate(str(metrics_path), min_f1=0.90)

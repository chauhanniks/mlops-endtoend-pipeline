import json

import pytest

from src.evaluate import evaluate
from src.train import train


def test_training_creates_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr("src.train.ARTIFACT_DIR", tmp_path)
    metrics = train()
    assert (tmp_path / "model.joblib").exists()
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "feature_names.json").exists()
    assert 0.0 <= metrics["f1_score"] <= 1.0


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

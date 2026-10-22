from fastapi.testclient import TestClient

from src import api


def test_health_without_model(monkeypatch):
    monkeypatch.setattr(api, "_model", None)
    response = TestClient(api.app).get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is False


def test_predict_without_model(monkeypatch):
    monkeypatch.setattr(api, "_model", None)
    response = TestClient(api.app).post(
        "/predict", json={"features": {"x": 1.0}}
    )
    assert response.status_code == 503


def test_metrics_endpoint_exposes_request_metrics():
    client = TestClient(api.app)
    client.get("/health")
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "mlops_http_requests_total" in response.text
    assert 'route="/health"' in response.text


def test_predict_records_prediction_metric(monkeypatch):
    class FakeModel:
        def predict(self, row):
            return [1]

    monkeypatch.setattr(api, "_model", FakeModel())
    monkeypatch.setattr(api, "_features", ["feature"])
    client = TestClient(api.app)

    prediction = client.post("/predict", json={"features": {"feature": 2.0}})
    metrics = client.get("/metrics")

    assert prediction.status_code == 200
    assert prediction.json()["prediction"] == 1
    assert 'mlops_predictions_total{prediction="1"}' in metrics.text


def test_load_model_selects_explicit_version(tmp_path, monkeypatch):
    version = "run-123"
    version_dir = tmp_path / "versions" / version
    version_dir.mkdir(parents=True)
    model_path = version_dir / "model.joblib"
    features_path = version_dir / "feature_names.json"
    model_path.write_text("placeholder")
    features_path.write_text('["feature"]')
    monkeypatch.setattr(api, "MODEL_PATH", tmp_path / "model.joblib")
    monkeypatch.setattr(api, "FEATURES_PATH", tmp_path / "feature_names.json")
    monkeypatch.setattr(api, "VERSION_MANIFEST_PATH", tmp_path / "model_version.json")
    monkeypatch.setenv("MODEL_VERSION", version)
    monkeypatch.setattr(api.joblib, "load", lambda path: {"loaded_from": path})
    monkeypatch.setattr(api, "_model", None)

    api.load_model()

    assert api._model == {"loaded_from": model_path}
    assert api._features == ["feature"]
    assert api._model_version == version

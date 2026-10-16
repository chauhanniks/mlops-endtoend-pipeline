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

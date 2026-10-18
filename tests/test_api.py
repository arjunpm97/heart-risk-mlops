import pytest
from fastapi.testclient import TestClient

from api.main import create_app


@pytest.fixture()
def client(trained_model_path):
    return TestClient(create_app(trained_model_path))


def test_health_and_ready(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}
    assert client.get("/ready").status_code == 200


def test_predict_returns_prediction_and_confidence(client, sample_record):
    resp = client.post("/predict", json=sample_record, headers={"x-request-id": "abc123"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["prediction"] in (0, 1)
    assert body["label"] in ("heart_disease", "no_heart_disease")
    assert 0.5 <= body["confidence"] <= 1
    assert body["risk_band"] in ("low", "moderate", "high")
    assert body["request_id"] == "abc123"
    assert resp.headers["x-request-id"] == "abc123"


def test_predict_accepts_missing_optional_fields(client, sample_record):
    sample_record.pop("ca")
    sample_record.pop("thal")
    assert client.post("/predict", json=sample_record).status_code == 200


@pytest.mark.parametrize("field,value", [("age", -5), ("cp", 7), ("chol", "high")])
def test_predict_rejects_invalid_input(client, sample_record, field, value):
    sample_record[field] = value
    assert client.post("/predict", json=sample_record).status_code == 422


def test_predict_rejects_unknown_fields(client, sample_record):
    sample_record["smoker"] = 1
    assert client.post("/predict", json=sample_record).status_code == 422


def test_batch_predict(client, sample_record):
    resp = client.post("/predict/batch", json={"records": [sample_record] * 3})
    assert resp.status_code == 200
    assert len(resp.json()["predictions"]) == 3


def test_metrics_endpoint_counts_predictions(client, sample_record):
    client.post("/predict", json=sample_record)
    text = client.get("/metrics").text
    assert "api_requests_total" in text
    assert "model_predictions_total" in text
    assert "model_risk_band_total" in text
    assert 'endpoint="/predict"' in text


def test_service_unavailable_without_model(tmp_path, sample_record):
    client = TestClient(create_app(tmp_path / "missing.joblib"))
    assert client.get("/health").json()["model_loaded"] is False
    assert client.get("/ready").status_code == 503
    assert client.post("/predict", json=sample_record).status_code == 503

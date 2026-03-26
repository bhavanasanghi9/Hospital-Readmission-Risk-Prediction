from fastapi.testclient import TestClient
from app.main import app


def test_health_check():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True
        assert data["version"] == "1.0.0"


def test_predict_low_risk(sample_patient):
    with TestClient(app) as client:
        response = client.post("/predict", json=sample_patient)
        assert response.status_code == 200

        data = response.json()
        assert "encounter_id" in data
        assert "risk_score" in data
        assert "risk_tier" in data
        assert data["encounter_id"] == sample_patient["encounter_id"]
        assert 0.0 <= data["risk_score"] <= 1.0
        assert data["risk_tier"] in ["Low", "Medium", "High"]


def test_predict_high_risk(high_risk_patient):
    with TestClient(app) as client:
        response = client.post("/predict", json=high_risk_patient)
        assert response.status_code == 200

        data = response.json()
        assert "encounter_id" in data
        assert "risk_score" in data
        assert "risk_tier" in data
        assert data["encounter_id"] == high_risk_patient["encounter_id"]
        assert 0.0 <= data["risk_score"] <= 1.0
        assert data["risk_tier"] == "High"


def test_predict_invalid_medication(sample_patient):
    with TestClient(app) as client:
        sample_patient["insulin"] = "banana"
        response = client.post("/predict", json=sample_patient)
        assert response.status_code == 422


def test_predict_invalid_gender(sample_patient):
    with TestClient(app) as client:
        sample_patient["gender"] = "Unknown"
        response = client.post("/predict", json=sample_patient)
        assert response.status_code == 422


def test_predict_missing_required_field(sample_patient):
    with TestClient(app) as client:
        del sample_patient["encounter_id"]
        response = client.post("/predict", json=sample_patient)
        assert response.status_code == 422


def test_predict_response_structure(sample_patient):
    with TestClient(app) as client:
        response = client.post("/predict", json=sample_patient)
        assert response.status_code == 200

        data = response.json()
        assert set(data.keys()) == {"encounter_id", "risk_score", "risk_tier"}
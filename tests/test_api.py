from fastapi.testclient import TestClient

import api.main as main  # <-- IMPORTANT
import api.services.predict_services as services
from api.main import app

client = TestClient(app)


def test_predict_endpoint(monkeypatch):
    # Mock MLflow model loader
    class DummyModel:
        def predict(self, df):
            return [1]

        def __getattr__(self, item):
            return None

    monkeypatch.setattr(services.mlflow.pyfunc, "load_model", lambda _: DummyModel())

    # Mock predict_churn used by FastAPI
    def mock_predict_churn(_):
        return {"prediction": 1, "probability": 0.82}

    # Patch the function inside api.main, NOT predict_services
    monkeypatch.setattr(main, "predict_churn", mock_predict_churn)

    sample = {"tenure": 12, "MonthlyCharges": 75.5, "Contract": 0}
    response = client.post("/predict", json=sample)

    assert response.status_code == 200
    result = response.json()

    assert result["prediction"] == 1
    assert result["probability"] == 0.82

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_predict_endpoint():
    sample = {"tenure": 12, "MonthlyCharges": 75.5, "Contract": 0}
    response = client.post("/predict", json=sample)
    assert response.status_code == 200
    assert "churn_prediction" in response.json()

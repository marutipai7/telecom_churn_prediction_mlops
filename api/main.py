# api/main.py
from fastapi import FastAPI

from api.models.predict_schema import PredictRequest
from api.services.predict_services import predict_churn

app = FastAPI(title="Churn Prediction API")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    return predict_churn(req.model_dump())

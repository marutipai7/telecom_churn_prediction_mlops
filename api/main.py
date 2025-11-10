from fastapi import FastAPI
import mlflow.sklearn
import pandas as pd
from api.models.predict_schema import PredictRequest
from api.services.predict_services import predict_churn

app = FastAPI(title="Churn Prediction API")

model = mlflow.sklearn.load_model("mlruns/0/<RUN_ID>/artifacts/model")

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    predictions = model.predict(df)[0]
    return {"churn_predictions": bool(predictions)}
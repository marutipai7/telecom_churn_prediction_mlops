import os

import mlflow.pyfunc
import pandas as pd

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "file:///E:/Code Space/churn_prediction_mlops/mlflow_tracking/mlruns",
)
MODEL_URI = os.getenv("MODEL_URI", "models:/telecom_churn/Production")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

model = None  # lazy-loaded model


def get_model():
    global model
    if model is None:
        model = mlflow.pyfunc.load_model(MODEL_URI)
    return model


def predict_churn(payload: dict):
    model = get_model()  # <--- load only when needed
    df = pd.DataFrame([payload])
    pred_proba = model.predict(df)

    if hasattr(model._model_impl, "predict_proba"):
        prob = float(model._model_impl.predict_proba(df)[:, 1][0])
        label = int(prob >= 0.5)
    else:
        label = int(pred_proba[0])
        prob = float(label)
    return {"prediction": label, "probability": prob}

# api/services/predict_services.py
import os
import pandas as pd
import mlflow.pyfunc

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:///E:/Code Space/churn_prediction_mlops/mlflow_tracking/mlruns")
MODEL_URI = os.getenv("MODEL_URI", "models:/telecom_churn/Production")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
model = mlflow.pyfunc.load_model(MODEL_URI)

def predict_churn(payload: dict):
    df = pd.DataFrame([payload])
    # Ensure payload matches training features (preprocess upstream)
    pred_proba = model.predict(df)
    # For tree models logged via sklearn flavor, predict returns labels; via pyfunc,
    # you may have a custom predict. If it's probability you can adapt accordingly:
    if hasattr(model._model_impl, "predict_proba"):
        prob = float(model._model_impl.predict_proba(df)[:,1][0])
        label = int(prob >= 0.5)
    else:
        label = int(pred_proba[0])
        prob = float(label)
    return {"churn": label, "probability": prob}

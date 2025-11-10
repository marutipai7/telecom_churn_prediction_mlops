import mlflow.sklearn
import pandas as pd

model = mlflow.sklearn.load_model("mlruns/0/latest/model")

def predict_churn(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    return bool(prediction)

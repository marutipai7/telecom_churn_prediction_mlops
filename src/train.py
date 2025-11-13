# src/train.py
import os

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# --- CONFIG ---
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "file:///E:/Code Space/churn_prediction_mlops/mlflow_tracking/mlruns",
)
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "telecom_churn_prediction")
REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "telecom_churn")

DATA_PATH = os.getenv(
    "DATA_PATH",
    r"E:\Code Space\churn_prediction_mlops\data\processed\telecom_churn_cleaned1.csv",
)
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["churn"])
    y = df["churn"]
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )


def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(X_test)[:, 1]
    else:
        probas = preds
    return {
        "accuracy": accuracy_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, probas),
    }


def train_one(model, model_name, X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name=model_name) as run:
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)
        # log params/metrics and model artifact
        mlflow.log_param("model_name", model_name)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        mlflow.sklearn.log_model(
            model, artifact_path="model", registered_model_name=REGISTERED_MODEL_NAME
        )
        run_id = run.info.run_id
        return run_id, metrics


def main(promote_best=True, metric_key="roc_auc"):
    X_train, X_test, y_train, y_test = load_data()

    models = [
        (
            "LightGBM",
            LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=7,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
        (
            "CatBoost",
            CatBoostClassifier(
                iterations=300,
                depth=8,
                learning_rate=0.05,
                eval_metric="AUC",
                verbose=False,
                random_seed=RANDOM_STATE,
            ),
        ),
        (
            "XGBoost",
            XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=7,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=RANDOM_STATE,
            ),
        ),
    ]

    results = []
    for name, mdl in models:
        run_id, metrics = train_one(mdl, name, X_train, X_test, y_train, y_test)
        results.append((name, run_id, metrics))

    # pick best by metric_key
    results.sort(key=lambda x: x[2][metric_key], reverse=True)
    best_name, best_run_id, best_metrics = results[0]
    print(
        f"Best: {best_name} | {metric_key}={best_metrics[metric_key]:.4f} | run_id={best_run_id}"
    )

    if promote_best:
        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        # get latest model version created by this run
        mv = next(
            mv
            for mv in client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
            if mv.run_id == best_run_id
        )
        # transition to Staging then Production (optional workflow)
        client.transition_model_version_stage(
            name=REGISTERED_MODEL_NAME,
            version=mv.version,
            stage="Staging",
            archive_existing_versions=False,
        )
        client.transition_model_version_stage(
            name=REGISTERED_MODEL_NAME,
            version=mv.version,
            stage="Production",
            archive_existing_versions=True,
        )
        print(f"Promoted model version {mv.version} to Production.")


if __name__ == "__main__":
    main()

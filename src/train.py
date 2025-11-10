import mlflow
import mlflow.sklearn
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.data_prep import load_data, preprocess_data, split_data

mlflow.set_experiment("Churn_Prediction_Experiment")

def train_model(data_path):
    df = load_data(data_path)
    df, label_encoders = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df)

    model = LGBMClassifier(n_estimators=200, learning_rate=0.05, random_state=42)

    with mlflow.start_run():
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("learning_rate", 0.05)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(model, "lgbm_churn_model")

        print(f"✅ Accuracy: {accuracy:.3f} | F1: {f1:.3f}")

if __name__ == "__main__":
    train_model("data/raw/data2.csv")
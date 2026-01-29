import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

import mlflow
import mlflow.sklearn

from config import MLFLOW_TRACKING_URI, MODEL_NAME, MODEL_DIR, MODEL_PATH
from data_gen import make_reference

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("ml-monitoring-drift-detection")

    df = make_reference()
    X = df[["x1", "x2"]]
    y = df["y"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )

    with mlflow.start_run(run_name="train-rf-regressor"):
        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("n_estimators", 200)

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = mean_squared_error(y_test, preds, squared=False)

        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)

        mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name=MODEL_NAME)

        joblib.dump(model, MODEL_PATH)

        print("✅ Model trained + logged to MLflow")
        print(f"✅ Saved local model -> {MODEL_PATH}")
        print(f"MAE={mae:.4f}, RMSE={rmse:.4f}")

if __name__ == "__main__":
    main()

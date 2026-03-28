import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from utils import FEATURE_COLS

PROCESSED_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "games_preprocessed.csv")
MODEL_PATH     = os.path.join(os.path.dirname(__file__), "..", "models", "points_rf.joblib")


def train(processed_path: str = PROCESSED_PATH, model_path: str = MODEL_PATH):
    df = pd.read_csv(processed_path)

    X = df[FEATURE_COLS]
    y = df["target_pts"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print(f"Test MAE: {mae:.2f} pts  |  RMSE: {rmse:.2f} pts")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved -> {model_path}")
    return model


if __name__ == "__main__":
    train()

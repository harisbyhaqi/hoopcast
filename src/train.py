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


CURRENT_SEASON_WEIGHT = 8.0   # current season games count 8x more than last season


def train(processed_path: str = PROCESSED_PATH, model_path: str = MODEL_PATH):
    df = pd.read_csv(processed_path)

    # Weight the most recent regular season more heavily.
    # Filter to regular season only (SEASON_ID like 22025) before finding max
    # so preseason (1xxxx) and playoff (4xxxx) IDs don't skew the result.
    regular = df[df["SEASON_ID"].astype(str).str.match(r"^2\d{4}$")]
    latest_season = regular["SEASON_ID"].astype(int).max()
    current_mask = df["SEASON_ID"].astype(int) == latest_season
    weights = np.where(current_mask, CURRENT_SEASON_WEIGHT, 1.0)

    X = df[FEATURE_COLS]
    y = df["target_pts"]

    X_train, X_test, y_train, y_test, w_train, _ = train_test_split(
        X, y, weights, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train, sample_weight=w_train)

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

import json
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from utils import FEATURE_COLS

PROCESSED_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "games_preprocessed.csv")
MODEL_PATH     = os.path.join(os.path.dirname(__file__), "..", "models", "points_rf.joblib")
METRICS_PATH   = os.path.join(os.path.dirname(__file__), "..", "models", "metrics.json")


CURRENT_SEASON_WEIGHT = 5.0   # current season games count 5x more than last season
HOLDOUT_FRACTION      = 0.15  # most recent games held out for evaluation


def new_model() -> RandomForestRegressor:
    return RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=5,
                                 random_state=42, n_jobs=-1)


def season_weights(df: pd.DataFrame) -> np.ndarray:
    # Weight the most recent season (regular + playoffs) more heavily.
    season = df["SEASON_ID"].astype(int) % 10000
    return np.where(season == season.max(), CURRENT_SEASON_WEIGHT, 1.0)


def time_split(df: pd.DataFrame):
    """Split by date so the hold-out set is strictly after the training set."""
    df = df.sort_values("GAME_DATE")
    cutoff = df["GAME_DATE"].iloc[int(len(df) * (1 - HOLDOUT_FRACTION))]
    return df[df["GAME_DATE"] < cutoff], df[df["GAME_DATE"] >= cutoff]


def score(model, test_df: pd.DataFrame) -> dict:
    """MAE/RMSE per team score, win-pick accuracy, and the std of the margin error (for win probabilities)."""
    test_df = test_df.assign(pred_pts=model.predict(test_df[FEATURE_COLS]))
    y, p = test_df["target_pts"], test_df["pred_pts"]

    paired = test_df.groupby("GAME_ID").filter(lambda g: len(g) == 2).sort_values(["GAME_ID", "home"])
    away, home = paired.iloc[0::2], paired.iloc[1::2]
    actual_margin = home["target_pts"].values - away["target_pts"].values
    pred_margin   = home["pred_pts"].values - away["pred_pts"].values

    return {
        "mae": round(float(mean_absolute_error(y, p)), 2),
        "rmse": round(float(np.sqrt(mean_squared_error(y, p))), 2),
        "win_accuracy": round(float(np.mean(np.sign(actual_margin) == np.sign(pred_margin))) * 100, 1),
        "margin_sigma": round(float(np.std(actual_margin - pred_margin)), 2),
        "test_games": int(len(home)),
        "test_from": str(pd.Timestamp(test_df["GAME_DATE"].min()).date()),
        "test_to": str(pd.Timestamp(test_df["GAME_DATE"].max()).date()),
    }


def holdout_metrics(df: pd.DataFrame) -> dict:
    train_df, test_df = time_split(df)
    model = new_model()
    model.fit(train_df[FEATURE_COLS], train_df["target_pts"], sample_weight=season_weights(train_df))
    return score(model, test_df)


def train(processed_path: str = PROCESSED_PATH, model_path: str = MODEL_PATH, metrics_path: str = METRICS_PATH):
    df = pd.read_csv(processed_path, parse_dates=["GAME_DATE"], dtype={"GAME_ID": str})

    metrics = holdout_metrics(df)
    print(f"Hold-out ({metrics['test_from']} → {metrics['test_to']}, {metrics['test_games']} games): "
          f"MAE {metrics['mae']} | RMSE {metrics['rmse']} | win acc {metrics['win_accuracy']}% | "
          f"margin σ {metrics['margin_sigma']}")

    # Final model uses every game, including the most recent ones.
    model = new_model()
    model.fit(df[FEATURE_COLS], df["target_pts"], sample_weight=season_weights(df))

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Model saved -> {model_path}")
    return model


if __name__ == "__main__":
    train()

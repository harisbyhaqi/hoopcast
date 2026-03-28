import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from utils import FEATURE_COLS

PROCESSED_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "games_preprocessed.csv")
MODEL_PATH     = os.path.join(os.path.dirname(__file__), "..", "models", "points_rf.joblib")


def evaluate(processed_path: str = PROCESSED_PATH, model_path: str = MODEL_PATH):
    df = pd.read_csv(processed_path, parse_dates=["GAME_DATE"])
    model = joblib.load(model_path)

    # Use last 20% of data (by date) as hold-out set
    df = df.sort_values("GAME_DATE")
    split = int(len(df) * 0.8)
    test_df = df.iloc[split:]

    X_test = test_df[FEATURE_COLS]
    y_test = test_df["target_pts"]
    preds  = model.predict(X_test)

    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    # Win prediction accuracy: pair games by GAME_ID and compare predicted winner vs actual
    test_df = test_df.copy()
    test_df["pred_pts"] = preds

    paired = test_df.groupby("GAME_ID").filter(lambda g: len(g) == 2)
    correct = 0
    total   = 0
    for _, grp in paired.groupby("GAME_ID"):
        if len(grp) != 2:
            continue
        a, b = grp.iloc[0], grp.iloc[1]
        pred_winner  = a["TEAM_ABBREVIATION"] if a["pred_pts"]   > b["pred_pts"]   else b["TEAM_ABBREVIATION"]
        actual_winner= a["TEAM_ABBREVIATION"] if a["target_pts"] > b["target_pts"] else b["TEAM_ABBREVIATION"]
        correct += int(pred_winner == actual_winner)
        total   += 1

    win_acc = correct / total if total > 0 else 0.0

    print(f"Hold-out set ({len(test_df)} rows from {test_df['GAME_DATE'].min().date()} to {test_df['GAME_DATE'].max().date()})")
    print(f"  MAE            : {mae:.2f} pts")
    print(f"  RMSE           : {rmse:.2f} pts")
    print(f"  Win accuracy   : {win_acc*100:.1f}%  ({correct}/{total} games)")
    return {"mae": mae, "rmse": rmse, "win_accuracy": win_acc}


if __name__ == "__main__":
    evaluate()

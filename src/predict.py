import os
import joblib
import numpy as np
import pandas as pd
from utils import FEATURE_COLS

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "points_rf.joblib")


def load_model(model_path: str = MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Run train.py first.")
    return joblib.load(model_path)


def predict_score(features: dict, model=None) -> float:
    """Predict points for a single team given feature dict."""
    if model is None:
        model = load_model()
    row = pd.DataFrame([{col: features.get(col, 0) for col in FEATURE_COLS}])
    return float(model.predict(row)[0])


def predict_game(team_a_feat: dict, team_b_feat: dict, model=None):
    """Return (score_a, score_b) predictions."""
    if model is None:
        model = load_model()
    score_a = predict_score(team_a_feat, model)
    score_b = predict_score(team_b_feat, model)
    return round(score_a, 1), round(score_b, 1)


def team_features_from_history(team_abbr: str, processed_path: str) -> dict:
    """Extract the most recent rolling features for a team from processed CSV."""
    df = pd.read_csv(processed_path, parse_dates=["GAME_DATE"])
    team_df = df[df["TEAM_ABBREVIATION"] == team_abbr].sort_values("GAME_DATE")
    if team_df.empty:
        raise ValueError(f"Team '{team_abbr}' not found in processed data.")
    last = team_df.iloc[-1]
    return {col: last[col] for col in FEATURE_COLS}


if __name__ == "__main__":
    import sys
    processed = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "games_preprocessed.csv")

    team_a = sys.argv[1] if len(sys.argv) > 1 else "LAL"
    team_b = sys.argv[2] if len(sys.argv) > 2 else "GSW"

    feat_a = team_features_from_history(team_a, processed)
    feat_b = team_features_from_history(team_b, processed)

    # Set home flags for the matchup
    feat_a["home"] = 1
    feat_b["home"] = 0

    score_a, score_b = predict_game(feat_a, feat_b)
    winner = team_a if score_a > score_b else team_b
    print(f"Predicted: {team_a} {score_a} - {team_b} {score_b}  →  Winner: {winner}")

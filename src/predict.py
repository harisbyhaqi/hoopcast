import os
import joblib
import pandas as pd
from utils import FEATURE_COLS, DEFENSE_COLS, team_states

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "points_rf.joblib")


def load_model(model_path: str = MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Run train.py first.")
    return joblib.load(model_path)


def matchup_row(team: dict, opp: dict) -> dict:
    """Model input for `team` facing `opp` (team states as returned by utils.team_states)."""
    row = {col: team.get(col, 0) for col in FEATURE_COLS}
    for c in DEFENSE_COLS:
        row[f"opp_{c}"] = opp[c]
    return row


def predict_games(pairs: list[tuple[dict, dict]], model=None) -> list[tuple[float, float]]:
    """Batch-predict [(score_a, score_b)] for a list of (team_a_state, team_b_state) pairs."""
    if not pairs:
        return []
    if model is None:
        model = load_model()
    rows = []
    for a, b in pairs:
        rows += [matchup_row(a, b), matchup_row(b, a)]
    preds = model.predict(pd.DataFrame(rows, columns=FEATURE_COLS))
    return [(round(float(preds[i]), 1), round(float(preds[i + 1]), 1)) for i in range(0, len(preds), 2)]


def predict_game(team_a_feat: dict, team_b_feat: dict, model=None):
    """Return (score_a, score_b) predictions."""
    return predict_games([(team_a_feat, team_b_feat)], model)[0]


def team_features_from_history(team_abbr: str, processed_path: str) -> dict:
    """A team's current state (after its most recent game) from the processed CSV."""
    df = pd.read_csv(processed_path, parse_dates=["GAME_DATE"], dtype={"GAME_ID": str})
    if team_abbr not in set(df["TEAM_ABBREVIATION"]):
        raise ValueError(f"Team '{team_abbr}' not found in processed data.")
    latest_sid = int(df.sort_values("GAME_DATE")["SEASON_ID"].iloc[-1])
    states = team_states(df, pd.Timestamp.today().normalize(), latest_sid)
    return states.loc[team_abbr].to_dict()


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

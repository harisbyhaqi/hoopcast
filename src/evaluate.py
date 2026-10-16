import os
import pandas as pd
from train import holdout_metrics

PROCESSED_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "games_preprocessed.csv")


def evaluate(processed_path: str = PROCESSED_PATH) -> dict:
    """Train on everything before the hold-out window, score on the window (no overlap)."""
    df = pd.read_csv(processed_path, parse_dates=["GAME_DATE"], dtype={"GAME_ID": str})
    m = holdout_metrics(df)
    print(f"Hold-out set ({m['test_games']} games from {m['test_from']} to {m['test_to']})")
    print(f"  MAE            : {m['mae']:.2f} pts")
    print(f"  RMSE           : {m['rmse']:.2f} pts")
    print(f"  Win accuracy   : {m['win_accuracy']:.1f}%")
    print(f"  Margin σ       : {m['margin_sigma']:.2f} pts")
    return m


if __name__ == "__main__":
    evaluate()

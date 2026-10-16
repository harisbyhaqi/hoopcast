import os
import pandas as pd
from utils import load_csv, rolling_team_stats, home_flag, nba_games_only, FEATURE_COLS

RAW_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "games_raw.csv")
OUT_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "games_preprocessed.csv")


def clean_games(raw_path: str = RAW_PATH) -> pd.DataFrame:
    """Finished NBA regular-season/playoff team-game rows with a home flag."""
    df = load_csv(raw_path)

    # Drop rows missing key columns or with corrupted data (e.g. partial live game snapshots)
    df = df.dropna(subset=["PTS", "FG_PCT", "FG3_PCT", "REB", "AST", "TOV", "MATCHUP", "WL"])
    df = df[df["PTS"] >= 60]  # a team scoring <60 in a finished game is impossible
    df = nba_games_only(df)

    df["home"] = df["MATCHUP"].apply(home_flag)
    return df


def preprocess(raw_path: str = RAW_PATH, out_path: str = OUT_PATH) -> pd.DataFrame:
    df = rolling_team_stats(clean_games(raw_path))

    # Drop rows where features are still NaN (first game ever per team, unpaired games)
    df = df.dropna(subset=FEATURE_COLS)

    # Target
    df["target_pts"] = df["PTS"]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Preprocessed {len(df)} rows -> {out_path}")
    return df


if __name__ == "__main__":
    preprocess()

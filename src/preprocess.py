import os
import pandas as pd
from utils import load_csv, rolling_team_stats, home_flag, FEATURE_COLS

RAW_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "games_raw.csv")
OUT_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "games_preprocessed.csv")


def preprocess(raw_path: str = RAW_PATH, out_path: str = OUT_PATH) -> pd.DataFrame:
    df = load_csv(raw_path)

    # Drop rows missing key columns
    df = df.dropna(subset=["PTS", "FG_PCT", "FG3_PCT", "REB", "AST", "TOV", "MATCHUP"])

    # Home flag
    df["home"] = df["MATCHUP"].apply(home_flag)

    # Rolling features
    df = rolling_team_stats(df)

    # Drop rows where rolling features are still NaN (first game ever per team)
    df = df.dropna(subset=FEATURE_COLS)

    # Target
    df["target_pts"] = df["PTS"]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Preprocessed {len(df)} rows -> {out_path}")
    return df


if __name__ == "__main__":
    preprocess()

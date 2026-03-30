import pandas as pd
import numpy as np


FEATURE_COLS = [
    "pts_roll5", "fgpct_roll5", "fg3pct_roll5",
    "reb_roll5", "ast_roll5", "tov_roll5",
    "rest_days", "home",
    "win_pct_last10",   # recent momentum (soft signal)
    "pts_diff_roll5",   # rolling point differential (net rating proxy)
]


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["GAME_DATE"])


def rolling_team_stats(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Compute per-team rolling averages sorted by date."""
    df = df.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"]).copy()

    for team, grp in df.groupby("TEAM_ABBREVIATION"):
        idx = grp.index
        df.loc[idx, "pts_roll5"]   = grp["PTS"].shift(1).rolling(window, min_periods=1).mean()
        df.loc[idx, "fgpct_roll5"] = grp["FG_PCT"].shift(1).rolling(window, min_periods=1).mean()
        df.loc[idx, "fg3pct_roll5"]= grp["FG3_PCT"].shift(1).rolling(window, min_periods=1).mean()
        df.loc[idx, "reb_roll5"]   = grp["REB"].shift(1).rolling(window, min_periods=1).mean()
        df.loc[idx, "ast_roll5"]   = grp["AST"].shift(1).rolling(window, min_periods=1).mean()
        df.loc[idx, "tov_roll5"]   = grp["TOV"].shift(1).rolling(window, min_periods=1).mean()
        df.loc[idx, "rest_days"]    = grp["GAME_DATE"].diff().dt.days.fillna(3).clip(1, 10)
        df.loc[idx, "win_pct_last10"] = (
            (grp["WL"] == "W").astype(float).shift(1).rolling(10, min_periods=1).mean()
        )
        df.loc[idx, "pts_diff_roll5"] = (
            grp["PLUS_MINUS"].shift(1).rolling(window, min_periods=1).mean()
        )

    return df


def home_flag(matchup: str) -> int:
    """1 if home game (contains 'vs.'), 0 if away (contains '@')."""
    return 1 if "vs." in matchup else 0

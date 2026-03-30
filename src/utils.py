import pandas as pd
import numpy as np


FEATURE_COLS = [
    "pts_roll5",        # EWM scoring avg (recent games weighted more)
    "fgpct_roll5",      # EWM FG%
    "fg3pct_roll5",     # EWM 3PT%
    "reb_roll5",        # EWM rebounds
    "ast_roll5",        # EWM assists
    "tov_roll5",        # EWM turnovers
    "efg_roll5",        # effective FG% = (FGM + 0.5*FG3M) / FGA
    "tov_rate_roll5",   # turnover rate = TOV / (FGA + 0.44*FTA + TOV)
    "rest_days",        # days since last game
    "home",             # 1 = home, 0 = away
    "home_win_pct",     # season win % in home games (prior games only)
    "away_win_pct",     # season win % in away games (prior games only)
    "win_pct_last10",   # win % over last 10 games (momentum)
]


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["GAME_DATE"])


def rolling_team_stats(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Compute per-team rolling/EWM stats sorted by date.
    Uses exponentially weighted means (span=window) so the most recent
    games carry more weight than games from 4-5 contests ago.
    """
    df = df.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"]).copy()

    for team, grp in df.groupby("TEAM_ABBREVIATION"):
        idx = grp.index
        wins   = (grp["WL"] == "W").astype(float)
        is_home = grp["home"].astype(float)

        fgm = grp["FGM"]
        fga = grp["FGA"].replace(0, np.nan)
        fg3m = grp["FG3M"]
        fta  = grp["FTA"]
        tov  = grp["TOV"]

        # Advanced stats per game
        efg      = (fgm + 0.5 * fg3m) / fga
        tov_denom = (fga + 0.44 * fta + tov).replace(0, np.nan)
        tov_rate  = tov / tov_denom

        # EWM rolling (shift first to avoid leakage — each row uses only prior games)
        ewm_kw = dict(span=window, min_periods=1)
        df.loc[idx, "pts_roll5"]      = grp["PTS"].shift(1).ewm(**ewm_kw).mean()
        df.loc[idx, "fgpct_roll5"]    = grp["FG_PCT"].shift(1).ewm(**ewm_kw).mean()
        df.loc[idx, "fg3pct_roll5"]   = grp["FG3_PCT"].shift(1).ewm(**ewm_kw).mean()
        df.loc[idx, "reb_roll5"]      = grp["REB"].shift(1).ewm(**ewm_kw).mean()
        df.loc[idx, "ast_roll5"]      = grp["AST"].shift(1).ewm(**ewm_kw).mean()
        df.loc[idx, "tov_roll5"]      = grp["TOV"].shift(1).ewm(**ewm_kw).mean()
        df.loc[idx, "efg_roll5"]      = efg.shift(1).ewm(**ewm_kw).mean()
        df.loc[idx, "tov_rate_roll5"] = tov_rate.shift(1).ewm(**ewm_kw).mean()

        df.loc[idx, "rest_days"]      = grp["GAME_DATE"].diff().dt.days.fillna(3).clip(1, 10)
        df.loc[idx, "win_pct_last10"] = wins.shift(1).rolling(10, min_periods=1).mean()

        # Home win % — cumulative over season using only prior home games
        home_wins_cum  = wins.where(is_home == 1).fillna(0).cumsum().shift(1)
        home_games_cum = is_home.cumsum().shift(1).replace(0, np.nan)
        df.loc[idx, "home_win_pct"]   = (home_wins_cum / home_games_cum).fillna(0.5)

        # Away win % — cumulative over season using only prior away games
        away_wins_cum  = wins.where(is_home == 0).fillna(0).cumsum().shift(1)
        away_games_cum = (1 - is_home).cumsum().shift(1).replace(0, np.nan)
        df.loc[idx, "away_win_pct"]   = (away_wins_cum / away_games_cum).fillna(0.5)

    return df


def home_flag(matchup: str) -> int:
    """1 if home game (contains 'vs.'), 0 if away (contains '@')."""
    return 1 if "vs." in matchup else 0

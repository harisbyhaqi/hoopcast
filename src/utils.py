import numpy as np
import pandas as pd

from espn import TEAM_NAMES


FEATURE_COLS = [
    "pts_roll5",              # EWM scoring avg (recent games weighted more)
    "fgpct_roll5",            # EWM FG%
    "fg3pct_roll5",           # EWM 3PT%
    "reb_roll5",              # EWM rebounds
    "ast_roll5",              # EWM assists
    "tov_roll5",              # EWM turnovers
    "efg_roll5",              # effective FG% = (FGM + 0.5*FG3M) / FGA
    "tov_rate_roll5",         # turnover rate = TOV / (FGA + 0.44*FTA + TOV)
    "rest_days",              # days since last game
    "home",                   # 1 = home, 0 = away
    "home_win_pct",           # this season's home win %, shrunk toward .500
    "away_win_pct",           # this season's away win %, shrunk toward .500
    "win_pct_last10",         # win % over last 10 games (momentum)
    "opp_pts_allowed_roll5",  # opponent's EWM points allowed (their defense)
    "opp_efg_allowed_roll5",  # opponent's EWM eFG% allowed
]

# Per-team state carried into the next game; opp_* features are filled from the opponent's state.
DEFENSE_COLS = ["pts_allowed_roll5", "efg_allowed_roll5"]

WIN_PCT_PRIOR_GAMES = 4   # shrink early-season home/away win % toward .500


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["GAME_DATE"], dtype={"GAME_ID": str})


def home_flag(matchup: str) -> int:
    """1 if home game (contains 'vs.'), 0 if away (contains '@')."""
    return 1 if "vs." in matchup else 0


def season_of(season_id) -> int:
    """22025 (regular) and 42025 (playoffs) both belong to season 2025."""
    return int(season_id) % 10000


def nba_games_only(df: pd.DataFrame) -> pd.DataFrame:
    """Keep regular season + playoffs between the 30 NBA teams (drops preseason, All-Star, intl. exhibitions)."""
    sid = df["SEASON_ID"].astype(str)
    opp = df["MATCHUP"].str.split().str[-1]
    return df[sid.str.match(r"^[24]\d{4}$") & df["TEAM_ABBREVIATION"].isin(TEAM_NAMES) & opp.isin(TEAM_NAMES)]


def rolling_team_stats(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Compute per-team pre-game features. Every value on a row uses only that team's
    prior games (shift before rolling), so rows are leakage-free training examples.
    EWM stats carry across seasons (end-of-last-season form is the best prior on opening night);
    home/away win % reset each season.
    """
    df = df.copy()
    df["_season"] = df["SEASON_ID"].map(season_of)
    df["_opp"] = df["MATCHUP"].fillna("").str.split().str[-1]
    df["_efg"] = (df["FGM"] + 0.5 * df["FG3M"]) / df["FGA"].replace(0, np.nan)
    df["_tov_rate"] = df["TOV"] / (df["FGA"] + 0.44 * df["FTA"] + df["TOV"]).replace(0, np.nan)

    # What the opponent scored/shot against this team in this game
    opp_game = df[["GAME_ID", "TEAM_ABBREVIATION", "PTS", "_efg"]].rename(
        columns={"TEAM_ABBREVIATION": "_opp", "PTS": "_opp_pts", "_efg": "_opp_efg"})
    df = df.merge(opp_game, on=["GAME_ID", "_opp"], how="left")

    df = df.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"]).reset_index(drop=True)
    ewm = lambda s: s.shift(1).ewm(span=window, min_periods=1).mean()

    for _, grp in df.groupby("TEAM_ABBREVIATION"):
        idx = grp.index
        df.loc[idx, "pts_roll5"]         = ewm(grp["PTS"])
        df.loc[idx, "fgpct_roll5"]       = ewm(grp["FG_PCT"])
        df.loc[idx, "fg3pct_roll5"]      = ewm(grp["FG3_PCT"])
        df.loc[idx, "reb_roll5"]         = ewm(grp["REB"])
        df.loc[idx, "ast_roll5"]         = ewm(grp["AST"])
        df.loc[idx, "tov_roll5"]         = ewm(grp["TOV"])
        df.loc[idx, "efg_roll5"]         = ewm(grp["_efg"])
        df.loc[idx, "tov_rate_roll5"]    = ewm(grp["_tov_rate"])
        df.loc[idx, "pts_allowed_roll5"] = ewm(grp["_opp_pts"])
        df.loc[idx, "efg_allowed_roll5"] = ewm(grp["_opp_efg"])

        wins = (grp["WL"] == "W").astype(float)
        df.loc[idx, "rest_days"]      = grp["GAME_DATE"].diff().dt.days.fillna(3).clip(1, 10)
        df.loc[idx, "win_pct_last10"] = wins.shift(1).rolling(10, min_periods=1).mean().fillna(0.5)

    # Home / away win % within each season, using prior games only
    k = WIN_PCT_PRIOR_GAMES
    for _, grp in df.groupby(["TEAM_ABBREVIATION", "_season"]):
        idx = grp.index
        wins = (grp["WL"] == "W").astype(float)
        is_home = grp["home"].astype(float)
        for col, mask in (("home_win_pct", is_home), ("away_win_pct", 1 - is_home)):
            w = (wins * mask).cumsum().shift(1).fillna(0)
            n = mask.cumsum().shift(1).fillna(0)
            df.loc[idx, col] = (w + 0.5 * k) / (n + k)

    # Opponent's pre-game defensive form for the same game
    opp_def = df[["GAME_ID", "TEAM_ABBREVIATION"] + DEFENSE_COLS].rename(
        columns={"TEAM_ABBREVIATION": "_opp", **{c: f"opp_{c}" for c in DEFENSE_COLS}})
    df = df.merge(opp_def, on=["GAME_ID", "_opp"], how="left")

    return df.drop(columns=["_season", "_opp", "_efg", "_tov_rate", "_opp_pts", "_opp_efg"])


def team_states(games: pd.DataFrame, as_of, season_id: int) -> pd.DataFrame:
    """
    Each team's features *after* its most recent game, i.e. what its next game would see.
    Works by appending one placeholder game per team dated `as_of` and reading its pre-game row.
    `season_id` is the season the placeholder belongs to (resets home/away win % on a new season).
    Returns a DataFrame indexed by team abbreviation.
    """
    as_of = pd.Timestamp(as_of)
    teams = sorted(games["TEAM_ABBREVIATION"].unique())
    placeholders = pd.DataFrame({
        "TEAM_ABBREVIATION": teams,
        "GAME_ID": [f"next_{t}" for t in teams],
        "GAME_DATE": as_of,
        "SEASON_ID": season_id,
        "MATCHUP": "",
        "home": 0,
    })
    out = rolling_team_stats(pd.concat([games, placeholders], ignore_index=True))
    out = out[out["GAME_ID"].str.startswith("next_")].set_index("TEAM_ABBREVIATION")
    return out[[c for c in FEATURE_COLS if not c.startswith("opp_")] + DEFENSE_COLS]

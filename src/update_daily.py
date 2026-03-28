"""
Daily update script — run by GitHub Actions.
1. Fetches recent games (last 60 days) to build current team rolling stats
2. Gets today's scheduled games from ScoreboardV2
3. Runs the pre-trained model + Monte Carlo for each matchup
4. Writes docs/predictions.json
"""
import os, sys, json, time
from datetime import date, timedelta

import numpy as np
import pandas as pd
import joblib

# ── Patch nba_api to use browser-like headers before any imports ──────────
# stats.nba.com blocks requests that don't look like they come from a browser.
from nba_api.stats.library import http as _nba_http

_nba_http.NBAStatsHTTP.HEADERS = {
    "Host": "stats.nba.com",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "Connection": "keep-alive",
}
_nba_http.NBAStatsHTTP.timeout = 120
# ─────────────────────────────────────────────────────────────────────────

sys.path.insert(0, os.path.dirname(__file__))
from utils import rolling_team_stats, home_flag, FEATURE_COLS
from simulate import monte_carlo

ROOT        = os.path.join(os.path.dirname(__file__), "..")
MODEL_PATH  = os.path.join(ROOT, "models", "points_rf.joblib")
OUT_PATH    = os.path.join(ROOT, "docs", "predictions.json")

TEAM_NAMES = {
    "ATL":"Atlanta Hawks","BOS":"Boston Celtics","BKN":"Brooklyn Nets",
    "CHA":"Charlotte Hornets","CHI":"Chicago Bulls","CLE":"Cleveland Cavaliers",
    "DAL":"Dallas Mavericks","DEN":"Denver Nuggets","DET":"Detroit Pistons",
    "GSW":"Golden State Warriors","HOU":"Houston Rockets","IND":"Indiana Pacers",
    "LAC":"LA Clippers","LAL":"Los Angeles Lakers","MEM":"Memphis Grizzlies",
    "MIA":"Miami Heat","MIL":"Milwaukee Bucks","MIN":"Minnesota Timberwolves",
    "NOP":"New Orleans Pelicans","NYK":"New York Knicks","OKC":"Oklahoma City Thunder",
    "ORL":"Orlando Magic","PHI":"Philadelphia 76ers","PHX":"Phoenix Suns",
    "POR":"Portland Trail Blazers","SAC":"Sacramento Kings","SAS":"San Antonio Spurs",
    "TOR":"Toronto Raptors","UTA":"Utah Jazz","WAS":"Washington Wizards",
}

TEAM_IDS = {
    "ATL":1610612737,"BOS":1610612738,"BKN":1610612751,"CHA":1610612766,
    "CHI":1610612741,"CLE":1610612739,"DAL":1610612742,"DEN":1610612743,
    "DET":1610612765,"GSW":1610612744,"HOU":1610612745,"IND":1610612754,
    "LAC":1610612746,"LAL":1610612747,"MEM":1610612763,"MIA":1610612748,
    "MIL":1610612749,"MIN":1610612750,"NOP":1610612740,"NYK":1610612752,
    "OKC":1610612760,"ORL":1610612753,"PHI":1610612755,"PHX":1610612756,
    "POR":1610612757,"SAC":1610612758,"SAS":1610612759,"TOR":1610612761,
    "UTA":1610612762,"WAS":1610612764,
}

def logo_url(abbr):
    tid = TEAM_IDS.get(abbr, 0)
    return f"https://cdn.nba.com/logos/nba/{tid}/global/L/logo.svg"


def _retry(fn, retries=4, base_delay=10):
    """Call fn(), retrying on exception with exponential back-off."""
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:
            if attempt == retries - 1:
                raise
            wait = base_delay * (2 ** attempt)
            print(f"  Attempt {attempt+1} failed ({exc.__class__.__name__}). Retrying in {wait}s ...")
            time.sleep(wait)


def fetch_recent_games(days: int = 60) -> pd.DataFrame:
    from nba_api.stats.endpoints import LeagueGameFinder
    today      = date.today()
    date_from  = (today - timedelta(days=days)).strftime("%m/%d/%Y")
    date_to    = today.strftime("%m/%d/%Y")
    print(f"Fetching games from {date_from} to {date_to} ...")
    time.sleep(2)

    def _fetch():
        lg = LeagueGameFinder(
            date_from_nullable=date_from,
            date_to_nullable=date_to,
            league_id_nullable="00",
            timeout=120,
        )
        return lg.get_data_frames()[0]

    df = _retry(_fetch)
    print(f"  Got {len(df)} team-game rows.")
    return df


def build_team_features(df: pd.DataFrame) -> dict:
    """Return {abbr: feature_dict} using each team's most recent rolling stats."""
    df = df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.dropna(subset=["PTS","FG_PCT","FG3_PCT","REB","AST","TOV","MATCHUP"])
    df["home"] = df["MATCHUP"].apply(home_flag)
    df = rolling_team_stats(df)
    df = df.dropna(subset=FEATURE_COLS)

    team_feats = {}
    for abbr, grp in df.groupby("TEAM_ABBREVIATION"):
        last = grp.sort_values("GAME_DATE").iloc[-1]
        team_feats[abbr] = {col: float(last[col]) for col in FEATURE_COLS}
    return team_feats


def fetch_todays_games(today_str: str) -> list[dict]:
    """Return list of {team_a, team_b, home_a, home_b} for today's schedule."""
    from nba_api.stats.endpoints import ScoreboardV2
    print(f"Fetching schedule for {today_str} ...")
    time.sleep(2)

    def _fetch():
        return ScoreboardV2(game_date=today_str, league_id="00", day_offset=0, timeout=120)

    sb     = _retry(_fetch)
    header = sb.get_data_frames()[0]   # GameHeader
    ls     = sb.get_data_frames()[1]   # LineScore (has TEAM_ABBREVIATION)

    if header.empty:
        print("  No games scheduled today.")
        return []

    games = []
    for _, row in header.iterrows():
        gid = row["GAME_ID"]
        teams = ls[ls["GAME_ID"] == gid].reset_index(drop=True)
        if len(teams) < 2:
            continue
        # home team is the one whose matchup contains "vs."
        home_row  = teams[teams["TEAM_ABBREVIATION"].apply(
            lambda a: _is_home(a, row.get("MATCHUP", ""), teams)
        )]
        # Simpler: home team ID is HOME_TEAM_ID in header
        home_id  = row.get("HOME_TEAM_ID")
        visitor_id = row.get("VISITOR_TEAM_ID")

        home_abbr    = teams[teams["TEAM_ID"] == home_id]["TEAM_ABBREVIATION"].values
        visitor_abbr = teams[teams["TEAM_ID"] == visitor_id]["TEAM_ABBREVIATION"].values

        if len(home_abbr) == 0 or len(visitor_abbr) == 0:
            continue

        games.append({
            "team_a": visitor_abbr[0],   # visitor = team_a
            "team_b": home_abbr[0],      # home    = team_b
            "home_a": 0,
            "home_b": 1,
        })
    print(f"  Found {len(games)} game(s).")
    return games


def _is_home(abbr, matchup_str, teams):
    return False   # unused helper kept for clarity


def main():
    today     = date.today()
    today_str = today.strftime("%m/%d/%Y")

    model = joblib.load(MODEL_PATH)

    # Step 1 – recent games → team features
    raw_df     = fetch_recent_games(days=60)
    team_feats = build_team_features(raw_df)

    # Step 2 – today's schedule
    schedule = fetch_todays_games(today_str)

    # Step 3 – predict
    results = []
    for g in schedule:
        a, b = g["team_a"], g["team_b"]
        if a not in team_feats or b not in team_feats:
            print(f"  Skipping {a} vs {b} — no recent data.")
            continue

        feat_a = {**team_feats[a], "home": g["home_a"]}
        feat_b = {**team_feats[b], "home": g["home_b"]}

        sim = monte_carlo(feat_a, feat_b, model, n=10_000)

        results.append({
            "team_a": {"abbr": a, "name": TEAM_NAMES.get(a, a), "logo": logo_url(a)},
            "team_b": {"abbr": b, "name": TEAM_NAMES.get(b, b), "logo": logo_url(b)},
            "score_a": sim["score_a"],
            "score_b": sim["score_b"],
            "win_prob_a": round(sim["win_prob_a"] * 100, 1),
            "win_prob_b": round(sim["win_prob_b"] * 100, 1),
        })
        print(f"  {a} {sim['score_a']} - {b} {sim['score_b']}  |  {a} {sim['win_prob_a']*100:.1f}%")

    payload = {
        "date": today.strftime("%Y-%m-%d"),
        "generated": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_stats": {"mae": 8.67, "rmse": 10.94, "win_accuracy": 68.5},
        "matchups": results,
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved {len(results)} prediction(s) -> {OUT_PATH}")


if __name__ == "__main__":
    main()

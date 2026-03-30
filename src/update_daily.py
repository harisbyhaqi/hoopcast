"""
Daily update script — run by GitHub Actions.

Data sources (both bypass the stats.nba.com block):
  * Today's schedule → cdn.nba.com live scoreboard JSON (public CDN, no auth)
  * Team rolling stats → data/processed/games_preprocessed.csv (committed to repo)

Flow:
  1. Fetch today's NBA schedule from CDN
  2. Load pre-computed team features from processed CSV
  3. Run pre-trained RF model + Monte Carlo for each matchup
  4. Write docs/predictions.json
"""
import os, sys, json, time
import numpy as np
import pandas as pd
import joblib
import requests
from datetime import date, datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))
from utils import FEATURE_COLS
from simulate import monte_carlo

ROOT           = os.path.join(os.path.dirname(__file__), "..")
MODEL_PATH     = os.path.join(ROOT, "models", "points_rf.joblib")
PROCESSED_PATH = os.path.join(ROOT, "data", "processed", "games_preprocessed.csv")
OUT_PATH       = os.path.join(ROOT, "docs", "predictions.json")

CDN_SCOREBOARD = "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json"

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


def fetch_todays_schedule() -> list[dict]:
    """
    Fetch today's games from the NBA CDN live scoreboard.
    Returns list of {"away": abbr, "home": abbr} dicts.
    """
    # Use America/New_York to correctly handle EST/EDT transitions automatically
    from zoneinfo import ZoneInfo
    today_et = datetime.now(ZoneInfo("America/New_York")).date()

    print(f"Fetching today's schedule from NBA CDN (ET date: {today_et}) ...")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://www.nba.com/",
    }
    resp = requests.get(CDN_SCOREBOARD, headers=headers, timeout=30)
    resp.raise_for_status()

    data       = resp.json()
    scoreboard = data.get("scoreboard", {})
    games      = scoreboard.get("games", [])

    # Validate that the CDN is serving today's game day (not yesterday's stale data)
    cdn_date_str = scoreboard.get("gameDate", "")  # e.g. "2026-03-28"
    if cdn_date_str:
        try:
            cdn_date = date.fromisoformat(cdn_date_str)
            if cdn_date != today_et:
                print(f"  WARNING: CDN scoreboard date ({cdn_date}) does not match today ET ({today_et}). "
                      f"Scoreboard not yet updated — returning no games.")
                return []
        except ValueError:
            pass  # unparseable date, proceed anyway

    print(f"  Found {len(games)} game(s).")

    schedule = []
    for g in games:
        away = g.get("awayTeam", {}).get("teamTricode")
        home = g.get("homeTeam", {}).get("teamTricode")
        if away and home:
            schedule.append({"away": away, "home": home})
    return schedule


def build_team_features(processed_path: str) -> dict:
    """
    Load processed CSV and return the most recent rolling feature row per team.
    {abbr: {feat_col: value, ...}}
    """
    df = pd.read_csv(processed_path, parse_dates=["GAME_DATE"])
    df = df.dropna(subset=FEATURE_COLS)
    features = {}
    for abbr, grp in df.groupby("TEAM_ABBREVIATION"):
        last = grp.sort_values("GAME_DATE").iloc[-1]
        features[abbr] = {col: float(last[col]) for col in FEATURE_COLS}
    print(f"Loaded features for {len(features)} teams from processed CSV.")
    return features


def main():
    today = date.today()
    model = joblib.load(MODEL_PATH)

    schedule   = fetch_todays_schedule()
    team_feats = build_team_features(PROCESSED_PATH)

    results = []
    for g in schedule:
        away, home = g["away"], g["home"]
        if away not in team_feats or home not in team_feats:
            print(f"  Skipping {away} @ {home} — team not in processed data.")
            continue

        feat_away = {**team_feats[away], "home": 0}
        feat_home = {**team_feats[home], "home": 1}

        sim = monte_carlo(feat_away, feat_home, model, n=10_000)

        results.append({
            "team_a": {"abbr": away, "name": TEAM_NAMES.get(away, away), "logo": logo_url(away)},
            "team_b": {"abbr": home, "name": TEAM_NAMES.get(home, home), "logo": logo_url(home)},
            "score_a": sim["score_a"],
            "score_b": sim["score_b"],
            "win_prob_a": round(sim["win_prob_a"] * 100, 1),
            "win_prob_b": round(sim["win_prob_b"] * 100, 1),
        })
        print(f"  {away} {sim['score_a']} @ {home} {sim['score_b']}  |  {away} wins {sim['win_prob_a']*100:.1f}%")

    payload = {
        "date": today.strftime("%Y-%m-%d"),
        "generated": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_stats": {"mae": 8.82, "rmse": 11.18, "win_accuracy": 69.7},
        "matchups": results,
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved {len(results)} prediction(s) -> {OUT_PATH}")


if __name__ == "__main__":
    main()

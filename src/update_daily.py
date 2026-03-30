"""
Daily update script — run by GitHub Actions.

Data sources:
  * Today's schedule  → cdn.nba.com live scoreboard JSON
  * Team rolling stats → data/processed/games_preprocessed.csv (current season only)
  * Injury report     → ESPN unofficial API (no key needed)
  * Player PPG        → BallDontLie API (for injury impact sizing)
  * H2H history       → data/processed/games_preprocessed.csv (current season matchups)

Flow:
  1. Fetch today's schedule from NBA CDN
  2. Load current-season-only team features from processed CSV
  3. For each matchup:
       a. Compute injury score offset (stepped model by PPG tier)
       b. Compute H2H offset (avg margin this season, capped at ±2.5 pts)
       c. Run Monte Carlo simulation with combined offsets
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

CDN_SCOREBOARD   = "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json"
BALLDONTLIE_BASE = "https://api.balldontlie.io/v1"

# Stepped injury impact: pts actually lost given star's PPG tier and status.
# Reflects that elite players are harder to replace than role players.
# Teammates absorb the rest of the usage so we don't remove the full PPG.
def _injury_pts_lost(ppg: float, status: str) -> float:
    if status in ("out", "doubtful"):
        if ppg >= 30:   return 8.0
        elif ppg >= 25: return 6.0
        elif ppg >= 20: return 4.5
        else:           return 3.0   # 15-20 PPG
    elif status == "questionable":
        if ppg >= 25:   return 3.0
        elif ppg >= 20: return 2.0
        else:           return 1.5
    elif status == "day-to-day":
        if ppg >= 25:   return 1.5
        else:           return 1.0
    return 0.0

KEY_PLAYER_PPG_THRESHOLD = 15.0

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

ESPN_TEAM_IDS = {
    "ATL":1,"BOS":2,"BKN":17,"CHA":30,"CHI":4,"CLE":5,"DAL":6,"DEN":7,
    "DET":8,"GSW":9,"HOU":10,"IND":11,"LAC":12,"LAL":13,"MEM":29,"MIA":14,
    "MIL":15,"MIN":16,"NOP":3,"NYK":18,"OKC":25,"ORL":19,"PHI":20,"PHX":21,
    "POR":22,"SAC":23,"SAS":24,"TOR":28,"UTA":26,"WAS":27,
}


def logo_url(abbr):
    tid = TEAM_IDS.get(abbr, 0)
    return f"https://cdn.nba.com/logos/nba/{tid}/global/L/logo.svg"


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------

def fetch_todays_schedule() -> list[dict]:
    from zoneinfo import ZoneInfo
    today_et = datetime.now(ZoneInfo("America/New_York")).date()

    print(f"Fetching today's schedule (ET: {today_et}) ...")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://www.nba.com/",
    }
    resp = requests.get(CDN_SCOREBOARD, headers=headers, timeout=30)
    resp.raise_for_status()

    data       = resp.json()
    scoreboard = data.get("scoreboard", {})
    games      = scoreboard.get("games", [])

    cdn_date_str = scoreboard.get("gameDate", "")
    if cdn_date_str:
        try:
            cdn_date = date.fromisoformat(cdn_date_str)
            if cdn_date != today_et:
                print(f"  WARNING: CDN date ({cdn_date}) != today ET ({today_et}). No games returned.")
                return []
        except ValueError:
            pass

    print(f"  Found {len(games)} game(s).")
    schedule = []
    for g in games:
        away = g.get("awayTeam", {}).get("teamTricode")
        home = g.get("homeTeam", {}).get("teamTricode")
        if away and home:
            schedule.append({"away": away, "home": home})
    return schedule


# ---------------------------------------------------------------------------
# Team features (current season only)
# ---------------------------------------------------------------------------

def build_team_features(processed_path: str) -> tuple[dict, pd.DataFrame]:
    """
    Returns (features_dict, current_season_df).
    features_dict: {abbr: {feat_col: value}} using each team's most recent game.
    current_season_df: full current-season df for H2H lookups.
    """
    df = pd.read_csv(processed_path, parse_dates=["GAME_DATE"])
    df = df.dropna(subset=FEATURE_COLS)

    # Regular season only (SEASON_ID like 22025)
    df = df[df["SEASON_ID"].astype(str).str.match(r"^2\d{4}$")]
    current_season = df["SEASON_ID"].astype(int).max()
    df = df[df["SEASON_ID"].astype(int) == current_season]

    features = {}
    for abbr, grp in df.groupby("TEAM_ABBREVIATION"):
        last = grp.sort_values("GAME_DATE").iloc[-1]
        features[abbr] = {col: float(last[col]) for col in FEATURE_COLS}

    print(f"Loaded features for {len(features)} teams (current season only).")
    return features, df


# ---------------------------------------------------------------------------
# H2H
# ---------------------------------------------------------------------------

def compute_h2h_offsets(away: str, home: str, season_df: pd.DataFrame) -> tuple[float, float]:
    """
    Look up this season's head-to-head games between away and home.
    Returns (away_offset, home_offset) — small score nudge based on avg margin.
    Capped at ±2.5 pts so H2H is a soft signal, not a dominant one.
    """
    away_games = season_df[season_df["TEAM_ABBREVIATION"] == away]
    home_games = season_df[season_df["TEAM_ABBREVIATION"] == home]

    shared_ids = set(away_games["GAME_ID"]) & set(home_games["GAME_ID"])
    if not shared_ids:
        return 0.0, 0.0

    away_h2h = away_games[away_games["GAME_ID"].isin(shared_ids)][["GAME_ID", "PTS"]].rename(columns={"PTS": "away_pts"})
    home_h2h = home_games[home_games["GAME_ID"].isin(shared_ids)][["GAME_ID", "PTS"]].rename(columns={"PTS": "home_pts"})

    merged = away_h2h.merge(home_h2h, on="GAME_ID")
    if merged.empty:
        return 0.0, 0.0

    avg_away_margin = float((merged["away_pts"] - merged["home_pts"]).mean())
    # Apply 30% of avg margin, capped at ±2.5 pts
    raw_offset = avg_away_margin * 0.30
    offset = float(np.clip(raw_offset, -2.5, 2.5))

    games_played = len(merged)
    print(f"  H2H ({away} vs {home}): {games_played} game(s) this season, "
          f"avg away margin {avg_away_margin:+.1f} → offset {offset:+.1f} pts")

    return offset, -offset


# ---------------------------------------------------------------------------
# Injuries
# ---------------------------------------------------------------------------

def fetch_team_injuries(team_abbr: str) -> list[dict]:
    """Fetch active injury report for a team from ESPN's unofficial API."""
    espn_id = ESPN_TEAM_IDS.get(team_abbr)
    if not espn_id:
        return []
    try:
        url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{espn_id}/injuries"
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        resp.raise_for_status()
        result = []
        for item in resp.json().get("injuries", []):
            name   = item.get("athlete", {}).get("displayName", "")
            status = item.get("status", "").lower()
            if name and any(s in status for s in ("out", "doubtful", "questionable", "day-to-day")):
                # Normalise to clean keys
                if "out" in status:          key = "out"
                elif "doubtful" in status:   key = "doubtful"
                elif "questionable" in status: key = "questionable"
                else:                         key = "day-to-day"
                result.append({"name": name, "status": key})
        return result
    except Exception as e:
        print(f"  Could not fetch injuries for {team_abbr}: {e}")
        return []


def get_player_ppg(api_key: str, player_name: str) -> float:
    """Look up a player's current season PPG via BallDontLie."""
    try:
        resp = requests.get(
            f"{BALLDONTLIE_BASE}/players",
            headers={"Authorization": api_key},
            params={"search": player_name, "per_page": 5},
            timeout=10,
        )
        resp.raise_for_status()
        players = resp.json().get("data", [])
        if not players:
            return 0.0
        player_id = players[0]["id"]
        time.sleep(0.3)

        today  = date.today()
        season = today.year if today.month >= 10 else today.year - 1
        resp = requests.get(
            f"{BALLDONTLIE_BASE}/season_averages",
            headers={"Authorization": api_key},
            params={"player_ids[]": player_id, "season": season},
            timeout=10,
        )
        resp.raise_for_status()
        avgs = resp.json().get("data", [])
        return float(avgs[0].get("pts", 0.0)) if avgs else 0.0
    except Exception:
        return 0.0


def compute_injury_offset(team_abbr: str, api_key: str) -> float:
    """
    Returns a negative pts offset for a team's predicted score.
    Uses a stepped model by PPG tier — elite players are harder to replace
    so the team loses more. Teammates absorb most of the usage.
    Only applied to key players (>15 PPG).
    """
    if not api_key:
        return 0.0

    injuries = fetch_team_injuries(team_abbr)
    if not injuries:
        return 0.0

    total_offset = 0.0
    for inj in injuries:
        ppg = get_player_ppg(api_key, inj["name"])
        if ppg < KEY_PLAYER_PPG_THRESHOLD:
            continue
        pts_lost = _injury_pts_lost(ppg, inj["status"])
        if pts_lost > 0:
            total_offset += pts_lost
            print(f"  Injury: {inj['name']} ({inj['status']}, {ppg:.1f} PPG) "
                  f"→ -{pts_lost:.1f} pts for {team_abbr}")
        time.sleep(0.3)

    return -total_offset


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    today   = date.today()
    model   = joblib.load(MODEL_PATH)
    bdl_key = os.environ.get("BALLDONTLIE_API_KEY", "")

    schedule              = fetch_todays_schedule()
    team_feats, season_df = build_team_features(PROCESSED_PATH)

    results = []
    for g in schedule:
        away, home = g["away"], g["home"]
        if away not in team_feats or home not in team_feats:
            print(f"  Skipping {away} @ {home} — team not in processed data.")
            continue

        feat_away = {**team_feats[away], "home": 0}
        feat_home = {**team_feats[home], "home": 1}

        # Injury adjustments
        offset_away = compute_injury_offset(away, bdl_key)
        offset_home = compute_injury_offset(home, bdl_key)

        # H2H adjustment
        h2h_away, h2h_home = compute_h2h_offsets(away, home, season_df)
        offset_away += h2h_away
        offset_home += h2h_home

        sim = monte_carlo(
            feat_away, feat_home, model, n=10_000,
            score_a_offset=offset_away,
            score_b_offset=offset_home,
        )

        results.append({
            "team_a": {"abbr": away, "name": TEAM_NAMES.get(away, away), "logo": logo_url(away)},
            "team_b": {"abbr": home, "name": TEAM_NAMES.get(home, home), "logo": logo_url(home)},
            "score_a": sim["score_a"],
            "score_b": sim["score_b"],
            "win_prob_a": round(sim["win_prob_a"] * 100, 1),
            "win_prob_b": round(sim["win_prob_b"] * 100, 1),
        })
        print(f"  {away} {sim['score_a']} @ {home} {sim['score_b']}  |  "
              f"{away} wins {sim['win_prob_a']*100:.1f}%")

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

"""
Daily update script — run by GitHub Actions.

Data sources:
  * Today's schedule  → cdn.nba.com live scoreboard JSON
  * Team rolling stats → data/processed/games_preprocessed.csv (current season only)
  * Injury report     → ESPN unofficial API (no key needed)
  * Player PPG        → BallDontLie API (for injury impact sizing)

Flow:
  1. Fetch today's NBA schedule from CDN
  2. Load pre-computed team features from current season only
  3. Fetch injury report and compute partial score adjustments for key injured players
  4. Run pre-trained RF model + Monte Carlo (with injury offsets) for each matchup
  5. Write docs/predictions.json
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

CDN_SCOREBOARD    = "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json"
BALLDONTLIE_BASE  = "https://api.balldontlie.io/v1"

# Fraction of an injured player's PPG the team loses.
# The rest (~67%) gets redistributed to teammates.
INJURY_IMPACT = {
    "out":          0.30,
    "doubtful":     0.30,
    "questionable": 0.12,
    "day-to-day":   0.08,
}
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


def fetch_todays_schedule() -> list[dict]:
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


def build_team_features(processed_path: str) -> dict:
    """
    Load current season features only — ensures predictions reflect
    how teams are actually playing this year, not past seasons.
    Regular season SEASON_IDs start with '2' (e.g. 22025). Filter to
    those only so preseason (1xxxx) and playoff (4xxxx) IDs don't skew max().
    """
    df = pd.read_csv(processed_path, parse_dates=["GAME_DATE"])
    df = df.dropna(subset=FEATURE_COLS)

    # Keep only regular season rows (SEASON_ID like 22025, 22024)
    df = df[df["SEASON_ID"].astype(str).str.match(r"^2\d{4}$")]

    # Restrict to current regular season only
    current_season = df["SEASON_ID"].astype(int).max()
    df = df[df["SEASON_ID"].astype(int) == current_season]

    features = {}
    for abbr, grp in df.groupby("TEAM_ABBREVIATION"):
        last = grp.sort_values("GAME_DATE").iloc[-1]
        features[abbr] = {col: float(last[col]) for col in FEATURE_COLS}
    print(f"Loaded features for {len(features)} teams (current season only).")
    return features


# ---------------------------------------------------------------------------
# Injury handling
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
        data = resp.json()
        result = []
        for item in data.get("injuries", []):
            name   = item.get("athlete", {}).get("displayName", "")
            status = item.get("status", "").lower()
            if name and status in INJURY_IMPACT:
                result.append({"name": name, "status": status})
        return result
    except Exception as e:
        print(f"  Could not fetch injuries for {team_abbr}: {e}")
        return []


def get_player_season_ppg(api_key: str, player_name: str) -> float:
    """Look up a player's current season PPG via BallDontLie."""
    try:
        # Find player ID
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

        # Get season averages for current season
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
    Returns a negative pts offset to apply to a team's predicted score.
    Only key players (>15 PPG) meaningfully shift the number — and even then
    only ~30% of their points are lost (teammates absorb the rest).
    """
    if not api_key:
        return 0.0

    injuries = fetch_team_injuries(team_abbr)
    if not injuries:
        return 0.0

    total_offset = 0.0
    for inj in injuries:
        ppg = get_player_season_ppg(api_key, inj["name"])
        if ppg < KEY_PLAYER_PPG_THRESHOLD:
            continue
        impact    = INJURY_IMPACT.get(inj["status"], 0.0)
        pts_lost  = ppg * impact
        total_offset += pts_lost
        print(f"  Injury: {inj['name']} ({inj['status']}, {ppg:.1f} PPG) "
              f"→ -{pts_lost:.1f} pts adjustment for {team_abbr}")
        time.sleep(0.3)

    return -total_offset  # negative = subtract from predicted score


def main():
    today     = date.today()
    model     = joblib.load(MODEL_PATH)
    bdl_key   = os.environ.get("BALLDONTLIE_API_KEY", "")

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

        # Injury adjustments (partial — teammates absorb most of the slack)
        offset_away = compute_injury_offset(away, bdl_key)
        offset_home = compute_injury_offset(home, bdl_key)

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

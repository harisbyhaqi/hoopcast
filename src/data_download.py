"""
Fetch NBA game data from BallDontLie API (works from GitHub Actions) and
incrementally append new games to data/raw/games_raw.csv.

Requires env var: BALLDONTLIE_API_KEY (free at balldontlie.io)
"""
import os
import time
from datetime import date, timedelta
import pandas as pd
import requests

RAW_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "games_raw.csv")
BASE_URL = "https://api.balldontlie.io/v1"

NBA_TEAM_IDS = {
    "ATL": 1610612737, "BOS": 1610612738, "BKN": 1610612751, "CHA": 1610612766,
    "CHI": 1610612741, "CLE": 1610612739, "DAL": 1610612742, "DEN": 1610612743,
    "DET": 1610612765, "GSW": 1610612744, "HOU": 1610612745, "IND": 1610612754,
    "LAC": 1610612746, "LAL": 1610612747, "MEM": 1610612763, "MIA": 1610612748,
    "MIL": 1610612749, "MIN": 1610612750, "NOP": 1610612740, "NYK": 1610612752,
    "OKC": 1610612760, "ORL": 1610612753, "PHI": 1610612755, "PHX": 1610612756,
    "POR": 1610612757, "SAC": 1610612758, "SAS": 1610612759, "TOR": 1610612761,
    "UTA": 1610612762, "WAS": 1610612764,
}

NBA_TEAM_NAMES = {
    "ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets", "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets", "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers",
    "LAC": "LA Clippers", "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat", "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans", "NYK": "New York Knicks", "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic", "PHI": "Philadelphia 76ers", "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers", "SAC": "Sacramento Kings", "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors", "UTA": "Utah Jazz", "WAS": "Washington Wizards",
}


def current_bdl_seasons() -> list[int]:
    """Return [current_season, previous_season] as BallDontLie season years."""
    today = date.today()
    year = today.year if today.month >= 10 else today.year - 1
    return [year, year - 1]


def _paginate(api_key: str, endpoint: str, params: list[tuple]) -> list[dict]:
    """Fetch all pages from a BallDontLie endpoint using cursor-based pagination."""
    headers = {"Authorization": api_key}
    results = []
    cursor = None
    while True:
        p = params + [("per_page", 100)]
        if cursor:
            p = p + [("cursor", cursor)]
        resp = requests.get(f"{BASE_URL}/{endpoint}", headers=headers, params=p, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        results.extend(data["data"])
        cursor = data.get("meta", {}).get("next_cursor")
        if not cursor:
            break
        time.sleep(0.5)
    return results


def fetch_completed_games(api_key: str, start_date: str) -> list[dict]:
    """Fetch all completed games on or after start_date (YYYY-MM-DD)."""
    seasons = current_bdl_seasons()
    params = (
        [("start_date", start_date)] +
        [("seasons[]", s) for s in seasons]
    )
    games = _paginate(api_key, "games", params)
    return [g for g in games if isinstance(g.get("status"), str) and "final" in g["status"].lower()]


def fetch_player_stats(api_key: str, game_ids: list[int]) -> list[dict]:
    """Fetch all player stats for the given game IDs in batches of 25."""
    all_stats = []
    for i in range(0, len(game_ids), 25):
        batch = game_ids[i:i + 25]
        params = [("game_ids[]", gid) for gid in batch]
        stats = _paginate(api_key, "stats", params)
        all_stats.extend(stats)
        time.sleep(1.0)
    return all_stats


def build_team_rows(games: list[dict], player_stats: list[dict]) -> pd.DataFrame:
    """Aggregate player stats into team-level rows matching the games_raw.csv schema."""
    if not player_stats:
        return pd.DataFrame()

    stats_df = pd.DataFrame(player_stats)
    stats_df["_game_id"] = stats_df["game"].apply(lambda g: g["id"])
    stats_df["_team_abbr"] = stats_df["team"].apply(lambda t: t["abbreviation"])

    # Normalise turnovers field name across API versions
    if "turnover" not in stats_df.columns:
        stats_df["turnover"] = stats_df.get("to", 0)

    # Ensure reb column exists
    if "reb" not in stats_df.columns:
        stats_df["reb"] = stats_df.get("oreb", 0) + stats_df.get("dreb", 0)

    rows = []
    for game in games:
        gid = game["id"]
        game_date = game["date"][:10]
        season = game["season"]
        home_abbr = game["home_team"]["abbreviation"]
        away_abbr = game["visitor_team"]["abbreviation"]
        home_score = game.get("home_team_score") or 0
        away_score = game.get("visitor_team_score") or 0

        for abbr, score, opp_score in [
            (home_abbr, home_score, away_score),
            (away_abbr, away_score, home_score),
        ]:
            ts = stats_df[(stats_df["_game_id"] == gid) & (stats_df["_team_abbr"] == abbr)]
            if ts.empty:
                continue

            fgm = int(ts["fgm"].sum())
            fga = int(ts["fga"].sum())
            fg3m = int(ts["fg3m"].sum())
            fg3a = int(ts["fg3a"].sum())
            ftm = int(ts["ftm"].sum())
            fta = int(ts["fta"].sum())

            matchup = (
                f"{home_abbr} vs. {away_abbr}"
                if abbr == home_abbr
                else f"{away_abbr} @ {home_abbr}"
            )

            rows.append({
                "SEASON_ID": int(f"2{season}"),
                "TEAM_ID": NBA_TEAM_IDS.get(abbr, 0),
                "TEAM_ABBREVIATION": abbr,
                "TEAM_NAME": NBA_TEAM_NAMES.get(abbr, abbr),
                "GAME_ID": f"bdl_{gid:08d}",
                "GAME_DATE": game_date,
                "MATCHUP": matchup,
                "WL": "W" if score > opp_score else "L",
                "MIN": 240,
                "PTS": int(ts["pts"].sum()),
                "FGM": fgm,
                "FGA": fga,
                "FG_PCT": round(fgm / fga, 3) if fga > 0 else 0.0,
                "FG3M": fg3m,
                "FG3A": fg3a,
                "FG3_PCT": round(fg3m / fg3a, 3) if fg3a > 0 else 0.0,
                "FTM": ftm,
                "FTA": fta,
                "FT_PCT": round(ftm / fta, 3) if fta > 0 else 0.0,
                "OREB": int(ts["oreb"].sum()),
                "DREB": int(ts["dreb"].sum()),
                "REB": int(ts["reb"].sum()),
                "AST": int(ts["ast"].sum()),
                "STL": int(ts["stl"].sum()),
                "BLK": int(ts["blk"].sum()),
                "TOV": int(ts["turnover"].sum()),
                "PF": int(ts["pf"].sum()),
                "PLUS_MINUS": float(score - opp_score),
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    api_key = os.environ.get("BALLDONTLIE_API_KEY", "")

    if not api_key:
        print("WARNING: BALLDONTLIE_API_KEY not set. Using cached data.")
        raise SystemExit(0)

    out = RAW_PATH

    # Determine start date: day after the last game already in the CSV
    if os.path.exists(out):
        existing = pd.read_csv(out)
        last_date = pd.to_datetime(existing["GAME_DATE"]).max().date()
        start_date = (last_date + timedelta(days=1)).isoformat()
        print(f"Existing data through {last_date}. Fetching from {start_date} onwards...")
    else:
        today = date.today()
        year = today.year if today.month >= 10 else today.year - 1
        start_date = f"{year}-10-01"
        print(f"No existing data. Fetching full season from {start_date}...")
        existing = None

    games = fetch_completed_games(api_key, start_date)
    print(f"Found {len(games)} completed game(s) since {start_date}.")

    if not games:
        print("No new games to add. CSV is already up to date.")
        raise SystemExit(0)

    game_ids = [g["id"] for g in games]
    print(f"Fetching player stats for {len(game_ids)} game(s)...")
    player_stats = fetch_player_stats(api_key, game_ids)
    print(f"Retrieved {len(player_stats)} player-stat records.")

    new_rows = build_team_rows(games, player_stats)
    if new_rows.empty:
        print("No rows built — player stats may not be available yet.")
        raise SystemExit(0)

    print(f"Built {len(new_rows)} new team-game rows.")

    if existing is not None:
        combined = pd.concat([existing, new_rows], ignore_index=True)
    else:
        combined = new_rows

    combined = combined.sort_values("GAME_DATE", ascending=False).reset_index(drop=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    combined.to_csv(out, index=False)
    print(f"Saved {len(combined)} total rows to {out}")

"""
Refresh the season schedule and append newly finished games from ESPN.

Outputs:
  data/schedule.csv            every game (past + future) for the current and previous season
  data/raw/games_raw.csv       one row per team per finished game (appended)
  data/raw/player_games.csv    one row per player per finished game (appended; feeds injury PPG)

No API key needed. Rows that came from nba_api before the ESPN switch are kept as-is;
ESPN only backfills games after the last nba_api date.
"""
import os
import time
from datetime import datetime

import pandas as pd

import espn

ROOT          = os.path.join(os.path.dirname(__file__), "..")
SCHEDULE_PATH = os.path.join(ROOT, "data", "schedule.csv")
RAW_PATH      = os.path.join(ROOT, "data", "raw", "games_raw.csv")
PLAYER_PATH   = os.path.join(ROOT, "data", "raw", "player_games.csv")

SEASON_PREFIX = {"regular": "2", "playoffs": "4"}


def season_id(season_end_year: int, season_type: str) -> int:
    """NBA-style SEASON_ID: 22025 = 2025-26 regular season, 42025 = its playoffs."""
    return int(f"{SEASON_PREFIX[season_type]}{season_end_year - 1}")


def refresh_schedule(today) -> pd.DataFrame:
    current = espn.season_end_year(today)
    old = pd.read_csv(SCHEDULE_PATH, dtype={"event_id": str}) if os.path.exists(SCHEDULE_PATH) else pd.DataFrame()

    frames = []
    for season in (current - 1, current):
        cached = old[old["season"] == season] if not old.empty else old
        # Past seasons never change once nothing is left to play; re-fetch only the live one.
        if season != current and not cached.empty and (cached["status"] != "pre").all():
            frames.append(cached)
            continue
        print(f"Fetching {season - 1}-{str(season)[2:]} schedule from ESPN ...")
        frames.append(pd.DataFrame(espn.fetch_season_schedule(season)))

    sched = pd.concat(frames, ignore_index=True)
    sched = sched[sched["season_type"].isin(SEASON_PREFIX)]
    sched = sched.sort_values(["tip_utc", "event_id"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(SCHEDULE_PATH), exist_ok=True)
    sched.to_csv(SCHEDULE_PATH, index=False)
    print(f"Schedule: {len(sched)} games ({(sched['status'] == 'final').sum()} final) -> {SCHEDULE_PATH}")
    return sched


def build_rows(game: pd.Series, team_rows: list[dict], player_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    sid = season_id(int(game["season"]), game["season_type"])
    gid = f"espn_{game['event_id']}"
    pts = {t["abbr"]: t["pts"] for t in team_rows}

    rows = []
    for t in team_rows:
        abbr = t["abbr"]
        opp = game["away"] if abbr == game["home"] else game["home"]
        matchup = f"{abbr} vs. {opp}" if abbr == game["home"] else f"{abbr} @ {opp}"
        rows.append({
            "SEASON_ID": sid,
            "TEAM_ID": espn.TEAM_IDS[abbr],
            "TEAM_ABBREVIATION": abbr,
            "TEAM_NAME": espn.TEAM_NAMES[abbr],
            "GAME_ID": gid,
            "GAME_DATE": game["date"],
            "MATCHUP": matchup,
            "WL": "W" if t["pts"] > pts[opp] else "L",
            "MIN": 240,
            "PTS": t["pts"],
            "FGM": t["fgm"], "FGA": t["fga"],
            "FG_PCT": round(t["fgm"] / t["fga"], 3) if t["fga"] else 0.0,
            "FG3M": t["fg3m"], "FG3A": t["fg3a"],
            "FG3_PCT": round(t["fg3m"] / t["fg3a"], 3) if t["fg3a"] else 0.0,
            "FTM": t["ftm"], "FTA": t["fta"],
            "FT_PCT": round(t["ftm"] / t["fta"], 3) if t["fta"] else 0.0,
            "OREB": t["oreb"], "DREB": t["dreb"], "REB": t["reb"],
            "AST": t["ast"], "STL": t["stl"], "BLK": t["blk"],
            "TOV": t["tov"], "PF": t["pf"],
            "PLUS_MINUS": float(t["pts"] - pts[opp]),
        })

    players = [{"GAME_ID": gid, "GAME_DATE": game["date"], "SEASON_ID": sid, **p} for p in player_rows]
    return rows, players


def download_new_games(sched: pd.DataFrame) -> int:
    raw = pd.read_csv(RAW_PATH, dtype={"GAME_ID": str}) if os.path.exists(RAW_PATH) else pd.DataFrame()
    players = pd.read_csv(PLAYER_PATH, dtype={"GAME_ID": str, "player_id": str}) if os.path.exists(PLAYER_PATH) else pd.DataFrame()

    have = set(raw["GAME_ID"]) if not raw.empty else set()
    # Games before this date came from nba_api with different IDs — don't duplicate them.
    legacy = raw[~raw["GAME_ID"].str.startswith("espn_")] if not raw.empty else raw
    cutoff = legacy["GAME_DATE"].max() if not legacy.empty else "0000-00-00"

    have_players = set(players["GAME_ID"]) if not players.empty else set()
    final = sched[sched["status"] == "final"]
    ids = "espn_" + final["event_id"]
    need_team = (final["date"] > cutoff) & ~ids.isin(have)
    need_players = ~ids.isin(have_players)
    todo = final[need_team | need_players]
    print(f"{need_team.sum()} game(s) need team rows (after {cutoff}), "
          f"{need_players.sum()} need player rows.")

    new_rows, new_players = [], []
    for _, game in todo.iterrows():
        try:
            team_rows, player_rows = espn.fetch_box_score(game["event_id"])
        except Exception as e:
            print(f"  Skipping {game['away']} @ {game['home']} ({game['date']}): {e}")
            continue
        if len(team_rows) != 2 or any(t["pts"] is None for t in team_rows):
            print(f"  Skipping {game['away']} @ {game['home']} ({game['date']}): incomplete box score")
            continue
        rows, plist = build_rows(game, team_rows, player_rows)
        if need_team[game.name]:
            new_rows += rows
        if need_players[game.name]:
            new_players += plist
        time.sleep(0.2)

    if new_rows:
        raw = pd.concat([raw, pd.DataFrame(new_rows)], ignore_index=True)
        raw = raw.sort_values("GAME_DATE", ascending=False).reset_index(drop=True)
        os.makedirs(os.path.dirname(RAW_PATH), exist_ok=True)
        raw.to_csv(RAW_PATH, index=False)

    if new_players:
        players = pd.concat([players, pd.DataFrame(new_players)], ignore_index=True)
        players = players.sort_values(["GAME_DATE", "GAME_ID"], ascending=False).reset_index(drop=True)
        players.to_csv(PLAYER_PATH, index=False)

    print(f"Added {len(new_rows)} team rows and {len(new_players)} player rows.")
    return len(new_rows) // 2


if __name__ == "__main__":
    today = datetime.now(espn.ET).date()
    schedule = refresh_schedule(today)
    download_new_games(schedule)

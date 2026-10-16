"""
Season forecaster — run by GitHub Actions every morning and evening.

Re-predicts every remaining game of the season from each team's current form, so the
whole calendar shifts as results, rosters and injuries change.

Data sources:
  * Schedule          → data/schedule.csv (refreshed from ESPN by data_download.py)
  * Team form         → data/raw/games_raw.csv (state after each team's latest game)
  * Injury report     → data/injuries.json (snapshotted by rosters.py)
  * Roster changes    → data/changes.json (logged by rosters.py)
  * Player PPG        → data/raw/player_games.csv (our own box scores, for injury sizing)
  * Model accuracy    → models/metrics.json (written by train.py)

Flow:
  1. Compute each team's current state; rest days per game come from the schedule
  2. For every upcoming game:
       a. Injury offset (stepped by PPG tier; Out until return date or next 3 games,
          Questionable/Day-to-day next game only; only players who've been playing)
       b. H2H offset (avg margin this season, capped at ±2.5 pts)
       c. Closed-form win probability with the hold-out margin σ
  3. Freeze the prediction for games tipping off today (data/frozen_predictions.csv)
     so finished games are scored against what we said before tip-off
  4. Write docs/season.json: calendar, projected standings, accuracy, injuries, changes

Usage: python src/update_daily.py [YYYY-MM-DD]   (defaults to today, Eastern time)
"""
import os, sys, json
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import espn
from preprocess import clean_games
from predict import load_model, predict_games
from simulate import simulate_game, DEFAULT_MARGIN_SIGMA
from utils import season_of, team_states

ROOT          = os.path.join(os.path.dirname(__file__), "..")
SCHEDULE_PATH = os.path.join(ROOT, "data", "schedule.csv")
PLAYER_PATH   = os.path.join(ROOT, "data", "raw", "player_games.csv")
METRICS_PATH  = os.path.join(ROOT, "models", "metrics.json")
INJURIES_PATH = os.path.join(ROOT, "data", "injuries.json")
CHANGES_PATH  = os.path.join(ROOT, "data", "changes.json")
FROZEN_PATH   = os.path.join(ROOT, "data", "frozen_predictions.csv")
OUT_PATH      = os.path.join(ROOT, "docs", "season.json")

KEY_PLAYER_PPG_THRESHOLD = 15.0
RECENT_TEAM_GAMES        = 10   # a player absent this long is already baked into team form
OUT_DEFAULT_GAMES        = 3    # Out/Doubtful with no return date: apply to the team's next N games
CHANGES_SHOWN            = 150


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


# ---------------------------------------------------------------------------
# Schedule & rest
# ---------------------------------------------------------------------------

def load_schedule() -> pd.DataFrame:
    return pd.read_csv(SCHEDULE_PATH, dtype={"event_id": str})


def rest_days(schedule: pd.DataFrame, team: str, game_date: str) -> float:
    """Days since the team's previous scheduled game (clipped 1–10, like the training feature)."""
    prior = schedule[((schedule["home"] == team) | (schedule["away"] == team)) & (schedule["date"] < game_date)]
    if prior.empty:
        return 10.0
    days = (date.fromisoformat(game_date) - date.fromisoformat(prior["date"].max())).days
    return float(np.clip(days, 1, 10))


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
    offset = float(np.clip(avg_away_margin * 0.30, -2.5, 2.5))
    return offset, -offset


# ---------------------------------------------------------------------------
# Injuries
# ---------------------------------------------------------------------------

class PlayerImpact:
    """PPG and recent-availability lookups from our own player box scores."""

    def __init__(self, player_path: str = PLAYER_PATH):
        if os.path.exists(player_path):
            df = pd.read_csv(player_path, dtype={"GAME_ID": str, "player_id": str})
            df["season"] = df["SEASON_ID"].map(season_of)
        else:
            df = pd.DataFrame(columns=["GAME_ID", "GAME_DATE", "season", "player_id", "player", "team", "pts"])
        self.df = df

    def ppg(self, player_id: str, name: str) -> float:
        rows = self.df[self.df["player_id"] == player_id] if player_id else self.df.iloc[0:0]
        if rows.empty:
            rows = self.df[self.df["player"] == name]
        if rows.empty:
            return 0.0
        latest = rows["season"].max()
        this_season = rows[rows["season"] == latest]
        if len(this_season) >= 3:
            return float(this_season["pts"].mean())
        prior = rows[rows["season"] == latest - 1]
        return float(prior["pts"].mean()) if len(prior) >= 10 else float(rows["pts"].mean())

    def played_recently(self, player_id: str, name: str, team: str) -> bool:
        """False if the player missed the team's last N games — team form already reflects it."""
        team_games = self.df[self.df["team"] == team].drop_duplicates("GAME_ID")
        recent = set(team_games.sort_values("GAME_DATE").tail(RECENT_TEAM_GAMES)["GAME_ID"])
        mine = self.df[(self.df["player_id"] == player_id) | (self.df["player"] == name)]
        return bool(recent & set(mine["GAME_ID"])) or not recent


def key_injuries(injuries: list[dict], impact: PlayerImpact) -> list[dict]:
    """
    Injuries that move the needle, with their score impact attached.
    Stepped by PPG tier — elite players are harder to replace. Only key players (>15 PPG)
    who have actually been playing count.
    """
    result = []
    for inj in injuries:
        ppg = impact.ppg(inj["player_id"], inj["player"])
        if ppg < KEY_PLAYER_PPG_THRESHOLD or not impact.played_recently(inj["player_id"], inj["player"], inj["team"]):
            continue
        lost = _injury_pts_lost(ppg, inj["status"])
        if lost > 0:
            result.append({**inj, "ppg": round(ppg, 1), "pts_lost": lost})
    return result


def injuries_by_game(injuries: list[dict], upcoming: pd.DataFrame) -> dict[str, list[dict]]:
    """
    Which upcoming games each injury applies to:
      Out/Doubtful → every game before ESPN's return date, else the team's next 3 games
      Questionable/Day-to-day → the team's next game only
    """
    upcoming = upcoming.sort_values("tip_utc")
    by_game: dict[str, list[dict]] = {}
    for inj in injuries:
        team_games = upcoming[(upcoming["home"] == inj["team"]) | (upcoming["away"] == inj["team"])]
        if inj["status"] in ("out", "doubtful"):
            rd = inj.get("return_date")
            if rd and not team_games.empty and rd > team_games["date"].iloc[0]:
                games = team_games[team_games["date"] < rd]
            else:
                games = team_games.head(OUT_DEFAULT_GAMES)
        else:
            games = team_games.head(1)
        for eid in games["event_id"]:
            by_game.setdefault(eid, []).append(inj)
    return by_game


def injury_offset(team: str, game_injuries: list[dict]) -> tuple[float, list[dict]]:
    """Negative pts offset for a team's predicted score, plus the injuries that drove it."""
    applied = [{"player": i["player"], "status": i["status"], "ppg": i["ppg"], "pts_lost": i["pts_lost"]}
               for i in game_injuries if i["team"] == team]
    return -sum(i["pts_lost"] for i in applied), applied


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def _load_json(path: str, default):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default


def load_injuries() -> list[dict]:
    """Latest snapshot from rosters.py; falls back to a live fetch if it hasn't run."""
    snap = _load_json(INJURIES_PATH, None)
    if snap is not None:
        return snap["injuries"]
    try:
        return espn.fetch_injuries()
    except Exception as e:
        print(f"  WARNING: no injury data ({e}); predicting without injury adjustments.")
        return []


def load_metrics() -> dict:
    return _load_json(METRICS_PATH, {"mae": None, "rmse": None, "win_accuracy": None,
                                     "margin_sigma": DEFAULT_MARGIN_SIGMA})


# ---------------------------------------------------------------------------
# Frozen pre-game predictions
# ---------------------------------------------------------------------------

FROZEN_COLS = ["event_id", "date", "home", "away", "pred_home", "pred_away", "p_home", "frozen_at"]


def load_frozen() -> pd.DataFrame:
    if os.path.exists(FROZEN_PATH):
        return pd.read_csv(FROZEN_PATH, dtype={"event_id": str})
    return pd.DataFrame(columns=FROZEN_COLS)


def freeze(frozen: pd.DataFrame, preds: list[dict], today: str, now_utc: datetime) -> pd.DataFrame:
    """Upsert predictions for today's games that haven't tipped off yet."""
    stamp = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = [
        {"event_id": p["id"], "date": p["date"], "home": p["home"], "away": p["away"],
         "pred_home": p["pred"]["home"], "pred_away": p["pred"]["away"], "p_home": p["pred"]["p_home"],
         "frozen_at": stamp}
        for p in preds
        if p["date"] == today and p["tip"] > stamp
    ]
    if not rows:
        return frozen
    new = pd.DataFrame(rows, columns=FROZEN_COLS)
    keep = frozen[~frozen["event_id"].isin(new["event_id"])]
    return pd.concat([keep, new], ignore_index=True).sort_values(["date", "event_id"])


# ---------------------------------------------------------------------------
# Standings & accuracy
# ---------------------------------------------------------------------------

def projected_standings(games: list[dict]) -> dict:
    """Actual W-L plus expected wins from remaining win probabilities (regular season only)."""
    table = {t: {"team": t, "name": espn.TEAM_NAMES[t], "w": 0, "l": 0, "exp_w": 0.0, "left": 0}
             for t in espn.TEAM_NAMES}
    for g in games:
        if g["type"] != "regular":
            continue
        h, a = table[g["home"]], table[g["away"]]
        if g["result"]:
            home_won = g["result"]["home"] > g["result"]["away"]
            (h if home_won else a)["w"] += 1
            (a if home_won else h)["l"] += 1
        elif g["pred"]:
            p = g["pred"]["p_home"] / 100
            h["exp_w"] += p
            a["exp_w"] += 1 - p
            h["left"] += 1
            a["left"] += 1

    out = {"East": [], "West": []}
    for t, r in table.items():
        proj_w = r["w"] + r["exp_w"]
        out[espn.CONFERENCE[t]].append({
            "team": t, "name": r["name"], "w": r["w"], "l": r["l"],
            "proj_w": round(proj_w, 1), "proj_l": round(r["w"] + r["l"] + r["left"] - proj_w, 1),
        })
    for conf in out.values():
        conf.sort(key=lambda r: (-r["proj_w"], -r["w"]))
    return out


def accuracy(games: list[dict], recent: int = 20) -> dict:
    """Score frozen pre-game predictions against final results."""
    scored = [g for g in games if g["result"] and g["frozen"]]
    if not scored:
        return {"games": 0, "correct": 0, "pct": None, "mae": None, "brier": None, "recent": []}
    rows = []
    for g in scored:
        f, r = g["frozen"], g["result"]
        home_won = r["home"] > r["away"]
        rows.append({
            "id": g["id"], "date": g["date"], "home": g["home"], "away": g["away"],
            "correct": (f["p_home"] >= 50) == home_won,
            "abs_err": (abs(f["home"] - r["home"]) + abs(f["away"] - r["away"])) / 2,
            "brier": (f["p_home"] / 100 - home_won) ** 2,
        })
    n = len(rows)
    correct = sum(r["correct"] for r in rows)
    return {
        "games": n,
        "correct": correct,
        "pct": round(100 * correct / n, 1),
        "mae": round(sum(r["abs_err"] for r in rows) / n, 2),
        "brier": round(sum(r["brier"] for r in rows) / n, 3),
        "recent": [{"id": r["id"], "correct": r["correct"]} for r in rows[-recent:]][::-1],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(target: str | None = None):
    now_utc = datetime.now(timezone.utc)
    today = target or now_utc.astimezone(espn.ET).date().isoformat()
    season = espn.season_end_year(date.fromisoformat(today))
    season_id = int(f"2{season - 1}")

    model    = load_model()
    metrics  = load_metrics()
    sigma    = metrics.get("margin_sigma") or DEFAULT_MARGIN_SIGMA
    schedule = load_schedule()
    games_df = clean_games()
    season_sched = schedule[schedule["season"] == season].sort_values(["tip_utc", "event_id"])

    upcoming = season_sched[(season_sched["status"] != "final") & (season_sched["date"] >= today)]
    print(f"{season - 1}-{str(season)[2:]}: {len(season_sched)} games scheduled, {len(upcoming)} to predict.")

    states = team_states(games_df[games_df["GAME_DATE"] < today], today, season_id)
    season_games = games_df[games_df["SEASON_ID"].map(season_of) == season - 1]

    impact = PlayerImpact()
    injuries = key_injuries(load_injuries(), impact)
    game_injuries = injuries_by_game(injuries, upcoming)
    for inj in injuries:
        print(f"  Key injury: {inj['team']} {inj['player']} ({inj['status']}, {inj['ppg']} PPG) → -{inj['pts_lost']} pts")

    # Batch-predict every upcoming game from current team states
    pairs = []
    for _, g in upcoming.iterrows():
        feat_away = {**states.loc[g["away"]].to_dict(), "home": 0, "rest_days": rest_days(schedule, g["away"], g["date"])}
        feat_home = {**states.loc[g["home"]].to_dict(), "home": 1, "rest_days": rest_days(schedule, g["home"], g["date"])}
        pairs.append((feat_away, feat_home))
    raw_preds = dict(zip(upcoming["event_id"], predict_games(pairs, model)))

    frozen = load_frozen()
    frozen_by_id = {}
    games = []
    for _, g in season_sched.iterrows():
        eid, away, home = g["event_id"], g["away"], g["home"]
        entry = {"id": eid, "date": g["date"], "tip": g["tip_utc"], "type": g["season_type"],
                 "home": home, "away": away, "status": g["status"],
                 "pred": None, "result": None, "frozen": None, "injuries": None}
        if g["status"] == "final":
            entry["result"] = {"home": int(g["home_pts"]), "away": int(g["away_pts"])}
        elif eid in raw_preds:
            score_away, score_home = raw_preds[eid]
            off_away, inj_away = injury_offset(away, game_injuries.get(eid, []))
            off_home, inj_home = injury_offset(home, game_injuries.get(eid, []))
            h2h_away, h2h_home = compute_h2h_offsets(away, home, season_games)
            sim = simulate_game(score_away, score_home, sigma,
                                score_a_offset=off_away + h2h_away, score_b_offset=off_home + h2h_home)
            entry["pred"] = {"home": sim["score_b"], "away": sim["score_a"], "p_home": round(sim["win_prob_b"] * 100, 1)}
            if inj_away or inj_home:
                entry["injuries"] = {"home": inj_home, "away": inj_away}
        games.append(entry)

    frozen = freeze(frozen, games, today, now_utc)
    os.makedirs(os.path.dirname(FROZEN_PATH), exist_ok=True)
    frozen.to_csv(FROZEN_PATH, index=False)
    for _, f in frozen.iterrows():
        frozen_by_id[f["event_id"]] = {"home": f["pred_home"], "away": f["pred_away"], "p_home": f["p_home"]}
    for entry in games:
        entry["frozen"] = frozen_by_id.get(entry["id"])

    for entry in (e for e in games if e["date"] == today and e["pred"]):
        p = entry["pred"]
        print(f"  {entry['away']} {p['away']} @ {entry['home']} {p['home']}  |  {entry['home']} wins {p['p_home']}%")

    injuries_now = {}
    for inj in sorted(load_injuries(), key=lambda i: (i["team"], i["player"])):
        injuries_now.setdefault(inj["team"], []).append(
            {k: inj.get(k) for k in ("player", "status", "injury", "return_date", "comment")})

    payload = {
        "season": f"{season - 1}-{str(season)[2:]}",
        "today": today,
        "generated": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_stats": metrics,
        "teams": {t: {"name": espn.TEAM_NAMES[t], "logo": espn.logo_url(t), "conf": espn.CONFERENCE[t]}
                  for t in espn.TEAM_NAMES},
        "games": games,
        "standings": projected_standings(games),
        "accuracy": accuracy(games),
        "injuries": injuries_now,
        "changes": _load_json(CHANGES_PATH, [])[:CHANGES_SHOWN],
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    print(f"Saved {len(games)} games ({len(raw_preds)} predicted) -> {OUT_PATH}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)

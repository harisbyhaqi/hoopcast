"""
Season forecaster — run by GitHub Actions every morning and evening.

Re-predicts every remaining game of the season, so the whole calendar shifts as results,
rosters and injuries change.

Each game blends two views of team strength:
  * Roster rating (ratings.py): who is on the team *now* — player BPM over the projected rotation,
    blended 50/50 with sportsbook win totals preseason. Doesn't care how the franchise did last year.
  * Current form (the Random Forest): this season's results only — rolling scoring, shooting,
    defense, rest, home/away record.
  Form weight = games played / (games played + 20): ~0 on opening night, ~50% after 20 games.

Injuries: injured players' minutes go to the rest of the roster and the rotation's BPM is
recomputed, so losing a star costs more than losing a bench player. Out/Doubtful apply until
ESPN's return date (else the next 3 games); Questionable / Day-to-day the next game only, at
reduced minutes.

Data sources:
  * Schedule          → data/schedule.csv (refreshed from ESPN by data_download.py)
  * Team form         → data/raw/games_raw.csv (this season's games)
  * Rosters / ratings → data/rosters.json (rosters.py), data/bpm/ + data/preseason_win_totals.csv (ratings.py)
  * Injury report     → data/injuries.json (rosters.py)
  * Roster changes    → data/changes.json (rosters.py)
  * Model accuracy    → models/metrics.json (train.py)

Writes docs/season.json: calendar, projected standings, accuracy, team ratings, injuries, changes.
Freezes today's pre-tip predictions in data/frozen_predictions.csv for scoring.

Usage: python src/update_daily.py [YYYY-MM-DD]   (defaults to today, Eastern time)
"""
import os, sys, json
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import espn
import ratings
from preprocess import clean_games
from predict import load_model, predict_games
from simulate import win_probability, DEFAULT_MARGIN_SIGMA
from utils import season_of, team_states, FEATURE_COLS

ROOT          = os.path.join(os.path.dirname(__file__), "..")
SCHEDULE_PATH = os.path.join(ROOT, "data", "schedule.csv")
METRICS_PATH  = os.path.join(ROOT, "models", "metrics.json")
INJURIES_PATH = os.path.join(ROOT, "data", "injuries.json")
CHANGES_PATH  = os.path.join(ROOT, "data", "changes.json")
FROZEN_PATH   = os.path.join(ROOT, "data", "frozen_predictions.csv")
OUT_PATH      = os.path.join(ROOT, "docs", "season.json")

FORM_GAMES          = 20     # games until current form and roster rating count equally
OUT_DEFAULT_GAMES   = 3      # Out/Doubtful with no return date: apply to the team's next N games
INJURY_NOTE_MINUTES = 12     # only list injured players who'd normally play this much
INJURY_NOTE_MIN_PTS = 0.5    # ...and only when the team's injuries cost at least this much
DEFAULT_HCA         = 2.5
DEFAULT_TEAM_PTS    = 114.0
CHANGES_SHOWN       = 150


# ---------------------------------------------------------------------------
# Schedule, rest & league context
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


def league_context(games_df: pd.DataFrame) -> tuple[float, float]:
    """(home-court advantage in points, average team points) over the last ~2 seasons of regular-season games."""
    reg = games_df[games_df["SEASON_ID"].astype(str).str.startswith("2")].sort_values("GAME_DATE").tail(2 * 2460)
    if reg.empty:
        return DEFAULT_HCA, DEFAULT_TEAM_PTS
    hca = float(reg.loc[reg["home"] == 1, "PLUS_MINUS"].mean())
    recent = reg.tail(2460)   # last season's scoring environment
    return round(hca, 2), round(float(recent["PTS"].mean()), 1)


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


def injury_notes(team: str, game_injuries: list[dict], team_ratings: ratings.TeamRatings) -> list[dict]:
    """Injured players who'd normally be in the rotation, for display."""
    usual = {x["player"]: x["min"] for x in ratings.rotation(team_ratings.rosters.get(team, []), team_ratings.table)}
    notes = [{"player": i["player"], "status": i["status"], "min": round(usual[i["player"]])}
             for i in game_injuries
             if i["team"] == team and usual.get(i["player"], 0) >= INJURY_NOTE_MINUTES]
    return sorted(notes, key=lambda n: -n["min"])


# ---------------------------------------------------------------------------
# Blend
# ---------------------------------------------------------------------------

def blend(model_scores: tuple[float, float] | None, games_played: float,
          rating_home: float, rating_away: float, hca: float, league_pts: float,
          injury_margin: float = 0.0, h2h_margin: float = 0.0) -> dict:
    """
    Combine current form (model_scores = (home, away), None before either team has played)
    with roster ratings. Returns predicted scores and the form weight used.
    Injuries and H2H shift the margin regardless of the blend.
    """
    w = games_played / (games_played + FORM_GAMES) if model_scores else 0.0
    rating_margin = rating_home - rating_away + hca
    if model_scores:
        model_margin, model_total = model_scores[0] - model_scores[1], model_scores[0] + model_scores[1]
    else:
        model_margin = model_total = 0.0
    margin = w * model_margin + (1 - w) * rating_margin + injury_margin + h2h_margin
    total = w * model_total + (1 - w) * 2 * league_pts
    return {"home": round(total / 2 + margin / 2, 1), "away": round(total / 2 - margin / 2, 1), "form_weight": round(w, 3)}


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

    past = games_df[games_df["GAME_DATE"] < today]
    states = team_states(past, today, season_id)
    season_games = past[past["SEASON_ID"].map(season_of) == season - 1]
    played = season_games.groupby("TEAM_ABBREVIATION").size().to_dict()
    hca, league_pts = league_context(past)
    print(f"Home-court advantage {hca:+.1f} pts, league average {league_pts} pts per team.")

    team_ratings = ratings.build(season, refresh_current=False)   # ratings.py refreshes the BPM cache
    injuries = load_injuries()
    game_injuries = injuries_by_game(injuries, upcoming)

    # Current-form model, only where both teams have played this season
    form_cols = [c for c in FEATURE_COLS if not c.startswith("opp_")]
    has_form = {t: not states.loc[t, form_cols].isna().any() for t in states.index}
    pairs, form_ids = [], []
    for _, g in upcoming.iterrows():
        if has_form.get(g["home"]) and has_form.get(g["away"]):
            feat_home = {**states.loc[g["home"]].to_dict(), "home": 1, "rest_days": rest_days(schedule, g["home"], g["date"])}
            feat_away = {**states.loc[g["away"]].to_dict(), "home": 0, "rest_days": rest_days(schedule, g["away"], g["date"])}
            pairs.append((feat_home, feat_away))
            form_ids.append(g["event_id"])
    form_preds = dict(zip(form_ids, predict_games(pairs, model)))
    print(f"Form model used for {len(form_preds)} game(s); roster ratings for the rest.")

    games = []
    for _, g in season_sched.iterrows():
        eid, away, home = g["event_id"], g["away"], g["home"]
        entry = {"id": eid, "date": g["date"], "tip": g["tip_utc"], "type": g["season_type"],
                 "home": home, "away": away, "status": g["status"],
                 "pred": None, "result": None, "frozen": None, "injuries": None}
        if g["status"] == "final":
            entry["result"] = {"home": int(g["home_pts"]), "away": int(g["away_pts"])}
        elif eid in upcoming["event_id"].values:
            inj = game_injuries.get(eid, [])
            inj_home, inj_away = team_ratings.injury_delta(home, inj), team_ratings.injury_delta(away, inj)
            h2h_away, h2h_home = compute_h2h_offsets(away, home, season_games)
            pred = blend(form_preds.get(eid), (played.get(home, 0) + played.get(away, 0)) / 2,
                         team_ratings.prior[home], team_ratings.prior[away], hca, league_pts,
                         injury_margin=inj_home - inj_away, h2h_margin=h2h_home - h2h_away)
            p_home = win_probability(pred["home"], pred["away"], sigma)
            entry["pred"] = {"home": pred["home"], "away": pred["away"], "p_home": round(p_home * 100, 1),
                             "form_weight": pred["form_weight"]}
            notes_home = injury_notes(home, inj, team_ratings) if inj_home <= -INJURY_NOTE_MIN_PTS else []
            notes_away = injury_notes(away, inj, team_ratings) if inj_away <= -INJURY_NOTE_MIN_PTS else []
            if notes_home or notes_away:
                entry["injuries"] = {"home": notes_home, "away": notes_away,
                                     "home_delta": round(inj_home, 1), "away_delta": round(inj_away, 1)}
        games.append(entry)

    frozen = freeze(load_frozen(), games, today, now_utc)
    os.makedirs(os.path.dirname(FROZEN_PATH), exist_ok=True)
    frozen.to_csv(FROZEN_PATH, index=False)
    frozen_by_id = {f["event_id"]: {"home": f["pred_home"], "away": f["pred_away"], "p_home": f["p_home"]}
                    for _, f in frozen.iterrows()}
    for entry in games:
        entry["frozen"] = frozen_by_id.get(entry["id"])

    next_day = min((e["date"] for e in games if e["pred"]), default=None)
    for entry in (e for e in games if e["date"] == next_day and e["pred"]):
        p = entry["pred"]
        print(f"  {next_day}  {entry['away']} {p['away']} @ {entry['home']} {p['home']}  |  "
              f"{entry['home']} wins {p['p_home']}%  (form weight {p['form_weight']})")

    injuries_now = {}
    for inj in sorted(injuries, key=lambda i: (i["team"], i["player"])):
        injuries_now.setdefault(inj["team"], []).append(
            {k: inj.get(k) for k in ("player", "status", "injury", "return_date", "comment")})

    payload = {
        "season": f"{season - 1}-{str(season)[2:]}",
        "today": today,
        "generated": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_stats": metrics,
        "context": {"hca": hca, "league_pts": league_pts, "form_games": FORM_GAMES},
        "teams": {t: {"name": espn.TEAM_NAMES[t], "logo": espn.logo_url(t), "conf": espn.CONFERENCE[t],
                      "played": int(played.get(t, 0)), **team_ratings.summary(t)}
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
    print(f"Saved {len(games)} games ({len(upcoming)} predicted) -> {OUT_PATH}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)

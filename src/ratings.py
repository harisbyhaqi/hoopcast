"""
Roster-based team strength: who is actually on each team *now*, not how the franchise did last year.

Player ratings are DARKO DPM (Daily Plus-Minus, www.darko.app by Kostya Medvedovsky): points per 100
possessions vs. an average player, built from box scores, on/off impact and tracking data, split
into offense (O-DPM) and defense (D-DPM), and re-estimated after every game. So defense that never
shows up as a steal or block still counts, and players who are playing badly (or well) move daily.
DARKO's projected minutes per game give the depth chart.

Fallback for players DARKO doesn't list (and for everyone if darko.app can't be reached):
Box Plus/Minus — Basketball-Reference for finished seasons (cached in data/bpm/), and for the
current season an estimate from our own ESPN box scores (BoxBPM) — mapped onto the DPM scale.
Players with neither (rookies, overseas signings) get a rookie default.

Team rating = minutes-weighted DPM of the projected rotation on the current ESPN roster
≈ net rating per 100 possessions. Preseason it's blended 50/50 with sportsbook win totals
(data/preseason_win_totals.csv), which see things ratings can't (rookies, new roles).

Injured players' minutes go to the next players on the depth chart, so availability changes the
rating per game.

Outputs:
  data/darko.csv              last successful DARKO pull (used if the site can't be reached)
  data/bpm/NBA_<year>.csv     cached Basketball-Reference advanced stats (finished seasons, fetched once)
  data/team_ratings.json      per-team roster rating, market rating, blended prior, rotation
"""
import io
import json
import os
import re
import time
import unicodedata
from datetime import datetime

import pandas as pd
import requests

import espn

ROOT          = os.path.join(os.path.dirname(__file__), "..")
BPM_DIR       = os.path.join(ROOT, "data", "bpm")
ROSTERS_PATH  = os.path.join(ROOT, "data", "rosters.json")
PLAYER_PATH   = os.path.join(ROOT, "data", "raw", "player_games.csv")
TOTALS_PATH   = os.path.join(ROOT, "data", "preseason_win_totals.csv")
RATINGS_PATH  = os.path.join(ROOT, "data", "team_ratings.json")
DARKO_PATH    = os.path.join(ROOT, "data", "darko.csv")
BR_URL        = "https://www.basketball-reference.com/leagues/NBA_{year}_advanced.html"
DARKO_URL     = "https://www.darko.app/__data.json"

SEASON_WEIGHTS     = (6, 5, 4, 3)   # current season first; weights multiply minutes played
REPLACEMENT_BPM    = -2.0           # what a freely available fill-in player is worth
PRIOR_MINUTES      = 1000           # weighted minutes of replacement level mixed into every player
ROOKIE_RATING      = -1.5           # no NBA history (rookies, overseas signings), DPM scale
ROOKIE_MPG         = 14.0
REPLACEMENT_RATING = -2.0           # fills minutes when a roster runs out, DPM scale
MAX_MPG            = 38.0
WINS_PER_NET_POINT = 2.7            # ≈ wins added per +1 net rating over 82 games
MARKET_WEIGHT      = 0.5            # share of the preseason prior that comes from win totals

# Minutes by rating rank among available players (best player first); sums to 240.
# Projected minutes = half this, half the player's usual minutes over recent seasons.
MINUTES_CURVE = [34, 33, 31, 29, 27, 24, 20, 16, 12, 8, 6]
CURVE_WEIGHT  = 0.5

# Share of a player's usual minutes expected given his injury status
AVAILABILITY = {"out": 0.0, "doubtful": 0.0, "questionable": 0.5, "day-to-day": 0.8}


# ESPN name → DARKO / Basketball-Reference name, where they differ beyond accents/punctuation/suffixes
# (applied to every source, so all spellings land on one key)
NAME_ALIASES = {
    "ronald holland": "ron holland",
    "bub carrington": "carlton carrington",
    "alex sarr": "alexandre sarr",
    "nic claxton": "nicolas claxton",
}


def name_key(name: str) -> str:
    """'Nikola Jokić' / 'Jabari Smith Jr.' → 'nikola jokic' / 'jabari smith' for cross-site matching."""
    n = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    n = re.sub(r"[.'’]", "", n)          # A.J. → aj, De'Aaron → deaaron
    n = n.replace("-", " ")
    n = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", n)
    n = " ".join(n.split())
    return NAME_ALIASES.get(n, n)


# ---------------------------------------------------------------------------
# Basketball-Reference BPM
# ---------------------------------------------------------------------------

EMPTY = pd.DataFrame(columns=["player", "g", "mp", "bpm"])


def parse_advanced(html: str) -> pd.DataFrame:
    if 'id="advanced"' not in html:
        return EMPTY.copy()   # season page exists but no games played yet
    df = pd.read_html(io.StringIO(html), attrs={"id": "advanced"}, flavor="lxml")[0]
    df = df[(df["Player"] != "Player") & (df["Player"] != "League Average")]
    # Traded players: the first row is the season total ("2TM"/"TOT"), then one row per team
    df = df.drop_duplicates("Player", keep="first")
    out = pd.DataFrame({
        "player": df["Player"],
        "g": pd.to_numeric(df["G"], errors="coerce"),
        "mp": pd.to_numeric(df["MP"], errors="coerce"),
        "bpm": pd.to_numeric(df["BPM"], errors="coerce"),
    }).dropna()
    return out[out["mp"] > 0]


def load_bpm(year: int) -> pd.DataFrame:
    """Basketball-Reference advanced stats for a finished season (ending in `year`); fetched once, then cached."""
    path = os.path.join(BPM_DIR, f"NBA_{year}.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    try:
        resp = requests.get(BR_URL.format(year=year), headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        if resp.status_code == 404:
            return EMPTY.copy()   # season hasn't started
        resp.raise_for_status()
        df = parse_advanced(resp.content.decode("utf-8"))   # requests guesses Latin-1 and mangles Jokić
        if df.empty:
            return df
        os.makedirs(BPM_DIR, exist_ok=True)
        df.to_csv(path, index=False)
        time.sleep(3)   # Basketball-Reference allows ~20 requests/minute
        return df
    except Exception as e:
        print(f"  WARNING: Basketball-Reference fetch failed for {year} ({e}); using cache if present.")
        return pd.read_csv(path) if os.path.exists(path) else EMPTY.copy()


# ---------------------------------------------------------------------------
# ESPN box-score BPM (current season — Basketball-Reference blocks GitHub Actions)
# ---------------------------------------------------------------------------

BOX_STATS = ["pts", "fga", "fta", "fg3m", "oreb", "dreb", "ast", "stl", "blk", "tov", "pf", "plus_minus"]
MIN_FIT_MINUTES = 500   # players with less are too noisy to calibrate on


def espn_season_totals(season: int, player_path: str = PLAYER_PATH) -> pd.DataFrame:
    """Regular-season per-player totals from our ESPN box scores: player, g, mp, BOX_STATS."""
    if not os.path.exists(player_path):
        return pd.DataFrame(columns=["player", "g", "mp"] + BOX_STATS)
    df = pd.read_csv(player_path, dtype={"GAME_ID": str, "player_id": str})
    if "fga" not in df.columns:
        return pd.DataFrame(columns=["player", "g", "mp"] + BOX_STATS)
    df = df[(df["SEASON_ID"] == int(f"2{season - 1}")) & (df["min"] > 0)]
    if df.empty:
        return pd.DataFrame(columns=["player", "g", "mp"] + BOX_STATS)
    df = df.assign(pid=df["player_id"].fillna("").replace("", None).fillna(df["player"]))
    g = df.sort_values("GAME_DATE").groupby("pid")
    totals = g[BOX_STATS + ["min"]].sum().rename(columns={"min": "mp"})
    totals["g"] = g.size()
    totals["player"] = g["player"].last()
    return totals.reset_index(drop=True)


def per36(totals: pd.DataFrame) -> pd.DataFrame:
    return totals[BOX_STATS].div(totals["mp"], axis=0) * 36


class BoxBPM:
    """
    Linear map from per-36 box-score stats (incl. plus/minus) to Basketball-Reference BPM,
    fitted on a past season where we have both. Lets us rate players on this season's ESPN
    box scores with BPM's scale and meaning.
    """

    def __init__(self, season: int):
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import cross_val_score

        espn_tot = espn_season_totals(season)
        br = load_bpm(season)
        espn_tot["key"] = espn_tot["player"].map(name_key)
        br = br.assign(key=br["player"].map(name_key))
        m = espn_tot.merge(br[["key", "bpm"]], on="key")
        m = m[m["mp"] >= MIN_FIT_MINUTES]
        if len(m) < 100:
            raise ValueError(f"only {len(m)} players to calibrate on for {season}")
        X, y, w = per36(m), m["bpm"], m["mp"]
        self.model = Ridge(alpha=1.0).fit(X, y, sample_weight=w)
        self.season, self.n = season, len(m)
        self.cv_r2 = float(cross_val_score(Ridge(alpha=1.0), X, y, cv=5, scoring="r2").mean())

    def predict(self, totals: pd.DataFrame) -> pd.DataFrame:
        """BPM-format table (player, g, mp, bpm) for the given ESPN season totals."""
        if totals.empty:
            return EMPTY.copy()
        bpm = self.model.predict(per36(totals)).clip(-10, 15)
        return pd.DataFrame({"player": totals["player"], "g": totals["g"], "mp": totals["mp"], "bpm": bpm})


def calibrated_box_bpm(season: int) -> "BoxBPM | None":
    """Fit on the most recent past season that has both Basketball-Reference and ESPN data."""
    for year in range(season - 1, season - len(SEASON_WEIGHTS), -1):
        if not os.path.exists(os.path.join(BPM_DIR, f"NBA_{year}.csv")):
            continue
        try:
            return BoxBPM(year)
        except ValueError:
            continue
    return None


def bpm_table(season: int) -> pd.DataFrame:
    """
    Blended BPM and minutes per game for every player, indexed by name_key.
    Past seasons: Basketball-Reference BPM (cached). Current season — and any past season BR
    isn't available for — BPM estimated from our ESPN box scores via BoxBPM.
    """
    box = None
    frames = []
    for i, w in enumerate(SEASON_WEIGHTS):
        year = season - i
        df = load_bpm(year) if i > 0 else EMPTY.copy()
        if df.empty:
            totals = espn_season_totals(year)
            if not totals.empty:
                box = box or calibrated_box_bpm(season)
                if box:
                    df = box.predict(totals)
        if not df.empty:
            frames.append(df.assign(w=w))
    if not frames:
        return pd.DataFrame(columns=["bpm", "mpg"])
    a = pd.concat(frames, ignore_index=True)
    a["key"] = a["player"].map(name_key)
    a = a.assign(wmp=a["mp"] * a["w"], wbpm=a["mp"] * a["w"] * a["bpm"], wg=a["g"] * a["w"])
    g = a.groupby("key").agg(wmp=("wmp", "sum"), wbpm=("wbpm", "sum"), wg=("wg", "sum"))
    return pd.DataFrame({
        "bpm": (g["wbpm"] + REPLACEMENT_BPM * PRIOR_MINUTES) / (g["wmp"] + PRIOR_MINUTES),
        "mpg": (g["wmp"] / g["wg"]).clip(upper=MAX_MPG),
    })


# ---------------------------------------------------------------------------
# DARKO DPM (primary)
# ---------------------------------------------------------------------------

DARKO_COLS = {"player_name": "player", "team_name": "team", "dpm": "dpm", "o_dpm": "o_dpm",
              "d_dpm": "d_dpm", "x_minutes": "mpg", "season_games": "games", "season": "season"}


def parse_darko(payload: dict) -> pd.DataFrame:
    """Decode darko.app's SvelteKit data payload (devalue-flattened) into one row per player."""
    data = next(n for n in reversed(payload["nodes"]) if n and n.get("type") == "data")["data"]

    def resolve(i):
        if isinstance(i, int) and i < 0:
            return None
        v = data[i]
        if isinstance(v, list):
            return [resolve(x) for x in v]
        if isinstance(v, dict):
            return {k: resolve(x) for k, x in v.items()}
        return v

    players = resolve(0)["players"]
    df = pd.DataFrame(dict(zip(players["keys"], players["values"])) if isinstance(players, dict) else players)
    df = df[[c for c in DARKO_COLS if c in df.columns]].rename(columns=DARKO_COLS)
    for c in ("dpm", "o_dpm", "d_dpm", "mpg"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["player", "dpm"])


def load_darko() -> pd.DataFrame:
    """Today's DARKO ratings; falls back to the last successful pull."""
    try:
        resp = requests.get(DARKO_URL, headers={"User-Agent": "Mozilla/5.0 (nba-predictor; daily)"}, timeout=30)
        resp.raise_for_status()
        df = parse_darko(resp.json())
        if len(df) < 300:
            raise ValueError(f"only {len(df)} players in response")
        os.makedirs(os.path.dirname(DARKO_PATH), exist_ok=True)
        df.to_csv(DARKO_PATH, index=False)
        return df
    except Exception as e:
        print(f"  WARNING: DARKO fetch failed ({e}); using last saved ratings if present.")
        return pd.read_csv(DARKO_PATH) if os.path.exists(DARKO_PATH) else pd.DataFrame()


def player_table(season: int) -> pd.DataFrame:
    """
    Rating (DPM scale) and projected minutes per game for every player, indexed by name_key.
    Columns: rating, mpg, o, d (offense/defense split, DARKO only), source ("darko" / "bpm").
    """
    bpm = bpm_table(season)
    darko = load_darko()
    if darko.empty:
        return pd.DataFrame({"rating": bpm["bpm"], "mpg": bpm["mpg"], "o": None, "d": None, "source": "bpm"})

    darko = darko.assign(key=darko["player"].map(name_key)).drop_duplicates("key").set_index("key")
    # Usual minutes: multi-season average where we have it (DARKO's projection follows the end of
    # last season, e.g. 17 mpg for a star who was hurt in March).
    mpg = bpm["mpg"].reindex(darko.index).fillna(darko["mpg"]).clip(upper=MAX_MPG)
    table = pd.DataFrame({"rating": darko["dpm"], "mpg": mpg,
                          "o": darko["o_dpm"], "d": darko["d_dpm"], "source": "darko"})

    # Players DARKO doesn't list: BPM mapped onto the DPM scale (fit on players in both)
    both = bpm.index.intersection(table.index)
    rest = bpm.drop(index=both, errors="ignore")
    if len(both) >= 50 and not rest.empty:
        import numpy as np
        slope, intercept = np.polyfit(bpm.loc[both, "bpm"], table.loc[both, "rating"], 1)
        extra = pd.DataFrame({"rating": slope * rest["bpm"] + intercept, "mpg": rest["mpg"],
                              "o": None, "d": None, "source": "bpm"})
        table = pd.concat([table, extra])
    return table


# ---------------------------------------------------------------------------
# Team ratings
# ---------------------------------------------------------------------------

def rotation(roster: list[dict], table: pd.DataFrame, availability: dict[str, float] | None = None) -> list[dict]:
    """
    Projected rotation filling 240 minutes. Each available player's minutes are half his usual
    minutes over recent seasons and half the slot his rating earns on this roster (MINUTES_CURVE),
    so a star coming back from an injured season still starts and end-of-season garbage-time
    minutes don't make a bench player a starter. Injured players drop out, everyone below moves
    up a slot, and if the roster runs out a replacement-level player fills in.
    """
    availability = availability or {}
    players = []
    for p in roster:
        key = name_key(p["player"])
        a = availability.get(key, 1.0)
        if a <= 0:
            continue
        if key in table.index:
            rating, usual = float(table.at[key, "rating"]), float(table.at[key, "mpg"])
            o, d = table.at[key, "o"], table.at[key, "d"]
        else:
            rating, usual, o, d = ROOKIE_RATING, ROOKIE_MPG, None, None
        players.append({"player": p["player"], "rating": rating, "o": o, "d": d, "usual": usual, "avail": a})

    players.sort(key=lambda x: -x["rating"])
    for rank, x in enumerate(players):
        slot = MINUTES_CURVE[rank] if rank < len(MINUTES_CURVE) else 0.0
        x["mpg"] = ((1 - CURVE_WEIGHT) * x["usual"] + CURVE_WEIGHT * slot) * x["avail"]
    players.sort(key=lambda x: -x["mpg"])

    remaining, rot = 240.0, []
    for x in players:
        if remaining <= 0 or x["mpg"] <= 0:
            break
        x["min"] = min(x["mpg"], remaining)
        remaining -= x["min"]
        rot.append(x)
    if remaining > 0:
        rot.append({"player": "Replacement", "rating": REPLACEMENT_RATING, "o": None, "d": None,
                    "mpg": remaining, "min": remaining})
    return rot


def roster_net(roster: list[dict], table: pd.DataFrame, availability: dict[str, float] | None = None) -> float:
    """≈ team net rating per 100 possessions from the projected rotation's player ratings."""
    return sum(x["rating"] * x["min"] for x in rotation(roster, table, availability)) / 48


def market_net(wins: float) -> float:
    return (wins - 41) / WINS_PER_NET_POINT


def availability_from(injuries: list[dict]) -> dict[str, float]:
    return {name_key(i["player"]): AVAILABILITY.get(i["status"], 1.0) for i in injuries}


class TeamRatings:
    """Preseason prior per team plus availability-adjusted ratings for specific games."""

    def __init__(self, rosters: dict[str, list[dict]], table: pd.DataFrame, win_totals: dict[str, float]):
        self.rosters, self.table = rosters, table
        healthy = {t: roster_net(rosters.get(t, []), table) for t in espn.TEAM_NAMES}
        mean = sum(healthy.values()) / len(healthy)
        self.healthy = healthy
        self.roster_c = {t: v - mean for t, v in healthy.items()}   # league-average team = 0
        self.market = {t: market_net(win_totals[t]) for t in win_totals}
        self.prior = {
            t: (MARKET_WEIGHT * self.market[t] + (1 - MARKET_WEIGHT) * self.roster_c[t]) if t in self.market
            else self.roster_c[t]
            for t in espn.TEAM_NAMES
        }

    def injury_delta(self, team: str, injuries: list[dict]) -> float:
        """Rating change from the listed injuries (≤ 0): rotation without them vs. healthy rotation."""
        team_inj = [i for i in injuries if i["team"] == team]
        if not team_inj:
            return 0.0
        delta = roster_net(self.rosters.get(team, []), self.table, availability_from(team_inj)) - self.healthy[team]
        # Reshuffling minutes can nudge the number up; an injury never makes a team better.
        return min(delta, 0.0)

    def summary(self, team: str) -> dict:
        rot = rotation(self.rosters.get(team, []), self.table)
        return {
            "roster": round(self.roster_c[team], 2),
            "market": round(self.market[team], 2) if team in self.market else None,
            "prior": round(self.prior[team], 2),
            "rotation": [{"player": x["player"], "rating": round(x["rating"], 1),
                          "o": None if pd.isna(x["o"]) else round(float(x["o"]), 1),
                          "d": None if pd.isna(x["d"]) else round(float(x["d"]), 1),
                          "min": round(x["min"], 1)} for x in rot],
        }


def load_win_totals(path: str = TOTALS_PATH) -> dict[str, float]:
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    return dict(zip(df["team"], df["wins"]))


def load_rosters(path: str = ROSTERS_PATH) -> dict[str, list[dict]]:
    with open(path) as f:
        return json.load(f)["teams"]


def build(season: int) -> TeamRatings:
    return TeamRatings(load_rosters(), player_table(season), load_win_totals())


if __name__ == "__main__":
    season = espn.season_end_year(datetime.now(espn.ET).date())
    box = calibrated_box_bpm(season)
    if box:
        print(f"Box-score BPM calibrated on {box.season - 1}-{str(box.season)[2:]}: "
              f"{box.n} players, cross-validated R² {box.cv_r2:.2f}")
    ratings = build(season)
    teams = {t: ratings.summary(t) for t in espn.TEAM_NAMES}

    unmatched = sorted(
        p["player"] for t in espn.TEAM_NAMES for p in ratings.rosters.get(t, [])
        if name_key(p["player"]) not in ratings.table.index
    )
    with open(RATINGS_PATH, "w") as f:
        json.dump({"generated": datetime.now(espn.ET).isoformat(timespec="seconds"),
                   "unmatched": unmatched, "teams": teams}, f, indent=1)

    print(f"{'team':<5}{'roster':>8}{'market':>8}{'prior':>8}")
    for t, r in sorted(teams.items(), key=lambda kv: -kv[1]["prior"]):
        print(f"{t:<5}{r['roster']:>8.1f}{(r['market'] if r['market'] is not None else float('nan')):>8.1f}{r['prior']:>8.1f}")
    src = ratings.table["source"].value_counts().to_dict()
    print(f"Player ratings: {src.get('darko', 0)} from DARKO DPM, {src.get('bpm', 0)} from BPM fallback.")
    print(f"{len(unmatched)} roster players without NBA history (rated as rookies) -> {RATINGS_PATH}")

"""
Roster-based team strength: who is actually on each team *now*, not how the franchise did last year.

Player ratings come from Basketball-Reference Box Plus/Minus (points per 100 possessions vs. an
average player), blended over the last few seasons and regressed toward replacement level by minutes
so stars coming off injured seasons aren't written off. The current season gets the most weight, so
players who are playing badly (or well) move their rating as the season goes.

Team rating = minutes-weighted BPM of the projected 10-man rotation on the current ESPN roster,
≈ net rating per 100 possessions. Preseason it's blended 50/50 with sportsbook win totals
(data/preseason_win_totals.csv), which see things box scores can't (rookies, new roles).

Injured players' minutes go to the rest of the roster, so availability changes the rating per game.

Outputs:
  data/bpm/NBA_<year>.csv     cached Basketball-Reference advanced stats (past seasons fetched once)
  data/team_ratings.json      per-team roster rating, market rating, blended prior, top rotation
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
TOTALS_PATH   = os.path.join(ROOT, "data", "preseason_win_totals.csv")
RATINGS_PATH  = os.path.join(ROOT, "data", "team_ratings.json")
BR_URL        = "https://www.basketball-reference.com/leagues/NBA_{year}_advanced.html"

SEASON_WEIGHTS     = (6, 5, 4, 3)   # current season first; weights multiply minutes played
REPLACEMENT_BPM    = -2.0           # what a freely available fill-in player is worth
PRIOR_MINUTES      = 1000           # weighted minutes of replacement level mixed into every player
ROOKIE_BPM         = -1.5           # no NBA history (rookies, overseas signings)
ROOKIE_MPG         = 14.0
MAX_MPG            = 38.0
WINS_PER_NET_POINT = 2.7            # ≈ wins added per +1 net rating over 82 games
MARKET_WEIGHT      = 0.5            # share of the preseason prior that comes from win totals

# Share of a player's usual minutes expected given his injury status
AVAILABILITY = {"out": 0.0, "doubtful": 0.0, "questionable": 0.5, "day-to-day": 0.8}


# ESPN name → Basketball-Reference name, where they differ beyond accents/punctuation/suffixes
NAME_ALIASES = {"ronald holland": "ron holland"}


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


def load_bpm(year: int, refresh: bool) -> pd.DataFrame:
    """Advanced stats for the season ending in `year`; cached on disk, re-fetched only if `refresh`."""
    path = os.path.join(BPM_DIR, f"NBA_{year}.csv")
    if os.path.exists(path) and not refresh:
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


def player_table(season: int, refresh_current: bool = True) -> pd.DataFrame:
    """Blended BPM and projected minutes per game for every player, indexed by name_key."""
    frames = []
    for i, w in enumerate(SEASON_WEIGHTS):
        year = season - i
        df = load_bpm(year, refresh=refresh_current and i == 0)
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
# Team ratings
# ---------------------------------------------------------------------------

def rotation(roster: list[dict], table: pd.DataFrame, availability: dict[str, float] | None = None) -> list[dict]:
    """
    Projected rotation filling 240 minutes in depth-chart order (usual minutes per game × availability).
    An injured player's minutes go to the next players down the depth chart — not to the starters —
    and if the roster runs out, to a replacement-level player.
    """
    availability = availability or {}
    players = []
    for p in roster:
        key = name_key(p["player"])
        if key in table.index:
            bpm, mpg = float(table.at[key, "bpm"]), float(table.at[key, "mpg"])
        else:
            bpm, mpg = ROOKIE_BPM, ROOKIE_MPG
        mpg *= availability.get(key, 1.0)
        if mpg > 0:
            players.append({"player": p["player"], "bpm": bpm, "mpg": mpg})
    players.sort(key=lambda x: -x["mpg"])

    remaining, rot = 240.0, []
    for x in players:
        if remaining <= 0:
            break
        x["min"] = min(x["mpg"], remaining)
        remaining -= x["min"]
        rot.append(x)
    if remaining > 0:
        rot.append({"player": "Replacement", "bpm": REPLACEMENT_BPM, "mpg": remaining, "min": remaining})
    return rot


def roster_net(roster: list[dict], table: pd.DataFrame, availability: dict[str, float] | None = None) -> float:
    """≈ team net rating per 100 possessions from the projected rotation's BPM."""
    return sum(x["bpm"] * x["min"] for x in rotation(roster, table, availability)) / 48


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
            "rotation": [{"player": x["player"], "bpm": round(x["bpm"], 1), "min": round(x["min"], 1)} for x in rot],
        }


def load_win_totals(path: str = TOTALS_PATH) -> dict[str, float]:
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    return dict(zip(df["team"], df["wins"]))


def load_rosters(path: str = ROSTERS_PATH) -> dict[str, list[dict]]:
    with open(path) as f:
        return json.load(f)["teams"]


def build(season: int, refresh_current: bool = True) -> TeamRatings:
    return TeamRatings(load_rosters(), player_table(season, refresh_current), load_win_totals())


if __name__ == "__main__":
    season = espn.season_end_year(datetime.now(espn.ET).date())
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
    print(f"{len(unmatched)} roster players without NBA history (rated as rookies) -> {RATINGS_PATH}")

"""
Thin client for ESPN's public NBA JSON API (no key needed, works from GitHub Actions).

Used for: season schedule, box scores, injuries, rosters.
ESPN uses some non-standard abbreviations (NY, SA, GS, NO, UTAH, WSH); everything
here is normalised to the NBA tricodes used throughout the rest of the project.
"""
import re
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import requests

BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
ET = ZoneInfo("America/New_York")

TEAM_NAMES = {
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

# NBA.com team IDs (used for logos and the TEAM_ID column)
TEAM_IDS = {
    "ATL": 1610612737, "BOS": 1610612738, "BKN": 1610612751, "CHA": 1610612766,
    "CHI": 1610612741, "CLE": 1610612739, "DAL": 1610612742, "DEN": 1610612743,
    "DET": 1610612765, "GSW": 1610612744, "HOU": 1610612745, "IND": 1610612754,
    "LAC": 1610612746, "LAL": 1610612747, "MEM": 1610612763, "MIA": 1610612748,
    "MIL": 1610612749, "MIN": 1610612750, "NOP": 1610612740, "NYK": 1610612752,
    "OKC": 1610612760, "ORL": 1610612753, "PHI": 1610612755, "PHX": 1610612756,
    "POR": 1610612757, "SAC": 1610612758, "SAS": 1610612759, "TOR": 1610612761,
    "UTA": 1610612762, "WAS": 1610612764,
}

ESPN_TEAM_IDS = {
    "ATL": 1, "BOS": 2, "BKN": 17, "CHA": 30, "CHI": 4, "CLE": 5, "DAL": 6, "DEN": 7,
    "DET": 8, "GSW": 9, "HOU": 10, "IND": 11, "LAC": 12, "LAL": 13, "MEM": 29, "MIA": 14,
    "MIL": 15, "MIN": 16, "NOP": 3, "NYK": 18, "OKC": 25, "ORL": 19, "PHI": 20, "PHX": 21,
    "POR": 22, "SAC": 23, "SAS": 24, "TOR": 28, "UTA": 26, "WAS": 27,
}
ESPN_ID_TO_ABBR = {str(v): k for k, v in ESPN_TEAM_IDS.items()}

EAST = {"ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DET", "IND",
        "MIA", "MIL", "NYK", "ORL", "PHI", "TOR", "WAS"}
CONFERENCE = {abbr: ("East" if abbr in EAST else "West") for abbr in TEAM_NAMES}

_ESPN_ABBR_FIX = {"NY": "NYK", "SA": "SAS", "GS": "GSW", "NO": "NOP", "UTAH": "UTA", "WSH": "WAS"}

SEASON_TYPES = {2: "regular", 3: "playoffs"}


def to_nba_abbr(espn_abbr: str) -> str:
    return _ESPN_ABBR_FIX.get(espn_abbr, espn_abbr)


_NBA_TO_ESPN_ABBR = {v: k for k, v in _ESPN_ABBR_FIX.items()}


def logo_url(abbr: str) -> str:
    # ESPN's CDN (cdn.nba.com logos are hit-or-miss behind its bot protection)
    return f"https://a.espncdn.com/i/teamlogos/nba/500/{_NBA_TO_ESPN_ABBR.get(abbr, abbr).lower()}.png"


def season_end_year(today) -> int:
    """ESPN labels seasons by end year: 2026-27 → 2027. New season from October."""
    return today.year + 1 if today.month >= 10 else today.year


def get_json(path: str, params: dict | None = None, retries: int = 3) -> dict:
    url = path if path.startswith("http") else f"{BASE}/{path}"
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------

def _score(competitor: dict):
    s = competitor.get("score")
    if isinstance(s, dict):
        s = s.get("value")
    return int(float(s)) if s not in (None, "") else None


def parse_event(event: dict) -> dict | None:
    comp = event["competitions"][0]
    sides = {c["homeAway"]: c for c in comp["competitors"]}
    if "home" not in sides or "away" not in sides:
        return None
    home = to_nba_abbr(sides["home"]["team"]["abbreviation"])
    away = to_nba_abbr(sides["away"]["team"]["abbreviation"])
    if home not in TEAM_NAMES or away not in TEAM_NAMES:
        return None  # All-Star / exhibition vs non-NBA teams
    tip_utc = datetime.fromisoformat(event["date"].replace("Z", "+00:00"))
    status = comp["status"]["type"]
    final = bool(status.get("completed")) and status.get("state") == "post"
    stype = (event.get("seasonType") or {}).get("type") or comp.get("type", {}).get("id")
    return {
        "event_id": str(event["id"]),
        "date": tip_utc.astimezone(ET).date().isoformat(),
        "tip_utc": tip_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "season_type": SEASON_TYPES.get(int(stype), "other") if stype else "other",
        "home": home,
        "away": away,
        "status": "final" if final else status.get("state", "pre"),
        "home_pts": _score(sides["home"]) if final else None,
        "away_pts": _score(sides["away"]) if final else None,
    }


def fetch_season_schedule(season: int, season_types=(2, 3)) -> list[dict]:
    """All games for a season (ESPN end-year label), deduped across the 30 team schedules."""
    games = {}
    for abbr, tid in ESPN_TEAM_IDS.items():
        for stype in season_types:
            data = get_json(f"teams/{tid}/schedule", {"season": season, "seasontype": stype})
            for ev in data.get("events", []):
                g = parse_event(ev)
                if g:
                    g["season"] = season
                    games[g["event_id"]] = g
        time.sleep(0.2)
    return sorted(games.values(), key=lambda g: (g["tip_utc"], g["event_id"]))


# ---------------------------------------------------------------------------
# Box scores
# ---------------------------------------------------------------------------

def _made_att(v: str) -> tuple[int, int]:
    m, a = v.split("-")
    return int(m), int(a)


def _to_int(v) -> int:
    try:
        return int(float(str(v).replace("+", "")))
    except ValueError:
        return 0


def parse_box_score(summary: dict) -> tuple[list[dict], list[dict]]:
    """
    Returns (team_rows, player_rows) for one finished game.
    team_rows: one dict per team with raw counting stats.
    player_rows: one dict per player who played.
    """
    box = summary["boxscore"]
    header = summary["header"]["competitions"][0]
    scores = {to_nba_abbr(c["team"]["abbreviation"]): _score(c) for c in header["competitors"]}

    team_rows = []
    for t in box["teams"]:
        abbr = to_nba_abbr(t["team"]["abbreviation"])
        s = {x["name"]: x["displayValue"] for x in t["statistics"]}
        fgm, fga = _made_att(s["fieldGoalsMade-fieldGoalsAttempted"])
        fg3m, fg3a = _made_att(s["threePointFieldGoalsMade-threePointFieldGoalsAttempted"])
        ftm, fta = _made_att(s["freeThrowsMade-freeThrowsAttempted"])
        team_rows.append({
            "abbr": abbr, "home_away": t.get("homeAway"), "pts": scores.get(abbr),
            "fgm": fgm, "fga": fga, "fg3m": fg3m, "fg3a": fg3a, "ftm": ftm, "fta": fta,
            "oreb": _to_int(s.get("offensiveRebounds", 0)), "dreb": _to_int(s.get("defensiveRebounds", 0)),
            "reb": _to_int(s.get("totalRebounds", 0)), "ast": _to_int(s.get("assists", 0)),
            "stl": _to_int(s.get("steals", 0)), "blk": _to_int(s.get("blocks", 0)),
            "tov": _to_int(s.get("totalTurnovers", s.get("turnovers", 0))),
            "pf": _to_int(s.get("fouls", 0)),
        })

    player_rows = []
    for team in box.get("players", []):
        abbr = to_nba_abbr(team["team"]["abbreviation"])
        for block in team.get("statistics", []):
            keys = block["keys"]
            for a in block.get("athletes", []):
                if a.get("didNotPlay") or not a.get("stats"):
                    continue
                st = dict(zip(keys, a["stats"]))
                ath = a.get("athlete", {})
                player_rows.append({
                    "player_id": str(ath.get("id", "")),   # ESPN occasionally ships an athlete with no id
                    "player": ath.get("displayName") or ath.get("shortName", ""),
                    "team": abbr,
                    "min": _to_int(st.get("minutes", 0)),
                    "pts": _to_int(st.get("points", 0)),
                })
    return team_rows, player_rows


def fetch_box_score(event_id: str) -> tuple[list[dict], list[dict]]:
    return parse_box_score(get_json("summary", {"event": event_id}))


# ---------------------------------------------------------------------------
# Injuries & rosters
# ---------------------------------------------------------------------------

_ATHLETE_ID_RE = re.compile(r"/id/(\d+)")


def _athlete_id(athlete: dict) -> str:
    for link in athlete.get("links", []):
        m = _ATHLETE_ID_RE.search(link.get("href", ""))
        if m:
            return m.group(1)
    return ""


def normalise_status(status: str) -> str:
    s = status.lower()
    for key in ("out", "doubtful", "questionable"):
        if key in s:
            return key
    if "suspen" in s:
        return "out"
    return "day-to-day"


def parse_injuries(data: dict) -> list[dict]:
    result = []
    for team in data.get("injuries", []):
        abbr = ESPN_ID_TO_ABBR.get(str(team.get("id")))
        if not abbr:
            continue
        for item in team.get("injuries", []):
            ath = item.get("athlete", {})
            details = item.get("details", {}) or {}
            result.append({
                "team": abbr,
                "player_id": _athlete_id(ath),
                "player": ath.get("displayName", ""),
                "status": normalise_status(item.get("status", "")),
                "status_raw": item.get("status", ""),
                "injury": " ".join(x for x in (details.get("type"), details.get("detail")) if x),
                "return_date": details.get("returnDate"),
                "updated": item.get("date"),
                "comment": item.get("shortComment", ""),
            })
    return result


def fetch_injuries() -> list[dict]:
    return parse_injuries(get_json("injuries"))


def fetch_roster(abbr: str) -> list[dict]:
    data = get_json(f"teams/{ESPN_TEAM_IDS[abbr]}/roster")
    return [
        {
            "player_id": str(a["id"]),
            "player": a.get("displayName", ""),
            "position": (a.get("position") or {}).get("abbreviation", ""),
            "jersey": a.get("jersey", ""),
        }
        for a in data.get("athletes", [])
    ]

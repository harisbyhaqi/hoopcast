"""
Snapshot all 30 rosters and the league injury report from ESPN, and log what changed
since the previous snapshot (new injuries, status changes, signings, trades, releases).

Outputs:
  data/rosters.json    {generated, teams: {abbr: [player, ...]}}
  data/injuries.json   {generated, injuries: [...]}
  data/changes.json    newest-first list of change events (capped)

Never fails the pipeline: if a fetch fails, the previous snapshot for that piece is kept.
"""
import json
import os
from datetime import datetime

import espn

ROOT          = os.path.join(os.path.dirname(__file__), "..")
ROSTERS_PATH  = os.path.join(ROOT, "data", "rosters.json")
INJURIES_PATH = os.path.join(ROOT, "data", "injuries.json")
CHANGES_PATH  = os.path.join(ROOT, "data", "changes.json")
MAX_CHANGES   = 1000


def _load(path: str, default):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default


def _save(path: str, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=1)


def diff_rosters(old: dict, new: dict) -> list[dict]:
    """Compare {team: [players]} snapshots. Only teams present in both are compared."""
    teams = set(old) & set(new)
    old_team = {p["player_id"]: (t, p["player"]) for t in teams for p in old[t]}
    new_team = {p["player_id"]: (t, p["player"]) for t in teams for p in new[t]}

    changes = []
    for pid, (team, name) in new_team.items():
        if pid not in old_team:
            changes.append({"kind": "signed", "team": team, "player": name})
        elif old_team[pid][0] != team:
            changes.append({"kind": "traded", "team": team, "player": name, "from": old_team[pid][0], "to": team})
    for pid, (team, name) in old_team.items():
        if pid not in new_team:
            changes.append({"kind": "released", "team": team, "player": name})
    return changes


def diff_injuries(old: list[dict], new: list[dict]) -> list[dict]:
    key = lambda i: i["player_id"] or i["player"]
    old_by, new_by = {key(i): i for i in old}, {key(i): i for i in new}

    changes = []
    for k, i in new_by.items():
        if k not in old_by:
            changes.append({"kind": "injury_new", "team": i["team"], "player": i["player"],
                            "to": i["status"], "injury": i["injury"]})
        elif old_by[k]["status"] != i["status"]:
            changes.append({"kind": "status_change", "team": i["team"], "player": i["player"],
                            "from": old_by[k]["status"], "to": i["status"], "injury": i["injury"]})
    for k, i in old_by.items():
        if k not in new_by:
            changes.append({"kind": "cleared", "team": i["team"], "player": i["player"], "from": i["status"]})
    return changes


def update(now: datetime | None = None) -> list[dict]:
    now = now or datetime.now(espn.ET)
    stamp = now.strftime("%Y-%m-%dT%H:%M:%S%z")

    old_rosters = _load(ROSTERS_PATH, None)
    old_injuries = _load(INJURIES_PATH, None)
    changes = []

    # Rosters — keep the old list for any team whose fetch fails
    teams = dict(old_rosters["teams"]) if old_rosters else {}
    fresh = {}
    for abbr in espn.TEAM_NAMES:
        try:
            fresh[abbr] = espn.fetch_roster(abbr)
        except Exception as e:
            print(f"  WARNING: roster fetch failed for {abbr}: {e}")
    if old_rosters:
        changes += diff_rosters({t: old_rosters["teams"][t] for t in fresh if t in old_rosters["teams"]}, fresh)
    teams.update(fresh)
    _save(ROSTERS_PATH, {"generated": stamp, "teams": teams})
    print(f"Rosters: {len(fresh)}/30 teams fetched.")

    # Injuries
    try:
        injuries = espn.fetch_injuries()
        if old_injuries:
            changes += diff_injuries(old_injuries["injuries"], injuries)
        _save(INJURIES_PATH, {"generated": stamp, "injuries": injuries})
        print(f"Injuries: {len(injuries)} entries.")
    except Exception as e:
        print(f"  WARNING: injury fetch failed, keeping previous report: {e}")

    for c in changes:
        c["time"] = stamp
    if changes:
        log = changes + _load(CHANGES_PATH, [])
        _save(CHANGES_PATH, log[:MAX_CHANGES])
    print(f"{len(changes)} roster/injury change(s) since last snapshot.")
    for c in changes[:20]:
        print(f"  {c['kind']:<13} {c['team']} {c['player']} {c.get('from', '')}→{c.get('to', '')}")
    return changes


if __name__ == "__main__":
    update()

import espn
from conftest import load_fixture


def test_abbreviation_map_covers_all_espn_codes():
    for code in ("NY", "SA", "GS", "NO", "UTAH", "WSH"):
        assert espn.to_nba_abbr(code) in espn.TEAM_NAMES
    assert espn.to_nba_abbr("BOS") == "BOS"
    assert set(espn.ESPN_TEAM_IDS) == set(espn.TEAM_NAMES) == set(espn.CONFERENCE)


def test_season_end_year_rolls_over_in_october():
    from datetime import date
    assert espn.season_end_year(date(2026, 9, 30)) == 2026
    assert espn.season_end_year(date(2026, 10, 1)) == 2027


def test_parse_event_final_and_eastern_date():
    events = load_fixture("schedule_bos.json")["events"]
    g = espn.parse_event(events[0])
    assert (g["home"], g["away"], g["status"]) == ("BOS", "PHI", "final")
    assert (g["home_pts"], g["away_pts"]) == (116, 117)
    assert g["season_type"] == "regular"
    assert g["date"] == "2025-10-22"   # 23:30Z is still Oct 22 in ET


def test_parse_box_score():
    teams, players = espn.parse_box_score(load_fixture("summary_401810433.json"))
    by_team = {t["abbr"]: t for t in teams}
    assert set(by_team) == {"MEM", "ORL"}
    assert by_team["ORL"]["pts"] == 118 and by_team["MEM"]["pts"] == 111
    assert (by_team["MEM"]["fgm"], by_team["MEM"]["fga"]) == (38, 83)
    # player lines add up to the team box score
    for abbr, t in by_team.items():
        mine = [p for p in players if p["team"] == abbr]
        assert sum(p["pts"] for p in mine) == t["pts"]
        for k in ("fgm", "fga", "fg3m", "ast", "stl", "blk"):
            assert sum(p[k] for p in mine) == t[k]
    jj = next(p for p in players if p["player"] == "Jaren Jackson Jr.")
    assert (jj["min"], jj["pts"], jj["fgm"], jj["fga"], jj["plus_minus"]) == (33, 30, 12, 22, -21)


def test_parse_injuries_normalises_status():
    rows = espn.parse_injuries(load_fixture("injuries.json"))
    assert rows
    for r in rows:
        assert r["team"] in espn.TEAM_NAMES
        assert r["status"] in {"out", "doubtful", "questionable", "day-to-day"}
        assert r["player_id"].isdigit()
    assert espn.normalise_status("Out") == "out"
    assert espn.normalise_status("Day-To-Day") == "day-to-day"
    assert espn.normalise_status("Suspension") == "out"

from datetime import datetime, timezone

import pandas as pd

import rosters
from update_daily import accuracy, blend, freeze, injuries_by_game, load_frozen, projected_standings


def _game(eid, date, home, away, result=None, p_home=None, frozen=None, type_="regular"):
    return {"id": eid, "date": date, "tip": f"{date}T23:00:00Z", "type": type_, "home": home, "away": away,
            "status": "final" if result else "pre",
            "result": result, "pred": None if result else {"home": 110.0, "away": 108.0, "p_home": p_home},
            "frozen": frozen, "injuries": None}


def test_projected_standings_adds_expected_wins():
    games = [
        _game("1", "2026-10-20", "BOS", "NYK", result={"home": 110, "away": 100}),
        _game("2", "2026-10-22", "NYK", "BOS", p_home=70.0),
    ]
    east = {r["team"]: r for r in projected_standings(games)["East"]}
    assert (east["BOS"]["w"], east["BOS"]["l"]) == (1, 0)
    assert east["BOS"]["proj_w"] == 1.3 and east["BOS"]["proj_l"] == 0.7
    assert east["NYK"]["proj_w"] == 0.7 and east["NYK"]["proj_l"] == 1.3


def test_accuracy_scores_frozen_predictions_only():
    games = [
        _game("1", "2026-10-20", "BOS", "NYK", result={"home": 110, "away": 100},
              frozen={"home": 112.0, "away": 104.0, "p_home": 65.0}),
        _game("2", "2026-10-20", "LAL", "GSW", result={"home": 100, "away": 120},
              frozen={"home": 110.0, "away": 106.0, "p_home": 60.0}),
        _game("3", "2026-10-20", "MIA", "ORL", result={"home": 100, "away": 99}),  # never frozen
    ]
    acc = accuracy(games)
    assert (acc["games"], acc["correct"], acc["pct"]) == (2, 1, 50.0)
    assert acc["recent"][0]["id"] == "2"


def test_injury_horizon():
    upcoming = pd.DataFrame([
        {"event_id": str(i), "date": f"2026-11-0{i}", "tip_utc": f"2026-11-0{i}T23:00:00Z", "home": "BOS", "away": "NYK"}
        for i in range(1, 7)
    ])
    inj = [
        {"team": "BOS", "player": "A", "status": "out", "return_date": None},
        {"team": "BOS", "player": "B", "status": "out", "return_date": "2026-11-05"},
        {"team": "NYK", "player": "C", "status": "questionable", "return_date": None},
    ]
    by_game = injuries_by_game(inj, upcoming)
    players = {eid: {i["player"] for i in v} for eid, v in by_game.items()}
    assert players["1"] == {"A", "B", "C"}
    assert players["3"] == {"A", "B"}
    assert players["4"] == {"B"}
    assert "5" not in players


def test_blend_is_roster_only_before_any_games():
    # Strong roster at home, no games played: form model ignored even if given
    p = blend((100.0, 130.0), 0, rating_home=5.0, rating_away=-5.0, hca=2.0, league_pts=115.0)
    assert p["form_weight"] == 0.0
    assert p["home"] - p["away"] == 12.0          # 10-pt rating gap + 2 home court
    assert p["home"] + p["away"] == 230.0         # league-average total


def test_blend_shifts_to_form_as_season_goes():
    early = blend((120.0, 100.0), 2, 0.0, 0.0, 0.0, 115.0)
    later = blend((120.0, 100.0), 40, 0.0, 0.0, 0.0, 115.0)
    assert 0 < early["form_weight"] < 0.1 < 0.5 < later["form_weight"]
    assert later["home"] - later["away"] > early["home"] - early["away"]
    assert blend((120.0, 100.0), 20, 0.0, 0.0, 0.0, 115.0)["form_weight"] == 0.5


def test_blend_applies_injuries_regardless_of_weight():
    base = blend(None, 0, 0.0, 0.0, 0.0, 115.0)
    hurt = blend(None, 0, 0.0, 0.0, 0.0, 115.0, injury_margin=-4.0)
    assert (hurt["home"] - hurt["away"]) - (base["home"] - base["away"]) == -4.0


def test_freeze_only_before_tip(tmp_path, monkeypatch):
    import update_daily
    monkeypatch.setattr(update_daily, "FROZEN_PATH", str(tmp_path / "frozen.csv"))
    games = [_game("1", "2026-10-20", "BOS", "NYK", p_home=60.0), _game("2", "2026-10-21", "LAL", "GSW", p_home=40.0)]
    games[0]["tip"] = "2026-10-20T23:00:00Z"
    before = freeze(load_frozen(), games, "2026-10-20", datetime(2026, 10, 20, 14, tzinfo=timezone.utc))
    assert list(before["event_id"]) == ["1"]
    games[0]["pred"]["p_home"] = 99.0
    after = freeze(before, games, "2026-10-20", datetime(2026, 10, 20, 23, 30, tzinfo=timezone.utc))
    assert after.iloc[0]["p_home"] == 60.0   # tipped off — evening run doesn't overwrite


def test_roster_diff_detects_moves():
    old = {"BOS": [{"player_id": "1", "player": "X"}, {"player_id": "2", "player": "Y"}], "LAL": []}
    new = {"BOS": [{"player_id": "2", "player": "Y"}], "LAL": [{"player_id": "1", "player": "X"},
                                                              {"player_id": "3", "player": "Z"}]}
    kinds = {(c["kind"], c["player"]) for c in rosters.diff_rosters(old, new)}
    assert kinds == {("traded", "X"), ("signed", "Z")}


def test_injury_diff():
    old = [{"player_id": "1", "player": "X", "team": "BOS", "status": "questionable", "injury": "Knee"},
           {"player_id": "2", "player": "Y", "team": "BOS", "status": "out", "injury": "Ankle"}]
    new = [{"player_id": "1", "player": "X", "team": "BOS", "status": "out", "injury": "Knee"},
           {"player_id": "3", "player": "Z", "team": "LAL", "status": "day-to-day", "injury": "Back"}]
    kinds = {(c["kind"], c["player"]) for c in rosters.diff_injuries(old, new)}
    assert kinds == {("status_change", "X"), ("cleared", "Y"), ("injury_new", "Z")}

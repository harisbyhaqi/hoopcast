import pandas as pd
import pytest

import ratings


TABLE = pd.DataFrame(
    {"rating": [8.0, 2.0, 0.0, -1.0, -2.0, -2.1, -3.0, -3.1, -4.0, -4.1, -5.0, -5.5],
     "mpg": [36, 34, 32, 30, 28, 26, 22, 18, 14, 12, 8, 6],
     "o": None, "d": None, "source": "darko"},
    index=["star", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12"],
)
ROSTER = [{"player": k} for k in TABLE.index]


def test_name_key_matches_across_sites():
    assert ratings.name_key("Nikola Jokić") == ratings.name_key("Nikola Jokic")
    assert ratings.name_key("Jabari Smith Jr.") == "jabari smith"
    assert ratings.name_key("A.J. Green") == ratings.name_key("AJ Green")
    assert ratings.name_key("Ronald Holland II") == ratings.name_key("Ron Holland")


def test_parse_advanced_keeps_season_total_for_traded_players():
    html = """<table id="advanced"><thead><tr><th>Rk</th><th>Player</th><th>Team</th><th>G</th><th>MP</th><th>BPM</th></tr></thead>
    <tbody><tr><td>1</td><td>Traded Guy</td><td>2TM</td><td>60</td><td>1800</td><td>1.5</td></tr>
    <tr><td>1</td><td>Traded Guy</td><td>AAA</td><td>30</td><td>900</td><td>1.0</td></tr>
    <tr><td></td><td>League Average</td><td></td><td></td><td></td><td>0.0</td></tr></tbody></table>"""
    df = ratings.parse_advanced(html)
    assert list(df["player"]) == ["Traded Guy"] and df["bpm"].iloc[0] == 1.5
    assert ratings.parse_advanced("<html>no stats yet</html>").empty


def test_rotation_fills_240_minutes():
    rot = ratings.rotation(ROSTER, TABLE)
    assert abs(sum(r["min"] for r in rot) - 240) < 1e-9
    assert rot[0]["player"] == "star"
    assert "p12" not in {r["player"] for r in rot}           # deep bench doesn't play when healthy


def test_star_back_from_injury_still_starts():
    # Usual minutes low (injured last season) but best rating on the roster → still a starter
    table = TABLE.copy()
    table.loc["star", "mpg"] = 12
    mins = {r["player"]: r["min"] for r in ratings.rotation(ROSTER, table)}
    assert mins["star"] >= 20


def test_injury_moves_minutes_down_the_bench():
    healthy = {r["player"]: r["min"] for r in ratings.rotation(ROSTER, TABLE)}
    hurt = {r["player"]: r["min"] for r in ratings.rotation(ROSTER, TABLE, {"star": 0.0})}
    assert "star" not in hurt
    assert all(hurt.get(p, 0) >= m - 1e-9 for p, m in healthy.items() if p != "star")


def test_short_roster_filled_with_replacement():
    rot = ratings.rotation([{"player": "star"}], TABLE)
    assert rot[-1]["player"] == "Replacement"
    assert rot[-1]["rating"] == ratings.REPLACEMENT_RATING


def test_losing_a_star_hurts_more_than_a_bench_player():
    healthy = ratings.roster_net(ROSTER, TABLE)
    no_star = ratings.roster_net(ROSTER, TABLE, {"star": 0.0})
    no_bench = ratings.roster_net(ROSTER, TABLE, {"p9": 0.0})
    assert no_star < no_bench <= healthy


def test_unknown_players_get_rookie_rating():
    rot = ratings.rotation([{"player": "Brand New"}], TABLE)
    assert rot[0]["rating"] == ratings.ROOKIE_RATING


def test_market_net_centered_on_41_wins():
    assert ratings.market_net(41) == 0
    assert ratings.market_net(54.5) == 5.0


def test_team_prior_blends_roster_and_market(monkeypatch):
    monkeypatch.setattr(ratings.espn, "TEAM_NAMES", {"AAA": "A", "BBB": "B"})
    rosters = {"AAA": ROSTER, "BBB": [{"player": "p12"}]}
    tr = ratings.TeamRatings(rosters, TABLE, {"AAA": 41.0, "BBB": 41.0})
    assert tr.roster_c["AAA"] == pytest.approx(-tr.roster_c["BBB"])   # centered on league average
    assert tr.roster_c["AAA"] > 0
    assert tr.prior["AAA"] == pytest.approx((1 - ratings.MARKET_WEIGHT) * tr.roster_c["AAA"])
    inj = [{"team": "AAA", "player": "star", "status": "out"}]
    assert tr.injury_delta("AAA", inj) < 0
    assert tr.injury_delta("BBB", inj) == 0


def test_espn_season_totals_aggregates_regular_season(tmp_path):
    rows = []
    for gid, sid, mins in (("g1", 22026, 30), ("g2", 22026, 20), ("g3", 42026, 40), ("g4", 22026, 0)):
        rows.append({"GAME_ID": gid, "GAME_DATE": "2027-01-0" + gid[1], "SEASON_ID": sid, "player_id": "7",
                     "player": "Some Guy", "team": "BOS", "min": mins, "pts": 10,
                     **{k: 1 for k in ratings.BOX_STATS if k != "pts"}})
    path = tmp_path / "pg.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    t = ratings.espn_season_totals(2027, str(path))
    assert len(t) == 1
    r = t.iloc[0]
    assert (r["player"], r["g"], r["mp"], r["pts"]) == ("Some Guy", 2, 50, 20)   # playoffs and DNPs excluded
    assert ratings.per36(t).iloc[0]["pts"] == pytest.approx(20 / 50 * 36)


def test_box_bpm_recovers_linear_relationship(monkeypatch):
    import numpy as np
    rng = np.random.default_rng(0)
    n = 300
    totals = pd.DataFrame({k: rng.uniform(0, 20, n) for k in ratings.BOX_STATS})
    totals["mp"] = rng.uniform(600, 2500, n)
    totals["g"] = 60
    totals["player"] = [f"Player {i}" for i in range(n)]
    rates = ratings.per36(totals)
    true_bpm = 0.3 * rates["pts"] - 0.4 * rates["tov"] + 0.2 * rates["plus_minus"] - 3
    totals_counts = totals.copy()
    br = pd.DataFrame({"player": totals["player"], "g": 60, "mp": totals["mp"], "bpm": true_bpm})
    monkeypatch.setattr(ratings, "espn_season_totals", lambda season, *a: totals_counts.copy())
    monkeypatch.setattr(ratings, "load_bpm", lambda year: br)

    box = ratings.BoxBPM(2026)
    assert box.cv_r2 > 0.95
    pred = box.predict(totals.head(5))
    assert list(pred.columns) == ["player", "g", "mp", "bpm"]
    assert np.allclose(pred["bpm"], true_bpm.head(5), atol=0.2)


def test_parse_darko_decodes_devalue_payload():
    # {"players": {"keys": [...], "values": [[...], ...]}} flattened the way SvelteKit ships it
    data = [
        {"players": 1},
        {"keys": 2, "values": 9},
        [3, 4, 5, 6, 7, 8],
        "player_name", "team_name", "dpm", "o_dpm", "d_dpm", "x_minutes",
        [10, 13, 16, 19, 22, 25],
        [11, 12], "Nikola Jokic", "Rudy Gobert",
        [14, 15], "Denver Nuggets", "Minnesota Timberwolves",
        [17, 18], 7.35, 2.45,
        [20, 21], 5.2, -1.5,
        [23, 24], 2.1, 4.0,
        [26, 27], 34.7, 25.3,
    ]
    df = ratings.parse_darko({"type": "data", "nodes": [None, {"type": "data", "data": data}]})
    assert list(df["player"]) == ["Nikola Jokic", "Rudy Gobert"]
    gobert = df.iloc[1]
    assert (gobert["dpm"], gobert["o_dpm"], gobert["d_dpm"], gobert["mpg"]) == (2.45, -1.5, 4.0, 25.3)


def test_player_table_prefers_darko_and_maps_bpm(monkeypatch):
    import numpy as np
    keys = [f"p{i}" for i in range(80)]
    bpm = pd.DataFrame({"bpm": np.linspace(-4, 8, 80), "mpg": 25.0}, index=keys + [])
    bpm.loc["only_bpm"] = [4.0, 20.0]
    darko = pd.DataFrame({"player": keys, "dpm": np.linspace(-4, 8, 80) / 2, "o_dpm": 0.0, "d_dpm": 0.0,
                          "mpg": 10.0})
    monkeypatch.setattr(ratings, "bpm_table", lambda season: bpm)
    monkeypatch.setattr(ratings, "load_darko", lambda: darko)
    t = ratings.player_table(2027)
    assert t.at["p79", "source"] == "darko" and t.at["p79", "rating"] == pytest.approx(4.0)
    assert t.at["p79", "mpg"] == 25.0                     # multi-season minutes beat DARKO's projection
    assert t.at["only_bpm", "source"] == "bpm"
    assert t.at["only_bpm", "rating"] == pytest.approx(2.0)   # BPM 4 → DPM scale (half)

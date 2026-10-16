import pandas as pd
import pytest

import ratings


TABLE = pd.DataFrame(
    {"bpm": [8.0, 2.0, 0.0, -1.0, -2.0, -2.0, -3.0, -3.0, -4.0, -4.0, -5.0],
     "mpg": [36, 34, 32, 30, 28, 26, 22, 18, 14, 12, 8]},
    index=["star", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11"],
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


def test_rotation_fills_240_minutes_in_depth_order():
    rot = ratings.rotation(ROSTER, TABLE)
    assert abs(sum(r["min"] for r in rot) - 240) < 1e-9
    assert "p11" not in {r["player"] for r in rot}           # 12th man doesn't play when healthy
    assert rot[0]["player"] == "star" and rot[0]["min"] == 36
    # star out: his minutes go down the bench, nobody else plays more than usual
    hurt = {r["player"]: r["min"] for r in ratings.rotation(ROSTER, TABLE, {"star": 0.0})}
    assert "star" not in hurt and hurt["p2"] == 34 and "p11" in hurt


def test_short_roster_filled_with_replacement():
    rot = ratings.rotation([{"player": "star"}], TABLE)
    assert rot[-1]["player"] == "Replacement" and rot[-1]["min"] == 240 - 36


def test_losing_a_star_hurts_more_than_a_bench_player():
    healthy = ratings.roster_net(ROSTER, TABLE)
    no_star = ratings.roster_net(ROSTER, TABLE, {"star": 0.0})
    no_bench = ratings.roster_net(ROSTER, TABLE, {"p9": 0.0})
    assert no_star < no_bench <= healthy


def test_unknown_players_get_rookie_rating():
    rot = ratings.rotation([{"player": "Brand New"}], TABLE)
    assert rot[0]["bpm"] == ratings.ROOKIE_BPM


def test_market_net_centered_on_41_wins():
    assert ratings.market_net(41) == 0
    assert ratings.market_net(54.5) == 5.0


def test_team_prior_blends_roster_and_market(monkeypatch):
    monkeypatch.setattr(ratings.espn, "TEAM_NAMES", {"AAA": "A", "BBB": "B"})
    rosters = {"AAA": ROSTER, "BBB": [{"player": "p11"}]}
    tr = ratings.TeamRatings(rosters, TABLE, {"AAA": 41.0, "BBB": 41.0})
    assert tr.roster_c["AAA"] == pytest.approx(-tr.roster_c["BBB"])   # centered on league average
    assert tr.roster_c["AAA"] > 0
    assert tr.prior["AAA"] == pytest.approx((1 - ratings.MARKET_WEIGHT) * tr.roster_c["AAA"])
    inj = [{"team": "AAA", "player": "star", "status": "out"}]
    assert tr.injury_delta("AAA", inj) < 0
    assert tr.injury_delta("BBB", inj) == 0

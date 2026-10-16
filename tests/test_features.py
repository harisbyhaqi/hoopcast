import numpy as np
import pandas as pd

from utils import rolling_team_stats, team_states, FEATURE_COLS


def _games():
    """Two teams, three games each: A wins at home, B wins at home, A wins at home (new season)."""
    rows = []
    spec = [  # (game_id, date, season_id, home, away, home_pts, away_pts)
        ("g1", "2025-04-01", 22024, "AAA", "BBB", 110, 100),
        ("g2", "2025-04-03", 22024, "BBB", "AAA", 120, 90),
        ("g3", "2025-10-25", 22025, "AAA", "BBB", 105, 95),
    ]
    for gid, d, sid, h, a, hp, ap in spec:
        for team, opp, pts, opp_pts, home in ((h, a, hp, ap, 1), (a, h, ap, hp, 0)):
            rows.append({
                "GAME_ID": gid, "GAME_DATE": pd.Timestamp(d), "SEASON_ID": sid,
                "TEAM_ABBREVIATION": team, "MATCHUP": f"{team} vs. {opp}" if home else f"{team} @ {opp}",
                "home": home, "WL": "W" if pts > opp_pts else "L", "PTS": pts,
                "FGM": 40, "FGA": 85, "FG3M": 12, "FG3A": 35, "FTA": 20, "TOV": 14,
                "FG_PCT": 40 / 85, "FG3_PCT": 12 / 35, "REB": 44, "AST": 25,
            })
    return pd.DataFrame(rows)


def test_features_use_only_prior_games():
    df = rolling_team_stats(_games()).set_index(["GAME_ID", "TEAM_ABBREVIATION"])
    # g2 row for AAA must reflect only g1 (AAA scored 110, allowed 100)
    assert df.loc[("g2", "AAA"), "pts_roll5"] == 110
    assert df.loc[("g2", "AAA"), "pts_allowed_roll5"] == 100
    # opponent defense on g2 for AAA = BBB's pts allowed before g2 = 110
    assert df.loc[("g2", "AAA"), "opp_pts_allowed_roll5"] == 110
    assert np.isnan(df.loc[("g1", "AAA"), "pts_roll5"])


def test_home_away_win_pct_resets_each_season():
    df = rolling_team_stats(_games()).set_index(["GAME_ID", "TEAM_ABBREVIATION"])
    # Before g2, AAA had 1 home win last season → shrunk above .500
    assert df.loc[("g2", "AAA"), "home_win_pct"] > 0.5
    # g3 is the first game of a new season → back to the .500 prior
    assert df.loc[("g3", "AAA"), "home_win_pct"] == 0.5
    # form doesn't carry over: first game of a season has no EWM features
    assert np.isnan(df.loc[("g3", "AAA"), "pts_roll5"])
    assert df.loc[("g3", "AAA"), "win_pct_last10"] == 0.5


def test_team_states_include_latest_game():
    states = team_states(_games(), "2025-10-28", 22025)
    assert set(states.index) == {"AAA", "BBB"}
    # AAA's last game (g3, 105 pts) must be part of its state
    pts = states.loc["AAA", "pts_roll5"]
    assert pts != rolling_team_stats(_games()).query("GAME_ID == 'g3' and TEAM_ABBREVIATION == 'AAA'")["pts_roll5"].iloc[0]
    assert states.loc["AAA", "rest_days"] == 3
    expected = [c for c in FEATURE_COLS if not c.startswith("opp_")]
    assert not states[expected].isna().any().any()


def test_team_states_new_season_has_no_form():
    states = team_states(_games(), "2026-10-20", 22026)
    assert states["pts_roll5"].isna().all()
    assert (states["rest_days"] == 10).all()

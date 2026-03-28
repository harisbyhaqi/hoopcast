"""
Generate predictions.json for the static GitHub Pages website.
Produces win probabilities and predicted scores for featured matchups.
"""
import os, json, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from predict import load_model, team_features_from_history
from simulate import monte_carlo

PROCESSED = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "games_preprocessed.csv")
OUT_PATH  = os.path.join(os.path.dirname(__file__), "..", "docs", "predictions.json")

TEAM_NAMES = {
    "LAL": "Los Angeles Lakers",   "GSW": "Golden State Warriors",
    "BOS": "Boston Celtics",       "MIL": "Milwaukee Bucks",
    "DEN": "Denver Nuggets",       "PHX": "Phoenix Suns",
    "MIA": "Miami Heat",           "NYK": "New York Knicks",
    "OKC": "Oklahoma City Thunder","MIN": "Minnesota Timberwolves",
    "DAL": "Dallas Mavericks",     "LAC": "LA Clippers",
    "SAC": "Sacramento Kings",     "CLE": "Cleveland Cavaliers",
    "IND": "Indiana Pacers",       "PHI": "Philadelphia 76ers",
    "CHI": "Chicago Bulls",        "ATL": "Atlanta Hawks",
    "NOP": "New Orleans Pelicans", "MEM": "Memphis Grizzlies",
}

MATCHUPS = [
    ("OKC", "BOS", 1, 0),
    ("LAL", "GSW", 1, 0),
    ("MIL", "MIA", 1, 0),
    ("DEN", "MIN", 1, 0),
    ("NYK", "IND", 1, 0),
    ("DAL", "PHX", 1, 0),
]

def get_logo_url(abbr):
    team_ids = {
        "ATL":1610612737,"BOS":1610612738,"CLE":1610612739,"NOP":1610612740,
        "CHI":1610612741,"DAL":1610612742,"DEN":1610612743,"GSW":1610612744,
        "HOU":1610612745,"IND":1610612754,"LAC":1610612746,"LAL":1610612747,
        "MEM":1610612763,"MIA":1610612748,"MIL":1610612749,"MIN":1610612750,
        "BKN":1610612751,"NYK":1610612752,"ORL":1610612753,"PHI":1610612755,
        "PHX":1610612756,"POR":1610612757,"SAC":1610612758,"SAS":1610612759,
        "OKC":1610612760,"TOR":1610612761,"UTA":1610612762,"WAS":1610612764,
        "CHA":1610612766,"DET":1610612765,
    }
    tid = team_ids.get(abbr, 0)
    return f"https://cdn.nba.com/logos/nba/{tid}/global/L/logo.svg"

def main():
    model = load_model()
    df = pd.read_csv(PROCESSED, parse_dates=["GAME_DATE"])

    # Get all available teams
    available = set(df["TEAM_ABBREVIATION"].unique())

    results = []
    for a, b, home_a, home_b in MATCHUPS:
        if a not in available or b not in available:
            print(f"Skipping {a} vs {b} — not in data")
            continue
        feat_a = team_features_from_history(a, PROCESSED)
        feat_b = team_features_from_history(b, PROCESSED)
        feat_a["home"] = home_a
        feat_b["home"] = home_b

        sim = monte_carlo(feat_a, feat_b, model, n=10000)

        results.append({
            "team_a": {"abbr": a, "name": TEAM_NAMES.get(a, a), "logo": get_logo_url(a)},
            "team_b": {"abbr": b, "name": TEAM_NAMES.get(b, b), "logo": get_logo_url(b)},
            "score_a": sim["score_a"],
            "score_b": sim["score_b"],
            "win_prob_a": round(sim["win_prob_a"] * 100, 1),
            "win_prob_b": round(sim["win_prob_b"] * 100, 1),
        })
        print(f"{a} {sim['score_a']} - {b} {sim['score_b']}  |  {a} wins {sim['win_prob_a']*100:.1f}%")

    # Model stats
    payload = {
        "generated": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "model_stats": {"mae": 8.67, "rmse": 10.94, "win_accuracy": 68.5},
        "matchups": results,
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved -> {OUT_PATH}")

if __name__ == "__main__":
    main()

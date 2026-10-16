import math

DEFAULT_MARGIN_SIGMA = 12.0   # std of (actual − predicted) margin; train.py measures the real value


def win_probability(score_a: float, score_b: float, margin_sigma: float = DEFAULT_MARGIN_SIGMA) -> float:
    """
    P(team A wins) assuming the actual margin is Normal(score_a − score_b, margin_sigma).
    Exact closed form of the old Monte Carlo — deterministic, and fast enough for a full season.
    """
    z = (score_a - score_b) / margin_sigma
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def simulate_game(score_a: float, score_b: float, margin_sigma: float = DEFAULT_MARGIN_SIGMA,
                  score_a_offset: float = 0.0, score_b_offset: float = 0.0) -> dict:
    """
    Apply offsets (injuries, H2H) to predicted scores and return win probabilities.
    """
    score_a = round(score_a + score_a_offset, 1)
    score_b = round(score_b + score_b_offset, 1)
    p = win_probability(score_a, score_b, margin_sigma)
    return {
        "score_a": score_a,
        "score_b": score_b,
        "win_prob_a": round(p, 4),
        "win_prob_b": round(1 - p, 4),
    }


if __name__ == "__main__":
    import sys, os
    from predict import predict_game, team_features_from_history
    processed = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "games_preprocessed.csv")

    team_a = sys.argv[1] if len(sys.argv) > 1 else "LAL"
    team_b = sys.argv[2] if len(sys.argv) > 2 else "GSW"

    feat_a = team_features_from_history(team_a, processed)
    feat_b = team_features_from_history(team_b, processed)
    feat_a["home"] = 1
    feat_b["home"] = 0

    result = simulate_game(*predict_game(feat_a, feat_b))
    print(f"{team_a} vs {team_b}")
    print(f"  Predicted: {result['score_a']} - {result['score_b']}")
    print(f"  Win prob:  {team_a} {result['win_prob_a']*100:.1f}%  |  {team_b} {result['win_prob_b']*100:.1f}%")

import numpy as np
from predict import predict_game, load_model, team_features_from_history


def monte_carlo(team_a_feat: dict, team_b_feat: dict,
                model=None, n: int = 10_000, noise_std: float = 8.0) -> dict:
    """
    Simulate n games by adding Gaussian noise to predicted scores.
    Returns win probability for team A and expected scores.
    """
    if model is None:
        model = load_model()

    score_a, score_b = predict_game(team_a_feat, team_b_feat, model)

    noise_a = np.random.normal(0, noise_std, n)
    noise_b = np.random.normal(0, noise_std, n)

    sims_a = score_a + noise_a
    sims_b = score_b + noise_b

    win_prob_a = float(np.mean(sims_a > sims_b))

    return {
        "score_a": score_a,
        "score_b": score_b,
        "win_prob_a": round(win_prob_a, 4),
        "win_prob_b": round(1 - win_prob_a, 4),
        "simulations": n,
    }


if __name__ == "__main__":
    import sys, os
    processed = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "games_preprocessed.csv")

    team_a = sys.argv[1] if len(sys.argv) > 1 else "LAL"
    team_b = sys.argv[2] if len(sys.argv) > 2 else "GSW"

    feat_a = team_features_from_history(team_a, processed)
    feat_b = team_features_from_history(team_b, processed)
    feat_a["home"] = 1
    feat_b["home"] = 0

    result = monte_carlo(feat_a, feat_b)
    print(f"{team_a} vs {team_b}")
    print(f"  Predicted: {result['score_a']} - {result['score_b']}")
    print(f"  Win prob:  {team_a} {result['win_prob_a']*100:.1f}%  |  {team_b} {result['win_prob_b']*100:.1f}%")

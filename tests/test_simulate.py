import pytest

from simulate import win_probability, simulate_game


def test_even_scores_are_coin_flip():
    assert win_probability(110, 110) == pytest.approx(0.5)


def test_symmetry_and_monotonic():
    p = win_probability(112, 105, 12)
    assert p + win_probability(105, 112, 12) == pytest.approx(1.0)
    assert 0.5 < p < win_probability(118, 105, 12)


def test_offsets_applied():
    r = simulate_game(110, 108, 12, score_a_offset=-4.5)
    assert r["score_a"] == 105.5
    assert r["win_prob_a"] < 0.5
    assert r["win_prob_a"] + r["win_prob_b"] == pytest.approx(1.0)

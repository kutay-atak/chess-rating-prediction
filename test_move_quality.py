import math

import numpy as np

from move_quality import GRID, move_losses, posterior, summary, table

NAN = float("nan")


def test_move_losses_follow_the_conventions():
    loss = move_losses([0.3, 0.2, -1.0, -0.8, -4.5, NAN])
    assert math.isnan(loss[0]) and math.isnan(loss[5])  # White's first move; no eval after the last
    assert np.allclose(loss[1:5], [0.0, 1.2, 0.2, 3.7])


def test_good_moves_raise_and_blunders_lower_only_the_mover():
    # two ratings bands: low players blunder often, high players rarely
    counts = np.zeros((1, len(GRID), 7))
    counts[0, : len(GRID) // 2] = [50, 10, 10, 10, 10, 10, 0]  # weaker half
    counts[0, len(GRID) // 2:] = [90, 6, 2, 1, 1, 0, 0]  # stronger half
    log_lik, log_prior = table(counts), np.log(np.full(len(GRID), 1 / len(GRID)))
    flat = summary(posterior(log_prior, log_lik, np.array([], int), np.array([], int), 1.0))[1]
    good = summary(posterior(log_prior, log_lik, np.zeros(10, int), np.zeros(10, int), 1.0))[1]
    bad = summary(posterior(log_prior, log_lik, np.full(10, 4), np.zeros(10, int), 1.0))[1]
    assert bad < flat < good


def test_summary_interval_contains_the_mean():
    p = np.exp(-0.5 * ((GRID - 1500) / 200) ** 2)
    lo, mean, hi = summary(p / p.sum())
    assert abs(mean - 1500) < 1 and lo < mean < hi and abs((hi - lo) / 2 - 1.645 * 200) < 30

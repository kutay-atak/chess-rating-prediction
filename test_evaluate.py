import math

import numpy as np

from evaluate import player_bootstrap, stratified


def test_stratified_estimate_and_se_by_hand():
    y, h = np.array([1.0, 3.0, 2.0, 4.0, 6.0]), np.array([0, 0, 1, 1, 1])
    N, n_all = np.array([100.0, 300.0]), np.array([10.0, 30.0])
    est, se = stratified(y, h, N, n_all)
    assert math.isclose(est, 0.25 * 2 + 0.75 * 4)
    var = 0.25 ** 2 * 0.9 * 2 / 2 + 0.75 ** 2 * 0.9 * 4 / 3  # W² (1 - f) s² / n per stratum
    assert math.isclose(se, math.sqrt(var))


def test_census_has_no_sampling_error():
    y, h = np.array([1.0, 3.0, 2.0, 4.0]), np.array([0, 0, 1, 1])
    assert stratified(y, h, np.array([2.0, 2.0]), np.array([2.0, 2.0]))[1] == 0


def test_missing_stratum_renormalises_weights():
    est, _ = stratified(np.array([5.0, 7.0]), np.array([1, 1]), np.array([100.0, 300.0]), np.array([10.0, 30.0]))
    assert est == 6


def test_player_bootstrap():
    err = np.array([10.0, 30.0, 20.0, 20.0])
    game, h, N = np.array([1, 2, 3, 4]), np.array([0, 0, 1, 1]), np.array([100.0, 100.0])
    assert player_bootstrap(err, game, np.array(["a", "a", "b", "b"]), h, N) == 0  # one player per stratum
    assert player_bootstrap(np.full(4, 5.0), game, np.array(["a", "b", "c", "d"]), h, N) == 0  # no spread
    assert player_bootstrap(err, game, np.array(["a", "b", "c", "d"]), h, N) > 0

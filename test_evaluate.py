import math

import numpy as np

import pandas as pd

from evaluate import by_band, interval_score, player_bootstrap, shard_jackknife, stratified


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


def test_shard_jackknife_is_the_se_of_a_mean_for_one_stratum():
    # one unit per shard: the delete-one-shard jackknife SE of a mean is s / sqrt(n)
    se = shard_jackknife(np.array([1.0, 2.0, 3.0]), np.zeros(3, int), np.array([0, 1, 2]), np.full((3, 1), 10.0))
    assert math.isclose(se, 1 / math.sqrt(3))


def test_interval_score_charges_twenty_times_the_miss():
    lo, hi = np.zeros(3), np.full(3, 10.0)
    assert np.allclose(interval_score(lo, hi, np.array([5.0, 12.0, -1.0])), [10, 50, 30])


def test_by_band_weights_sides_by_game_weight():
    preds = pd.DataFrame({"game_id": [1, 2, 3], "side": "white", "t": 2, "mean": [1500.0, 1500, 1500],
                          "lo": [1000.0, 1450, 1000], "hi": [2000.0, 1550, 2000]})
    test = pd.DataFrame({"game_id": [1, 2, 3], "side": "white", "rating": [1600.0, 1700, 2300], "weight": [3.0, 1, 1]})
    band = by_band(preds, test).set_index("band")
    assert band.loc["1400-1799", "n"] == 2 and band.loc["1400-1799", "bias"] == (3 * -100 + 1 * -200) / 4
    assert band.loc["1400-1799", "coverage"] == 0.75 and band.loc["2200+", "mae"] == 800

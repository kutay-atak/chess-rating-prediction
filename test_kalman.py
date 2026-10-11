import math

import numpy as np
import pandas as pd

from kalman import autocorrelation, checkpoint_preds, filter_ar1, filter_iid, move_rows, tc_prior

NAN = float("nan")


def test_move_rows():
    # the 3+2 game of test_features.py: spent White 0, 5, 3 and Black 0, 7, 7
    games = pd.DataFrame({"clocks": [[180, 180, 177, 175, 176, 170]], "TimeControl": ["180+2"]})
    m = move_rows(games).sort_values("ply")
    assert list(m.ply) == [0, 1, 2, 3, 4, 5] and list(m.side) == [0, 1, 0, 1, 0, 1] and list(m.k) == [0, 0, 1, 1, 2, 2]
    assert np.allclose(m.log_spent, np.log1p([0, 0, 5, 7, 3, 7]))
    assert np.allclose(m.clock_share, np.array([180, 180, 177, 175, 176, 170]) / 180)


def test_iid_filter_is_precision_weighted_averaging():
    z, s = np.array([[1700.0, 1600.0, NAN]]), np.array([[200.0, 100.0, NAN]])
    mean, var = filter_iid(z, s, np.array([1500.0]), np.array([400.0 ** 2]))
    prec = 1 / 400 ** 2 + 1 / 200 ** 2 + 1 / 100 ** 2
    assert math.isclose(var[0, 1], 1 / prec)
    assert math.isclose(mean[0, 1], (1500 / 400 ** 2 + 1700 / 200 ** 2 + 1600 / 100 ** 2) / prec)


def test_ar1_filter_with_rho_0_is_the_iid_filter():
    rng = np.random.default_rng(0)
    z, s = rng.normal(1500, 300, (5, 8)), rng.uniform(100, 400, (5, 8))
    z[0, 5:] = s[0, 5:] = NAN  # a shorter game
    pm, pv = rng.normal(1500, 100, 5), rng.uniform(1e4, 2e5, 5)
    iid, ar1 = filter_iid(z, s, pm, pv), filter_ar1(z, s, pm, pv, rho=0.0)
    ok = ~np.isnan(z)
    assert np.allclose(iid[0][ok], ar1[0][ok]) and np.allclose(iid[1][ok], ar1[1][ok])
    assert np.isnan(ar1[0][~ok]).all()


def test_correlated_errors_add_less_information():
    z, s = np.full((1, 10), 1600.0), np.full((1, 10), 200.0)
    pm, pv = np.array([1500.0]), np.array([300.0 ** 2])
    iid_var, half_var, same_var = filter_iid(z, s, pm, pv)[1], filter_ar1(z, s, pm, pv, 0.5)[1], filter_ar1(z, s, pm, pv, 0.999)[1]
    assert iid_var[0, -1] < half_var[0, -1] < same_var[0, -1]
    assert math.isclose(same_var[0, -1], same_var[0, 0], rel_tol=0.02)  # identical errors: later moves add almost nothing


def test_autocorrelation_recovers_an_ar1_rho():
    rng = np.random.default_rng(0)
    n, K, rho = 3000, 20, 0.6
    u = np.zeros((n, K))
    u[:, 0] = rng.normal(size=n)
    for k in range(1, K):
        u[:, k] = rho * u[:, k - 1] + np.sqrt(1 - rho ** 2) * rng.normal(size=n)
    seq, k = np.repeat(np.arange(n), K), np.tile(np.arange(K), n)
    assert abs(autocorrelation(u.ravel(), seq, k, np.ones(n * K), 1) - rho) < 0.02
    assert abs(autocorrelation(u.ravel(), seq, k, np.ones(n * K), 2) - rho ** 2) < 0.02


def test_checkpoint_preds_take_the_sides_own_moves_among_the_first_t_plies():
    means = np.tile(np.arange(1.0, 41.0), (2, 1))  # one game: White row 0, Black row 1; value = own moves seen
    preds = checkpoint_preds(means, np.ones((2, 40)), np.array([7]), np.array([5]))
    at5 = preds[preds.t == 5].set_index("side")["mean"]
    assert set(preds.t) == {2, 5} and at5["white"] == 3 and at5["black"] == 2


def test_tc_prior_falls_back_for_rare_time_controls():
    games = pd.DataFrame({"TimeControl": ["180+0"] * 40 + ["15+10"], "WhiteElo": [1500] * 20 + [1700] * 20 + [2500],
                          "BlackElo": [1500] * 20 + [1700] * 20 + [2500], "weight": 1.0})
    mean, var = tc_prior(games, np.ones(41, bool))
    assert mean[0] == 1600 and var[0] == 100 ** 2
    assert mean[-1] != 2500 and var[-1] > var[0]  # 15+10 has one game: overall moments instead

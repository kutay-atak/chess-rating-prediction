import math

import numpy as np
import pandas as pd
import torch

from features import CHECKPOINTS
from gru import GRU, UNK, batch, conformal, encode, load, nll, ply_inputs, predict, vocab

NAN = float("nan")
# the 3+2 game of test_features.py: spent White 0, 5, 3 and Black 0, 7, 7
CLOCKS = [180, 180, 177, 175, 176, 170]


def test_ply_inputs():
    x = ply_inputs(CLOCKS, 180, 2)
    assert x.shape == (6, 5)
    assert np.allclose(x[:, 0], np.log1p([0, 0, 5, 7, 3, 7]))  # seconds spent by each ply's mover
    assert np.allclose(x[:, 1], np.array(CLOCKS) / 180)
    assert list(x[:, 2]) == [0, 1, 0, 1, 0, 1]
    assert np.allclose(x[:, 3], 1) and np.allclose(x[:, 4], 0.4)


def test_rare_moves_are_unknown():
    words = vocab([["e4", "e5"], ["e4", "d5"]], min_count=2)
    assert words == ["e4"]
    assert list(encode(["e4", "Nf3"], {m: i + 2 for i, m in enumerate(words)})) == [2, UNK]


def test_nll_skips_padding_and_unscored_sides():
    mean = torch.tensor([[[0.0, 1.0], [1.0, 9.0]], [[2.0, 0.0], [NAN, NAN]]])
    logvar = torch.full((2, 2, 2), math.log(4))  # variance 4, so the log-variance term counts
    y = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    keep = torch.tensor([[[True, False], [True, False]], [[True, True], [False, False]]])  # Black unscored in game 0
    w = torch.tensor([2.0, 1.0])
    half_log4 = 0.5 * math.log(4)
    expected = (2 * (half_log4 + 1 / 8 + half_log4 + 0) + 1 * (half_log4 + 4 / 8 + half_log4 + 0)) / 2
    assert math.isclose(nll(mean, logvar, y, keep, w).item(), expected, rel_tol=1e-6)  # float32


def games(lengths):
    rng = np.random.default_rng(0)
    return pd.DataFrame({"moves": [["e4"] * n for n in lengths],
                         "clocks": [list(180 - rng.integers(0, 3, n).cumsum()) for n in lengths], "TimeControl": "180+2"})


def test_outputs_depend_only_on_plies_seen():
    torch.manual_seed(0)
    model = GRU(3)
    data = load(games([7, 4]), {"e4": 2})
    moves, x, lengths = batch(data, np.array([0, 1]))  # the 4-ply game is right-padded to 7
    mean, logvar = model(moves, x)
    for t in range(1, 5):
        alone = model(moves[1:, :t], x[1:, :t])
        assert torch.allclose(mean[1, t - 1], alone[0][0, -1], atol=1e-6)
        assert torch.allclose(logvar[1, t - 1], alone[1][0, -1], atol=1e-6)


def test_predict_reads_the_output_after_ply_t():
    def plies_seen(moves, x):  # a "model" whose mean after each ply is the number of plies seen
        t = torch.arange(1, moves.shape[1] + 1, dtype=torch.float32)[None, :, None].expand(len(moves), -1, 2)
        return t, torch.zeros_like(t)

    data = load(games([85, 12]), {"e4": 2})
    preds = predict(plies_seen, data, np.array([0, 1]), np.array([10, 11]), m=1500, sd=100)
    assert set(preds.t[preds.game_id == 11]) == {2, 5, 10} and set(preds.t[preds.game_id == 10]) == set(CHECKPOINTS)
    assert np.allclose(preds["mean"], 1500 + 100 * preds.t) and np.allclose(preds.sd, 100)
    short = predict(plies_seen, data, np.array([1]), np.array([10, 11]), m=1500, sd=100)  # a batch of short games only
    assert set(short.t) == {2, 5, 10}


def test_conformal_scales_by_sd_and_skips_unscored_sides():
    preds = pd.DataFrame({"game_id": [1, 1, 2, 3, 4], "side": ["white", "black", "white", "white", "white"],
                          "t": 2, "mean": 1500.0, "sd": [100.0, 100.0, 200.0, 100.0, 50.0]})
    truth = pd.DataFrame({"game_id": [1, 1, 2, 3, 4], "side": ["white", "black", "white", "white", "white"],
                          "rating": [1600.0, 9999.0, 1800.0, 1550.0, 1600.0], "scored": [True, False, True, True, True],
                          "weight": 1.0})
    # scored |error| / sd: 1, 1.5, 0.5, 2 (the unscored black side is ignored); 90% weighted quantile: 2
    assert conformal(preds, truth) == {2: 2.0}
    assert conformal(preds, truth.assign(weight=[1.0, 1, 1, 1, 0.01])) == {2: 1.5}  # unweighted it would be 2

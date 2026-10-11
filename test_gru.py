import math

import numpy as np
import pandas as pd
import torch

from features import CHECKPOINTS
from gru import GRU, UNK, batch, encode, load, nll, ply_inputs, predict, vocab

NAN = float("nan")
# the 3+2 game of test_features.py: spent White 0, 5, 3 and Black 0, 7, 7
CLOCKS = [180, 180, 177, 175, 176, 170]
EVALS = [0.3, 0.2, -1.0, -0.8, -4.5, NAN]


def test_ply_inputs():
    x = ply_inputs(CLOCKS, EVALS, 180, 2)
    assert x.shape == (6, 9)
    assert np.allclose(x[:, 0], np.log1p([0, 0, 5, 7, 3, 7]))  # seconds spent by each ply's mover
    assert np.allclose(x[:, 1], np.array(CLOCKS) / 180)
    assert list(x[:, 2]) == [0, 1, 0, 1, 0, 1]
    assert np.allclose(x[:, 3], 1) and np.allclose(x[:, 4], 0.4)
    assert np.allclose(x[:, 5], [0.03, 0.02, -0.1, -0.08, -0.45, 0]) and list(x[:, 6]) == [0, 0, 0, 0, 0, 1]
    # White's first move has no loss; Black's last has no eval after it
    assert np.allclose(x[:, 7], [0, 0, 0.12, 0.02, 0.37, 0]) and list(x[:, 8]) == [1, 0, 0, 0, 0, 1]


def test_rare_moves_are_unknown():
    words = vocab([["e4", "e5"], ["e4", "d5"]], min_count=2)
    assert words == ["e4"]
    assert list(encode(["e4", "Nf3"], {m: i + 2 for i, m in enumerate(words)})) == [2, UNK]


def test_nll_skips_padding_and_unscored_sides():
    mean = torch.tensor([[[0.0, 1.0], [1.0, 9.0]], [[2.0, 0.0], [NAN, NAN]]])
    logvar = torch.zeros(2, 2, 2)
    y = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    keep = torch.tensor([[[True, False], [True, False]], [[True, True], [False, False]]])  # Black unscored in game 0
    w = torch.tensor([2.0, 1.0])
    expected = (2 * 0.5 * (1 + 0) + 1 * 0.5 * (4 + 0)) / 2
    assert math.isclose(nll(mean, logvar, y, keep, w).item(), expected)


def games(lengths):
    rng = np.random.default_rng(0)
    return pd.DataFrame({"moves": [["e4"] * n for n in lengths],
                         "clocks": [list(180 - np.arange(n)) for n in lengths],
                         "evals": [list(rng.normal(size=n)) for n in lengths], "TimeControl": "180+2"})


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
    assert np.allclose(preds["mean"], 1500 + 100 * preds.t)
    assert np.allclose(preds.hi - preds["mean"], 1.645 * 100)

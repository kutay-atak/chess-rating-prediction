import numpy as np
import pandas as pd

from sample import LABELS, draw, score_sides, stratum
from split import player_split


def test_stratum_bands():
    white = [900, 1000, 1199, 1200, 2399, 2400, 3000]
    black = [1000, 1000, 1200, 1200, 2400, 2400, 3000]
    assert [LABELS[h] for h in stratum(white, black)] == [
        "<1000", "1000-1199", "1000-1199", "1200-1399", "2200-2399", "2400+", "2400+"]


def test_draw_caps_each_stratum_without_replacement():
    strata = np.array([0] * 5 + [1] * 100 + [3] * 3)
    drawn, N, n = draw(strata, cap=10, seed=0)
    assert N.tolist() == [5, 100, 0, 3, 0, 0, 0, 0, 0]
    assert n.tolist() == [5, 10, 0, 3, 0, 0, 0, 0, 0]
    assert len(set(drawn)) == len(drawn) == 18
    assert np.bincount(strata[drawn], minlength=len(LABELS)).tolist() == n.tolist()
    np.testing.assert_array_equal(drawn, draw(strata, cap=10, seed=0)[0])  # same seed, same sample


def test_only_sides_from_the_games_split_are_scored():
    names = [f"user{i}" for i in range(1000)]
    one = {s: next(p for p in names if player_split(p) == s) for s in ("train", "val", "test")}
    white = pd.Series([one["test"], one["train"], one["val"]])
    black = pd.Series([one["train"], one["train"], one["test"]])
    split, score_white, score_black = score_sides(white, black)
    assert split.tolist() == ["test", "train", "test"]
    assert score_white.tolist() == [True, True, False]
    assert score_black.tolist() == [False, True, True]

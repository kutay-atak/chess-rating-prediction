import math

import numpy as np
import pandas as pd

from sem_data import indicators, one_per_player

NAN = float("nan")


def test_indicators_on_a_hand_made_game():
    # 3+2, clocks never move (2 s spent, 2 s added back); White's losses: none on move 1, then 1.2, 3.7, then 0s
    clocks = [180] * 20
    evals = [0.0, 0.0, -1.2, -1.2] + [-4.9] * 16  # White's view, after each ply
    ind = indicators(clocks, evals, 180, 2, side=0)
    assert ind is None  # only 9 moves have a loss
    clocks, evals = clocks + [180, 180], evals + [-4.9, -4.9]
    ind = indicators(clocks, evals, 180, 2, side=0)
    losses = np.array([1.2, 3.7] + [0.0] * 8)
    assert math.isclose(ind["log_avg_loss"], np.log1p(losses.mean()))
    assert ind["mistake_rate"] == 0.2 and ind["blunder_rate"] == 0.1 and ind["low_clock_share"] == 0


def test_one_per_player_shares_no_player_or_game():
    cands = pd.DataFrame({"game": [1, 1, 2, 2, 3, 3], "side": [0, 1] * 3,
                          "player": ["a", "b", "a", "c", "b", "c"]})
    picked = one_per_player(cands)
    assert picked.player.is_unique and picked.game.is_unique
    assert picked.equals(one_per_player(cands))

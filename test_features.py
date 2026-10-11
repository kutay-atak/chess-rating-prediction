import math

import numpy as np
import pandas as pd

from features import game_features, rows

NAN = float("nan")
# 3+2. White: 180, 177, 176 (spent 0, 5, 3); Black: 180, 175, 170 (spent 0, 7, 7)
CLOCKS = [180, 180, 177, 175, 176, 170]
# White's 2nd move loses 1.2 (a mistake), its 3rd 3.7 (a blunder); Black's moves lose 0 and 0.2;
# the last ply has no eval (as after a mating move)
EVALS = [0.3, 0.2, -1.0, -0.8, -4.5, NAN]


def side(features, name, t):
    i = np.flatnonzero((features["side"] == name) & (features["t"] == t))[0]
    return {c: v[i] for c, v in features.items()}


def test_game_features():
    f = game_features(CLOCKS, EVALS, 180, 2)
    assert sorted(set(f["t"])) == [2, 5]  # only checkpoints the 6-ply game reaches

    w2, b2 = side(f, "white", 2), side(f, "black", 2)
    assert w2["own_moves"] == 1 and b2["own_moves"] == 1
    assert w2["log_spent_mean"] == 0 and math.isnan(w2["log_spent_sd"]) and w2["fast_share"] == 1
    assert w2["clock_share"] == 1 and w2["has_evals"] and b2["has_evals"]
    assert math.isnan(w2["avg_loss"]) and math.isnan(w2["mistakes"])  # White's first move has no loss
    assert b2["avg_loss"] == 0 and b2["mistakes"] == 0

    w5, b5 = side(f, "white", 5), side(f, "black", 5)
    assert w5["own_moves"] == 3 and b5["own_moves"] == 2
    assert math.isclose(w5["log_spent_mean"], np.mean(np.log1p([0, 5, 3])))
    assert math.isclose(w5["log_spent_sd"], np.std(np.log1p([0, 5, 3]), ddof=1))
    assert math.isclose(w5["fast_share"], 1 / 3) and math.isclose(w5["clock_share"], 176 / 180)
    assert math.isclose(w5["avg_loss"], 2.45) and w5["mistakes"] == 2 and w5["blunders"] == 1
    assert math.isclose(b5["avg_loss"], 0.1) and b5["mistakes"] == 0


def test_unanalysed_game_has_no_eval_features():
    f = game_features(CLOCKS, [NAN] * 6, 180, 2)
    assert not f["has_evals"].any() and np.isnan(f["avg_loss"]).all() and np.isnan(f["blunders"]).all()


def test_zero_base_time_control():
    f = game_features([5, 5, 9, 9], [NAN] * 4, 0, 5)  # 0+5: the first clock shows the increment
    w = side(f, "white", 2)
    assert w["log_spent_mean"] == 0 and math.isnan(w["clock_share"]) and not w["berserk"]


def test_berserk_side():
    f = game_features([90, 180, 88, 177, 87, 175], [NAN] * 6, 180, 2)  # White berserked: half clock, no increment
    w, b = side(f, "white", 5), side(f, "black", 5)
    assert w["berserk"] and not b["berserk"]
    assert math.isclose(w["log_spent_mean"], np.mean(np.log1p([0, 2, 1])))  # spent 0, 90 - 88, 88 - 87
    assert math.isclose(w["clock_share"], 87 / 180)


def test_rows_masks_eco_before_ply_10():
    game = dict(game_id=7, WhiteElo=1500, BlackElo=1600, ECO="C20", TimeControl="180+2", split="test",
                weight=2.0, score_white=True, score_black=False,
                clocks=[180.0] * 12, evals=[NAN] * 12)
    df = rows(pd.DataFrame([game]))
    assert df.eco[df.t < 10].isna().all() and (df.eco[df.t == 10] == "C").all()
    white = df[df.side == "white"]
    assert (white.rating == 1500).all() and white.scored.all() and not df[df.side == "black"].scored.any()

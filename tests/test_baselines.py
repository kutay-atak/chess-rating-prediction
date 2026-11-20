import numpy as np
import pandas as pd

from baselines import conformal, timecontrol_baseline, weighted_quantile


def test_weighted_quantile():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    assert weighted_quantile(values, np.ones(4), 0.5) == 2
    assert weighted_quantile(values, np.array([1.0, 1.0, 1.0, 7.0]), 0.5) == 4  # the heavy value dominates


def test_conformal_uses_weighted_val_residuals_per_checkpoint():
    df = pd.DataFrame({
        "split": ["val"] * 4 + ["test"], "scored": [True, True, True, False, True],
        "t": [2, 2, 2, 2, 2], "rating": [1000.0, 1100, 1300, 5000, 1000], "weight": [1.0, 1, 1, 1, 1]})
    lo, hi = conformal(df, np.full(5, 1000.0))
    # scored val residuals 0, 100, 300 (the unscored 4000 is ignored): 90% quantile 300
    assert lo[-1] == 700 and hi[-1] == 1300


def test_rare_time_controls_fall_back_to_the_overall_mean():
    n = 40
    df = pd.DataFrame({"split": "train", "t": 2, "game_id": range(n + 1), "weight": 1.0,
                       "TimeControl": ["180+0"] * n + ["15+10"], "rating": [1500.0] * n + [2500.0]})
    pred = timecontrol_baseline(df)
    assert pred[0] == 1500 and pred[-1] == np.average(df.rating)

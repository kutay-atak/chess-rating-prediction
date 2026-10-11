"""Three baselines, each writing results/preds_<model>.parquet for the test games.

- mean: the month's mean rating
- timecontrol: the month's mean rating per TimeControl
- lgbm: LightGBM on one side's features after t plies (features.py), one model for every t

Every fit is weighted by N_h/n_h: the strata are bands of the very ratings being predicted, so an
unweighted fit would learn the oversampled sample's rating mix instead of the month's.
Intervals: split conformal per model and checkpoint, from absolute residuals on val sides.

Usage: python baselines.py   (reads sample.parquet)
"""
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

from features import FEATURES, rows

LEVEL = 0.9  # 90% intervals
MIN_TC_GAMES = 30  # a rarer TimeControl's mean rests on one or two players; use the overall mean instead


def weighted_quantile(values, weights, q):
    order = np.argsort(values)
    cum = np.cumsum(weights[order])
    return values[order][min(np.searchsorted(cum, q * cum[-1]), len(cum) - 1)]


def conformal(df, pred):
    """(lo, hi) for every row: pred ± the weighted 90% quantile of |rating - pred| over val scored sides
    at the same t. Weighted so the intervals cover 90% of the month's sides, not the sample's."""
    calib = (df.split == "val") & df.scored
    resid = (df.rating - pred).abs()
    q = {t: weighted_quantile(resid[calib & (df.t == t)].to_numpy(), df.weight[calib & (df.t == t)].to_numpy(), LEVEL)
         for t in df.t.unique()}
    half = df.t.map(q).to_numpy()
    return pred - half, pred + half


def mean_baseline(df):
    train = (df.split == "train") & (df.t == df.t.min())  # one row per train side
    return np.full(len(df), np.average(df.rating[train], weights=df.weight[train]))


def timecontrol_baseline(df):
    train = df[(df.split == "train") & (df.t == df.t.min())]
    means = train.groupby("TimeControl").apply(lambda g: np.average(g.rating, weights=g.weight), include_groups=False)
    games = train.groupby("TimeControl").game_id.nunique()
    means = means[games >= MIN_TC_GAMES]
    return df.TimeControl.map(means).fillna(mean_baseline(df)[0]).to_numpy()


def lgbm_baseline(df):
    X = df[FEATURES].astype({"berserk": int, "has_evals": int})
    train_games = np.sort(df.game_id[df.split == "train"].unique())
    # early stopping on 10% of train games, so the val sides stay untouched for the conformal step
    stop_games = np.random.default_rng(0).choice(train_games, len(train_games) // 10, replace=False)
    stop = df.game_id.isin(stop_games).to_numpy()
    fit = (df.split == "train").to_numpy() & ~stop
    data = lambda m: lgb.Dataset(X[m], df.rating[m], weight=df.weight[m], categorical_feature=["eco"])  # noqa: E731
    dfit = data(fit)
    params = dict(objective="regression", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
                  seed=0, num_threads=os.cpu_count(), verbose=-1)
    model = lgb.train(params, dfit, num_boost_round=2000, valid_sets=[lgb.Dataset(X[stop], df.rating[stop], weight=df.weight[stop],
                      categorical_feature=["eco"], reference=dfit)],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
    return model.predict(X, num_iteration=model.best_iteration)


def main():
    df = rows(pd.read_parquet("sample.parquet"))
    print(f"{len(df):,} side-checkpoint rows", flush=True)
    os.makedirs("results", exist_ok=True)
    test = (df.split == "test").to_numpy()
    for name, fit in [("mean", mean_baseline), ("timecontrol", timecontrol_baseline), ("lgbm", lgbm_baseline)]:
        pred = fit(df)
        lo, hi = conformal(df, pred)
        out = pd.DataFrame({"game_id": df.game_id, "side": df.side, "t": df.t, "mean": pred, "lo": lo, "hi": hi})[test]
        out.to_parquet(f"results/preds_{name}.parquet", index=False)
        print(f"{name}: {len(out):,} test rows -> results/preds_{name}.parquet", flush=True)


if __name__ == "__main__":
    main()

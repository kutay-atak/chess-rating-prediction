"""Kalman filters over a player's own moves: the rating estimate sharpens as moves arrive.

Each move gives an observation z = the mover's rating as predicted by a per-move LightGBM (seconds
spent, clock left, ply), taken as z = rating + e with var(e) = sigma²(ply bin) from val residuals.
The state is the player's rating, constant within a game, with a prior from the time control (the
`timecontrol` baseline's mean, plus the variance of ratings within that TimeControl).
- kalman_iid: independent errors; the filter is then precision-weighted averaging of prior and z's
- kalman_ar1: AR(1) errors between a player's consecutive moves, with rho from val residuals; the
  state is augmented with the current error
Intervals: posterior mean ± 1.645 sd, model-based on purpose, to see whether each model's own
uncertainty is honest.

Usage: python kalman.py   (reads sample.parquet; writes results/preds_kalman_{iid,ar1}.parquet)
"""
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

from baselines import MIN_TC_GAMES
from features import CHECKPOINTS, own_moves

MAX_PLY = max(CHECKPOINTS)  # the filters are scored up to ply 80, so later moves are not needed
SIGNAL = ["log_spent", "clock_share", "ply"]  # no time control: the prior holds it, and every move would count it again
N_MOVES = 2_000_000  # the per-move model's training sample, as PLAN.md sets
BIN_EDGES = [2, 10, 20, 40]  # ply bins for sigma²: the signal changes fastest in the opening (first moves spend no clock)
Z = 1.645  # 90% Gaussian interval
LAGS = 10  # residual autocorrelations printed, to compare with rho^lag


def move_rows(games):
    """One row per move among the first MAX_PLY plies: game index, side (0 White, 1 Black), own-move
    number k (0-based), ply and the SIGNAL features."""
    parts = []
    for i, g in enumerate(games[["clocks", "TimeControl"]].itertuples(index=False)):
        base, inc = map(int, g.TimeControl.split("+"))
        clocks = np.asarray(g.clocks[:MAX_PLY], dtype=float)
        for side in (0, 1):
            spent, _, left, _ = own_moves(clocks, np.full(len(clocks), np.nan), base, inc, side)
            parts.append((np.full(len(spent), i), np.full(len(spent), side), np.arange(len(spent)),
                          np.arange(side, len(clocks), 2), np.log1p(spent),
                          np.clip(left / (base if base > 0 else 40 * inc), 0, 2)))  # 0+N games: Lichess's 40-move estimate
    cols = ["game", "side", "k", "ply", "log_spent", "clock_share"]
    return pd.DataFrame({c: np.concatenate([p[j] for p in parts]) for j, c in enumerate(cols)})


def tc_prior(games, train):
    """(mean, variance) of the rating per TimeControl from train sides, weighted by N_h/n_h; rarer
    TimeControls (fewer than MIN_TC_GAMES train games) get the overall mean and variance."""
    g = games[train]
    sides = pd.DataFrame({"tc": np.concatenate([g.TimeControl, g.TimeControl]),
                          "y": np.concatenate([g.WhiteElo, g.BlackElo]).astype(float),
                          "w": np.concatenate([g.weight, g.weight])})

    def moments(d):
        m = np.average(d.y, weights=d.w)
        return pd.Series({"mean": m, "var": np.average((d.y - m) ** 2, weights=d.w), "games": len(d) / 2})

    by_tc = sides.groupby("tc").apply(moments, include_groups=False)
    overall = moments(sides)
    by_tc = by_tc[by_tc.games >= MIN_TC_GAMES]
    mean = games.TimeControl.map(by_tc["mean"]).fillna(overall["mean"]).to_numpy()
    var = games.TimeControl.map(by_tc["var"]).fillna(overall["var"]).to_numpy()
    return mean, var


def fit_signal(moves, rating, weight, train_games, rng):
    """LightGBM on single moves predicting the mover's rating: N_MOVES random moves of train games,
    weighted by N_h/n_h; early stopping on moves of a held-out 10% of train games, so val stays for sigma² and rho."""
    stop_games = rng.choice(train_games, len(train_games) // 10, replace=False)
    in_train, in_stop = np.isin(moves.game, train_games), np.isin(moves.game, stop_games)
    fit = rng.choice(np.flatnonzero(in_train & ~in_stop), N_MOVES, replace=False)
    stop = np.flatnonzero(in_stop)

    def data(rows, **kw):
        return lgb.Dataset(moves[SIGNAL].iloc[rows], rating[rows], weight=weight[rows], **kw)

    dfit = data(fit)
    params = dict(objective="regression", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, seed=0,
                  num_threads=os.cpu_count(), verbose=-1)
    return lgb.train(params, dfit, num_boost_round=2000, valid_sets=[data(stop, reference=dfit)],
                     callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])


def ply_bin(ply):
    return np.searchsorted(BIN_EDGES, ply, side="right")


def residual_sd(resid, bins, weight):
    """sigma per ply bin: the weighted root mean square of rating - z (z is taken as unbiased)."""
    return np.array([np.sqrt(np.average(resid[bins == b] ** 2, weights=weight[bins == b]))
                     for b in range(len(BIN_EDGES) + 1)])


def autocorrelation(u, seq, k, weight, lag):
    """Weighted correlation of standardised residuals u between own moves k and k + lag of the same
    (game, side) sequence `seq`."""
    d = pd.DataFrame({"seq": seq, "k": k, "u": u, "w": weight})
    pair = d.merge(d.assign(k=d.k - lag), on=["seq", "k"], suffixes=("", "_next"))
    return np.sum(pair.w * pair.u * pair.u_next) / np.sqrt(np.sum(pair.w * pair.u ** 2) * np.sum(pair.w * pair.u_next ** 2))


def filter_iid(z, s, prior_mean, prior_var):
    """Posterior (mean, variance) of the rating after each own move (columns of z; NaN after the
    game's last move) with independent errors: precision-weighted averaging of the prior and the z's."""
    prec = 1 / s ** 2
    total = 1 / prior_var[:, None] + np.nancumsum(np.where(np.isnan(z), 0, prec), axis=1)
    weighted = prior_mean[:, None] / prior_var[:, None] + np.nancumsum(np.where(np.isnan(z), 0, z * prec), axis=1)
    return weighted / total, 1 / total


def filter_ar1(z, s, prior_mean, prior_var, rho):
    """Like filter_iid, with e_k = phi_k e_(k-1) + eta_k, phi_k = rho s_k / s_(k-1), var(eta_k) = s_k² (1 - rho²),
    so every error keeps variance s_k² and consecutive errors correlate by rho. State: (rating, current error)."""
    G, K = z.shape
    x = np.stack([prior_mean, np.zeros(G)], axis=1)
    P = np.zeros((G, 2, 2))
    P[:, 0, 0] = prior_var
    means, variances = np.full((G, K), np.nan), np.full((G, K), np.nan)
    for k in range(K):
        ok = ~np.isnan(z[:, k])
        if k == 0:
            P[:, 1, 1] = s[:, 0] ** 2
        else:  # predict: the rating stays, the error decays toward 0 and gets new noise
            phi = np.where(ok, rho * s[:, k] / np.where(ok, s[:, k - 1], 1), 1)
            x[:, 1] *= phi
            P[:, 0, 1] *= phi
            P[:, 1, 0] *= phi
            P[:, 1, 1] = phi ** 2 * P[:, 1, 1] + np.where(ok, s[:, k] ** 2 * (1 - rho ** 2), 0)
        # update with z = rating + error: H = [1, 1], no extra noise
        S = np.where(ok, P[:, 0, 0] + P[:, 0, 1] + P[:, 1, 0] + P[:, 1, 1], 1)  # 1: ended games are not updated
        gain = np.stack([P[:, 0, 0] + P[:, 0, 1], P[:, 1, 0] + P[:, 1, 1]], axis=1) / S[:, None]
        innov = np.where(ok, z[:, k] - x[:, 0] - x[:, 1], 0)
        x += np.where(ok[:, None], gain * innov[:, None], 0)
        P -= np.where(ok[:, None, None], gain[:, :, None] * gain[:, None, :] * S[:, None, None], 0)
        means[:, k], variances[:, k] = np.where(ok, x[:, 0], np.nan), np.where(ok, P[:, 0, 0], np.nan)
    return means, variances


def grid(values, seq, k, n_seq):
    """[n_seq, MAX_PLY // 2] matrix of per-move values (NaN where a sequence has no k-th move)."""
    out = np.full((n_seq, (MAX_PLY + 1) // 2), np.nan)
    out[seq, k] = values
    return out


def checkpoint_preds(means, variances, game_ids, plies):
    """Preds rows at every checkpoint t a game reaches: the posterior after the side's own moves among the first t plies.
    Rows of means/variances are sequences: game i White is row 2i, Black row 2i + 1."""
    out = []
    for t in CHECKPOINTS:
        for side, name in enumerate(("white", "black")):
            k = (t + 1 - side) // 2  # own moves among the first t plies
            mu, var = means[side::2, k - 1], variances[side::2, k - 1]
            ok = ~np.isnan(mu) & (plies >= t)
            half = Z * np.sqrt(var[ok])
            out.append(pd.DataFrame({"game_id": game_ids[ok], "side": name, "t": t, "mean": mu[ok],
                                     "lo": mu[ok] - half, "hi": mu[ok] + half}))
    return pd.concat(out, ignore_index=True)


def main():
    rng = np.random.default_rng(0)
    games = pd.read_parquet("sample.parquet", columns=["game_id", "WhiteElo", "BlackElo", "TimeControl", "clocks",
                                                       "weight", "split", "score_white", "score_black"])
    split = games.split.to_numpy()
    moves = move_rows(games)
    print(f"{len(moves):,} moves among the first {MAX_PLY} plies", flush=True)
    seq = 2 * moves.game.to_numpy() + moves.side.to_numpy()  # one sequence per (game, side)
    white = moves.side.to_numpy() == 0
    g = moves.game.to_numpy()
    rating = np.where(white, games.WhiteElo.to_numpy()[g], games.BlackElo.to_numpy()[g]).astype(float)
    scored = np.where(white, games.score_white.to_numpy()[g], games.score_black.to_numpy()[g])
    weight = games.weight.to_numpy()[g]

    model = fit_signal(moves, rating, weight, np.flatnonzero(split == "train"), rng)
    rest = np.flatnonzero(np.isin(split[g], ["val", "test"]))
    z = np.full(len(moves), np.nan)
    z[rest] = model.predict(moves[SIGNAL].iloc[rest], num_iteration=model.best_iteration)

    val = (split[g] == "val") & scored
    bins = ply_bin(moves.ply.to_numpy())
    s_bin = residual_sd(rating[val] - z[val], bins[val], weight[val])
    u = (rating[val] - z[val]) / s_bin[bins[val]]
    acf = [autocorrelation(u, seq[val], moves.k.to_numpy()[val], weight[val], lag) for lag in range(1, LAGS + 1)]
    rho = acf[0]
    print("sigma by ply bin", dict(zip(["0-1", "2-9", "10-19", "20-39", "40-79"], s_bin.round(1))), flush=True)
    print(f"rho {rho:.3f}; residual autocorrelation at lags 1-{LAGS}: {np.round(acf, 3).tolist()} "
          f"(AR(1) implies {np.round(rho ** np.arange(1, LAGS + 1), 3).tolist()})", flush=True)

    prior_mean, prior_var = tc_prior(games, split == "train")
    test = np.flatnonzero(split == "test")
    in_test = np.isin(g, test)
    n_seq = 2 * len(games)
    zs = grid(z[in_test], seq[in_test], moves.k.to_numpy()[in_test], n_seq)
    ss = grid(s_bin[bins[in_test]], seq[in_test], moves.k.to_numpy()[in_test], n_seq)
    rows = np.sort(np.concatenate([2 * test, 2 * test + 1]))  # test sequences, White then Black of each game
    pm, pv = np.repeat(prior_mean, 2)[rows], np.repeat(prior_var, 2)[rows]
    plies = games.clocks.map(len).to_numpy()[test]
    os.makedirs("results", exist_ok=True)
    for name, (means, variances) in [("kalman_iid", filter_iid(zs[rows], ss[rows], pm, pv)),
                                     ("kalman_ar1", filter_ar1(zs[rows], ss[rows], pm, pv, rho))]:
        preds = checkpoint_preds(means, variances, games.game_id.to_numpy()[test], plies)
        preds.to_parquet(f"results/preds_{name}.parquet", index=False)
        print(f"{name}: {len(preds):,} test rows -> results/preds_{name}.parquet", flush=True)


if __name__ == "__main__":
    main()

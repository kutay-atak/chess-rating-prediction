"""Score every results/preds_<model>.parquet the same way, on scored test sides.

Per model and checkpoint t:
- mae, coverage of the 90% interval, width (mean hi - lo), n: plain averages over scored test sides
- weighted_mae and se: the stratified estimator of the month's MAE. Unit = test game, y = mean absolute
  error over its scored sides; stratum weights W_h = N_h(t)/N(t), N_h(t) = games in games/ of stratum h
  that reach t plies; variance with finite-population correction
- se_player: the same estimate's SE from resampling whole test players within strata, since one
  player's errors move together and `se` treats every game as independent
- weighted_coverage: the stratified estimator applied to the share of a game's scored sides covered
Writes results/metrics.csv, results/mae.png and results/coverage.png.

Usage: python evaluate.py
"""
import glob
import os

import matplotlib
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

from features import CHECKPOINTS
from sample import LABELS, stratum

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPLICATES = 200  # player bootstrap replicates for se_player
COLORS = {"mean": "#2a78d6", "timecontrol": "#eb6834", "lgbm": "#1baf7a", "gru": "#eda100"}  # validated slots 1-4
MARKERS = {"mean": "o", "timecontrol": "s", "lgbm": "^", "gru": "D"}


def population(folder="games"):
    """N_h(t): games in `folder` per stratum (rows) that reach each checkpoint (columns)."""
    counts = np.zeros((len(LABELS), len(CHECKPOINTS)))
    for path in sorted(glob.glob(f"{folder}/part-*.parquet")):
        table = pq.read_table(path, columns=["WhiteElo", "BlackElo", "moves"])
        h = stratum(table["WhiteElo"].to_numpy(), table["BlackElo"].to_numpy())
        plies = pc.list_value_length(table["moves"]).to_numpy()
        for j, t in enumerate(CHECKPOINTS):
            counts[:, j] += np.bincount(h[plies >= t], minlength=len(LABELS))
    return counts


def stratified(y, h, N, n_all):
    """Stratified mean of unit values y (stratum h) and its SE with finite-population correction.

    N[k]: population units in stratum k; n_all[k]: sampled units of any split. The test games are a domain
    of the stratified sample, so the sampling fraction in the FPC is the whole sample's, n_all/N. Strata
    without test units are left out and the weights renormalised."""
    present = np.array([np.any(h == k) for k in range(len(N))])
    W = np.where(present, N, 0) / np.where(present, N, 0).sum()
    est = var = 0.0
    for k in np.flatnonzero(present):
        yk = y[h == k]
        est += W[k] * yk.mean()
        if len(yk) > 1:
            var += W[k] ** 2 * (1 - n_all[k] / N[k]) * yk.var(ddof=1) / len(yk)
    return est, np.sqrt(var)


def player_bootstrap(err, game, player, h, N, replicates=REPLICATES, seed=0):
    """SE of the stratified MAE when whole players are resampled within strata.

    Each scored side carries err/m and 1/m, m = scored sides in its game, so a stratum's mean of the
    per-game errors is sum(err/m) / sum(1/m) and a game still counts once."""
    m = pd.Series(game).map(pd.Series(game).value_counts()).to_numpy()
    present = np.array([np.any(h == k) for k in range(len(N))])
    W = np.where(present, N, 0) / np.where(present, N, 0).sum()
    rng = np.random.default_rng(seed)
    reps = np.zeros(replicates)
    for k in np.flatnonzero(present):
        sel = h == k
        codes, _ = pd.factorize(player[sel])
        num, den = np.bincount(codes, err[sel] / m[sel]), np.bincount(codes, 1 / m[sel])
        draw = rng.integers(0, len(num), size=(replicates, len(num)))
        reps += W[k] * num[draw].sum(axis=1) / den[draw].sum(axis=1)
    return reps.std(ddof=1)


def truth():
    """Scored test sides: game_id, side, rating, player, stratum index, plies."""
    s = pd.read_parquet("sample.parquet", columns=["game_id", "White", "Black", "WhiteElo", "BlackElo", "split",
                                                   "score_white", "score_black", "moves"])
    s["h"] = stratum(s.WhiteElo, s.BlackElo)
    s["plies"] = s.moves.map(len)
    sides = [s.assign(side=c, rating=s[f"{c.title()}Elo"], player=s[c.title()].str.lower(), scored=s[f"score_{c}"])
             for c in ("white", "black")]
    sides = pd.concat(sides)
    test = sides[(sides.split == "test") & sides.scored]
    n_all = np.array([[((s.h == k) & (s.plies >= t)).sum() for t in CHECKPOINTS] for k in range(len(LABELS))])
    return test[["game_id", "side", "rating", "player", "h"]], n_all


def score(preds, test, N, n_all):
    rows = []
    d = preds.merge(test, on=["game_id", "side"])
    for j, t in enumerate(CHECKPOINTS):
        x = d[d.t == t]
        err = (x.rating - x["mean"]).abs().to_numpy()
        covered = ((x.lo <= x.rating) & (x.rating <= x.hi)).to_numpy().astype(float)
        games = pd.DataFrame({"game_id": x.game_id.to_numpy(), "h": x.h.to_numpy(), "err": err, "covered": covered})
        per_game = games.groupby("game_id").agg(h=("h", "first"), err=("err", "mean"), covered=("covered", "mean"))
        h = per_game.h.to_numpy()
        wmae, se = stratified(per_game.err.to_numpy(), h, N[:, j], n_all[:, j])
        wcov, _ = stratified(per_game.covered.to_numpy(), h, N[:, j], n_all[:, j])
        rows.append(dict(t=t, mae=err.mean(), weighted_mae=wmae, se=se,
                         se_player=player_bootstrap(err, x.game_id.to_numpy(), x.player.to_numpy(), x.h.to_numpy(), N[:, j]),
                         coverage=covered.mean(), weighted_coverage=wcov, width=(x.hi - x.lo).mean(), n=len(x)))
    return pd.DataFrame(rows)


def plot(metrics, column, path, ylabel, title, band=None, line=None, dashed=None, legend=False):
    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for model, m in metrics.groupby("model", sort=False):
        color = COLORS.get(model, "#52514e")
        ax.plot(m.t, m[column], color=color, lw=2, marker=MARKERS.get(model, "o"), ms=6, label=model)
        if band:
            ax.fill_between(m.t, m[column] - 1.96 * m[band], m[column] + 1.96 * m[band], color=color, alpha=0.15, lw=0)
        if dashed:
            ax.plot(m.t, m[dashed], color=color, lw=1.5, ls="--")
        if not legend:  # direct labels at the line ends
            ax.annotate(model, (m.t.iloc[-1], m[column].iloc[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", color="#0b0b0b", fontsize=9)
    if line is not None:
        ax.axhline(line, color="#52514e", lw=1, ls=":")
    ax.set(xlabel="plies seen (t)", ylabel=ylabel, title=title, xticks=CHECKPOINTS)
    ax.grid(axis="y", color="#e4e3df", lw=0.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    if legend:  # one row between the title and the plot, clear of the lines
        ax.legend(frameon=False, ncol=len(COLORS), loc="lower center", bbox_to_anchor=(0.5, 1.0))
        ax.set_title(title, pad=28)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    N = population()
    test, n_all = truth()
    parts = []
    for path in sorted(glob.glob("results/preds_*.parquet")):
        model = os.path.basename(path)[len("preds_"):-len(".parquet")]
        parts.append(score(pd.read_parquet(path), test, N, n_all).assign(model=model))
        print(f"scored {model}", flush=True)
    order = [m for m in COLORS if any(p.model.iloc[0] == m for p in parts)]
    metrics = pd.concat(parts)
    metrics = metrics.set_index("model").loc[order + sorted(set(metrics.model) - set(order))].reset_index()
    metrics.to_csv("results/metrics.csv", index=False, float_format="%.4f")
    print(metrics.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    plot(metrics, "weighted_mae", "results/mae.png", "MAE (rating points)",
         "Weighted MAE on unseen players (band: ±1.96 player-bootstrap SE)", band="se_player")
    plot(metrics, "weighted_coverage", "results/coverage.png", "share of sides inside the interval",
         "Coverage of the 90% intervals (solid: weighted, dashed: unweighted)", line=0.9, dashed="coverage",
         legend=True)  # the weighted lines all sit on 0.9, so end labels would collide


if __name__ == "__main__":
    main()

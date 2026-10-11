"""Score every results/preds_<model>.parquet the same way, on scored test sides.

Per model and checkpoint t:
- mae, coverage of the 90% interval, width (mean hi - lo), n: plain averages over scored test sides
- weighted_mae and se: the stratified estimator of the month's MAE. Unit = test game, y = mean absolute
  error over its scored sides; stratum weights W_h = N_h(t)/N(t), N_h(t) = games in games/ of stratum h
  that reach t plies; variance with finite-population correction
- se_player: the same estimate's SE from resampling whole test players within strata, since one
  player's errors move together and `se` treats every game as independent
- se_shard: the same estimate's SE from deleting one downloaded shard at a time (jackknife), since the
  12 shards are a random sample of the month's and games in a shard share its hours of play
- weighted_mae_80: the same estimator over games that reach ply 80 only, so the curve follows the same
  games at every t and shows the gain from more plies alone
- weighted_coverage, interval_score: the stratified estimator applied to the share of a game's scored
  sides covered, and to the interval score (width plus 2/0.1 times any miss; lower is better)
Per model, t and band of the side's own rating: bias, MAE, coverage and width (by_band).
Writes results/metrics.csv, results/metrics_by_band.csv, results/mae.png, results/coverage.png, results/width.png and
results/kalman_coverage.png (the two Kalman filters' model-based intervals, against the GRU's calibrated ones).

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
LEVEL = 0.9  # the intervals' nominal coverage
BANDS = [1000, 1400, 1800, 2200]  # 400-point bands of a side's own rating: several thousand test sides each
BAND_LABELS = ["<1000", "1000-1399", "1400-1799", "1800-2199", "2200+"]
COLORS = {"mean": "#2a78d6", "timecontrol": "#eb6834", "lgbm": "#1baf7a", "gru": "#eda100", "kalman_iid": "#e87ba4",
          "kalman_ar1": "#008300"}  # validated categorical slots 1-6
MARKERS = {"mean": "o", "timecontrol": "s", "lgbm": "^", "gru": "D", "kalman_iid": "v", "kalman_ar1": "P"}


def population(folder="games"):
    """N_h(t) per shard: games in each part of `folder` (axis 0, the `shard` of sample.py) per stratum
    (axis 1) that reach each checkpoint (axis 2)."""
    paths = sorted(glob.glob(f"{folder}/part-*.parquet"))
    counts = np.zeros((len(paths), len(LABELS), len(CHECKPOINTS)))
    for i, path in enumerate(paths):
        table = pq.read_table(path, columns=["WhiteElo", "BlackElo", "moves"])
        h = stratum(table["WhiteElo"].to_numpy(), table["BlackElo"].to_numpy())
        plies = pc.list_value_length(table["moves"]).to_numpy()
        for j, t in enumerate(CHECKPOINTS):
            counts[i, :, j] = np.bincount(h[plies >= t], minlength=len(LABELS))
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


def shard_jackknife(y, h, shard, N_shard):
    """SE of the stratified mean of y from deleting one shard at a time (JK1): both its test games and its
    games in N. N_shard[i, k]: population units of stratum k in shard i."""
    S = len(N_shard)
    est = np.array([stratified(y[shard != i], h[shard != i], N_shard.sum(0) - N_shard[i], np.zeros(N_shard.shape[1]))[0]
                    for i in range(S)])
    return np.sqrt((S - 1) / S * ((est - est.mean()) ** 2).sum())


def truth():
    """Scored test sides: game_id, side, rating, player, stratum index, shard, plies, weight."""
    s = pd.read_parquet("sample.parquet", columns=["game_id", "White", "Black", "WhiteElo", "BlackElo", "split",
                                                   "score_white", "score_black", "moves", "shard", "weight"])
    s["h"] = stratum(s.WhiteElo, s.BlackElo)
    s["plies"] = s.moves.map(len)
    sides = [s.assign(side=c, rating=s[f"{c.title()}Elo"], player=s[c.title()].str.lower(), scored=s[f"score_{c}"])
             for c in ("white", "black")]
    sides = pd.concat(sides)
    test = sides[(sides.split == "test") & sides.scored]
    n_all = np.array([[((s.h == k) & (s.plies >= t)).sum() for t in CHECKPOINTS] for k in range(len(LABELS))])
    return test[["game_id", "side", "rating", "player", "h", "shard", "plies", "weight"]], n_all


def interval_score(lo, hi, y, level=LEVEL):
    """Width plus 2/alpha times the miss (Gneiting and Raftery 2007): rewards narrow intervals only if they still cover."""
    return hi - lo + 2 / (1 - level) * (np.maximum(lo - y, 0) + np.maximum(y - hi, 0))


def score(preds, test, N_shard, n_all):
    N = N_shard.sum(0)
    rows = []
    d = preds.merge(test, on=["game_id", "side"])
    for j, t in enumerate(CHECKPOINTS):
        x = d[d.t == t]
        err = (x.rating - x["mean"]).abs().to_numpy()
        covered = ((x.lo <= x.rating) & (x.rating <= x.hi)).to_numpy().astype(float)
        games = pd.DataFrame({"game_id": x.game_id.to_numpy(), "h": x.h.to_numpy(), "shard": x.shard.to_numpy(),
                              "plies": x.plies.to_numpy(), "err": err, "covered": covered,
                              "iscore": interval_score(x.lo, x.hi, x.rating).to_numpy(), "width": (x.hi - x.lo).to_numpy()})
        g = games.groupby("game_id").agg(h=("h", "first"), shard=("shard", "first"), plies=("plies", "first"),
                                         err=("err", "mean"), covered=("covered", "mean"), iscore=("iscore", "mean"),
                                         width=("width", "mean"))
        h = g.h.to_numpy()
        wmae, se = stratified(g.err.to_numpy(), h, N[:, j], n_all[:, j])
        long = g.plies.to_numpy() >= CHECKPOINTS[-1]
        rows.append(dict(
            t=t, mae=err.mean(), weighted_mae=wmae, se=se,
            se_player=player_bootstrap(err, x.game_id.to_numpy(), x.player.to_numpy(), x.h.to_numpy(), N[:, j]),
            se_shard=shard_jackknife(g.err.to_numpy(), h, g.shard.to_numpy(), N_shard[:, :, j]),
            weighted_mae_80=stratified(g.err.to_numpy()[long], h[long], N[:, -1], n_all[:, -1])[0],
            coverage=covered.mean(), weighted_coverage=stratified(g.covered.to_numpy(), h, N[:, j], n_all[:, j])[0],
            width=(x.hi - x.lo).mean(), weighted_half_width=stratified(g.width.to_numpy(), h, N[:, j], n_all[:, j])[0] / 2,
            interval_score=stratified(g.iscore.to_numpy(), h, N[:, j], n_all[:, j])[0],
            n=len(x)))
    return pd.DataFrame(rows)


def by_band(preds, test):
    """Per t and band of the side's own rating: n, bias (mean - rating), MAE, coverage and width. Each is a
    mean weighted by the game weight N_h/n_h, so a band is summarised as it is in the month, not as sampled."""
    d = preds.merge(test, on=["game_id", "side"])
    d["band"] = pd.Categorical(np.array(BAND_LABELS)[np.searchsorted(BANDS, d.rating, side="right")], BAND_LABELS)
    d["bias"] = d["mean"] - d.rating
    d["mae"], d["coverage"], d["width"] = d.bias.abs(), (d.lo <= d.rating) & (d.rating <= d.hi), d.hi - d.lo
    cols = ["bias", "mae", "coverage", "width"]
    return d.groupby(["t", "band"], observed=True).apply(
        lambda g: pd.Series({"n": len(g), **{c: np.average(g[c], weights=g.weight) for c in cols}}),
        include_groups=False).reset_index()


def plot(metrics, column, path, ylabel, title, band=None, line=None, dashed=None, legend=False):
    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ends = []
    for model, m in metrics.groupby("model", sort=False):
        color = COLORS.get(model, "#52514e")
        ax.plot(m.t, m[column], color=color, lw=2, marker=MARKERS.get(model, "o"), ms=6, label=model)
        if band:
            ax.fill_between(m.t, m[column] - 1.96 * m[band], m[column] + 1.96 * m[band], color=color, alpha=0.15, lw=0)
        if dashed:
            ax.plot(m.t, m[dashed], color=color, lw=1.5, ls="--")
        ends.append([m[column].iloc[-1], model])
    if not legend:  # direct labels at the line ends, pushed apart where lines end close together
        low, high = ax.get_ylim()
        ends.sort()
        for i in range(1, len(ends)):
            ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.045 * (high - low))
        for y, model in ends:
            ax.annotate(model, (CHECKPOINTS[-1], y), xytext=(6, 0), textcoords="offset points", va="center",
                        color="#0b0b0b", fontsize=9, annotation_clip=False)
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
    N_shard = population()
    test, n_all = truth()
    parts, bands = [], []
    for path in sorted(glob.glob("results/preds_*.parquet")):
        model = os.path.basename(path)[len("preds_"):-len(".parquet")]
        preds = pd.read_parquet(path)
        parts.append(score(preds, test, N_shard, n_all).assign(model=model))
        bands.append(by_band(preds, test).assign(model=model))
        print(f"scored {model}", flush=True)
    order = [m for m in COLORS if any(p.model.iloc[0] == m for p in parts)]
    order += sorted({p.model.iloc[0] for p in parts} - set(order))
    metrics = pd.concat(parts).set_index("model").loc[order].reset_index()
    metrics.to_csv("results/metrics.csv", index=False, float_format="%.4f")
    print(metrics.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    bands = pd.concat(bands).set_index("model").loc[order].reset_index()
    bands.to_csv("results/metrics_by_band.csv", index=False, float_format="%.4f")
    metrics = metrics[metrics.model.isin(COLORS)]  # other models (gru_v1, the first GRU) stay in the CSVs only
    plot(metrics, "weighted_mae", "results/mae.png", "MAE (rating points)",
         "Weighted MAE on unseen players (band: ±1.96 player-bootstrap SE)", band="se_player")
    calibrated = metrics[~metrics.model.str.startswith("kalman")]  # the Kalman filters get their own plot below
    plot(calibrated, "weighted_coverage", "results/coverage.png", "share of sides inside the interval",
         "Coverage of the 90% intervals (solid: weighted, dashed: unweighted)", line=0.9, dashed="coverage",
         legend=True)  # the weighted lines all sit on 0.9, so end labels would collide
    plot(metrics[metrics.model.isin(["timecontrol", "lgbm", "gru"])], "weighted_half_width", "results/width.png",
         "half-width of the 90% interval (rating points)", "How fast the 90% intervals narrow (all calibrated to cover 90%)")
    kalman = metrics[metrics.model.isin(["gru", "kalman_iid", "kalman_ar1"])]
    if kalman.model.str.startswith("kalman").any():
        plot(kalman, "weighted_coverage", "results/kalman_coverage.png", "share of sides inside the interval",
             "Weighted coverage of the 90% intervals: do correlated errors explain the gap?", line=0.9)


if __name__ == "__main__":
    main()

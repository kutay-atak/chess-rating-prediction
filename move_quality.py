"""Each player's rating from the quality of their own moves only.

From games in games/ that Lichess analysed, a table of how likely each size of move loss is for a player
of each rating, in the opening, middlegame and endgame. A player's estimate starts from the month's rating
distribution and, after each of their own moves, is multiplied by that move's likelihood at every rating
(tempered by alpha, since one player's moves are not independent evidence). The two players never share
anything, so a good move raises only the mover and the gap between them can grow.

- table from train players, alpha per checkpoint from val players (so 90% intervals cover 90%),
  scores on unseen test players, including how well the gap between the two players is found
- writes results/move_quality.csv (test metrics, with the GRU on the same games) and web/quality.json

Usage: python move_quality.py
"""
import glob
import json

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

from features import CHECKPOINTS
from split import game_split, player_split

GRID = np.arange(600, 3001, 50.0)  # rating grid (Lichess blitz ratings sit inside 600-3000)
LOSS_EDGES = [0.05, 0.2, 0.5, 1, 2, 4]  # pawns: best/near-best, small, inaccuracy, mistake, blunder, big blunder
PHASE_EDGES = [20, 60]  # plies: opening, middlegame, endgame lose at different rates
ALPHAS = np.round(np.arange(0.02, 1.01, 0.02), 2)
LEVEL = 0.9
MAX_PLY = max(CHECKPOINTS)


def move_losses(evals):
    """Loss of each ply's mover in pawns (CLAUDE.md conventions), NaN where an eval is missing or on White's first move."""
    evals = np.asarray(evals, dtype=float)
    before = np.concatenate([[np.nan], evals[:-1]])
    sign = np.where(np.arange(len(evals)) % 2 == 0, 1.0, -1.0)
    return np.clip(sign * (before - evals), 0, None)


def bins(loss, ply):
    return np.searchsorted(LOSS_EDGES, loss, side="right"), np.searchsorted(PHASE_EDGES, ply, side="right")


def rating_index(r):
    return np.clip(np.round((np.asarray(r, float) - GRID[0]) / 50).astype(int), 0, len(GRID) - 1)


def table(counts):
    """log P(loss bin | rating, phase) from counts[phase, rating, loss bin]: add-one, then smoothed over ±100 points."""
    p = (counts + 1) / (counts + 1).sum(axis=2, keepdims=True)
    kernel = np.array([1, 2, 3, 2, 1]) / 9
    padded = np.pad(p, ((0, 0), (2, 2), (0, 0)), mode="edge")
    smooth = sum(k * padded[:, i:i + p.shape[1]] for i, k in enumerate(kernel))
    return np.log(smooth / smooth.sum(axis=2, keepdims=True))


def posterior(log_prior, log_lik, loss_bins, phases, alpha):
    """Rating posterior on GRID after a player's moves (loss bin and phase of each, NaN-loss moves skipped)."""
    lp = log_prior + alpha * log_lik[phases, :, loss_bins].sum(axis=0)
    p = np.exp(lp - lp.max())
    return p / p.sum()


def summary(p, level=LEVEL):
    """(lo, mean, hi): posterior mean and central `level` interval, interpolated on the grid."""
    cdf = np.cumsum(p)
    lo, hi = np.interp([(1 - level) / 2, (1 + level) / 2], cdf, GRID)
    return lo, float(p @ GRID), hi


def side_moves(evals, side, t):
    """(loss bins, phases) of `side`'s moves with a loss among the first t plies."""
    loss = move_losses(evals[:t])
    ply = np.arange(len(loss))
    keep = (ply % 2 == side) & ~np.isnan(loss)
    return bins(loss[keep], ply[keep])


def collect():
    """Counts from train players' moves, and val/test sides (rating, evals, game id) for calibration and scoring."""
    counts = np.zeros((len(PHASE_EDGES) + 1, len(GRID), len(LOSS_EDGES) + 1))
    held, prior = [], np.zeros(len(GRID))
    for shard, path in enumerate(sorted(glob.glob("games/part-*.parquet"))):
        t = pq.read_table(path, columns=["White", "Black", "WhiteElo", "BlackElo", "evals"])
        flat = pc.list_flatten(t["evals"]).to_numpy(zero_copy_only=False)
        parents = pc.list_parent_indices(t["evals"]).to_numpy()
        analysed = np.flatnonzero(np.bincount(parents[~np.isnan(flat)], minlength=len(t)) > 0)
        g = t.take(analysed).to_pandas()
        split = game_split(g.White, g.Black)
        for i, row in enumerate(g.itertuples(index=False)):
            evals = np.asarray(row.evals, dtype=float)
            for side, (name, elo) in enumerate([(row.White, row.WhiteElo), (row.Black, row.BlackElo)]):
                if player_split(name) != split[i]:
                    continue  # only sides whose player belongs to the game's split
                if split[i] == "train":
                    loss = move_losses(evals)
                    ply = np.arange(len(loss))
                    keep = (ply % 2 == side) & ~np.isnan(loss)
                    lb, ph = bins(loss[keep], ply[keep])
                    np.add.at(counts, (ph, rating_index(elo), lb), 1)
                    prior[rating_index(elo)] += 1
                else:
                    held.append((split[i], f"{shard}-{analysed[i]}", side, float(elo), evals[:MAX_PLY]))
        print(f"{path}: {len(g):,} analysed games", flush=True)
    return counts, prior, pd.DataFrame(held, columns=["split", "game", "side", "rating", "evals"])


def estimates(sides, log_prior, log_lik, alpha_of_t):
    """(lo, mean, hi) for every held-out side at every checkpoint its game reaches."""
    rows = []
    for s in sides.itertuples(index=False):
        for t in CHECKPOINTS:
            if t > len(s.evals):
                break
            lb, ph = side_moves(s.evals, s.side, t)
            lo, mean, hi = summary(posterior(log_prior, log_lik, lb, ph, alpha_of_t[t]))
            rows.append((s.game, s.side, t, s.rating, lo, mean, hi))
    return pd.DataFrame(rows, columns=["game", "side", "t", "rating", "lo", "mean", "hi"])


def calibrate(val, log_prior, log_lik):
    """alpha per checkpoint: the largest (sharpest) alpha whose 90% intervals still cover 90% of val sides."""
    best = {}
    for t in CHECKPOINTS:
        v = val[val.evals.map(len) >= t]
        moves = [side_moves(s.evals, s.side, t) for s in v.itertuples(index=False)]
        best[t] = ALPHAS[0]
        for a in ALPHAS:
            cov = np.mean([lo <= r <= hi for (lb, ph), r in zip(moves, v.rating)
                           for lo, _, hi in [summary(posterior(log_prior, log_lik, lb, ph, a))]])
            if cov < LEVEL:
                break
            best[t] = a
    return best


def gap_metrics(est):
    """Per t: per-side MAE and coverage, and how well White - Black is found."""
    out = []
    for t, x in est.groupby("t"):
        w, b = x[x.side == 0].set_index("game"), x[x.side == 1].set_index("game")
        both = w.index.intersection(b.index)
        pred, true = (w["mean"] - b["mean"])[both], (w.rating - b.rating)[both]
        big = true.abs() > 200
        out.append(dict(t=t, mae=(x.rating - x["mean"]).abs().mean(),
                        coverage=((x.lo <= x.rating) & (x.rating <= x.hi)).mean(),
                        gap_corr=np.corrcoef(pred, true)[0, 1], gap_mae=(pred - true).abs().mean(),
                        stronger_found=(np.sign(pred[big]) == np.sign(true[big])).mean(), n=len(x)))
    return pd.DataFrame(out)


def main():
    counts, prior, held = collect()
    log_lik = table(counts)
    log_prior = np.log((prior + 1) / (prior + 1).sum())
    val, test = held[held.split == "val"], held[held.split == "test"]
    print(f"{int(counts.sum()):,} train moves; {len(val):,} val and {len(test):,} test sides", flush=True)
    alpha = calibrate(val.sample(min(len(val), 20_000), random_state=0), log_prior, log_lik)
    print("alpha by checkpoint", alpha, flush=True)
    metrics = gap_metrics(estimates(test, log_prior, log_lik, alpha)).assign(model="move_quality")
    metrics.to_csv("results/move_quality.csv", index=False, float_format="%.4f")
    print(metrics.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    with open("web/quality.json", "w") as f:
        json.dump({"grid": GRID.tolist(), "loss_edges": LOSS_EDGES, "phase_edges": PHASE_EDGES,
                   "log_prior": log_prior.round(5).tolist(), "log_lik": log_lik.round(5).tolist(),
                   "alpha": {str(k): v for k, v in alpha.items()}, "checkpoints": CHECKPOINTS}, f)


if __name__ == "__main__":
    main()

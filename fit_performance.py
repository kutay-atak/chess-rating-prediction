"""Fit the accuracy -> rating table behind the per-game performance rating (performance.py), then check it.

From Lichess games with engine evals (train players only): for every player after each of their own
moves k, their running game accuracy and their rating. Per (k, accuracy bin): weighted median and
10th / 90th percentile rating, widening to neighbouring bins until a cell has enough players, then
made monotone in accuracy. Analysed games are weighted to the rating mix of all games, since players
who ask for analysis aren't typical.

Checks on test players' analysed games: the mover's drop on a blunder (and that the opponent's
estimate doesn't move), error against the players' real ratings, and the same with evals from our
own Stockfish instead of Lichess's.

Usage: python fit_performance.py games [--stockfish-games 200]
Writes model/performance.json and reports/performance.json.
"""
import argparse
import glob
import json
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from performance import Book, Table, estimates, game_accuracy, move_accuracy, win_percent
from split import game_split

K_MAX, STEP, MIN_CELL = 40, 2, 300
OUT, REPORT, BOOK = "model/performance.json", "reports/performance.json", "model/opening_book.npz"


def load(folder, split, columns, analysed=True, limit=None):
    """Games of one split (only those Lichess analysed, if `analysed`), at most `limit`."""
    frames, total = [], 0
    for path in sorted(glob.glob(f"{folder}/*.parquet")):
        df = pq.read_table(path, columns=["White", "Black"] + columns).to_pandas()
        df = df[game_split(df.White, df.Black).to_numpy() == split]
        if analysed:
            df = df[df.evals.notna()]
        frames.append(df[columns])
        total += len(df)
        if limit and total >= limit:
            break
    df = pd.concat(frames, ignore_index=True)
    return df if limit is None else df.iloc[:limit]


def rating_weights(all_ratings, ratings):
    """Weight per rating so analysed players match the rating mix of all players."""
    bins = lambda r: np.clip((np.asarray(r) - 400) // 100, 0, 25).astype(int)  # noqa: E731
    target = np.bincount(bins(all_ratings), minlength=26).astype(float)
    have = np.bincount(bins(ratings), minlength=26).astype(float)
    ratio = (target / target.sum()) / np.maximum(have / have.sum(), 1e-9)
    return np.clip(ratio, 0.1, 10)[bins(ratings)]


def rows(games):
    """(k, running accuracy, rating) for every player after every own move."""
    ks, accs, ratings = [], [], []
    for evals, w, b in zip(games.evals, games.WhiteElo, games.BlackElo):
        acc = move_accuracy(evals)
        for side, rating in ((0, w), (1, b)):
            running = game_accuracy(acc[side::2])
            ks.append(np.minimum(np.arange(1, len(running) + 1), K_MAX))
            accs.append(running)
            ratings.append(np.full(len(running), rating))
    return np.concatenate(ks), np.concatenate(accs), np.concatenate(ratings).astype(np.float64)


def weighted_quantiles(values, weights, qs):
    order = np.argsort(values)
    cum = np.cumsum(weights[order])
    return [values[order][min(np.searchsorted(cum, q * cum[-1]), len(cum) - 1)] for q in qs]


def fit(k, acc, rating, weight):
    n_bins = 100 // STEP
    out = {name: np.zeros((K_MAX + 1, n_bins)) for name in ("low", "median", "high")}
    for name, value in zip(("low", "median", "high"), weighted_quantiles(rating, weight, (0.1, 0.5, 0.9))):
        out[name][0] = value  # before any own move: everyone
    for kk in range(1, K_MAX + 1):
        sel = k == kk
        a, r, w = acc[sel], rating[sel], weight[sel]
        bins = np.clip((a // STEP).astype(int), 0, n_bins - 1)
        order = np.argsort(bins, kind="stable")
        bins, r, w = bins[order], r[order], w[order]
        starts = np.searchsorted(bins, np.arange(n_bins + 1))  # rows of bin b: starts[b]:starts[b + 1]
        for b in range(n_bins):
            lo = hi = b
            while starts[hi + 1] - starts[lo] < MIN_CELL and (lo > 0 or hi < n_bins - 1):
                lo, hi = max(lo - 1, 0), min(hi + 1, n_bins - 1)
            cell = slice(starts[lo], starts[hi + 1])
            for name, value in zip(("low", "median", "high"), weighted_quantiles(r[cell], w[cell], (0.1, 0.5, 0.9))):
                out[name][kk, b] = value
    for name, t in out.items():  # a more accurate game never means a lower rating; averaging both sides avoids bias
        t[1:] = (np.maximum.accumulate(t[1:], axis=1) + np.minimum.accumulate(t[1:, ::-1], axis=1)[:, ::-1]) / 2
    return {"step": STEP, **{name: np.round(t).tolist() for name, t in out.items()}}


def check(games, table, book):
    """Blunder drops and errors against the real ratings on these games."""
    drops, other, end_err, m20_err, covered = [], [], [], [], []
    for sans, evals, w, b in zip(games.moves, games.evals, games.WhiteElo, games.BlackElo):
        est = estimates(list(sans), list(evals), table, book)
        white = win_percent([20] + list(evals))
        for t in range(1, len(est)):
            mover, opp = ("white", "black") if t % 2 else ("black", "white")
            if (white[t - 1] - white[t]) * (1 if mover == "white" else -1) >= 30:  # gave away 30% win chance
                drops.append(est[t - 1][mover]["rating"] - est[t][mover]["rating"])
                other.append(est[t][opp]["rating"] - est[t - 1][opp]["rating"])
        last = est[-1]
        end_err += [abs(last["white"]["rating"] - w), abs(last["black"]["rating"] - b)]
        covered += [last[c]["low"] <= r <= last[c]["high"] for c, r in (("white", w), ("black", b))]
        if len(est) > 40:
            m20_err += [abs(est[40]["white"]["rating"] - w), abs(est[40]["black"]["rating"] - b)]
    return {"games": len(games), "blunders": len(drops),
            "mover's drop on a blunder (median)": float(np.median(drops)) if drops else None,
            "opponent's change on that move (largest)": float(np.max(np.abs(other))) if other else None,
            "MAE vs real rating, end of game": round(float(np.mean(end_err)), 1),
            "MAE vs real rating, after move 20": round(float(np.mean(m20_err)), 1) if m20_err else None,
            "real rating inside the 10-90% range, end of game": round(float(np.mean(covered)), 3)}


def mean_estimate(games, table, book):
    """Average end estimate per player, to compare the same games under two sets of evals."""
    return float(np.mean([[e["white"]["rating"], e["black"]["rating"]] for e in
                          (estimates(list(s), list(v), table, book)[-1] for s, v in zip(games.moves, games.evals))]))


_engine = None


def _stockfish(sans):
    global _engine
    if _engine is None:
        from web.engine import Engine
        _engine = Engine()
    return _engine.evals(sans)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("games", help="folder of prepared parquet games (with evals)")
    ap.add_argument("--test-games", type=int, default=10_000)
    ap.add_argument("--stockfish-games", type=int, default=200)
    args = ap.parse_args()

    every = load(args.games, "train", ["WhiteElo", "BlackElo"], analysed=False).to_numpy().ravel()
    train = load(args.games, "train", ["WhiteElo", "BlackElo", "evals"])
    print(f"{len(train):,} analysed train games out of {len(every) // 2:,}", flush=True)
    k, acc, rating = rows(train)
    del train
    table_data = fit(k, acc, rating, rating_weights(every, rating))
    del k, acc, rating
    os.makedirs("model", exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(table_data, f)
    print(f"wrote {OUT}", flush=True)

    table = Table(table_data)
    book = Book(BOOK) if os.path.exists(BOOK) else None
    test = load(args.games, "test", ["WhiteElo", "BlackElo", "moves", "evals"], limit=args.test_games)
    report = {"test, Lichess evals": check(test, table, book)}
    print(json.dumps(report, indent=2), flush=True)

    if args.stockfish_games:
        sample = test.iloc[:args.stockfish_games].copy()
        for col in ("moves", "evals"):
            sample[col] = [list(x[:40]) for x in sample[col]]
        lichess = mean_estimate(sample, table, book)
        report["first 20 moves, Lichess evals"] = check(sample, table, book)
        with Pool(os.cpu_count()) as pool:
            sample["evals"] = pool.map(_stockfish, sample.moves, chunksize=4)
        report["first 20 moves, our Stockfish"] = check(sample, table, book)
        report["first 20 moves, mean estimate: our Stockfish minus Lichess evals"] = round(
            mean_estimate(sample, table, book) - lichess, 1)
    os.makedirs("reports", exist_ok=True)
    with open(REPORT, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

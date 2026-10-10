"""Fit the accuracy -> rating table behind the per-game performance rating (performance.py), then check it.

From Lichess games with engine evals (train players only): for every player after every own move k,
their running game accuracy and their rating. Per (k, accuracy bin): weighted median and 10th / 90th
percentile rating, widening to neighbouring bins until a cell has enough players, then made
monotone in accuracy. Analysed games are weighted to the rating mix of all games, since players who
ask for analysis aren't typical. The running-accuracy metric ("mean" or "blend") is picked by
end-of-game error on val games.

Checks on test players' analysed games: drop in the mover's estimate on blunders, how well the two
players' estimates separate when their ratings differ a lot, error against the true rating (next to
the moves-and-clocks model), and the same with evals from our own Stockfish instead of Lichess's.

Usage: python fit_performance.py games2 [--stockfish-games 300]
Writes model/performance.json and reports/performance.json.
"""
import argparse
import glob
import json
import os
from multiprocessing import Pool

import numpy as np
import pyarrow.parquet as pq

from performance import Book, Table, estimates, game_accuracy, move_accuracy
from split import game_split
from train import load_games

K_MAX, STEP, MIN_CELL = 40, 2, 300
OUT, REPORT, BOOK = "model/performance.json", "reports/performance.json", "model/opening_book.npz"
COLUMNS = ["WhiteElo", "BlackElo", "moves", "evals"]


def rating_weights(folder, ratings):
    """Weight per rating so analysed players match the rating mix of all train players."""
    every = []
    for path in sorted(glob.glob(f"{folder}/*.parquet")):
        df = pq.read_table(path, columns=["White", "Black", "WhiteElo", "BlackElo"]).to_pandas()
        df = df[game_split(df.White, df.Black).to_numpy() == "train"]
        every.append(df[["WhiteElo", "BlackElo"]].to_numpy().ravel())
    bins = lambda r: np.clip((np.asarray(r) - 400) // 100, 0, 25).astype(int)  # noqa: E731
    target = np.bincount(bins(np.concatenate(every)), minlength=26).astype(float)
    have = np.bincount(bins(ratings), minlength=26).astype(float)
    ratio = (target / target.sum()) / np.maximum(have / have.sum(), 1e-9)
    return lambda r: np.clip(ratio, 0.1, 10)[bins(r)]


def rows(games, metric, weight):
    """(k, running accuracy, rating, weight) for every player after every own move."""
    ks, accs, ratings = [], [], []
    for evals, w, b in zip(games.evals, games.WhiteElo, games.BlackElo):
        acc = move_accuracy(evals)
        for side, rating in ((0, w), (1, b)):
            running = game_accuracy(acc[side::2], metric)
            ks.append(np.minimum(np.arange(1, len(running) + 1), K_MAX))
            accs.append(running)
            ratings.append(np.full(len(running), rating))
    ratings = np.concatenate(ratings).astype(np.float64)
    return np.concatenate(ks), np.concatenate(accs), ratings, weight(ratings)


def weighted_quantiles(values, weights, qs):
    order = np.argsort(values)
    cum = np.cumsum(weights[order])
    return [values[order][min(np.searchsorted(cum, q * cum[-1]), len(cum) - 1)] for q in qs]


def fit(k, acc, rating, weight, metric):
    n_bins = 100 // STEP
    out = {name: np.zeros((K_MAX + 1, n_bins)) for name in ("low", "median", "high")}
    prior = weighted_quantiles(rating, weight, (0.1, 0.5, 0.9))
    for name, value in zip(("low", "median", "high"), prior):
        out[name][0] = value  # before any own move: everyone
    for kk in range(1, K_MAX + 1):
        sel = k == kk
        a, r, w = acc[sel], rating[sel], weight[sel]
        bins = np.clip((a // STEP).astype(int), 0, n_bins - 1)
        counts = np.bincount(bins, minlength=n_bins)
        for b in range(n_bins):
            lo = hi = b
            while counts[lo:hi + 1].sum() < MIN_CELL and (lo > 0 or hi < n_bins - 1):
                lo, hi = max(lo - 1, 0), min(hi + 1, n_bins - 1)
            cell = (bins >= lo) & (bins <= hi)
            for name, value in zip(("low", "median", "high"), weighted_quantiles(r[cell], w[cell], (0.1, 0.5, 0.9))):
                out[name][kk, b] = value
    for name in out:  # a more accurate game never means a lower rating
        out[name][1:] = np.maximum.accumulate(out[name][1:], axis=1)
    return {"metric": metric, "step": STEP, **{name: np.round(t).tolist() for name, t in out.items()}}


def end_of_game_mae(games, table):
    err = []
    for sans, evals, w, b in zip(games.moves, games.evals, games.WhiteElo, games.BlackElo):
        last = estimates(list(sans), list(evals), table)[-1]
        err += [abs(last["white"]["rating"] - w), abs(last["black"]["rating"] - b)]
    return float(np.mean(err))


def check(games, table, book, fallback=None):
    """Blunder drops, gap separation and errors on these games."""
    drops, other, end_err, m20_err, covered, true_gap, pred_gap, fb_gap, fb_end, fb_m20 = ([] for _ in range(10))
    for sans, evals, clocks, tc, w, b in zip(games.moves, games.evals, games.clocks, games.TimeControl,
                                             games.WhiteElo, games.BlackElo):
        sans, evals = list(sans), list(evals)
        est = estimates(sans, evals, table, book)
        white = 100 / (1 + np.exp(-0.00368208 * np.asarray([20] + evals, dtype=float)))
        for t in range(1, len(est)):
            mover, opp = ("white", "black") if t % 2 else ("black", "white")
            lost = (white[t - 1] - white[t]) * (1 if mover == "white" else -1)
            if lost >= 30 and t > 10:  # a blunder after the first five moves
                drops.append(est[t - 1][mover]["rating"] - est[t][mover]["rating"])
                other.append(est[t][opp]["rating"] - est[t - 1][opp]["rating"])
        last = est[-1]
        end_err += [abs(last["white"]["rating"] - w), abs(last["black"]["rating"] - b)]
        covered += [last[c]["low"] <= r <= last[c]["high"] for c, r in (("white", w), ("black", b))]
        if len(est) > 40:
            m20_err += [abs(est[40]["white"]["rating"] - w), abs(est[40]["black"]["rating"] - b)]
        true_gap.append(w - b)
        pred_gap.append(last["white"]["rating"] - last["black"]["rating"])
        if fallback:
            base, inc = map(int, tc.split("+"))
            f = fallback(sans, list(clocks), base, inc)
            fb_end += [abs(f[-1]["white"]["rating"] - w), abs(f[-1]["black"]["rating"] - b)]
            if len(f) > 40:
                fb_m20 += [abs(f[40]["white"]["rating"] - w), abs(f[40]["black"]["rating"] - b)]
            fb_gap.append(f[-1]["white"]["rating"] - f[-1]["black"]["rating"])
    true_gap, pred_gap = np.array(true_gap), np.array(pred_gap)
    big = np.abs(true_gap) > 300
    out = {"games": len(games), "blunders": len(drops),
           "mover's drop on a blunder (median)": float(np.median(drops)) if drops else None,
           "opponent's change on that move (median)": float(np.median(other)) if other else None,
           "MAE vs true rating, end of game": round(float(np.mean(end_err)), 1),
           "MAE vs true rating, after move 20": round(float(np.mean(m20_err)), 1),
           "share of true ratings inside the 10-90% range, end of game": round(float(np.mean(covered)), 3),
           "big-gap games (|gap| > 300)": int(big.sum()),
           "big-gap slope, predicted gap vs true gap": round(float(np.polyfit(true_gap[big], pred_gap[big], 1)[0]), 2) if big.sum() > 10 else None}
    if fallback:
        fb_gap = np.array(fb_gap)
        out["moves-and-clocks model"] = {
            "MAE vs true rating, end of game": round(float(np.mean(fb_end)), 1),
            "MAE vs true rating, after move 20": round(float(np.mean(fb_m20)), 1),
            "big-gap slope, predicted gap vs true gap": round(float(np.polyfit(true_gap[big], fb_gap[big], 1)[0]), 2) if big.sum() > 10 else None}
    return out


_engine = None


def _stockfish(sans):
    global _engine
    if _engine is None:
        from web.engine import Engine
        _engine = Engine(depth=14)
    return _engine.evals(sans)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("games", help="folder of prepared parquet games (with evals)")
    ap.add_argument("--val-games", type=int, default=10_000)
    ap.add_argument("--test-games", type=int, default=10_000)
    ap.add_argument("--stockfish-games", type=int, default=300)
    args = ap.parse_args()

    train = load_games(args.games, "train", None, analysed=True, columns=COLUMNS)
    weight = rating_weights(args.games, train[["WhiteElo", "BlackElo"]].to_numpy().ravel())
    print(f"{len(train):,} analysed train games", flush=True)
    val = load_games(args.games, "val", args.val_games, seed=1, analysed=True, columns=COLUMNS)
    fits, val_mae = {}, {}
    for metric in ("mean", "blend"):
        fits[metric] = fit(*rows(train, metric, weight), metric)
        val_mae[metric] = end_of_game_mae(val, Table(fits[metric]))
        print(f"{metric}: val MAE at end of game {val_mae[metric]:.1f}", flush=True)
    metric = min(val_mae, key=val_mae.get)
    table = Table(fits[metric])
    os.makedirs("model", exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(fits[metric], f)
    del train

    from web.predictor import Predictor
    book = Book(BOOK) if os.path.exists(BOOK) else None
    fallback = Predictor("model") if os.path.exists("model/rating_model.onnx") else None
    test = load_games(args.games, "test", args.test_games, seed=1, analysed=True)
    report = {"metric": metric, "val MAE at end of game by metric": val_mae,
              "test, Lichess evals": check(test, table, book, fallback)}
    print(json.dumps(report, indent=2), flush=True)

    if args.stockfish_games:
        sample = test.iloc[:args.stockfish_games].copy()
        for col in ("moves", "evals", "clocks"):
            sample[col] = [list(x[:40]) for x in sample[col]]
        before = check(sample, table, book)
        with Pool(os.cpu_count()) as pool:
            sample["evals"] = pool.map(_stockfish, sample.moves, chunksize=4)
        report["first 20 moves, Lichess evals"] = before
        report["first 20 moves, our Stockfish (depth 14)"] = check(sample, table, book)
    os.makedirs("reports", exist_ok=True)
    with open(REPORT, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

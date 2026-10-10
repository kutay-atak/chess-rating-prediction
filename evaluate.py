"""Calibrate the exported model's intervals on val games, then score it on test games.

Runs the ONNX model the website serves (model/), not the torch checkpoint.
1. Val: per ply bucket, scale the predicted sigmas so that 80% intervals hold 80% of true ratings,
   separately for games with engine evals and without. The scales go into model/meta.json.
2. Test (players never seen in training), split into games Lichess analysed (with evals) and the
   rest: MAE per player after each full move for the model with and without evals, an optional
   baseline model (--baseline, e.g. the moves-and-clocks-only model) and two simple baselines (mean
   rating; mean rating per time control); interval coverage and width. Headline: after move 20.
3. --stockfish-games N: re-score N analysed test games with our own Stockfish (as the website does)
   and compare the model's error using those evals against using Lichess's.
Writes reports/metrics.json and reports/mae_by_move.png.

Usage: python evaluate.py games [--baseline model_v1] [--stockfish-games 1000]
"""
import argparse
import json
import os
from multiprocessing import Pool

import matplotlib
import numpy as np
import onnxruntime as ort

from export_onnx import META, ONNX
from features import MAX_PLIES, Z80, split_outputs, vocab_index
from train import Encoded, batches, load_games

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BUCKETS = [0, 1, 11, 21, 41, 61, 101, MAX_PLIES + 1]  # ply bucket edges for calibration
MOVES = [0, 5, 10, 15, 20, 30, 40]
HEADLINE_MOVE = 20


class Model:
    def __init__(self, folder):
        self.meta = json.load(open(f"{folder}/{META}"))
        self.session = ort.InferenceSession(f"{folder}/{ONNX}")
        self.n_numeric = self.session.get_inputs()[1].shape[2]

    def predict(self, data, hide_evals=False):
        """Predicted (mu, sd), each [n_tokens, 2] in rating points, flat and aligned with data.tokens."""
        raw = np.zeros((len(data.tokens), 5), dtype=np.float32)
        for ids in batches(data.lengths, 32_768):
            tokens, numeric, _ = data.batch(ids, np.full(len(ids), hide_evals))
            out = self.session.run(None, {"tokens": tokens.numpy(), "numeric": numeric.numpy()[..., :self.n_numeric]})[0]
            for row, i in enumerate(ids):
                raw[data.offsets[i]:data.offsets[i + 1]] = out[row, :data.lengths[i]]
        mu_w, mu_b, sd_w, sd_b, _ = split_outputs(raw, self.meta["center"], self.meta["scale"])
        return np.stack([mu_w, mu_b], 1), np.stack([sd_w, sd_b], 1)


def plies(data):
    return np.concatenate([np.arange(n) for n in data.lengths])


def truth(data):
    return np.repeat(data.ratings, data.lengths, axis=0)


def calibrate(ply, true, mu, sd):
    """Per-ply sigma scales (length MAX_PLIES + 1) so central 80% intervals cover 80% per bucket."""
    z = np.abs(true - mu) / sd
    scales = np.ones(MAX_PLIES + 1)
    for lo, hi in zip(BUCKETS[:-1], BUCKETS[1:]):
        inside = (ply >= lo) & (ply < hi)
        if inside.any():
            scales[lo:hi] = np.quantile(z[inside], 0.8) / Z80
    return scales


def score(data, runs, tc_mean, center, shipped):
    """({move: {run: MAE, ..., coverage_80, interval_width, games}}, per-token errors, plies).

    runs: name -> (mu, sd); shipped: (run name, calibration scales) whose intervals get scored."""
    ply, true = plies(data), truth(data)
    errors = {name: np.abs(true - mu) for name, (mu, _) in runs.items()}
    errors["mean rating"] = np.abs(true - center)
    errors["mean per time control"] = np.abs(true - np.repeat(
        [tc_mean.get(tc, center) for tc in data.time_control], data.lengths)[:, None])
    name, scales = shipped
    mu, sd = runs[name]
    half = Z80 * np.asarray(scales)[ply][:, None] * sd
    covered = np.abs(true - mu) <= half

    def row(sel):
        return {**{k: round(float(e[sel].mean()), 1) for k, e in errors.items()},
                "coverage_80": round(float(covered[sel].mean()), 3),
                "interval_width": round(float(2 * half[sel].mean()), 1), "games": int(sel.sum())}

    last = np.isin(np.arange(len(ply)), np.cumsum(data.lengths) - 1)
    rows = {str(m): row(ply == 2 * m) for m in MOVES if (ply == 2 * m).any()}
    return {**rows, "end of game": row(last)}, errors, ply


_engine = None


def _stockfish_evals(sans):
    global _engine
    if _engine is None:
        from web.engine import Engine
        _engine = Engine()
    return _engine.evals(sans)


def stockfish_check(model, games, index, n):
    """MAE after moves 10 and 20 on n analysed games: Lichess's evals vs our Stockfish's."""
    games = games.iloc[:n].copy()
    for col in ["moves", "clocks", "evals"]:  # moves after 20 can't change the estimate at move 20
        games[col] = [list(x[:2 * HEADLINE_MOVE]) for x in games[col]]
    lichess = Encoded(games, index)
    with Pool(os.cpu_count()) as pool:
        games["evals"] = pool.map(_stockfish_evals, games.moves, chunksize=8)
    ours = Encoded(games, index)
    ply, true = plies(lichess), truth(lichess)
    out = {"games": len(games), "depth": int(os.environ.get("STOCKFISH_DEPTH", 12))}
    for name, data, hide in [("Lichess evals", lichess, False), ("our Stockfish", ours, False), ("no evals", lichess, True)]:
        mu, _ = model.predict(data, hide_evals=hide)
        out[name] = {str(m): round(float(np.abs(true - mu)[ply == 2 * m].mean()), 1) for m in (10, HEADLINE_MOVE)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("games", help="folder of prepared parquet games")
    ap.add_argument("--model", default="model")
    ap.add_argument("--baseline", help="folder of another exported model to compare against")
    ap.add_argument("--val-games", type=int, default=50_000, help="per kind (with / without evals)")
    ap.add_argument("--test-games", type=int, default=100_000, help="per kind")
    ap.add_argument("--train-games", type=int, default=200_000, help="games for the time-control baseline")
    ap.add_argument("--stockfish-games", type=int, default=0)
    args = ap.parse_args()

    model = Model(args.model)
    meta = model.meta
    index = vocab_index(meta["vocab"])
    kinds = {"evals": True, "no_evals": False}

    for kind, analysed in kinds.items():
        val = Encoded(load_games(args.games, "val", args.val_games, seed=1, analysed=analysed), index)
        mu, sd = model.predict(val)
        meta["calibration"][kind] = calibrate(plies(val), truth(val), mu, sd).tolist()
        del val
    with open(f"{args.model}/{META}", "w") as f:
        json.dump(meta, f)

    train_df = load_games(args.games, "train", args.train_games, seed=1, columns=["TimeControl", "WhiteElo", "BlackElo"])
    tc_mean = train_df.groupby("TimeControl")[["WhiteElo", "BlackElo"]].mean().mean(axis=1).to_dict()
    baseline = Model(args.baseline) if args.baseline else None

    metrics, curves = {}, {}
    for title, analysed in [("analysed games", True), ("other games", False)]:
        games = load_games(args.games, "test", args.test_games, seed=1, analysed=analysed)
        test = Encoded(games, index)
        runs = {}
        if analysed:
            runs["model with Stockfish evals"] = model.predict(test)
        runs["model without evals"] = model.predict(test, hide_evals=True)
        if baseline:
            runs["baseline model"] = baseline.predict(test, hide_evals=True)
        shipped = (("model with Stockfish evals", meta["calibration"]["evals"]) if analysed
                   else ("model without evals", meta["calibration"]["no_evals"]))
        rows, errors, ply = score(test, runs, tc_mean, meta["center"], shipped)
        metrics[title] = {"test games": len(test), f"after move {HEADLINE_MOVE}": rows[str(HEADLINE_MOVE)], "by move": rows}
        curves[title] = (errors, ply)
        if analysed and args.stockfish_games:
            metrics["stockfish check"] = stockfish_check(model, games, index, args.stockfish_games)
        del games, test
    metrics["calibration scales"] = {kind: {f"{lo}-{hi - 1}": round(meta["calibration"][kind][lo], 3)
                                            for lo, hi in zip(BUCKETS[:-1], BUCKETS[1:])} for kind in kinds}
    os.makedirs("reports", exist_ok=True)
    with open("reports/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, (title, (errors, ply)) in zip(axes, curves.items()):
        moves = [m for m in range(61) if (ply == 2 * m).any()]
        for name, e in errors.items():
            ax.plot(moves, [e[ply == 2 * m].mean() for m in moves], label=name)
        ax.axvline(HEADLINE_MOVE, color="grey", ls=":")
        ax.set(xlabel="full moves played", title=f"Rating error, test players, {title}")
        ax.legend()
    axes[0].set(ylabel="mean absolute error (rating points)")
    fig.tight_layout()
    fig.savefig("reports/mae_by_move.png", dpi=120)


if __name__ == "__main__":
    main()

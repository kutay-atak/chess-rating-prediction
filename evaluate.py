"""Calibrate the exported model's intervals on val games, then score it against baselines on test games.

Runs the ONNX model the website serves (model/rating_model.onnx), not the torch checkpoint.
1. Val: per ply bucket, scale the predicted sigmas so that 80% intervals hold 80% of true ratings;
   the scales are written into model/meta.json.
2. Test (players never seen in training): MAE per player vs ply for the model and two baselines
   (global mean rating; mean rating per time control), interval coverage and width vs ply.
Writes reports/metrics.json and reports/mae_by_ply.png.

Usage: python evaluate.py games [--val-games 50000] [--test-games 100000]
"""
import argparse
import json
import os

import matplotlib
import numpy as np
import onnxruntime as ort

from export_onnx import META, ONNX
from features import MAX_PLIES, Z80, split_outputs, vocab_index
from train import Encoded, batches, load_games

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BUCKETS = [0, 1, 11, 21, 41, 61, 101, MAX_PLIES + 1]  # ply bucket edges for calibration
REPORT_PLIES = [0, 5, 10, 20, 30, 40, 60, 80, 100]


def predict(session, data):
    """Raw outputs for every token of `data`, flat and aligned with data.tokens: [n_tokens, 5]."""
    out = np.zeros((len(data.tokens), 5), dtype=np.float32)
    for ids in batches(data.lengths, 32_768):
        tokens, numeric, _ = data.batch(ids)
        raw = session.run(None, {"tokens": tokens.numpy(), "numeric": numeric.numpy()})[0]
        for row, i in enumerate(ids):
            out[data.offsets[i]:data.offsets[i + 1]] = raw[row, :data.lengths[i]]
    return out


def per_token(data, raw, meta):
    """Flat per-token arrays: ply, true ratings [n, 2], predicted means [n, 2], sigmas [n, 2]."""
    ply = np.concatenate([np.arange(n) for n in data.lengths])
    truth = np.repeat(data.ratings, data.lengths, axis=0)
    mu_w, mu_b, sd_w, sd_b, _ = split_outputs(raw, meta["center"], meta["scale"])
    return ply, truth, np.stack([mu_w, mu_b], 1), np.stack([sd_w, sd_b], 1)


def calibrate(ply, truth, mu, sd):
    """Per-ply sigma scales (length MAX_PLIES + 1) so central 80% intervals cover 80% per bucket."""
    z = np.abs(truth - mu) / sd
    scales = np.ones(MAX_PLIES + 1)
    for lo, hi in zip(BUCKETS[:-1], BUCKETS[1:]):
        inside = (ply >= lo) & (ply < hi)
        if inside.any():
            scales[lo:hi] = np.quantile(z[inside], 0.8) / Z80
    return scales


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("games", help="folder of prepared parquet games")
    ap.add_argument("--val-games", type=int, default=50_000)
    ap.add_argument("--test-games", type=int, default=100_000)
    ap.add_argument("--train-games", type=int, default=200_000, help="games for the time-control baseline")
    args = ap.parse_args()

    meta = json.load(open(META))
    session = ort.InferenceSession(ONNX)
    index = vocab_index(meta["vocab"])

    val = Encoded(load_games(args.games, "val", args.val_games, seed=1), index)
    scales = calibrate(*per_token(val, predict(session, val), meta))
    meta["calibration"] = scales.tolist()
    with open(META, "w") as f:
        json.dump(meta, f)

    train_df = load_games(args.games, "train", args.train_games, seed=1, columns=["TimeControl", "WhiteElo", "BlackElo"])
    tc_mean = (train_df.groupby("TimeControl")[["WhiteElo", "BlackElo"]].mean().mean(axis=1)).to_dict()
    del train_df

    test = Encoded(load_games(args.games, "test", args.test_games, seed=1), index)
    ply, truth, mu, sd = per_token(test, predict(session, test), meta)
    half = Z80 * scales[ply][:, None] * sd
    tc_pred = np.repeat([tc_mean.get(tc, meta["center"]) for tc in test.time_control], test.lengths)[:, None]
    err = {"model": np.abs(truth - mu), "mean rating": np.abs(truth - meta["center"]),
           "mean per time control": np.abs(truth - tc_pred)}
    covered = np.abs(truth - mu) <= half
    last = np.cumsum(test.lengths) - 1  # each game's final position

    def row(sel):
        return {**{name: round(float(e[sel].mean()), 1) for name, e in err.items()},
                "coverage_80": round(float(covered[sel].mean()), 3),
                "interval_width": round(float(2 * half[sel].mean()), 1), "games": int(sel.sum())}

    metrics = {"test_games": len(test), "test_players_unseen_in_training": True,
               "mae_by_ply": {str(p): row(ply == p) for p in REPORT_PLIES if (ply == p).any()},
               "end_of_game": row(np.isin(np.arange(len(ply)), last)),
               "calibration_scales": {f"{lo}-{hi - 1}": round(float(scales[lo]), 3) for lo, hi in zip(BUCKETS[:-1], BUCKETS[1:])}}
    os.makedirs("reports", exist_ok=True)
    with open("reports/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

    plies = np.arange(0, 121)
    has = [(ply == p).any() for p in plies]
    plies = plies[has]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, e in err.items():
        a.plot(plies, [e[ply == p].mean() for p in plies], label=name)
    a.set(xlabel="half-moves played", ylabel="mean absolute error (rating points)", title="Rating error, test players")
    a.legend()
    b.plot(plies, [2 * half[ply == p].mean() for p in plies], label="80% interval width")
    b.set(xlabel="half-moves played", ylabel="rating points", title="Interval width (coverage on right axis)")
    c = b.twinx()
    c.plot(plies, [covered[ply == p].mean() for p in plies], color="C3", label="coverage")
    c.axhline(0.8, color="C3", ls=":")
    c.set(ylim=(0.5, 1), ylabel="coverage")
    fig.tight_layout()
    fig.savefig("reports/mae_by_ply.png", dpi=120)


if __name__ == "__main__":
    main()

"""Model inputs and outputs, shared by training, evaluation and the web server.

A game becomes a sequence of tokens: a start token (prediction before any move), then one token per
half-move. Each token has a move id (SAN, or <start>/<unk>) and a few numeric features from the
clocks and the time control. The model outputs, per token, a bivariate Gaussian over the two
ratings; `to_ratings` turns that into per-player estimates and calibrated intervals.

Numpy only, so the web server needs no torch.
"""
from collections import Counter

import numpy as np

PAD, START, UNK = 0, 1, 2
MAX_PLIES = 320  # moves past this are not seen; the estimate stays at the last one
N_NUMERIC = 9
Z80 = 1.2816  # half-width of a central 80% normal interval, in sigmas


def time_control(tc):
    """'180+2' -> (180, 2)."""
    base, inc = tc.split("+")
    return int(base), int(inc)


def build_vocab(san_lists, min_count=5):
    counts = Counter(san for sans in san_lists for san in sans[:MAX_PLIES])
    return sorted(san for san, n in counts.items() if n >= min_count)


def numeric_features(clocks, base, inc):
    """[len(clocks) + 1, N_NUMERIC] float32; row 0 is the start token.

    clocks[i] is the mover's seconds left after half-move i (White moves at even i). Time spent on a
    move is the mover's previous clock (the base time for their first move) plus the increment
    minus their clock now.
    """
    clocks = np.asarray(clocks, dtype=np.float64)
    n = len(clocks)
    prev = np.full(n, float(base))
    prev[2:] = clocks[:-2]
    spent = np.clip(prev + inc - clocks, 0, None)
    other = np.full(n, float(base))  # opponent's clock while this move was made
    other[1:] = clocks[:-1]

    f = np.zeros((n + 1, N_NUMERIC))
    f[0, 7] = 1  # start token
    f[1:, 0] = np.where(np.arange(n) % 2 == 0, 1, -1)  # White / Black move
    f[1:, 1] = clocks / base
    f[1:, 2] = np.log1p(clocks) / 5
    f[1:, 3] = np.log1p(spent) / 3
    f[1:, 4] = spent / np.maximum(prev + inc, 1)
    f[1:, 8] = other / base
    f[:, 5] = np.log(base / 180)
    f[:, 6] = inc / 5
    return f.astype(np.float32)


def encode(sans, clocks, base, inc, index):
    """(tokens [T+1] int64, numeric [T+1, N_NUMERIC] float32), cut to MAX_PLIES moves.

    index maps SAN -> id, as from `vocab_index`.
    """
    sans, clocks = sans[:MAX_PLIES], clocks[:MAX_PLIES]
    tokens = np.array([START] + [index.get(san, UNK) for san in sans], dtype=np.int64)
    return tokens, numeric_features(clocks, base, inc)


def vocab_index(vocab):
    return {san: i + 3 for i, san in enumerate(vocab)}


def split_outputs(raw, center, scale):
    """Raw model outputs [..., 5] -> (mu_w, mu_b, sd_w, sd_b, rho) in rating points."""
    raw = np.asarray(raw, dtype=np.float64)
    mu = raw[..., :2] * scale + center
    sd = np.exp(raw[..., 2:4]) * scale
    rho = 0.99 * np.tanh(raw[..., 4])
    return mu[..., 0], mu[..., 1], sd[..., 0], sd[..., 1], rho


def to_ratings(raw, meta):
    """Raw outputs [T+1, 5] for one game -> list of per-ply dicts with estimates and 80% intervals.

    Interval half-widths are Z80 * sd * the calibration scale for that ply, fitted on val games.
    """
    mu_w, mu_b, sd_w, sd_b, _ = split_outputs(raw, meta["center"], meta["scale"])
    cal = np.asarray(meta["calibration"])
    k = Z80 * cal[np.minimum(np.arange(len(mu_w)), len(cal) - 1)]
    return [
        {"ply": t,
         "white": {"rating": round(mu_w[t]), "low": round(mu_w[t] - k[t] * sd_w[t]), "high": round(mu_w[t] + k[t] * sd_w[t])},
         "black": {"rating": round(mu_b[t]), "low": round(mu_b[t] - k[t] * sd_b[t]), "high": round(mu_b[t] + k[t] * sd_b[t])}}
        for t in range(len(mu_w))
    ]

"""Model inputs and outputs, shared by training, evaluation and the web server.

A game becomes a sequence of tokens: a start token (prediction before any move), then one token per
half-move. Each token has a move id (SAN, or <start>/<unk>) and numeric features from the clocks,
the time control and, when available, engine evals. Without evals those features are zero and the
model falls back to moves and clocks. It outputs, per token, a bivariate Gaussian over the two
ratings; `to_ratings` turns that into per-player estimates and calibrated intervals.

Numpy only, so the web server needs no torch.
"""
from collections import Counter

import numpy as np

PAD, START, UNK = 0, 1, 2
MAX_PLIES = 320  # moves past this are not seen; the estimate stays at the last one
N_NUMERIC = 12
Z80 = 1.2816  # half-width of a central 80% normal interval, in sigmas
START_CP = 20  # roughly Stockfish's eval of the starting position, in centipawns


def time_control(tc):
    """'180+2' -> (180, 2)."""
    base, inc = tc.split("+")
    return int(base), int(inc)


def build_vocab(san_lists, min_count=5):
    counts = Counter(san for sans in san_lists for san in sans[:MAX_PLIES])
    return sorted(san for san, n in counts.items() if n >= min_count)


def win_chance(cp):
    """White's expected score (0..1) from an eval in centipawns, with Lichess's formula."""
    return 1 / (1 + np.exp(-0.00368208 * np.asarray(cp, dtype=np.float64)))


def numeric_features(clocks, base, inc, evals=None):
    """[len(clocks) + 1, N_NUMERIC] float32; row 0 is the start token.

    clocks[i] is the mover's seconds left after half-move i (White moves at even i). Time spent on a
    move is the mover's previous clock (the base time for their first move) plus the increment
    minus their clock now. Clocks are scaled by Lichess's estimated game length, base + 40 * inc,
    which unlike the base time is never 0 for blitz (0+5 is blitz).

    evals[i] is the engine eval after half-move i in centipawns from White's side, or evals is None.
    From it: White's win chance after each move, and how much win chance the mover gave away.
    """
    clocks = np.asarray(clocks, dtype=np.float64)
    total = base + 40 * inc
    n = len(clocks)
    prev = np.full(n, float(base))
    prev[2:] = clocks[:-2]
    spent = np.clip(prev + inc - clocks, 0, None)
    other = np.full(n, float(base))  # opponent's clock while this move was made
    other[1:] = clocks[:-1]

    f = np.zeros((n + 1, N_NUMERIC))
    f[0, 7] = 1  # start token
    f[1:, 0] = np.where(np.arange(n) % 2 == 0, 1, -1)  # White / Black move
    f[1:, 1] = clocks / total
    f[1:, 2] = np.log1p(clocks) / 5
    f[1:, 3] = np.log1p(spent) / 3
    f[1:, 4] = spent / np.maximum(prev + inc, 1)
    f[1:, 8] = other / total
    f[:, 5] = base / 300
    f[:, 6] = inc / 5
    if evals is not None:
        white = win_chance(evals)
        before = np.concatenate([win_chance([START_CP]), white[:-1]])
        f[:, 9] = 1  # this game has evals
        f[1:, 10] = white - 0.5
        f[1:, 11] = np.clip(f[1:, 0] * (before - white), 0, 1)  # win chance the mover lost
    return f.astype(np.float32)


def encode(sans, clocks, base, inc, index, evals=None):
    """(tokens [T+1] int64, numeric [T+1, N_NUMERIC] float32), cut to MAX_PLIES moves.

    index maps SAN -> id, as from `vocab_index`.
    """
    sans, clocks = sans[:MAX_PLIES], clocks[:MAX_PLIES]
    evals = None if evals is None else evals[:MAX_PLIES]
    tokens = np.array([START] + [index.get(san, UNK) for san in sans], dtype=np.int64)
    return tokens, numeric_features(clocks, base, inc, evals)


def vocab_index(vocab):
    return {san: i + 3 for i, san in enumerate(vocab)}


def split_outputs(raw, center, scale):
    """Raw model outputs [..., 5] -> (mu_w, mu_b, sd_w, sd_b, rho) in rating points."""
    raw = np.asarray(raw, dtype=np.float64)
    mu = raw[..., :2] * scale + center
    sd = np.exp(raw[..., 2:4]) * scale
    rho = 0.99 * np.tanh(raw[..., 4])
    return mu[..., 0], mu[..., 1], sd[..., 0], sd[..., 1], rho


def to_ratings(raw, meta, has_evals=False):
    """Raw outputs [T+1, 5] for one game -> list of per-ply dicts with estimates and 80% intervals.

    Interval half-widths are Z80 * sd * the calibration scale for that ply, fitted on val games
    separately for games with and without engine evals.
    """
    mu_w, mu_b, sd_w, sd_b, _ = split_outputs(raw, meta["center"], meta["scale"])
    cal = np.asarray(meta["calibration"]["evals" if has_evals else "no_evals"])
    k = Z80 * cal[np.minimum(np.arange(len(mu_w)), len(cal) - 1)]
    return [
        {"ply": t,
         "white": {"rating": round(mu_w[t]), "low": round(mu_w[t] - k[t] * sd_w[t]), "high": round(mu_w[t] + k[t] * sd_w[t])},
         "black": {"rating": round(mu_b[t]), "low": round(mu_b[t] - k[t] * sd_b[t]), "high": round(mu_b[t] + k[t] * sd_b[t])}}
        for t in range(len(mu_w))
    ]

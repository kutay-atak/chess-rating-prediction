"""Features of one side of a game after t plies, for the baselines (baselines.py).

For a side after the first t plies: how many moves it has made, how it used its clock, the time
control, the opening family (from ply 10 on) and, if Lichess analysed the game, how much its moves
lost by the engine's evaluation. Conventions (seconds spent, losses) are in CLAUDE.md.
"""
import numpy as np
import pandas as pd

CHECKPOINTS = [2, 5, 10, 20, 30, 40, 60, 80]
ECO_FROM = 10  # the ECO header is set from moves after t; its letter is settled by ply 10 in 99.8% of games
FEATURES = ["t", "own_moves", "log_spent_mean", "log_spent_sd", "fast_share", "clock_share", "base", "inc",
            "berserk", "eco", "has_evals", "avg_loss", "mistakes", "blunders"]


def own_moves(clocks, evals, base, inc, side):
    """(seconds spent, loss in pawns, clock left) for each move of `side` (0 White, 1 Black), and berserk."""
    plies = np.arange(side, len(clocks), 2)
    left = clocks[plies]
    berserk = bool(base > 0 and len(left) and left[0] == base / 2)  # Lichess halves the clock and drops the increment
    spent = np.concatenate([[np.nan], left[:-1]]) - left + (0 if berserk else inc)
    if len(left):
        spent[0] = 0 if berserk else base - left[0]
    spent = np.clip(spent, 0, None)  # whole-second clocks and time the opponent added make a few slightly negative
    before = np.concatenate([[np.nan], evals[:-1]])[plies]  # White's first move has no eval before it
    loss = np.clip((1 if side == 0 else -1) * (before - evals[plies]), 0, None)  # NaN where an eval is missing
    return spent, loss, left, berserk


def game_features(clocks, evals, base, inc, checkpoints=CHECKPOINTS):
    """Features of both sides at each checkpoint t the game reaches, as a dict of equal-length arrays."""
    clocks, evals = np.asarray(clocks, dtype=float), np.asarray(evals, dtype=float)
    ts = np.array([t for t in checkpoints if t <= len(clocks)], dtype=int)
    evals_seen = np.cumsum(~np.isnan(evals))
    cols = {}
    for side in (0, 1):
        spent, loss, left, berserk = own_moves(clocks, evals, base, inc, side)
        k = (ts + 1 - side) // 2  # own moves among the first t plies: White ⌈t/2⌉, Black ⌊t/2⌋

        def upto(x):
            return np.concatenate([[0.0], np.cumsum(x)])[k]  # sum over the first k own moves

        log_spent = np.log1p(spent)  # log1p: 0-second moves (premoves, the untimed first move) are common
        s1, s2, n_loss = upto(log_spent), upto(log_spent ** 2), upto(~np.isnan(loss))
        part = {
            "side": np.where(side == 0, "white", "black").repeat(len(ts)),
            "t": ts,
            "own_moves": k,
            "log_spent_mean": s1 / k,
            "log_spent_sd": np.where(k > 1, np.sqrt(np.maximum(s2 - s1 ** 2 / k, 0) / np.maximum(k - 1, 1)), np.nan),
            "fast_share": upto(spent < 2) / k,
            "clock_share": left[k - 1] / base if base > 0 else np.full(len(ts), np.nan),  # 0+N games have no base
            "berserk": np.full(len(ts), berserk),
            "has_evals": evals_seen[ts - 1] > 0,  # the same for both sides: was this game analysed (so far)?
            "avg_loss": np.where(n_loss > 0, upto(np.nan_to_num(loss)) / np.maximum(n_loss, 1), np.nan),
            "mistakes": np.where(n_loss > 0, upto(loss >= 1), np.nan),
            "blunders": np.where(n_loss > 0, upto(loss >= 3), np.nan),
        }
        for c, v in part.items():
            cols.setdefault(c, []).append(v)
    return {c: np.concatenate(v) for c, v in cols.items()}


def rows(games, checkpoints=CHECKPOINTS):
    """One row per side of each game at each checkpoint it reaches: features plus what the models and
    evaluation need (rating, split, scored, weight, TimeControl)."""
    parts = []
    for g in games.itertuples(index=False):
        base, inc = map(int, g.TimeControl.split("+"))
        f = game_features(g.clocks, g.evals, base, inc, checkpoints)
        f["game_id"] = np.full(len(f["t"]), g.game_id)
        parts.append(f)
    df = pd.DataFrame({c: np.concatenate([p[c] for p in parts]) for c in parts[0]})
    g = games.set_index("game_id").loc[df.game_id]
    white = df.side.to_numpy() == "white"
    df["rating"] = np.where(white, g.WhiteElo, g.BlackElo)
    df["scored"] = np.where(white, g.score_white, g.score_black)
    df["split"], df["weight"], df["TimeControl"] = g.split.to_numpy(), g.weight.to_numpy(), g.TimeControl.to_numpy()
    tc = df.TimeControl.str.split("+", expand=True).astype(int)
    df["base"], df["inc"] = tc[0], tc[1]
    letter = g.ECO.str[0].where(df.t.to_numpy() >= ECO_FROM).to_numpy()
    df["eco"] = pd.Categorical(letter, categories=list("ABCDE"))
    return df

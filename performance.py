"""Per-game performance rating: how strong each player has played in this game, up to this move.

Two sources, blended per player after every half-move:
- Opening book: while the position has been reached in many training games, the average rating of
  the players who reached it (keyed by position, so different move orders count together).
- Move accuracy: Stockfish's evals before and after each of the player's own moves give a move
  accuracy 0-100 (Lichess's formula; a hung queen is about 0). The player's running game accuracy
  is looked up in a table, fitted on real games, of the ratings of players who had that accuracy
  after that many moves (fit_performance.py).
The book's weight is its game count n against BOOK_PRIOR; after the game leaves the book, that n
halves with every own move, so each line slides from the book to accuracy instead of jumping.

Numpy and python-chess only, so the web server needs no torch.
"""
import json

import chess
import chess.polyglot
import numpy as np

from features import START_CP, Z80, win_chance

BOOK_PRIOR = 20  # pseudo-games of weight the accuracy estimate gets against the book
MIN_ACCURACY = 10  # floor in the harmonic mean, so a single 0% move doesn't zero the whole game


def move_accuracy(evals):
    """Accuracy 0-100 of each half-move, from the eval after each move (centipawns, White's side)."""
    white = 100 * win_chance(evals)
    before = np.concatenate([[100 * win_chance(START_CP)], white[:-1]])
    mover = np.where(np.arange(len(white)) % 2 == 0, 1, -1)
    lost = mover * (before - white)  # win% the mover gave away; negative if they gained
    return np.clip(103.1668 * np.exp(-0.04354 * lost) - 3.1669, 0, 100)


def game_accuracy(acc, metric="blend"):
    """Running game accuracy after each of one player's moves, given that player's move accuracies.

    "mean" is the plain average; "blend" averages it with the harmonic mean, which a blunder pulls
    down much harder (as in Lichess's game accuracy)."""
    acc = np.asarray(acc, dtype=np.float64)
    k = np.arange(1, len(acc) + 1)
    mean = np.cumsum(acc) / k
    if metric == "mean":
        return mean
    return (mean + k / np.cumsum(1 / np.maximum(acc, MIN_ACCURACY))) / 2


def own_moves(n, side):
    """Number of moves `side` (0 White, 1 Black) has made after each ply 0..n."""
    return (np.arange(n + 1) + 1 - side) // 2


def position_keys(sans):
    """Zobrist hash of the position after each ply 0..len(sans) (ply 0: the starting position)."""
    board = chess.Board()
    keys = [chess.polyglot.zobrist_hash(board)]
    for san in sans:
        board.push_san(san)
        keys.append(chess.polyglot.zobrist_hash(board))
    return np.array(keys, dtype=np.uint64)


class Table:
    """Running accuracy -> (median, 10th, 90th percentile) rating, per number of own moves."""

    def __init__(self, data):
        self.metric, self.step = data["metric"], data["step"]
        self.median, self.low, self.high = (np.array(data[x]) for x in ("median", "low", "high"))
        self.k_max = self.median.shape[0] - 1  # row k_max holds every k >= k_max; row 0 is before any move

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(json.load(f))

    def lookup(self, k, acc):
        """Arrays (median, low, high) for own-move counts k and running accuracies acc (nan when k=0)."""
        k = np.minimum(np.asarray(k), self.k_max)
        pos = np.clip((np.nan_to_num(np.asarray(acc, dtype=np.float64)) - self.step / 2) / self.step,
                      0, self.median.shape[1] - 1)
        i0 = np.floor(pos).astype(int)
        i1 = np.minimum(i0 + 1, self.median.shape[1] - 1)
        f = pos - i0
        return tuple(t[k, i0] * (1 - f) + t[k, i1] * f for t in (self.median, self.low, self.high))


class Book:
    """Opening book from opening_book.py: per position, game count and mean White / Black rating."""

    def __init__(self, path):
        data = np.load(path)
        self.keys, self.n = data["keys"], data["n"]
        self.white, self.black, self.sd = data["white"], data["black"], data["sd"]

    def find(self, keys):
        """Index of each key in the book, or -1."""
        i = np.minimum(np.searchsorted(self.keys, keys), len(self.keys) - 1)
        return np.where(self.keys[i] == keys, i, -1)


def book_weights(found, n_book, side):
    """Per ply, the book's weight for `side` and which book entry it refers to (-1: none yet).

    In the book: weight = that position's game count. After leaving it: the last in-book count,
    halved for every move this side has made since."""
    weight, entry = np.zeros(len(found)), np.full(len(found), -1)
    last, since = -1, 0
    for t, i in enumerate(found):
        if i >= 0:
            last, since = i, 0
        elif last >= 0 and t > 0 and (t - 1) % 2 == side:
            since += 1
        if last >= 0:
            weight[t], entry[t] = n_book[last] * 0.5 ** since, last
    return weight, entry


def estimates(sans, evals, table, book=None):
    """Per ply 0..len(evals): {ply, white: {rating, low, high}, black: {...}}.

    evals[i] is the engine eval after half-move i; sans the moves played (at least len(evals))."""
    evals = np.asarray(evals, dtype=np.float64)
    n = len(evals)
    acc = move_accuracy(evals) if n else np.zeros(0)
    found = book.find(position_keys(sans[:n])) if book else np.full(n + 1, -1)
    out = [{"ply": t} for t in range(n + 1)]
    for side, name in enumerate(("white", "black")):
        k = own_moves(n, side)
        running = np.concatenate([[np.nan], game_accuracy(acc[side::2], table.metric)])[k]
        a_mid, a_low, a_high = table.lookup(k, running)
        weight, entry = book_weights(found, book.n if book else None, side)
        mean = (book.white if side == 0 else book.black)[entry] if book else 0
        spread = Z80 * book.sd[entry] if book else 0
        mix = lambda b, a: (weight * b + BOOK_PRIOR * a) / (weight + BOOK_PRIOR)  # noqa: E731
        mid, low, high = mix(mean, a_mid), mix(mean - spread, a_low), mix(mean + spread, a_high)
        for t in range(n + 1):
            out[t][name] = {"rating": round(mid[t]), "low": round(low[t]), "high": round(high[t])}
    return out

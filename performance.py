"""Per-game performance rating: how strong each player has played in this game, up to this move.

Two sources, blended per player after every half-move:
- Opening book: while the position has been reached in many training games, the average rating of
  the players who reached it (keyed by position, so different move orders count together).
- Move accuracy: Stockfish's evals before and after each of the player's own moves give a move
  accuracy 0-100 (Lichess's formula; a hung queen is about 0). The player's running game accuracy
  is looked up in a table, fitted on real games, of the ratings of players who had that accuracy
  after that many moves (fit_performance.py).
The book's weight is its game count n against BOOK_PRIOR; after the game leaves the book, n starts
from at most BOOK_MIN (the position now has fewer games than that) and halves with every own move, so
each line slides from the book to accuracy instead of jumping, yet a blunder still shows at once.

A player's estimate only changes when that player moves: their book position is the one right after
their own move, and their accuracy only counts their own moves.

Numpy and python-chess only.
"""
import json

import chess
import chess.polyglot
import numpy as np

BOOK_PRIOR = 20  # pseudo-games of weight the accuracy estimate gets against the book
BOOK_MIN = 20  # opening_book.py keeps positions seen in at least this many games
MIN_ACCURACY = 10  # floor in the harmonic mean, so a single 0% move doesn't zero the whole game
START_CP = 20  # roughly Stockfish's eval of the starting position, in centipawns
Z80 = 1.2816  # half-width of a central 80% normal interval, in sigmas


def win_percent(cp):
    """White's winning chance 0-100 from an eval in centipawns (Lichess's formula)."""
    return 100 / (1 + np.exp(-0.00368208 * np.asarray(cp, dtype=np.float64)))


def move_accuracy(evals):
    """Accuracy 0-100 of each half-move, from the eval after each move (centipawns, White's side)."""
    white = win_percent(evals)
    before = np.concatenate([[win_percent(START_CP)], white[:-1]])
    mover = np.where(np.arange(len(white)) % 2 == 0, 1, -1)
    lost = mover * (before - white)  # win% the mover gave away; negative if they gained
    return np.clip(103.1668 * np.exp(-0.04354 * lost) - 3.1669, 0, 100)


def game_accuracy(acc):
    """Running game accuracy after each of one player's moves, given that player's move accuracies:
    the average of the mean and the harmonic mean, which a blunder pulls down much harder."""
    acc = np.asarray(acc, dtype=np.float64)
    k = np.arange(1, len(acc) + 1)
    mean = np.cumsum(acc) / k
    harmonic = k / np.cumsum(1 / np.maximum(acc, MIN_ACCURACY))
    return (mean + harmonic) / 2


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
        self.step = data["step"]
        self.median, self.low, self.high = (np.array(data[x]) for x in ("median", "low", "high"))
        self.k_max = self.median.shape[0] - 1  # row k_max holds every k >= k_max; row 0 is before any move

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(json.load(f))

    def lookup(self, k, acc):
        """Arrays (median, low, high) for own-move counts k and running accuracies acc (ignored when k=0),
        interpolated between bin centres."""
        k = np.minimum(np.asarray(k), self.k_max)
        pos = np.clip((np.nan_to_num(np.asarray(acc, dtype=np.float64)) - self.step / 2) / self.step,
                      0, self.median.shape[1] - 1)
        i0 = np.floor(pos).astype(int)
        i1 = np.minimum(i0 + 1, self.median.shape[1] - 1)
        f = pos - i0
        return tuple(t[k, i0] * (1 - f) + t[k, i1] * f for t in (self.median, self.low, self.high))


class Book:
    """Opening book from opening_book.py: per position, game count, mean White / Black rating and the
    spread of both players' ratings."""

    def __init__(self, path):
        data = np.load(path)
        self.keys, self.n = data["keys"], data["n"]
        self.white, self.black, self.sd = data["white"], data["black"], data["sd"]

    def find(self, keys):
        """Index of each key in the book, or -1."""
        i = np.minimum(np.searchsorted(self.keys, keys), len(self.keys) - 1)
        return np.where(self.keys[i] == keys, i, -1)


def book_weights(found, n_book, side):
    """Per ply 0..T, the book's weight for `side` and the book entry it refers to (-1: none).

    Only the start position and the positions right after this side's own moves count, so the
    opponent's moves never change this side's estimate. In the book: weight = that position's game
    count. After leaving it: the last in-book count, capped at BOOK_MIN, halved for every own move since."""
    weight, entry = np.zeros(len(found)), np.full(len(found), -1)
    last, since = -1, 0
    for t, i in enumerate(found):
        if t == 0 or (t - 1) % 2 == side:
            if i >= 0:
                last, since = i, 0
            elif last >= 0:
                since += 1
        if last >= 0:
            weight[t], entry[t] = (min(n_book[last], BOOK_MIN) if since else n_book[last]) * 0.5 ** since, last
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
        running = np.concatenate([[np.nan], game_accuracy(acc[side::2])])[k]
        mid, low, high = table.lookup(k, running)
        if book:
            weight, entry = book_weights(found, book.n, side)
            mean = (book.white if side == 0 else book.black)[entry]
            spread = Z80 * book.sd[entry]
            mix = lambda b, a: (weight * b + BOOK_PRIOR * a) / (weight + BOOK_PRIOR)  # noqa: E731
            mid, low, high = mix(mean, mid), mix(mean - spread, low), mix(mean + spread, high)
        for t in range(n + 1):
            out[t][name] = {"rating": round(mid[t]), "low": round(low[t]), "high": round(high[t])}
    return out

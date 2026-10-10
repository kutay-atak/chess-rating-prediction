from pathlib import Path

import numpy as np
import pytest

from performance import BOOK_PRIOR, Book, Table, book_weights, estimates, game_accuracy, move_accuracy, position_keys

MODEL = Path(__file__).parent / "model"
SANS = ["e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6", "Ng5", "d5", "exd5", "Qe7", "Nxf7", "Qxe2+"]  # evals below are made up
QUEEN_BLUNDER = ["e4", "e5", "Qh5", "Nf6", "Qxf7+", "Kxf7"]
QUEEN_BLUNDER_EVALS = [30, 34, -38, -19, -614, -618]  # Stockfish 17.1, depth 14


def toy_table():
    """Rating rises linearly with accuracy: 600 at 0% to 2600 at 100%, the same for every k."""
    centers = np.arange(1, 100, 2)
    row = 600 + 20 * centers
    rows = np.vstack([np.full(50, 1600.0)] + [row] * 40)
    return Table({"step": 2, "median": rows.tolist(), "low": (rows - 300).tolist(), "high": (rows + 300).tolist()})


def line(out, side):
    return [e[side]["rating"] for e in out]


def test_move_accuracy():
    assert move_accuracy([20, 20, 20, -900]).round().tolist() == [100, 100, 100, 100]  # nobody gave anything away
    assert move_accuracy([20, 900])[1] < 15  # Black's move hands White nine pawns


def test_blunder_drops_only_the_movers_line():
    evals = [30] * 9 + [-50, -900]  # Black plays well; White's 11th half-move throws away ~9 pawns
    out = estimates(SANS, evals, toy_table())
    white, black = line(out, "white"), line(out, "black")
    assert white[-2] - white[-1] > 300  # steep drop on the blunder
    assert black[-1] == black[-2]  # Black didn't move, its line doesn't change
    assert black[-1] > 2000  # Black's moves were all accurate


def test_harmonic_blend_punishes_a_blunder_more_than_the_mean():
    acc = [95] * 10 + [0]
    assert game_accuracy(acc)[-1] < np.mean(acc) - 10


def test_transpositions_share_a_position():
    a = position_keys(["e4", "e5", "Nf3", "Nc6"])[-1]
    b = position_keys(["Nf3", "Nc6", "e4", "e5"])[-1]
    assert a == b and position_keys(["e4", "e5", "Nf3", "Nf6"])[-1] != a


def toy_book(tmp_path):
    """Start position (1000 games), after 1.e4 (400) and after 1.e4 e5 (200)."""
    keys = position_keys(["e4", "e5"])
    order = np.argsort(keys)
    np.savez(tmp_path / "book.npz", keys=keys[order], n=np.array([1000, 400, 200])[order],
             white=np.array([1500.0, 1520, 1540])[order], black=np.array([1450.0, 1470, 1490])[order],
             sd=np.full(3, 300.0)[order])
    return Book(tmp_path / "book.npz")


def test_book_counts_only_positions_after_own_moves(tmp_path):
    book = toy_book(tmp_path)
    found = book.find(position_keys(["e4", "e5", "Qh5", "Ke7", "Qxe5#"]))
    assert (found[:3] >= 0).all() and (found[3:] == -1).all()
    # White: start, after 1.e4, then 20 halved after each of its out-of-book moves (plies 3 and 5)
    assert book_weights(found, book.n, side=0)[0].tolist() == [1000, 400, 400, 10, 10, 5]
    # Black: start until it moves, after 1...e5, then 20 halved after 2...Ke7
    assert book_weights(found, book.n, side=1)[0].tolist() == [1000, 1000, 200, 200, 10, 10]
    out = estimates(["e4", "e5"], [30, 30], toy_table(), book)
    assert out[0]["black"]["rating"] == round((1000 * 1450 + BOOK_PRIOR * 1600) / (1000 + BOOK_PRIOR))
    assert out[1]["black"] == out[0]["black"]  # 1.e4 doesn't change Black's estimate


def test_each_line_moves_only_on_its_own_moves(tmp_path):
    out = estimates(QUEEN_BLUNDER, QUEEN_BLUNDER_EVALS, toy_table(), toy_book(tmp_path))
    for t in range(1, len(out)):
        idle = "black" if t % 2 else "white"
        assert out[t][idle] == out[t - 1][idle]
    white = line(out, "white")
    assert white[4] - white[5] > 300  # 3.Qxf7+?? drops White steeply


@pytest.mark.skipif(not (MODEL / "performance.json").exists(), reason="no fitted table in model/")
def test_fitted_table_drops_steeply_on_a_hung_queen():
    table = Table.load(MODEL / "performance.json")
    assert (np.diff(table.median[1:], axis=1) >= 0).all()  # never lower for a higher accuracy
    book = Book(MODEL / "opening_book.npz") if (MODEL / "opening_book.npz").exists() else None
    out = estimates(QUEEN_BLUNDER, QUEEN_BLUNDER_EVALS, table, book)
    white, black = line(out, "white"), line(out, "black")
    assert white[4] - white[5] > 200
    assert black[5] == black[4]

from pathlib import Path

import numpy as np
import pytest

from performance import BOOK_PRIOR, Book, Table, book_weights, estimates, game_accuracy, move_accuracy, position_keys

MODEL = Path(__file__).parent / "model"
SANS = ["e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6", "Ng5", "d5", "exd5", "Qe7", "Nxf7", "Qxe2+"]  # evals below are made up


def toy_table():
    """Rating rises linearly with accuracy: 600 at 0% to 2600 at 100%, the same for every k."""
    centers = np.arange(1, 100, 2)
    row = 600 + 20 * centers
    rows = np.vstack([np.full(50, 1600.0)] + [row] * 40)
    return Table({"metric": "blend", "step": 2, "median": rows.tolist(),
                  "low": (rows - 300).tolist(), "high": (rows + 300).tolist()})


def test_move_accuracy():
    assert move_accuracy([20, 20, 20, -900]).round().tolist() == [100, 100, 100, 100]  # nobody gave anything away
    assert move_accuracy([20, 900])[1] < 15  # Black's move hands White nine pawns


def test_blunder_drops_only_the_movers_line():
    evals = [30] * 9 + [-50, -900]  # Black plays well; White's 11th half-move throws away ~9 pawns
    out = estimates(SANS, evals, toy_table())
    white = [e["white"]["rating"] for e in out]
    black = [e["black"]["rating"] for e in out]
    assert white[-2] - white[-1] > 300  # steep drop on the blunder
    assert black[-1] == black[-2]  # Black didn't move, its line doesn't change
    assert black[-1] > 2000  # Black's moves were all accurate


def test_harmonic_blend_punishes_a_blunder_more_than_the_mean():
    acc = [95] * 10 + [0]
    assert game_accuracy(acc, "blend")[-1] < game_accuracy(acc, "mean")[-1] - 10


def test_transpositions_share_a_position():
    a = position_keys(["e4", "e5", "Nf3", "Nc6"])[-1]
    b = position_keys(["Nf3", "Nc6", "e4", "e5"])[-1]
    assert a == b and position_keys(["e4", "e5", "Nf3", "Nf6"])[-1] != a


def test_book_weight_halves_per_own_move_after_leaving(tmp_path):
    keys = position_keys(["e4", "e5"])
    path = tmp_path / "book.npz"
    order = np.argsort(keys)
    np.savez(path, keys=keys[order], n=np.array([1000, 400, 200])[order], white=np.full(3, 1500.0)[order],
             black=np.full(3, 1450.0)[order], sd=np.full(3, 300.0)[order])
    book = Book(path)
    found = book.find(position_keys(["e4", "e5", "Qh5", "Ke7", "Qxe5#"]))
    assert (found[:3] >= 0).all() and (found[3:] == -1).all()
    weight, _ = book_weights(found, book.n, side=0)
    assert weight.tolist() == [1000, 400, 200, 100, 100, 50]  # White moves at plies 3 and 5
    out = estimates(["e4", "e5"], [30, 30], toy_table(), book)
    assert out[0]["black"]["rating"] == round((1000 * 1450 + BOOK_PRIOR * 1600) / (1000 + BOOK_PRIOR))


@pytest.mark.skipif(not (MODEL / "performance.json").exists(), reason="no fitted table in model/")
def test_fitted_table_drops_steeply_on_a_hung_queen():
    table = Table.load(MODEL / "performance.json")
    assert (np.diff(table.median[1:], axis=1) >= 0).all()  # never lower for a higher accuracy
    evals = [30] * 19 + [-900]  # White's 10th move hangs a queen
    sans = ["Nf3", "Nf6", "Ng1", "Ng8"] * 5
    out = estimates(sans, evals, table)
    assert out[-2]["white"]["rating"] - out[-1]["white"]["rating"] > 150
    assert out[-1]["black"]["rating"] == out[-2]["black"]["rating"]

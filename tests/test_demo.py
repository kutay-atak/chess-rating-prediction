import numpy as np
import pytest

from demo import bands, game_id, load_model, read_pgn

PGN = """[Event "Rated blitz game"]
[White "a"]
[Black "b"]
[WhiteElo "1500"]
[BlackElo "1520"]
[TimeControl "180+2"]

1. e4 { [%clk 0:03:00] } 1... e5 { [%clk 0:03:00] } 2. Nf3 { [%clk 0:02:58] } 2... Nc6 { [%clk 0:02:55] }
3. Bb5 { [%clk 0:02:57] } 3... a6 { [%clk 0:02:50] } 1-0
"""


def test_game_id():
    assert game_id("https://lichess.org/q7ZvsdUF") == "q7ZvsdUF"
    assert game_id("lichess.org/q7ZvsdUF/black#12") == "q7ZvsdUF"
    assert game_id("https://example.com/q7ZvsdUF") is None


def test_read_pgn():
    headers, sans, clocks, base, inc = read_pgn(PGN)
    assert headers["WhiteElo"] == "1500" and sans[:2] == ["e4", "e5"] and clocks[2] == 178 and (base, inc) == (180, 2)
    with pytest.raises(ValueError, match="blitz"):
        read_pgn(PGN.replace("180+2", "600+0"))
    with pytest.raises(ValueError, match="clock"):
        read_pgn(PGN.replace("{ [%clk 0:02:50] }", ""))


def test_bands_per_ply():
    model, saved = load_model()
    _, sans, clocks, base, inc = read_pgn(PGN)
    b = bands(model, saved, sans, clocks, base, inc)
    assert b.shape == (6, 2, 3) and np.isfinite(b).all()
    assert (b[..., 0] < b[..., 1]).all() and (b[..., 1] < b[..., 2]).all()

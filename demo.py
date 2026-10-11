"""The GRU on one game, for the demo app (app.py): rating bands for both players after every ply.

Plain functions, so they are testable without Streamlit.
"""
import io
import re
import urllib.request

import chess.pgn
import numpy as np
import torch

from features import CHECKPOINTS
from gru import GRU, encode, ply_inputs
from prepare_games import parse

GAME_ID = re.compile(r"lichess\.org/([A-Za-z0-9]{8})")


def game_id(link):
    """The 8-character game id in a lichess.org link (game, player-view or analysis URLs), or None."""
    m = GAME_ID.search(link)
    return m[1] if m else None


def fetch_pgn(gid):
    """PGN with clocks from Lichess's game export API."""
    req = urllib.request.Request(f"https://lichess.org/game/export/{gid}?clocks=true&evals=false",
                                 headers={"Accept": "application/x-chess-pgn"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return r.read().decode()


def read_pgn(text):
    """(headers, SAN moves, clocks, base, inc) of a blitz game with clocks; ValueError otherwise."""
    game = chess.pgn.read_game(io.StringIO(text))
    if game is None:
        raise ValueError("That doesn't look like a PGN.")
    headers = dict(game.headers)
    tc = re.fullmatch(r"(\d+)\+(\d+)", headers.get("TimeControl", ""))
    if not tc:
        raise ValueError("The game has no standard time control.")
    base, inc = int(tc[1]), int(tc[2])
    if not 180 <= base + 40 * inc <= 479:  # Lichess's own blitz bounds, the only games the model has seen
        raise ValueError(f"This is a {base // 60}+{inc} game; the model only knows blitz (3+0 up to 8+0).")
    movetext = text.split("\n\n", 1)[1] if "\n\n" in text else text
    parsed = parse(" ".join(movetext.split()))
    if parsed is None:
        raise ValueError("The game needs a clock on every move (export it with clocks).")
    sans, clocks, _ = parsed  # evals are ignored: the model never uses them
    return headers, sans, clocks, base, inc


def load_model(path="model.pt"):
    saved = torch.load(path)
    model = GRU(len(saved["vocab"]) + 2)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return model, saved


def bands(model, saved, sans, clocks, base, inc):
    """[plies, 2, 3] array: for White and Black after each ply, (lo, mean, hi) of the rating in points.

    The interval multiplier is the conformal one of the last checkpoint at or before the ply (ply 1 uses
    ply 2's), since the model was calibrated only at the checkpoints."""
    index = {m: i + 2 for i, m in enumerate(saved["vocab"])}
    moves = torch.from_numpy(encode(sans, index).astype(np.int64))[None]
    x = torch.from_numpy(ply_inputs(clocks, base, inc))[None]
    with torch.no_grad():
        mean, logvar = model(moves, x)
    m, sd = saved["rating_mean"], saved["rating_sd"]
    mu = mean[0].numpy() * sd + m
    s = np.exp(logvar[0].numpy() / 2) * sd
    q = saved["conformal"]
    t = np.arange(1, len(sans) + 1)
    mult = np.array([q[max([c for c in CHECKPOINTS if c <= k] or [CHECKPOINTS[0]])] for k in t])[:, None]
    return np.stack([mu - mult * s, mu, mu + mult * s], axis=2)

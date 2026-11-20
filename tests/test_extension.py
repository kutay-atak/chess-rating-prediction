import json
import random
import shutil
import subprocess

import chess
import pytest

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="needs Node")


def legal_count(board, color, square):
    """What extension/control.js counts: pieces of `color` that could legally capture on `square`."""
    n = 0
    for a in board.attackers(color, square):
        if board.piece_type_at(a) == chess.KING:
            without = board.copy(stack=False)
            without.remove_piece_at(a)
            if without.is_attacked_by(not color, square):
                continue  # a king can't step onto a covered square
        elif board.is_pinned(color, a) and square not in board.pin(color, a):
            continue  # a pinned piece stays on the pin line
        n += 1
    return n


def positions():
    fens = ["4k3/8/2n5/1B6/8/8/8/4K3 b - - 0 1",  # knight pinned to its king
            "4k3/8/8/8/8/8/3r4/4K3 w - - 0 1",  # king beside a rook
            "1n1k2n1/3pN3/2p4P/pR1P1p2/PbN4p/2P1pp1P/8/2BRKQ1r w - - 6 49"]  # pinned queen beside its king
    rng = random.Random(0)
    for _ in range(60):  # random games, so pins, checks and promotions all turn up
        board = chess.Board()
        for _ in range(rng.randint(4, 120)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
        fens.append(board.fen())
    return [chess.Board(f) for f in fens]


def test_control_matches_python_chess():
    boards = positions()
    squares = [[(p.symbol() if p else None) for p in map(b.piece_at, chess.SQUARES)] for b in boards]
    script = f"const c = require('./extension/control.js'); console.log(JSON.stringify({json.dumps(squares)}.map((b) => c.control(b))))"
    js = json.loads(subprocess.run([NODE, "-e", script], capture_output=True, text=True, check=True).stdout)
    assert js == [[[legal_count(b, chess.WHITE, s), legal_count(b, chess.BLACK, s)] for s in chess.SQUARES] for b in boards]

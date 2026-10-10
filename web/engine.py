"""Stockfish evals for live games, on the same scale as the training data's Lichess evals:
centipawns from White's side after each move, a forced mate as ±MATE."""
import os
import shutil
import threading

import chess
import chess.engine

from prepare_games import MATE

DEPTH = int(os.environ.get("STOCKFISH_DEPTH", 14))


def find_stockfish():
    return os.environ.get("STOCKFISH") or shutil.which("stockfish")


class Engine:
    def __init__(self, path=None, depth=DEPTH):
        self.engine = chess.engine.SimpleEngine.popen_uci(path or find_stockfish())
        self.engine.configure({"Threads": 1, "Hash": 32})
        self.depth = depth
        self.lock = threading.Lock()  # one search at a time; the web server calls from threads

    def eval(self, board):
        if board.is_checkmate():
            return -MATE if board.turn == chess.WHITE else MATE
        if board.is_game_over():
            return 0
        with self.lock:
            info = self.engine.analyse(board, chess.engine.Limit(depth=self.depth))
        return info["score"].white().score(mate_score=MATE)

    def evals(self, sans):
        """Eval after each of these SAN moves from the starting position."""
        board, out = chess.Board(), []
        for san in sans:
            board.push_san(san)
            out.append(self.eval(board))
        return out

    def close(self):
        self.engine.quit()

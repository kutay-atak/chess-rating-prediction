"""One blitz game: rules from python-chess, clocks kept the way Lichess records them.

Lichess (where the training games come from) doesn't run a side's clock for its first move, then
takes the thinking time off and adds the increment after every move; `clocks` holds the mover's
seconds left after each move, the same series prepare_games.py extracts. Like chess.com, a side that
doesn't make its first move within FIRST_MOVE_SECONDS aborts the game.
"""
import time

import chess

FIRST_MOVE_SECONDS = 30
WIN = {chess.WHITE: "1-0", chess.BLACK: "0-1"}


class Game:
    def __init__(self, base, inc, now=time.monotonic):
        self.base, self.inc, self.now = base, inc, now
        self.board = chess.Board()
        self.sans, self.ucis, self.clocks = [], [], []
        self.remaining = {chess.WHITE: float(base), chess.BLACK: float(base)}
        self.turn_started = now()
        self.result = self.reason = None
        self.draw_offer = None  # colour that offered a draw

    @property
    def ply(self):
        return len(self.sans)

    @property
    def over(self):
        return self.result is not None

    def start(self):
        """Restart the first-move countdown, e.g. once a friend has joined."""
        self.turn_started = self.now()

    def clock_running(self):
        return not self.over and self.ply >= 2

    def time_left(self, color):
        left = self.remaining[color]
        if self.clock_running() and self.board.turn == color:
            left -= self.now() - self.turn_started
        return max(left, 0.0)

    def first_move_left(self):
        """Seconds left for the side to move to make its first move, or None after both have."""
        if self.over or self.ply >= 2:
            return None
        return max(FIRST_MOVE_SECONDS - (self.now() - self.turn_started), 0.0)

    def tick(self):
        """End the game if the side to move has run out of time. True if this call ended it."""
        if self.over:
            return False
        if self.ply < 2:
            if self.first_move_left() <= 0:
                self._finish("*", "abandoned")
                return True
            return False
        mover = self.board.turn
        if self.time_left(mover) > 0:
            return False
        self.remaining[mover] = 0.0
        if self.board.has_insufficient_material(not mover):
            self._finish("1/2-1/2", "timeout vs insufficient material")
        else:
            self._finish(WIN[not mover], "time")
        return True

    def move(self, color, uci):
        """Play `uci` (e.g. 'e2e4', 'e7e8q') for `color`; ValueError if not allowed."""
        if self.tick() or self.over:
            raise ValueError("game over")
        if color != self.board.turn:
            raise ValueError("not your turn")
        move = chess.Move.from_uci(uci)
        if move not in self.board.legal_moves:
            raise ValueError("illegal move")
        now = self.now()
        if self.ply >= 2:
            self.remaining[color] += self.inc - (now - self.turn_started)
        self.turn_started = now
        self.sans.append(self.board.san(move))
        self.ucis.append(move.uci())
        self.board.push(move)
        self.clocks.append(float(round(self.remaining[color])))  # Lichess records whole seconds
        if self.draw_offer == (not color):
            self.draw_offer = None  # moving declines the opponent's offer
        self._check_end()

    def resign(self, color):
        if not self.over:
            self._finish(WIN[not color], "resignation")

    def offer_draw(self, color):
        if self.over:
            return
        if self.draw_offer == (not color):
            self._finish("1/2-1/2", "agreement")
        else:
            self.draw_offer = color

    def decline_draw(self, color):
        if self.draw_offer == (not color):
            self.draw_offer = None

    def abort(self):
        """Allowed until both sides have made their first move."""
        if not self.over and self.ply < 2:
            self._finish("*", "aborted")

    def winner(self):
        return {"1-0": "white", "0-1": "black"}.get(self.result)

    def _check_end(self):
        b = self.board
        if b.is_checkmate():
            self._finish(WIN[not b.turn], "checkmate")
        elif b.is_stalemate():
            self._finish("1/2-1/2", "stalemate")
        elif b.is_insufficient_material():
            self._finish("1/2-1/2", "insufficient material")
        elif b.is_repetition(3):
            self._finish("1/2-1/2", "repetition")
        elif b.is_fifty_moves():
            self._finish("1/2-1/2", "50-move rule")

    def _finish(self, result, reason):
        if self.clock_running():
            mover = self.board.turn
            self.remaining[mover] = self.time_left(mover)
        self.result, self.reason, self.draw_offer = result, reason, None

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import web.server
from web.game import FIRST_MOVE_SECONDS, Game
from web.server import app

MODEL = Path(__file__).parent / "model"


def fake_predictor(sans, clocks, base, inc):
    """Stands in for the trained model so these tests don't need model/."""
    return [{"ply": t, "white": {"rating": 1500 + t, "low": 1400, "high": 1600 + t},
             "black": {"rating": 1500 - t, "low": 1400 - t, "high": 1600}} for t in range(len(sans) + 1)]


@pytest.fixture(autouse=True)
def no_model(monkeypatch):
    monkeypatch.setattr(web.server, "load_predictor", lambda: fake_predictor)

WHITE, BLACK = True, False
FOOLS_MATE = ["f2f3", "e7e5", "g2g4", "d8h4"]


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def play(game, ucis, think=0.0):
    for uci in ucis:
        game.now.t += think
        game.move(game.board.turn, uci)


def test_checkmate():
    game = Game(180, 0, FakeClock())
    play(game, FOOLS_MATE)
    assert (game.result, game.reason, game.winner()) == ("0-1", "checkmate", "black")
    assert game.sans == ["f3", "e5", "g4", "Qh4#"]


def test_rejects_illegal_and_out_of_turn_moves():
    game = Game(180, 0, FakeClock())
    with pytest.raises(ValueError):
        game.move(WHITE, "e2e5")
    with pytest.raises(ValueError):
        game.move(BLACK, "e7e5")
    with pytest.raises(ValueError):
        game.move(WHITE, "nonsense")
    assert game.ply == 0


def test_clocks_follow_lichess_convention():
    game = Game(180, 2, FakeClock())
    play(game, ["e2e4", "e7e5", "g1f3", "b8c6"], think=5)
    # first move per side is free; then thinking time comes off and the increment goes on
    assert game.clocks == [180, 180, 177, 177]
    game.now.t += 10
    assert game.time_left(WHITE) == 167 and game.time_left(BLACK) == 177


def test_flag_fall_and_insufficient_material():
    game = Game(180, 0, FakeClock())
    play(game, ["e2e4", "e7e5"])
    game.now.t += 181
    assert game.tick() and (game.result, game.reason) == ("0-1", "time")

    game = Game(180, 0, FakeClock())
    game.board.set_fen("4k3/8/8/8/8/8/8/R3K3 w - - 0 1")  # Black has only a king: can't win on time
    game.sans, game.clocks = ["a", "b"], [180, 180]
    game.now.t += 200
    assert game.tick() and game.result == "1/2-1/2" and game.reason == "timeout vs insufficient material"


def test_first_move_timeout_abandons():
    game = Game(180, 0, FakeClock())
    game.now.t += FIRST_MOVE_SECONDS + 1
    assert game.tick() and (game.result, game.reason) == ("*", "abandoned")


def test_draw_offer_accept_and_decline_by_moving():
    game = Game(180, 0, FakeClock())
    play(game, ["e2e4"])
    game.offer_draw(WHITE)
    play(game, ["e7e5"])  # Black moves instead of accepting
    assert game.draw_offer is None
    game.offer_draw(WHITE)
    game.offer_draw(BLACK)
    assert (game.result, game.reason) == ("1/2-1/2", "agreement")


def test_repetition():
    game = Game(180, 0, FakeClock())
    play(game, ["g1f3", "g8f6", "f3g1", "f6g8"] * 2)
    assert game.reason == "repetition"


def hello(ws, token, name):
    ws.send_json({"type": "hello", "token": token, "name": name})


def next_state(ws):
    while True:
        msg = ws.receive_json()
        if msg["type"] == "state":
            return msg


def test_matchmaking_game_over_websocket():
    with TestClient(app) as client, client.websocket_connect("/ws") as a, client.websocket_connect("/ws") as b:
        hello(a, "token-a", "alice")
        hello(b, "token-b", "bob")
        a.send_json({"type": "seek", "tc": "3+0"})
        assert a.receive_json()["type"] == "seeking"
        b.send_json({"type": "seek", "tc": "3+0"})
        room = a.receive_json()["id"]
        assert b.receive_json() == {"type": "matched", "id": room}
        a.send_json({"type": "join", "id": room})
        next_state(a)
        b.send_json({"type": "join", "id": room})
        state = next_state(b)
        assert {state["players"]["white"]["name"], state["players"]["black"]["name"]} == {"alice", "bob"}
        assert state["model"] and len(state["estimates"]) == 1  # estimate before any move
        sockets = {"white": a if next_state(a)["you"] == ["white"] else b}
        sockets["black"] = b if sockets["white"] is a else a

        for i, uci in enumerate(FOOLS_MATE):
            sockets["white" if i % 2 == 0 else "black"].send_json({"type": "move", "id": room, "uci": uci})
            state = next_state(a)
            next_state(b)
        assert state["result"] == "0-1" and state["reason"] == "checkmate"
        assert len(state["estimates"]) == 5
        est = state["estimates"][-1]["white"]
        assert est["low"] <= est["rating"] <= est["high"]


def test_pass_and_play_moves_both_sides():
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        hello(ws, "token-local", "me")
        ws.send_json({"type": "local", "tc": "5+3", "names": ["Ann", "Ben"]})
        room = ws.receive_json()["id"]
        ws.send_json({"type": "join", "id": room})
        assert next_state(ws)["you"] == ["white", "black"]
        for uci in ["e2e4", "e7e5"]:
            ws.send_json({"type": "move", "id": room, "uci": uci})
            state = next_state(ws)
        assert state["sans"] == ["e4", "e5"]
        ws.send_json({"type": "draw", "id": room})
        assert next_state(ws)["reason"] == "agreement"


def test_friend_link_seats_the_second_player():
    with TestClient(app) as client, client.websocket_connect("/ws") as a, client.websocket_connect("/ws") as b:
        hello(a, "token-host", "host")
        hello(b, "token-guest", "guest")
        a.send_json({"type": "friend", "tc": "5+0"})
        room = a.receive_json()["id"]
        a.send_json({"type": "join", "id": room})
        assert not next_state(a)["started"]
        b.send_json({"type": "join", "id": room})
        state = next_state(b)
        assert state["started"] and len(state["you"]) == 1
        assert {p["name"] for p in state["players"].values()} == {"host", "guest"}


@pytest.mark.skipif(not (MODEL / "rating_model.onnx").exists(), reason="no trained model in model/")
def test_trained_model_gives_an_estimate_per_ply():
    from web.predictor import Predictor
    out = Predictor(MODEL)(["e4", "e5", "Nf3"], [180, 180, 178], 180, 2)
    assert [e["ply"] for e in out] == [0, 1, 2, 3]
    for e in out:
        for side in ("white", "black"):
            assert 0 < e[side]["low"] <= e[side]["rating"] <= e[side]["high"] < 4000

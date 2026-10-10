"""Two-player blitz with each player's live game performance rating.

  uvicorn web.server:app        (from the repo root; MODEL_DIR defaults to model/)

One WebSocket per browser tab at /ws. A client says hello with a token it keeps in localStorage (so
a reload gets its seat back), then seeks a game (matchmaking per time control), creates a friend
game (invite link /game/<id>) or a pass-and-play game (one client plays both sides), and joins
games by id. After every change the server sends each client in a game the full game state,
including each player's estimate after every ply so far.

Each move is scored by Stockfish (on PATH, or STOCKFISH=/path) in the background, and the estimates
are per-game performance ratings (performance.py: opening book, then move accuracy); the estimate for
a move arrives once its eval is in. Without Stockfish or the fitted files in model/, or if the engine
fails mid-game, games still play and the page says the estimate is unavailable.
"""
import asyncio
import os
import random
import secrets
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import chess
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import performance
from web.engine import Engine, find_stockfish
from web.game import Game

STATIC = Path(__file__).parent / "static"
TIME_CONTROLS = {"3+0": (180, 0), "3+2": (180, 2), "5+0": (300, 0), "5+3": (300, 3)}  # blitz only, like the training games
COLOR = {chess.WHITE: "white", chess.BLACK: "black"}
FINISHED_ROOM_TTL = 30 * 60


@dataclass(eq=False)
class Client:
    ws: WebSocket
    token: str = ""
    name: str = "Guest"


@dataclass(eq=False)
class Room:
    id: str
    mode: str  # "online", "friend" or "local"
    tc: str
    game: Game
    seats: dict  # chess colour -> token or None
    names: dict  # chess colour -> display name
    started: bool = True
    clients: set = field(default_factory=set)
    estimates: list = field(default_factory=list)
    evals: list | None = None  # engine eval after each ply so far; None: no estimate for this game
    analysing: bool = False
    rematch: set = field(default_factory=set)
    next_id: str | None = None
    ended_at: float | None = None


rooms: dict[str, Room] = {}
seeks: dict[str, Client] = {}  # time control -> the client waiting for an opponent
MODEL_DIR = Path(os.environ.get("MODEL_DIR", Path(__file__).parent.parent / "model"))
perf = None  # (accuracy table, opening book or None) for the performance rating
engine = None
unavailable = None  # why there is no estimate, or None


def load_performance():
    if not (MODEL_DIR / "performance.json").exists():
        return None
    book = MODEL_DIR / "opening_book.npz"
    return performance.Table.load(MODEL_DIR / "performance.json"), performance.Book(book) if book.exists() else None


def load_engine():
    try:
        return Engine() if find_stockfish() else None
    except Exception as e:
        print(f"Stockfish didn't start ({e}); games will have no estimate")
        return None


def new_room(mode, tc, seats, names):
    room = Room(id=secrets.token_urlsafe(6), mode=mode, tc=tc, game=Game(*TIME_CONTROLS[tc]),
                seats=seats, names=names, started=all(seats.values()), evals=[] if engine and perf else None)
    update_estimates(room)
    rooms[room.id] = room
    return room


def update_estimates(room):
    """Estimates after every ply whose eval is in."""
    room.estimates = performance.estimates(room.game.sans, room.evals, *perf) if room.evals is not None else []


async def analyse(room):
    """Score the plies that have no eval yet, in order, then refresh the estimates."""
    if room.analysing:
        return
    room.analysing = True
    try:
        while room.evals is not None and len(room.evals) < room.game.ply:
            board = chess.Board()
            for uci in room.game.ucis[:len(room.evals) + 1]:
                board.push_uci(uci)
            try:
                room.evals.append(await asyncio.to_thread(engine.eval, board))
            except Exception:
                room.evals = None  # the engine failed: this game carries on without an estimate
        update_estimates(room)
        await broadcast(room)
    finally:
        room.analysing = False


def state(room, client):
    g = room.game
    running = COLOR[g.board.turn] if g.clock_running() and room.started else None
    first = g.first_move_left() if room.started else None
    return {
        "type": "state", "id": room.id, "mode": room.mode, "tc": room.tc, "base": g.base, "inc": g.inc,
        "started": room.started,
        "players": {COLOR[c]: {"name": room.names[c], "seated": room.seats[c] is not None,
                               "online": any(o.token == room.seats[c] for o in room.clients)} for c in COLOR},
        "you": [COLOR[c] for c in COLOR if room.seats[c] == client.token],
        "ucis": g.ucis, "sans": g.sans, "fen": g.board.fen(), "turn": COLOR[g.board.turn],
        "check": g.board.is_check(),
        "clocks": {COLOR[c]: round(g.time_left(c) * 1000) for c in COLOR}, "running": running,
        "first_move_left": None if first is None else round(first * 1000),
        "result": g.result, "reason": g.reason, "winner": g.winner(),
        "draw_offer": COLOR[g.draw_offer] if g.draw_offer is not None else None,
        "rematch": sorted(COLOR[c] for c in room.rematch), "next_id": room.next_id,
        "estimates": room.estimates,
        "unavailable": None if room.evals is not None else unavailable or "Stockfish stopped during this game",
    }


async def send(client, message):
    try:
        await client.ws.send_json(message)
    except Exception:  # the socket closed under us; the disconnect handler cleans up
        pass


async def broadcast(room):
    if room.game.over and room.ended_at is None:
        room.ended_at = time.monotonic()
    for client in list(room.clients):
        await send(client, state(room, client))


async def join(client, room):
    """Attach a client to a room; a friend game's empty seat goes to the first newcomer."""
    for c in COLOR:
        if room.seats[c] is None and client.token not in room.seats.values():
            room.seats[c], room.names[c] = client.token, client.name
            room.started = True
            room.game.start()
    room.clients.add(client)
    await broadcast(room)


def seat_of(room, client):
    """The colour this client moves for: its only seat, or the side to move in pass-and-play."""
    mine = [c for c in COLOR if room.seats[c] == client.token]
    if len(mine) == 2:
        return room.game.board.turn
    return mine[0] if mine else None


async def handle(client, msg):
    kind = msg.get("type")
    if kind == "hello":
        client.token = str(msg.get("token") or secrets.token_urlsafe(16))[:64]
        client.name = (str(msg.get("name") or "").strip() or "Guest")[:24]
        return
    if kind in ("seek", "friend", "local"):
        tc = msg.get("tc")
        if tc not in TIME_CONTROLS:
            return await send(client, {"type": "error", "message": "unknown time control"})
        if kind == "local":
            names = msg.get("names") or ["White", "Black"]
            room = new_room("local", tc, {chess.WHITE: client.token, chess.BLACK: client.token},
                            {chess.WHITE: str(names[0])[:24] or "White", chess.BLACK: str(names[1])[:24] or "Black"})
        elif kind == "friend":
            color = random.choice(list(COLOR))
            room = new_room("friend", tc, {color: client.token, not color: None}, {color: client.name, not color: "Waiting…"})
        else:
            other = seeks.pop(tc, None)
            if other is None or other.token == client.token:
                seeks[tc] = client
                return await send(client, {"type": "seeking", "tc": tc})
            first, second = random.sample([client, other], 2)
            room = new_room("online", tc, {chess.WHITE: first.token, chess.BLACK: second.token},
                            {chess.WHITE: first.name, chess.BLACK: second.name})
            await send(other, {"type": "matched", "id": room.id})
        return await send(client, {"type": "matched", "id": room.id})
    if kind == "cancel_seek":
        for tc, c in list(seeks.items()):
            if c is client:
                del seeks[tc]
        return
    room = rooms.get(msg.get("id"))
    if room is None:
        return await send(client, {"type": "error", "message": "game not found"})
    if kind == "join":
        for other in rooms.values():
            other.clients.discard(client)
        return await join(client, room)
    color, g = seat_of(room, client), room.game
    if color is None or (not room.started and kind != "abort"):
        return
    if kind == "move":
        try:
            g.move(color, str(msg.get("uci")))
        except ValueError as e:
            await send(client, {"type": "error", "message": str(e)})
            return await broadcast(room)
        if room.evals is None:
            update_estimates(room)
        else:
            asyncio.create_task(analyse(room))
    elif kind == "resign":
        g.resign(color)
    elif kind == "draw":
        g.offer_draw(color)
        if room.mode == "local":  # both players share this screen, so an offer is an agreement
            g.offer_draw(not color)
    elif kind == "decline_draw":
        g.decline_draw(color)
    elif kind == "abort":
        g.abort()
    elif kind == "rematch" and g.over and room.next_id is None:
        room.rematch.update(c for c in COLOR if room.seats[c] == client.token)
        if len(room.rematch) == 2:
            nxt = new_room(room.mode, room.tc, {chess.WHITE: room.seats[chess.BLACK], chess.BLACK: room.seats[chess.WHITE]},
                           {chess.WHITE: room.names[chess.BLACK], chess.BLACK: room.names[chess.WHITE]})
            room.next_id = nxt.id
    await broadcast(room)


async def ticker():
    """Flag fall and first-move timeouts happen without any message arriving, so poll for them."""
    while True:
        await asyncio.sleep(0.1)
        now = time.monotonic()
        for room in list(rooms.values()):
            if room.started and room.game.tick():
                await broadcast(room)
            if room.ended_at and now - room.ended_at > FINISHED_ROOM_TTL and not room.clients:
                del rooms[room.id]


@asynccontextmanager
async def lifespan(app):
    global perf, engine, unavailable
    perf = load_performance()
    engine = load_engine() if perf else None  # no point scoring moves otherwise
    unavailable = ("No fitted performance table in model/ (see the README)" if not perf else
                   "Stockfish isn't installed where this site runs (see the README)" if not engine else None)
    if unavailable:
        print(f"Estimate unavailable: {unavailable}")
    task = asyncio.create_task(ticker())
    yield
    task.cancel()
    if engine:
        engine.close()


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
@app.get("/game/{room_id}")
async def page(room_id: str = ""):
    return FileResponse(STATIC / "index.html")


@app.websocket("/ws")
async def websocket(ws: WebSocket):
    await ws.accept()
    client = Client(ws)
    try:
        while True:
            await handle(client, await ws.receive_json())
    except WebSocketDisconnect:
        pass
    finally:
        for tc, c in list(seeks.items()):
            if c is client:
                del seeks[tc]
        for room in list(rooms.values()):
            if client in room.clients:
                room.clients.discard(client)
                await broadcast(room)

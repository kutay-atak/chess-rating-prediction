import { Chess } from "/static/vendor/chess.js";
import { Board } from "/static/js/board.js";
import { drawChart } from "/static/js/chart.js";
import { icon } from "/static/js/icons.js";
import { play, soundOn, toggleSound } from "/static/js/sound.js";

const START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
const TIME_CONTROLS = { "3+0": "3 min", "3+2": "3 | 2", "5+0": "5 min", "5+3": "5 | 3" };
const REASONS = {
  checkmate: "by checkmate", resignation: "by resignation", time: "on time", agreement: "by agreement",
  stalemate: "by stalemate", "insufficient material": "by insufficient material", repetition: "by repetition",
  "50-move rule": "by 50-move rule", "timeout vs insufficient material": "timeout vs insufficient material",
  aborted: "Game aborted", abandoned: "No first move was played",
};
const FIGURINE = { N: "♞", B: "♝", R: "♜", Q: "♛", K: "♚" };
const VALUE = { p: 1, n: 3, b: 3, r: 5, q: 9 };
const START_COUNT = { p: 8, n: 2, b: 2, r: 2, q: 1 };

const $ = (sel) => document.querySelector(sel);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
const store = {
  get(k, d) { try { return localStorage.getItem(k) ?? d; } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch {} },
};

function randomId() {
  const bytes = crypto.getRandomValues(new Uint8Array(12));
  return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
}

// One token per tab (sessionStorage): a reload keeps your seat, a second tab is a second player.
function tabToken() {
  try {
    let t = sessionStorage.getItem("token");
    if (!t) { t = randomId(); sessionStorage.setItem("token", t); }
    return t;
  } catch { return randomId(); }
}

const S = {
  ws: null,
  token: tabToken(),
  name: store.get("name", `Guest${Math.floor(1000 + Math.random() * 9000)}`),
  tc: TIME_CONTROLS[store.get("tc")] ? store.get("tc") : "3+0",
  view: "home", // home | seeking | local | game
  tab: "game",
  tcOpen: false,
  roomId: null,
  game: null,
  receivedAt: 0,
  positions: [],
  viewPly: 0,
  optimistic: null,
  soundPly: 0,
  flipped: false,
  modalFor: null,
  confirmResign: false,
  lowWarned: false,
};
store.set("name", S.name);

const board = new Board($("#board"), { onMove });

// ---------- connection ----------

function connect() {
  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  S.ws = ws;
  ws.onopen = () => {
    send({ type: "hello", token: S.token, name: S.name });
    if (S.roomId) send({ type: "join", id: S.roomId });
    else if (S.view === "seeking") send({ type: "seek", tc: S.tc });
  };
  ws.onmessage = (e) => onMessage(JSON.parse(e.data));
  ws.onclose = () => setTimeout(connect, 1000);
}

function send(msg) {
  if (S.ws && S.ws.readyState === WebSocket.OPEN) S.ws.send(JSON.stringify(msg));
}

function onMessage(msg) {
  if (msg.type === "seeking") { S.view = "seeking"; render(); }
  else if (msg.type === "matched") go(`/game/${msg.id}`);
  else if (msg.type === "state") onState(msg);
  else if (msg.type === "error") {
    if (msg.message === "game not found") { toast("That game doesn't exist any more."); go("/"); }
    else if (S.optimistic) { S.optimistic = null; render(); }
  }
}

function go(path, push = true) {
  if (push && location.pathname !== path) history.pushState(null, "", path);
  const m = path.match(/^\/game\/([\w-]+)/);
  S.roomId = m ? m[1] : null;
  S.optimistic = null;
  S.confirmResign = false;
  $("#game-over").hidden = true;
  if (S.roomId) { S.tab = "game"; send({ type: "join", id: S.roomId }); }
  else { S.game = null; if (S.view !== "seeking") S.view = "home"; }
  render();
}

window.addEventListener("popstate", () => go(location.pathname, false));

// ---------- game state ----------

function buildPositions(ucis) {
  // reuse the positions already computed when the move list only grew
  const known = S.positions;
  let start = 0;
  while (start < ucis.length && start + 1 < known.length && known[start + 1].uci === ucis[start]) start++;
  const positions = known.slice(0, start + 1);
  if (!positions.length) positions.push(positionInfo(new Chess(), null, null));
  const chess = new Chess(positions[positions.length - 1].fen);
  for (let i = start; i < ucis.length; i++) {
    const uci = ucis[i];
    const move = chess.move({ from: uci.slice(0, 2), to: uci.slice(2, 4), promotion: uci[4] });
    positions.push(positionInfo(chess, move, uci));
  }
  return positions;
}

function positionInfo(chess, move, uci) {
  let check = null;
  if (chess.inCheck()) {
    for (const row of chess.board()) for (const sq of row) if (sq && sq.type === "k" && sq.color === chess.turn()) check = sq.square;
  }
  return { fen: chess.fen(), uci, lastMove: move ? [move.from, move.to] : null, check, move };
}

function destsFor(fen) {
  const dests = new Map();
  for (const m of new Chess(fen).moves({ verbose: true })) {
    if (!dests.has(m.from)) dests.set(m.from, []);
    dests.get(m.from).push({ to: m.to, promotion: m.promotion });
  }
  return dests;
}

function soundFor(move) {
  if (!move) return;
  if (move.san.includes("+") || move.san.includes("#")) play("check");
  else if (move.captured) play("capture");
  else play("move");
}

function onState(g) {
  if (g.id !== S.roomId) return; // a broadcast from a game this tab has left
  const prev = S.game && S.game.id === g.id ? S.game : null;
  S.game = g;
  S.receivedAt = performance.now();
  S.view = "game";
  const atEnd = !prev || S.viewPly >= prev.ucis.length;
  S.positions = buildPositions(g.ucis);
  if (!prev) {
    S.viewPly = g.ucis.length;
    S.soundPly = g.ucis.length;
    S.flipped = false;
    S.modalFor = g.result ? g.id : null; // don't pop up the result of a game that ended before we came
    S.lowWarned = false;
    if (g.started && g.you.length && !g.ucis.length && !g.result) play("start");
  } else if (!prev.started && g.started) play("start");
  if (atEnd) S.viewPly = g.ucis.length;
  if (S.optimistic && (g.ucis.length >= S.optimistic.ply || g.result)) S.optimistic = null;
  if (g.ucis.length > S.soundPly) {
    soundFor(S.positions[g.ucis.length].move);
    S.soundPly = g.ucis.length;
  }
  if (g.result && S.modalFor !== g.id) {
    S.modalFor = g.id;
    S.confirmResign = false;
    play("end");
    showGameOver();
  }
  if (g.next_id && g.you.length) return go(`/game/${g.next_id}`);
  render();
}

function onMove(from, to, promotion) {
  const g = S.game;
  if (!g) return;
  const chess = new Chess(S.positions[g.ucis.length].fen);
  let move;
  try { move = chess.move({ from, to, promotion }); } catch { return; }
  const uci = from + to + (promotion || "");
  S.optimistic = { ply: g.ucis.length + 1, ...positionInfo(chess, move, uci) };
  S.soundPly = g.ucis.length + 1;
  soundFor(move);
  send({ type: "move", id: g.id, uci });
  render();
}

// ---------- helpers for the view ----------

function orientation() {
  const g = S.game;
  const base = g && g.you.length === 1 ? g.you[0] : "white";
  return S.flipped ? (base === "white" ? "black" : "white") : base;
}

function estimateAt(color, ply) {
  const est = S.game?.estimates || [];
  if (!est.length) return null;
  return est[Math.min(ply, est.length - 1)][color];
}

function timeLeft(color) {
  const g = S.game;
  if (!g) return +S.tc.split("+")[0] * 60 * 1000; // "3+0" is minutes + seconds
  let ms = g.clocks[color];
  if (g.running === color) ms -= performance.now() - S.receivedAt;
  return Math.max(0, ms);
}

function clockText(ms) {
  const s = ms / 1000;
  if (s < 20) return `0:${String(Math.floor(s)).padStart(2, "0")}.${Math.floor((s * 10) % 10)}`;
  const whole = Math.ceil(s);
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
}

function material(fen) {
  const count = { w: { p: 0, n: 0, b: 0, r: 0, q: 0 }, b: { p: 0, n: 0, b: 0, r: 0, q: 0 } };
  for (const ch of fen.split(" ")[0]) {
    const p = ch.toLowerCase();
    if (p in VALUE) count[ch === p ? "b" : "w"][p]++;
  }
  return count;
}

function capturedHtml(color, fen) {
  const count = material(fen);
  const mine = color === "white" ? "w" : "b", theirs = mine === "w" ? "b" : "w";
  let html = "", score = 0;
  for (const p of ["p", "n", "b", "r", "q"]) {
    score += VALUE[p] * (count[mine][p] - count[theirs][p]);
    const n = Math.max(0, START_COUNT[p] - count[theirs][p]);
    for (let i = 0; i < n; i++) html += `<img class="${i === 0 && html ? "gap" : ""}" src="/static/pieces/${theirs}${p.toUpperCase()}.svg" alt="">`;
  }
  return html + (score > 0 ? `<span class="adv">+${score}</span>` : "");
}

function avatar(color, cls = "") {
  return `<div class="avatar ${color} ${cls}"><img src="/static/pieces/${color === "white" ? "w" : "b"}P.svg" alt=""></div>`;
}

function plyLabel(ply) {
  if (ply === 0) return "before the first move";
  return `after ${Math.ceil(ply / 2)}${ply % 2 ? "." : "…"} ${S.positions[ply]?.move?.san ?? ""}`;
}

function sanHtml(san) {
  return FIGURINE[san[0]] ? `<span class="fig">${FIGURINE[san[0]]}</span>${esc(san.slice(1))}` : esc(san);
}

// ---------- rendering ----------

function render() {
  renderBoard();
  renderBars();
  const typing = document.activeElement && document.activeElement.tagName === "INPUT" && $("#panel").contains(document.activeElement);
  if (!typing) renderPanel(); // don't wipe a name being typed in the New Game tab
  renderBanner();
  updateClocks();
  const g = S.game;
  const myTurn = g && g.started && !g.result && g.you.includes(g.turn) && g.you.length === 1;
  document.title = myTurn ? "Your Turn · EloSight" : "EloSight · Play Chess";
}

function renderBoard() {
  const g = S.game;
  board.setOrientation(orientation());
  if (!g) return board.set({ fen: START_FEN });
  const latest = !S.optimistic && S.viewPly === g.ucis.length;
  const pos = S.optimistic || S.positions[Math.min(S.viewPly, S.positions.length - 1)];
  const canMove = latest && g.started && !g.result && g.you.includes(g.turn);
  board.set({ fen: pos.fen, lastMove: pos.lastMove, check: pos.check,
              movable: canMove ? g.turn[0] : null, dests: canMove ? destsFor(pos.fen) : new Map() });
}

function renderBars() {
  const bottom = orientation(), top = bottom === "white" ? "black" : "white";
  $("#bar-top").innerHTML = barHtml(top);
  $("#bar-bottom").innerHTML = barHtml(bottom);
}

function barHtml(color) {
  const g = S.game;
  let name, online = true, est = null, delta = 0, fen = START_FEN;
  if (g) {
    const p = g.players[color];
    name = p.name;
    online = p.online || !p.seated;
    const ply = S.optimistic ? g.ucis.length : S.viewPly;
    est = estimateAt(color, ply);
    const before = ply > 0 ? estimateAt(color, ply - 1) : null;
    if (est && before) delta = est.rating - before.rating;
    fen = (S.optimistic || S.positions[Math.min(S.viewPly, S.positions.length - 1)]).fen;
  } else {
    name = color === "white" ? S.name : "Opponent";
  }
  const rating = est ? `<span class="rating" title="80% range ${est.low}–${est.high}">(${est.rating} ± ${Math.round((est.high - est.low) / 2)})</span>` : "";
  const deltaHtml = delta ? `<span class="delta show ${delta > 0 ? "up" : "down"}">${delta > 0 ? "▲" : "▼"}${Math.abs(delta)}</span>` : "";
  const dot = g && g.mode !== "local" ? `<span class="status-dot ${online ? "" : "off"}" title="${online ? "online" : "offline"}"></span>` : "";
  return `${avatar(color)}
    <div class="pinfo">
      <div class="pname"><span class="name">${esc(name)}</span>${rating}${deltaHtml}${dot}</div>
      <div class="captured" data-first="${color}">${capturedHtml(color, fen)}</div>
    </div>
    <div class="clock ${color}" data-clock="${color}">${icon("tick", "tick")}<span class="time"></span></div>`;
}

function renderBanner() {
  const g = S.game, banner = $("#banner");
  let text = "";
  if (S.view === "seeking") text = "Searching for an opponent…";
  else if (g && !g.started) text = "Waiting for your friend to open the invite link…";
  banner.textContent = text;
  banner.hidden = !text;
}

function updateClocks() {
  const g = S.game;
  for (const el of document.querySelectorAll("[data-clock]")) {
    const color = el.dataset.clock;
    const ms = timeLeft(color);
    el.querySelector(".time").textContent = clockText(ms);
    const active = g && (g.running === color || (g.started && !g.result && g.turn === color));
    el.classList.toggle("active", !!active);
    el.classList.toggle("low", !!g && ms < 20000);
    if (g && g.running === color && g.you.length === 1 && g.you[0] === color && ms < 10000 && !S.lowWarned) {
      S.lowWarned = true;
      play("low");
    }
  }
  if (g && g.first_move_left != null && !g.result) {
    const left = Math.max(0, Math.ceil((g.first_move_left - (performance.now() - S.receivedAt)) / 1000));
    const el = document.querySelector(`[data-first="${g.turn}"]`);
    if (el) el.innerHTML = `<span style="color:#f7c631;font-size:12px;font-weight:600">${left} seconds to make the first move</span>`;
  }
}

function tick() {
  updateClocks();
  requestAnimationFrame(tick);
}

function renderPanel() {
  const panel = $("#panel");
  const g = S.game;
  const playing = g && g.you.length && !g.result;
  if (S.view === "game" && g) {
    const tabs = `<div class="tabs">
      <button class="tab ${S.tab === "game" ? "active" : ""}" data-act="tab" data-tab="game">${icon("board")}<span>Play</span></button>
      <button class="tab ${S.tab === "new" ? "active" : ""}" data-act="tab" data-tab="new">${icon("plus")}<span>New Game</span></button>
      <button class="tab" data-act="about">${icon("info")}<span>How it works</span></button></div>`;
    panel.innerHTML = tabs + (S.tab === "game" ? gamePanel(g) : homePanel(playing));
    if (S.tab === "game") {
      drawChart($("#chart"), g.estimates, S.optimistic ? g.ucis.length : S.viewPly, (ply) => setPly(ply));
      const cur = panel.querySelector(".san.current");
      const moves = $("#moves");
      if (cur) cur.scrollIntoView({ block: "nearest" });
      else if (S.viewPly >= g.ucis.length) moves.scrollTop = moves.scrollHeight;
    }
  } else if (S.view === "local") {
    panel.innerHTML = localPanel();
  } else {
    panel.innerHTML = homePanel(false);
  }
}

function tcPicker() {
  const options = Object.entries(TIME_CONTROLS).map(([tc, label]) =>
    `<button class="tc-option ${tc === S.tc ? "active" : ""}" data-act="tc" data-tc="${tc}">${label}</button>`).join("");
  return `<button class="tc-current" data-act="tc-toggle">${icon("bolt")}<span>${TIME_CONTROLS[S.tc]}</span><span style="color:var(--muted)">▾</span></button>
    <div class="tc-group" ${S.tcOpen ? "" : "hidden"}><h4>${icon("bolt")}Blitz</h4><div class="tc-grid">${options}</div></div>`;
}

function homePanel(busy) {
  const seeking = S.view === "seeking";
  const start = seeking
    ? `<div class="searching"><div class="spinner"></div>Searching for an opponent (${TIME_CONTROLS[S.tc]})…<button class="btn" data-act="cancel">Cancel</button></div>`
    : `<button class="btn-green big" data-act="seek" ${busy ? "disabled" : ""}>Start Game</button>`;
  return `<div class="panel-body">
    <div class="panel-title"><img src="/static/pieces/wN.svg" alt=""><h2>Play Chess</h2></div>
    <label class="field">Playing as<input id="name-input" maxlength="24" value="${esc(S.name)}"></label>
    ${tcPicker()}
    ${start}
    <button class="btn-row" data-act="friend" ${busy || seeking ? "disabled" : ""}>${icon("link")}<span>Play a Friend<small>Send an invite link</small></span></button>
    <button class="btn-row" data-act="local-setup" ${busy || seeking ? "disabled" : ""}>${icon("people")}<span>Pass and Play<small>Two players, one screen</small></span></button>
    ${busy ? `<p class="note">Finish your current game before starting another.</p>` : ""}
    <p class="note">While you play, a model trained on Lichess blitz games estimates both players' ratings after every move.
      <a class="link" data-act="about">How it works</a></p>
  </div>`;
}

function localPanel() {
  return `<div class="panel-body">
    <div class="panel-title">${icon("people")}<h2>Pass and Play</h2></div>
    <label class="field">White<input id="local-white" maxlength="24" value="${esc(store.get("localWhite", "Player 1"))}"></label>
    <label class="field">Black<input id="local-black" maxlength="24" value="${esc(store.get("localBlack", "Player 2"))}"></label>
    ${tcPicker()}
    <button class="btn-green big" data-act="local-start">Start Game</button>
    <a class="link" data-act="back">← Back</a>
  </div>`;
}

function gamePanel(g) {
  const ply = S.optimistic ? g.ucis.length : S.viewPly;
  const sans = S.optimistic ? [...g.sans, S.optimistic.move.san] : g.sans;
  const rows = [];
  for (let i = 0; i < sans.length; i += 2) {
    const cell = (j) => j < sans.length ? `<span class="san ${j + 1 === ply ? "current" : ""}" data-act="ply" data-ply="${j + 1}">${sanHtml(sans[j])}</span>` : "<span></span>";
    rows.push(`<div class="move-row"><span class="num">${i / 2 + 1}.</span>${cell(i)}${cell(i + 1)}</div>`);
  }
  if (g.result) {
    const head = g.result === "*" ? "Game aborted" : g.result === "1/2-1/2" ? "½-½ · Draw" : `${g.result} · ${g.winner === "white" ? "White" : "Black"} won`;
    rows.push(`<div class="result-line">${head}<small>${esc(REASONS[g.reason] || g.reason)}</small></div>`);
  }

  const est = (color) => {
    const e = estimateAt(color, ply);
    return `<div class="est-row"><span class="swatch" style="background:var(--${color}-line)"></span>
      <span class="who">${esc(g.players[color].name)}</span>
      <span class="val">${e ? e.rating : "—"}</span><span class="rng">${e ? `${e.low}–${e.high}` : ""}</span></div>`;
  };
  const estimate = g.model
    ? `<section class="estimate"><div class="estimate-head"><h3>Live rating estimate</h3><span>${plyLabel(Math.min(ply, g.estimates.length - 1))}</span></div>
        ${est("white")}${est("black")}<div class="chart" id="chart"></div></section>`
    : `<section class="estimate"><div class="no-model">No model loaded: run train.py and export_onnx.py to see rating estimates.</div></section>`;

  const notices = [];
  const mine = g.you.length === 1 ? g.you[0] : null;
  if (g.mode === "friend" && !g.started) {
    const link = `${location.origin}/game/${g.id}`;
    notices.push(`<div class="notice"><span class="grow">Invite a friend to play</span>
      <div class="invite"><input readonly value="${esc(link)}" id="invite"><button class="btn" data-act="copy">Copy</button></div></div>`);
  }
  if (g.draw_offer && !g.result && mine) {
    notices.push(g.draw_offer === mine
      ? `<div class="notice"><span class="grow">Draw offer sent</span></div>`
      : `<div class="notice"><span class="grow">${esc(g.players[g.draw_offer].name)} offers a draw</span>
         <button class="icon-btn yes" data-act="draw" title="Accept">✓</button><button class="icon-btn" data-act="decline" title="Decline">✕</button></div>`);
  }
  if (g.result && mine && g.rematch.length && g.mode !== "local") {
    notices.push(g.rematch.includes(mine)
      ? `<div class="notice"><span class="grow">Rematch offer sent</span></div>`
      : `<div class="notice"><span class="grow">${esc(g.players[g.rematch[0]].name)} wants a rematch</span><button class="icon-btn yes" data-act="rematch" title="Accept">✓</button></div>`);
  }

  const latest = g.ucis.length;
  const navs = `<div class="nav-controls">
    <button class="btn" data-act="first" ${ply === 0 ? "disabled" : ""}>${icon("first")}</button>
    <button class="btn" data-act="prev" ${ply === 0 ? "disabled" : ""}>${icon("prev")}</button>
    <button class="btn" data-act="next" ${ply >= latest ? "disabled" : ""}>${icon("next")}</button>
    <button class="btn" data-act="last" ${ply >= latest ? "disabled" : ""}>${icon("last")}</button></div>`;

  let actions = "";
  if (g.you.length && !g.result) {
    actions = g.ucis.length < 2
      ? `<button class="btn" data-act="abort">${icon("close")}Abort</button>`
      : `<button class="btn" data-act="draw" ${g.draw_offer === mine && mine ? "disabled" : ""}>½ Draw</button>
         <button class="btn" data-act="resign">${icon("flag")}${S.confirmResign ? "Confirm resign?" : "Resign"}</button>`;
  } else if (g.result) {
    actions = `<button class="btn" data-act="new-same">${icon("plus")}New ${TIME_CONTROLS[g.tc]}</button>`
      + (g.you.length && g.result !== "*" ? `<button class="btn" data-act="rematch" ${mine && g.rematch.includes(mine) ? "disabled" : ""}>${icon("rematch")}Rematch</button>` : "");
  }
  return `<div class="panel-body">${estimate}<div class="moves" id="moves">${rows.join("")}</div>
    ${notices.join("")}${navs}<div class="actions">${actions}</div></div>`;
}

function showGameOver() {
  const g = S.game, el = $("#game-over");
  const mine = g.you.length === 1 ? g.you[0] : null;
  let title, mood = "draw";
  if (g.result === "*") title = g.reason === "aborted" ? "Game Aborted" : "Game Abandoned";
  else if (g.result === "1/2-1/2") title = "Draw";
  else if (mine) { title = mine === g.winner ? "You Won!" : "You Lost"; mood = mine === g.winner ? "win" : "loss"; }
  else { title = g.winner === "white" ? "White Won" : "Black Won"; mood = "win"; }
  const last = g.estimates.length - 1;
  const player = (color) => {
    const e = last >= 0 ? g.estimates[last][color] : null;
    return `<div class="modal-player ${g.winner === color ? "won" : ""}">${avatar(color)}
      <span class="who">${esc(g.players[color].name)}</span>
      ${e ? `<span class="est">${e.rating}</span><span class="range">${e.low}–${e.high}</span>` : ""}</div>`;
  };
  const rematch = g.you.length && g.result !== "*"
    ? `<button class="btn" data-act="rematch">${icon("rematch")}Rematch</button>` : `<button class="btn" data-act="home">Home</button>`;
  el.innerHTML = `<div class="modal-head ${mood}"><button class="modal-close" data-act="close-modal" aria-label="Close">✕</button>
      <svg class="trophy" viewBox="0 0 24 24"><path d="M7 3h10v2h4v3a5 5 0 0 1-4.6 5A6 6 0 0 1 13 16.9V19h3v2H8v-2h3v-2.1A6 6 0 0 1 7.6 13 5 5 0 0 1 3 8V5h4zm0 4H5v1a3 3 0 0 0 2 2.8zm10 0v3.8A3 3 0 0 0 19 8V7z"/></svg>
      <h2>${title}</h2><p>${esc(g.result === "*" ? REASONS[g.reason] : REASONS[g.reason] || g.reason)}</p></div>
    <div class="modal-players">${player("white")}<span class="modal-vs">vs</span>${player("black")}</div>
    ${g.model && last > 0 ? `<div class="modal-caption">Estimated Lichess blitz ratings after ${Math.ceil(last / 2)} moves (80% range)</div>` : ""}
    <div class="modal-actions"><button class="btn-green" data-act="new-same">New ${TIME_CONTROLS[g.tc]}</button>
      ${rematch}<button class="btn" data-act="close-modal">${icon("chart")}Review</button></div>`;
  el.hidden = false;
}

// ---------- actions ----------

function setPly(ply) {
  const g = S.game;
  if (!g) return;
  S.viewPly = Math.max(0, Math.min(ply, g.ucis.length));
  render();
}

function startNew(tc) {
  const g = S.game;
  S.tc = tc;
  store.set("tc", tc);
  $("#game-over").hidden = true;
  if (g && g.mode === "local") return send({ type: "local", tc, names: [g.players.white.name, g.players.black.name] });
  if (g && g.mode === "friend") return send({ type: "friend", tc });
  history.pushState(null, "", "/");
  S.roomId = null;
  S.game = null;
  S.view = "seeking";
  send({ type: "seek", tc });
  render();
}

const ACTIONS = {
  home: (e) => { e.preventDefault(); if (S.view === "seeking") send({ type: "cancel_seek" }); S.view = "home"; go("/"); },
  about: () => { $("#about").hidden = false; },
  "close-about": () => { $("#about").hidden = true; },
  flip: () => { S.flipped = !S.flipped; render(); },
  sound: () => { toggleSound(); renderSoundButton(); },
  tab: (e, t) => { S.tab = t.dataset.tab; render(); },
  "tc-toggle": () => { S.tcOpen = !S.tcOpen; renderPanel(); },
  tc: (e, t) => {
    S.tc = t.dataset.tc;
    store.set("tc", S.tc);
    S.tcOpen = false;
    renderPanel();
    if (!S.game) { renderBars(); updateClocks(); }
  },
  seek: () => { S.view = "seeking"; send({ type: "seek", tc: S.tc }); render(); },
  cancel: () => { send({ type: "cancel_seek" }); S.view = "home"; render(); },
  friend: () => send({ type: "friend", tc: S.tc }),
  "local-setup": () => { S.view = "local"; S.game = null; S.roomId = null; history.pushState(null, "", "/"); render(); },
  "local-start": () => {
    const w = $("#local-white").value.trim() || "Player 1", b = $("#local-black").value.trim() || "Player 2";
    store.set("localWhite", w);
    store.set("localBlack", b);
    send({ type: "local", tc: S.tc, names: [w, b] });
  },
  back: () => { S.view = "home"; render(); },
  ply: (e, t) => setPly(+t.dataset.ply),
  first: () => setPly(0),
  prev: () => setPly(S.viewPly - 1),
  next: () => setPly(S.viewPly + 1),
  last: () => setPly(S.game ? S.game.ucis.length : 0),
  abort: () => send({ type: "abort", id: S.roomId }),
  draw: () => send({ type: "draw", id: S.roomId }),
  decline: () => send({ type: "decline_draw", id: S.roomId }),
  resign: () => {
    if (S.game?.mode === "local" || S.confirmResign) { S.confirmResign = false; send({ type: "resign", id: S.roomId }); return; }
    S.confirmResign = true;
    renderPanel();
    setTimeout(() => { if (S.confirmResign) { S.confirmResign = false; renderPanel(); } }, 3000);
  },
  rematch: () => send({ type: "rematch", id: S.roomId }),
  "new-same": () => startNew(S.game ? S.game.tc : S.tc),
  "close-modal": () => { $("#game-over").hidden = true; },
  copy: () => {
    const input = $("#invite");
    (navigator.clipboard ? navigator.clipboard.writeText(input.value) : Promise.reject()).then(
      () => toast("Invite link copied"), () => { input.select(); document.execCommand("copy"); toast("Invite link copied"); });
  },
};

document.addEventListener("click", (e) => {
  const t = e.target.closest("[data-act]");
  if (!t || t.disabled) return;
  const act = ACTIONS[t.dataset.act];
  if (act) act(e, t);
});

document.addEventListener("change", (e) => {
  if (e.target.id === "name-input") {
    S.name = e.target.value.trim().slice(0, 24) || S.name;
    store.set("name", S.name);
    send({ type: "hello", token: S.token, name: S.name });
    if (!S.game) renderBars();
  }
});

document.addEventListener("keydown", (e) => {
  if (e.target.closest("input")) return;
  if (e.key === "ArrowLeft") setPly(S.viewPly - 1);
  else if (e.key === "ArrowRight") setPly(S.viewPly + 1);
  else if (e.key === "Escape") { $("#about").hidden = true; $("#game-over").hidden = true; }
});

$("#about").addEventListener("click", (e) => { if (e.target.id === "about") e.target.hidden = true; });

function toast(text) {
  const el = document.createElement("div");
  el.className = "toast";
  el.textContent = text;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 2200);
}

function renderSoundButton() {
  $("#sound-btn").innerHTML = `${icon(soundOn() ? "sound" : "mute")}<span>Sound ${soundOn() ? "on" : "off"}</span>`;
}

// ---------- layout ----------

function layout() {
  const narrow = innerWidth <= 900;
  const root = document.documentElement.style;
  const bar = 52;
  let size;
  if (narrow) size = Math.min(innerWidth - 32, innerHeight - 2 * bar - 80);
  else size = Math.min(innerHeight - 2 * bar - 32, innerWidth - 145 - 420 - 60);
  size = Math.max(240, Math.floor(size / 8) * 8);
  root.setProperty("--board", `${size}px`);
  if (S.game && S.tab === "game" && S.view === "game") drawChart($("#chart"), S.game.estimates, S.viewPly, setPly);
}

window.addEventListener("resize", layout);

for (const el of document.querySelectorAll("[data-icon]")) el.outerHTML = icon(el.dataset.icon);
renderSoundButton();
layout();
go(location.pathname, false);
connect();
requestAnimationFrame(tick);

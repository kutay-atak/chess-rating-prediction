// Chessboard view: renders a FEN, lets the player drag or click pieces along the legal moves it is
// given, and reports moves through onMove(from, to, promotion). Right-click marks squares and
// right-drag draws arrows. It never changes the position itself; the caller sets the next one.
const FILES = "abcdefgh";
const PIECE_URL = (p) => `/static/pieces/${p}.svg`;

function parseFen(fen) {
  const pieces = new Map();
  fen.split(" ")[0].split("/").forEach((row, r) => {
    let f = 0;
    for (const ch of row) {
      if (/\d/.test(ch)) { f += +ch; continue; }
      const color = ch === ch.toUpperCase() ? "w" : "b";
      pieces.set(FILES[f] + (8 - r), color + ch.toUpperCase());
      f++;
    }
  });
  return pieces;
}

export class Board {
  constructor(el, { onMove }) {
    this.el = el;
    this.onMove = onMove;
    this.orientation = "white";
    this.pieces = new Map(); // square -> element
    this.fen = null;
    this.lastMove = null;
    this.check = null;
    this.movable = null; // "w", "b" or null
    this.dests = new Map(); // from -> [{to, promotion}]
    this.selected = null;
    this.marks = new Set();
    this.arrows = [];
    el.innerHTML = `<div class="layer cells"></div><div class="layer coords"></div><div class="layer pieces"></div>
      <svg class="arrows" viewBox="0 0 8 8"><defs><marker id="ah" markerWidth="3" markerHeight="3" refX="1.4" refY="1.5" orient="auto"><path d="M0,0 L3,1.5 L0,3 z" fill="rgba(255,170,0,.85)"/></marker></defs><g></g></svg>`;
    this.cellLayer = el.querySelector(".cells");
    this.coordLayer = el.querySelector(".coords");
    this.pieceLayer = el.querySelector(".pieces");
    this.arrowLayer = el.querySelector("svg.arrows g");
    el.addEventListener("pointerdown", (e) => this.down(e));
    el.addEventListener("pointermove", (e) => this.move(e));
    el.addEventListener("pointerup", (e) => this.up(e));
    el.addEventListener("pointercancel", () => this.cancelDrag());
    el.addEventListener("contextmenu", (e) => e.preventDefault());
    this.drawCoords();
  }

  setOrientation(color) {
    if (color === this.orientation) return;
    this.orientation = color;
    this.drawCoords();
    for (const [sq, el] of this.pieces) this.place(el, sq);
    this.drawCells();
    this.drawArrows();
  }

  // fen, lastMove [from, to], check square, movable colour ("w"/"b"/null), dests Map
  set({ fen, lastMove = null, check = null, movable = null, dests = new Map() }) {
    if (fen !== this.fen) {
      this.diffPieces(parseFen(fen));
      this.fen = fen;
      this.selected = null;
    }
    this.lastMove = lastMove;
    this.check = check;
    this.movable = movable;
    this.dests = dests;
    if (this.selected && !this.dests.has(this.selected)) this.selected = null;
    this.closePromotion();
    this.drawCells();
  }

  xy(sq) {
    const f = FILES.indexOf(sq[0]), r = +sq[1] - 1;
    return this.orientation === "white" ? [f, 7 - r] : [7 - f, r];
  }

  squareAt(e) {
    const rect = this.el.getBoundingClientRect();
    const x = Math.floor(((e.clientX - rect.left) / rect.width) * 8);
    const y = Math.floor(((e.clientY - rect.top) / rect.height) * 8);
    if (x < 0 || x > 7 || y < 0 || y > 7) return null;
    return this.orientation === "white" ? FILES[x] + (8 - y) : FILES[7 - x] + (y + 1);
  }

  place(el, sq) {
    const [x, y] = this.xy(sq);
    el.style.transform = `translate(${x * 100}%, ${y * 100}%)`;
  }

  diffPieces(next) {
    const old = new Map(this.pieces);
    const kept = new Map();
    for (const [sq, p] of next) {
      const el = old.get(sq);
      if (el && el.dataset.piece === p) { kept.set(sq, el); old.delete(sq); }
    }
    for (const [sq, p] of next) {
      if (kept.has(sq)) continue;
      let best = null, bestDist = Infinity;
      for (const [osq, el] of old) {
        if (el.dataset.piece !== p) continue;
        const [x1, y1] = this.xy(osq), [x2, y2] = this.xy(sq);
        const d = Math.hypot(x1 - x2, y1 - y2);
        if (d < bestDist) { best = osq; bestDist = d; }
      }
      let el;
      if (best) { el = old.get(best); old.delete(best); }
      else {
        el = document.createElement("div");
        el.className = "piece";
        el.dataset.piece = p;
        el.style.backgroundImage = `url(${PIECE_URL(p)})`;
        el.style.transition = "none";
        this.pieceLayer.appendChild(el);
        this.place(el, sq);
        el.offsetWidth; // commit position before re-enabling transitions
        el.style.transition = "";
      }
      el.classList.remove("dragging");
      this.place(el, sq);
      kept.set(sq, el);
    }
    for (const el of old.values()) {
      el.classList.add("fading");
      setTimeout(() => el.remove(), 160);
    }
    this.pieces = kept;
  }

  drawCoords() {
    let html = "";
    for (let i = 0; i < 8; i++) {
      const rank = this.orientation === "white" ? 8 - i : i + 1;
      const file = this.orientation === "white" ? FILES[i] : FILES[7 - i];
      const rankLight = i % 2 === 0; // left column: top-left square is light
      const fileLight = i % 2 === 1; // bottom row: bottom-left square is dark
      html += `<div class="cell" style="transform:translate(0,${i * 100}%)"><span class="coord rank ${rankLight ? "on-light" : "on-dark"}">${rank}</span></div>`;
      html += `<div class="cell" style="transform:translate(${i * 100}%,700%)"><span class="coord file ${fileLight ? "on-light" : "on-dark"}">${file}</span></div>`;
    }
    this.coordLayer.innerHTML = html;
  }

  drawCells() {
    const cls = new Map();
    const add = (sq, c) => cls.set(sq, (cls.get(sq) || "") + " " + c);
    if (this.lastMove) this.lastMove.forEach((sq) => add(sq, "last"));
    for (const sq of this.marks) add(sq, "mark");
    if (this.check) add(this.check, "check");
    if (this.selected) {
      add(this.selected, "selected");
      for (const m of this.dests.get(this.selected) || []) add(m.to, this.pieces.has(m.to) ? "dest capture" : "dest");
    }
    if (this.hover) add(this.hover, "hover");
    let html = "";
    for (const [sq, c] of cls) {
      const [x, y] = this.xy(sq);
      html += `<div class="cell${c}" style="transform:translate(${x * 100}%,${y * 100}%)"></div>`;
    }
    this.cellLayer.innerHTML = html;
  }

  canMove(sq) {
    return this.movable && this.dests.has(sq);
  }

  isDest(from, to) {
    return (this.dests.get(from) || []).some((m) => m.to === to);
  }

  down(e) {
    if (this.promo) return;
    const sq = this.squareAt(e);
    if (!sq) return;
    if (e.button === 2) { this.rightStart = sq; return; }
    if (e.button !== 0) return;
    if (this.marks.size || this.arrows.length) { this.marks.clear(); this.arrows = []; this.drawArrows(); }
    if (this.selected && this.isDest(this.selected, sq)) {
      this.tryMove(this.selected, sq);
      return;
    }
    if (!this.canMove(sq)) {
      this.selected = null;
      this.drawCells();
      return;
    }
    const wasSelected = this.selected === sq;
    this.selected = sq;
    const el = this.pieces.get(sq);
    this.drag = { from: sq, el, wasSelected, moved: false, x: e.clientX, y: e.clientY };
    this.el.setPointerCapture(e.pointerId);
    this.drawCells();
  }

  move(e) {
    if (!this.drag) return;
    const d = this.drag;
    if (!d.moved && Math.hypot(e.clientX - d.x, e.clientY - d.y) < 4) return;
    d.moved = true;
    const rect = this.el.getBoundingClientRect();
    const size = rect.width / 8;
    d.el.classList.add("dragging");
    d.el.style.transform = `translate(${e.clientX - rect.left - size / 2}px, ${e.clientY - rect.top - size / 2}px)`;
    const over = this.squareAt(e);
    if (over !== this.hover) { this.hover = over; this.drawCells(); }
  }

  up(e) {
    if (e.button === 2 && this.rightStart) {
      const to = this.squareAt(e);
      const from = this.rightStart;
      this.rightStart = null;
      if (!to) return;
      if (to === from) { this.marks.has(to) ? this.marks.delete(to) : this.marks.add(to); this.drawCells(); }
      else {
        const i = this.arrows.findIndex((a) => a[0] === from && a[1] === to);
        i >= 0 ? this.arrows.splice(i, 1) : this.arrows.push([from, to]);
        this.drawArrows();
      }
      return;
    }
    if (!this.drag) return;
    const d = this.drag;
    this.drag = null;
    this.hover = null;
    const to = this.squareAt(e);
    if (!d.moved) {
      if (d.wasSelected) this.selected = null; // second click on a selected piece deselects it
      this.drawCells();
      return;
    }
    if (to && to !== d.from && this.isDest(d.from, to)) {
      this.tryMove(d.from, to);
    } else {
      d.el.classList.remove("dragging");
      this.place(d.el, d.from);
      this.drawCells();
    }
  }

  cancelDrag() {
    if (!this.drag) return;
    this.drag.el.classList.remove("dragging");
    this.place(this.drag.el, this.drag.from);
    this.drag = null;
    this.hover = null;
    this.drawCells();
  }

  tryMove(from, to) {
    const options = (this.dests.get(from) || []).filter((m) => m.to === to);
    if (options.some((m) => m.promotion)) return this.openPromotion(from, to);
    this.selected = null;
    this.onMove(from, to);
  }

  openPromotion(from, to) {
    const color = this.pieces.get(from)?.dataset.piece[0] || "w";
    const [x, y] = this.xy(to);
    const up = y !== 0; // choices grow away from the edge the pawn reached
    const wrap = document.createElement("div");
    wrap.className = "promotion";
    const choices = ["Q", "N", "R", "B"].map((p) => `<button data-p="${p.toLowerCase()}"><img src="${PIECE_URL(color + p)}" alt="${p}"></button>`).join("");
    wrap.innerHTML = `<div class="choices ${up ? "up" : ""}" style="left:${x * 12.5}%;${up ? `bottom:${(7 - y) * 12.5}%` : `top:0`}">${choices}<button class="close">✕</button></div>`;
    wrap.addEventListener("pointerdown", (e) => {
      e.stopPropagation();
      const b = e.target.closest("button");
      this.closePromotion();
      if (b && b.dataset.p) { this.selected = null; this.onMove(from, to, b.dataset.p); }
      else { const el = this.pieces.get(from); if (el) { el.classList.remove("dragging"); this.place(el, from); } this.drawCells(); }
    });
    this.el.appendChild(wrap);
    this.promo = wrap;
  }

  closePromotion() {
    if (this.promo) { this.promo.remove(); this.promo = null; }
  }

  drawArrows() {
    const center = (sq) => { const [x, y] = this.xy(sq); return [x + 0.5, y + 0.5]; };
    this.arrowLayer.innerHTML = this.arrows.map(([from, to]) => {
      const [x1, y1] = center(from), [x2, y2] = center(to);
      const dx = x2 - x1, dy = y2 - y1;
      const shorten = (ax, ay, bx, by) => { const l = Math.hypot(bx - ax, by - ay); return [bx - ((bx - ax) / l) * 0.35, by - ((by - ay) / l) * 0.35]; };
      let points;
      if (Math.abs(dx) * Math.abs(dy) === 2) { // knight move: L-shaped like an actual knight path
        const corner = Math.abs(dx) > Math.abs(dy) ? [x2, y1] : [x1, y2];
        points = [[x1, y1], corner, shorten(...corner, x2, y2)];
      } else points = [[x1, y1], shorten(x1, y1, x2, y2)];
      return `<polyline points="${points.map((p) => p.join(",")).join(" ")}" fill="none" stroke="rgba(255,170,0,.85)" stroke-width=".2" stroke-linejoin="round" marker-end="url(#ah)"/>`;
    }).join("");
  }
}

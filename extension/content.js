// Draws the control map on chess.com's board. Runs only on bot games and analysis pages.
(() => {
  const ALLOWED = /^\/(play\/(computer|bots?)(\/|$)|analysis|game\/computer\/)/;
  const COLORS = { white: "#2f7ae0", black: "#e0352b", equal: "#9a5fd0" };
  const WIDTH = [0.05, 0.09, 0.14];                   // frame width per majority of 1, 2, 3+, as a share of a square
  let on = localStorage.getItem("scm") !== "off", last = "";

  const button = document.createElement("button");
  button.style.cssText = "position:fixed;left:16px;bottom:16px;z-index:2147483647;padding:9px 13px;border-radius:999px;" +
    "border:0;font:600 13px system-ui;color:#fff;cursor:pointer;box-shadow:0 2px 8px #0006";
  button.title = "Blue: White controls the square · Red: Black · Purple: equal (Alt+C)";
  button.onclick = () => { on = !on; localStorage.setItem("scm", on ? "on" : "off"); last = ""; };
  document.addEventListener("keydown", (e) => e.altKey && e.code === "KeyC" && button.click());
  document.body.append(button);

  function readFen(el) {
    const fen = el?.game?.getFEN?.();
    if (!fen) return null;
    return fen.split(" ")[0].split("/").reverse().flatMap((row) =>
      [...row].flatMap((c) => (/\d/.test(c) ? Array(+c).fill(null) : [c])));
  }

  function overlay(el) {
    let ov = el.querySelector(":scope > .scm");
    if (!ov) {
      ov = document.createElement("div");
      ov.className = "scm";
      ov.style.cssText = "position:absolute;inset:0;pointer-events:none;z-index:5";
      ov.innerHTML = '<div style="position:absolute;width:12.5%;height:12.5%"></div>'.repeat(64);
      el.append(ov);
    }
    return ov;
  }

  function frame([w, b], size) {
    if (!w && !b) return "none";
    const level = w === b ? 0 : Math.min(Math.abs(w - b), 3) - 1;
    const color = w === b ? COLORS.equal : w > b ? COLORS.white : COLORS.black;
    return `inset 0 0 0 ${Math.max(2, Math.round(WIDTH[level] * size))}px ${color}`;
  }

  function render() {
    const allowed = ALLOWED.test(location.pathname);
    const el = document.querySelector("wc-chess-board");
    const board = allowed && on ? readFen(el) : null;
    button.hidden = !allowed;
    button.textContent = `Control map: ${on ? "on" : "off"}`;
    button.style.background = on ? "#2f7ae0" : "#262421";
    document.querySelectorAll(".scm").forEach((ov) => (ov.hidden = !board));
    if (!board) return;
    const flipped = el.classList.contains("flipped") || !!el.game.getOptions?.().flipped;
    const size = el.getBoundingClientRect().width / 8;
    const key = flipped + board.join() + Math.round(size);
    const ov = overlay(el);
    if (key === last) return;
    last = key;
    const control = SquareControl.control(board);
    [...ov.children].forEach((cell, sq) => {
      const f = sq % 8, r = sq >> 3;
      cell.style.left = (flipped ? 7 - f : f) * 12.5 + "%";
      cell.style.top = (flipped ? r : 7 - r) * 12.5 + "%";
      cell.style.boxShadow = frame(control[sq], size);
      cell.title = `White ${control[sq][0]} · Black ${control[sq][1]}`;
    });
  }

  setInterval(render, 300);                           // the board is a canvas, so poll instead of watching elements
})();

// How many pieces of each side could legally capture on each square.
// board: 64 entries (a1 = 0, h8 = 63), each null or "P", "n", ... (upper case = White).
const SquareControl = (() => {
  const ROOK = [[1, 0], [-1, 0], [0, 1], [0, -1]];
  const BISHOP = [[1, 1], [1, -1], [-1, 1], [-1, -1]];
  const KNIGHT = [[1, 2], [2, 1], [2, -1], [1, -2], [-1, -2], [-2, -1], [-2, 1], [-1, 2]];
  const KING = [...ROOK, ...BISHOP];
  const inside = (f, r) => f >= 0 && f < 8 && r >= 0 && r < 8;
  const isWhite = (p) => p === p.toUpperCase();
  const mine = (p, white, types) => p && isWhite(p) === white && types.includes(p.toLowerCase());

  function attackers(board, white, sq) {
    const f = sq % 8, r = sq >> 3, out = [];
    const step = (df, dr, types) => {
      const i = (r + dr) * 8 + f + df;
      if (inside(f + df, r + dr) && mine(board[i], white, types)) out.push(i);
    };
    const pawnRank = white ? -1 : 1;
    step(-1, pawnRank, "p"); step(1, pawnRank, "p");
    KNIGHT.forEach(([a, b]) => step(a, b, "n"));
    KING.forEach(([a, b]) => step(a, b, "k"));
    for (const [dirs, types] of [[ROOK, "rq"], [BISHOP, "bq"]]) {
      for (const [a, b] of dirs) {
        for (let ff = f + a, rr = r + b; inside(ff, rr); ff += a, rr += b) {
          const p = board[rr * 8 + ff];
          if (p) { if (mine(p, white, types)) out.push(rr * 8 + ff); break; }
        }
      }
    }
    return out;
  }

  // Squares a piece pinned to its own king may still reach, or null if it isn't pinned.
  function pinLine(board, sq) {
    const white = isWhite(board[sq]);
    const k = board.findIndex((p) => p === (white ? "K" : "k"));
    const df = (sq % 8) - (k % 8), dr = (sq >> 3) - (k >> 3);
    const straight = !df || !dr;
    if (k < 0 || (!straight && Math.abs(df) !== Math.abs(dr))) return null;
    const line = new Set([k]);
    let passed = false;
    for (let f = k % 8 + Math.sign(df), r = (k >> 3) + Math.sign(dr); inside(f, r); f += Math.sign(df), r += Math.sign(dr)) {
      const i = r * 8 + f, p = board[i];
      line.add(i);
      if (!p) continue;
      if (!passed) { if (i !== sq) return null; passed = true; continue; }
      return isWhite(p) !== white && (straight ? "rq" : "bq").includes(p.toLowerCase()) ? line : null;
    }
    return null;
  }

  function count(board, white, sq) {
    return attackers(board, white, sq).filter((a) => {
      if (board[a].toLowerCase() === "k") {           // a king can't step onto a covered square
        const without = board.slice(); without[a] = null;
        return !attackers(without, !white, sq).length;
      }
      const line = pinLine(board, a);                 // a pinned piece stays on the pin line
      return !line || line.has(sq);
    }).length;
  }

  const control = (board) => board.map((_, sq) => [count(board, true, sq), count(board, false, sq)]);
  return { control };
})();
if (typeof module === "object" && module && module.exports) module.exports = SquareControl;

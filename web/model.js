// The GRU of gru.py, in plain JS: per-ply inputs, one recurrent step per ply, rating bands.
// Shared by index.html and test_web.py (run in Node), so the page uses exactly the tested code.
(function (root) {
  const CHECKPOINTS = [2, 5, 10, 20, 30, 40, 60, 80];

  function sigmoid(v) { return 1 / (1 + Math.exp(-v)); }

  // Inputs for ply i (0-based) given every clock so far: what gru.ply_inputs computes for row i.
  // Seconds spent = mover's previous clock - clock + increment; first own move base - clock;
  // berserk (first own clock = base/2) spends 0 then gets no increment.
  function plyInput(clocks, i, base, inc) {
    const side = i % 2;
    const first = clocks[side];
    const berserk = base > 0 && first === base / 2;
    let spent;
    if (i === side) spent = berserk ? 0 : base - clocks[i];
    else spent = clocks[i - 2] - clocks[i] + (berserk ? 0 : inc);
    spent = Math.max(spent, 0);
    const share = Math.min(Math.max(clocks[i] / (base > 0 ? base : 40 * inc), 0), 2);
    return [Math.log1p(spent), share, side, base / 180, inc / 5];
  }

  function makeModel(json) {
    const W = {};
    for (const [k, v] of Object.entries(json.weights)) W[k] = { shape: v.shape, data: Float64Array.from(v.data) };
    const index = new Map(json.vocab.map((m, i) => [m, i + 2]));
    const E = W["emb.weight"], H = W["gru.weight_hh_l0"].shape[1], IN = W["gru.weight_ih_l0"].shape[1];
    const Wih = W["gru.weight_ih_l0"].data, Whh = W["gru.weight_hh_l0"].data;
    const bih = W["gru.bias_ih_l0"].data, bhh = W["gru.bias_hh_l0"].data;
    const Wo = W["head.weight"].data, bo = W["head.bias"].data;
    const q = Object.fromEntries(Object.entries(json.conformal).map(([k, v]) => [Number(k), v]));

    function matvec(M, rowStart, cols, x) {
      let s = 0;
      for (let j = 0; j < cols; j++) s += M[rowStart + j] * x[j];
      return s;
    }

    // One GRU step (PyTorch gate order r, z, n). Returns the new hidden state.
    function step(h, san, inputs) {
      const id = index.get(san.replace(/[?!]+$/, "")) ?? 1;  // 1 = unknown move
      const emb = E.data.subarray(id * E.shape[1], (id + 1) * E.shape[1]);
      const x = new Float64Array(IN);
      x.set(emb); x.set(inputs, E.shape[1]);
      const out = new Float64Array(H);
      for (let u = 0; u < H; u++) {
        const r = sigmoid(matvec(Wih, u * IN, IN, x) + bih[u] + matvec(Whh, u * H, H, h) + bhh[u]);
        const z = sigmoid(matvec(Wih, (H + u) * IN, IN, x) + bih[H + u] + matvec(Whh, (H + u) * H, H, h) + bhh[H + u]);
        const n = Math.tanh(matvec(Wih, (2 * H + u) * IN, IN, x) + bih[2 * H + u] +
                            r * (matvec(Whh, (2 * H + u) * H, H, h) + bhh[2 * H + u]));
        out[u] = (1 - z) * n + z * h[u];
      }
      return out;
    }

    // [[lo, mean, hi] for White, [lo, mean, hi] for Black] after `ply` plies, from hidden state h.
    function band(h, ply) {
      const o = [0, 1, 2, 3].map(k => matvec(Wo, k * H, H, h) + bo[k]);
      let c = CHECKPOINTS[0];
      for (const t of CHECKPOINTS) if (t <= ply) c = t;
      return [[o[0], o[1]], [o[2], o[3]]].map(([mu, lv]) => {
        const m = mu * json.rating_sd + json.rating_mean;
        const s = Math.exp(Math.min(Math.max(lv, -7), 3) / 2) * json.rating_sd;
        return [m - q[c] * s, m, m + q[c] * s];
      });
    }

    // Bands after every ply of a whole game.
    function run(sans, clocks, base, inc) {
      let h = new Float64Array(H);
      return sans.map((san, i) => { h = step(h, san, plyInput(clocks, i, base, inc)); return band(h, i + 1); });
    }

    return { step, band, run, hidden: () => new Float64Array(H) };
  }

  // SAN moves and clocks (seconds) from a PGN whose moves carry [%clk h:mm:ss] comments.
  function parsePgn(text) {
    const headers = {};
    for (const m of text.matchAll(/\[(\w+)\s+"([^"]*)"\]/g)) headers[m[1]] = m[2];
    const body = text.replace(/\[(\w+)\s+"([^"]*)"\]/g, " ");
    const sans = [], clocks = [];
    for (const m of body.matchAll(/([^\s{}]+)\s*\{[^}]*?\[%clk (\d+):(\d+):([\d.]+)\][^}]*\}/g)) {
      sans.push(m[1].replace(/^\d+\.+/, "").replace(/[?!]+$/, ""));
      clocks.push(Math.round(Number(m[2]) * 3600 + Number(m[3]) * 60 + Number(m[4])));
    }
    return { headers, sans, clocks };
  }

  const api = { CHECKPOINTS, plyInput, makeModel, parsePgn };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.RatingModel = api;
})(typeof globalThis !== "undefined" ? globalThis : this);

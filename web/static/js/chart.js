// Line chart of both players' estimated ratings after every half-move, with 80% interval bands.
const L = 34, R = 6, T = 8, B = 16;

function niceStep(range) {
  return [50, 100, 200, 250, 500].find((s) => range / s <= 5) || 1000;
}

export function drawChart(el, estimates, viewPly, onPick) {
  if (!estimates.length) { el.innerHTML = ""; return; }
  const W = el.clientWidth || 380, H = el.clientHeight || 150;
  const n = Math.max(estimates.length - 1, 20);
  const lows = estimates.flatMap((e) => [e.white.low, e.black.low]);
  const highs = estimates.flatMap((e) => [e.white.high, e.black.high]);
  const step0 = niceStep(Math.max(...highs) - Math.min(...lows));
  const lo = Math.floor(Math.min(...lows) / step0) * step0;
  const hi = Math.ceil(Math.max(...highs) / step0) * step0;
  const step = niceStep(hi - lo);
  const x = (i) => L + (i / n) * (W - L - R);
  const y = (v) => T + (1 - (v - lo) / (hi - lo || 1)) * (H - T - B);

  let grid = "";
  for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) {
    grid += `<line x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}"/><text x="${L - 4}" y="${y(v) + 3}" text-anchor="end">${v}</text>`;
  }
  for (let m = 10; m <= n / 2; m += m >= 40 ? 20 : 10) {
    grid += `<text x="${x(2 * m)}" y="${H - 3}" text-anchor="middle">${m}</text>`;
  }
  const series = (side, color, fill) => {
    const band = estimates.map((e, i) => `${x(i)},${y(e[side].high)}`).concat(
      estimates.map((e, i) => `${x(i)},${y(e[side].low)}`).reverse()).join(" ");
    const line = estimates.map((e, i) => `${x(i)},${y(e[side].rating)}`).join(" ");
    return `<polygon points="${band}" fill="${fill}"/><polyline points="${line}" fill="none" stroke="${color}" stroke-width="2" stroke-linejoin="round"/>`;
  };
  const css = getComputedStyle(document.documentElement);
  const wc = css.getPropertyValue("--white-line").trim(), bc = css.getPropertyValue("--black-line").trim();
  const v = Math.min(viewPly, estimates.length - 1);
  const marker = `<line x1="${x(v)}" x2="${x(v)}" y1="${T}" y2="${H - B}" stroke="#6f6d6a" stroke-dasharray="3 3"/>
    <circle cx="${x(v)}" cy="${y(estimates[v].white.rating)}" r="3.5" fill="${wc}"/>
    <circle cx="${x(v)}" cy="${y(estimates[v].black.rating)}" r="3.5" fill="${bc}"/>`;
  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}"><g class="grid">${grid}</g>
    ${series("white", wc, "rgba(242,241,239,.12)")}${series("black", bc, "rgba(155,195,99,.18)")}${marker}</svg><div class="tip" hidden></div>`;

  const svg = el.querySelector("svg"), tip = el.querySelector(".tip");
  const plyAt = (e) => {
    const r = svg.getBoundingClientRect();
    const px = ((e.clientX - r.left) / r.width) * W;
    return Math.max(0, Math.min(estimates.length - 1, Math.round(((px - L) / (W - L - R)) * n)));
  };
  svg.addEventListener("mousemove", (e) => {
    const i = plyAt(e), est = estimates[i], r = svg.getBoundingClientRect();
    tip.hidden = false;
    tip.style.left = `${(x(i) / W) * r.width}px`;
    tip.style.top = `${(y(Math.max(est.white.high, est.black.high)) / H) * r.height}px`;
    tip.textContent = `${i === 0 ? "Start" : `Move ${Math.ceil(i / 2)}${i % 2 ? "" : "…"}`} · W ${est.white.rating} · B ${est.black.rating}`;
  });
  svg.addEventListener("mouseleave", () => { tip.hidden = true; });
  svg.addEventListener("click", (e) => onPick(plyAt(e)));
}

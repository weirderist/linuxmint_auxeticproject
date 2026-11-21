"use strict";
// chartkit.js - the one scatter-chart renderer shared by the Campaigns page and the pure-shell page, so every chart has the same size, aspect ratio, margins, ticks, fonts and marker size.
// Each chart is drawn at the MEASURED pixel width of its grid column (no CSS scaling of the SVG, so text and strokes stay crisp at any zoom or device pixel ratio); lines sit on half pixels;
// nothing is rotated (the y-axis label is horizontal above the plot). Colours and stroke widths come from theme.css by class names (cl-ok, g0, .mk, .dl, ...).
const ChartKit = (() => {
  const CH = { aspect: 0.58, L: 58, R: 16, T: 30, B: 46, r: 4, yTicks: 5, xTicks: 5, minW: 260, fallbackW: 560 };
  const esc = s => String(s == null ? "" : s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const num = v => { if (v == null || v === "") return ""; if (Number.isInteger(v)) return String(v); const a = Math.abs(v); return (a !== 0 && (a < 1e-3 || a >= 1e5)) ? v.toExponential(2) : String(+v.toPrecision(4)); };
  const half = v => Math.round(v) + 0.5;
  function ticks(lo, hi, n) {
    if (!(hi > lo)) return [lo];
    const raw = (hi - lo) / n, p = Math.pow(10, Math.floor(Math.log10(raw))), m = raw / p, st = (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * p, out = [];
    for (let v = Math.ceil(lo / st) * st; v <= hi + st * 1e-9; v += st) out.push(+v.toPrecision(10));
    return out;
  }
  function size(el) { const w = Math.floor(el && el.clientWidth) || CH.fallbackW; const W = Math.max(CH.minW, w); return { W, H: Math.round(W * CH.aspect) }; }
  function mark(p, x, y) {          // marker size is fixed; the shape says the class, the colour comes from the class name
    const r = CH.r, c = "mk " + (p.cls || "g0"), t = "<title>" + esc(p.tip) + "</title>", d = r + 1;
    if (p.shape === "diamond") return '<path class="' + c + '" d="M' + x + " " + (y - d) + "l" + d + " " + d + "l-" + d + " " + d + "l-" + d + " -" + d + 'z">' + t + "</path>";
    if (p.shape === "triangle") return '<path class="' + c + '" d="M' + x + " " + (y - d) + "l" + d + " " + (2 * r) + "l-" + (2 * r + 2) + ' 0z">' + t + "</path>";
    if (p.shape === "cross") return '<path class="' + c + ' cross" d="M' + (x - r) + " " + (y - r) + "l" + (2 * r) + " " + (2 * r) + "m0 -" + (2 * r) + "l-" + (2 * r) + " " + (2 * r) + '">' + t + "</path>";
    return '<circle class="' + c + '" cx="' + x + '" cy="' + y + '" r="' + r + '">' + t + "</circle>";
  }
  // pts: [{x, y, cls, shape, tip}]; o: {xlog, xlabel, ylabel, line}; S: {W, H} from size()
  function scatter(pts, o, S) {
    const { W, H } = S, L = CH.L, R = CH.R, T = CH.T, B = CH.B;
    if (!pts.length) return '<svg class="nodata" width="' + W + '" height="' + H + '" role="img" aria-label="no data yet"><rect x="0.5" y="0.5" width="' + (W - 1) + '" height="' + (H - 1) + '" rx="6"/><text x="' + (W / 2) + '" y="' + (H / 2) + '" text-anchor="middle">no data yet for ' + esc(o.xlabel) + "</text></svg>";
    const fx = o.xlog ? Math.log : (v => v);
    const xs = pts.map(p => fx(p.x)), ys = pts.map(p => p.y);
    let x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(0, ...ys), y1 = Math.max(...ys);
    if (x1 === x0) { x0 -= 1; x1 += 1; } if (y1 === y0) y1 = y0 + 1;
    const pad = (x1 - x0) * 0.05; x0 -= pad; x1 += pad; y1 += (y1 - y0) * 0.06;
    const X = v => L + (fx(v) - x0) / (x1 - x0) * (W - L - R), Y = v => H - B - (v - y0) / (y1 - y0) * (H - T - B), f1 = v => v.toFixed(1);
    let s = '<svg width="' + W + '" height="' + H + '" viewBox="0 0 ' + W + " " + H + '" role="img" aria-label="' + esc(o.ylabel + " against " + o.xlabel) + '">';
    for (const v of ticks(y0, y1, CH.yTicks)) s += '<line class="gd" x1="' + L + '" x2="' + (W - R) + '" y1="' + half(Y(v)) + '" y2="' + half(Y(v)) + '"/><text x="' + (L - 8) + '" y="' + f1(Y(v) + 4) + '" text-anchor="end">' + esc(num(v)) + "</text>";
    const xt = o.xlog ? (() => { const a = Math.exp(x0), b = Math.exp(x1); return [a, Math.sqrt(a * b), b]; })() : ticks(x0, x1, CH.xTicks);
    for (const v of xt) s += '<line class="ax" x1="' + half(X(v)) + '" x2="' + half(X(v)) + '" y1="' + (H - B) + '" y2="' + (H - B + 5) + '"/><text x="' + f1(X(v)) + '" y="' + (H - B + 20) + '" text-anchor="middle">' + esc(num(+v.toPrecision(3))) + "</text>";
    s += '<line class="ax" x1="' + L + '" x2="' + (W - R) + '" y1="' + half(H - B) + '" y2="' + half(H - B) + '"/><line class="ax" x1="' + half(L) + '" x2="' + half(L) + '" y1="' + T + '" y2="' + (H - B) + '"/>';
    s += '<text x="' + L + '" y="16" text-anchor="start">' + esc(o.ylabel) + '</text><text x="' + f1(L + (W - L - R) / 2) + '" y="' + (H - 8) + '" text-anchor="middle">' + esc(o.xlabel + (o.xlog ? " (log scale)" : "")) + "</text>";
    if (o.line) { const q = pts.slice().sort((a, b) => a.x - b.x); s += '<polyline class="dl" points="' + q.map(p => f1(X(p.x)) + "," + f1(Y(p.y))).join(" ") + '"/>'; }
    for (const p of pts) s += mark(p, +X(p.x).toFixed(1), +Y(p.y).toFixed(1));
    return s + "</svg>";
  }
  return { CH, size, scatter, ticks, num };
})();

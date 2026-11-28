"use strict";
// Per-run graphs, hand-drawn SVG, no libraries: loop-length histogram, loop length along each wale / course, fold profile along the fold direction.
const $ = s => document.querySelector(s);
const esc = s => String(s == null ? '' : s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const q = new URLSearchParams(location.search).get('run');
if (!q || !/^[0-9A-Za-z_-]{1,64}\/[0-9]{1,6}$/.test(q)) { $('#err').textContent = 'Open this page from a run: /run?run=<campaign>/<id>'; throw new Error('no run'); }
const [campaign, run] = q.split('/'), id = 'run~' + campaign + '~' + run;
$('#rname').textContent = q; $('#lv').href = '/sim?run=' + q; $('#ls').href = '/strain?run=' + q;
const get = async p => { const r = await fetch('/api/results/' + id + p); if (!r.ok) { const e = await r.json().catch(() => ({})); throw new Error(e.detail || 'HTTP ' + r.status); } return r.json(); };
const ticks = (lo, hi, n) => { if (!(hi > lo)) return [lo]; const raw = (hi - lo) / n, p = Math.pow(10, Math.floor(Math.log10(raw))), m = raw / p, st = (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * p, o = []; for (let v = Math.ceil(lo / st) * st; v <= hi + st * 1e-9; v += st) o.push(+v.toPrecision(10)); return o; };
const num = v => +(+v).toPrecision(4);
function frame(w, h, L, R, T, B, x0, x1, y0, y1, xl, yl, xfmt) {
  const X = v => L + (v - x0) / (x1 - x0) * (w - L - R), Y = v => h - B - (v - y0) / (y1 - y0) * (h - T - B);
  let s = '<svg viewBox="0 0 ' + w + ' ' + h + '">';
  for (const v of ticks(y0, y1, 5)) s += '<line class="gd" x1="' + L + '" x2="' + (w - R) + '" y1="' + Y(v) + '" y2="' + Y(v) + '"/><text x="' + (L - 4) + '" y="' + (Y(v) + 3) + '" text-anchor="end">' + num(v) + '</text>';
  for (const v of ticks(x0, x1, 8)) s += '<line class="ax" x1="' + X(v) + '" x2="' + X(v) + '" y1="' + (h - B) + '" y2="' + (h - B + 3) + '"/><text x="' + X(v) + '" y="' + (h - B + 14) + '" text-anchor="middle">' + (xfmt ? xfmt(v) : num(v)) + '</text>';
  s += '<line class="ax" x1="' + L + '" x2="' + (w - R) + '" y1="' + (h - B) + '" y2="' + (h - B) + '"/><line class="ax" x1="' + L + '" x2="' + L + '" y1="' + T + '" y2="' + (h - B) + '"/>';
  s += '<text x="' + (L + (w - L - R) / 2) + '" y="' + (h - 4) + '" text-anchor="middle">' + esc(xl) + '</text><text x="11" y="' + (T + (h - T - B) / 2) + '" text-anchor="middle" transform="rotate(-90 11 ' + (T + (h - T - B) / 2) + ')">' + esc(yl) + '</text>';
  return { s, X, Y };
}
let loops = null, hRel = true, pRel = true, pWale = true;
function histogram() {
  const g = (hRel ? loops.ratio_relaxed : loops.ratio_asbuilt), el = $('#hist');
  if (!g) { el.textContent = 'No relaxed state for this run.'; return; }
  const v = g.flat().filter(x => x != null), lo = Math.min(...v), hi = Math.max(...v), nb = 40, w = (hi - lo) / nb || 1, c = new Array(nb).fill(0);
  for (const x of v) c[Math.min(nb - 1, Math.floor((x - lo) / w))]++;
  const f = frame(760, 280, 48, 10, 10, 38, lo, lo + w * nb, 0, Math.max(...c), 'loop length / as-built mean loop length', 'loops');
  let s = f.s; c.forEach((n, i) => { s += '<rect x="' + f.X(lo + i * w) + '" y="' + f.Y(n) + '" width="' + Math.max(1, f.X(lo + (i + 1) * w) - f.X(lo + i * w) - 1) + '" height="' + (f.Y(0) - f.Y(n)) + '" fill="var(--chart-line)"><title>' + num(lo + i * w) + ' to ' + num(lo + (i + 1) * w) + ': ' + n + ' loops</title></rect>'; });
  s += '<line x1="' + f.X(1) + '" x2="' + f.X(1) + '" y1="10" y2="242" stroke="currentColor" stroke-dasharray="3 3" opacity=".6"/>';
  el.innerHTML = s + '</svg><div class="small mut">' + v.length + ' loops; min ' + num(lo) + ', max ' + num(hi) + ', mean ' + num(v.reduce((a, b) => a + b, 0) / v.length) + '. Dashed line: 1 = the as-built mean loop length (' + num(loops.mean_loop_asbuilt) + ' ' + esc(loops.units === 'mm' ? 'mm' : 'length units') + ').</div>';
}
function profile() {
  const g = pRel ? loops.ratio_relaxed : loops.ratio_asbuilt, el = $('#prof'), sel = $('#pSel');
  if (!g) { el.textContent = 'No relaxed state for this run.'; return; }
  const nc = loops.ncourse, nw = loops.ncol, n = pWale ? nc : nw, m = pWale ? nw : nc;
  $('#pT').textContent = pWale ? 'wale (column)' : 'course';
  if (sel.dataset.mode !== (pWale ? 'w' : 'c')) { sel.innerHTML = Array.from({ length: m }, (_, i) => '<option>' + i + '</option>').join(''); sel.dataset.mode = pWale ? 'w' : 'c'; sel.value = '0'; }
  const k = +sel.value || 0, at = (i, j) => pWale ? g[i][j] : g[j][i], ys = [], lo = [], hi = [], mean = [];
  for (let i = 0; i < n; i++) { const vals = []; for (let j = 0; j < m; j++) { const v = at(i, j); if (v != null) vals.push(v); } lo.push(Math.min(...vals)); hi.push(Math.max(...vals)); mean.push(vals.reduce((a, b) => a + b, 0) / vals.length); ys.push(at(i, k)); }
  const all = [...lo, ...hi].filter(Number.isFinite), y0 = Math.min(...all), y1 = Math.max(...all), pad = (y1 - y0) * .08 || .01;
  const f = frame(760, 280, 48, 10, 10, 38, 0, Math.max(n - 1, 1), y0 - pad, y1 + pad, pWale ? 'course (0 = bottom)' : 'wale (column, left to right)', 'loop length / as-built mean', v => Math.round(v));
  const pts = a => a.map((v, i) => v == null || !Number.isFinite(v) ? null : f.X(i) + ',' + f.Y(v)).filter(Boolean);
  let s = f.s + '<polygon points="' + pts(hi).concat(pts(lo).reverse()).join(' ') + '" fill="var(--chart-line)" opacity=".15"/><polyline fill="none" stroke="var(--chart-line)" stroke-width="1" opacity=".5" stroke-dasharray="4 3" points="' + pts(mean).join(' ') + '"/><polyline fill="none" stroke="var(--chart-line)" stroke-width="2" points="' + pts(ys).join(' ') + '"/>';
  ys.forEach((v, i) => { if (v != null) s += '<circle cx="' + f.X(i) + '" cy="' + f.Y(v) + '" r="3.5" fill="var(--chart-line)"><title>' + (pWale ? 'course ' : 'wale ') + i + ': ' + num(v) + '</title></circle>'; });
  el.innerHTML = s + '</svg><div class="small mut">Line: ' + (pWale ? 'wale ' : 'course ') + k + '. Band: range over all ' + (pWale ? 'wales' : 'courses') + '; dashed: their mean. Loops at the fabric edge have fewer neighbours and may differ.</div>';
}
function foldGraph(f) {
  const dir = f.direction, th = f.threshold_rad, el = $('#fold');
  $('#foldTxt').innerHTML = 'Fold direction (from the campaign metrics): <b>' + esc(dir || 'not recorded') + '</b>' + (f.amplitude != null ? '; amplitude ' + num(f.amplitude) + ' mm' : '') +
    (th != null ? '; classification threshold ' + th + ' rad of turning across the sheet (turn along courses ' + (f.turn_x_rad != null ? num(f.turn_x_rad) : '?') + ', along wales ' + (f.turn_y_rad != null ? num(f.turn_y_rad) : '?') + ')' : '') +
    '.<br><span class="mut">' + esc(f.threshold_note) + ' ' + esc(f.note) + '</span>';
  const show = dir === 'course' ? ['course'] : dir === 'wale' ? ['wale'] : ['course', 'wale'];
  let h = '';
  for (const which of show) {
    const xs = which === 'course' ? f.x_mm : f.y_mm, ys = which === 'course' ? f.profile_course : f.profile_wale, ok = ys.map((v, i) => v == null ? null : [xs[i], v]).filter(Boolean);
    if (!ok.length) continue;
    const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ok.map(p => p[1]), 0), y1 = Math.max(...ok.map(p => p[1]), 0), pad = (y1 - y0) * .08 || .01;
    const fr = frame(760, 240, 52, 10, 10, 38, x0, x1, y0 - pad, y1 + pad, (which === 'course' ? 'x along the courses' : 'y along the wales') + ' (' + (f.units === 'mm' ? 'mm' : 'length units') + ')', 'mean height after plane removal');
    h += '<div class="small">' + (which === 'course' ? 'Along the courses (x)' : 'Along the wales (y)') + '</div>' + fr.s + '<line x1="52" x2="750" y1="' + fr.Y(0) + '" y2="' + fr.Y(0) + '" stroke="currentColor" stroke-dasharray="3 3" opacity=".6"/><polyline fill="none" stroke="var(--chart-line)" stroke-width="2" points="' + ok.map(p => fr.X(p[0]) + ',' + fr.Y(p[1])).join(' ') + '"/>' + ok.map(p => '<circle cx="' + fr.X(p[0]) + '" cy="' + fr.Y(p[1]) + '" r="3" fill="var(--chart-line)"><title>' + num(p[0]) + ': ' + num(p[1]) + '</title></circle>').join('') + '</svg>';
  }
  el.innerHTML = h || '<div class="small mut">No profile available.</div>';
}
$('#hA').onclick = () => { hRel = false; $('#hA').classList.add('on'); $('#hR').classList.remove('on'); histogram(); };
$('#hR').onclick = () => { hRel = true; $('#hR').classList.add('on'); $('#hA').classList.remove('on'); histogram(); };
$('#pA').onclick = () => { pRel = false; $('#pA').classList.add('on'); $('#pR').classList.remove('on'); profile(); };
$('#pR').onclick = () => { pRel = true; $('#pR').classList.add('on'); $('#pA').classList.remove('on'); profile(); };
$('#pW').onclick = () => { pWale = true; $('#pW').classList.add('on'); $('#pC').classList.remove('on'); profile(); };
$('#pC').onclick = () => { pWale = false; $('#pC').classList.add('on'); $('#pW').classList.remove('on'); profile(); };
$('#pSel').onchange = profile;
get('/loops').then(l => { loops = l; $('#notes').textContent = l.note; histogram(); profile(); }).catch(e => { $('#err').textContent = 'Loop data: ' + e.message; });
get('/fold').then(foldGraph).catch(e => { $('#fold').textContent = 'Fold profile: ' + e.message; });

import { createViewer, syncCameras, loadTubeBase, tubeMesh, fetchColors, legendHTML, legendTip, rid } from '/viewer_core.js';
// Strain page for one campaign run: A = loop length (3D), B = axial strain of the relaxed fabric (3D or flat map), C = strain graph (SVG).
// Both 3D viewers share one loaded geometry per variant (same position / normal / index arrays); only the colour attribute differs.
const $ = s => document.querySelector(s);
const esc = s => String(s == null ? '' : s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const q = new URLSearchParams(location.search).get('run');
if (!q || !/^[0-9A-Za-z_-]{1,64}\/[0-9]{1,6}$/.test(q)) { $('#err').textContent = 'Open this page from a run: /strain?run=<campaign>/<id>'; throw new Error('no run'); }
const [campaign, run] = q.split('/'), id = rid(campaign, run);
$('#rname').textContent = q; $('#lv').href = '/sim?run=' + q; $('#lg').href = '/run?run=' + q;
// Structure for a later stretch-test state: every data call takes `step` (0 = the relaxed fabric). Only step 0 exists; the server refuses any other value.
const STEP = 0;
const fail = e => { $('#err').textContent = String(e && e.message || e); };

const vA = createViewer($('#vA')), vB = createViewer($('#vB'));
syncCameras([vA, vB], () => $('#sync').checked);
const bases = {};                                                   // variant -> Promise of the shared geometry
const getBase = (variant, on) => bases[variant] || (bases[variant] = loadTubeBase(id, variant, on).catch(e => { delete bases[variant]; throw e; }));
let meshA = null, meshB = null, aVariant = 'relaxed', flipB = false, firstA = true, firstB = true;
const disposeMesh = m => { if (m) { m.geometry.dispose(); m.material.dispose(); } };
const clipV = () => Math.min(100, Math.max(50, +$('#clip').value || 98));

async function showA(refit) {
  const st = $('#sA'), v = aVariant;
  try {
    const base = await getBase(v, t => { st.textContent = t; });
    st.textContent = 'colouring…';
    const c = await fetchColors(id, 'loop', v, 'tube');
    if (v !== aVariant) return;
    disposeMesh(meshA); meshA = tubeMesh(base, c.rgb); vA.show(meshA, refit || firstA); firstA = false;
    $('#lA').innerHTML = legendHTML(c.legend); $('#lA').title = legendTip(c.legend); $('#lA').hidden = false; st.textContent = base.nv.toLocaleString() + ' vertices';
  } catch (e) { st.textContent = ''; fail(e); }
}
async function showB() {
  const st = $('#sB');
  try {
    const base = await getBase('relaxed', t => { st.textContent = t; });
    st.textContent = 'colouring…';
    const c = await fetchColors(id, 'strain', 'relaxed', 'tube', '&flip=' + (flipB ? 1 : 0) + '&clip=' + clipV());
    disposeMesh(meshB); meshB = tubeMesh(base, c.rgb); vB.show(meshB, firstB); firstB = false;
    $('#lB').innerHTML = legendHTML(c.legend); $('#lB').title = legendTip(c.legend); $('#lB').hidden = false; st.textContent = base.nv.toLocaleString() + ' vertices';
  } catch (e) { st.textContent = ''; fail(e); }
}
Promise.all([showA(true), showB()]);          // both viewers fit the same geometry, so their cameras start identical

$('#aBase').onclick = () => { aVariant = 'baseline'; $('#aBase').classList.add('on'); $('#aRel').classList.remove('on'); showA(true); };
$('#aRel').onclick = () => { aVariant = 'relaxed'; $('#aRel').classList.add('on'); $('#aBase').classList.remove('on'); showA(true); };
$('#flip').onclick = () => { flipB = !flipB; showB(); drawMap(); };
$('#clip').onchange = () => { showB(); drawMap(); };
$('#b3d').onclick = () => { $('#b3d').classList.add('on'); $('#b2d').classList.remove('on'); $('#vB').style.display = ''; $('#map').style.display = 'none'; vB.resize(); };
$('#b2d').onclick = () => { $('#b2d').classList.add('on'); $('#b3d').classList.remove('on'); $('#vB').style.display = 'none'; $('#map').style.display = 'block'; drawMap(); };

// ---------------------------------------------------------------- data
let strainP = null;
const getStrain = () => strainP || (strainP = fetch('/api/results/' + id + '/strain?step=' + STEP).then(async r => { if (!r.ok) { const e = await r.json().catch(() => ({})); throw new Error(e.detail || 'HTTP ' + r.status); } return r.json(); }).catch(e => { strainP = null; throw e; }));
const pct = v => (v == null ? '' : (100 * v).toFixed(3) + '%');

// ---------------------------------------------------------------- flat map: courses against wales, one cell per loop (mean strain of its segments)
async function drawMap() {
  const el = $('#map'); if (el.style.display === 'none') return;
  try {
    const d = await getStrain();
    const fi = await (await fetch('/api/results/' + id + '/field-info?field=strain&variant=relaxed&flip=' + (flipB ? 1 : 0) + '&clip=' + clipV())).json();
    const L = fi.legend, c = L.vmax, [neg, mid, pos] = L.stops, lerp = (a, b, t) => a.map((x, i) => Math.round(x + (b[i] - x) * t));
    const col = v => { if (v == null) return '#8884'; const t = Math.max(-1, Math.min(1, v / c)); return 'rgb(' + (t >= 0 ? lerp(mid, pos, t) : lerp(mid, neg, -t)).join(',') + ')'; };
    const nc = d.ncourse, nw = d.ncol, cs = Math.max(6, Math.min(24, Math.floor(620 / nw))), pl = 34, pb = 26, W = pl + nw * cs + 6, H = nc * cs + pb + 6;
    let s = '<svg viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="axial strain map, courses against wales">';
    for (let ci = 0; ci < nc; ci++) for (let j = 0; j < nw; j++) {
      const v = d.grid[ci][j]; s += '<rect x="' + (pl + j * cs) + '" y="' + ((nc - 1 - ci) * cs) + '" width="' + cs + '" height="' + cs + '" fill="' + col(v) + '"><title>course ' + ci + ', wale ' + j + ': ' + (v == null ? 'no data' : pct(v)) + '</title></rect>';
    }
    for (let j = 0; j < nw; j += Math.max(1, Math.round(nw / 8))) s += '<text x="' + (pl + (j + .5) * cs) + '" y="' + (nc * cs + 12) + '" text-anchor="middle">' + j + '</text>';
    for (let ci = 0; ci < nc; ci += Math.max(1, Math.round(nc / 8))) s += '<text x="' + (pl - 4) + '" y="' + ((nc - 1 - ci + .7) * cs) + '" text-anchor="end">' + ci + '</text>';
    s += '<text x="' + (pl + nw * cs / 2) + '" y="' + (H - 2) + '" text-anchor="middle">wale (column, left to right)</text></svg>';
    el.title = 'one cell = one loop (mean axial strain of its segments); course 0 = bottom row'; el.innerHTML = s + '<div class="legend lg-static">' + legendHTML(L) + '</div>';
  } catch (e) { el.textContent = String(e.message || e); }
}

// ---------------------------------------------------------------- graph (hand-drawn SVG)
let gMode = 'hist';
const ticks = (lo, hi, n) => { if (!(hi > lo)) return [lo]; const raw = (hi - lo) / n, p = Math.pow(10, Math.floor(Math.log10(raw))), m = raw / p, st = (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * p, o = []; for (let v = Math.ceil(lo / st) * st; v <= hi + st * 1e-9; v += st) o.push(+v.toPrecision(10)); return o; };
function frame(w, h, L, R, T, B, x0, x1, y0, y1, xl, yl, xfmt, yfmt) {
  const X = v => L + (v - x0) / (x1 - x0) * (w - L - R), Y = v => h - B - (v - y0) / (y1 - y0) * (h - T - B);
  let s = '<svg viewBox="0 0 ' + w + ' ' + h + '">';
  for (const v of ticks(y0, y1, 5)) s += '<line class="gd" x1="' + L + '" x2="' + (w - R) + '" y1="' + Y(v) + '" y2="' + Y(v) + '"/><text x="' + (L - 4) + '" y="' + (Y(v) + 3) + '" text-anchor="end">' + yfmt(v) + '</text>';
  for (const v of ticks(x0, x1, 8)) s += '<line class="ax" x1="' + X(v) + '" x2="' + X(v) + '" y1="' + (h - B) + '" y2="' + (h - B + 3) + '"/><text x="' + X(v) + '" y="' + (h - B + 14) + '" text-anchor="middle">' + xfmt(v) + '</text>';
  s += '<line class="ax" x1="' + L + '" x2="' + (w - R) + '" y1="' + (h - B) + '" y2="' + (h - B) + '"/><line class="ax" x1="' + L + '" x2="' + L + '" y1="' + T + '" y2="' + (h - B) + '"/>';
  s += '<text x="' + (L + (w - L - R) / 2) + '" y="' + (h - 4) + '" text-anchor="middle">' + esc(xl) + '</text><text x="11" y="' + (T + (h - T - B) / 2) + '" text-anchor="middle" transform="rotate(-90 11 ' + (T + (h - T - B) / 2) + ')">' + esc(yl) + '</text>';
  return { s, X, Y };
}
async function drawGraph() {
  const el = $('#gr');
  try {
    const d = await getStrain(), W = 760, H = 300, st = d.stats;
    $('#notes').innerHTML = esc(d.notes.join(' ')) + '<br>' + st.n.toLocaleString() + ' segments; strain min ' + pct(st.min) + ', mean ' + pct(st.mean) + ', max ' + pct(st.max) + '.';
    const selL = $('#gSelL'), sel = $('#gSel');
    if (gMode === 'hist') {
      selL.hidden = true;
      const e = d.hist.edges, c = d.hist.counts, ymax = Math.max(...c, 1), f = frame(W, H, 52, 10, 10, 38, e[0] * 100, e[e.length - 1] * 100, 0, ymax, 'axial strain of the relaxed fabric (%)', 'segments', v => +v.toPrecision(3), v => v);
      let s = f.s; for (let i = 0; i < c.length; i++) s += '<rect x="' + f.X(e[i] * 100) + '" y="' + f.Y(c[i]) + '" width="' + Math.max(1, f.X(e[i + 1] * 100) - f.X(e[i] * 100) - 1) + '" height="' + (f.Y(0) - f.Y(c[i])) + '" fill="var(--s1)"><title>' + pct(e[i]) + ' to ' + pct(e[i + 1]) + ': ' + c[i] + ' segments</title></rect>';
      if (e[0] < 0 && e[e.length - 1] > 0) s += '<line x1="' + f.X(0) + '" x2="' + f.X(0) + '" y1="10" y2="' + (H - 38) + '" stroke="currentColor" stroke-dasharray="3 3" opacity=".6"/>';
      el.innerHTML = s + '</svg>'; return;
    }
    const wale = gMode === 'wale', nc = d.ncourse, nw = d.ncol, n = wale ? nc : nw, m = wale ? nw : nc;
    selL.hidden = false; $('#gSelT').textContent = wale ? 'wale (column)' : 'course';
    if (sel.dataset.mode !== gMode) { sel.innerHTML = Array.from({ length: m }, (_, i) => '<option>' + i + '</option>').join(''); sel.dataset.mode = gMode; sel.value = '0'; }
    const k = +sel.value || 0, at = (i, j) => wale ? d.grid[i][j] : d.grid[j][i];            // i = position along the profile, j = which wale / course
    const ys = [], lo = [], hi = [], mean = [];
    for (let i = 0; i < n; i++) { const vals = []; for (let j = 0; j < m; j++) { const v = at(i, j); if (v != null) vals.push(v); } lo.push(vals.length ? Math.min(...vals) : null); hi.push(vals.length ? Math.max(...vals) : null); mean.push(vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null); ys.push(at(i, k)); }
    const all = [...lo, ...hi, ...ys].filter(v => v != null), y0 = Math.min(...all, 0) * 100, y1 = Math.max(...all, 0) * 100, pad = (y1 - y0) * .06 || 1;
    const f = frame(W, H, 52, 10, 10, 38, 0, Math.max(n - 1, 1), y0 - pad, y1 + pad, wale ? 'course (0 = bottom)' : 'wale (column, left to right)', 'axial strain (%)', v => Math.round(v), v => +v.toPrecision(3));
    const pts = a => a.map((v, i) => v == null ? null : f.X(i) + ',' + f.Y(v * 100)).filter(Boolean);
    let s = f.s + '<line x1="52" x2="' + (W - 10) + '" y1="' + f.Y(0) + '" y2="' + f.Y(0) + '" stroke="currentColor" stroke-dasharray="3 3" opacity=".6"/>';
    s += '<polygon points="' + pts(hi).concat(pts(lo).reverse()).join(' ') + '" fill="var(--s1)" opacity=".15"/><polyline fill="none" stroke="var(--s1)" stroke-width="1" opacity=".5" stroke-dasharray="4 3" points="' + pts(mean).join(' ') + '"/>';
    s += '<polyline fill="none" stroke="var(--s1)" stroke-width="2" points="' + pts(ys).join(' ') + '"/>';
    ys.forEach((v, i) => { if (v != null) s += '<circle cx="' + f.X(i) + '" cy="' + f.Y(v * 100) + '" r="3.5" fill="var(--s1)"><title>' + (wale ? 'course ' : 'wale ') + i + ': ' + pct(v) + '</title></circle>'; });
    el.innerHTML = s + '</svg><div class="small mut">Line: ' + (wale ? 'wale ' : 'course ') + k + '. Shaded band: range over all ' + (wale ? 'wales' : 'courses') + '; dashed: their mean. Each point is the mean strain of the segments of one loop.</div>';
  } catch (e) { el.textContent = String(e.message || e); }
}
for (const [b, m] of [['#gH', 'hist'], ['#gW', 'wale'], ['#gC', 'course']]) $(b).onclick = () => { gMode = m; for (const x of ['#gH', '#gW', '#gC']) $(x).classList.toggle('on', x === b); drawGraph(); };
$('#gSel').onchange = drawGraph;
drawGraph();

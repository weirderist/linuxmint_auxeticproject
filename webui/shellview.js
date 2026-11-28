import { createViewer, syncCameras, loadTubeBase, tubeMesh } from '/viewer_core.js';
// Shell viewer: the folded sheet of one run in 3D, coloured by a membrane field, with a second view (another field of the same run, or another run) side by side. Colours and sizes come from theme.css.
const $ = s => document.querySelector(s);
const esc = s => String(s == null ? '' : s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
// ?c=<campaign>&run=S000001[&b=...]: a campaign run; ?sim=<id>[&b=<id>]: a single shell run started from the Simulator page (webui/shellsim.py)
const P = new URLSearchParams(location.search), SIM = P.get('sim'), camp = SIM ? null : P.get('c'), runB0 = P.get('b'); let runA = SIM || P.get('run');
const ok = s => /^[0-9A-Za-z_-]{1,64}$/.test(s || ''), okRun = s => (SIM ? /^[0-9]{8}-[0-9]{6}$/ : /^S[0-9]{6}$/).test(s || '');
if ((!SIM && !ok(camp)) || !okRun(runA)) { $('#err').textContent = 'Open this page from a run: /shellview?c=<campaign>&run=S000001 or /shellview?sim=<id>'; throw new Error('no run'); }
const SELS = [...new Set([runA, runB0, ...(P.get('sel') || '').split(',')].filter(okRun))].slice(0, 40);          // every run picked on the shell page (2026-11-28)
$('#rname').textContent = SIM ? 'simulator / ' + runA : camp + ' / ' + (SELS.length > 2 ? SELS.length + ' runs' : SELS.join(', ')); $('#back').href = SIM ? '/sim' : '/shell'; if (SIM) $('#back').textContent = 'Simulator';
const FIELDS = [['area_strain', 'membrane area strain'], ['stretch_max', 'largest principal stretch - 1'], ['stretch_min', 'smallest principal stretch - 1'], ['z', 'height z (mm)'], ['tilt', 'surface tilt (degrees)']];
const STRAIN = ['area_strain', 'stretch_max', 'stretch_min'];
for (const sel of [$('#fA'), $('#fB')]) for (const [k, t] of FIELDS) sel.add(new Option(t, k));
$('#fA').value = P.get('f') || 'area_strain'; $('#fB').value = P.get('g') || 'z';
const url = r => SIM ? '/api/shellsim/' + encodeURIComponent(r) : '/api/shell/' + encodeURIComponent(camp) + '/runs/' + encodeURIComponent(r);
async function colours(r, field, clip) {
  const resp = await fetch(url(r) + '/colors?field=' + field + '&clip=' + clip); if (!resp.ok) { const e = await resp.json().catch(() => ({})); throw new Error(e.detail || 'HTTP ' + resp.status); }
  return { rgb: new Uint8Array(await resp.arrayBuffer()), legend: JSON.parse(resp.headers.get('X-Legend')) };
}
// Legend (compact, 2026-11-28 K: it hid the shell): short name, colour bar, three ticks; the strain fields mark the ends squeezed / stretched.
// The range note and the data minimum and maximum are in the tooltip. The only inline style is the colour ramp itself, which the server sends with the data.
const SHORT = { area_strain: 'area strain', stretch_max: 'max stretch - 1', stretch_min: 'min stretch - 1', z: 'height z (mm)', tilt: 'tilt (deg)' };
function legendHTML(l) {
  const f = x => (+x).toPrecision(3), e = STRAIN.includes(l.field) ? ['squeezed', 'stretched'] : (l.end_labels || ['low', 'high']), ramp = l.stops.map(c => 'rgb(' + c.join(',') + ')').join(',');
  return '<div class="lg-title">' + esc(SHORT[l.field] || l.label) + '</div><div class="lg-bar" style="background:linear-gradient(to right,' + ramp + ')"></div>' +
    '<div class="lg-ticks"><span>' + f(l.vmin) + '</span><span>' + f(l.ticks[1]) + '</span><span>' + f(l.vmax) + '</span></div>' +
    '<div class="lg-ticks lg-ends"><span>' + esc(e[0]) + '</span><span>' + esc(e[1]) + '</span></div>';          // the end words on their own row: same place on both sides (2026-11-28)
}
const legendTip = l => l.label + ' | data ' + (+l.data_min).toPrecision(4) + ' to ' + (+l.data_max).toPrecision(4) + (l.note ? ' | ' + l.note : '');
const views = { A: createViewer($('#vA')), B: createViewer($('#vB')) }; syncCameras([views.A, views.B], () => $('#sync').checked);
for (const v of [views.A, views.B]) { v.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2)); v.resize(); }          // shell meshes are small: render at up to 2x the CSS pixels so the 3D view is crisp on high-density screens
const bases = {}; let runBsel = runB0 || SELS[1] || runA;
async function base(r, st) { if (!bases[r]) bases[r] = await loadTubeBase(null, 'baseline', t => st.textContent = t, url(r) + '/mesh'); return bases[r]; }
async function draw(which, r, field) {
  const st = $('#s' + which), lg = $('#l' + which), v = views[which]; st.textContent = 'loading';
  try {
    const b = await base(r, st), c = await colours(r, field, +$('#clip').value || 98), first = !v.__shown; v.show(tubeMesh(b, c.rgb), first); v.__shown = true;
    lg.innerHTML = legendHTML(c.legend); lg.title = legendTip(c.legend); lg.hidden = false; st.textContent = '';
    $('#t' + which).textContent = r + ' · ' + (SHORT[field] || c.legend.label);
  } catch (e) { st.textContent = e.message; lg.hidden = true; }
}
// Matrices (2026-11-28, K): one card per selected run: the stitch matrix as the shell reads it (row 1 = bottom course, so it looks like the 3D view from above), its settings and descriptors,
// and Left / Right to show that run in a 3D view. Colours from theme.css (--knit, --purl).
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
function drawMatrix(cv, M) {
  const R = M.length, C = M[0].length, cell = Math.max(3, Math.min(14, Math.floor(200 / Math.max(R, C)))), dpr = Math.min(window.devicePixelRatio || 1, 2);
  cv.width = C * cell * dpr; cv.height = R * cell * dpr; cv.style.width = C * cell + 'px'; cv.style.height = R * cell + 'px';
  const g = cv.getContext('2d'); g.scale(dpr, dpr); const k = css('--knit'), p = css('--purl'), line = css('--card');
  for (let i = 0; i < R; i++) for (let j = 0; j < C; j++) { g.fillStyle = M[i][j] === 1 ? k : p; g.fillRect(j * cell, (R - 1 - i) * cell, cell, cell); }
  if (cell >= 6) { g.strokeStyle = line; g.lineWidth = 1; for (let i = 0; i <= R; i++) { g.beginPath(); g.moveTo(0, i * cell + .5); g.lineTo(C * cell, i * cell + .5); g.stroke(); } for (let j = 0; j <= C; j++) { g.beginPath(); g.moveTo(j * cell + .5, 0); g.lineTo(j * cell + .5, R * cell); g.stroke(); } }
}
const details = {};
async function info() {
  const box = $('#mats'); box.innerHTML = '';
  for (const r of SELS) {
    const card = document.createElement('div'); card.className = 'mat' + (r === runA ? ' on-a' : '') + (r === runBsel ? ' on-b' : ''); box.append(card);
    try {
      const d = details[r] || (details[r] = await (await fetch(url(r))).json()), s = d.settings || {}, m = d.metrics || {}, sd = d.seed || {};
      const head = document.createElement('div'); head.className = 'mat-h';
      head.innerHTML = '<b>' + esc(d.run || r) + '</b> <span class="mut">' + esc(d.label || '') + '</span>';
      const cv = document.createElement('canvas'); if (s.matrix && s.matrix.length) drawMatrix(cv, s.matrix);
      const txt = document.createElement('div'); txt.className = 'small mut';
      txt.innerHTML = (s.matrix ? s.matrix.length + ' x ' + s.matrix[0].length + ' × ' + esc(s.tiles) + ' tiles' : '') + ' · CPI ' + esc(s.cpi) + ' WPI ' + esc(s.wpi) +
        (m.plan_x != null ? '<br>plan ' + (+m.plan_x).toFixed(1) + ' × ' + (+m.plan_y).toFixed(1) + ' mm · thick ' + (+m.thickness).toFixed(1) + ' mm · fold ' + esc(m.fold_direction || '') : '') +
        (sd.real_fold ? '<br>real fold: ' + esc(sd.real_fold) : '') + '<br>' + esc(d.source || '');
      const bt = document.createElement('div'); bt.className = 'btnrow';
      for (const [w, t] of [['A', 'Left'], ['B', 'Right']]) { const b = document.createElement('button'); b.textContent = t; b.onclick = () => show(w, r); bt.append(b); }
      card.append(head, cv, txt, bt);
    } catch (e) { card.textContent = r + ': ' + e.message; }
  }
}
function show(which, r) {
  if (which === 'A') { runA = r; views.A.__shown = false; draw('A', runA, $('#fA').value); } else { runBsel = r; $('#rB').value = r; views.B.__shown = false; draw('B', runBsel, $('#fB').value); }
  for (const c of document.querySelectorAll('#mats .mat')) { const id = (c.querySelector('b') || {}).textContent; c.classList.toggle('on-a', id === runA); c.classList.toggle('on-b', id === runBsel); }
}
async function runList() {
  try { const sel = $('#rB'); let rows;
    if (SIM) rows = (await (await fetch('/api/shellsim')).json()).filter(r => r.state === 'done').map(r => ({ run: r.id, label: r.name }));
    else rows = (await (await fetch('/api/shell/' + encodeURIComponent(camp))).json()).rows.filter(r => r.status === 'done' && !r.heavy_deleted);
    sel.innerHTML = ''; for (const r of rows) sel.add(new Option(r.run + ' ' + (r.label || ''), r.run)); sel.value = runBsel; if (!sel.value) { sel.add(new Option(runBsel, runBsel)); sel.value = runBsel; } } catch (e) {}
}
const redraw = () => { draw('A', runA, $('#fA').value); draw('B', runBsel, $('#fB').value); };
$('#fA').onchange = () => draw('A', runA, $('#fA').value); $('#fB').onchange = () => draw('B', runBsel, $('#fB').value);
$('#rB').onchange = () => show('B', $('#rB').value); $('#clip').onchange = redraw;
async function waitSim() {          // a simulator shell run still in progress: show its growth steps, draw when metrics.json exists
  for (;;) {
    let d; try { d = await (await fetch(url(runA))).json(); } catch (e) { return; }
    if (d.state === 'done') return;
    const msg = d.state === 'running' ? 'running: growth step ' + (d.progress.growth_steps_done || 0) + ' of ' + (d.progress.steps || '?') : d.state + ' (' + ((d.log_tail || []).slice(-1)[0] || 'no log') + ')';
    $('#sA').textContent = msg; $('#sB').textContent = msg; if (d.state !== 'running') throw new Error(d.state);
    await new Promise(r => setTimeout(r, 4000));
  }
}
(SIM ? waitSim() : Promise.resolve()).then(runList).then(() => { redraw(); info(); }).catch(() => {});

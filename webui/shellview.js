import { createViewer, syncCameras, loadTubeBase, tubeMesh } from '/viewer_core.js';
// Shell viewer: the folded sheet of one run in 3D, coloured by a membrane field, with a second view (another field of the same run, or another run) side by side. Colours and sizes come from theme.css.
const $ = s => document.querySelector(s);
const esc = s => String(s == null ? '' : s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
// ?c=<campaign>&run=S000001[&b=...]: a campaign run; ?sim=<id>[&b=<id>]: a single shell run started from the Simulator page (webui/shellsim.py)
const P = new URLSearchParams(location.search), SIM = P.get('sim'), camp = SIM ? null : P.get('c'), runA = SIM || P.get('run'), runB0 = P.get('b');
const ok = s => /^[0-9A-Za-z_-]{1,64}$/.test(s || ''), okRun = s => (SIM ? /^[0-9]{8}-[0-9]{6}$/ : /^S[0-9]{6}$/).test(s || '');
if ((!SIM && !ok(camp)) || !okRun(runA)) { $('#err').textContent = 'Open this page from a run: /shellview?c=<campaign>&run=S000001 or /shellview?sim=<id>'; throw new Error('no run'); }
$('#rname').textContent = SIM ? 'simulator / ' + runA : camp + ' / ' + runA; $('#back').href = SIM ? '/sim' : '/shell'; if (SIM) $('#back').textContent = 'Simulator';
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
const bases = {}; let runBsel = runB0 || runA;
async function base(r, st) { if (!bases[r]) bases[r] = await loadTubeBase(null, 'baseline', t => st.textContent = t, url(r) + '/mesh'); return bases[r]; }
async function draw(which, r, field) {
  const st = $('#s' + which), lg = $('#l' + which), v = views[which]; st.textContent = 'loading';
  try {
    const b = await base(r, st), c = await colours(r, field, +$('#clip').value || 98), first = !v.__shown; v.show(tubeMesh(b, c.rgb), first); v.__shown = true;
    lg.innerHTML = legendHTML(c.legend); lg.title = legendTip(c.legend); lg.hidden = false; st.textContent = '';
    $('#t' + which).textContent = r + ' · ' + (SHORT[field] || c.legend.label);
  } catch (e) { st.textContent = e.message; lg.hidden = true; }
}
async function info() {
  try {
    const rows = [await (await fetch(url(runA))).json()]; if (runBsel !== runA) rows.push(await (await fetch(url(runBsel))).json());
    $('#info').innerHTML = rows.map(d => { const s = d.settings || {}, m = d.metrics || {}, sd = d.seed || {};
      return '<p><b>' + esc(d.run) + '</b> ' + esc(d.label || '') + ' <span class="mut">' + esc(d.source) + '</span> · ' + (s.matrix ? s.matrix.length + 'x' + s.matrix[0].length + ' x' + esc(s.tiles) : '') + ' · CPI ' + esc(s.cpi) + ' WPI ' + esc(s.wpi) + ' R ' + (s.cpi / s.wpi).toFixed(3) + ' · plan ' + (+m.plan_x).toFixed(1) + ' x ' + (+m.plan_y).toFixed(1) + ' mm · thick ' + (+m.thickness).toFixed(1) + ' mm · waves ' + esc(m.wave_k) +
        (d.tightness && d.tightness.tf_rel != null ? ' · TF_rel ' + (+d.tightness.tf_rel).toFixed(2) : '') + (sd.real_fold ? '<br><span class="mut">real fold (' + esc(sd.sample || '') + '): ' + esc(sd.real_fold) + '</span>' : '') + '</p>'; }).join('');
  } catch (e) { $('#info').textContent = ''; }
}
async function runList() {
  try { const sel = $('#rB'); let rows;
    if (SIM) rows = (await (await fetch('/api/shellsim')).json()).filter(r => r.state === 'done').map(r => ({ run: r.id, label: r.name }));
    else rows = (await (await fetch('/api/shell/' + encodeURIComponent(camp))).json()).rows.filter(r => r.status === 'done' && !r.heavy_deleted);
    sel.innerHTML = ''; for (const r of rows) sel.add(new Option(r.run + ' ' + (r.label || ''), r.run)); sel.value = runBsel; if (!sel.value) { sel.add(new Option(runBsel, runBsel)); sel.value = runBsel; } } catch (e) {}
}
const redraw = () => { draw('A', runA, $('#fA').value); draw('B', runBsel, $('#fB').value); };
$('#fA').onchange = () => draw('A', runA, $('#fA').value); $('#fB').onchange = () => draw('B', runBsel, $('#fB').value);
$('#rB').onchange = () => { runBsel = $('#rB').value; views.B.__shown = false; draw('B', runBsel, $('#fB').value); info(); }; $('#clip').onchange = redraw;
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

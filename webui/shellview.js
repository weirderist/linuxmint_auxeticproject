import { createViewer, syncCameras, loadTubeBase, tubeMesh, legendHTML } from '/viewer_core.js';
// Shell viewer: the folded sheet of one run in 3D, coloured by a membrane field, with a second view (another field of the same run, or another run) side by side.
const $ = s => document.querySelector(s);
const esc = s => String(s == null ? '' : s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const P = new URLSearchParams(location.search), camp = P.get('c'), runA = P.get('run'), runB0 = P.get('b');
const ok = s => /^[0-9A-Za-z_-]{1,64}$/.test(s || ''), okRun = s => /^S[0-9]{6}$/.test(s || '');
if (!ok(camp) || !okRun(runA)) { $('#err').textContent = 'Open this page from a run: /shellview?c=<campaign>&run=S000001[&b=S000002]'; throw new Error('no run'); }
$('#rname').textContent = camp + ' / ' + runA; $('#back').href = '/shell';
const FIELDS = [['area_strain', 'membrane area strain'], ['stretch_max', 'largest principal stretch - 1'], ['stretch_min', 'smallest principal stretch - 1'], ['z', 'height z (mm)'], ['tilt', 'surface tilt (degrees)']];
for (const sel of [$('#fA'), $('#fB')]) for (const [k, t] of FIELDS) sel.add(new Option(t, k));
$('#fA').value = P.get('f') || 'area_strain'; $('#fB').value = P.get('g') || 'z';
const url = r => '/api/shell/' + encodeURIComponent(camp) + '/runs/' + encodeURIComponent(r);
async function colours(r, field, clip) {
  const resp = await fetch(url(r) + '/colors?field=' + field + '&clip=' + clip); if (!resp.ok) { const e = await resp.json().catch(() => ({})); throw new Error(e.detail || 'HTTP ' + resp.status); }
  return { rgb: new Uint8Array(await resp.arrayBuffer()), legend: JSON.parse(resp.headers.get('X-Legend')) };
}
const views = { A: createViewer($('#vA')), B: createViewer($('#vB')) }; syncCameras([views.A, views.B], () => $('#sync').checked);
const bases = {}; let runBsel = runB0 || runA;
async function base(r, st) { if (!bases[r]) bases[r] = await loadTubeBase(null, 'baseline', t => st.textContent = t, url(r) + '/mesh'); return bases[r]; }
async function draw(which, r, field) {
  const st = $('#s' + which), lg = $('#l' + which), v = views[which]; st.textContent = 'loading…';
  try {
    const b = await base(r, st), c = await colours(r, field, +$('#clip').value || 98), first = !v.__shown; v.show(tubeMesh(b, c.rgb), first); v.__shown = true;
    lg.innerHTML = legendHTML(c.legend); lg.hidden = false; st.textContent = '';
    $('#t' + which).textContent = (which === 'A' ? 'Left: ' : 'Right: ') + r + ' - ' + c.legend.label;
  } catch (e) { st.textContent = e.message; lg.hidden = true; }
}
async function info() {
  try {
    const rows = [await (await fetch(url(runA))).json()]; if (runBsel !== runA) rows.push(await (await fetch(url(runBsel))).json());
    $('#info').innerHTML = rows.map(d => { const s = d.settings || {}, m = d.metrics || {}, sd = d.seed || {};
      return '<div style="margin-bottom:.5rem"><b>' + esc(d.run) + '</b> ' + esc(d.label || '') + ' <span class="mut">(' + esc(d.source) + ')</span>: ' + (s.matrix ? s.matrix.length + ' x ' + s.matrix[0].length + ' per tile, ' : '') + 'tiles ' + esc(s.tiles) + ', CPI ' + esc(s.cpi) + ', WPI ' + esc(s.wpi) + ' (R ' + (s.cpi / s.wpi).toFixed(3) + '); plan ' + (+m.plan_x).toFixed(1) + ' x ' + (+m.plan_y).toFixed(1) + ' mm, thickness ' + (+m.thickness).toFixed(1) + ' mm, waves ' + esc(m.wave_k) +
        (sd.real_fold ? '<br><span class="mut">Real fold (K\'s report, ' + esc(sd.sample || '') + '): ' + esc(sd.real_fold) + '</span>' : '') + '</div>'; }).join('');
  } catch (e) { $('#info').textContent = ''; }
}
async function runList() {
  try { const d = await (await fetch('/api/shell/' + encodeURIComponent(camp))).json(); const sel = $('#rB'); sel.innerHTML = '';
    for (const r of d.rows.filter(r => r.status === 'done' && !r.heavy_deleted)) sel.add(new Option(r.run + ' ' + (r.label || ''), r.run)); sel.value = runBsel; if (!sel.value) { sel.add(new Option(runBsel, runBsel)); sel.value = runBsel; } } catch (e) {}
}
const redraw = () => { draw('A', runA, $('#fA').value); draw('B', runBsel, $('#fB').value); };
$('#fA').onchange = () => draw('A', runA, $('#fA').value); $('#fB').onchange = () => draw('B', runBsel, $('#fB').value);
$('#rB').onchange = () => { runBsel = $('#rB').value; views.B.__shown = false; draw('B', runBsel, $('#fB').value); info(); }; $('#clip').onchange = redraw;
runList().then(() => { redraw(); info(); });

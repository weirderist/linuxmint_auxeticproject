import { createViewer, syncCameras, loadTubeBase, tubeMesh } from '/viewer_core.js';
// Shell viewer: the folded sheet of one run in 3D, coloured by a membrane field, with a second view (another field of the same run, or another run) side by side. Colours and sizes come from theme.css.
const $ = s => document.querySelector(s);
const esc = s => String(s == null ? '' : s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const P = new URLSearchParams(location.search), camp = P.get('c'), runA = P.get('run'), runB0 = P.get('b');
const ok = s => /^[0-9A-Za-z_-]{1,64}$/.test(s || ''), okRun = s => /^S[0-9]{6}$/.test(s || '');
if (!ok(camp) || !okRun(runA)) { $('#err').textContent = 'Open this page from a run: /shellview?c=<campaign>&run=S000001[&b=S000002]'; throw new Error('no run'); }
$('#rname').textContent = camp + ' / ' + runA; $('#back').href = '/shell';
const FIELDS = [['area_strain', 'membrane area strain'], ['stretch_max', 'largest principal stretch - 1'], ['stretch_min', 'smallest principal stretch - 1'], ['z', 'height z (mm)'], ['tilt', 'surface tilt (degrees)']];
const STRAIN = ['area_strain', 'stretch_max', 'stretch_min'];
for (const sel of [$('#fA'), $('#fB')]) for (const [k, t] of FIELDS) sel.add(new Option(t, k));
$('#fA').value = P.get('f') || 'area_strain'; $('#fB').value = P.get('g') || 'z';
const url = r => '/api/shell/' + encodeURIComponent(camp) + '/runs/' + encodeURIComponent(r);
async function colours(r, field, clip) {
  const resp = await fetch(url(r) + '/colors?field=' + field + '&clip=' + clip); if (!resp.ok) { const e = await resp.json().catch(() => ({})); throw new Error(e.detail || 'HTTP ' + resp.status); }
  return { rgb: new Uint8Array(await resp.arrayBuffer()), legend: JSON.parse(resp.headers.get('X-Legend')) };
}
// Legend: a plain-language first line for the strain fields, then the technical lines (label, colour bar with compression / tension ends, range note) and the data minimum and maximum.
// The only inline style is the colour ramp itself, which the server sends with the data.
function legendHTML(l) {
  const f = x => (+x).toPrecision(4), e = l.end_labels || ['low', 'high'], ramp = l.stops.map(c => 'rgb(' + c.join(',') + ')').join(',');
  return (STRAIN.includes(l.field) ? '<div class="lg-plain">blue = squeezed (compression), red = stretched (tension)</div>' : '') +
    '<div class="lg-title">' + esc(l.label) + '</div><div class="lg-bar" style="background:linear-gradient(to right,' + ramp + ')"></div>' +
    '<div class="lg-ticks"><span>' + f(l.vmin) + ' (' + esc(e[0]) + ')</span><span>' + f(l.ticks[1]) + '</span><span>' + f(l.vmax) + ' (' + esc(e[1]) + ')</span></div>' +
    (l.note ? '<div class="lg-note">' + esc(l.note) + '</div>' : '') + '<div class="lg-data">data minimum ' + f(l.data_min) + ', maximum ' + f(l.data_max) + '</div>';
}
const views = { A: createViewer($('#vA')), B: createViewer($('#vB')) }; syncCameras([views.A, views.B], () => $('#sync').checked);
for (const v of [views.A, views.B]) { v.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2)); v.resize(); }          // shell meshes are small: render at up to 2x the CSS pixels so the 3D view is crisp on high-density screens
const bases = {}; let runBsel = runB0 || runA;
async function base(r, st) { if (!bases[r]) bases[r] = await loadTubeBase(null, 'baseline', t => st.textContent = t, url(r) + '/mesh'); return bases[r]; }
async function draw(which, r, field) {
  const st = $('#s' + which), lg = $('#l' + which), v = views[which]; st.textContent = 'loading';
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
      return '<p><b>' + esc(d.run) + '</b> ' + esc(d.label || '') + ' <span class="mut">(' + esc(d.source) + ')</span>: ' + (s.matrix ? s.matrix.length + ' x ' + s.matrix[0].length + ' per tile, ' : '') + 'tiles ' + esc(s.tiles) + ', CPI ' + esc(s.cpi) + ', WPI ' + esc(s.wpi) + ' (R ' + (s.cpi / s.wpi).toFixed(3) + '); plan ' + (+m.plan_x).toFixed(1) + ' x ' + (+m.plan_y).toFixed(1) + ' mm, thickness ' + (+m.thickness).toFixed(1) + ' mm, waves ' + esc(m.wave_k) +
        (d.tightness ? '<br>Nominal tightness (display only): S = CPI x WPI = ' + esc(d.tightness.stitch_density == null ? 'n/a' : (+d.tightness.stitch_density).toFixed(1)) + ' stitches per square inch; TF_rel = sqrt(tex x S) = ' + esc(d.tightness.tf_rel == null ? 'n/a' : (+d.tightness.tf_rel).toFixed(2)) + ' tex^0.5 per inch (Ne ' + esc(d.tightness.ne) + ')' + (d.tightness.constant_note ? ' (S and TF_rel: ' + esc(d.tightness.constant_note) + ')' : '') + '; TF_c_rel = sqrt(tex) x CPI = ' + esc(d.tightness.tf_c_rel == null ? 'n/a' : (+d.tightness.tf_c_rel).toFixed(2)) + ', TF_w_rel = sqrt(tex) x WPI = ' + esc(d.tightness.tf_w_rel == null ? 'n/a' : (+d.tightness.tf_w_rel).toFixed(2)) + ' tex^0.5 per inch. <span class="mut">All three are proportional to the tightness factor TF = sqrt(tex)/l (l = loop length): TF_rel from the stitch density, TF_c_rel from the courses (cpi = k_c / l), TF_w_rel from the wales (wpi = k_w / l); the scale constants are UNSOURCED, so TF itself is not shown.</span>' : '') +
        (sd.real_fold ? '<br><span class="mut">Real fold (K\'s report, ' + esc(sd.sample || '') + '): ' + esc(sd.real_fold) + '</span>' : '') + '</p>'; }).join('');
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

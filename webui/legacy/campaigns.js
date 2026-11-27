"use strict";
// Campaigns page. No libraries: tables and inline-SVG charts are built by hand. Everything shown comes from /api/campaigns*, /api/datasets, /api/campaign-help.
const $ = s => document.querySelector(s);
const esc = s => String(s == null ? "" : s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const code = s => esc(s).replace(/`([^`]+)`/g, "<code>$1</code>");
let cur = null, data = null, sortKey = "run", sortDir = 1, selRun = null, busyAction = false;
const SHOW = ["exit_code", "rel_dE_last", "min_dist_over_2R_end", "fold_direction", "fold_amplitude_mm", "fold_local_amplitude_mm", "wall_s"];
const FOLD_Y = "fold_amplitude_mm";
// display-only tightness columns, computed by the server from each run's CPI, WPI and Ne (definitions in the note under the table)
const TIGHT = ["stitch_density", "tf_rel", "tf_c_rel", "tf_w_rel"], HEAD = { stitch_density: "S", tf_rel: "TF_rel", tf_c_rel: "TF_c_rel", tf_w_rel: "TF_w_rel" }, hname = h => HEAD[h] || h;
function tnote() { const t = data.tightness; $("#tnote").innerHTML = t ? "<b>S</b>: " + esc(t.label_stitch_density) + ". <b>TF_rel</b>: " + esc(t.label_tf_rel) + (t.constant_note ? ". S and TF_rel are " + esc(t.constant_note) : "") + ". <b>TF_c_rel</b>: " + esc(t.label_tf_c_rel) + ". <b>TF_w_rel</b>: " + esc(t.label_tf_w_rel) + (t.constant_note ? ". TF_c_rel and TF_w_rel are NOT constant: they change with CPI and WPI separately (in opposite directions at constant S)" : "") + ". " + esc(t.note) + " The yarn-level tightness factor recorded in meta.json (from the as-built loop length) is a separate value." : ""; }

async function api(path, opt) {
  const r = await fetch(path, opt);
  let j = null; try { j = await r.json(); } catch (e) {}
  if (!r.ok) throw new Error((j && j.detail) ? (typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail)) : "HTTP " + r.status);
  return j;
}
function fmt(v, k) {
  if (v == null || v === "") return "";
  if (typeof v === "number") {
    if (k === "wall_s" || k === "build_s") return Math.round(v) + "";
    if (Number.isInteger(v)) return String(v);
    const a = Math.abs(v); return (a !== 0 && (a < 1e-3 || a >= 1e5)) ? v.toExponential(2) : String(+v.toPrecision(4));
  }
  return String(v);
}
function dur(s) {
  if (s == null) return "";
  s = Math.round(s); const h = Math.floor(s / 3600), m = Math.floor(s % 3600 / 60);
  return h ? h + " h " + m + " min" : m ? m + " min" : s + " s";
}
function msg(t, err) { const m = $("#msg"); m.textContent = t ? (err ? "Error: " : "") + t : ""; m.className = err ? "err" : ""; }

// ---------------------------------------------------------------- campaign list and header
async function loadList() {
  const j = await api("/api/campaigns");
  const t = $("#tabs"); t.innerHTML = "";
  $("#empty").hidden = j.campaigns.length > 0;
  if (!j.campaigns.some(c => c.name === cur)) cur = j.campaigns.length ? j.campaigns[0].name : null;
  for (const c of j.campaigns) {
    const b = document.createElement("button"); b.textContent = c.name + " (" + c.state + ")"; b.className = c.name === cur ? "on" : "";
    b.onclick = () => { cur = c.name; selRun = null; msg(""); refresh(); }; t.appendChild(b);
  }
  $("#cur").hidden = !cur;
  return j;
}
function header(ov, list) {
  $("#cname").textContent = ov.name;
  const p = $("#pill"); p.textContent = ov.state; p.className = "pill " + ov.state;
  let d = [];
  if (ov.runner_alive) d.push("runner alive (pid " + ov.runner_pid + ")"); else if (ov.job_alive) d.push("a run is still working without a runner");
  if (ov.stopping) d.push("pause requested; the run in progress is finishing");
  else if (ov.state === "paused" && !ov.runner_alive) d.push("stopped with runs still queued");
  if (ov.complete && ov.planned) d.push("no runs left");
  if (list.web_job_running) d.push("a simulation started from the web page is running");
  $("#pdetail").textContent = d.join("; ");
  const canStart = ov.planned && ov.can_start && !ov.runner_alive && !ov.job_alive && !list.busy && !list.web_job_running && !busyAction;
  const canStop = (ov.runner_alive || ov.job_alive) && !ov.stop_requested && !busyAction;
  $("#bStart").disabled = !canStart; $("#bStart").textContent = ov.state === "paused" ? "Resume" : "Start";
  $("#bStop").disabled = !canStop;
  $("#bStart").title = canStart ? "" : (!ov.planned ? ov.message : list.busy || (ov.runner_alive ? "already running" : ov.can_start ? "" : "no runs left"));
  if (!ov.planned) msg(ov.message, false);
  const c = ov.counts || {}, n = ov.n_runs || 0, bar = $("#bar"); bar.innerHTML = "";
  const seg = k => { const w = n ? (c[k] || 0) / n * 100 : 0; if (w) { const i = document.createElement("i"); i.style.width = w + "%"; i.className = k; i.title = k + " " + c[k]; bar.appendChild(i); } };          // colours come from theme.css (.bar i.done, .running, .failed)
  seg("done"); seg("failed"); seg("running");
  $("#barTxt").textContent = n ? [(c.done || 0) + " of " + n + " done"].concat(["failed", "running", "queued"].map(k => (c[k] || 0) + " " + k)).join(" · ") : "";
}
function etaText(e) {
  if (!e) return "";
  if (e.estimate_s == null) return e.note;
  return "Estimated time left: about " + dur(e.estimate_s) + "  (estimate). " + e.note;
}

// ---------------------------------------------------------------- table
function sortedRuns(runs) {
  const k = sortKey, f = sortDir;
  return runs.slice().sort((a, b) => {
    const x = a[k], y = b[k];
    if (x == null && y == null) return 0; if (x == null) return 1; if (y == null) return -1;
    return (typeof x === "number" && typeof y === "number" ? x - y : String(x).localeCompare(String(y))) * f;
  });
}
function flagTags(f) {
  let s = "";
  if (f.failed) s += '<span class="tag fail">failed</span>';
  if (f.not_settled) s += '<span class="tag ns">not settled</span>';
  if (f.interpenetration) s += '<span class="tag ip">interpenetration</span>';
  if (f.no_fold) s += '<span class="tag nf">no fold</span>';
  return s;
}
function table() {
  const cols = ["run", "status"].concat(data.params, TIGHT, SHOW), runs = sortedRuns(data.runs);
  let h = "<thead><tr>" + cols.map(c => '<th data-k="' + esc(c) + '"' + (c === "fold_direction" || c === "status" ? ' class="l"' : "") + ">" + esc(hname(c)) + (c === sortKey ? (sortDir > 0 ? " ▲" : " ▼") : "") + "</th>").join("") + '<th class="l" data-k="">flags</th></tr></thead><tbody>';
  for (const r of runs) {
    const f = r.flags || {};
    h += '<tr data-run="' + esc(r.run) + '"' + (r.run === selRun ? ' class="sel"' : "") + ">";
    for (const c of cols) {
      const cl = c === "fold_direction" || c === "status" ? "l" : "";          // no coloured cells: the flags column carries the flags as grey pills
      h += '<td class="' + cl + '">' + esc(fmt(r[c], c)) + "</td>";
    }
    h += '<td class="l">' + flagTags(f) + "</td></tr>";
  }
  const t = $("#tbl"); t.innerHTML = h + "</tbody>";
  t.querySelectorAll("th[data-k]").forEach(th => th.onclick = () => { const k = th.dataset.k; if (!k) return; if (k === sortKey) sortDir = -sortDir; else { sortKey = k; sortDir = 1; } table(); });
  t.querySelectorAll("tbody tr").forEach(tr => tr.onclick = () => openRun(tr.dataset.run)); tnote();
}
async function openRun(id) {
  selRun = id; table();
  const el = $("#detail");
  try {
    const j = await api("/api/campaigns/" + encodeURIComponent(cur) + "/runs/" + encodeURIComponent(id));
    const kv = o => "<dl>" + Object.keys(o).map(k => "<dt>" + esc(k) + "</dt><dd>" + esc(fmt(o[k], k)) + "</dd>").join("") + "</dl>";
    el.innerHTML = '<div class="card mt"><h2>Run ' + esc(j.run) + " &ndash; " + esc(j.status) + (j.stage ? " (stage " + esc(j.stage) + ")" : "") + " " + flagTags(j.flags || {}) + "</h2>" +
      (j.reason ? '<div class="small"><b>Reason:</b> ' + esc(j.reason) + "</div>" : "") +
      '<div class="small links">' + [["/?run=", "Open in viewer"], ["/strain?run=", "Strain page"], ["/run?run=", "Run graphs"]].map(l => '<a href="' + l[0] + encodeURIComponent(j.campaign) + "/" + encodeURIComponent(j.run) + '" target="_blank" rel="noopener">' + l[1] + "</a>").join(" &middot; ") + "</div>" +
      '<div class="small mut">attempts ' + j.attempts + ", interruptions " + j.interrupted + ", exit code " + esc(j.exit_code == null ? "-" : j.exit_code) + ", wall time " + esc(j.wall_s == null ? "-" : dur(j.wall_s)) + "</div>" +
      '<div class="row"><div class="col"><b>Parameters</b>' + kv(j.params) + (j.nominal ? '<b>Nominal tightness (display only)</b>' + kv({ "S (stitches per square inch)": j.nominal.stitch_density, "TF_rel (tex^0.5 per inch)": j.nominal.tf_rel, "TF_c_rel = sqrt(tex) x CPI": j.nominal.tf_c_rel, "TF_w_rel = sqrt(tex) x WPI": j.nominal.tf_w_rel }) : "") + '</div><div class="col col2"><b>Metrics</b>' + kv(j.metrics) + "</div></div>" +
      "<b>Last 40 lines of relax.log</b><pre>" + esc(j.log_tail || "(no log yet)") + "</pre></div>";
    el.scrollIntoView({ block: "nearest", behavior: "smooth" });
  } catch (e) { el.innerHTML = '<div class="card mt"><b>Error:</b> ' + esc(e.message) + "</div>"; }
}

// ---------------------------------------------------------------- charts (drawn by chartkit.js: the same size, aspect ratio, ticks, fonts and marker size for every chart; colours come from theme.css classes)
const CLS = { ok: { n: "finished, settled", shape: "circle" }, ns: { n: "did not settle", shape: "diamond" }, ip: { n: "closest approach below 0.97", shape: "triangle" }, fail: { n: "failed", shape: "cross" } };
function cls(r) { const f = r.flags || {}; return f.failed ? "fail" : f.interpenetration ? "ip" : f.not_settled ? "ns" : "ok"; }
function pt(r, x, y, tip) { const k = cls(r); return { x, y, cls: "cl-" + k, shape: CLS[k].shape, tip }; }
function charts() {
  $("#cleg").innerHTML = Object.keys(CLS).map(k => '<span class="cl-' + k + '">' + esc(CLS[k].n) + "</span>").join("");
  let vary = data.varied.length ? data.varied : data.params.map(p => ({ name: p, scale: "linear" })), S = ChartKit.size($("#order"));
  if (data.tightness) vary = vary.concat(TIGHT.filter(k => data.tightness.constant === false || k === "tf_c_rel" || k === "tf_w_rel").map(k => ({ name: k, scale: "linear" })));          // S and TF_rel only when S differs between runs (otherwise every point sits at one x); TF_c_rel and TF_w_rel always (they change with CPI and WPI separately)
  let h = "";
  for (const v of vary) {
    const pts = data.runs.filter(r => r[v.name] != null && r[FOLD_Y] != null && (v.scale !== "log" || r[v.name] > 0)).map(r => pt(r, r[v.name], r[FOLD_Y], "run " + r.run + ": " + v.name + " " + fmt(r[v.name]) + ", " + FOLD_Y + " " + fmt(r[FOLD_Y])));
    h += '<div class="card chart"><div class="card-h"><span class="card-t">' + esc(FOLD_Y) + " against " + esc(hname(v.name)) + "</span></div>" + ChartKit.scatter(pts, { xlog: v.scale === "log", xlabel: hname(v.name), ylabel: "fold amplitude (mm)" }, S) + "</div>";
  }
  $("#scat").innerHTML = h;
  const sel = $("#ymet"), keep = sel.value || "min_dist_over_2R_end";
  if (!sel.options.length) {
    const nums = data.metrics.filter(m => !["relax_state", "settled_by", "fold_direction", "reason"].includes(m));
    sel.innerHTML = nums.map(m => '<option value="' + esc(m) + '">' + esc(m) + "</option>").join(""); sel.value = nums.includes(keep) ? keep : nums[0]; sel.onchange = orderChart;
  }
  orderChart();
}
function orderChart() {
  const m = $("#ymet").value;
  const pts = data.runs.filter(r => r[m] != null).map(r => pt(r, +r.run, r[m], "run " + r.run + ": " + m + " " + fmt(r[m])));
  $("#order").innerHTML = ChartKit.scatter(pts, { xlabel: "run order (run number)", ylabel: m, line: true }, ChartKit.size($("#order")));
}
let resizeTimer = null;
window.addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(() => { if (data) charts(); }, 150); });

// ---------------------------------------------------------------- datasets, help
async function loadDatasets() {
  try {
    const d = await api("/api/datasets");
    const msgs = (n, word, items) => n ? '<details class="msgs"><summary>' + n + " " + word + (n === 1 ? "" : "s") + "</summary><ul>" + items.map(m => "<li>" + esc(m) + "</li>").join("") + "</ul></details>" : "";
    const pill = s => s ? '<span class="pill-s">' + esc(s) + "</span>" : "";
    $("#dst").innerHTML = '<colgroup><col class="c1"><col class="c2"><col class="c3"><col class="c4"><col class="c5"><col class="c6"></colgroup>' +
      '<thead><tr><th>dataset</th><th>kind</th><th>role</th><th>provenance</th><th class="r">rows</th><th>checker</th></tr></thead><tbody>' +
      d.map(x => '<tr><td><div class="dsname">' + esc(x.id) + "</div><div>" + esc(x.fabric || "") + "</div></td><td>" + pill(x.kind) + "</td><td>" + pill(x.role) + "</td><td>" + pill(x.provenance || "not stated") + (x.source ? '<div class="mut small">' + esc(x.source) + "</div>" : "") +
        '</td><td class="r">' + esc(x.rows == null ? "?" : x.rows) + (x.usable != null ? '<div class="mut small">' + x.usable + " usable</div>" : "") + "</td><td>" +
        (msgs(x.errors.length, "error", x.errors) + msgs(x.warnings.length, "warning", x.warnings) || '<span class="mut small">none</span>') + "</td></tr>").join("") + "</tbody>";
  } catch (e) { $("#dst").innerHTML = "<tr><td>" + esc(e.message) + "</td></tr>"; }
}
async function loadHelp() {
  try {
    const j = await api("/api/campaign-help");
    $("#help").innerHTML = "<p>" + code(j.intro) + '</p><div class="tw tall"><table class="cols"><tbody>' + j.rows.map(r => "<tr><td>" + code(r[0]) + "</td><td>" + code(r[1]) + "</td></tr>").join("") + "</tbody></table></div><p>" + code(j.caveats) + "</p>";
  } catch (e) { $("#help").textContent = e.message; }
}

// ---------------------------------------------------------------- actions and refresh
async function act(kind) {
  if (busyAction || !cur) return; busyAction = true; msg("Working…");
  try {
    const j = await api("/api/campaigns/" + encodeURIComponent(cur) + "/" + kind, { method: "POST" });
    msg(j.message || "Done.", false);
  } catch (e) { msg(e.message, true); }
  busyAction = false; setTimeout(refresh, 1500); refresh();
}
$("#bStart").onclick = () => { if (confirm("Start campaign '" + cur + "'? Finished runs are not repeated.")) act("start"); };
$("#bStop").onclick = () => { if (confirm("Pause campaign '" + cur + "'? The run in progress will finish first; nothing is killed.")) act("stop"); };
async function refresh() {
  try {
    const list = await loadList(); if (!cur) return;
    const ov = list.campaigns.find(c => c.name === cur);
    header(ov, list);
    if (!ov.planned) { $("#sum").innerHTML = ""; $("#tbl").innerHTML = ""; $("#scat").innerHTML = ""; $("#order").innerHTML = ""; $("#eta").textContent = ""; return; }
    data = await api("/api/campaigns/" + encodeURIComponent(cur));
    header(data.overview, list);
    $("#eta").textContent = etaText(data.eta);
    $("#sum").innerHTML = data.summary.map(s => '<li class="' + esc(s.level) + '"><span>' + (s.level === "warn" ? "<b>Warning.</b>" : "<b>Note.</b>") + " " + esc(s.text) + "</span></li>").join("");
    table(); charts();
  } catch (e) { msg(e.message, true); }
}
loadHelp(); loadDatasets(); refresh(); setInterval(() => { if (!document.hidden) refresh(); }, 20000);


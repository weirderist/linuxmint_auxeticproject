"use strict";
// Campaigns page. No libraries: tables and inline-SVG charts are built by hand. Everything shown comes from /api/campaigns*, /api/datasets, /api/campaign-help.
const $ = s => document.querySelector(s);
const esc = s => String(s == null ? "" : s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const code = s => esc(s).replace(/`([^`]+)`/g, "<code>$1</code>");
let cur = null, data = null, sortKey = "run", sortDir = 1, selRun = null, busyAction = false;
const SHOW = ["exit_code", "rel_dE_last", "min_dist_over_2R_end", "fold_direction", "fold_amplitude_mm", "fold_local_amplitude_mm", "wall_s"];
const FOLD_Y = "fold_amplitude_mm";

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
  const cols = ["run", "status"].concat(data.params, SHOW), runs = sortedRuns(data.runs);
  let h = "<thead><tr>" + cols.map(c => '<th data-k="' + esc(c) + '"' + (c === "fold_direction" || c === "status" ? ' class="l"' : "") + ">" + esc(c) + (c === sortKey ? (sortDir > 0 ? " ▲" : " ▼") : "") + "</th>").join("") + '<th class="l" data-k="">flags</th></tr></thead><tbody>';
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
  t.querySelectorAll("tbody tr").forEach(tr => tr.onclick = () => openRun(tr.dataset.run));
}
async function openRun(id) {
  selRun = id; table();
  const el = $("#detail");
  try {
    const j = await api("/api/campaigns/" + encodeURIComponent(cur) + "/runs/" + encodeURIComponent(id));
    const kv = o => "<dl>" + Object.keys(o).map(k => "<dt>" + esc(k) + "</dt><dd>" + esc(fmt(o[k], k)) + "</dd>").join("") + "</dl>";
    el.innerHTML = '<div class="card" style="margin-top:var(--gap)"><h2>Run ' + esc(j.run) + " &ndash; " + esc(j.status) + (j.stage ? " (stage " + esc(j.stage) + ")" : "") + " " + flagTags(j.flags || {}) + "</h2>" +
      (j.reason ? '<div class="small"><b>Reason:</b> ' + esc(j.reason) + "</div>" : "") +
      '<div class="small" style="margin:.3rem 0">' + [["/?run=", "Open in viewer"], ["/strain?run=", "Strain page"], ["/run?run=", "Run graphs"]].map(l => '<a href="' + l[0] + encodeURIComponent(j.campaign) + "/" + encodeURIComponent(j.run) + '" target="_blank" rel="noopener">' + l[1] + "</a>").join(" &middot; ") + "</div>" +
      '<div class="small mut">attempts ' + j.attempts + ", interruptions " + j.interrupted + ", exit code " + esc(j.exit_code == null ? "-" : j.exit_code) + ", wall time " + esc(j.wall_s == null ? "-" : dur(j.wall_s)) + "</div>" +
      '<div class="row"><div style="flex:1;min-width:240px"><b>Parameters</b>' + kv(j.params) + '</div><div style="flex:2;min-width:300px"><b>Metrics</b>' + kv(j.metrics) + "</div></div>" +
      "<b>Last 40 lines of relax.log</b><pre>" + esc(j.log_tail || "(no log yet)") + "</pre></div>";
    el.scrollIntoView({ block: "nearest", behavior: "smooth" });
  } catch (e) { el.innerHTML = '<div class="card" style="margin-top:var(--gap)"><b>Error:</b> ' + esc(e.message) + "</div>"; }
}

// ---------------------------------------------------------------- charts (inline SVG)
// Every chart is drawn at the measured width of its grid column with one fixed aspect ratio, the same margins, tick counts, marker size and 11 px text (theme.css), so all charts are the same size.
// Colours are CSS variables from theme.css; the run class is also shown by marker shape (circle, diamond, triangle, cross).
const CLS = { ok: { c: "var(--chart-1)", n: "finished, settled" }, ns: { c: "var(--chart-2)", n: "did not settle" }, ip: { c: "var(--chart-3)", n: "closest approach below 0.97" }, fail: { c: "var(--chart-4)", n: "failed" } };
const CH = { aspect: 0.58, L: 54, R: 14, T: 12, B: 40, r: 4, yTicks: 5, xTicks: 5, minW: 240, fallbackW: 560 };
function cls(r) { const f = r.flags || {}; return f.failed ? "fail" : f.interpenetration ? "ip" : f.not_settled ? "ns" : "ok"; }
function mark(cl, x, y, tip) {
  const fill = "fill:" + CLS[cl].c + ";stroke:var(--bg);stroke-width:1.5", r = CH.r, t = "<title>" + esc(tip) + "</title>";
  if (cl === "ns") return '<path d="M' + x + " " + (y - r - 1) + "l" + (r + 1) + " " + (r + 1) + "l-" + (r + 1) + " " + (r + 1) + "l-" + (r + 1) + " -" + (r + 1) + 'z" style="' + fill + '">' + t + "</path>";
  if (cl === "ip") return '<path d="M' + x + " " + (y - r - 1) + "l" + (r + 1) + " " + (2 * r) + "l-" + (2 * r + 2) + ' 0z" style="' + fill + '">' + t + "</path>";
  if (cl === "fail") return '<path d="M' + (x - r) + " " + (y - r) + "l" + (2 * r) + " " + (2 * r) + "m0 -" + (2 * r) + "l-" + (2 * r) + " " + (2 * r) + '" style="fill:none;stroke:' + CLS[cl].c + ';stroke-width:2">' + t + "</path>";
  return '<circle cx="' + x + '" cy="' + y + '" r="' + r + '" style="' + fill + '">' + t + "</circle>";
}
function ticks(lo, hi, n) {
  if (!(hi > lo)) return [lo];
  const raw = (hi - lo) / n, p = Math.pow(10, Math.floor(Math.log10(raw))), m = raw / p, st = (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * p, out = [];
  for (let v = Math.ceil(lo / st) * st; v <= hi + st * 1e-9; v += st) out.push(+v.toPrecision(10));
  return out;
}
function chartSize() { const w = Math.floor($("#order").clientWidth) || CH.fallbackW; const W = Math.max(CH.minW, w); return { W, H: Math.round(W * CH.aspect) }; }
function chart(pts, o, S) {        // pts: [{x,y,cl,tip}], o: {xlog, xlabel, ylabel, line}, S: {W,H}
  const { W, H } = S, L = CH.L, R = CH.R, T = CH.T, B = CH.B;
  if (!pts.length) return '<div class="nodata" style="width:' + W + "px;height:" + H + 'px">no data yet for ' + esc(o.xlabel) + "</div>";
  const fx = o.xlog ? Math.log : (v => v);
  let xs = pts.map(p => fx(p.x)), ys = pts.map(p => p.y);
  let x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(0, ...ys), y1 = Math.max(...ys);
  if (x1 === x0) { x0 -= 1; x1 += 1; } if (y1 === y0) y1 = y0 + 1;
  const pad = (x1 - x0) * 0.05; x0 -= pad; x1 += pad; y1 += (y1 - y0) * 0.06;
  const X = v => L + (fx(v) - x0) / (x1 - x0) * (W - L - R), Y = v => H - B - (v - y0) / (y1 - y0) * (H - T - B);
  let s = '<svg width="' + W + '" height="' + H + '" viewBox="0 0 ' + W + " " + H + '" role="img" aria-label="' + esc(o.ylabel + " against " + o.xlabel) + '">';
  for (const v of ticks(y0, y1, CH.yTicks)) s += '<line class="gd" x1="' + L + '" x2="' + (W - R) + '" y1="' + Y(v).toFixed(1) + '" y2="' + Y(v).toFixed(1) + '"/><text x="' + (L - 6) + '" y="' + (Y(v) + 4).toFixed(1) + '" text-anchor="end">' + esc(fmt(v)) + "</text>";
  let xt;
  if (o.xlog) { const a = Math.exp(x0), b = Math.exp(x1); xt = [a, Math.sqrt(a * b), b]; } else xt = ticks(x0, x1, CH.xTicks);
  for (const v of xt) s += '<line class="ax" x1="' + X(v).toFixed(1) + '" x2="' + X(v).toFixed(1) + '" y1="' + (H - B) + '" y2="' + (H - B + 4) + '"/><text x="' + X(v).toFixed(1) + '" y="' + (H - B + 17) + '" text-anchor="middle">' + esc(fmt(+v.toPrecision(3))) + "</text>";
  s += '<line class="ax" x1="' + L + '" x2="' + (W - R) + '" y1="' + (H - B) + '" y2="' + (H - B) + '"/><line class="ax" x1="' + L + '" x2="' + L + '" y1="' + T + '" y2="' + (H - B) + '"/>';
  s += '<text x="' + (L + (W - L - R) / 2).toFixed(1) + '" y="' + (H - 6) + '" text-anchor="middle">' + esc(o.xlabel + (o.xlog ? " (log scale)" : "")) + "</text>";
  s += '<text x="12" y="' + (T + (H - T - B) / 2).toFixed(1) + '" text-anchor="middle" transform="rotate(-90 12 ' + (T + (H - T - B) / 2).toFixed(1) + ')">' + esc(o.ylabel) + "</text>";
  if (o.line) { const q = pts.slice().sort((a, b) => a.x - b.x); s += '<polyline style="fill:none;stroke:var(--chart-line);stroke-width:1.5;opacity:.6" points="' + q.map(p => X(p.x).toFixed(1) + "," + Y(p.y).toFixed(1)).join(" ") + '"/>'; }
  for (const p of pts) s += mark(p.cl, +X(p.x).toFixed(1), +Y(p.y).toFixed(1), p.tip);
  return s + "</svg>";
}
function charts() {
  $("#cleg").innerHTML = Object.keys(CLS).map(k => '<span style="--c:' + CLS[k].c + '">' + esc(CLS[k].n) + "</span>").join("");
  const vary = data.varied.length ? data.varied : data.params.map(p => ({ name: p, scale: "linear" })), S = chartSize();
  let h = "";
  for (const v of vary) {
    const pts = data.runs.filter(r => r[v.name] != null && r[FOLD_Y] != null && (v.scale !== "log" || r[v.name] > 0)).map(r => ({ x: r[v.name], y: r[FOLD_Y], cl: cls(r), tip: "run " + r.run + ": " + v.name + " " + fmt(r[v.name]) + ", " + FOLD_Y + " " + fmt(r[FOLD_Y]) }));
    h += '<div class="card chart"><div class="card-h"><span class="card-t">' + esc(FOLD_Y) + " against " + esc(v.name) + "</span></div>" + chart(pts, { xlog: v.scale === "log", xlabel: v.name, ylabel: "fold amplitude (mm)" }, S) + "</div>";
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
  const pts = data.runs.filter(r => r[m] != null).map(r => ({ x: +r.run, y: r[m], cl: cls(r), tip: "run " + r.run + ": " + m + " " + fmt(r[m]) }));
  $("#order").innerHTML = chart(pts, { xlabel: "run order (run number)", ylabel: m, line: true }, chartSize());
}
let resizeTimer = null;
window.addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(() => { if (data) charts(); }, 150); });

// ---------------------------------------------------------------- datasets, help
async function loadDatasets() {
  try {
    const d = await api("/api/datasets");
    const msgs = (n, word, items) => n ? '<details class="msgs"><summary>' + n + " " + word + (n === 1 ? "" : "s") + "</summary><ul>" + items.map(m => "<li>" + esc(m) + "</li>").join("") + "</ul></details>" : "";
    const pill = s => s ? '<span class="pill-s">' + esc(s) + "</span>" : "";
    $("#dst").innerHTML = '<colgroup><col style="width:24%"><col style="width:11%"><col style="width:13%"><col style="width:22%"><col style="width:11%"><col style="width:19%"></colgroup>' +
      '<thead><tr><th>dataset</th><th>kind</th><th>role</th><th>provenance</th><th class="r">rows</th><th>checker</th></tr></thead><tbody>' +
      d.map(x => '<tr><td><div class="dsname">' + esc(x.id) + "</div><div>" + esc(x.fabric || "") + "</div></td><td>" + pill(x.kind) + "</td><td>" + pill(x.role) + "</td><td>" + pill(x.provenance || "not stated") + (x.source ? '<div class="mut small">' + esc(x.source) + "</div>" : "") +
        '</td><td class="r">' + esc(x.rows == null ? "?" : x.rows) + (x.usable != null ? '<div class="mut small">' + x.usable + " usable</div>" : "") + "</td><td>" +
        (msgs(x.errors.length, "error", x.errors) + msgs(x.warnings.length, "warning", x.warnings) || '<span class="mut small">none</span>') + "</td></tr>").join("") + "</tbody>";
  } catch (e) { $("#dst").innerHTML = "<tr><td>" + esc(e.message) + "</td></tr>"; }
}
async function loadHelp() {
  try {
    const j = await api("/api/campaign-help");
    $("#help").innerHTML = "<p>" + code(j.intro) + '</p><div class="tw" style="max-height:none"><table class="cols"><tbody>' + j.rows.map(r => "<tr><td>" + code(r[0]) + "</td><td>" + code(r[1]) + "</td></tr>").join("") + "</tbody></table></div><p>" + code(j.caveats) + "</p>";
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


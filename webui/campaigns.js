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
function msg(t, err) { const m = $("#msg"); m.textContent = t || ""; m.className = err ? "err" : ""; }

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
  const seg = (k, col) => { const w = n ? (c[k] || 0) / n * 100 : 0; if (w) { const i = document.createElement("i"); i.style.width = w + "%"; i.style.background = col; i.title = k + " " + c[k]; bar.appendChild(i); } };
  seg("done", "var(--ok)"); seg("failed", "var(--crit)"); seg("running", "var(--s1)");
  $("#barTxt").textContent = n ? ["done", "failed", "running", "queued"].map(k => (c[k] || 0) + " " + k).join(", ") + " of " + n + " runs" : "";
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
      let cl = c === "fold_direction" || c === "status" ? "l" : "";
      if (c === "exit_code" && f.not_settled) cl += " f-ns";
      if (c === "min_dist_over_2R_end" && f.interpenetration) cl += " f-ip";
      if (c === "fold_direction" && f.no_fold) cl += " f-nf";
      if (c === "status" && f.failed) cl += " f-fail";
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
    el.innerHTML = '<div class="card" style="margin-top:.6rem"><h2>Run ' + esc(j.run) + " &ndash; " + esc(j.status) + (j.stage ? " (stage " + esc(j.stage) + ")" : "") + " " + flagTags(j.flags || {}) + "</h2>" +
      (j.reason ? '<div class="small" style="color:var(--crit)">Reason: ' + esc(j.reason) + "</div>" : "") +
      '<div class="small" style="margin:.3rem 0">' + [["/?run=", "Open in viewer"], ["/strain?run=", "Strain page"], ["/run?run=", "Run graphs"]].map(l => '<a href="' + l[0] + encodeURIComponent(j.campaign) + "/" + encodeURIComponent(j.run) + '" target="_blank" rel="noopener">' + l[1] + "</a>").join(" &middot; ") + "</div>" +
      '<div class="small mut">attempts ' + j.attempts + ", interruptions " + j.interrupted + ", exit code " + esc(j.exit_code == null ? "-" : j.exit_code) + ", wall time " + esc(j.wall_s == null ? "-" : dur(j.wall_s)) + "</div>" +
      '<div class="row"><div style="flex:1;min-width:240px"><b>Parameters</b>' + kv(j.params) + '</div><div style="flex:2;min-width:300px"><b>Metrics</b>' + kv(j.metrics) + "</div></div>" +
      "<b>Last 40 lines of relax.log</b><pre>" + esc(j.log_tail || "(no log yet)") + "</pre></div>";
    el.scrollIntoView({ block: "nearest", behavior: "smooth" });
  } catch (e) { el.innerHTML = '<div class="card err" style="color:var(--crit)">' + esc(e.message) + "</div>"; }
}

// ---------------------------------------------------------------- charts (inline SVG)
const CLS = { ok: { c: "var(--s1)", n: "finished, settled" }, ns: { c: "var(--warn)", n: "did not settle" }, ip: { c: "var(--crit)", n: "closest approach below 0.97" }, fail: { c: "var(--gray)", n: "failed" } };
function cls(r) { const f = r.flags || {}; return f.failed ? "fail" : f.interpenetration ? "ip" : f.not_settled ? "ns" : "ok"; }
function mark(cl, x, y, tip) {          // shape as well as colour: circle, diamond, triangle, cross
  const c = CLS[cl].c, t = "<title>" + esc(tip) + "</title>";
  if (cl === "ns") return '<path d="M' + x + " " + (y - 5) + "l5 5l-5 5l-5 -5z" + '" fill="' + c + '" stroke="var(--bg)" stroke-width="1.5">' + t + "</path>";
  if (cl === "ip") return '<path d="M' + x + " " + (y - 5.5) + "l5.5 10l-11 0z" + '" fill="' + c + '" stroke="var(--bg)" stroke-width="1.5">' + t + "</path>";
  if (cl === "fail") return '<path d="M' + (x - 4) + " " + (y - 4) + "l8 8m0 -8l-8 8" + '" stroke="' + c + '" stroke-width="2" fill="none">' + t + "</path>";
  return '<circle cx="' + x + '" cy="' + y + '" r="4.5" fill="' + c + '" stroke="var(--bg)" stroke-width="1.5">' + t + "</circle>";
}
function ticks(lo, hi, n) {
  if (!(hi > lo)) return [lo];
  const raw = (hi - lo) / n, p = Math.pow(10, Math.floor(Math.log10(raw))), m = raw / p, st = (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * p, out = [];
  for (let v = Math.ceil(lo / st) * st; v <= hi + st * 1e-9; v += st) out.push(+v.toPrecision(10));
  return out;
}
function chart(pts, o) {        // pts: [{x,y,cl,tip}], o: {xlog, xlabel, ylabel, line}
  const W = 320, H = 210, L = 44, R = 10, T = 10, B = 36;
  if (!pts.length) return '<div class="small mut" style="padding:1rem">no data yet for ' + esc(o.xlabel) + "</div>";
  const fx = o.xlog ? Math.log : (v => v);
  let xs = pts.map(p => fx(p.x)), ys = pts.map(p => p.y);
  let x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(0, ...ys), y1 = Math.max(...ys);
  if (x1 === x0) { x0 -= 1; x1 += 1; } if (y1 === y0) y1 = y0 + 1;
  const pad = (x1 - x0) * 0.05; x0 -= pad; x1 += pad; y1 += (y1 - y0) * 0.06;
  const X = v => L + (fx(v) - x0) / (x1 - x0) * (W - L - R), Y = v => H - B - (v - y0) / (y1 - y0) * (H - T - B);
  let s = '<svg viewBox="0 0 ' + W + " " + H + '" role="img" aria-label="' + esc(o.ylabel + " against " + o.xlabel) + '">';
  for (const v of ticks(y0, y1, 4)) s += '<line class="gd" x1="' + L + '" x2="' + (W - R) + '" y1="' + Y(v) + '" y2="' + Y(v) + '"/><text x="' + (L - 4) + '" y="' + (Y(v) + 3) + '" text-anchor="end">' + esc(fmt(v)) + "</text>";
  let xt;
  if (o.xlog) { const a = Math.exp(x0), b = Math.exp(x1); xt = [a, Math.sqrt(a * b), b]; } else xt = ticks(x0, x1, 4);
  for (const v of xt) s += '<line class="ax" x1="' + X(v) + '" x2="' + X(v) + '" y1="' + (H - B) + '" y2="' + (H - B + 3) + '"/><text x="' + X(v) + '" y="' + (H - B + 14) + '" text-anchor="middle">' + esc(fmt(+v.toPrecision(3))) + "</text>";
  s += '<line class="ax" x1="' + L + '" x2="' + (W - R) + '" y1="' + (H - B) + '" y2="' + (H - B) + '"/><line class="ax" x1="' + L + '" x2="' + L + '" y1="' + T + '" y2="' + (H - B) + '"/>';
  s += '<text x="' + (L + (W - L - R) / 2) + '" y="' + (H - 4) + '" text-anchor="middle">' + esc(o.xlabel + (o.xlog ? " (log scale)" : "")) + "</text>";
  s += '<text x="10" y="' + (T + (H - T - B) / 2) + '" text-anchor="middle" transform="rotate(-90 10 ' + (T + (H - T - B) / 2) + ')">' + esc(o.ylabel) + "</text>";
  if (o.line) { const q = pts.slice().sort((a, b) => a.x - b.x); s += '<polyline fill="none" stroke="var(--s1)" stroke-width="1.5" opacity=".5" points="' + q.map(p => X(p.x) + "," + Y(p.y)).join(" ") + '"/>'; }
  for (const p of pts) s += mark(p.cl, +X(p.x).toFixed(1), +Y(p.y).toFixed(1), p.tip);
  return s + "</svg>";
}
function charts() {
  const leg = $("#cleg"); leg.innerHTML = '<span style="--c:var(--s1)">finished, settled</span><span style="--c:var(--warn);--r:2px">did not settle</span><span style="--c:var(--crit);--r:2px">closest approach below 0.97</span><span style="--c:var(--gray);--r:2px">failed (no fold value)</span>';
  const vary = data.varied.length ? data.varied : data.params.map(p => ({ name: p, scale: "linear" }));
  let h = "";
  for (const v of vary) {
    const pts = data.runs.filter(r => r[v.name] != null && r[FOLD_Y] != null && (v.scale !== "log" || r[v.name] > 0)).map(r => ({ x: r[v.name], y: r[FOLD_Y], cl: cls(r), tip: "run " + r.run + ": " + v.name + " " + fmt(r[v.name]) + ", " + FOLD_Y + " " + fmt(r[FOLD_Y]) }));
    h += '<div class="card"><b class="small">' + esc(FOLD_Y) + " against " + esc(v.name) + "</b>" + chart(pts, { xlog: v.scale === "log", xlabel: v.name, ylabel: "fold amplitude (mm)" }) + "</div>";
  }
  $("#scat").innerHTML = h || '<div class="small mut">no varied parameters</div>';
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
  $("#order").innerHTML = '<div class="card">' + chart(pts, { xlabel: "run order (run number)", ylabel: m, line: true }) + "</div>";
}

// ---------------------------------------------------------------- datasets, help
async function loadDatasets() {
  try {
    const d = await api("/api/datasets");
    $("#dst").innerHTML = "<thead><tr><th>dataset</th><th>kind</th><th>role</th><th>provenance</th><th>rows</th><th>checker messages</th></tr></thead><tbody>" +
      d.map(x => "<tr><td><b>" + esc(x.id) + "</b><br><span class='mut'>" + esc(x.fabric || "") + "</span></td><td>" + esc(x.kind) + "</td><td>" + esc(x.role) + "</td><td>" + esc(x.provenance || "not stated") + (x.source ? "<br><span class='mut'>" + esc(x.source) + "</span>" : "") + "</td><td>" + esc(x.rows == null ? "?" : x.rows) + (x.usable != null ? " (" + x.usable + " usable)" : "") + "</td><td>" +
        (x.errors.map(e => '<div style="color:var(--crit)">error: ' + esc(e) + "</div>").join("") + x.warnings.map(w => "<div>warning: " + esc(w) + "</div>").join("") || "none") + "</td></tr>").join("") + "</tbody>";
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
    $("#sum").innerHTML = data.summary.map(s => '<li class="' + s.level + '">' + esc(s.text) + "</li>").join("");
    table(); charts();
  } catch (e) { msg(e.message, true); }
}
loadHelp(); loadDatasets(); refresh(); setInterval(() => { if (!document.hidden) refresh(); }, 20000);

// ---------------------------------------------------------------- Pure shell only runs (thin-shell fold only; /api/shell/..., scripts/shell_campaign.py). Separate from the yarn campaign code above.
let shCur = null, shData = null, shBusy = false; const shSel = new Set();
const SH_COLS = ["status", "tiles", "cpi", "wpi", "kappa_scale", "h_scale", "matrix", "knit_fraction", "plan_x", "plan_y", "thickness", "wave_k", "wave_amp_median_mm", "mid_over_end", "steps_at_cap", "wall_s"];
function shMsg(t, err) { const m = $("#shmsg"); m.textContent = t || ""; m.className = err ? "err" : ""; }
async function shLoadList() {
  const j = await api("/api/shell"); const t = $("#shtabs"); t.innerHTML = ""; $("#shempty").hidden = j.campaigns.length > 0;
  if (!j.campaigns.some(c => c.name === shCur)) shCur = j.campaigns.length ? j.campaigns[0].name : null;
  for (const c of j.campaigns) {
    const b = document.createElement("button"); b.textContent = c.name + " (" + (c.runner_alive ? (c.state || "alive") : "no runner") + ", " + c.control + ")"; b.className = c.name === shCur ? "on" : "";
    b.onclick = () => { shCur = c.name; shSel.clear(); shMsg(""); shRefresh(); }; t.appendChild(b);
  }
  $("#shcur").hidden = !shCur;
}
function shRender() {
  const d = shData, rn = d.runner || {}, alive = d.runner_alive, state = alive ? (rn.state || "alive") : (rn.state === "guard_stopped" ? "guard_stopped" : "no runner");
  $("#shname").textContent = d.name; const p = $("#shpill"); p.textContent = state; p.className = "pill " + (alive && rn.state === "running" ? "running" : alive && rn.state === "paused" ? "paused" : "idle");
  $("#shdetail").textContent = "control: " + d.control + (rn.reason ? " | " + rn.reason : "");
  $("#shStart").disabled = shBusy || (alive && d.control === "run"); $("#shPause").disabled = shBusy || !alive || d.control === "pause"; $("#shCont").disabled = shBusy || !alive || d.control === "run"; $("#shStop").disabled = shBusy || !alive;
  const c = d.counts, pr = d.progress || {}, cur = rn.current ? d.rows.find(r => r.run === rn.current) : null;
  const parts = [];
  if (alive && rn.current) parts.push("current run " + esc(rn.current) + (pr.growth_steps_done != null ? ": growth step " + pr.growth_steps_done + (cur && cur.steps ? " of " + cur.steps : "") : ": starting") + (rn.timeout_s ? " (timeout " + dur(rn.timeout_s) + ")" : ""));
  parts.push("queued " + c.queued + ", done " + c.done + ", failed " + c.failed + ", timeout " + c.timeout + (c.needs_approval ? ", needs approval " + c.needs_approval : ""));
  parts.push("mean run " + (d.mean_wall_s ? dur(d.mean_wall_s) : "n/a") + (d.eta_s ? ", queue about " + dur(d.eta_s) : ""));
  parts.push("free disk " + d.free_gb + " GB (stops below " + d.limits.min_free_gb + " GB), this folder " + d.total_gb + " GB (limit " + d.limits.max_total_gb + " GB)");
  $("#shstat").innerHTML = parts.map(s => "<div>" + s + "</div>").join("");
  const head = ["run"].concat(SH_COLS), rows = d.rows.slice().reverse();
  $("#shtbl").innerHTML = "<thead><tr><th></th>" + head.map(h => "<th" + (h === "run" ? ' class="l"' : "") + ">" + esc(h) + "</th>").join("") + "</tr></thead><tbody>" + rows.map(r =>
    '<tr><td><input type="checkbox" data-r="' + esc(r.run) + '"' + (shSel.has(r.run) ? " checked" : "") + (r.status === "done" ? "" : " disabled") + "></td>" +
    head.map(h => "<td" + (h === "run" ? ' class="l"' : "") + ">" + esc(fmt(h === "status" ? (r.status + (r.heavy_deleted ? " (3D data deleted)" : "")) : r[h], h)) + "</td>").join("") + "</tr>").join("") + "</tbody>";
  $("#shtbl").querySelectorAll("input[data-r]").forEach(i => i.onchange = () => { i.checked ? shSel.add(i.dataset.r) : shSel.delete(i.dataset.r); $("#shDel").disabled = shSel.size === 0; });
  $("#shDel").disabled = shSel.size === 0; $("#shDel").textContent = "Delete heavy files of the selected runs (" + shSel.size + ")";
}
async function shAct(kind) {
  shBusy = true; shMsg("…"); try { const j = await api("/api/shell/" + encodeURIComponent(shCur) + "/" + kind, { method: "POST" }); shMsg(j.message || "Done.", false); } catch (e) { shMsg(e.message, true); }
  shBusy = false; setTimeout(shRefresh, 1500); shRefresh();
}
$("#shStart").onclick = () => { if (confirm("Start shell campaign '" + shCur + "'? Runs go on until you pause; finished runs are never repeated.")) shAct("start"); };
$("#shPause").onclick = () => shAct("pause"); $("#shCont").onclick = () => shAct("continue");
$("#shStop").onclick = () => { if (confirm("Stop the runner after the run in progress?")) shAct("stop"); };
$("#shDel").onclick = async () => {
  const ids = Array.from(shSel); if (!ids.length) return;
  try {
    const pv = await api("/api/shell/" + encodeURIComponent(shCur) + "/delete-preview?ids=" + encodeURIComponent(ids.join(",")));
    const word = "DELETE " + pv.ids.length, ans = prompt("This removes shell.npz and shell.bin of " + pv.ids.length + " run(s): " + pv.ids.join(", ") + " (" + (pv.bytes / 1e6).toFixed(1) + " MB, " + pv.files + " files). Settings, descriptors and the log stay.\n\nType " + word + " to confirm:");
    if (ans !== word) { shMsg("Delete cancelled.", false); return; }
    const j = await api("/api/shell/" + encodeURIComponent(shCur) + "/delete", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ids: pv.ids, confirm: ans }) });
    shSel.clear(); shMsg("Removed " + j.files + " files (" + (j.bytes / 1e6).toFixed(1) + " MB) of " + j.ids.length + " run(s). Logged in deletions.log.", false);
  } catch (e) { shMsg(e.message, true); }
  shRefresh();
};
async function shRefresh() {
  try { await shLoadList(); if (!shCur) return; shData = await api("/api/shell/" + encodeURIComponent(shCur)); shRender(); } catch (e) { shMsg(e.message, true); }
}
shRefresh(); setInterval(() => { if (!document.hidden) shRefresh(); }, 5000);

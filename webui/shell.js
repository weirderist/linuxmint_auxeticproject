"use strict";
// Pure shell runs page. Everything shown comes from /api/shell*. No libraries: tables are built by hand, charts by chartkit.js. Colours and sizes come from theme.css (no colour literals here).
const $ = s => document.querySelector(s);
const esc = s => String(s == null ? "" : s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
async function api(path, opt) {
  const r = await fetch(path, opt); let j = null; try { j = await r.json(); } catch (e) {}
  if (!r.ok) throw new Error((j && j.detail) ? (typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail)) : "HTTP " + r.status);
  return j;
}
function fmt(v, k) {
  if (v == null || v === "") return "";
  if (typeof v === "number") { if (k === "wall_s") return Math.round(v) + ""; return ChartKit.num(v); }
  return String(v);
}
function dur(s) { if (s == null) return ""; s = Math.round(s); const h = Math.floor(s / 3600), m = Math.floor(s % 3600 / 60); return h ? h + " h " + m + " min" : m ? m + " min" : s + " s"; }
let cur = null, data = null, busy = false, sortKey = "run", sortDir = 1, openRun = null; const sel = new Set();
const COLS = ["status", "source", "label", "tiles", "cpi", "wpi", "ratio", "matrix", "knit_fraction", "plan_x", "plan_y", "thickness", "shrink_x", "shrink_y", "wave_k", "wave_amp_median_mm", "mid_over_end", "steps_at_cap", "score", "wall_s", "git_commit"];
const NUM = ["tiles", "cpi", "wpi", "ratio", "knit_fraction", "plan_x", "plan_y", "thickness", "shrink_x", "shrink_y", "fold_kx_per_mm", "fold_ky_per_mm", "wave_k", "wave_amp_median_mm", "mid_over_end", "steps_at_cap", "iterations_total", "wall_s", "solver_seconds", "tier", "score"];
const TEXTCOL = ["run", "label", "source", "status", "matrix", "git_commit"];
const SETT = ["tiles", "cpi", "wpi", "kappa_scale", "h_scale", "sub", "steps", "iter", "nu", "springs", "seed", "ne", "matrix"];
function msg(t, err) { const m = $("#msg"); m.textContent = t ? (err ? "Error: " : "") + t : ""; m.className = err ? "err" : ""; }
const kind = r => (r.source || "").split(":")[0], seedOf = r => (r.source || "").split(":")[1] || "";

async function loadList() {
  const j = await api("/api/shell"); const t = $("#tabs"); t.innerHTML = ""; $("#empty").hidden = j.campaigns.length > 0;
  if (!j.campaigns.some(c => c.name === cur)) cur = j.campaigns.length ? j.campaigns[0].name : null;
  for (const c of j.campaigns) {
    const b = document.createElement("button"); b.textContent = c.name + " (" + (c.runner_alive ? (c.state || "alive") : "no runner") + ", " + c.control + ")"; b.className = c.name === cur ? "on" : "";
    b.onclick = () => { cur = c.name; sel.clear(); openRun = null; msg(""); refresh(); }; t.appendChild(b);
  }
  $("#cur").hidden = !cur;
}
function header() {
  const d = data, rn = d.runner || {}, alive = d.runner_alive, state = alive ? (rn.state || "alive") : (rn.state === "guard_stopped" ? "guard_stopped" : "no runner");
  $("#cname").textContent = d.name; const p = $("#pill"); p.textContent = state; p.className = "pill " + (alive && rn.state === "running" ? "running" : alive && rn.state === "paused" ? "paused" : (!alive || state === "idle") ? "idle" : "paused");
  $("#pdetail").textContent = "control: " + d.control + (rn.reason ? " | " + rn.reason : "");
  $("#bStart").disabled = busy || (alive && d.control === "run"); $("#bPause").disabled = busy || !alive || d.control === "pause"; $("#bCont").disabled = busy || !alive || d.control === "run"; $("#bStop").disabled = busy || !alive;
  const c = d.counts, pr = d.progress || {}, cr = rn.current ? d.rows.find(r => r.run === rn.current) : null, parts = [];
  if (alive && rn.current) parts.push("current run " + esc(rn.current) + (cr && cr.label ? " (" + esc(cr.label) + ")" : "") + (pr.growth_steps_done != null ? ": growth step " + pr.growth_steps_done + (cr && cr.steps ? " of " + cr.steps : "") : ": starting") + (rn.timeout_s ? ", timeout " + dur(rn.timeout_s) : ""));
  parts.push("queued " + c.queued + ", done " + c.done + ", failed " + c.failed + ", timeout " + c.timeout + (c.rejected ? ", rejected " + c.rejected : "") + (c.needs_approval ? ", needs approval " + c.needs_approval : ""));
  parts.push("mean run " + (d.mean_wall_s ? dur(d.mean_wall_s) : "n/a") + (d.eta_s ? ", queue about " + dur(d.eta_s) + (d.eta_s > 86400 ? " (" + (d.eta_s / 86400).toFixed(1) + " days)" : "") : ""));
  parts.push("free disk " + d.free_gb + " GB (the runner stops below " + d.limits.min_free_gb + " GB), this campaign " + d.total_gb + " GB (limit " + d.limits.max_total_gb + " GB)");
  $("#stat").innerHTML = parts.map(s => "<div>" + s + "</div>").join("");
  const code = d.code || {}; $("#code").textContent = code.git_commit ? "frozen code: commit " + String(code.git_commit).slice(0, 10) + (code.git_dirty ? " (UNCOMMITTED changes at freeze time)" : " (clean)") + ", shell_sim.py sha256 " + String(code.shell_sim_sha256 || "").slice(0, 12) + ", frozen " + (code.frozen || "") : "";
}
function rowsView() {
  let rows = data.rows.map(r => Object.assign({ ratio: (r.cpi && r.wpi) ? r.cpi / r.wpi : null }, r));
  const fs = $("#fStatus").value, fc = $("#fSource").value; if (fs) rows = rows.filter(r => r.status === fs); if (fc) rows = rows.filter(r => kind(r) === fc);
  rows.sort((a, b) => { const x = a[sortKey], y = b[sortKey]; if (x == null || x === "") return 1; if (y == null || y === "") return -1; return (typeof x === "number" && typeof y === "number" ? x - y : String(x).localeCompare(String(y))) * sortDir; });
  return rows;
}
function table() {
  const rows = rowsView(), head = ["run"].concat(COLS), left = h => TEXTCOL.includes(h) ? ' class="l"' : "";
  $("#tbl").innerHTML = "<thead><tr><th></th>" + head.map(h => '<th data-k="' + h + '"' + left(h) + ">" + esc(h) + (h === sortKey ? (sortDir > 0 ? " &uarr;" : " &darr;") : "") + "</th>").join("") + "</tr></thead><tbody>" +
    rows.slice(0, 1500).map(r => '<tr data-r="' + esc(r.run) + '" class="' + (r.run === openRun ? "sel" : "") + '"><td><input type="checkbox" data-c="' + esc(r.run) + '"' + (sel.has(r.run) ? " checked" : "") + (r.status === "done" ? "" : " disabled") + "></td>" +
    head.map(h => "<td" + left(h) + ">" + (h === "status" ? esc(r.status) + (r.heavy_deleted ? " (3D deleted)" : "") + (r.control ? ' <span class="tag">control</span>' : "") : esc(fmt(r[h], h))) + "</td>").join("") + "</tr>").join("") + "</tbody>";
  $("#tbl").querySelectorAll("th[data-k]").forEach(th => th.onclick = () => { const k = th.dataset.k; if (sortKey === k) sortDir = -sortDir; else { sortKey = k; sortDir = 1; } table(); });
  $("#tbl").querySelectorAll("tbody tr").forEach(tr => tr.onclick = ev => { if (ev.target.tagName === "INPUT") return; openRun = tr.dataset.r; table(); showDetail(openRun); });
  $("#tbl").querySelectorAll("input[data-c]").forEach(i => i.onchange = () => { i.checked ? sel.add(i.dataset.c) : sel.delete(i.dataset.c); buttons(); });
  buttons();
}
function buttons() { const n = sel.size; $("#bDel").disabled = n === 0; $("#bDel").textContent = "Delete heavy files of the selected runs (" + n + ")"; $("#bCompare").disabled = n < 2; $("#bView").disabled = n < 1 || n > 2; }
async function showDetail(rid) {
  const box = $("#detail"); box.innerHTML = '<div class="card note">loading</div>';
  try {
    const d = await api("/api/shell/" + encodeURIComponent(cur) + "/runs/" + encodeURIComponent(rid)), s = d.settings || {}, m = d.metrics || {}, pv = d.provenance || {}, sd = d.seed || {};
    const dl = o => "<dl>" + Object.keys(o).map(k => "<dt>" + esc(k) + "</dt><dd>" + esc(fmt(o[k], k)) + "</dd>").join("") + "</dl>";
    const sett = {}; SETT.forEach(k => { if (s[k] !== undefined) sett[k] = k === "matrix" ? (s.matrix.length + " x " + s.matrix[0].length + " per tile") : s[k]; });
    const met = {}; ["plan_x", "plan_y", "thickness", "shrink_x", "shrink_y", "fold_direction", "fold_kx_per_mm", "fold_ky_per_mm", "wave_k", "wave_power_fraction", "wave_amp_median_mm", "mid_over_end", "steps_at_cap", "iterations_total", "U_final", "solver_seconds"].forEach(k => { if (m[k] !== undefined) met[k] = m[k]; });
    const prov = { frozen: pv.frozen, matches_manifest: pv.matches_manifest, git_commit: String(pv.git_commit || "").slice(0, 10), git_dirty: pv.git_dirty, shell_sim_sha256: String(pv.shell_sim_sha256 || "").slice(0, 16), python: pv.python, numpy: pv.numpy, scipy: pv.scipy, jax: pv.jax };
    const link = d.files["shell.bin"] ? '<a href="/shellview?c=' + encodeURIComponent(cur) + "&run=" + encodeURIComponent(rid) + '">Open the 3D viewer and strain maps</a>' : '<span class="mut">3D data deleted</span>';
    box.innerHTML = '<div class="card"><h3>' + esc(rid) + " " + esc(d.label || "") + ' <span class="mut small">' + esc(d.source) + " | tier " + esc(d.tier) + '</span></h3><div class="row"><div class="col"><span class="bold">settings</span>' + dl(sett) + '</div><div class="col"><span class="bold">descriptors</span>' + dl(met) + '</div><div class="col"><span class="bold">code provenance</span>' + dl(prov) + "</div></div>" +
      (d.control ? '<p class="note"><span class="tag">control</span> ' + esc(d.control.reason) + " (changes: " + esc((d.control.keys || []).join(", ")) + ")</p>" : "") +
      (sd.real_fold ? '<p class="note"><span class="bold">' + (sd.kind === "seed" ? "Real fold of this pattern" : "Real fold of the pattern it was edited from (" + esc(sd.seed) + ")") + "</span> (" + esc(sd.sample || "") + "; TM axis: " + esc(sd.real_tm_axis || "") + "; K's report): " + esc(sd.real_fold) + "</p>" : "") +
      '<p class="note">' + link + '</p><details class="msgs"><summary>log (last lines)</summary><pre>' + esc((d.log_tail || []).join("\n")) + "</pre></details></div>";
  } catch (e) { box.innerHTML = '<div class="card"><b>Error:</b> ' + esc(e.message) + "</div>"; }
}
function compare() {
  const rows = data.rows.filter(r => sel.has(r.run)), keys = ["label", "source"].concat(SETT.filter(k => k !== "matrix"), ["matrix", "knit_fraction", "plan_x", "plan_y", "thickness", "shrink_x", "shrink_y", "wave_k", "wave_amp_median_mm", "mid_over_end", "steps_at_cap", "score", "wall_s"]);
  $("#cmpsec").hidden = rows.length < 2;
  $("#cmp").innerHTML = '<thead><tr><th class="l">quantity</th>' + rows.map(r => "<th>" + esc(r.run) + "</th>").join("") + "</tr></thead><tbody>" + keys.map(k => { const vals = rows.map(r => r[k]), diff = new Set(vals.map(v => String(v))).size > 1;
    return '<tr><td class="l">' + esc(k) + "</td>" + vals.map(v => '<td class="' + (diff ? "diff" : "") + '">' + esc(fmt(v, k)) + "</td>").join("") + "</tr>"; }).join("") + "</tbody>";
  $("#cmpsec").scrollIntoView({ behavior: "smooth" });
}
// ---- comparison graph: any two columns, coloured by source / seed pattern / tiles (same renderer, size and rules as every other chart)
function charts() {
  const rows = data.rows.filter(r => r.status === "done").map(r => Object.assign({ ratio: (r.cpi && r.wpi) ? r.cpi / r.wpi : null }, r)); const gx = $("#gx"), gy = $("#gy");
  if (!gx.options.length) { for (const k of NUM) { gx.add(new Option(k, k)); gy.add(new Option(k, k)); } gx.value = "ratio"; gy.value = "thickness"; gx.onchange = gy.onchange = $("#gc").onchange = charts; }
  const xk = gx.value, yk = gy.value, ck = $("#gc").value, pts = rows.filter(r => typeof r[xk] === "number" && typeof r[yk] === "number");
  const grp = r => ck === "src" ? kind(r) : ck === "seed" ? (seedOf(r) || "-") : "tiles " + r.tiles, names = Array.from(new Set(pts.map(grp))).sort();
  $("#gleg").innerHTML = names.map((n, i) => '<span class="g' + (i % 8) + '">' + esc(n) + "</span>").join("");
  const P = pts.map(p => { const i = names.indexOf(grp(p)); return { x: p[xk], y: p[yk], cls: "g" + (i % 8), shape: "circle", tip: p.run + " " + (p.label || "") + ": " + xk + " " + fmt(p[xk]) + ", " + yk + " " + fmt(p[yk]), run: p.run }; });
  $("#gr").innerHTML = ChartKit.scatter(P, { xlabel: xk, ylabel: yk }, ChartKit.size($("#gr")));
  $("#gr").querySelectorAll(".mk").forEach((c, i) => c.onclick = () => { openRun = P[i].run; table(); showDetail(openRun); });
}
async function rules() {
  const box = $("#rules"); try {
    const r = await api("/api/shell/" + encodeURIComponent(cur) + "/rules"); if (!r.targets) { box.innerHTML = "No rules yet: the job has not run (" + esc(r.n_finished) + " finished runs; rules are shown from " + esc(r.min_runs) + ")."; return; }
    box.innerHTML = '<div><span class="bold">' + esc(r.label) + "</span>. " + esc(r.n_finished) + " finished runs at " + esc(r.generated) + ". " + esc(r.method) + ".</div>" + r.targets.map(t => {
      if (t.status !== "ok") return "<h3>" + esc(t.target) + '</h3><div class="mut">' + esc(t.note || t.status) + " (" + esc(t.n_runs) + " runs)</div>";
      const la = t.lasso; return "<h3>" + esc(t.target) + "</h3><div>sparse regression: cross-validated R&sup2; " + (+la.cv_r2).toFixed(3) + " against the mean predictor, " + la.n_selected + " features selected, support " + la.support + " runs; tree cross-check R&sup2; " + (+t.tree.cv_r2).toFixed(3) + "</div>" +
        '<div class="tw"><table><thead><tr><th class="l">feature</th><th>per unit</th><th>standardised</th><th>selected in bootstrap refits</th></tr></thead><tbody>' + la.coefficients.map(c => '<tr><td class="l">' + esc(c.feature) + "</td><td>" + fmt(c.per_unit) + "</td><td>" + fmt(c.coefficient_std) + "</td><td>" + Math.round(100 * c.stability) + " %</td></tr>").join("") + "</tbody></table></div><pre>" + esc(t.tree.rules.join("\n")) + "</pre>"; }).join("");
  } catch (e) { box.textContent = e.message; }
}
async function act(kindName) { busy = true; msg("Working"); try { const j = await api("/api/shell/" + encodeURIComponent(cur) + "/" + kindName, { method: "POST" }); msg(j.message || "Done.", false); } catch (e) { msg(e.message, true); } busy = false; setTimeout(refresh, 1500); refresh(); }
$("#bStart").onclick = () => { if (confirm("Start shell campaign '" + cur + "'? It goes on until you pause; finished runs are never repeated.")) act("start"); };
$("#bPause").onclick = () => act("pause"); $("#bCont").onclick = () => act("continue");
$("#bStop").onclick = () => { if (confirm("Stop the runner after the run in progress?")) act("stop"); };
$("#bCompare").onclick = compare;
$("#bView").onclick = () => { const ids = Array.from(sel); location.href = "/shellview?c=" + encodeURIComponent(cur) + "&run=" + encodeURIComponent(ids[0]) + (ids[1] ? "&b=" + encodeURIComponent(ids[1]) : ""); };
$("#fStatus").onchange = $("#fSource").onchange = () => { if (data) table(); };
$("#bDel").onclick = async () => {
  const ids = Array.from(sel); if (!ids.length) return;
  try {
    const pv = await api("/api/shell/" + encodeURIComponent(cur) + "/delete-preview?ids=" + encodeURIComponent(ids.join(","))), word = "DELETE " + pv.ids.length;
    const ans = prompt("This removes shell.npz and shell.bin of " + pv.ids.length + " run(s): " + pv.ids.join(", ") + " (" + (pv.bytes / 1e6).toFixed(1) + " MB, " + pv.files + " files). Settings, descriptors and the log stay.\n\nType " + word + " to confirm:");
    if (ans !== word) { msg("Delete cancelled.", false); return; }
    const j = await api("/api/shell/" + encodeURIComponent(cur) + "/delete", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ids: pv.ids, confirm: ans }) });
    sel.clear(); msg("Removed " + j.files + " files (" + (j.bytes / 1e6).toFixed(1) + " MB) of " + j.ids.length + " run(s). Logged in deletions.log.", false);
  } catch (e) { msg(e.message, true); }
  refresh();
};
async function refresh() { try { await loadList(); if (!cur) return; data = await api("/api/shell/" + encodeURIComponent(cur)); header(); table(); charts(); rules(); if (openRun && !$("#detail").firstChild) showDetail(openRun); } catch (e) { msg(e.message, true); } }
let resizeTimer = null;
window.addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(() => { if (data) charts(); }, 150); });
refresh(); setInterval(() => { if (!document.hidden) refresh(); }, 15000);

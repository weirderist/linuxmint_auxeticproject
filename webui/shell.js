"use strict";
// Pure shell runs page. Everything shown comes from /api/shell*. No libraries: tables and SVG charts are built by hand.
const $ = s => document.querySelector(s);
const esc = s => String(s == null ? "" : s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
async function api(path, opt) {
  const r = await fetch(path, opt); let j = null; try { j = await r.json(); } catch (e) {}
  if (!r.ok) throw new Error((j && j.detail) ? (typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail)) : "HTTP " + r.status);
  return j;
}
function fmt(v, k) {
  if (v == null || v === "") return "";
  if (typeof v === "number") { if (k === "wall_s") return Math.round(v) + ""; if (Number.isInteger(v)) return String(v); const a = Math.abs(v); return (a !== 0 && (a < 1e-3 || a >= 1e5)) ? v.toExponential(2) : String(+v.toPrecision(4)); }
  return String(v);
}
function dur(s) { if (s == null) return ""; s = Math.round(s); const h = Math.floor(s / 3600), m = Math.floor(s % 3600 / 60); return h ? h + " h " + m + " min" : m ? m + " min" : s + " s"; }
let cur = null, data = null, busy = false, sortKey = "run", sortDir = 1, openRun = null; const sel = new Set();
const COLS = ["status", "source", "label", "tiles", "cpi", "wpi", "ratio", "matrix", "knit_fraction", "plan_x", "plan_y", "thickness", "shrink_x", "shrink_y", "wave_k", "wave_amp_median_mm", "mid_over_end", "steps_at_cap", "wall_s", "git_commit"];
const NUM = ["tiles", "cpi", "wpi", "ratio", "knit_fraction", "plan_x", "plan_y", "thickness", "shrink_x", "shrink_y", "fold_kx_per_mm", "fold_ky_per_mm", "wave_k", "wave_amp_median_mm", "mid_over_end", "steps_at_cap", "iterations_total", "wall_s", "solver_seconds", "tier"];
const SETT = ["tiles", "cpi", "wpi", "kappa_scale", "h_scale", "sub", "steps", "iter", "nu", "springs", "seed", "ne", "matrix"];
function msg(t, err) { const m = $("#msg"); m.textContent = t || ""; m.className = err ? "err" : ""; }
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
  $("#cname").textContent = d.name; const p = $("#pill"); p.textContent = state; p.className = "pill " + (alive && rn.state === "running" ? "running" : alive && rn.state === "paused" ? "paused" : "idle");
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
  const rows = rowsView(), head = ["run"].concat(COLS);
  $("#tbl").innerHTML = "<thead><tr><th></th>" + head.map(h => '<th data-k="' + h + '"' + (h === "run" || h === "label" || h === "source" ? ' class="l"' : "") + ">" + esc(h) + (h === sortKey ? (sortDir > 0 ? " &uarr;" : " &darr;") : "") + "</th>").join("") + "</tr></thead><tbody>" +
    rows.slice(0, 1500).map(r => '<tr data-r="' + esc(r.run) + '" class="' + (r.run === openRun ? "sel" : "") + '"><td><input type="checkbox" data-c="' + esc(r.run) + '"' + (sel.has(r.run) ? " checked" : "") + (r.status === "done" ? "" : " disabled") + "></td>" +
    head.map(h => "<td" + (h === "run" || h === "label" || h === "source" ? ' class="l"' : "") + ">" + (h === "status" ? esc(r.status) + (r.heavy_deleted ? " (3D deleted)" : "") + (r.control ? ' <span class="tag ctl">control</span>' : "") : esc(fmt(r[h], h))) + "</td>").join("") + "</tr>").join("") + "</tbody>";
  $("#tbl").querySelectorAll("th[data-k]").forEach(th => th.onclick = () => { const k = th.dataset.k; if (sortKey === k) sortDir = -sortDir; else { sortKey = k; sortDir = 1; } table(); });
  $("#tbl").querySelectorAll("tbody tr").forEach(tr => tr.onclick = ev => { if (ev.target.tagName === "INPUT") return; openRun = tr.dataset.r; table(); showDetail(openRun); });
  $("#tbl").querySelectorAll("input[data-c]").forEach(i => i.onchange = () => { i.checked ? sel.add(i.dataset.c) : sel.delete(i.dataset.c); buttons(); });
  buttons();
}
function buttons() { const n = sel.size; $("#bDel").disabled = n === 0; $("#bDel").textContent = "Delete heavy files of the selected runs (" + n + ")"; $("#bCompare").disabled = n < 2; $("#bView").disabled = n < 1 || n > 2; }
async function showDetail(rid) {
  const box = $("#detail"); box.innerHTML = '<div class="card small mut">loading…</div>';
  try {
    const d = await api("/api/shell/" + encodeURIComponent(cur) + "/runs/" + encodeURIComponent(rid)), s = d.settings || {}, m = d.metrics || {}, pv = d.provenance || {}, sd = d.seed || {};
    const dl = o => "<dl>" + Object.keys(o).map(k => "<dt>" + esc(k) + "</dt><dd>" + esc(fmt(o[k], k)) + "</dd>").join("") + "</dl>";
    const sett = {}; SETT.forEach(k => { if (s[k] !== undefined) sett[k] = k === "matrix" ? (s.matrix.length + " x " + s.matrix[0].length + " per tile") : s[k]; });
    const met = {}; ["plan_x", "plan_y", "thickness", "shrink_x", "shrink_y", "fold_direction", "fold_kx_per_mm", "fold_ky_per_mm", "wave_k", "wave_power_fraction", "wave_amp_median_mm", "mid_over_end", "steps_at_cap", "iterations_total", "U_final", "solver_seconds"].forEach(k => { if (m[k] !== undefined) met[k] = m[k]; });
    const prov = { frozen: pv.frozen, matches_manifest: pv.matches_manifest, git_commit: String(pv.git_commit || "").slice(0, 10), git_dirty: pv.git_dirty, shell_sim_sha256: String(pv.shell_sim_sha256 || "").slice(0, 16), python: pv.python, numpy: pv.numpy, scipy: pv.scipy, jax: pv.jax };
    const link = d.files["shell.bin"] ? '<a href="/shellview?c=' + encodeURIComponent(cur) + "&run=" + encodeURIComponent(rid) + '">Open the 3D viewer and strain maps</a>' : '<span class="mut">3D data deleted</span>';
    box.innerHTML = '<div class="card"><h3>' + esc(rid) + " " + esc(d.label || "") + ' <span class="mut small">' + esc(d.source) + " | tier " + esc(d.tier) + "</span></h3><div class=\"row\"><div class=\"act\"><b>settings</b>" + dl(sett) + "</div><div class=\"act\"><b>descriptors</b>" + dl(met) + "</div><div class=\"act\"><b>code provenance</b>" + dl(prov) + "</div></div>" +
      (d.control ? '<div class="small"><span class="tag ctl">control</span> ' + esc(d.control.reason) + " (changes: " + esc((d.control.keys || []).join(", ")) + ")</div>" : "") +
      (sd.real_fold ? '<div class="small" style="margin:.4rem 0"><b>' + (sd.kind === "seed" ? "Real fold of this pattern" : "Real fold of the pattern it was edited from (" + esc(sd.seed) + ")") + "</b> (" + esc(sd.sample || "") + "; TM axis: " + esc(sd.real_tm_axis || "") + "; K's report): " + esc(sd.real_fold) + "</div>" : "") +
      '<div class="small">' + link + "</div><details><summary class=\"small\">log (last lines)</summary><pre>" + esc((d.log_tail || []).join("\n")) + "</pre></details></div>";
  } catch (e) { box.innerHTML = '<div class="card small" style="color:var(--crit)">' + esc(e.message) + "</div>"; }
}
function compare() {
  const rows = data.rows.filter(r => sel.has(r.run)), keys = ["label", "source"].concat(SETT.filter(k => k !== "matrix"), ["matrix", "knit_fraction", "plan_x", "plan_y", "thickness", "shrink_x", "shrink_y", "wave_k", "wave_amp_median_mm", "mid_over_end", "steps_at_cap", "wall_s"]);
  $("#cmpsec").hidden = rows.length < 2;
  $("#cmp").innerHTML = "<thead><tr><th class=\"l\">quantity</th>" + rows.map(r => "<th>" + esc(r.run) + "</th>").join("") + "</tr></thead><tbody>" + keys.map(k => { const vals = rows.map(r => r[k]), diff = new Set(vals.map(v => String(v))).size > 1;
    return "<tr><td class=\"l\">" + esc(k) + "</td>" + vals.map(v => "<td class=\"" + (diff ? "diff" : "") + "\">" + esc(fmt(v, k)) + "</td>").join("") + "</tr>"; }).join("") + "</tbody>";
  $("#cmpsec").scrollIntoView({ behavior: "smooth" });
}
// ---- scatter chart: any two columns, coloured by source / seed pattern / tiles
const PAL = ["#2a78d6", "#eb6834", "#1b8a5a", "#9b59b6", "#d4a017", "#17a2b8", "#e34948", "#6c757d", "#8e6e53", "#c2185b", "#00897b"];
function charts() {
  const rows = data.rows.filter(r => r.status === "done").map(r => Object.assign({ ratio: (r.cpi && r.wpi) ? r.cpi / r.wpi : null }, r)); const gx = $("#gx"), gy = $("#gy");
  if (!gx.options.length) { for (const k of NUM) { gx.add(new Option(k, k)); gy.add(new Option(k, k)); } gx.value = "ratio"; gy.value = "thickness"; gx.onchange = gy.onchange = $("#gc").onchange = charts; }
  const xk = gx.value, yk = gy.value, ck = $("#gc").value, pts = rows.filter(r => typeof r[xk] === "number" && typeof r[yk] === "number");
  const grp = r => ck === "src" ? kind(r) : ck === "seed" ? (seedOf(r) || "-") : "tiles " + r.tiles, names = Array.from(new Set(pts.map(grp))).sort();
  $("#gleg").innerHTML = names.map((n, i) => '<span style="--c:' + PAL[i % PAL.length] + '">' + esc(n) + "</span>").join("");
  if (!pts.length) { $("#gr").innerHTML = '<div class="small mut">no finished runs with both columns yet</div>'; return; }
  const W = 760, H = 380, L = 56, R = 14, T = 10, B = 40, xs = pts.map(p => p[xk]), ys = pts.map(p => p[yk]), pad = (a, b) => (b - a) || 1;
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys), sx = v => L + (v - x0) / pad(x0, x1) * (W - L - R), sy = v => H - B - (v - y0) / pad(y0, y1) * (H - B - T);
  let g = ""; for (let i = 0; i <= 4; i++) { const xv = x0 + (x1 - x0) * i / 4, yv = y0 + (y1 - y0) * i / 4; g += '<line class="gd" x1="' + sx(xv) + '" x2="' + sx(xv) + '" y1="' + T + '" y2="' + (H - B) + '"/><text x="' + sx(xv) + '" y="' + (H - B + 13) + '" text-anchor="middle">' + fmt(+xv.toPrecision(3)) + '</text><line class="gd" x1="' + L + '" x2="' + (W - R) + '" y1="' + sy(yv) + '" y2="' + sy(yv) + '"/><text x="' + (L - 4) + '" y="' + (sy(yv) + 3) + '" text-anchor="end">' + fmt(+yv.toPrecision(3)) + "</text>"; }
  const dots = pts.map(p => '<circle cx="' + sx(p[xk]) + '" cy="' + sy(p[yk]) + '" r="3.4" fill="' + PAL[names.indexOf(grp(p)) % PAL.length] + '" fill-opacity=".75" data-r="' + esc(p.run) + '"><title>' + esc(p.run + " " + (p.label || "") + ": " + xk + " " + fmt(p[xk]) + ", " + yk + " " + fmt(p[yk])) + "</title></circle>").join("");
  $("#gr").innerHTML = '<svg viewBox="0 0 ' + W + " " + H + '"><line class="ax" x1="' + L + '" x2="' + (W - R) + '" y1="' + (H - B) + '" y2="' + (H - B) + '"/><line class="ax" x1="' + L + '" x2="' + L + '" y1="' + T + '" y2="' + (H - B) + '"/>' + g + dots +
    '<text x="' + (W / 2) + '" y="' + (H - 4) + '" text-anchor="middle">' + esc(xk) + '</text><text transform="translate(12 ' + (H / 2) + ') rotate(-90)" text-anchor="middle">' + esc(yk) + "</text></svg>";
  $("#gr").querySelectorAll("circle[data-r]").forEach(c => c.onclick = () => { openRun = c.dataset.r; table(); showDetail(openRun); });
}
async function act(kindName) { busy = true; msg("…"); try { const j = await api("/api/shell/" + encodeURIComponent(cur) + "/" + kindName, { method: "POST" }); msg(j.message || "Done.", false); } catch (e) { msg(e.message, true); } busy = false; setTimeout(refresh, 1500); refresh(); }
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
async function refresh() { try { await loadList(); if (!cur) return; data = await api("/api/shell/" + encodeURIComponent(cur)); header(); table(); charts(); if (openRun && !$("#detail").firstChild) showDetail(openRun); } catch (e) { msg(e.message, true); } }
refresh(); setInterval(() => { if (!document.hidden) refresh(); }, 15000);

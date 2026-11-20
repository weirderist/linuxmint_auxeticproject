// Shared 3D viewer code for strain.html (same loaders and tube/axis handling as the main viewer in index.html, which keeps its own inline copy).
// Geometry is loaded once and shared: several viewers can show the same position / normal / index arrays and differ only in their colour attribute.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
export { THREE };

export const rid = (campaign, run) => 'run~' + campaign + '~' + run;

export function createViewer(container) {
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
  container.appendChild(renderer.domElement);
  const scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera(40, 1, 0.1, 1000), controls = new OrbitControls(camera, renderer.domElement);
  const sun = new THREE.DirectionalLight(0xffffff, 1.2); sun.position.set(0, 0, 1); sun.target.position.set(0, 0, -1);
  camera.add(sun); camera.add(sun.target); scene.add(camera); scene.add(new THREE.HemisphereLight(0xffffff, 0x666666, 1.1));
  let need = false, shown = null;
  const v = { renderer, scene, camera, controls, container };
  v.invalidate = () => { if (!need) { need = true; requestAnimationFrame(() => { need = false; renderer.render(scene, camera); }); } };
  v.resize = () => { const w = container.clientWidth, h = container.clientHeight || 1; renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix(); v.invalidate(); };
  v.show = (obj, refit) => { if (shown) scene.remove(shown); shown = obj; if (obj) { scene.add(obj); if (refit) v.fit(obj); } v.invalidate(); };
  v.fit = obj => {
    const box = new THREE.Box3().setFromObject(obj), c = box.getCenter(new THREE.Vector3()), s = box.getSize(new THREE.Vector3());
    const r = Math.max(s.x, s.y, s.z, 1e-3) / 2, dist = r / Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) * 1.4;
    controls.target.copy(c); camera.position.set(c.x, c.y, c.z + dist); camera.near = dist / 100; camera.far = dist * 100; camera.updateProjectionMatrix(); controls.update(); v.invalidate();
  };
  controls.addEventListener('change', v.invalidate);
  new ResizeObserver(v.resize).observe(container); v.resize();
  return v;
}

// Keep the cameras of several viewers together: moving one moves the others. enabled() is asked on every move.
export function syncCameras(viewers, enabled) {
  let busy = false;
  for (const a of viewers) a.controls.addEventListener('change', () => {
    if (busy || !enabled()) return; busy = true;
    for (const b of viewers) if (b !== a) {
      b.camera.position.copy(a.camera.position); b.camera.quaternion.copy(a.camera.quaternion); b.camera.zoom = a.camera.zoom;
      b.camera.near = a.camera.near; b.camera.far = a.camera.far; b.camera.updateProjectionMatrix(); b.controls.target.copy(a.controls.target); b.controls.update(); b.invalidate();
    }
    busy = false;
  });
}

async function readWithProgress(resp, onProgress) {
  const total = +resp.headers.get('Content-Length') || 0, rd = resp.body.getReader(); let got = 0, out = total ? new Uint8Array(total) : null; const chunks = [];
  for (;;) {
    const { done, value } = await rd.read(); if (done) break;
    if (out && got + value.length <= out.length) out.set(value, got); else { if (out) { chunks.push(out.subarray(0, got)); out = null; } chunks.push(value); }
    got += value.length; onProgress && onProgress('loading ' + (got / 1048576).toFixed(0) + (total ? '/' + (total / 1048576).toFixed(0) : '') + ' MB');
  }
  if (out) return got === out.length ? out.buffer : out.buffer.slice(0, got);
  const all = new Uint8Array(got); let o = 0; for (const c of chunks) { all.set(c, o); o += c.length; } return all.buffer;
}
const tick = () => new Promise(r => setTimeout(r, 0));
async function vertexNormals(pos, idx, nv, onProgress) {
  const n = new Float32Array(nv * 3), nt = idx.length / 3, CH = 150000;
  for (let t0 = 0; t0 < nt; t0 += CH) {
    const t1 = Math.min(nt, t0 + CH);
    for (let t = t0; t < t1; t++) {
      const a = idx[3 * t] * 3, b = idx[3 * t + 1] * 3, c = idx[3 * t + 2] * 3;
      const ux = pos[b] - pos[a], uy = pos[b + 1] - pos[a + 1], uz = pos[b + 2] - pos[a + 2], vx = pos[c] - pos[a], vy = pos[c + 1] - pos[a + 1], vz = pos[c + 2] - pos[a + 2];
      const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
      n[a] += nx; n[a + 1] += ny; n[a + 2] += nz; n[b] += nx; n[b + 1] += ny; n[b + 2] += nz; n[c] += nx; n[c + 1] += ny; n[c + 2] += nz;
    }
    onProgress && onProgress('shading ' + Math.round(100 * t1 / nt) + '%'); await tick();
  }
  for (let i = 0; i < nv * 3; i += 3) { const l = Math.hypot(n[i], n[i + 1], n[i + 2]) || 1; n[i] /= l; n[i + 1] /= l; n[i + 2] /= l; }
  return n;
}

// The heavy arrays of a tube mesh: position, index, normal. Loaded once per (run, variant).
export async function loadTubeBase(id, variant, onProgress) {
  onProgress && onProgress('preparing mesh…');
  const r = await fetch('/api/results/' + id + '/mesh' + (variant === 'relaxed' ? '?v=relaxed' : ''));
  if (!r.ok) { const e = await r.json().catch(() => ({})); throw new Error('mesh: ' + (e.detail || 'HTTP ' + r.status)); }
  const buf = await readWithProgress(r, onProgress), hd = new DataView(buf);
  if (hd.getUint32(0, false) !== 0x4b4d5348) throw new Error('mesh: bad header');
  const nv = hd.getUint32(8, true), nf = hd.getUint32(12, true);
  if (buf.byteLength < 16 + nv * 12 + nf * 12) throw new Error('mesh: file is shorter than its header says');
  const pos = new Float32Array(buf, 16, nv * 3), idx = new Uint32Array(buf, 16 + nv * 12, nf * 3);
  return { nv, position: new THREE.BufferAttribute(pos, 3), index: new THREE.BufferAttribute(idx, 1), normal: new THREE.BufferAttribute(await vertexNormals(pos, idx, nv, onProgress), 3) };
}

// A mesh that SHARES the attributes of `base` (no copy) and has its own colour attribute (Uint8 RGB per vertex, or null for a plain colour).
export function tubeMesh(base, rgb) {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', base.position); g.setAttribute('normal', base.normal); g.setIndex(base.index);
  if (rgb) g.setAttribute('color', new THREE.BufferAttribute(rgb, 3, true));
  const m = new THREE.MeshLambertMaterial(rgb ? { vertexColors: true, side: THREE.DoubleSide } : { color: '#d8c8a0', side: THREE.DoubleSide });
  return new THREE.Mesh(g, m);
}
export function recolour(mesh, rgb) {
  if (mesh.geometry.attributes.color) mesh.geometry.attributes.color.array = rgb, mesh.geometry.attributes.color.needsUpdate = true;
  else { mesh.geometry.setAttribute('color', new THREE.BufferAttribute(rgb, 3, true)); mesh.material.dispose(); mesh.material = new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide }); }
}

export async function fetchColors(id, field, variant, kind, extra) {
  const r = await fetch('/api/results/' + id + '/colors?field=' + field + '&variant=' + variant + '&kind=' + kind + (extra || ''));
  if (!r.ok) { const e = await r.json().catch(() => ({})); throw new Error(e.detail || 'HTTP ' + r.status); }
  return { rgb: new Uint8Array(await r.arrayBuffer()), legend: JSON.parse(r.headers.get('X-Legend')) };
}

export function legendHTML(l) {
  const g = l.stops.map(c => 'rgb(' + c.join(',') + ')').join(','), f = x => (+x).toPrecision(3), e = l.end_labels || ['low', 'high'];
  return '<b>' + l.label + '</b><div class="bar" style="background:linear-gradient(to right,' + g + ')"></div><div class="tk"><span>' + f(l.vmin) + ' (' + e[0] + ')</span><span>' + f(l.ticks[1]) + '</span><span>' + f(l.vmax) + ' (' + e[1] + ')</span></div>';
}

#!/usr/bin/env python3
"""round_diag.py - per-round diagnostics of relax_job.py (--diagnostics), written to diagnostics.csv in the run folder. numpy only (no JAX); reads the solver state, changes nothing.

One row per outer round:
  round, E, max_dl0 (largest single rest-length transfer of the round, from the solver)
  n_slips, transferred_total, transferred_max   number of slip transfers of the round, sum and largest of |d| (same unit as the lengths; needs relax_state.migrate statistics)
  drive_strength                                 the persistent drive strength e(k) of the round (0 when the drive is off)
  rest_* / geo_* for the regions all, outer, inner:  mean, min, max over the loops of
      rest = (sum of l0 of the loop) / (the same sum at the start of the run)        -> the rest-length change of the loop (migration, creep)
      geo  = (chord length of the loop now) / (chord length at the start)             -> how much the loop's yarn has been stretched or compressed in space
  share_l0                                       the share of the total length change stored in the rest length:  sum|l0 - l0_start| / (sum|l0 - l0_start| + sum|l - l0|), segments of the whole fabric;
                                                 0 = everything is elastic strain, 1 = everything is rest-length change (UNSOURCED definition; 0 at the start when l0 = l0_start)
  mean_elastic_strain                            mean of l/l0 - 1 over all segments (unweighted)
  net_l0_change                                  (sum l0 - sum l0_start) / sum l0_start (creep does not conserve the total; slip does)
  fold_amplitude                                 peak-to-peak of the stitch-cell mean heights after removing a best-fit plane (the same definition as campaign.py's fold_amplitude_mm; in the units of the
                                                 result); empty when more than 10 % of the cells are empty
  z_range_over_R                                 z range of all vertices over the yarn radius
Regions (UNSOURCED thresholds on the boundary term G_b of the sectioned loop field, scripts/loop_grade.py): outer = G_b >= 0.75 (next to a section boundary), inner = G_b <= 0.25 (far from every boundary),
all = every loop. For single jersey outer is the free-edge ring and inner the centre.
"""
import csv, os
import numpy as np
import loop_grade as lg

REGION_OUTER, REGION_INNER = 0.75, 0.25          # UNSOURCED
REGIONS = ('all', 'outer', 'inner')
COLUMNS = (['round', 'E', 'max_dl0', 'n_slips', 'transferred_total', 'transferred_max', 'drive_strength'] + ['%s_%s_%s' % (k, r, s) for k in ('rest', 'geo') for r in REGIONS for s in ('mean', 'min', 'max')] +
           ['share_l0', 'mean_elastic_strain', 'net_l0_change', 'fold_amplitude', 'z_range_over_R'])


def fold_amplitude(Xs, nx, ny):
    """Peak-to-peak of the cell-mean heights after removing a best-fit plane, on an nx x ny grid of cells over the bounding box (campaign.py's fold_amplitude_mm); None if > 10 % of the cells are empty."""
    P = np.vstack(Xs); x, y, z = P[:, 0], P[:, 1], P[:, 2]; xc, yc = x - x.mean(), y - y.mean()
    A = np.c_[xc, yc, np.ones_like(xc)]; res = z - A @ np.linalg.lstsq(A, z, rcond=None)[0]
    xe = np.linspace(x.min(), x.max(), nx + 1); ye = np.linspace(y.min(), y.max(), ny + 1)
    cnt, _, _ = np.histogram2d(y, x, bins=[ye, xe])
    if (cnt == 0).mean() > 0.10: return None
    s, _, _ = np.histogram2d(y, x, bins=[ye, xe], weights=res); g = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)
    return float(np.nanmax(g) - np.nanmin(g))


class RoundDiag:
    """Built once right after relax_state.load() (and the optional starting grade): remembers the loop mapping, the start rest lengths and the start chord lengths."""
    def __init__(self, sim, folder, meta, s_min=lg.DEFAULT_MIN_STRENGTH):
        f = lg.segment_field(sim, folder, meta, 'both', s_min); G_b = lg.loop_G_parts(f['T'], s_min)[0]
        self.betas, self.nrow, self.ncol, self.courses = f['betas'], f['nrow'], f['ncol'], [int(y.course) for y in sim.yarns]
        self.masks = dict(all=np.ones(G_b.shape, bool), outer=G_b >= REGION_OUTER, inner=G_b <= REGION_INNER)
        self.l0_start = [np.asarray(y.l0, float).copy() for y in sim.yarns]; self.chord_start = [np.linalg.norm(np.diff(y.X, axis=0), axis=1) for y in sim.yarns]
        self.rest_start = self._grid(self.l0_start); self.geo_start = self._grid(self.chord_start); self.sum_l0_start = float(sum(a.sum() for a in self.l0_start))

    def _grid(self, per_yarn):
        g = np.zeros((self.nrow, self.ncol))
        for c, v, beta in zip(self.courses, per_yarn, self.betas): g[c] = lg.loop_sum(v, beta)
        return g

    def row(self, sim, k, E, dmax, slip_stats=None, drive_strength=0.0):
        chord = [np.linalg.norm(np.diff(y.X, axis=0), axis=1) for y in sim.yarns]; l0 = [np.asarray(y.l0, float) for y in sim.yarns]
        rest = self._grid(l0) / self.rest_start; geo = self._grid(chord) / self.geo_start
        out = dict(zip(COLUMNS, [None] * len(COLUMNS))); out.update(round=k + 1, E=E, max_dl0=dmax, drive_strength=drive_strength)
        ss = slip_stats or {}; out.update(n_slips=ss.get('n'), transferred_total=ss.get('total'), transferred_max=ss.get('max'))
        for name, grid in (('rest', rest), ('geo', geo)):
            for r, m in self.masks.items():
                v = grid[m]; out['%s_%s_mean' % (name, r)] = float(v.mean()) if v.size else None; out['%s_%s_min' % (name, r)] = float(v.min()) if v.size else None; out['%s_%s_max' % (name, r)] = float(v.max()) if v.size else None
        dl0 = sum(float(np.abs(a - b).sum()) for a, b in zip(l0, self.l0_start)); el = sum(float(np.abs(c - a).sum()) for c, a in zip(chord, l0))
        out['share_l0'] = dl0 / (dl0 + el) if dl0 + el > 0 else None
        out['mean_elastic_strain'] = float(np.concatenate([c / a - 1.0 for c, a in zip(chord, l0)]).mean())
        out['net_l0_change'] = (float(sum(a.sum() for a in l0)) - self.sum_l0_start) / self.sum_l0_start
        Xs = [y.X for y in sim.yarns]; out['fold_amplitude'] = fold_amplitude(Xs, self.ncol, self.nrow); P = np.vstack(Xs); out['z_range_over_R'] = float(np.ptp(P[:, 2]) / sim.R)
        return out

    @staticmethod
    def append(path, row):
        """Append one row (the header is written once, also after a resume)."""
        new = not os.path.exists(path) or os.path.getsize(path) == 0
        with open(path, 'a', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=COLUMNS)
            if new: w.writeheader()
            w.writerow({k: ('' if row.get(k) is None else row[k]) for k in COLUMNS})

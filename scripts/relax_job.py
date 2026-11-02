#!/usr/bin/env python3
"""relax_job.py - run the baseline -> relaxation transition for one result folder. This is what the UI's Simulate button launches (as a subprocess).
Usage: python3 relax_job.py --folder webui/results/<id> --E <modulus> --mu <friction> --k-contact <kc> [--perturb 0.15] [--outer 40] [--maxiter 300]
Units follow the folder: mm if built with --ne --cpi --wpi (E in N/mm^2 = MPa, kc in N/mm), otherwise R-units (dimensionless).
Writes into the folder: state.json (progress, atomic, every outer iteration), relaxed_axis.csv (course,x,y,z), relaxed_tube.obj. Never touches meta.json.
"""
import argparse, json, math, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import relax_state as rs, relax_solver as sv
from build_knit import write_obj

def main():
    a = argparse.ArgumentParser()
    a.add_argument('--folder', required=True); a.add_argument('--E', type=float, required=True, help='effective yarn modulus')
    a.add_argument('--mu', type=float, required=True, help='yarn-yarn friction coefficient'); a.add_argument('--k-contact', type=float, required=True)
    a.add_argument('--perturb', type=float, default=0.15, help='out-of-plane symmetry-breaking amplitude, fraction of R (numerical device)')
    a.add_argument('--outer', type=int, default=40); a.add_argument('--maxiter', type=int, default=300); a.add_argument('--seed', type=int, default=0)
    a.add_argument('--sides', type=int, default=8)
    a.add_argument('--curl', type=float, default=0.0, help='stored curl, turning angle (rad) per vertex, front loops +, back loops -')
    a.add_argument('--tension', type=float, default=0.0, help='feeder tension in cN (mm results only); the yarn is knitted stretched by this force')
    a.add_argument('--creep-tau', type=float, default=0.0, help='creep relaxation time in hours (0 = no creep)'); a.add_argument('--hours', type=float, default=0.0, help='simulated duration in hours (with creep)')
    g = a.parse_args()
    meta = json.load(open(os.path.join(g.folder, 'meta.json')))
    EA, EI = sv.section(g.E, meta['R'])
    if g.tension and meta.get('structure', {}).get('units') != 'mm': sys.exit('feeder tension needs a result built in mm (--ne --cpi --wpi)')
    if g.curl and 'matrix' not in meta: sys.exit('curl needs the stitch matrix in meta.json (result built by an older pipeline)')
    if g.creep_tau and g.hours <= 0: sys.exit('creep needs --hours > 0')
    ratio = 1.0/(1.0 + 0.01*g.tension/EA)                          # rest length = baseline length / (1 + T/EA), T in N (cN x 0.01)
    p = rs.Params(mu=g.mu, EA=EA, EI=EI, GJ=0.0, k_contact=g.k_contact, outer_max=g.outer, rest_length_ratio=ratio, curl=g.curl, creep_tau=g.creep_tau, hours=g.hours)       # GJ unused: twist not in the energy (see relax_solver)
    a_creep = 1.0 - math.exp(-(g.hours/g.outer)/g.creep_tau) if g.creep_tau else 0.0
    sim = rs.load(g.folder, p); rs.release(sim)
    e0 = sv.energy_parts(sim, g.k_contact, g.E); d0 = sv.min_axis_distance(sim)
    rng = np.random.default_rng(g.seed)
    for y in sim.yarns: y.X[:, 2] += g.perturb*sim.R*rng.standard_normal(len(y.X))
    print('curl %g, feeder tension %g cN (strain %.3g), creep tau %g h over %g h' % (g.curl, g.tension, 0.01*g.tension/EA, g.creep_tau, g.hours), flush=True)
    print('start: E %.6g (stretch %.4g bend %.4g contact %.4g), min axis distance %.4f = %.1f%% of 2R' % (e0['total'], e0['stretch'], e0['bend'], e0['contact'], d0, 100*d0/(2*sim.R)), flush=True)
    t0 = time.time()
    def relax(s):
        sv.relax(s, g.E, g.k_contact, maxiter=g.maxiter, verbose=True)
        if a_creep: sv.creep(s, a_creep)
    energy = lambda s: sv.energy_parts(s, g.k_contact, g.E)['total']
    rs.run(sim, relax, energy)
    e1 = sv.energy_parts(sim, g.k_contact, g.E); d1 = sv.min_axis_distance(sim)
    tot_l0 = sum(float(y.l0.sum()) for y in sim.yarns)
    print('end (%s, %.0f s): E %.6g (stretch %.4g bend %.4g contact %.4g), min axis distance %.4f = %.1f%% of 2R, total rest length %.6g' % (sim.state.value, time.time()-t0, e1['total'], e1['stretch'], e1['bend'], e1['contact'], d1, 100*d1/(2*sim.R), tot_l0))
    np.savetxt(os.path.join(g.folder, 'relaxed_axis.csv'), np.vstack([np.c_[np.full(len(y.X), y.course), y.X] for y in sim.yarns]), delimiter=',', header='course,x,y,z', comments='', fmt=['%d', '%.6f', '%.6f', '%.6f'])
    write_obj(os.path.join(g.folder, 'relaxed_tube.obj'), [y.X for y in sim.yarns], sim.R, g.sides, 'relaxed state of %s' % os.path.basename(os.path.abspath(g.folder)))
    d = json.load(open(os.path.join(g.folder, 'state.json'))); d['summary'] = dict(start=e0, end=e1, min_dist_over_2R_start=d0/(2*sim.R), min_dist_over_2R_end=d1/(2*sim.R))
    json.dump(d, open(os.path.join(g.folder, 'state.json.tmp'), 'w'), indent=1); os.replace(os.path.join(g.folder, 'state.json.tmp'), os.path.join(g.folder, 'state.json'))
    sys.exit(0 if sim.state == rs.S.SETTLED else 2)

if __name__ == '__main__':
    main()
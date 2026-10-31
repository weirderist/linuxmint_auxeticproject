#!/usr/bin/env python3
"""relax_job.py - run the baseline -> relaxation transition for one result folder. This is what the UI's Simulate button launches (as a subprocess).
Usage: python3 relax_job.py --folder webui/results/<id> --E <modulus> --mu <friction> --k-contact <kc> [--perturb 0.02] [--outer 40] [--maxiter 300]
Units follow the folder: mm if built with --ne --cpi --wpi (E in N/mm^2 = MPa, kc in N/mm), otherwise R-units (dimensionless).
Writes into the folder: state.json (progress, atomic, every outer iteration), relaxed_axis.csv (course,x,y,z), relaxed_tube.obj. Never touches meta.json.
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import relax_state as rs, relax_solver as sv
from build_knit import write_obj

def main():
    a = argparse.ArgumentParser()
    a.add_argument('--folder', required=True); a.add_argument('--E', type=float, required=True, help='effective yarn modulus')
    a.add_argument('--mu', type=float, required=True, help='yarn-yarn friction coefficient'); a.add_argument('--k-contact', type=float, required=True)
    a.add_argument('--perturb', type=float, default=0.02, help='out-of-plane symmetry-breaking amplitude, fraction of R (numerical device)')
    a.add_argument('--outer', type=int, default=40); a.add_argument('--maxiter', type=int, default=300); a.add_argument('--seed', type=int, default=0)
    a.add_argument('--sides', type=int, default=8)
    g = a.parse_args()
    EA, EI = sv.section(g.E, json.load(open(os.path.join(g.folder, 'meta.json')))['R'])
    p = rs.Params(mu=g.mu, EA=EA, EI=EI, GJ=0.0, k_contact=g.k_contact, outer_max=g.outer)       # GJ unused: twist not in the energy (see relax_solver)
    sim = rs.load(g.folder, p); rs.release(sim)
    e0 = sv.energy_parts(sim, g.k_contact, g.E); d0 = sv.min_axis_distance(sim)
    rng = np.random.default_rng(g.seed)
    for y in sim.yarns: y.X[:, 2] += g.perturb*sim.R*rng.standard_normal(len(y.X))
    print('start: E %.6g (stretch %.4g bend %.4g contact %.4g), min axis distance %.4f = %.1f%% of 2R' % (e0['total'], e0['stretch'], e0['bend'], e0['contact'], d0, 100*d0/(2*sim.R)), flush=True)
    t0 = time.time()
    relax = lambda s: sv.relax(s, g.E, g.k_contact, maxiter=g.maxiter, verbose=True)
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

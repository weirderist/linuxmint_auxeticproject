#!/usr/bin/env python3
"""fold_scan.py - does the yarn model prefer a folded fabric, without any shell? Runs relax_job.py from several bent starts (--warp-kx / --warp-ky, curvatures in 1/mm) on copies of one result folder,
then prints, for each start, the relaxed energy and the curvature the relaxed fabric ended with (fold_metric fit). The lowest energy is the model's preferred state among the starts you tried;
if it is the flat start (0 0), the model does not fold with those inputs. Needs the same JAX environment as relax_job.py.
Usage: python3 fold_scan.py --folder results/<id> --kx 0,0.02,-0.02 [--ky 0] [-- any relax_job.py options, e.g. -- --E 3700 --ref-scale 0 --bend-scale 0.003 --outer 6 --maxiter 300 --macro]
Set FOLD_SCAN_RUNNER to a command prefix if relax_job.py must be started differently (default: this Python).
Energies are only comparable between starts run with the same options. Runs are sequential; each is a full relax_job run (the original folder is not touched)."""
import argparse, itertools, os, shutil, subprocess, sys, tempfile, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fold_metric import fit
def main():
    a = argparse.ArgumentParser(); a.add_argument('--folder', required=True); a.add_argument('--kx', default='0'); a.add_argument('--ky', default='0'); a.add_argument('rest', nargs=argparse.REMAINDER)
    g = a.parse_args(); rest = [r for r in g.rest if r != '--']; here = os.path.dirname(os.path.abspath(__file__)); rows = []
    for kx, ky in itertools.product([float(v) for v in g.kx.split(',')], [float(v) for v in g.ky.split(',')]):
        d = tempfile.mkdtemp(prefix='foldscan_'); shutil.copytree(g.folder, d, dirs_exist_ok=True)
        pr = subprocess.run(os.environ.get('FOLD_SCAN_RUNNER', sys.executable).split() + [os.path.join(os.path.dirname(here), 'yarn_model', 'relax_job.py'), '--folder', d, '--warp-kx', repr(kx), '--warp-ky', repr(ky)] + rest, stdout=subprocess.DEVNULL)
        if pr.returncode not in (0, 2): sys.exit('relax_job.py failed (exit %d) for the start kx %g ky %g; run it by hand with the same options to see why' % (pr.returncode, kx, ky))     # exit 2 = the run ended as "failed" (not settled): its relaxed state is still used
        meta = json.load(open(os.path.join(d, 'meta.json'))); e = meta['simulations'][-1].get('energy_final', None)
        if e is None: e = meta['simulations'][-1]['rounds'][-1]['E']
        r = fit(np.loadtxt(os.path.join(d, 'relaxed_axis.csv'), delimiter=',', skiprows=1)[:, 1:]); rows.append((e, kx, ky, r['kx'], r['ky'])); print('start kx %+.4f ky %+.4f -> E %.6g, relaxed kx %+.4f ky %+.4f /mm' % (kx, ky, e, r['kx'], r['ky']), flush=True)
    rows.sort(); print('\nlowest energy: start kx %+.4f ky %+.4f (E %.6g), relaxed curvature kx %+.4f ky %+.4f /mm' % (rows[0][1], rows[0][2], rows[0][0], rows[0][3], rows[0][4]))
if __name__ == '__main__': main()

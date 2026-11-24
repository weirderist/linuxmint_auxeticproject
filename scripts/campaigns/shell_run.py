#!/usr/bin/env python3
"""shell_run.py - ONE pure-shell run (no yarn mesh, no build_knit). Called by shell_campaign.py; can also be run by hand.

    python3 shell_run.py --settings settings.json --out DIR

Reads the normalised settings (shell_campaign.normalise), calls shell_sim.fold exactly as relax_job.run_shell does (kappa 1.2/C, h 0.5/1.2 C, curvatures kx = -k x kappa_scale, ky = +k x kappa_scale,
so settings of the 13 Nov success shell reproduce its shell.npz bit for bit), and writes into DIR: shell.npz (same keys as relax_job), shell.bin (KMSH v1 mesh, shell_metrics), shell.log (printed by fold),
metrics.json LAST (its presence = the run is complete). Nothing outside DIR is written.
Test hook (never set in real campaigns): environment SHELL_RUN_STUB=1 replaces the solver with a tiny synthetic fold; settings 'stub_seconds', 'stub_hang', 'stub_fail' steer it.
"""
import argparse, json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
_S = os.path.dirname(HERE)
if os.path.basename(_S) == 'scripts': sys.path.extend(os.path.join(_S, _d) for _d in sorted(os.listdir(_S)) if _d not in ('legacy', '__pycache__') and os.path.isdir(os.path.join(_S, _d)))          # the other folders of scripts/ (not in a flat frozen copy)
import shell_metrics as sm

SHELL_KAPPA_C, SHELL_H_C = 1.2, 0.5 / 1.2          # as relax_job.py: Niu et al. kappa = 1, h = 0.5 for a stitch 1.4 x 1.2, written per course spacing C


def wjson(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f: json.dump(obj, f, indent=1); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


FILES = ('campaign.py', 'shell_campaign.py', 'shell_run.py', 'shell_metrics.py', 'shell_sim.py', 'shell_policy.py')


def provenance():
    """What actually ran: sha256 of the code files next to this script, compared with the frozen MANIFEST.json (if this is a campaign's frozen copy), git commit, library versions."""
    import hashlib, importlib.metadata as im, platform, subprocess
    sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
    GITDIR = os.path.dirname(HERE) if os.path.basename(os.path.dirname(HERE)) == 'scripts' else HERE          # git is asked about the whole of scripts/, as before the sort into folders
    def where(f):          # next to this script (flat copy) or in a sibling folder of scripts/ (shell_sim.py and shell_metrics.py live in scripts/shell_model/)
        p = os.path.join(HERE, f)
        if os.path.isfile(p): return p
        for d in sorted(os.listdir(os.path.dirname(HERE))):
            q = os.path.join(os.path.dirname(HERE), d, f)
            if os.path.isfile(q): return q
        return p
    files = {f: sha(where(f)) for f in FILES if os.path.isfile(where(f))}
    try: man = json.load(open(os.path.join(HERE, 'MANIFEST.json')))
    except (OSError, ValueError): man = None
    commit, dirty = (man or {}).get('git_commit', ''), (man or {}).get('git_dirty')
    if not man:                                                       # run by hand from scripts/: ask git (best effort)
        try:
            commit = subprocess.run(['git', '-C', GITDIR, 'rev-parse', 'HEAD'], capture_output=True, text=True, timeout=20).stdout.strip()
            dirty = bool(subprocess.run(['git', '-C', GITDIR, 'status', '--porcelain', '--', '.'], capture_output=True, text=True, timeout=20).stdout.strip())
        except Exception: pass
    out = dict(frozen=bool(man), git_commit=commit, git_dirty=dirty, shell_sim_sha256=files.get('shell_sim.py', ''), files=files, matches_manifest=(files == man.get('files')) if man else None,
               manifest_frozen=(man or {}).get('frozen'), python=platform.python_version(), host=platform.node(), t=time.strftime('%Y-%m-%dT%H:%M:%S'))
    for p in ('numpy', 'scipy', 'jax', 'jaxlib'):
        try: out[p] = im.version(p)
        except Exception: out[p] = ''
    return out


def stub_fold(s, W, C):
    """Synthetic fold for tests: a sinusoidal sheet of the right grid size. Not physics."""
    M = s['matrix']; nx, ny = len(M[0]) * s['tiles'], len(M) * s['tiles']; sub = s['sub']
    time.sleep(float(s.get('stub_seconds', 0)))
    while s.get('stub_hang'): time.sleep(1)
    if s.get('stub_fail'): sys.exit(3)
    jj, ii = np.meshgrid(np.arange(ny * sub + 1), np.arange(nx * sub + 1), indexing='ij')
    X = np.stack([ii * W / sub + 0.5 * np.sin(jj / 7.0), jj * C / sub, 2.0 * np.sin(jj / 9.0) * np.cos(ii / 11.0)], -1)
    return X, nx, ny, [(1.0, 0.0, 1, float(np.ptp(X[..., 2])))]


def main():
    a = argparse.ArgumentParser(); a.add_argument('--settings', required=True); a.add_argument('--out', required=True); g = a.parse_args()
    s = json.load(open(g.settings)); t0 = time.time(); os.makedirs(g.out, exist_ok=True)
    W, C = 25.4 / s['wpi'], 25.4 / s['cpi']; kappa, h = SHELL_KAPPA_C / C, SHELL_H_C * C * s['h_scale']
    kxy = (s['kappa_scale'] * (-SHELL_KAPPA_C / C), s['kappa_scale'] * (SHELL_KAPPA_C / C))      # same expression order as relax_job.main with --shell-kappa guess
    if os.environ.get('SHELL_RUN_STUB') == '1': X, nx, ny, log = stub_fold(s, W, C)
    else:
        import shell_sim as ss
        print('shell stage: %d x %d stitches, sub %d, %d growth steps, h %.4g, digit-1 curvature kx %.4g ky %.4g (1/mm, > 0 toward +z)' % (len(s['matrix'][0]) * s['tiles'], len(s['matrix']) * s['tiles'], s['sub'], s['steps'], h, *kxy), flush=True)
        X0, X, tri, (nx, ny), log = ss.fold(s['matrix'], W, C, kappa=kappa, h=h, nu=s['nu'], sub=s['sub'], ntile=(s['tiles'], s['tiles']), steps=s['steps'], maxiter=s['iter'], springs=s['springs'], seed=s['seed'], verbose=True, kx=kxy[0], ky=kxy[1])
    Xg = np.asarray(X).reshape(ny * s['sub'] + 1, nx * s['sub'] + 1, 3)
    with open(os.path.join(g.out, 'shell.npz.tmp'), 'wb') as f: np.savez(f, X=Xg, W=W, C=C, sub=s['sub'], nx=nx, ny=ny, kappa=kappa, h=h)
    os.replace(os.path.join(g.out, 'shell.npz.tmp'), os.path.join(g.out, 'shell.npz'))
    with open(os.path.join(g.out, 'shell.bin.tmp'), 'wb') as f: f.write(sm.kmsh_bytes(Xg))
    os.replace(os.path.join(g.out, 'shell.bin.tmp'), os.path.join(g.out, 'shell.bin'))
    wjson(os.path.join(g.out, 'provenance.json'), provenance())
    m = sm.describe(Xg, W, C, s['sub'], float(s.get('fold_min_turn_rad', 0.2)))
    m.update(steps_at_cap=int(sum(1 for r in log if r[2] >= s['iter'])), steps=len(log), iterations_total=int(sum(r[2] for r in log)), U_final=float(log[-1][1]), W=W, C=C, kappa=kappa, h=h, solver_seconds=round(time.time() - t0, 1))
    wjson(os.path.join(g.out, 'metrics.json'), m)                      # last: its presence marks a complete run
    print('shell run complete in %.0f s: plan %.2f x %.2f mm, thickness %.2f mm' % (time.time() - t0, m['plan_x'], m['plan_y'], m['thickness']), flush=True)


if __name__ == '__main__':
    main()

"""probe_kernel.py - runs ON KAGGLE (2026-11-27 feasibility probe): which JAX devices exist, and the time of one small shell2-type run (6x6 matrix, tiles 2, FIRE 2000, plate) from the frozen shell2 code.
Input: the private dataset with the frozen code (mounted under /kaggle/input/). Output: /kaggle/working/probe.json and the run folder."""
import glob, json, os, subprocess, sys, time
t0 = time.time(); out = dict(t_start=time.strftime('%Y-%m-%dT%H:%M:%S'))
code = os.path.dirname(sorted(glob.glob('/kaggle/input/**/shell_run.py', recursive=True))[0]); out['code_dir'] = code
try:
    import jax; out['jax'] = jax.__version__; out['devices'] = [str(d) for d in jax.devices()]
except Exception as e:
    out['jax_import_error'] = repr(e)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'jax'], check=False); import jax; out['jax'] = jax.__version__; out['devices'] = [str(d) for d in jax.devices()]
try: out['nproc'] = os.cpu_count(); out['gpu'] = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True).stdout.strip()
except Exception as e: out['gpu'] = repr(e)
S = dict(matrix=[[1, 1, 1, 2, 2, 2]] * 3 + [[2, 2, 2, 1, 1, 1]] * 3, tiles=2, cpi=22.0, wpi=16.0, ne=12.0, kappa_scale=1.0, h_scale=1.0, nu=0.4, springs=0.0, sub=2, steps=15, iter=2000, seed=0,
         bending='plate', solver='fire', check_stability=True)
os.makedirs('/kaggle/working/run', exist_ok=True); json.dump(S, open('/kaggle/working/run/settings.json', 'w'))
for name, envx in (('default', {}), ('cpu', {'JAX_PLATFORMS': 'cpu'})):
    t = time.time(); p = subprocess.run([sys.executable, os.path.join(code, 'shell_run.py'), '--settings', '/kaggle/working/run/settings.json', '--out', '/kaggle/working/run/out_' + name],
                                        capture_output=True, text=True, env=dict(os.environ, **envx))
    m = {}
    try: m = json.load(open('/kaggle/working/run/out_%s/metrics.json' % name))
    except Exception: pass
    out[name] = dict(rc=p.returncode, seconds=round(time.time() - t, 1), tail=(p.stdout + p.stderr)[-1500:], U_final=m.get('U_final'), stability_min=m.get('stability_min'), fold_direction=m.get('fold_direction'))
out['seconds_total'] = round(time.time() - t0, 1); json.dump(out, open('/kaggle/working/probe.json', 'w'), indent=1); print(json.dumps(out, indent=1))

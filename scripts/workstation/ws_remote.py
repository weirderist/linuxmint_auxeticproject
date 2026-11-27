#!/usr/bin/env python3
"""ws_remote.py - SERVER side of the workstation worker (2026-11-27). Called by ws_agent.py over ssh; every command prints one JSON object.

    ws_remote.py status  <campaign> --worker W [--current RID]          config, allowance left, leases held, frozen-code manifest, inbox path; records the agent heartbeat
    ws_remote.py claim   <campaign> --worker W --ttl S                   one run (a volunteered id first, else the head of the queue) if the workstation is allowed and has allowance left
    ws_remote.py renew   <campaign> --worker W --ids A,B --ttl S
    ws_remote.py release <campaign> --worker W --ids A,B [--why TEXT]
    ws_remote.py import  <campaign> --worker W --id RID --src DIR [--wall S]   DIR is inside <campaign>/workstation/inbox/

K decides from the /shell page (Workstation box): <campaign>/workstation/config.json {enabled, n (how many runs), ids (specific runs, optional), seq}. Every new decision gets a new seq, and the
allowance (n) counts from it; 'Cancel' sets enabled false: the agent stops its run and its lease is released, so the run goes back to the server and Kaggle. Leases are short (the agent renews them
every poll), so after a power cut on the workstation the run is queued again by itself."""
import argparse, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'campaigns'))
import shell_campaign as sc, campaign as cm          # noqa: E402


def wsdir(c): d = os.path.join(c.d, 'workstation'); os.makedirs(os.path.join(d, 'inbox'), exist_ok=True); return d
def config(c): return cm.rjson(os.path.join(wsdir(c), 'config.json'), {}) or {}
def state(c): return cm.rjson(os.path.join(wsdir(c), 'state.json'), {}) or {}

def allowance(c):
    cf, st = config(c), state(c)
    if st.get('seq') != cf.get('seq'): st = dict(seq=cf.get('seq'), granted=0)
    return cf, st, max(0, int(cf.get('n') or 0) - int(st.get('granted') or 0))

def main(argv=None):
    a = argparse.ArgumentParser(); a.add_argument('cmd', choices=['status', 'claim', 'renew', 'release', 'import']); a.add_argument('campaign'); a.add_argument('--worker', required=True)
    a.add_argument('--ttl', type=float, default=1800); a.add_argument('--ids', default=''); a.add_argument('--id', default=''); a.add_argument('--src', default=''); a.add_argument('--wall', type=float)
    a.add_argument('--why', default=''); a.add_argument('--current', default=''); g = a.parse_args(argv)
    if not g.worker.startswith('workstation'): raise SystemExit('worker names of the workstation start with "workstation"')
    c = sc.Camp(g.campaign)
    if not c.exists(): print(json.dumps(dict(ok=False, error='no such campaign'))); return 2
    d = wsdir(c); out = dict(ok=True)
    if g.cmd == 'status':
        cf, st, left = allowance(c); man = c.manifest() or {}
        cm.wjson(os.path.join(d, 'agent.json'), dict(worker=g.worker, t=cm.now(), t_epoch=time.time(), current=g.current or None))
        held = [r for r, L in c.active_leases().items() if L.get('worker') == g.worker]
        out.update(config=cf, allowance_left=left, held=held, control=c.get_control(), manifest=dict(files=man.get('files') or {}, shell_sim_sha256=man.get('shell_sim_sha256')), code_dir=c.code_dir(),
                   inbox=os.path.join(d, 'inbox'), thresholds=c.spec_().get('thresholds') or {})
    elif g.cmd == 'claim':
        cf, st, left = allowance(c); items = []
        if cf.get('enabled') and left > 0 and c.get_control() != 'stop':
            want = [x for x in (cf.get('ids') or []) if isinstance(x, str)]
            for rid in want:          # volunteered runs first, one at a time (a run already done or held elsewhere is skipped)
                items = c.claim_ids(g.worker, [rid], g.ttl, 'workstation')
                if items: break
            if not items and (not want or cf.get('then_queue')): items = c.claim(g.worker, 1, g.ttl, 'workstation')          # no ids given (or 'then the queue'): the head of the queue
            if items: st['granted'] = int(st.get('granted') or 0) + len(items); cm.wjson(os.path.join(d, 'state.json'), st)
        thr = (c.spec_().get('thresholds') or {}).get('fold_min_turn_rad', 0.2)
        out.update(items=[dict(id=x['id'], settings=dict(x['settings'], fold_min_turn_rad=thr), expires=x['lease']['expires']) for x in items], allowance_left=allowance(c)[2])
    elif g.cmd == 'renew': out.update(renewed=c.renew([x for x in g.ids.split(',') if x], g.worker, g.ttl))
    elif g.cmd == 'release': out.update(released=c.release([x for x in g.ids.split(',') if x], g.worker, g.why or 'workstation'))
    elif g.cmd == 'import':
        src = os.path.realpath(g.src); inbox = os.path.realpath(os.path.join(d, 'inbox'))
        if os.path.commonpath([src, inbox]) != inbox: out.update(ok=False, error='src must be inside the inbox'); print(json.dumps(out)); return 1
        try: out.update(result=c.import_result(g.id, src, g.worker, wall_s=g.wall))
        except ValueError as e:
            out.update(ok=False, error=str(e)); c.release([g.id], g.worker, 'import refused'); sc.append_line(c.index, dict(id=g.id, status='failed', t=cm.now(), reason='workstation import refused: %s' % e, worker=g.worker))
        else: c.write_results()
    print(json.dumps(out)); return 0 if out.get('ok') else 1

if __name__ == '__main__':
    sys.exit(main())

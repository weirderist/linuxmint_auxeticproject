#!/usr/bin/env python3
"""kaggle_cli.py - run the Kaggle command line tool with the credentials of the project's .env (KAGGLE_USERNAME, KAGGLE_KEY), without printing them.

    python3 scripts/kaggle/kaggle_cli.py <kaggle arguments...>        e.g.  kernels status <user>/<slug>
Library use: from kaggle_cli import run, username.  The CLI lives in its own venv (~/kaggle_venv, installed 2026-11-27 so the project venv is untouched); override with KAGGLE_BIN.
Only lines that start with KAGGLE_ are read from .env; the key is removed from every output this module returns or prints."""
import os, subprocess, sys

ROOT = os.environ.get('AUXETIC_ROOT') or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENV = os.environ.get('AUXETIC_ENV') or os.path.join(ROOT, '.env')
BIN = os.environ.get('KAGGLE_BIN') or os.path.expanduser('~/kaggle_venv/bin/kaggle')


def load_env():
    """KAGGLE_* values from .env (KEY=VALUE lines, optional quotes, optional 'export '). Never printed."""
    out = {}
    try:
        for line in open(ENV, encoding='utf8'):
            line = line.strip()
            if line.startswith('export '): line = line[7:].strip()
            if not line.startswith('KAGGLE_') or '=' not in line: continue
            k, v = line.split('=', 1); out[k.strip()] = v.strip().strip('"').strip("'")
    except OSError: pass
    k = out.get('KAGGLE_KEY', '')
    if k.startswith('KGAT_') and 'KAGGLE_API_TOKEN' not in out: out['KAGGLE_API_TOKEN'] = out.pop('KAGGLE_KEY')          # a new-style API token (Kaggle CLI 2.x) stored under the old name
    return out


def username():
    return load_env().get('KAGGLE_USERNAME') or os.environ.get('KAGGLE_USERNAME', '')


def _redact(text, env):
    for k, v in env.items():
        if 'KEY' in k or 'TOKEN' in k:
            if v: text = text.replace(v, '<redacted>')
    return text


def run(args, timeout=600, cwd=None):
    """(returncode, output) of `kaggle <args>`; output = stdout + stderr with the key redacted. Network errors give a non-zero code, never an exception."""
    env = load_env(); full = dict(os.environ); full.update(env)
    try:
        p = subprocess.run([BIN] + list(args), capture_output=True, text=True, timeout=timeout, env=full, cwd=cwd)
        return p.returncode, _redact((p.stdout or '') + (p.stderr or ''), env)
    except subprocess.TimeoutExpired: return 124, 'kaggle %s: timed out after %d s' % (args[:2], timeout)
    except OSError as e: return 127, 'kaggle CLI not runnable (%s): %s' % (BIN, e)


if __name__ == '__main__':
    rc, out = run(sys.argv[1:]); print(out, end=''); sys.exit(rc)

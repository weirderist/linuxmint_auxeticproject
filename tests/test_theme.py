#!/usr/bin/env python3
"""Tests of the shared stylesheet webui/theme.css and of the pages that use it (presentation rules): dark only, every colour defined once in :root, no colour literal and no undefined variable in the
page files, the stylesheet is served. No server, no solver. Run from the repository root: python tests/test_theme.py"""
import os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); W = os.path.join(ROOT, 'webui')
sys.path.insert(0, W)
os.environ.setdefault('CAMPAIGN_DIR', os.path.join(ROOT, 'smoke_test', 'campaigns'))
bad = []
def check(name, ok, info=''):
    print('%s  %s %s' % ('PASS' if ok else 'FAIL', name, info)); ok or bad.append(name)
read = lambda f: open(os.path.join(W, f), encoding='utf8').read()
css = read('theme.css'); html = read('campaigns.html'); js = read('campaigns.js')
root = re.search(r':root\s*\{(.*?)\n\}', css, re.S).group(1); defined = set(re.findall(r'(--[a-z0-9-]+)\s*:', root))
check('theme.css is dark only: color-scheme dark, no automatic scheme switch, no data-theme', 'color-scheme: dark' in css and 'prefers-color-scheme' not in css and 'data-theme' not in css and 'prefers-color-scheme' not in html)
check('background is pure black and text white, defined as variables', re.search(r'--bg:\s*#000\b', css) and re.search(r'--fg:\s*#fff\b', css))
check('every colour literal in theme.css sits inside :root (rules use only variables)', not re.findall(r'#[0-9a-fA-F]{3,8}\b|rgba?\(|hsla?\(', css.replace(root, '')))
for f, src in (('campaigns.html', html), ('campaigns.js', js)):
    lits = re.findall(r"['\"(]#[0-9a-fA-F]{3,8}\b|rgba?\(|hsla?\(|\borange\b|\bred\b|\bbrown\b", src)
    check('%s has no colour literal (only var(--...) from theme.css)' % f, not lits, str(lits[:4]))
check('campaigns.html links /theme.css and has no inline <style> block or style attribute with a colour', '<link rel="stylesheet" href="/theme.css">' in html and '<style' not in html and not re.findall(r'style="[^"]*(?:color|background)', html))
used = set(re.findall(r'var\((--[a-z0-9-]+)', html + js + css)); missing = sorted(used - defined - {'--c'})
check('every var(--x) used by the page, script and stylesheet is defined in theme.css (--c is the legend dot set inline)', not missing, str(missing))
check('the old light/dark variables are gone from the page files (--ok, --crit, --warn, --s1, --paused, --gray)', not re.findall(r'var\(--(?:ok|crit|warn|s1|paused|gray|flagbg|yelbg)\)', html + js))
check('no coloured table cells: the f-ns / f-ip / f-nf / f-fail fills are removed', not re.findall(r'f-ns|f-ip|f-nf|f-fail', js + css + html))
check('charts: one aspect ratio, tick counts and marker size set once for all charts', 'aspect: 0.58' in js and 'yTicks: 5' in js and 'xTicks: 5' in js and js.count('const CH =') == 1)
check('summary severity is a bold lead-in word (not red text)', 'Warning.' in js and 'Note.' in js and 'li.warn { color' not in css)
check('flags legend is plain text', 'id="rleg"' in html and 'class="legend-text"' in html)
check('datasets checker messages are collapsed to "N warnings" lists', 'details class="msgs"' in js and 'warning' in js)
try:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    import campaign_api as ca
    app = FastAPI(); ca.setup(app, lambda: False); cl = TestClient(app); r = cl.get('/theme.css')
    check('/theme.css is served as text/css and equals the file', r.status_code == 200 and r.headers['content-type'].startswith('text/css') and r.text == css.replace('\r\n', '\n') or r.text == css)
    check('/campaigns links it and /campaigns.js is served', 'href="/theme.css"' in cl.get('/campaigns').text and cl.get('/campaigns.js').status_code == 200)
except ImportError: print('SKIP  fastapi not installed: route test skipped')
print('\n' + ('ALL OK' if not bad else 'FAILURES: ' + ', '.join(bad))); sys.exit(1 if bad else 0)

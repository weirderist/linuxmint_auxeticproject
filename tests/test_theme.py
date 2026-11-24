#!/usr/bin/env python3
"""Tests of the shared stylesheet webui/theme.css, the shared chart code webui/chartkit.js and the pages that use them (Campaigns page and the pure-shell pages): presentation rules only.
Dark only; every colour defined once in :root; no colour literal, no inline colour and no undefined variable in the page files; larger fonts; crisp, unscaled, unrotated charts of one size; the strain
legend wording. No server, no solver. Run from the repository root: python tests/test_theme.py"""
import os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); W = os.path.join(ROOT, 'webui')
sys.path.insert(0, W); sys.path.insert(0, os.path.join(ROOT, 'scripts')); import scriptdirs
os.environ.setdefault('CAMPAIGN_DIR', os.path.join(ROOT, 'smoke_test', 'campaigns'))
bad = []
def check(name, ok, info=''):
    print('%s  %s %s' % ('PASS' if ok else 'FAIL', name, info)); ok or bad.append(name)
read = lambda f: open(os.path.join(W, f), encoding='utf8').read()
css = read('theme.css'); kit = read('chartkit.js')
PAGES = {f: read(f) for f in ('campaigns.html', 'campaigns.js', 'shell.html', 'shell.js', 'shellview.html', 'shellview.js')}
root = re.search(r':root\s*\{(.*?)\n\}', css, re.S).group(1); defined = set(re.findall(r'(--[a-z0-9-]+)\s*:', root))
var = lambda name: re.search(r'%s:\s*([^;]+);' % re.escape(name), root).group(1).strip()
rgb = lambda h: tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))

# ---- scheme
check('dark only: color-scheme dark, no automatic scheme switch, no data-theme', 'color-scheme: dark' in css and 'prefers-color-scheme' not in css and 'data-theme' not in css and all('prefers-color-scheme' not in s for s in PAGES.values()))
bg, card = rgb(var('--bg')), rgb(var('--card'))
check('page background is a soft dark grey (about #1f1f1e to #262624, UNSOURCED), not pure black', all(0x1f <= c <= 0x26 for c in bg) and var('--bg') != '#000', var('--bg'))
check('cards are slightly lighter than the page, text is white', sum(card) > sum(bg) and sum(card) - sum(bg) < 60 and var('--fg') == '#fff', '%s on %s' % (var('--card'), var('--bg')))
check('borders and muted text are grey (R = G = B within 12)', all(max(rgb(var(v))) - min(rgb(var(v))) <= 12 for v in ('--line', '--line-strong', '--mut', '--hover', '--sel', '--track')))
fill, idle, run = rgb(var('--fill')), rgb(var('--st-idle')), rgb(var('--st-run'))
check('progress bar fill is a light green; idle pill light yellow; running pill green (variables)', fill[1] > fill[0] and fill[1] > fill[2] and min(fill) > 120 and idle[0] > 200 and idle[1] > 200 and idle[2] < idle[0] and run[1] > run[0] and run[1] > run[2], '%s %s %s' % (var('--fill'), var('--st-idle'), var('--st-run')))
check('pill classes use the status variables; other states stay muted', re.search(r'\.pill\.idle \{[^}]*var\(--st-idle\)', css) and re.search(r'\.pill\.running, \.pill\.active \{[^}]*var\(--st-run\)', css) and re.search(r'\.pill::before \{[^}]*var\(--st-other\)', css))
check('every colour literal in theme.css sits inside :root (rules use only variables)', not re.findall(r'#[0-9a-fA-F]{3,8}\b|rgba?\(|hsla?\(', css.replace(root, '')))
px = lambda n: int(var(n).replace('px', ''))
check('fonts are 2 px larger than round 1 (16 / 14 / 13 px)', (px('--fs'), px('--fs-sm'), px('--fs-xs')) == (16, 14, 13), str((px('--fs'), px('--fs-sm'), px('--fs-xs'))))

# ---- the page files: no colour literal, no inline colour, no style block, no undefined variable
for f, src in PAGES.items():
    lits = re.findall(r"['\"(]#[0-9a-fA-F]{3,8}\b|\brgba?\(|hsla?\(|\borange\b|\bbrown\b", src.replace("'rgb(' + c.join(',') + ')'", ''))
    check('%s has no colour literal (only var(--...) from theme.css)' % f, not lits, str(lits[:3]))
for f in ('campaigns.html', 'shell.html', 'shellview.html'):
    s = PAGES[f]
    check('%s links /theme.css, has no <style> block and no style attribute' % f, '<link rel="stylesheet" href="/theme.css">' in s and '<style' not in s and 'style="' not in s)
for f in ('campaigns.js', 'shell.js', 'chartkit.js'):
    s = PAGES.get(f) or kit
    check('%s has no inline style attribute' % f, 'style="' not in s)
inl = re.findall(r'style="[^"]*"', PAGES['shellview.js'])
check('shellview.js has exactly one inline style: the colour ramp the server sends with the legend data', len(inl) == 1 and 'linear-gradient' in inl[0], str(inl))
allsrc = css + kit + ''.join(PAGES.values()); used = set(re.findall(r'var\((--[a-z0-9-]+)', allsrc)); missing = sorted(used - defined - {'--c'})          # --c is the per-class colour set by .cl-* / .g* in theme.css
check('every var(--x) used anywhere is defined in :root (--c is set per class by the chart colour classes)', not missing, str(missing))
check('the old round-1 variables are gone', not re.findall(r'var\(--(?:ok|crit|warn|s1|paused|gray|flagbg|yelbg|dot-run|chart-ok)\)', allsrc))

# ---- charts: one renderer, native pixel size, crisp, unrotated, thicker lines
check('both pages draw charts with ChartKit.scatter and size them with ChartKit.size; no private chart code is left', 'ChartKit.scatter' in PAGES['campaigns.js'] and 'ChartKit.scatter' in PAGES['shell.js'] and 'function chart(' not in PAGES['campaigns.js'] and 'function ticks(' not in PAGES['campaigns.js'] and 'function ticks(' not in PAGES['shell.js'])
check('one aspect ratio, tick counts and marker size for every chart (defined once in chartkit.js)', 'aspect: 0.58' in kit and 'yTicks: 5' in kit and 'xTicks: 5' in kit and 'r: 4' in kit and kit.count('const CH =') == 1)
check('SVG is drawn at its measured pixel size: width/height attributes, no viewBox scaling by CSS, no rotate(), lines on half pixels', 'clientWidth' in kit and "width=\"' + W" in kit and 'rotate(' not in kit and 'transform' not in kit and 'half(' in kit and not re.search(r'\.chart[^{]*\{[^}]*(?:transform|zoom|scale)', css))
check('chart text is 13 px and lines use crisp edges; data line 2.25 px (1.5 x the old 1.5), axes 1.5 px', 'font-size: 13px' in css and 'shape-rendering: crispEdges' in css and re.search(r'\.chart \.dl \{[^}]*stroke-width: 2\.25', css) and re.search(r'\.chart \.ax \{[^}]*stroke-width: 1\.5', css))
check('chart colours are classes (cl-ok, g0 ...), not inline colours', 'cl-ok' in css and '.g7' in css and 'cl-' in PAGES['campaigns.js'] and '"g" +' in PAGES['shell.js'])
check('3D views render at up to 2x the CSS pixels (device pixel ratio) in the shell viewer', 'setPixelRatio(Math.min(window.devicePixelRatio || 1, 2))' in PAGES['shellview.js'])

# ---- strain legend wording (shell viewer)
sv = PAGES['shellview.js']
check('legend: plain first line "blue = squeezed (compression), red = stretched (tension)" for the strain fields', 'blue = squeezed (compression), red = stretched (tension)' in sv and "STRAIN.includes(l.field)" in sv)
check('legend keeps compression / tension end labels, the technical note, and the data minimum and maximum', 'l.end_labels' in sv and 'l.note' in sv and 'data minimum' in sv and 'maximum' in sv)
check('plain line comes before the technical lines in the legend markup', sv.index('lg-plain') < sv.index('lg-title') < sv.index('lg-bar') < sv.index('lg-note') < sv.index('lg-data'))
check('legend is crisp and readable: 14 px text, solid card, 14 px colour bar (theme.css)', re.search(r'\.lg \{[^}]*font-size: var\(--fs-sm\)', css) and re.search(r'\.lg \{[^}]*background: var\(--card\)', css) and re.search(r'\.lg-bar \{[^}]*height: 14px', css))
try:
    import numpy as np, shell_strain as ss
    rgbv, lg = ss.colours('area_strain', np.array([-0.2, 0.0, 0.1, 0.3]))
    check('the server legend still carries compression / tension, the 98th percentile note, "not stretch-test strain" and the data range', lg['end_labels'] == ['compression', 'tension'] and '98-th percentile' in lg['note'] and 'not stretch-test strain' in lg['note'] and abs(lg['data_min'] + 0.2) < 1e-12 and abs(lg['data_max'] - 0.3) < 1e-12)
except ImportError: print('SKIP  numpy not installed: legend data test skipped')

# ---- served
try:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    import campaign_api as ca
    app = FastAPI(); ca.setup(app, lambda: False); cl = TestClient(app)
    for u, ct in (('/theme.css', 'text/css'), ('/chartkit.js', 'javascript')):
        r = cl.get(u); check('%s is served (%s)' % (u, ct), r.status_code == 200 and ct in r.headers['content-type'] and len(r.text) > 500)
    check('the pages link the stylesheet and load chartkit.js before their own script', all('href="/theme.css"' in cl.get(p).text for p in ('/campaigns', '/shell', '/shellview')) and all('src="/chartkit.js"' in cl.get(p).text and cl.get(p).text.index('/chartkit.js') < cl.get(p).text.index(s) for p, s in (('/campaigns', '/campaigns.js'), ('/shell', '/shell.js'))))
except ImportError: print('SKIP  fastapi not installed: route test skipped')
print('\n' + ('ALL OK' if not bad else 'FAILURES: ' + ', '.join(bad))); sys.exit(1 if bad else 0)

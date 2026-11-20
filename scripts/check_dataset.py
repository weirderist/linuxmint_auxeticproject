#!/usr/bin/env python3
"""Validate measured datasets under data/ against the schema in data/README.md.

Usage:
    python scripts/check_dataset.py                # every folder in data/
    python scripts/check_dataset.py data/<id> ...  # one or more named dataset folders

Prints a summary table. Exit status 0 if there is no error, 1 if any error is found, 2 for a usage problem.
Warnings (for example a value that is null because it is unknown) never change the exit status.
Standard library only, so it runs on the workstation without the simulation environment.
"""
import csv
import json
import math
import os
import sys

KINDS = {
    # kind: (required columns, numeric columns, categorical columns {name: allowed}, optional numeric columns)
    'npr_curve':      (['eps_long', 'eps_trans', 'usable'], ['eps_long', 'eps_trans'], {'usable': {'0', '1'}}, ['force_N']),
    'fold_geometry':  (['amplitude_mm', 'wavelength_mm', 'direction'], ['amplitude_mm', 'wavelength_mm'], {'direction': {'course', 'wale'}}, []),
    'curl':           (['radius_mm', 'direction'], ['radius_mm'], {'direction': {'course', 'wale'}}, []),
    'yarn_tensile':   (['strain', 'force_cN'], ['strain', 'force_cN'], {}, []),
    'loop_geometry':  (['loop_length_mm', 'cpi', 'wpi', 'thickness_mm', 'gsm'], ['loop_length_mm', 'cpi', 'wpi', 'thickness_mm', 'gsm'], {}, ['rest_time_h']),
}
ROLES = {'calibrate', 'validate', 'reference'}
DIRECTIONS = {'course', 'wale'}
TOP_KEYS = ['id', 'kind', 'fabric', 'test', 'units', 'uncertainty', 'notes', 'role']
FABRIC_KEYS = ['name', 'matrix', 'ne', 'cpi', 'wpi', 'loop_length_mm', 'yarn']
FABRIC_NUMERIC = ['ne', 'cpi', 'wpi', 'loop_length_mm']          # number or null (null = unknown)
TEST_KEYS = ['direction', 'gauge_L0_mm', 'gauge_W0_mm', 'repeat', 'date', 'operator', 'protocol']
TEST_NUMERIC = ['gauge_L0_mm', 'gauge_W0_mm', 'repeat']
# Strain may fall slightly between consecutive usable samples (tracking jitter, hand stretching). Every fall is reported as a warning;
# a fall larger than this tolerance is an error. Reason for 0.02: tape lengths in the report are quantised to about one pixel, and
# the largest fall observed in Pattern 1 is 0.0116 (CSV line 103), so 0.02 clears that with margin while still catching real reversals.
STRAIN_FALL_TOL = 0.02
PROVENANCE = {'measured', 'literature', 'expert_estimate'}


def is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def parse_float(s):
    try:
        v = float(s)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def check_meta(meta, folder_name, err, warn):
    for k in TOP_KEYS:
        if k not in meta:
            err.append("meta.json: missing key '%s'" % k)
    if err:
        return None
    if meta['id'] != folder_name:
        err.append("meta.json: id '%s' does not match folder name '%s'" % (meta['id'], folder_name))
    kind = meta['kind']
    if kind not in KINDS:
        err.append("meta.json: kind '%s' not one of %s" % (kind, sorted(KINDS)))
    if meta['role'] not in ROLES:
        err.append("meta.json: role '%s' not one of %s" % (meta['role'], sorted(ROLES)))
    fab = meta['fabric']
    if not isinstance(fab, dict):
        err.append("meta.json: fabric must be an object")
    else:
        for k in FABRIC_KEYS:
            if k not in fab:
                err.append("meta.json: fabric is missing key '%s'" % k)
        for k in FABRIC_NUMERIC:
            v = fab.get(k)
            if v is None:
                if k in fab:
                    warn.append("fabric.%s is null (unknown)" % k)
            elif not is_num(v) or v <= 0:
                err.append("meta.json: fabric.%s must be a positive number or null" % k)
        if not fab.get('name'):
            err.append("meta.json: fabric.name is empty")
    test = meta['test']
    if not isinstance(test, dict):
        err.append("meta.json: test must be an object")
    else:
        for k in TEST_KEYS:
            if k not in test:
                err.append("meta.json: test is missing key '%s'" % k)
        d = test.get('direction')
        if d is not None and d not in DIRECTIONS:
            err.append("meta.json: test.direction must be 'course', 'wale' or null")
        if d is None and 'direction' in test:
            warn.append("test.direction is null (unknown)")
        for k in TEST_NUMERIC:
            v = test.get(k)
            if v is None:
                if k in test:
                    warn.append("test.%s is null (unknown)" % k)
            elif not is_num(v) or v <= 0:
                err.append("meta.json: test.%s must be a positive number or null" % k)
    units = meta['units']
    if not isinstance(units, dict) or not units:
        err.append("meta.json: units must be a non-empty object {column: unit}")
    elif kind in KINDS:
        req, _, _, opt = KINDS[kind]
        for c in req + [c for c in opt if c in units]:
            if not isinstance(units.get(c), str) or not units[c].strip():
                err.append("meta.json: units has no entry for column '%s'" % c)
    u = meta['uncertainty']
    if not isinstance(u, dict) or 'value' not in u or 'meaning' not in u:
        err.append("meta.json: uncertainty must be an object with 'value' (number or null) and 'meaning' (text)")
    else:
        if u['value'] is None:
            warn.append("uncertainty.value is null (not yet quantified)")
        elif not is_num(u['value']) or u['value'] < 0:
            err.append("meta.json: uncertainty.value must be a non-negative number or null")
        if not isinstance(u['meaning'], str) or not u['meaning'].strip():
            err.append("meta.json: uncertainty.meaning is empty")
    if not isinstance(meta['notes'], str):
        err.append("meta.json: notes must be text")
    # optional provenance of the values: measured | literature | expert_estimate, with an optional source string
    src = meta.get('source')
    if 'source' in meta and (not isinstance(src, str) or (src != '' and not src.strip())):
        err.append("meta.json: source must be text")
    if 'provenance' in meta:
        pv = meta['provenance']
        if pv not in PROVENANCE:
            err.append("meta.json: provenance '%s' not one of %s" % (pv, sorted(PROVENANCE)))
        elif pv != 'measured' and not (isinstance(src, str) and src.strip()):
            warn.append("provenance is '%s' but no source is given" % pv)
    return kind if kind in KINDS else None


def check_rising(vals, label, err, warn):
    """vals: list of (row_number, strain). Strain must be >= 0 and rise overall; small falls are warnings."""
    if not vals:
        err.append("%s: no rows to check" % label)
        return
    neg = [r for r, v in vals if v < 0]
    if neg:
        err.append("%s: negative strain at row(s) %s" % (label, neg[:5]))
    if len(vals) > 1 and vals[-1][1] <= vals[0][1]:
        err.append("%s: strain does not rise (last %.4g <= first %.4g)" % (label, vals[-1][1], vals[0][1]))
    falls = [(r1, a - b) for (r0, a), (r1, b) in zip(vals, vals[1:]) if b < a]
    big = [(r, d) for r, d in falls if d > STRAIN_FALL_TOL]
    if big:
        err.append("%s: strain falls by more than %.2g at row(s) %s" % (label, STRAIN_FALL_TOL, [r for r, _ in big[:5]]))
    if falls:
        warn.append("%s: %d fall(s) in strain, largest %.4g (tolerance %.2g), at line(s) %s" % (label, len(falls), max(d for _, d in falls), STRAIN_FALL_TOL, [r for r, _ in falls[:5]]))


def check_csv(path, kind, err, warn):
    """Return (rows, usable_rows, strain_min, strain_max)."""
    req, numeric, categorical, optional = KINDS[kind]
    with open(path, newline='', encoding='utf-8') as fh:
        rd = csv.DictReader(fh)
        cols = rd.fieldnames or []
        rows = list(rd)
    for c in req:
        if c not in cols:
            err.append("data.csv: missing column '%s'" % c)
    if any(c not in cols for c in req):
        return len(rows), None, None, None
    if not rows:
        err.append("data.csv: no data rows")
        return 0, None, None, None
    nums = numeric + [c for c in optional if c in cols]
    for c in nums:
        bad = [i + 2 for i, r in enumerate(rows) if parse_float(r[c]) is None]
        if bad:
            err.append("data.csv: column '%s' has non-numeric or missing values at line(s) %s" % (c, bad[:5]))
    for c, allowed in categorical.items():
        bad = [i + 2 for i, r in enumerate(rows) if (r[c] or '').strip() not in allowed]
        if bad:
            err.append("data.csv: column '%s' must be one of %s (line(s) %s)" % (c, sorted(allowed), bad[:5]))
    if err:
        return len(rows), None, None, None
    usable_n = strain_min = strain_max = None
    if kind in ('npr_curve', 'yarn_tensile'):
        scol = 'eps_long' if kind == 'npr_curve' else 'strain'
        vals = [(i + 2, float(r[scol])) for i, r in enumerate(rows)]
        if kind == 'npr_curve':
            use = [(ln, v) for (ln, v), r in zip(vals, rows) if r['usable'].strip() == '1']
            usable_n = len(use)
            neg_all = [ln for ln, v in vals if v < 0]
            if neg_all:
                err.append("data.csv: negative eps_long at line(s) %s" % neg_all[:5])
            if usable_n == 0:
                err.append("data.csv: no row has usable = 1")
            else:
                check_rising(use, 'usable rows of eps_long', err, warn)
            vals = use or vals
        else:
            check_rising(vals, 'strain', err, warn)
        if vals:
            strain_min = min(v for _, v in vals)
            strain_max = max(v for _, v in vals)
    else:
        for c in numeric:
            nonpos = [i + 2 for i, r in enumerate(rows) if float(r[c]) <= 0]
            if nonpos:
                err.append("data.csv: column '%s' must be positive (line(s) %s)" % (c, nonpos[:5]))
        if kind == 'loop_geometry':
            if 'rest_time_h' in cols:
                neg = [i + 2 for i, r in enumerate(rows) if float(r['rest_time_h']) < 0]
                if neg:
                    err.append("data.csv: column 'rest_time_h' must be >= 0 (line(s) %s)" % neg[:5])
            if 'location' in cols:
                blank = [i + 2 for i, r in enumerate(rows) if not (r['location'] or '').strip()]
                if blank:
                    err.append("data.csv: column 'location' is blank at line(s) %s (give a label such as section_centre)" % blank[:5])
    return len(rows), usable_n, strain_min, strain_max


def check_dataset(folder):
    name = os.path.basename(os.path.normpath(folder))
    err, warn = [], []
    info = {'id': name, 'kind': '?', 'role': '?', 'rows': '-', 'usable': '-', 'strain': '-'}
    mpath, dpath = os.path.join(folder, 'meta.json'), os.path.join(folder, 'data.csv')
    meta = None
    if not os.path.isfile(mpath):
        err.append("missing meta.json")
    else:
        try:
            with open(mpath, encoding='utf-8') as fh:
                meta = json.load(fh)
        except ValueError as e:
            err.append("meta.json is not valid JSON: %s" % e)
        if meta is not None and not isinstance(meta, dict):
            err.append("meta.json must contain a JSON object")
            meta = None
    if not os.path.isfile(dpath):
        err.append("missing data.csv")
    kind = None
    if meta is not None:
        info['kind'], info['role'] = str(meta.get('kind', '?')), str(meta.get('role', '?'))
        kind = check_meta(meta, name, err, warn)
    if kind and os.path.isfile(dpath):
        n, usable, lo, hi = check_csv(dpath, kind, err, warn)
        info['rows'] = str(n)
        if usable is not None:
            info['usable'] = str(usable)
        if lo is not None:
            info['strain'] = '%.3g to %.3g' % (lo, hi)
    info['errors'], info['warnings'] = err, warn
    return info


def main(argv):
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
    if len(argv) > 1:
        folders = argv[1:]
    else:
        if not os.path.isdir(root):
            print("no data/ folder at %s" % os.path.normpath(root))
            return 2
        folders = sorted(os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    if not folders:
        print("no datasets found")
        return 2
    results = []
    for f in folders:
        if not os.path.isdir(f):
            print("not a folder: %s" % f)
            return 2
        results.append(check_dataset(f))
    head = ['id', 'kind', 'role', 'rows', 'usable', 'strain', 'errors', 'warnings', 'status']
    table = [head] + [[r['id'], r['kind'], r['role'], r['rows'], r['usable'], r['strain'], str(len(r['errors'])), str(len(r['warnings'])), 'FAIL' if r['errors'] else 'ok'] for r in results]
    w = [max(len(row[i]) for row in table) for i in range(len(head))]
    for i, row in enumerate(table):
        print('  '.join(c.ljust(w[j]) for j, c in enumerate(row)))
        if i == 0:
            print('  '.join('-' * x for x in w))
    for r in results:
        for m in r['errors']:
            print("ERROR   [%s] %s" % (r['id'], m))
        for m in r['warnings']:
            print("warning [%s] %s" % (r['id'], m))
    return 1 if any(r['errors'] for r in results) else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))

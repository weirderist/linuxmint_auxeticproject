"""scriptdirs.py - puts scripts/ and every purpose folder of scripts/ (not legacy/) on sys.path, so that the tests import the script modules by their plain names
(scripts/ is sorted into folders; each script finds its own siblings by itself, see the bootstrap lines in the scripts). Imported by the tests right after their sys.path line."""
import os, sys
S = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts')
for _d in [S] + [os.path.join(S, n) for n in sorted(os.listdir(S)) if n not in ('legacy', '__pycache__') and os.path.isdir(os.path.join(S, n))]:
    if _d not in sys.path: sys.path.insert(0, _d)

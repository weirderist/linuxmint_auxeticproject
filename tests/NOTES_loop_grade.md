# Parked: starting loop-length field (`--loop-grade`)

**Status: parked, 2026-11-20.** It belongs to the later yarn-mesh stage. The final code is not written and nothing is committed. The design prototype is `tests/analysis_loop_grade_design.py` (analysis only, no solver).
Every constant below is UNSOURCED until K measures loop length by region.

## Proposed rule (prototype)
- A *section* is a 4-connected region of one loop type; the fabric edge counts as a section boundary.
- Boundary term `G_b`: 1 next to the section boundary, falling linearly with the Euclidean distance (in loops) to 0 at the farthest loop of the section.
- TM axis strength per wale and per course: `f` = fraction of the majority type, `s = max(0, 2 (f - 0.5))`; a line is an axis if `s >= 0.2`; consecutive axis lines of the same majority type form one band.
- TM term `G_t`: `s` times a linear fall-off with the distance across the band, over the section the band runs through.
- Mode `boundary`, `tm` or `both` (default `both`); field `L = 1 + a (G - mean of G over the section)`, so every section keeps its total rest length exactly and a uniform term drops out. `a` is the length difference between the target loops and the farthest loops as a fraction of the section's mean loop length (example value 0.02; the option stays off by default).

## K's decisions on the open points
1. **Two-loop-wide ridge at shared boundaries** (loops of both sections next to a shared boundary get the full boundary gain): accepted as a first guess. To be tested against the shell results.
2. **An axis acts only on the section it runs through:** accepted.
3. **Square contours on single jersey** (distance to the nearest edge, so corners get the same gain as edge middles): **undecided.** In real single jersey the corners roll the most (two free edges), so corners may deserve a larger gain than the edge middles. To be decided by comparison with shell results later, not now. (The first radial draft, kept in the scripts, gives corners sqrt(2) times the edge-middle gain.)
4. **Strength mapping `s = 2 (f - 0.5)` and the minimum strength 0.2:** accepted as UNSOURCED first guesses.

## Known sensitivity, to be checked against the shell
In the dilution test (Pattern 1 with every k-th loop of its pure axis wales flipped to the other type) the `both` gain on the axis wales is **not smooth** with dilution (1.18, 0.75, 0.65, 0.69, 0.44 % for pure, 1 in 10, 1 in 5, 1 in 3, 1 in 2). The reason is that flipped loops split the sections and move the section boundaries, so the boundary term changes with the dilution. The TM-only gain is smooth (0.79, 0.46, 0.36, 0.24, 0.00 %). This sensitivity to section fragmentation needs checking against the shell.

## UNSOURCED constants of the proposal
`a` (example 0.02), minimum axis strength 0.2, the linear strength mapping, the linear fall-off with distance, the Euclidean distance to the section boundary, the mode weights (both 1), the additive per-section normalisation, the merging of consecutive same-type axis lines into one band with their mean strength.

## Not done
The final implementation of the sectioned rule, its tests (option off identical; mean conserved per section; sign and monotonic shape; rest-length colouring on a synthetic case), the dry-run printout and `meta.json` record for it, the web UI field, and `--tm-boost` (not part of this work).
Smallest server test, when the work is resumed: 8x8 single jersey, tiles 1, Ne 12, CPI 22, WPI 16, with and without the grade; Pattern 1 tiles 1 with and without.

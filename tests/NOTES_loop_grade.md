# Starting loop-length field (`--loop-grade`)

**Status, 2026-11-20 (later): implemented and tested without the solver; not yet run in the solver; not committed.** Code: `scripts/loop_grade.py` (numpy only), called from `scripts/relax_job.py`. Prototype and reference: `tests/analysis_loop_grade_design.py`. Tests: `tests/test_loop_grade.py`. The earlier radial draft (`tests/loop_grade_draft.patch`) is superseded and kept only as history.
Every constant is UNSOURCED until K measures loop length by region.

## The rule (K's decisions, implemented)
- A *section* is a 4-connected region of one loop type, computed on the matrix **as built** (tiled `tiles` times). The fabric edge counts as a section boundary.
- Boundary term `G_b`: 1 next to the section boundary, falling linearly with the Euclidean distance (in loops) to 0 at the farthest loop of the section. Loops of two sections that share a boundary both get 1 next to it (a two-loop-wide ridge).
- TM axis strength per wale and per course: `f` = fraction of the majority type, `s = max(0, 2 (f - 0.5))`; a line is an axis if `s >= --loop-grade-min-strength` (default 0.2); consecutive axis lines of the same majority type form one band (strength = mean of `s`).
- TM term `G_t`: `s` times a linear fall-off with the distance across the band, over the section the band runs through (a type-2 band acts only on a section that contains loops of its lines).
- `--loop-grade-target boundary | tm | both` (default `both`; `--tm-boost` is superseded by `tm`).
- Per solver segment: `factor = 1 + a (G - G_S)`; `G` is the overlap-weighted `G` of the loops the segment lies in, `G_S` the rest-length-weighted (= chord-weighted) mean of `G` over the segments of its section `S`. **Every section keeps exactly the total rest length of the uniform start** (tested to 1e-16), and a term that is uniform over a section drops out. `a` = length difference between the target loops and the farthest loops as a fraction of the section's mean loop length. A segment belongs to the section of the loop with the larger overlap.
- Opt-in: `--loop-grade 0` (default) = the uniform start, bit-identical (the grade code is not even called). Range of `a`: -0.4 to 1; a numerical guard refuses factors below 0.2.

## K's decisions on the open points (recorded earlier)
1. Two-loop-wide ridge at shared boundaries: accepted as a first guess; to be tested against the shell results.
2. An axis acts only on the section it runs through: accepted.
3. Square contours on single jersey (distance to the nearest edge, so corners get the same gain as edge middles): **undecided**. In real single jersey the corners roll the most (two free edges), so corners may deserve a larger gain; decide by comparison with shell results later.
4. Strength mapping `s = 2 (f - 0.5)` and the minimum 0.2: accepted as UNSOURCED first guesses.
5. Known sensitivity, to be checked against the shell: the `both` gain on the axis wales is not smooth with dilution (1.18, 0.75, 0.65, 0.69, 0.44 %) because flipped loops split the sections and move the section boundaries; the TM-only gain is smooth (0.79, 0.46, 0.36, 0.24, 0.00 %).

## Findings while implementing
- **Tile joins merge sections** (printed before the conservation was coded): Pattern 1 has 3 sections at tiles 1 (sizes 60, 100, 160), 6 at tiles 2 (60, 60, 200, 320, 320, 320) and 9 at tiles 3 (60 x3, 300, 480 x5); a tile repeat alone would give 3, 12 and 27. Type-2 forms one vertical zig-zag section per tile column; type-1 chains upward along wales 14-15. Single jersey is one section at any tiles. Consequence: the sections grow with `tiles`, the linear fall-off runs over the whole section, so **the same `a` gives a different field at a different `tiles`**. Compare runs at the same `tiles` only.
- **The shell does not need to be refused.** The draft refused `--shell` by copying the "no shell" restriction of `--ramp` and `--seed-stitch`; there is no technical reason. The grade reads the as-built loop geometry and only changes the rest lengths, right after `load()`, before the shell moves any vertex; the shell hold acts on positions, not on rest lengths. The refusal is removed; `meta.json` records `with_shell`.
- **Creep does not conserve the total.** `--creep-tau`/`--hours` move every rest length toward its current length each round (a fraction `1 - exp(-(hours/outer)/tau)` per round), so a starting grade can be partly erased. Creep is unchanged; the `--help`, the code comments, the Simulate form and `campaigns/README.md` say so, and the job log prints a note when both are on.
- **Viewer:** the rest-length colouring is `sum of l0 per loop / as-built mean rest length`. On a mixed pattern the as-built loops already differ by loop type (5.9 % spread on Pattern 1), so that variation is part of the colour (it is not part of single jersey). The viewer's notes now say so and give the size. A per-loop relative mode (each loop against its own start) would show only the change; it is not built.

## Recording
`meta.json` (`simulations[-1].params`: `loop_grade`, `loop_grade_target`, `loop_grade_min_strength`; block `loop_grade`: factor range, per-section loops / segments / total rest length before and after / factor range, the axis bands, `with_shell`, the UNSOURCED list), only when the grade is on. `job_request.json` and presets: the three `SimBody` fields. Dry-run printout and the allowed-flags logic of `campaign.py`: picked up from `relax_job.py` without edits. Simulate form: a block with the three fields, one-line explanations and UNSOURCED marks.

## UNSOURCED constants
`a` (example 0.02; off by default), minimum axis strength 0.2, the linear strength mapping, the linear fall-off with distance, the Euclidean distance to the section boundary, the mode weights (boundary 1, tm 1), the additive per-section normalisation, the merging of consecutive same-type axis lines into one band with their mean strength, the numerical factor floor 0.2.

## Not verified
Never run in the solver (needs JAX), never opened in a browser, nothing on the server. Smallest server test: 8x8 single jersey, tiles 1, Ne 12, CPI 22, WPI 16, with and without `--loop-grade 0.02`, same other settings; then the same for Pattern 1 tiles 1. Compare the folding and open both with the rest-length colouring (a new run writes `relaxed_l0.csv`).

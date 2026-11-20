# Measured datasets

Each dataset is one folder, `data/<id>/`, holding two files:

- `meta.json` describes the fabric, the test and the units.
- `data.csv` holds the measurements.

The folder name must equal the `id` in `meta.json`. Optional photographs go in `data/<id>/photos/`; these are excluded from git. Check the layout with `python scripts/check_dataset.py` (all datasets) or `python scripts/check_dataset.py data/<id>` (one).

Rule: record only what was measured or is documented. Write `null` for an unknown value; never estimate one. Anything without a source must say so in `notes`.

## meta.json

All keys below are required. `null` is allowed where stated.

| Key | Content |
|---|---|
| `id` | Dataset id, equal to the folder name. |
| `kind` | One of `npr_curve`, `fold_geometry`, `curl`, `yarn_tensile`, `loop_geometry`. |
| `fabric` | Object: `name`; `matrix` (the 1/2 front-back pattern, or `null`); `ne` (English cotton count, number or `null`); `cpi` (courses per inch) and `wpi` (wales per inch), numbers or `null`; `loop_length_mm` (yarn length per loop, number or `null`); `yarn` (free text). |
| `test` | Object: `direction` (`course`, `wale` or `null`); `gauge_L0_mm` and `gauge_W0_mm` (initial gauge length and width, numbers or `null`); `repeat` (number of repeats, or `null`); `date`; `operator`; `protocol` (free text or `null`). |
| `units` | Object mapping each CSV column to its unit (text). Must cover every required column of the kind. |
| `uncertainty` | Object: `value` (number or `null`) and `meaning` (text saying what the number is, for example "one standard deviation of strain, from 5 repeats"). |
| `notes` | Free text: history, caveats, anything unsourced. |
| `role` | `calibrate` (used to fit model parameters), `validate` (held out to test the model) or `reference` (context only, not used to fit or test). |

Optional keys:

| Key | Content |
|---|---|
| `provenance` | Where the values come from: `measured` (own test), `literature` (published value) or `expert_estimate` (a person's estimate). |
| `source` | Text citing the source (author and year, document and page, or the name of the person). The checker warns if `provenance` is not `measured` and `source` is missing. |

## data.csv columns by kind

Extra columns are allowed; the columns listed here must be present.

| Kind | Required columns | Optional |
|---|---|---|
| `npr_curve` | `eps_long`, `eps_trans`, `usable` | `force_N` |
| `fold_geometry` | `amplitude_mm`, `wavelength_mm`, `direction` | |
| `curl` | `radius_mm`, `direction` | |
| `yarn_tensile` | `strain`, `force_cN` | |
| `loop_geometry` | `loop_length_mm`, `cpi`, `wpi`, `thickness_mm`, `gsm` | `location`, `rest_time_h` |

- `eps_long` and `eps_trans` are engineering strains along and across the stretch direction. `usable` is 1 for a row to be used and 0 for a row to be excluded.
- `direction` is `course` or `wale`.
- For `loop_geometry`, `location` says where on the swatch the row was taken (free text; suggested labels: `section_centre`, `section_border`, `tm_axis` for the transfer-medium axis, `whole_course`) and `rest_time_h` is the time in hours the swatch rested after knitting or relaxation before the measurement (number, 0 or more). Both are optional; a blank `location` is an error.
- All numeric values must be finite; geometry values must be positive.

## Checks made by check_dataset.py

- `meta.json` is valid JSON with every required key; `id` matches the folder; `kind` and `role` are allowed values; `test.direction` is `course`, `wale` or `null`; numeric fabric and test fields are positive numbers or `null`; `units` covers the required columns; `uncertainty` has `value` and `meaning`; if present, `provenance` is one of the three allowed values.
- `data.csv` has the required columns, numeric values where required, and `usable` equal to 0 or 1.
- Strain (`eps_long` on usable rows, or `strain`) is non-negative and rises overall. Every fall between neighbouring rows is reported as a warning; a fall larger than 0.02 strain is an error. The tolerance (`STRAIN_FALL_TOL` in the script) is 0.02 because tape lengths in the report are quantised to about one pixel and the largest fall in Pattern 1 is 0.0116.
- A `null` that means "unknown" gives a warning, not an error. Any error gives a non-zero exit status.

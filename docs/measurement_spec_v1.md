# Measurement spec v1 (frozen 2026-11-16)

Every Metrology number that is compared across code changes is measured with this spec. The machine-readable
copy is `meshsolid/metrology_spec.py` (`SPEC`, `SPEC_VERSION = "v1"`, `spec_hash()`), and every harness row
records `spec_version` and `spec_hash`. **Any change to a value below, or to how a metric is computed,
bumps the version** (v2, v3 ...). `tests/test_metrology_harness.py` pins the hash, so an edit without a
bump fails the test suite.

## Units
Millimetres throughout. Rotations in degrees.

## Distance
* One-sided: every **scan vertex** to the nearest point on the solid's tessellation (vertices, not
  area-weighted samples).
* Signed: positive = scan outside the solid (normal of the closest solid triangle).
* Solid tessellation: linear deflection 0.01 mm, angular 0.1 rad (Metrology 1.0.1 defaults).
  The engine's own gate uses 0.002 mm / 0.05 rad; on the pinion this changes rms by 0.0006 mm (0.3903 vs 0.3909).

## Alignment
* Metrology 1.0.1 `align_scan`, frozen as-is: trimmed point-to-point ICP from the 4 proper PCA starts,
  25,000 area-uniform samples on scan and solid, seed 0, 40 iterations, 90 % trim, the start with the lowest
  median residual wins. On by default.
* One bug fix was made while freezing: 1.0.1 capped the solid samples at the tessellation's vertex count
  (a box was aligned from 8 points). No effect on the accuracy set (every solid tessellation has far more
  than 25,000 vertices).
* Logged per row: rotation angle (deg), rotation axis, translation (mm), and the ICP's sampled median residual
  (a pre-alignment figure, not the final deviation).
* Known limits, deliberately not fixed in v1: lands on another symmetric copy (pinion 21.9 / 109 / 131 deg,
  STG1 23.9 / 168 deg), stops ~0.05-0.1 deg short. The alignment fix will be spec v2.
* A per-part `align: false` is allowed only as a logged deviation (`deviations` column).

## Exclusions
* Only named, listed exclusions:
  1. hand-entered features from the conversion report beside the STEP (1.0.1 auto-exclude, e.g. a bore
     entered with `--measure`);
  2. per-part exclusions declared in the harness manifest (`zones` + `zones_reason`, or
     `exclude_mask_npy` + `exclude_mask_reason`).
* **No zone tags by default**: no 'tip', 'backface' or 'bore' zones. In 1.0.1 the 'tip' gear zone removed an
  impeller's rim and blade tips (44,214 points on STG1) and hid a 2 mm rim slot.
* `pct_excluded` = named-excluded vertices / all scan vertices x 100. Beyond-cap points are **not**
  exclusions; they are failures (below).
* Per-reason breakdown is logged in `excluded_by_reason`.

## Outlier cap
* Cap = 0.2 mm (the value used for the 1.0.1 baseline).
* **The cap never removes points from the v1 numbers.** Points with |d| > cap are included in
  `rms_uncapped`, `p95`, `p99` and the maxima, and count as **failures** in `pass_pct_v1`.
* The cap only defines the legacy (1.0.1) numbers `rms_capped`, `p95_capped`, `pass_pct_legacy`, plus
  `capped_count` / `pct_capped`.

## Pass percentage (logged in both forms in every row)
Tolerance ±0.05 mm. *Inspected* = scan vertices not removed by a named exclusion, **including** beyond-cap points.

| column | definition |
|---|---|
| `pass_pct_v1` | 100 x #(inspected, not beyond cap, \|d\| <= 0.05) / #(inspected) — capped points are failures |
| `pass_pct_legacy` | 100 x #(inspected, not beyond cap, \|d\| <= 0.05) / #(inspected, not beyond cap) — 1.0.1 counting |

Target (unchanged from the review): rms_uncapped <= 0.03 mm, pass_pct_v1 >= 90 %, pct_excluded <= 5 %
(`target_met_v1`). Uncapped rms is never shown alone: it sits beside p95, p99, the maxima and pass %.

## Harness row (results/harness.csv)
`timestamp, spec_version, spec_hash, git_commit, label, part, status, scan, scan_sha12, solid, solid_sha12,
n_scan, n_inspected, n_excluded, pct_excluded, excluded_by_reason, cap_mm, capped_count, pct_capped, tol_mm,
rms_uncapped, rms_capped, mean_signed, p95, p95_capped, p99, max_plus, max_minus, pass_pct_v1,
pass_pct_legacy, target_met_v1, align, rot_deg, rot_axis, trans_x_mm, trans_y_mm, trans_z_mm,
icp_median_pre_mm, seconds, device, deviations, error`

A part that fails still gets a row (`status=error`), so a failure is never silent. If the columns change,
the harness refuses to append and a new CSV is started.

## Running it
```
python scripts/metrology_harness.py --manifest accuracy_set.json --csv results/harness.csv --label "what changed"
```
On the PC the manifests live in `%LOCALAPPDATA%\meshsolid\harness\` (the part files never leave the PC and
are never committed).

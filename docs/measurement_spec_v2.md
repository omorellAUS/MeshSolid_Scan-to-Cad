# Measurement spec v2 (2026-11-16)

Spec v2 is spec v1 (`docs/measurement_spec_v1.md`) with **only the alignment changed**. Distance, sign,
population, exclusions, the 0.2 mm cap, both pass-% forms and the targets are identical. The hashes are pinned
in `tests/test_metrology_harness.py`: v1 `05d7048b9f9d`, v2 `42ceb29d6b7b`. Every harness row carries
`spec_version` + `spec_hash`. v1 rows stay in `results/harness.csv` and v2 rows go to `results/harness_v2.csv`
(the v2 CSV adds `n_fold` and `align_detail`).

Step 5a (per-face / per-blade columns and stats) did not change the measurement: the v1 rerun is identical
to the earlier v1 rows, so 5a needed no spec bump.

## Alignment (`meshsolid/align_v2.py`, `metrology.run(align_method="v2")`)
1. Point-to-plane ICP on >= 500,000 points (scan vertices, topped up with area-uniform samples, seed 0).
   The worst 5 % of |residual| is trimmed, and solid points within 0.25 mm of a sharp (>20 deg) edge are
   left out. ICP stops when a step moves < 0.0005 deg and < 0.001 mm (max 100 iterations).
2. Start pose: identity (MeshSolid builds the solid in the scan frame) or a given transform. For N-fold parts
   (N from the manifest, else the conversion report, else estimated) the rotation about the axis is bounded
   to +/- half a pitch from the start, so a blade can never lock onto its neighbour.
3. Rigid refinement on the signed deviations of the inspected vertices. Named exclusions are removed, and a
   robust 6-sigma MAD gate is used instead of a trim, because a 5 % trim biased the pose by 0.009 deg under
   per-blade scatter. Refinement above 0.01 deg / 0.01 mm raises a warning.
4. `direct_check`: the rms is re-measured at +/- 0.005 deg / 0.005 mm in all 6 DOF. The final pose must not
   be beaten.

By design, a mean per-blade rotation equals a rigid rotation, and a once-per-revolution blade pattern equals
a rigid translation. The pose absorbs both, and only the remaining blade-to-blade scatter is left in the
deviation.

## Self-test (`tests/test_align_v2.py`)
The test part is synthetic and N-fold. It has 0.013 mm noise, balanced per-blade rotation and thickness
scatter, and a known motion of ~0.1 deg plus ~0.03 mm applied together.
- Recovery: 0.00027 deg / 0.00009 mm (requirement <= 0.002 / 0.002).
- Sign: verified by direct re-measurement (the wrong sign measures clearly worse).
- Determinism: two runs give a bit-identical T.
- The half-pitch bound holds.

## Real pairs (PC, 2026-11-16)
All 6 pairs converged in 2-4 ICP iterations with the bound not hit and `direct_check` ok. The repeat run
produced identical rows; only `seconds` differs. The refinement magnitudes (all below the 0.01 warning) are
in `align_detail`.

## Per-blade test (`scripts/per_blade_test.py`, `tests/test_per_blade.py`)
This is a preview of plan step 6. On the blade-surface vertices (|n . e_t| > 0.5), it fits the rotation about
the axis, the axial shift and a uniform normal offset (half the thickness change):
- first one common set for all blades,
- then one set per blade.

A column the blade cannot observe (rms(n.u) < 0.05) is dropped and the axial term is reported as 0 with
`axial_observable=false`. The synthetic test recovers the per-blade rotation and thickness with
corr > 0.999 and the same sign. Sector numbering starts at an arbitrary blade, so the comparison uses the
best cyclic shift. Results: `results/harness/per_blade_*_2026-11-16.json`.

# Step 7 plan: STG1 rim slots as modelled cuts

Status 2026-11-16: implemented in meshsolid/rim_slots.py (8902af5, 96df1e0, then per-slot auto): detection + cut, wired into the impeller and revolve builders behind --rim-slots auto|on|off. auto cuts each confident slot and leaves non-confident patches listed (STG1 job 20261116-140538: 4 patches, 1 confident flat slot at 69.46 deg / 8.76 mm / floor rms 0.026, 3 real but non-flat back-face depressions rejected on floor_rms 0.077-0.154). Replayed on the STG1 scan with all-or-nothing auto (none cut); per-slot auto not yet replayed.

## What we know
- STG1 (X2-B1941, 15 blades, rim r ~ 83 mm) has flat-floored gaps in the back face near the rim:
  floors at about z = -2.85 mm (axis frame, back plane z = 0), 3-4 slots spread over about 196-320 deg,
  radial band r 76-83 mm (open to the rim). About 2 mm deep against the nominal back face.
- Today they are hidden: spec v1 caught it (the old 'tip' exclusion + outlier cap masked them); the
  balance-cut detector (meshsolid/balance.py) treats back-face scallops as non-design and only *lists* them
  ("modelled_as: not modelled"). Rim slots are design geometry (flat floor, straight walls), so they must be
  *cut into the solid*, not masked.

## Design
1. **Detect (meshsolid/rim_slots.py, new):** after the impeller build and deviation, in the axis frame take
   scan vertices with r in [0.88 R_rim, R_rim + 0.5] whose nominal surface is the back face (|n.z| > 0.85,
   |z| < 0.5) and the signed deviation < -0.5 mm (scan inside). Grow patches (link 1.2 mm), then per patch:
   - floor: RANSAC plane on the patch's scan points with |n.z| > 0.95 -> z_floor; accept if the floor
     rms < 2x noise (about 0.03 mm) and depth > 0.5 mm;
   - walls: the two side walls from the points with |n.theta| > 0.8 -> angles th0/th1 (or a
     constant-width slot: fit two parallel lines in the back-plane projection, width w, centre angle);
   - inner end: r_in from the patch's inner boundary (flat end vs radius = tool radius, report which).
   Reject patches that overlap a balance cut (compact scallop, no flat floor) - those stay listed only.
2. **Model:** per slot a prism in the axis frame (polygon th0..th1 x r_in..R_rim + 1 mm, z from z_floor to
   +0.1 mm) -> BRepAlgoAPI_Cut from the built solid (reuse the fuzzy retry pattern from impeller._common).
   Corner radius = tool radius if the inner end is round. Snap the floor to 0.05 mm, widths to 0.1 mm only if
   the fit residual allows (logged, never silently).
3. **Report:** rep["rim_slots"] = list of {centre_deg, width_mm or th0/th1, r_in, z_floor, floor_rms,
   n_points, modelled: true}; balance_cuts unchanged.
4. **Gate:** spec v2 rows keep no zone for the slots (they are measured, not excluded). Expected: STG1 p99
   0.313 -> well below, max_minus -2.04 -> ~ -0.3, pass_v1 up a few points.

## Tests
- Synthetic: a revolved disc with 3 milled slots (known z_floor, widths, angles) scanned with 0.013 mm
  noise -> detect within 0.02 mm / 0.2 deg; cut solid vs truth rms < 0.01.
- Negative: disc with ground balance scallops only -> zero rim slots, scallops still listed.
- Robustness: Stage 3 and pinion -> no slots found (no false positives).

## Replay and measure
- Replay on the PC (part files stay there): add `--rim-slots auto|off` to the CLI, rebuild STG1 into a new
  job, add it as `stg1_impeller` engine-output pair in the v2 manifest, and log harness rows with
  _build_commit.txt (scripts/make_harness_code_zip.py).
- Accept only if rms_capped, p99 and pass_v1 improve and Stage 3 / pinion rows are unchanged.

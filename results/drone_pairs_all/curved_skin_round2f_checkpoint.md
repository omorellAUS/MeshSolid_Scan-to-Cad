# Curved-skin round 2f checkpoint (2026-11-09 ~12:09 AWST)

## Ask
Tessellation-aware loft `self_check` from first principles (not tuned to pass WINGTIP); validate vs bad lofts + regressions; if justified apply → WINGTIP L/R 2 STEP; continue WING 2 L planar-cap; wire loft + limited pair rerun only if green.

## First-principles investigation

| quantity (WINGTIP L 2) | value |
|---|---|
| facet sag_p90 (`w·θ/8`) | **0.0077 mm** |
| fit tol (`1.5·sag_p90`) | 0.0115 mm |
| mesh verts → truth STEP | rms **0.020**, p95 0.031 |
| mesh verts → loft solid | rms **0.158**, p95 0.333 |
| loft → STEP (area samples) | rms **0.035**, ≤0.02 mm ~84–88% |
| residual location | TE/LE + tips (mid-skin away from edges ~0.06 rms) |

**Conclusion**

1. **Vertex gates must stay tight** (`1.5/3·tol`). `facets.py`: CAD tessellation vertices lie on the true surface. Empirically verts→STEP ≪ verts→loft, so the 0.158 rms is a real loft miss (TE/LE/tip), **not** chord sag. Widening vertex gates enough to pass this part (~10×) would be per-part tuning, not first principles.
2. **`solid→mesh` should be tessellation-aware**, same basis as `pipeline.accept`: face interiors sit ~sag off the design surface → `lim_s2m = 3·tol + sag_p90` when faceted; non-faceted keeps `4·tol`.
3. **"Vertices only vs interiors"** for the *vertex* check is already the right primary gate; dropping it or comparing only interiors would hide the TE/LE miss. Sag-aware s2m is the correct treatment of interiors.

This gate change **does not** make WINGTIP self_check green (verts still fail hard). That is intended.

## Gate change applied

`meshsolid/loft.py` self_check:
- detect faceted via `facet_stats` (`coplanar_len_frac>0.02` ∧ `turning_smooth_len_frac>0.15`)
- `lim_s2m = 3·tol + sag_p90` if faceted else `4·tol`
- vertex limits unchanged
- records `faceted`, `facet_sag_p90`, `solid_to_mesh_p95_limit` in `self_check`

Unit test: `test_faceted_self_check_s2m_uses_chord_sag_vertices_stay_tight`  
TDD: `tests/test_loft_smooth_skin.py` **6 passed**

## Validation (must still reject / must not change)

| case | result |
|---|---|
| bad tip-cap loft rms 0.89 | still fails `1.5·tol` (vertex gate) |
| 43% underfill | still fails volume 2% gate before deviation |
| WINGTIP L/R 2 after change | still **FAIL** self_check (rms ~0.158) |
| exact/scan regressions (M3, coupling, s04–s06, p01, screw, impellers) | loft path not wired into pipeline — **no outcome change**; imports OK |
| synthetic loft tests | 6/6 green |

## WINGTIP L/R 2 STEP bidirectional (self_check still red)

| part | self_check | loft→STEP rms | ≤0.02 mm | vol rel |
|---|---|---|---|---|
| WINGTIP L 2 | FAIL rms 0.158 | **0.036** | 84% | −0.30% |
| WINGTIP R 2 | FAIL rms 0.158 | **0.035** | 88% | −0.32% |

Artifacts: `wingtip_l2_loft.step`, `wingtip_r2_loft.step`, truth `*_truth_s0.step`

## WING 2 L planar-cap topology

- `capf≈0.10` → tip-apex path correctly **off**; `_cap_levels` = `[-395, -290]` = vertex AABB (true planar caps)
- Truth STEP: 2 solids, 29 Plane + 4 BSpline + 6 Cyl (complex; mesh pairs to solid #0)
- Loft vs mesh still bad: rms **0.93** (40 sec) / **0.72** (80 sec); mid-skin worse than caps
- `corner_deg` 25/35 → corner-sign mismatch across sections (not adopted)
- **Not green** — construction miss, not cap-topology mis-detect

## Pipeline wire / matched-pair rerun

**Skipped** — no green self_check on wingtips or wing. Same rules: no facet-as-BRep pass claims, no per-part gate loosen.

## Honest status
- Tessellation-aware **s2m** gate applied from first principles; **vertex** gates unchanged
- WINGTIP loft ≈ truth STEP (~0.035 rms) but correctly fails mesh vertex self_check
- WING 2 L planar-cap path selected; skin loft quality still insufficient
- Loft **not** wired into pipeline

## Next
1. Fix loft construction at TE/LE (and WING 2 L mid-skin) until vertex self_check greens — without loosening gates
2. Then STEP already looks good on wingtips; recheck WING 2 L vs solid #0
3. Only then wire loft + limited matched-pair rerun (wingtips, wings, ailerons)

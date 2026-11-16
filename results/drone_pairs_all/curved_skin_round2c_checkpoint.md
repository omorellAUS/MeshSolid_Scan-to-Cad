# Curved-skin round 2c checkpoint (2026-11-09 ~11:25 AWST)

## Goal this slice
1. Close tapered ThruSections into a true solid (fix ~43% underfill)
2. WINGTIP L 2 independent bidirectional STEP validation
3. Second distinct matched solid

## Root cause (solidification / underfill) — FOUND + FIXED
**`native_closed=True` kept unequal CAD vertex counts per section** (e.g. 57 / 221 / 69 edges).
`BRepOffsetAPI_ThruSections(isSolid=True)` then returned a shell / invalid solid with ~43% volume.

**Fix (generic, no per-part tuning):**
- Always arc-length-resample closed smooth loops to a shared `Ms` (cap 64) before ThruSections
- Keep polyline edges (not B-spline) so concave airfoils do not balloon
- Prismatic path still uses native vertices + `MakePrism`
- Cyclic `_align_closed` (phase + optional reverse) to reduce loft twist on tapers
- `_thru` tries smooth then ruled; picks the valid solid closest to target volume
- Capless skins: target = mesh volume when slab covers >85% of AABB; else trapezoid of section areas
- Denser smooth seed (up to 9) + refine re-enabled (safe with equal-count polylines)

## 1. Synthetic TDD
`tests/test_loft_smooth_skin.py` **3 passed**
- corner-free tessellation turns
- extruded airfoil (prismatic prism)
- **new** tapered airfoil → true solid, volume within 2%, self_check green on dense fixture

## 2. WINGTIP L 2
| gate | result |
|---|---|
| BRep solid | **1 valid solid** (was shell / 0 solids) |
| volume vs mesh | **0.9985** (was ~0.43) |
| loft self_check vs mesh | **FAIL** rms 0.89 / p95 1.89 (tol 0.0115) |
| STEP bidirectional (ICP+mirror, 60k samples) | volume rel_diff **-0.21%**; surface **rms ~1.06 mm**, p95 ~1.74, ≤0.02 mm band **7.4%** — **not a green pair** |

STEP artifact: `results/drone_pairs_all/wingtip_l2_loft.step` (truth slice `wingtip_l2_truth_s0.step`).

## 3. Second distinct solids (mesh volume/solid gates only)
| part | solids | vol ratio | self_check |
|---|---|---|---|
| WINGTIP R 2 | 1 valid | 0.9985 | FAIL rms 0.89 |
| WING 2 L | 1 valid | 0.9983 | FAIL rms 1.41 |

Solidification class is fixed for these tapered skins; surface fidelity (~1 mm) still blocks matched-pair green.

## Honest status
- ThruSections → **true solid** with mesh-volume match: **done**
- STEP bidirectional validation: **ran**; volume OK; **surface not within gate**
- Pipeline wire / install package: **not yet** (no green matched pair)
- Gates not loosened; STEP remains truth; no per-part constants

## Next
1. Drive surface deviation down (section correspondence / LE sampling / tip span) until self_check and STEP p95 are within pipeline accept bands
2. Re-run WINGTIP L 2 STEP bidirectional as the green gate
3. Confirm second solid (WINGTIP R 2 or WING 2 L) the same way
4. Only then wire loft into the pipeline and package

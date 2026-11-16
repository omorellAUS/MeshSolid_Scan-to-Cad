# Curved-skin round 2b checkpoint (2026-11-09 ~11:05 AWST)

## 1. Synthetic extruded airfoil — DONE (TDD green)
- `tests/test_loft_smooth_skin.py` **2 passed**
- Fixes: native closed section polylines; prismatic `MakePrism` when section area/extent stable; no refine on smooth; seed respects `max_sections`; OCP polyline wires; volume matches mesh within 2% + self_check

## 2. WINGTIP L 2 (mesh-only) — NOT YET a valid solid
- Sweep frac ≈ 0.998, cap_frac = 0 → `_area_span` trims empty tips (generic)
- ThruSections produces a **shell** (not solid); `ShapeFix_Shape` → valid shell, `MakeSolid` still invalid
- Volume ~43% of mesh (26560 / 61123) — underfill / open loft
- **No STEP bidirectional validation** (no rebuilt solid to compare)
- Failure reported honestly; gates not loosened

## 3. Second representatives (probes only)
| part | result |
|---|---|
| WINGTIP R 2 | same class: valid shell-ish, vol ratio ~0.48, 0 solids |
| BOOM MOUNT L | open/mismatched loop counts along sweep |
| WING 2 L | 0 solids, vol ratio ~0.61 |
| AILERON 1R VTOL | open cross-section (dangling points) |

## Commits this slice
- (pending) loft smooth prismatic + area_span + ShapeFix attempt; synthetic TDD green

## Next remaining
1. Close ThruSections output to a true solid on tapered airfoil skins (wire compatibility / end caps / ShapeFix_Solid path).
2. Re-run WINGTIP L 2 → independent STEP truth bidirectional deviation.
3. Second distinct solid (e.g. WING 2 or opposite wingtip) once (1) works.
4. Wire loft into pipeline only after a real matched-pair solid validates.

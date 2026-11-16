# Curved-skin round 2e checkpoint (2026-11-09 ~11:52 AWST)

## Ask
Preserve sharp TE (and LE where sharp) as loft corners / C0 knots without breaking tip apexes; WINGTIP L 2 self_check green → STEP bidirectional → WINGTIP R 2 / WING 2 L.

## Diagnosis continued
| observation | detail |
|---|---|
| Truth STEP topology | **4 faces**: 2× BSpline + 2× Plane — TE/LE split skins with **planar** end caps |
| Mesh self_check vs STEP | Faceted STL TE/LE verts stick out past smooth loft; mesh vertex rms ≫ STEP surface rms |
| Closed + tip-apex loft | Best mesh self_check so far (~0.09–0.16 rms) and **best STEP match** |
| TE+LE open-run corners | ThruSections solid OK when TE/LE stay separated; tip sections collapse (sep→1); **worse** mesh rms (~0.15–0.67) and **worse** STEP (~0.24 rms) than closed+apex |
| Double-TE corners (both at TE) | Collapse at tips; invalid/poor solids |
| TE-cusp forced in closed polyline | Align fights cusp; regresses |
| Dense ruled (80 sections) | Mesh rms ~0.13 — still far from gate 0.017 |

## Code landed
- `_te_le_corners(Q, te_hint=None)` — tracked TE (sharp turn) + LE (farthest); raises on collapse
- Unit test `test_te_le_corners_splits_sharp_trailing_edge` green
- Auto TE+LE-in-smooth path **reverted** (broke extruded/tapered TDD + wingtip wires); helper kept for next attempt
- Tip-apex + nonempty span + Chebyshev from 2d retained

TDD: `tests/test_loft_smooth_skin.py` **5 passed**

## WINGTIP L 2 (max_sections=40, tip-apex closed)
| gate | result |
|---|---|
| solid / vol vs mesh | 1 solid, ratio **0.9975** |
| self_check vs mesh | **FAIL** rms 0.158 / p95 0.333 / s2m 0.047 (gates 0.017 / 0.035 / 0.046) |
| STEP bidirectional (ICP+mirror, 60k) | test→ref rms **0.035**, p95 **0.045**, ≤0.02 mm **88.4%**, vol rel_diff **−0.30%** |

STEP artifact: `results/drone_pairs_all/wingtip_l2_loft.step`

## Other parts
| part | self_check |
|---|---|
| WINGTIP R 2 | same class: rms 0.158 (not green) |
| WING 2 L | planar caps (`capf≈0.10`); rms 0.93 — still separate |

## Honest status
- TE/LE C0 corner loft **probed**; not adopted as default — regresses mesh and STEP vs closed+apex
- Mesh self_check **not green** (faceted TE/LE vs smooth loft)
- STEP bidirectional **strong** (~0.04 mm) with tip-apex closed loft — closer to “matched solid” than mesh self_check suggests
- Gates not loosened; no per-part constants

## Next
1. TE+LE corner loft that matches truth topology (2 BSplines + 2 planes) **without** tip collapse — e.g. stop sections before collapse and use **planar** end caps (not apexes) at those stations; or sew upper/lower lofted faces explicitly
2. Or reconcile mesh self_check with faceting (still must not loosen gates): force loft through TE crease edges / denser span stations at high TE lateral speed
3. Then R 2 + WING 2 L; pipeline wire only after green matched pair

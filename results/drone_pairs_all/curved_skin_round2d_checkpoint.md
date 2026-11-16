# Curved-skin round 2d checkpoint (2026-11-09 ~11:45 AWST)

## Ask
Diagnose ~1 mm rms loft-vs-mesh despite 0.998 volume; fix until WINGTIP L 2 self_check green; then STEP bidirectional; then WINGTIP R 2 / WING 2 L.

## Diagnosis (where the error lives)

| locus | finding |
|---|---|
| **Volume** | Trapezoid of section areas ≈ mesh vol; solid vol ratio ~0.998 — extraction/span OK |
| **Median vertex→solid** | **p50 ≈ 0.04 mm** (before tip fix) / **≈ 0.01 mm** (after) — most of the skin is fine |
| **Tips (outside area_span)** | 141 verts; 3D dist **≡** beyond-span along d (corr=1.0). Planar ThruSections end caps vs pointed mesh tips. Dominated rms (~3.5 mm tip rms → overall ~0.89) |
| **solid→mesh at tip band** | End-cap face samples p95 ~5.6 mm before apex closure |
| **In-plane section fit** | loop residuals 0.02–0.13 mm — section extraction OK |
| **Holes / multi-loop** | Single outer loop all along span — no missing internals |
| **After tip-apex fix** | Remaining bad verts (dev>0.15) are **100% LE or TE** (0% mid-chord). Hot band ~h∈[574,601] |
| **Heatmap** | `results/drone_pairs_all/wingtip_l2_dev_heatmap.png` |

Root cause of the ~1 mm rms: **planar end caps on a pointed, capless taper**, not wrong span integral / missing holes. Secondary residual: LE/TE fidelity under uniform closed-loop resample + corner-free smooth path (sharp TE ~86–92° is discarded when `corner_deg≤0`).

## Fixes landed (generic)
1. **Tip apex closure** for capless skins (`capf<0.05` and AABB extends past section span): `_tip_apexes` + ThruSections `AddVertex` before/after wires
2. **Non-empty `_area_span`** (min_frac=0, denser samples) so tip sections are included
3. **Chebyshev section seeding** near tips; Ms cap 96; no end-height remap when apexes present
4. Clamp “new sharp corners” in `_corners_like` (prep for TE-preserving path; not yet green on wingtip)
5. Open-run wires also polyline (no B-spline balloon)

TDD: `tests/test_loft_smooth_skin.py` **4 passed** (incl. pointed spindle needing apexes).

## WINGTIP L 2 status
| gate | before 2d | after tip apex | need |
|---|---|---|---|
| solid | 1 valid | 1 valid | ✓ |
| vol ratio | 0.998 | 0.999 | ✓ |
| self_check rms | 0.89 | **0.116** | ≤0.017 |
| self_check p95 | 1.89 | **0.230** | ≤0.035 |
| self_check s2m p95 | 1.79 | **0.057** | ≤0.046 |

**Self_check still FAIL** (LE/TE residual). STEP bidirectional **not re-run** this slice (no green self_check). Prior STEP (pre-apex) had vol −0.21% but surface rms ~1 mm.

## Other parts
| part | self_check |
|---|---|
| WINGTIP R 2 | same class: rms 0.118 / p95 0.253 (improved, not green) |
| WING 2 L | still bad (rms 1.25); has planar caps (`capf≈0.10`) — separate issue, apex gated off |

## Next
1. Preserve sharp TE (≥~70°) as loft corners without breaking tip area follow / ThruSections solid
2. Or curvature-aware LE/TE sampling that does not regress mid-span (uniform curvature attempt regressed)
3. Re-gate WINGTIP L 2 self_check → STEP bidirectional → R 2 / WING 2 L
4. Pipeline wire only after a green matched pair

Gates not loosened; STEP truth only; no per-part constants.

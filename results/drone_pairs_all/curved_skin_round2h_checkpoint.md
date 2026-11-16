# Curved-skin round 2h checkpoint (2026-11-09 ~12:55 AWST)

## Ask
Degenerate tip section wires + hard TE generator without underfill; then R2 / WING 2 L; wire only if L2 vertex self_check green. Also objectively compare a direct B-spline skin fit (section/LSCM param, C0 TE, degenerate tip) if ThruSections cannot reach tolerance.

## ThruSections construction

### Degenerate tip micro-sections (landed)
`_outer_section_span`: when tip apexes are used, extend the loft span to the outermost **valid** cross-sections near the AABB, then still `AddVertex` apexes. Ablation:

| config | L2 rms | p95 | s2m |
|---|---|---|---|
| 2g baseline (area_span + apex, ≥80) | 0.065 | 0.121 | 0.014 |
| **micro-span + apex** | **0.059** | **0.102** | **0.014** |
| micro-span + scaled tip wires | 0.059 | 0.104 | 0.012 |
| micro-span, no apex | 0.066 | 0.140 | 0.014 |

### Hard TE generator (probed, not landed)
| attempt | result |
|---|---|
| TE end-cluster Chebyshev resample | **regress** 0.059→0.086–0.118 |
| TE blunt 2-corner (flat TE edge) | wire not connected |
| TE+LE upper/lower open (prior) | underfill / worse rms |
| TE seam start (2g, kept) | still in place; helps with smooth-prefer |

**Conclusion:** keep TE seam start + smooth-prefer; do **not** cluster samples at TE. Tip win is micro-section span + apex.

## B-spline alternative (objective compare)

Section-parameterised least-squares cubic B-splines (reuse `meshsolid.freeform` solver):

| method | vert rms | vs loft 0.059 | vs gate 0.017 |
|---|---|---|---|
| UL open surfaces (grid LS) | ~0.16 | worse | fail |
| Periodic tube LS on all mesh verts | ~0.18 (sampler) | worse | fail |
| ThruSections (current) | **0.059** | — | fail (~3.4×) |

**Verdict:** ThruSections remains the better approach for this part. Direct B-spline skin fit does **not** clear the vertex gate and is not closer than the loft. No sew/caps solid built from the B-spline (would not pass on surface fit alone). Not adopted.

## Results this round

| part | self_check | rms | p95 | s2m | vol |
|---|---|---|---|---|---|
| WINGTIP L 2 | FAIL | **0.059** | 0.102 | 0.014 pass | 0.999 |
| WINGTIP R 2 | FAIL | **0.060** | 0.108 | 0.013 pass | 0.999 |
| WING 2 L | FAIL | ~0.83 (prior; this-round eval hung) | — | — | — |

Vertex gate still ~3.4× too tight (need ≤0.017 / ≤0.035). Progress 0.158→0.065→**0.059**.

## Pipeline wire / matched-pair rerun
**Skipped** — L2 vertex self_check not green.

## Code
- `meshsolid/loft.py`: `_outer_section_span` + tip path uses it before apex loft
- TDD: `tests/test_loft_smooth_skin.py` **6 passed**

## Next
1. Tip residual still dominates — try true OCC degenerate tip (multiple AddVertex / cone) or tip-local freeform patch sewn onto mid loft
2. TE residual after tip fix: revisit upper/lower sew matching truth (2 BSpline + 2 Plane) with volume-safe solidification
3. WING 2 L mid-skin once tip path is within ~2× of gate
4. Wire only after L2 vertex self_check green

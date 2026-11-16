# Curved-skin round 2o — section vs spanwise diagnosis

**When:** 2026-11-09 ~16:40 AWST (Australia/Perth)  
**Branch:** `wip-local-trim`  
**Floors:** `tests/test_loft_drone_regression.py` + `test_loft_smooth_skin.py` — **9 passed** (wingtip rms 0.05855 ≤ 0.059).

## Question
If the belly is a smooth concave region, a correctly sampled closed section loft should capture it. Separate **per-section 2D fit** (native section polyline vs resampled curve/polyline) from **between-section interpolation** on **WING 2 L**.

## Measurements (`curved_skin_round2o_section_vs_span.json`)
| Quantity | median | max | notes |
|---|---|---|---|
| Section poly M=96 | 0.308 | 0.454 | undersampled |
| Section poly M=160 | 0.130 | 0.229 | prior shared-wire floor |
| Section poly belly@160 | 0.222 | 0.376 | lower y∈[12,18] |
| Section B-spline@160 | 0.120 | 0.189 | similar to poly |
| Section poly M=320 (post) | ~0.067 | ~0.087 | belly~0.07 |
| Span blend @ dh=2 | **0.0004** | — | near-zero |
| Span blend @ dh=5 | **0.007** | — | near-zero |
| Span @ dh≈10 (coarse) | 1.90 | 9.06 | mostly TE-align artifact |

**Span axis:** `sweep_axis` d=`[-1,0,0]` · |d·pca1|=0.96 (span) · |d·pca0|=0.28 (chord). **Axis OK** — not a PCA-max mistake.

**Verdict:** dominant cause was **section resample/param** (chordal undersample of closed loops), not span density or axis. Operational span spacing already fine.

## Fix applied (`meshsolid/loft.py`)
1. Closed shared-wire **uniform arc M floored at max(n_nat, 320), cap 320** (belly section floor ~0.07).
2. Prefers **smooth** before te_le/cornered.
3. `_densify_closed` kept but **off for loft wires** — densify zeros section fit but breaks spanwise homology → ThruSections **balloon** (smooth fail → te_le).
4. Curvature-weighted resample remains off for sew (starves low-turn belly).
5. Strip path still tries first; sew still open (`free=37`) → section fallback.

## Probe (`curved_skin_round2o_probe.json`)
| Part | vol/mesh | rms | mode |
|---|---|---|---|
| WING 2 L | 1.0124 | **0.573** | smooth (M320) |
| WINGTIP L 2 | 0.9988 | **0.05855** | (floor held) |

Prior smooth@M≤160 rms~0.589 → M320 **0.573** (Δ~0.016). Section floor fixed; **loft rms still ≫0.2**.

## Implication
Correctly sampled closed sections are no longer the limiter. Residual ~0.57 is **non-section** (mesh verts off silhouette / chordwise features / strip tracks). **Continue strip sew closure** next (shared TopoDS rails / free-edge fix); do not chase more section M.

## Not done
- Strip MakeSolid free edges 37 → closed solid
- Aileron cutouts
- WING 2 L rms&lt;0.2 unmet

# Curved-skin round 2i checkpoint (2026-11-09 ~13:05 AWST)

## Ask
Time-box tip/TE (try tip-local patch); shift breadth to WING 2 L mid-skin 0.83 + hang (add time limit); report L2 per-zone residual map. No gate changes.

## WINGTIP L 2 — per-zone residual map (gates unchanged: rms≤0.017, p95≤0.035)

Current loft (micro-tip span + apex, ≥80 sections, TE seam start, smooth-prefer): **overall rms 0.0585 / p95 0.102** (still FAIL vertex self_check).

| zone | n verts | rms | p95 | mean | max |
|---|---|---|---|---|---|
| **ALL** | 3013 | **0.0585** | 0.102 | 0.022 | 0.522 |
| tip (5% span ends) | 298 | **0.026** | 0.055 | 0.013 | 0.206 |
| TE band (8% chord) | 110 | 0.043 | 0.096 | 0.022 | 0.213 |
| **LE band (8% chord)** | 187 | **0.119** | **0.267** | 0.073 | 0.404 |
| mid skin | 2418 | 0.055 | 0.077 | 0.020 | 0.522 |

**User judgment:** after tip micro-sections, **tip is no longer the bottleneck** (0.026 rms, near gate). **LE band dominates** overall error; mid has a few high outliers (max 0.52) but typical mid is OK (p95 0.077).

Artifact: `results/drone_pairs_all/wingtip_l2_zone_map.json`

## Tip-local patch (time-boxed, not adopted)

| attempt | tip-vert rms | vs loft tip |
|---|---|---|
| freeform `fit_freeform` on tip verts | failed (no plane/tube param) | — |
| section-grid periodic LS on hi-tip | **0.225** | loft tip **0.072** |

Patch worse than loft → **not adopted**. Tip work for this round stops here.

## Hang fix — time limits landed

Prior WING 2 L “hang” was post-build: build itself ~32s; fine `deviation` tessellation (`lin=tol/4`) can run for minutes. Added:

- `t_stop` checks in section seed loop and before ThruSections
- abort before UnifySameDomain / ShapeFix if over budget (return reason)
- self_check: if `<20s` remain, use coarser `lin=max(tol,0.02)` and 3000 samples; if already over budget, fail with explicit reason

Gates unchanged. TDD: `tests/test_loft_smooth_skin.py` **6 passed**.

## WING 2 L mid-skin 0.83 — diagnosis

| fact | detail |
|---|---|
| Topology | planar caps `capf≈0.10`; single slab `[-395,-290]`; smooth loft (0 corners) |
| Volume | 1.009 — solidifies; miss is geometry not underfill |
| Zone rms | near_caps **0.31**; TE_band **1.36**; LE_band **0.43**; **mid_skin 1.06** |
| Section→loft | resampled sections only ~0.1–0.2 rms — verts between features miss more |
| Spanwise structure | root half: **~10–14 corners >60°**; tip half: **2 corners**; area ramps 2500→3160 then flat |
| `corner_deg>0` | fails: “corner turning signs do not match” across the 14→2 transition |

**Root cause (breadth):** smooth corner-free loft on a wing that has **real feature corners** (TE + spars/hinges) on the root side. Mid-skin 0.83 is not tip-cap error.

Area-jump slab split did not fire (gradual ramp). Split at corner-count transition (~−347) was started but fuse/eval exceeded the round; not landed.

Artifact: `results/drone_pairs_all/wing2l_zone_map.json`

## Pipeline wire
**Skipped** — no green self_check.

## Next (breadth-first)
1. Auto slab breaks (or adaptive corners) where sharp-corner count changes along span — unblocks wings/ailerons class
2. LE-aware sampling on wingtips once wing path is healthier (LE band is L2’s remaining error)
3. Wire loft only after matched-pair greens

# Curved-skin round 2p — reconcile 0.57 rms vs section/span floors

**When:** 2026-11-09 ~17:00 AWST (Australia/Perth)  
**Branch:** `wip-local-trim`  
**Floors:** unchanged from 2o (9 passed; wingtip 0.05855). Not re-run this slice (loft.py unchanged for floors).

## Reconcile
| Floor | Value |
|---|---|
| Section poly M320 / belly | ~0.07 |
| Span blend @ dh≤5 | ~0 |
| **Observed loft rms** | **0.573** |

**Cause:** residual is **not** belly undersample or span density. **~91.8% of SSE** is in **212 verts** in two on-skin pocket clusters (root half). Skin-only rms excluding them: **0.188**. `d<0.5` skin rms: **0.176**.

## Localisation (`curved_skin_round2p_residual_localise.json`)
- **SSE:** root half 87.6% / tip 12.4%; rms root 0.65 / tip 0.36
- **n_loops = 1** at all stations — features are **outer-silhouette recesses**, not inner holes
- **Corners@35°:** 10–14 on root half (h ≲ −350), drop to 2 on clean mid (−342…−325, rms~0), 6 near tip break
- **Top clusters:**
  1. n=111, h≈−353, cen≈(353, 17.3, 468), rms 0.99, max 1.64, **pocket_or_boss**, sil=0.95 (on skin)
  2. n=97, h≈−370, cen≈(370, 17.7, 509), rms 0.96, max 1.59, **pocket_or_boss**, sil=0.97
  3–4. n=2 outliers max~4.4 (tip/root pins), off-skin
- Heat-map shows rectilinear/triangular cutout on lower/root skin (servo pocket / channel)

### Artifacts
- `wing2l_2p_deviation_heatmap.png` (4-view, cyan cluster centroids)
- `wing2l_2p_deviation_preview.png`
- `wing2l_2p_residual_vs_span.png`
- `wing2l_2p_deviation.ply`

## Plan — outer skin loft + features via primitives
1. **Keep** smooth shared-wire loft as outer skin (already ~0.19 on non-feature verts).
2. **Cluster** high residuals (`meshsolid/loft_features.py`).
3. **Build tools:** section-recess polygons swept spanwise / OBB / `fit_cylinder`; **Cut** pockets (or Fuse bosses) — same BOP pattern as `build_local`.
4. Optional later: feature-suppressed section bridges for loft itself (full convex envelope **failed**, rms 3.2 / vol 1.08 — too aggressive).

## Started this round
- Residual localisation + heatmaps + span/corner profile
- Skin-only SSE split (92% features)
- Probe cuts: PCA-OBB prisms (vol ok, rms **worsened** 0.573→0.586)
- Probe cuts: single-station recess prisms (1/2 ok, rms →0.592) — depth/alignment wrong
- New helper module `meshsolid/loft_features.py` (cluster / classify / skin mask)

## Next
- Multi-station pocket rails (match recess across span) + depth from mesh-vs-skin signed gap
- Prefer Cut only where mesh is inside loft (true cavity)
- Then aileron cutouts; keep wingtip ≤0.059

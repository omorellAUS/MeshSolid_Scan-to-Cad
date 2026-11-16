# Curved-skin round 2n checkpoint

**Ask:** Generic strip loft (spanwise crease/guide tracks → one B-spline face per strip → sew + caps); investigate lower skin y∈[12,18]; goal WING 2 L rms <0.2; keep floors; then aileron cutouts. ~25 min.

## Floors (9/9)

| part | floor | 2n |
|------|-------|-----|
| WINGTIP L 2 | rms ≤ 0.059 | **0.05855** |
| AILERON | vol ±5%, 1 solid | 1.016 / nsol=1 |
| WING 2 L | ≥3 levels, no hang | 4 levels; strip tries then te_le fallback |

## Lower skin y∈[12,18]

- **42% of mesh verts**; band rms ~0.66 under te_le shared-wire (upper y∈[30,35] rms ~0.005)
- Worst points ~`(373, 15.4, 508)` — mid-chord lower, forward of mid
- Section min-y has **turn ≈ 0.01** (smooth concave belly, not a sharp crease)
- TE/LE pins miss this region → ThruSections chords across the belly
- Mesh dihedral creases are **mostly chordwise** (n≈92 chains with dspan≈5); only one partial spanwise crease near z≈493 on the lower skin (dspan≈46 of ~105)

## Strip loft (generic)

Implemented in `meshsolid/loft.py`:

- `_track_guides` — TE/LE + soft section corners + belly/top extrema, nearest-matched along span; ordered in section (u,v) angle
- `_fit_strip_surface` — bbox-guarded `GeomAPI_PointsToBSplineSurface` (impeller-style)
- `_strip_loft_skin` — one face per angular sector between guides; rail-matched end caps; sew
- `build_loft` tries strip first on multi-slab non-tip; falls back to shared-wire (te_le) then single slab

**Status:** strips build (4 guides, areas sane after geometric short-arc fix); sew still leaves **free≈37** edges → MakeSolid fails → **fallback te_le rms 0.618** (same as 2m). Not yet <0.2.

Progress on sew: free 297 → 216 → 49 → **37** (UV order, length-based short arc, rail-endpoint caps). Still need shared TopoDS crease edges (e.g. MakeFilling with boundary = shared rail edges) for a watertight shell.

## Goal vs status

| goal | status |
|------|--------|
| Lower-skin diagnosis | **done** (concave belly, not missed sharp corner) |
| Generic strip pipeline | **in tree**; sew closure incomplete |
| WING 2 L rms < 0.2 | **not met** (0.618 via te_le fallback) |
| Floors | **held** |
| Aileron cutouts | **not started** (sections still nloops=1) |

## Next

1. Close strips with **shared TopoDS rail edges** (`BRepOffsetAPI_MakeFilling` / edge-bounded faces) so Sewing free → 0
2. Re-measure lower-band rms once solid; densify across-arc on belly sector if still high
3. Aileron cutout / hinge as separate path after outer skin <0.2

## Artifacts

- `meshsolid/loft.py` — strip loft helpers + build_loft try/fallback
- `results/drone_pairs_all/curved_skin_round2n_{checkpoint.md,probe.json}`

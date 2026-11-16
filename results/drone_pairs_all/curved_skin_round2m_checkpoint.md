# Curved-skin round 2m checkpoint

**Ask:** Cornered shared-wire slabs on WING 2 L with intents 10/6/2 (corner-matched within slab; break wires = union of both sides); goal rms ≪ 0.2; then aileron cutout loops. Floors kept. ~25 min.

## Floors (9/9)

| part | floor | 2m |
|------|-------|-----|
| WINGTIP L 2 | rms ≤ 0.059 | **0.05855** |
| AILERON | vol within 5%, 1 solid | vol 1.016, nsol=1 |
| WING 2 L | ≥3 levels, no hang | 4 levels, t~8s build |

## WING 2 L shared-wire cornered

**Intents** measured 10 / 6 / 2. Break-union upgrade cascades to 10/10/10 (ThruSections needs matching edge counts on both sides of each shared break).

**True shared TopoDS wires** (same object as end of slab i and start of i+1) are required for Sewing to close. Caps must use those same outer wires.

| attempt | result |
|---------|--------|
| Independent cornered slabs + geometric sew | free edges remain; invalid / wrong vol |
| Shared wires + forced n=10 follow | corner correspondence jumps (max step ≫ chord); ThruSections balloons (area 10^11+) |
| Nearest-corner reassignment at n=10 | still unstable on tip-adjacent slabs |
| Per-slab `_loft_slab` solids face-sew | slab1/2 often inverted; free edges persist |
| **TE/LE 2-corner shared-wire** (stable pins) | **vol 1.012, rms 0.618** (was ~1.07 smooth @2l) |
| Smooth denser (80 sect, no corners) | rms ~0.69 |
| TE/LE denser (48 sect) | rms ~0.62 |

Mode cascade now in `_sew_slabs_shared_wire`: `cornered` (if max n_use≤4) → `te_le` → `smooth`, with shell-area / volume sanity to reject balloons. WING 2 L takes **`te_le`**; intents still recorded on each slab.

### Error map (smooth denser baseline)

- Lower surface y∈[12,18] dominates (rms~0.94); upper y∈[30,35] ~0.01
- Root span worse than tip; mid-chord z∈[480,540] worst
- Sections are **single outer loop** along span — no aileron cutout holes to loft as inner wires on this sweep axis

## Aileron cutout loops

Not started. WING 2 L sections show nloops=1 at all stations; cutouts are not spanwise holes under current sweep. Next: hinge-local / chordwise treatment or separate AILERON hinge-corner path (still rms 0.84).

## Goal vs status

| goal | status |
|------|--------|
| intents 10/6/2 recorded + break union | **done** (upgrade→10; loft uses te_le pins) |
| true shared-wire sew, 1 solid | **done** |
| WING 2 L rms ≪ 0.2 | **not met** (0.618; was 1.067) |
| aileron cutout loops | **not started** |

## Next

1. Stable multi-corner follow (or strip lofts) so upgraded 10/6/2 can run without ballooning — TE/LE alone cannot pin the lower-skin mid-chord error enough for ≪0.2
2. Adaptive mid-section insert where section chord error is high (partial trial: slow, small gains)
3. Aileron cutout / hinge: different sweep or local boolean after outer skin
4. LE-local still off by default (mid/TE floors)

## Artifacts

- `meshsolid/loft.py` — `_sew_slabs_shared_wire` mode cascade + intent metadata + balloon guards
- `results/drone_pairs_all/curved_skin_round2m_probe.json`

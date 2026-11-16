# Curved-skin round 2l checkpoint (2026-11-09 ~14:40 AWST)

## Ask
Account for bisect 0.046 vs fixed 0.0586; recover 0.046 if real (tighten floor). Then shared-wire shell sew join, aileron hinge corners, LE-local sampling. Gates unchanged. ~25 min.

## 0.046 vs 0.0586 — not a durable baseline

| source | WINGTIP L 2 rms |
|---|---|
| Bisect toggle `dense_at=None` on 2j tree (one run) | 0.0457 |
| Current (no dense_at), 3 repeats | **0.05855** (identical) |
| `git show 1d96ac0:meshsolid/loft.py` on same mesh | **0.05855** |
| Ms cap 96 / 128 / 160 | all **0.05855** |

**Conclusion:** 0.046 was a one-off (not reproduced; not the 1d96ac0 baseline). Real restored floor is **0.05855**. Regression test stays **rms ≤ 0.059** — do not tighten to 0.046.

## Shared-wire shell sew (multi-slab, no BOP) — landed

`_sew_slabs_shared_wire`: one section wire per break height, reused as end of slab i and start of slab i+1 → shell `ThruSections(isSolid=False)` → `BRepBuilderAPI_Sewing` + planar outer caps → `MakeSolid`. Outer AABB caps inset for section hits.

**WING 2 L:** nsol=1, vol **1.010**, t~4s build / ~39s with coarse self_check, 3 slabs with `shared_wire=True`, `corner_count_intent` 10/6/2. rms still ~1.07 (smooth sections; cornered shared wires not yet).

BOP fuse path retained as fallback for tip/single-slab only.

## Aileron hinge

- `corner_deg=0`: vol 1.016, rms 0.84 (unchanged)
- `corner_deg=35`: 4 corners, **vol ×29** balloon — not adopted
- Hinge still needs open-section / cutout-aware loops (deferred)

## LE-local sampling

- Added `le_band=(lo,hi)` on `_resample` (half samples in band)
- Enabled on tip TE-seam → overall rms **0.148** (LE band better, mid/TE worse) — **disabled by default**
- Same lesson as `dense_at=0.5`: LE densify without mid/TE budget regresses overall

## Probe (`curved_skin_round2l_probe.json`)

| part | nsol | vol | rms | notes |
|---|---|---|---|---|
| WINGTIP L 2 | 1 | 0.999 | **0.0586** | floor held |
| WING 2 L | 1 | 1.010 | 1.067 | shared-wire sew; corner intents recorded |
| AILERON | 1 | 1.016 | 0.84 | hinge corner loft balloons |

Tests: `test_loft_smooth_skin` + `test_loft_drone_regression` → **9 passed**.

## Next
1. Shared-wire sew with **per-slab cornered** interface wires (intent 10/6/2) to cut WING 2 L mid-skin rms
2. Aileron: hinge cutout as inner loop / partial span, not global corner_deg
3. LE-local only inside tip micro-span or with explicit mid/TE sample floors
4. Wire loft after greens

## Artifacts
- `results/drone_pairs_all/curved_skin_round2l_checkpoint.md`
- `results/drone_pairs_all/curved_skin_round2l_probe.json`

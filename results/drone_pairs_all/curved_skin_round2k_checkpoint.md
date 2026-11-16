# Curved-skin round 2k checkpoint (2026-11-09 ~14:13 AWST)

## Ask
WINGTIP L 2 regressed 0.059→0.104; bisect `1d96ac0..2c32a06`, restore ≤0.059 while keeping slab work; add fixed regression check for wingtip/aileron/WING 2 L; continue Next list. Gates unchanged. ~25 min.

## Regression cause (bisect)

Only one commit in range (`2c32a06`). Feature toggle on current tree:

| config | WINGTIP L 2 rms |
|---|---|
| current with `dense_at=0.5` on TE-seamed closed runs | **0.104** |
| same code, `dense_at=None` (1d96ac0 uniform arc-length) | **0.046** |

**Cause:** LE mid-arc densify (`dense_at=0.5` in `_runs`) introduced in 2j. Not tip-apex gate, not Ms=128, not key_tol ladder.

**Fix:** removed `dense_at` from `_runs` (API kept on `_resample` for later LE-local work); Ms cap back to 96. Restored **rms 0.05855 ≤ 0.059**.

## Regression floors (`tests/test_loft_drone_regression.py`)

Skip if user scans absent. Floors:

- **WINGTIP L 2:** rms ≤ **0.059**, \|vol−1\|<2%, nsol=1, <60s
- **AILERON:** \|vol−1\|<5%, nsol=1 (anti tip-apex balloon)
- **WING 2 L:** ≥3 levels (corner breaks fire), returns <100s (no silent OCC hang)

`pytest tests/test_loft_smooth_skin.py tests/test_loft_drone_regression.py` → **9 passed**.

## Next-list progress

1. **Cornered slabs individually work** after fixing `_align_closed` misuse on open corner runs (was reversing runs → `section wire is not connected`). Tip slab 2-corner OK; root 10-corner OK.
2. **Fuse of cornered (and even multi smooth) slabs** via `BRepAlgoAPI_Fuse` either **hangs minutes** or returns **crumb solids** (n_solids=25, invalid). Added timed BREP-subprocess fuse + validation; on failure **fall back to single-span smooth loft**.
3. Corner-count **breaks still recorded** in `levels` / `corner_count_intent`; multi-slab loft stays smooth until non-BOP join lands.
4. Aileron/FUS hinge work not started this round (time).

## Probe (`curved_skin_round2k_probe.json`)

| part | ok | nsol | vol | rms | notes |
|---|---|---|---|---|---|
| **WINGTIP L 2** | no (gates) | 1 | 0.999 | **0.0586** | restored; LE densify reverted |
| **AILERON** | no | 1 | 1.016 | 0.84 | vol floor held |
| **WING 2 L** | no | 1 | 1.009 | 0.96 | breaks fire; multi-fuse rejected → single-slab fallback |

## Next
1. Non-BOP multi-slab join: shared interface wire + shell ThruSections + sew + MakeSolid (oriented).
2. Or simplify end caps before BOP so fuse does not hang/crumb.
3. Aileron hinge corners once join works.
4. LE-local sampling (not global dense_at) for wingtip LE band 0.119.
5. Wire loft only after greens.

## Artifacts
- `tests/test_loft_drone_regression.py`
- `results/drone_pairs_all/curved_skin_round2k_checkpoint.md`
- `results/drone_pairs_all/curved_skin_round2k_probe.json`

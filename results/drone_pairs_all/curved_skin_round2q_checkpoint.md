# Curved-skin round 2q — printed accuracy class + pocket cuts

**When:** 2026-11-09 ~17:31 AWST (Australia/Perth)  
**Branch:** `wip-local-trim`  
**Tests:** `tests/test_accuracy.py` + loft floors + smooth skin — **13 passed**.

## Accuracy class (user: 0.05 mm for printed drone parts)
New module `meshsolid/accuracy.py` + CLI `--accuracy {machined,printed}` + `Params.accuracy`.

| Class | vertex→solid rms | p95 | When |
|---|---|---|---|
| **machined** (default) | 1.5·tol | 3·tol | machined / scanned; **unchanged** |
| **printed** | **0.05 mm** | **0.10 mm** | opt-in for FDM/SLA drone parts |

- Report field `accuracy` states class + limits + basis.
- Explicit `accept_rms` / `accept_p95` still override.
- **Never** auto-applied to machined parts; default remains machined so existing regressions are identical.
- Loft `build_loft(..., accuracy=)` uses the same gates in `self_check`.

## WINGTIP L/R 2 re-score (honest)
| Part | self_check rms | p95 | printed (≤0.05/0.10) | regression floor ≤0.059 |
|---|---|---|---|---|
| WINGTIP L 2 | **0.05855** | 0.1021 | **FAIL** (rms & p95) | **PASS** |
| WINGTIP R 2 | **0.06036** | 0.1075 | **FAIL** | **FAIL** |

Machined gate (1.5·tol ≈ 0.017) also fails both — same as before; printed is looser but still not met. **Do not nudge.** Artifact: `curved_skin_round2q_wingtip_accuracy.json`.

## Pocket cuts (WING 2 L)
- Direction: mesh vertex normals (≈ −Y); depth from `| (cp−V)·n |` capped at 2.5 mm (was ~5 mm → invalid BOP).
- Tools build; cut#0 → **nsol=2** (rejected); cut#1 accepted; full rms **0.573 → 0.573** (no meaningful gain). Skin-only target still **0.188**.
- Next: keep single solid (smaller rim / common-volume cut), multi-station pocket rails, then belly skin-only.

Helpers: `meshsolid/loft_features.py` (grow / floor+walls / hull prism / cut). Probe: `curved_skin_round2q_pocket_cut.json`.

## Not done
- Pocket rms → ~0.188
- Belly skin-only 0.188 → lower
- Matched-pair rerun with `--accuracy printed` once loft+pockets wired end-to-end

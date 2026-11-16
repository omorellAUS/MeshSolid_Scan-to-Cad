# Curved-skin round 2g checkpoint (2026-11-09 ~12:34 AWST)

## Ask
Fix TE/LE/tip construction so loft passes **vertex** self_check on WINGTIP L 2 (TE crease C0 / upper-lower matching truth / degenerate tip); then R 2, WING 2 L mid-skin; wire + limited rerun only if green.

## Construction probes (not adopted as defaults)

| attempt | result |
|---|---|
| TE+LE upper/lower open runs + tip apex, span trimmed on sep collapse | **worse** rms 0.25 (tip verts outside trimmed span); tip sep collapses only near extreme tip |
| Closed loft with TE start only (major-axis → TE seam) | alone can regress if span/apex mishandled |
| Single TE corner (open run TE→TE as C0 seam) | underfill vol ~0.54 — wire closure / solidification break |
| Truth topology note | 2×BSpline + 2×Plane: planes are **TE/LE-parallel** (n·d≈0.08), **not** end caps; tips are pointed → apex path stays correct |
| Smooth vs ruled ThruSections | volume-pick preferred **ruled**; smooth often underfills at high section counts but when vol OK is **better** skin fit |

## Landed construction changes (`meshsolid/loft.py`)

1. **`_thru`**: prefer **smooth** when volume within 2% of target; else closest |dV| (still tries smooth then ruled)
2. **TE seam start** on pointed (tip_apex) smooth skins: `_te_le_corners` locks closed-loop start at TE; followed across sections
3. **`_area_span`**: default samples 81→**201** (capture tip micro-sections)
4. **Refine**: tip/smooth uses `0.15·tol` mid-err insert threshold (was 0.25·tol)
5. **Seed**: Chebyshev seed up to **40** stations; tip_apex skins auto **`max_sections = max(..., 80)`**

TDD: `tests/test_loft_smooth_skin.py` **6 passed**. Gates unchanged (vertex still 1.5/3·tol).

## WINGTIP L / R 2 (still not green)

| part | before (2e/2f, ms40) | now (default≥80) | gate |
|---|---|---|---|
| L 2 rms | 0.158 | **0.065** | ≤0.017 |
| L 2 p95 | 0.333 | **0.121** | ≤0.035 |
| L 2 s2m | 0.047 (fail) | **0.014 (pass)** | ≤0.042 |
| L 2 vol | 0.997 | 1.002 | ±|
| R 2 rms | 0.158 | **0.066** | ≤0.017 |

~2.4× rms improvement; s2m now green; **vertex self_check still FAIL** (~4× above rms gate). No STEP re-bidirectional this round (prior ~0.035 still the reference).

## WING 2 L mid-skin

- Planar-cap path unchanged (`capf≈0.10`, no tip apex / no auto-80)
- rms 0.93 → **0.83** (ms=40) — still far; mid-skin construction remains open

## Pipeline wire / matched-pair rerun

**Skipped** — no green vertex self_check.

## Honest status
- TE-aligned closed seam + smooth-prefer + denser tip sampling **help** but do not clear the vertex gate
- Upper/lower TE+LE split and single-TE-corner open run **not** ready (regress vol or rms)
- Remaining error still concentrated at tips + TE band; need a tip treatment beyond AddVertex (true degenerate tip sections / better apex) and/or a non-underfilling smooth loft that preserves the TE crease as a hard generator

## Next
1. Degenerate tip **section wires** (tiny loops near AABB) instead of/in addition to AddVertex — match mesh tip bluntness
2. Hard TE generator edge through ThruSections (or sew upper/lower shells) without underfill
3. WING 2 L mid-skin once tip path greens
4. Wire + limited pair rerun only after WINGTIP L 2 self_check green

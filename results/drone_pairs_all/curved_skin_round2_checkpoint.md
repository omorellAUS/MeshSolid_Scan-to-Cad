# Curved-skin round 2 checkpoint (2026-11-09 Australia/Perth)

## Done this slice
- Screw isolated retry **5/5 ok** committed (`cd8dd72`); `results/m8_screw_rotors.json` seconds refreshed only.
- Pairs_all baseline **preserved** (0/35 solids); no full 35 rerun.
- Investigated WIP `meshsolid/loft.py` + `lscm.py` (LSCM is parametrisation helper; not yet a solid builder).

## Root cause (construction)
Local-trim open shells on matched skins remain the all-35 failure mode. Section-loft is the intended curved-skin path but was unfinished:
1. **Corner-free loops rejected tessellation turns** (`_corners_like` m==0 raised on polyline dihedrals) — **fixed**; unit test green.
2. **OCP 8.x imports** for interpolate arrays — **fixed** (`OCP.collections`).
3. **Float height keys** in section cache — **fixed**.
4. **ThruSections** now falls back to ruled; synthetic extruded airfoil produces a solid but **volume ~10× mesh** — volume gate correctly rejects (not loosened).
5. **WINGTIP L 2** (representative, sweep_frac≈0.998): after smooth mode, fails on **tapered section area change** along span; area-ratio gate relaxed generically for taper; still unsolved end-to-end.

## Tests
- `tests/test_loft_smooth_skin.py`: `test_corners_like_corner_free_allows_tessellation_turns` **PASS**
- `test_loft_builds_faceted_extruded_airfoil` **FAIL** (honest): solid built but volume gate fails (245324 vs 25933)

## Not done
- No genuine valid closed solid yet on a real matched drone STL with independent STEP bidirectional deviation.
- Loft not wired into `pipeline.py`.
- Package zip **not** rebuilt (no tested green solid yet).
- Second representative not attempted beyond probes.

## Active PIDs
- None for pairs/screw (complete). Loft experiments were foreground.

## Next remaining work
1. Fix closed-section start alignment / resample so ThruSections volume matches mesh on synthetic extruded airfoil (TDD until green).
2. Retry WINGTIP L 2 mesh-only smooth loft; validate vs matched STEP with bidirectional samples (truth only).
3. Second distinct representative (e.g. AILERON 1R VTOL or WING 2).
4. Wire loft into pipeline behind sweep/cad gates; only then consider limited pair reruns.
5. Rebuild install zip only after tested code change.

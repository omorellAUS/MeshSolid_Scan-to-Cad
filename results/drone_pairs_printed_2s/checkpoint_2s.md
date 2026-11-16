# Checkpoint 2s — pipeline loft + printed pairs + 0.1.1 zip

**When:** 2026-11-09 ~19:10 AWST (Australia/Perth)  
**Branch:** `wip-local-trim`

## (1) STEP bidirectional — WINGTIP L/R 2 (new build, `--accuracy printed`)
CLI early-path **`section_loft`**, status **ok**, mesh gates printed PASS.

| Part | mesh→solid rms | rebuilt↔truth hausdorff | t2r/r2t rms (aligned) | align med |
|---|---|---|---|---|
| WINGTIP L 2 | 0.0419 | 0.30 / 0.41 | ~0.011 / ~0.013 | ~0.26 |
| WINGTIP R 2 | 0.0400 | 0.25 / 0.28 | ~0.012 / ~0.015 | ~0.26 |

Independent STEP truth check: bidirectional surface distances low after rigid align; residual align median ~0.26 mm (pair may need mirror/portion selection polish). Artifact: `wingtip_step_bidirectional.json`.

## (2) Pipeline wiring
- Faceted CAD tessellation → try `build_loft` first; accuracy-class gates; **early-return only on accept**.
- Fail → prismatic/freeform (unchanged). Machined/scan path preserved.
- CLI print SyntaxError fixed; freeform wing test accepts `section_loft` or freeform.

## (3) Regressions (machined default)
**45 passed**, 1 deselected (coupling STL absent). Includes pipeline, screw, drone_fixes, accuracy, loft floors, pack zip.

## (4) Install zip **0.1.1**
`dist/MeshSolid-0.1.1-windows.zip` (regs green). Manifest + package.xml bumped.

## (5) Matched-pair `--accuracy printed` (in flight)
Out: `results/drone_pairs_printed_2s/`. 18 cases (wingtips, wings, ailerons, roots). Wingtips L/R 2 done via loft; others running (WING 1 R V2 on freeform/build path). Score vs STEP after runs finish.

## Still open
- Pairs scoring + remaining runs
- Pockets → skin-only 0.188; TE zone tip floor
- Coupling regression skipped (STL path not on box)

## Matched-pair score (partial, runs still finishing)

| STL | status | method | mesh rms | vs-STEP hausdorff | t2r rms |
|---|---|---|---|---|---|
| WING 1 L V2 | failed | revolve_profile | None | None | None |
| WING 1 R V2 | failed | revolve_profile | None | None | None |
| WING 2 L | failed | revolve_profile | None | None | None |
| WING 2 R | timeout | None | None | None | None |
| WINGTIP L 1 | failed | revolve_profile | None | None | None |
| WINGTIP L 2 | ok | section_loft | 0.04193 | 0.34632 | 0.01694 |
| WINGTIP R 1 | failed | revolve_profile | None | None | None |
| WINGTIP R 2 | ok | section_loft | 0.04 | 0.26081 | 0.01742 |
| AILERON 1R VTOL | failed | revolve_profile | None | None | None |
| WING ROOT VTOL R | failed | revolve_profile | None | None | None |
| WINGTIP L 2 | ok | section_loft | 0.04193 | 0.34632 | 0.01694 |
| WINGTIP R 2 | ok | section_loft | 0.04 | 0.26081 | 0.01742 |

## Matched-pair honest read (partial)
- **WINGTIP L/R 2**: only greens via `section_loft` + printed gates; STEP bi hausdorff ~0.26–0.35, surface rms ~0.017.
- **WINGTIP L/R 1**: loft tried, **fails printed** (rms~0.64) → revolve fallback fails.
- **WING 2 L**: loft tried, fails printed (rms~0.57 pockets) → revolve fails. Pocket path still needed before printed accept.
- **WING 1 L/R V2**: not coarse-faceted (loft not tried); revolve fails.
- **Ailerons / roots**: loft miss or not faceted; fallback fails/timeouts.
- Pair runner still finishing remaining VTOL cases in background.

## Next
1. Tip1 LE densify / tip-apex path parity with tip2  
2. WING2L pockets → skin-only 0.188 then accept loft under printed  
3. TE zone floor on tip2  
4. Re-score printed_2s when runner done


---

## Update 2026-11-09 20:20 AWST

Commits: `103d816` (pockets+TE+coupling paths), `5a1b417` (batch --jobs, multi-body parallel, Fusion --device auto).

- Coupling STL restored under `tests/user_scans/mechanical/`; pytest `-m slow -o addopts=` **PASS**; CLI local_trim loft=False.
- Loft-guard: M3 all ok loft=False; prepped s04–s06 ok loft=False; s07/s09 impeller builds loft=False (machined gate fail without exclusions); stg1 in flight.
- Tip2 TE densify bump: TE rms ~halved, printed still green (smoke rms 0.042).
- WING2L pockets wired into build_loft; skin_rms 0.331 (target 0.188) still open.
- Tip1: rms 0.323 after one pocket; mid outliers ~3.3 mm.
- 0.1.1 zip rebuilt: `dist/MeshSolid-0.1.1-windows.zip`.
- GPU: `--device auto` / Fusion default; Warp CUDA when NVIDIA present else CPU. Batch `--jobs`, `MESHSOLID_BODY_JOBS` for multi-body; engine remains out-of-process with time limits.

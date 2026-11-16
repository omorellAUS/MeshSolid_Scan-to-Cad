# Drone / VTOL STL ↔ STEP pairing inventory

Source: `/workspace/meshsolid_testset/drone/inventory.json` (from `tools/testset/pairing.py`).
Inputs on disk: 27 STEP + 52 STL under `tests/user_scans/drone` (gitignored).
Generated: 2026-11-09 08:57 AWST.

## Counts

- STEP files: 27
- STL files: 52
- Verified matches: **35** (of which mirrored: 5)
- Unmatched STLs: **17**
- L/R fingerprint twins recorded: 17

Method: pose-invariant fingerprint → rigid ICP (proper + mirrored) → point-to-surface check.
Units: mm. No unverified reference matches. Pair test must rebuild from the STL and compare to the matched STEP portion — importing the source STEP is not success.

## Matched (35) — include both mirrors in the pair test

| group | STL | → | how | STEP product | solid | pair p95 mm | mir |
|---|---|---|---|---|---|---|---|
| drone | REAR HATCH 2 | → | STEP solid #1+3 | REAR HATCH | 1+3 | 0.0494 | False |
| drone | MOTOR MOUNT D19x16mm | → | whole STEP | MOTOR MOUNT D19x16mm | -1 | 0.0113 | False |
| drone | REAR HATCH 1 | → | STEP solid #0+2 | REAR HATCH | 0+2 | 0.025 | False |
| drone | WINGTIP L 1 | → | mirror of STEP solid #1 | WINGTIP R | 1 | 0.035 | True |
| drone | WING 2 R | → | mirror of STEP solid #0 | WING 2L MODIFIER | 0 | 0.0222 | True |
| drone | WINGTIP L 2 | → | mirror of STEP solid #0 | WINGTIP R | 0 | 0.0196 | True |
| drone | FRONT HATCH 1 | → | STEP solid #0+3 | FRONT HATCH | 0+3 | 0.2205 | False |
| drone | NOSE TOP | → | whole STEP | NOSE TOP | -1 | 0.0065 | False |
| drone | FUS 1 L | → | STEP solid #0+4 | FUS 1 MODIFIER | 0+4 | 0.0366 | False |
| drone | NOSE CLEAN | → | whole STEP | NOSE CLEAN 1 | -1 | 0.0192 | False |
| drone | WING 1 R V2 | → | mirror of STEP solid #0 | WING 1L MODIFIER | 0 | 0.0275 | True |
| drone | WINGTIP R 2 | → | STEP solid #0 | WINGTIP R | 0 | 0.0154 | False |
| drone | FUS 4 R V2 | → | STEP solid #0 | FUS4 MODIFIER | 0 | 0.0243 | False |
| drone | FRONT HATCH 2 | → | mirror of STEP solid #1+2 | FRONT HATCH | 1+2 | 0.0781 | True |
| drone | WING 2 L | → | STEP solid #0 | WING 2L MODIFIER | 0 | 0.0373 | False |
| drone | NOSE | → | whole STEP | NOSE | -1 | 0.0155 | False |
| drone | WINGTIP R 1 | → | STEP solid #1 | WINGTIP R | 1 | 0.0684 | False |
| drone | FUS 4 L V2 | → | STEP solid #1 | FUS4 MODIFIER | 1 | 0.0251 | False |
| drone | WING 1 L V2 | → | STEP solid #0 | WING 1L MODIFIER | 0 | 0.0466 | False |
| drone | FUS 1 R | → | STEP solid #1+5 | FUS 1 MODIFIER | 1+5 | 0.0226 | False |
| drone | MOTOR MOUNT D19mm | → | whole STEP | MOTOR MOUNT Dmm | -1 | 0.011 | False |
| vtol | AILERON 1R VTOL | → | STEP solid #0 | AILERON R VTOL | 0 | 0.0109 | False |
| vtol | WING ROOT VTOL R | → | whole STEP | WING ROOT VTOL R | -1 | 0.011 | False |
| vtol | AILERON 1L VTOL | → | STEP solid #1 | AILERON L VTOL | 1 | 0.0212 | False |
| vtol | MOTOR MOUNT TOP VTOL | → | whole STEP | MOTOR MOUNT TOP VTOL | -1 | 0.0093 | False |
| vtol | WING 2 R VTOL | → | whole STEP | WING 2 R VTOL | -1 | 0.0354 | False |
| vtol | WING 1 R VTOL | → | whole STEP | WING 1 R VTOL | -1 | 0.0824 | False |
| vtol | BOOM MOUNT L | → | whole STEP | BOOM MOUNT L | -1 | 0.039 | False |
| vtol | AILERON 2L VTOL | → | STEP solid #0 | AILERON L VTOL | 0 | 0.0144 | False |
| vtol | WING 1 L VTOL | → | STEP solid #0 | WING 1 L VTOL MODIFIER2 | 0 | 0.0327 | False |
| vtol | WING 2 L VTOL | → | whole STEP | WING 2 L VTOL | -1 | 0.0198 | False |
| vtol | AILERON 2R VTOL | → | STEP solid #1 | AILERON R VTOL | 1 | 0.0374 | False |
| vtol | BOOM MOUNT R | → | whole STEP | BOOM MOUNT R | -1 | 0.0453 | False |
| vtol | WING ROOT VTOL L | → | whole STEP | WING ROOT VTOL L | -1 | 0.0705 | False |
| vtol | MOTOR MOUNT BOTOM VTOL | → | whole STEP | MOTOR MOUNT BOTOM VTOL | -1 | 0.0063 | False |

## Unmatched (17)

| group | STL | class | segment_of | note |
|---|---|---|---|---|
| drone | AIL 1 R | segment | AIL R | print segment of one-piece STEP |
| drone | AIL 2 L | segment | AIL R | print segment of one-piece STEP |
| drone | FUS ROOT R V2 | no STEP | — | no STEP was sent |
| drone | BATTERY PAD | no STEP | — | no STEP was sent |
| drone | WING ROOT R V2 | no STEP | — | no STEP was sent |
| drone | LOCK 2 | no STEP | — | no STEP was sent |
| drone | FUS 2 R | no STEP | — | no STEP was sent |
| drone | FUS ROOT L V2 | no STEP | — | no STEP was sent |
| drone | FUS 3R V2 | no STEP | — | no STEP was sent |
| drone | LOCK 1 | no STEP | — | no STEP was sent |
| drone | FUS 2 L | no STEP | — | no STEP was sent |
| drone | AIL 2 R | segment | AIL R | print segment of one-piece STEP |
| drone | FRONT REINFORCEMENT | no STEP | — | no STEP was sent |
| drone | AIL 1 L | segment | AIL R | print segment of one-piece STEP |
| drone | WING ROOT L V2 | no STEP | — | no STEP was sent |
| drone | FUS 3L V2 | no STEP | — | no STEP was sent |
| drone | SERVO COVER | no STEP | — | no STEP was sent |

## Notes

- FRONT HATCH 1 match has pair p95 ≈ 0.22 mm (above typical 0.006–0.08 mm facet sag); kept because frac_within/vol/area gates passed in pairing.py.
- WING ROOT VTOL L/R STLs are ~1.6% lighter than their STEP (revision difference) but still matched.
- VTOL WING 2 L/R STL equals drone WING 2 L/R (~0.7%); treated as separate files in the inventory.
- Baseline engine runs live under `/workspace/meshsolid_testset/drone` (do not overwrite). New pair runs go to `drone_pairs_all`.

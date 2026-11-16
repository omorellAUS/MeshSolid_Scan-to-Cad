# Regressions + pair-test outcome (r15_pairsession)

Generated: 2026-11-09 10:49 AWST

## Matched drone STL↔STEP pairs

- `/workspace/meshsolid_testset/drone_pairs_all` — **35/35**, ok=0, failed=22, timeout=13, STEP solids=0
- Artifacts: `results/drone_pairs_all/pair_results.{csv,md,json}`
- Baseline `/workspace/meshsolid_testset/drone` untouched

## Fresh this session (`out/user/r15_pairsession/`)

| suite | result | evidence |
|---|---|---|
| M3 (10) | ok 10/10 | m3.log |
| coupling | ok | coupling.report.json |
| s04/s05/s06 | ok | *.report.json |
| p01 snapped | ok | p01_snapped.report.json |
| screw rotors | **ok 5/5** | screw2.log exit 0 (male_4, female_6, pump_twin, pump_triple, variable_pitch). First attempt OOM under pair load; isolated retry green. |
| units gear/addin/pack | passed | /workspace/scratch/reg_light1.log |
| units drone_fixes+extrude | 18 passed | /workspace/scratch/reg_light2.log |

## Prior only

| suite | note |
|---|---|
| impellers stg1/s07/s09 | not rebuilt; r14 / regressions_drone_fixes.json |

Parts: {'male_4': 'ok', 'female_6': 'ok', 'pump_twin': 'ok', 'pump_triple': 'ok', 'variable_pitch': 'ok'}

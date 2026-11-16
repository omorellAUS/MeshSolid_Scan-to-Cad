# Drone matched-pair engine table (checkpoint)
Generated: 2026-11-09 10:41 AWST
Run dir: `/workspace/meshsolid_testset/drone_pairs_all` (baseline `/workspace/meshsolid_testset/drone` untouched).
Expected matched pairs: **35**. Status counts: **{'failed': 22, 'timeout': 13}**.
Pending: 0. OK: 0. Steps written: 0.
Deviation scoring vs STEP truth runs next for ok solids only; failed/timeout have no rebuilt solid to compare.

| id | status | eng s | wall s | truth | how | mir | step | failure |
|---|---|---|---|---|---|---|---|---|
| drone__rear_hatch_2__userstl | failed | 251.1 | 252.6 | REAR HATCH #1+3 | STEP solid #1+3 | False | False | analytic solid construction failed (local_trim: 130 free (unsewn) edges - shell not closed |
| drone__motor_mount_d19x16mm__userstl | failed | 370.4 | 371.9 | MOTOR MOUNT D19x16mm #-1 | whole STEP | False | False | analytic solid construction failed (local_trim: 25 regions produced no trimmed face; 336 f |
| drone__rear_hatch_1__userstl | timeout | 0.0 | 391.0 | REAR HATCH #0+2 | STEP solid #0+2 | False | False | time budget of 390 s exceeded during stage 'build'; the engine was stopped. No STEP writte |
| drone__wingtip_l_1__userstl | failed | 301.0 | 302.6 | WINGTIP R #1 | mirror of STEP solid #1 | True | False | analytic solid construction failed (local_trim: 6 regions produced no trimmed face; 92 fre |
| drone__wing_2_r__userstl | failed | 188.8 | 191.0 | WING 2L MODIFIER #0 | mirror of STEP solid #0 | True | False | analytic solid construction failed (local_trim: 8 regions produced no trimmed face; 108 fr |
| drone__wingtip_l_2__userstl | failed | 116.5 | 117.8 | WINGTIP R #0 | mirror of STEP solid #0 | True | False | analytic solid construction failed (local_trim: 2 regions produced no trimmed face; 14 fre |
| drone__front_hatch_1__userstl | timeout | 0.0 | 608.9 | FRONT HATCH #0+3 | STEP solid #0+3 | False | False | time budget of 390 s exceeded during stage 'build'; the engine was stopped. No STEP writte |
| drone__nose_top__userstl | failed | 177.7 | 179.2 | NOSE TOP #-1 | whole STEP | False | False | analytic solid construction failed (local_trim: no trimmed faces; local_trim_retry1: no tr |
| drone__fus_1_l__userstl | timeout | 0.0 | 391.0 | FUS 1 MODIFIER #0+4 | STEP solid #0+4 | False | False | time budget of 390 s exceeded during stage 'freeform'; the engine was stopped. No STEP wri |
| drone__nose_clean__userstl | timeout | 0.0 | 394.9 | NOSE CLEAN 1 #-1 | whole STEP | False | False | time budget of 390 s exceeded during stage 'build'; the engine was stopped. No STEP writte |
| drone__wing_1_r_v2__userstl | failed | 330.0 | 331.5 | WING 1L MODIFIER #0 | mirror of STEP solid #0 | True | False | analytic solid construction failed (local_trim: 9 regions produced no trimmed face; 280 fr |
| drone__wingtip_r_2__userstl | timeout | 0.0 | 425.5 | WINGTIP R #0 | STEP solid #0 | False | False | time budget of 390 s exceeded during stage 'segmentation'; the engine was stopped. No STEP |
| drone__fus_4_r_v2__userstl | failed | 376.7 | 378.4 | FUS4 MODIFIER #0 | STEP solid #0 | False | False | analytic solid construction failed (local_trim: 3 regions produced no trimmed face; 9 free |
| drone__front_hatch_2__userstl | failed | 214.1 | 218.1 | FRONT HATCH #1+2 | mirror of STEP solid #1+2 | True | False | analytic solid construction failed (local_trim: 2 regions produced no trimmed face; 106 fr |
| drone__wing_2_l__userstl | failed | 123.2 | 124.8 | WING 2L MODIFIER #0 | STEP solid #0 | False | False | analytic solid construction failed (local_trim: 7 regions produced no trimmed face; 51 fre |
| drone__nose__userstl | timeout | 0.0 | 601.9 | NOSE #-1 | whole STEP | False | False | time budget of 390 s exceeded during stage 'build'; the engine was stopped. No STEP writte |
| drone__wingtip_r_1__userstl | timeout | 0.0 | 521.0 | WINGTIP R #1 | STEP solid #1 | False | False | time budget of 390 s exceeded during stage 'build'; the engine was stopped. No STEP writte |
| drone__fus_4_l_v2__userstl | timeout | 0.0 | 890.4 | FUS4 MODIFIER #1 | STEP solid #1 | False | False | time budget of 390 s exceeded during stage 'build'; the engine was stopped. No STEP writte |
| drone__wing_1_l_v2__userstl | failed | 343.7 | 346.4 | WING 1L MODIFIER #0 | STEP solid #0 | False | False | analytic solid construction failed (local_trim: 13 regions produced no trimmed face; 334 f |
| drone__fus_1_r__userstl | timeout | 0.0 | 543.9 | FUS 1 MODIFIER #1+5 | STEP solid #1+5 | False | False | time budget of 390 s exceeded during stage 'segmentation'; the engine was stopped. No STEP |
| drone__motor_mount_d19mm__userstl | timeout | 0.0 | 447.5 | MOTOR MOUNT Dmm #-1 | whole STEP | False | False | time budget of 390 s exceeded during stage 'build'; the engine was stopped. No STEP writte |
| vtol__aileron_1r_vtol__userstl | failed | 59.5 | 62.1 | AILERON R VTOL #0 | STEP solid #0 | False | False | analytic solid construction failed (local_trim: 40 free (unsewn) edges - shell not closed; |
| vtol__wing_root_vtol_r__userstl | failed | 153.1 | 155.2 | WING ROOT VTOL R #-1 | whole STEP | False | False | analytic solid construction failed (local_trim: 18 regions produced no trimmed face; 296 f |
| vtol__aileron_1l_vtol__userstl | timeout | 0.0 | 546.3 | AILERON L VTOL #1 | STEP solid #1 | False | False | time budget of 390 s exceeded during stage 'build'; the engine was stopped. No STEP writte |
| vtol__motor_mount_top_vtol__userstl | failed | 324.8 | 326.3 | MOTOR MOUNT TOP VTOL #-1 | whole STEP | False | False | analytic solid construction failed (local_trim: 1 regions produced no trimmed face; 12 fre |
| vtol__wing_2_r_vtol__userstl | failed | 148.2 | 151.0 | WING 2 R VTOL #-1 | whole STEP | False | False | analytic solid construction failed (local_trim: 2 regions produced no trimmed face; 53 fre |
| vtol__wing_1_r_vtol__userstl | failed | 157.1 | 158.6 | WING 1 R VTOL #-1 | whole STEP | False | False | analytic solid construction failed (local_trim: 8 regions produced no trimmed face; 70 fre |
| vtol__boom_mount_l__userstl | failed | 38.8 | 40.2 | BOOM MOUNT L #-1 | whole STEP | False | False | analytic solid construction failed (local_trim: 14 free (unsewn) edges - shell not closed; |
| vtol__aileron_2l_vtol__userstl | timeout | 0.0 | 585.0 | AILERON L VTOL #0 | STEP solid #0 | False | False | time budget of 390 s exceeded during stage 'build'; the engine was stopped. No STEP writte |
| vtol__wing_1_l_vtol__userstl | timeout | 0.0 | 562.5 | WING 1 L VTOL MODIFIER2 #0 | STEP solid #0 | False | False | time budget of 390 s exceeded during stage 'build'; the engine was stopped. No STEP writte |
| vtol__wing_2_l_vtol__userstl | failed | 129.9 | 133.2 | WING 2 L VTOL #-1 | whole STEP | False | False | analytic solid construction failed (local_trim: 4 regions produced no trimmed face; 43 fre |
| vtol__aileron_2r_vtol__userstl | failed | 356.2 | 359.5 | AILERON R VTOL #1 | STEP solid #1 | False | False | analytic solid construction failed (local_trim: 5 regions produced no trimmed face; 125 fr |
| vtol__boom_mount_r__userstl | failed | 58.7 | 60.0 | BOOM MOUNT R #-1 | whole STEP | False | False | analytic solid construction failed (local_trim: 20 free (unsewn) edges - shell not closed; |
| vtol__wing_root_vtol_l__userstl | failed | 148.3 | 149.7 | WING ROOT VTOL L #-1 | whole STEP | False | False | analytic solid construction failed (local_trim: 16 regions produced no trimmed face; 236 f |
| vtol__motor_mount_botom_vtol__userstl | failed | 367.7 | 369.3 | MOTOR MOUNT BOTOM VTOL #-1 | whole STEP | False | False | analytic solid construction failed (local_trim: 6 regions produced no trimmed face; 132 fr |

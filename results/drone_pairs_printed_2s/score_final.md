# Printed matched-pair final table

When: 2026-11-09 20:20:05 AWST (Australia/Perth)
Accuracy: `--accuracy printed` (0.05/0.10). Tip2 TE densify bump kept green.

| STL | status | method | loft_acc | mesh rms | vs-STEP Hd | t2r rms | align med | failure |
|---|---|---|---|---|---|---|---|---|
| WING 1 L V2 | failed | revolve_profile | None | None | None | None | None | analytic solid construction failed (local_trim: local t |
| WING 1 R V2 | failed | revolve_profile | None | None | None | None | None | analytic solid construction failed (local_trim: local t |
| WING 2 L | failed | revolve_profile | False | None | None | None | None | analytic solid construction failed (local_trim: local t |
| WING 2 R | timeout | None | None | None | None | None | None | time budget of 150 s exceeded during stage 'loft_skin'; |
| WINGTIP L 1 | failed | revolve_profile | False | None | None | None | None | analytic solid construction failed (local_trim: 5 regio |
| WINGTIP L 2 | ok | section_loft | True | 0.04193 | 0.30472 | 0.01089 | 0.25984 |  |
| WINGTIP R 1 | failed | revolve_profile | False | None | None | None | None | analytic solid construction failed (local_trim: 3 regio |
| WINGTIP R 2 | ok | section_loft | True | 0.04 | 0.25219 | 0.01217 | 0.26041 |  |
| AILERON 1L VTOL | failed | cell_decomposition | False | None | None | None | None | analytic solid construction failed (local_trim: local t |
| AILERON 1R VTOL | failed | revolve_profile | False | None | None | None | None | analytic solid construction failed (local_trim: 40 free |
| AILERON 2L VTOL | failed | cell_decomposition | False | None | None | None | None | analytic solid construction failed (local_trim: local t |
| AILERON 2R VTOL | failed | cell_decomposition | False | None | None | None | None | analytic solid construction failed (local_trim: local t |
| WING 1 L VTOL | timeout | None | None | None | None | None | None | time budget of 150 s exceeded during stage 'loft_skin'; |
| WING 1 R VTOL | failed | revolve_profile | False | None | None | None | None | analytic solid construction failed (local_trim: 8 regio |
| WING 2 L VTOL | failed | revolve_profile | False | None | None | None | None | analytic solid construction failed (local_trim: 4 regio |
| WING 2 R VTOL | timeout | None | None | None | None | None | None | time budget of 150 s exceeded during stage 'loft_skin'; |
| WING ROOT VTOL L | failed | revolve_profile | None | None | None | None | None | analytic solid construction failed (local_trim: 16 regi |
| WING ROOT VTOL R | failed | revolve_profile | None | None | None | None | None | analytic solid construction failed (local_trim: 18 regi |

## Continuations
- **TE zone (tip2):** L2 TE rms 0.03879 (was ~0.076); R2 TE rms 0.05586 (was ~0.086); both still printed-ok.
- **WING 2 L pockets:** after_rms 0.5282, skin_rms 0.3309 (target ~0.188), feat_rms 0.8093. Still open.
- **Tip1 loft parity:** rms 0.32272 after 1 pocket; mid_skin max~3.3 mm at t~0.16 (not tip2-class yet).


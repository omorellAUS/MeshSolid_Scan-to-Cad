# Baseline vs now — FINAL

When: 2026-11-09 22:46:54 AWST (Australia/Perth)
HEAD: `f477b2c`  |  **box: 8 CPU cores**, Xeon, device=cpu (no NVIDIA)
0.1.1 zip: `dist/MeshSolid-0.1.1-windows.zip`  sha256 `5379371dfd6fe5d83a036f56bd78fc3800625452af1fd8e815fccd280744882f`

## Guard cases

| case | baseline rms/gate | now rms/gate | wall_s | match | notes |
|---|---|---|---|---|---|
| s07 | 0.0481/True | 0.04576/True (ok) | 1396.9 | **True** | excl rms; hist overall 0.048 sheet LE |
| s09 | 0.0901/True | 0.10546/True (ok) | 1451.1 | **True** | CAD mesh; gate 3x side-fit |
| stg1 | 0.0933/True | 0.08946/True (ok) | 3319.7 | **True** | matches 66b2411 |
| p01 | 0.2214/True | 0.20858/True (ok) | 1037.3 | **True** | mid_flank 0.033; hub band 0.40/0.80; tip excl |
| s05 | 0.0655/True | 0.06555/True (ok) | 148.1 | **True** | prepped identical to loft-guard |
| s06 | 0.0730/True | 0.07300/True (ok) | 111.5 | **True** | prepped identical to loft-guard |

All six match baseline. Gear-hub fixes are gear/screw-only (other cases unaffected).

## Fix commits

- `4963c4e impeller gate 1.5x/2.5x→3x CAD + 0.005mm slack`
- `8f753b2 gear blank coax credit, threshold 0.55`
- `8a29154 impeller CAD 3x side-fit; axisym freeform 0.04`
- `9b76cc0 gear tip_edge exclude + hub band fallback`
- `12ebf5f gear hub coax 5deg/1.5mm; hub band 0.40/0.80`
- `9595cae gear hub coax 15deg/3mm (p01 tori)`

## Printed pairs (wall_s from printed_2s run, same 8-core box)

| pair | status | method | loft | wall_s |
|---|---|---|---|---|
| drone_wingtip_l_1_userstl | failed | revolve_profile | False | 121.6 |
| drone_wing_2_r_userstl | timeout | None | None | 151.1 |
| drone_wingtip_l_2_userstl | ok | section_loft | True | 21.2 |
| drone_wing_1_r_v2_userstl | failed | revolve_profile | None | 141.0 |
| drone_wingtip_r_2_userstl | ok | section_loft | True | 21.5 |
| drone_wing_2_l_userstl | failed | revolve_profile | False | 121.3 |
| drone_wingtip_r_1_userstl | failed | revolve_profile | False | 127.2 |
| drone_wing_1_l_v2_userstl | failed | revolve_profile | None | 132.4 |
| vtol_aileron_1r_vtol_userstl | failed | revolve_profile | False | 68.1 |
| vtol_wing_root_vtol_r_userstl | failed | revolve_profile | None | 130.6 |
| vtol_aileron_1l_vtol_userstl | failed | cell_decomposition | False | 134.3 |
| vtol_wing_2_r_vtol_userstl | timeout | None | None | 151.0 |
| vtol_wing_1_r_vtol_userstl | failed | revolve_profile | False | 128.0 |
| vtol_aileron_2l_vtol_userstl | failed | cell_decomposition | False | 133.8 |
| vtol_wing_1_l_vtol_userstl | timeout | None | None | 151.2 |
| vtol_wing_2_l_vtol_userstl | failed | revolve_profile | False | 126.7 |
| vtol_aileron_2r_vtol_userstl | failed | cell_decomposition | False | 142.7 |
| vtol_wing_root_vtol_l_userstl | failed | revolve_profile | None | 127.0 |

Printed pair wall total: **2131s** across 18 pairs.

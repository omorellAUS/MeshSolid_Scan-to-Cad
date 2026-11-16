# Fresh loft-guard regression table

When: 2026-11-09 20:20:05 AWST (Australia/Perth)
Branch: `wip-local-trim` @ 5a1b417. Loft early-path must not divert these.

| case | status | method | loft? | rms | gate | notes |
|---|---|---|---|---|---|---|
| coupling_pytest | ok | — | — | — | — | 1 passed (-m slow -o addopts=); local mechanical STL restored |
| coupling | ok | local_trim+maker_volume | False | 0.00048 | True |  |
| smoke_coupling | ok | local_trim+maker_volume | False | 0.00048 | True |  |
| smoke_tip2L | ok | section_loft | True | 0.04229 | True |  |
| s04 | failed | cell_decomposition | False | None | None | analytic solid construction failed (local_trim: no trimmed f |
| s04_prepped | ok | revolve_profile | False | 0.12785 | True |  |
| s05 | failed | revolve_profile | False | 6.46751 | False | solid is valid but deviates from the mesh beyond acceptance  |
| s05_prepped | ok | revolve_profile | False | 0.12228 | True |  |
| s06 | failed | revolve_profile | False | 2.72144 | False | solid is valid but deviates from the mesh beyond acceptance  |
| s06_prepped | ok | revolve_profile | False | 0.1226 | True |  |
| p01 | failed | revolve_profile | False | None | None | analytic solid construction failed (revolve_profile: not a t |
| s07 | failed | impeller | False | 1.15814 | False | impeller solid is valid but deviates beyond the acceptance l |
| s09 | failed | impeller | False | 0.10546 | False | impeller solid is valid but deviates beyond the acceptance l |
| stg1 | incomplete | None | False | None | None | process ended during stage 'impeller' (crash or kill) |

## M3 (all loft=False)
| case | status | method | loft? | rms | gate |
|---|---|---|---|---|---|
| bolt_circle_flange.report | ok | local_trim+maker_volume | False | 0.01961 | True |
| chamfer_fillet_block.report | ok | local_trim+maker_volume | False | 0.01992 | True |
| chamfered_block.report | ok | local_trim+maker_volume | False | 0.01982 | True |
| counterbore_plate.report | ok | local_trim+maker_volume | False | 0.01968 | True |
| countersunk_plate.report | ok | local_trim+maker_volume | False | 0.01985 | True |
| drafted_block.report | ok | local_trim+maker_volume | False | 0.01988 | True |
| pipe_elbow.report | ok | local_trim+maker_volume | False | 0.02092 | True |
| pocket_plate.report | ok | local_trim+maker_volume | False | 0.01972 | True |
| slotted_plate.report | ok | local_trim+maker_volume | False | 0.01976 | True |
| threaded_stud.report | ok | local_trim+maker_volume | False | 0.5758 | True |

**Any routed_to_loft on guard set (excl. tip2 smoke)?** `False`  
(tip2 smoke correctly uses section_loft under `--accuracy printed`.)

Impellers (s07/s09): build via `impeller` (vol matches historical); machined gate rejects without impeller zone exclusions — not a loft diversion. stg1 still running (scaled 0.1 mesh + bore_d=14.29).
Tip2 smoke: section_loft ok, device auto→cpu (no NVIDIA on box). Coupling: local_trim, loft=None.

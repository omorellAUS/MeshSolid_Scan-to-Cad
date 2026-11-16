# Pinion X2.B1872 — where the 2.6 % / 92 % error came from (2026-11-16)

Scan: same SHA (129f2347a5eb) as job 20261114-060647. Axis frame of the engine build; h = height above the
low-z end (z = -147.11). Box replay of `build_revolve` on the job's dumped regions (`/tmp/pin`, not committed),
measured with Metrology, spec-v2 settings, alignment off (same frame), 33-fold. |d| > 0.3 mm binned by h.

| cluster | h (mm) | solid faces (deviation.csv) | cause | fix |
|---|---|---|---|---|
| A | 195-225 | 3, 5, 6, 7, 8 (neck), 10, 11, 12 (groove) | freeform region 0 spans the r 11.05 journal **and** the r 9.21 neck behind a filleted step (6.8 % off one line): the robust refit kept r 11.05 and its segment ends came from all points, so the neck z 56-74 got +1.85 mm of material; the circlip groove r 10.0-10.9 at z 48-50 was 1,940 unassigned faces bridged by the straight r 11.07 line (up to 1 mm extra) | segment ends from inliers only; outlier bands that are lines of revolution become own segments (`_axisym_bands`); full-ring unassigned faces fitted as arcs (`_unassigned_revolution_segments`) |
| B | 50-55 | 427-430 | the scan is **two watertight shells** (lower shaft 25k faces, gear+upper shaft 179k) never merged: the shaft shell's cut face at z -95.5 and the gear shell's face at z -93 look at each other across a 2-2.5 mm gap from r 0 to r ~15, open to the outside at some angles. Region 3 (gear face plane) carried those points to r 0.8; the profile synthesis closed it with a cone back to the shaft: **a neck to r 0.8 in the solid** | faces of two shells facing each other across <= 4 mm (ray along the normal hits the OTHER shell, normals opposed) are left out of the profile (`_joint_caps`, multi-shell scans only). The solid is now a sound shaft-to-gear-face joint |

Cluster B cannot be fixed in the solid: 1,058 scan vertices (1.04 %) lie on internal faces no real surface
has, up to 10.9 mm inside any sound part (this is the 10.6 mm "max" seen in the 1.0.1 report). Fix in the
scan (merge the two captures in Fusion / close the gap) or a spec-v3 exclusion of capture-joint faces.

| solid (box replay) | rms unc | rms cap | p95 | p99 | max | pass_v1 % |
|---|---|---|---|---|---|---|
| before (1.0.1 revolve) | 0.358 | 0.0524 | 0.451 | 1.790 | 7.44 | 66.19 |
| A fixed | 0.176 | 0.0505 | 0.129 | 0.663 | 3.15 | 70.71 |
| A + B fixed | 0.353 | 0.0504 | 0.124 | 0.702 | 10.89 | 71.09 |
| A + B, joint vertices excluded (info only, not spec) | 0.093 | — | 0.115 | 0.272 | 1.58 | 71.76 |

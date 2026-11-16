# Pinion X2.B1872: controlled cap-off test (2026-11-16)

Scan: engine job 20261114-060647 `mesh_body.stl` (sha 129f2347..., identical to the Metrology jobs' scan.stl).
Engine STEP: 20261114-060647 `mesh_body.step`. Fusion body: Metrology job 20261114-062754 `solid.step` (the run that gave 0.05616 / 68.8 %).
Engine gate (from its report): rms 0.3908 / p95 0.2878 (gate exclusions), 0.4269 all vertices.
Engine exclusions rebuilt on Metrology's vertex order: 35794 tip-edge faces (engine: 35,789) + 434 capture-cap faces (engine: 430); 43 hole-fill faces not rebuilt; 20815 vertices excluded (20.4 %).

Pre-registered reading: near 0.39 = the cap explains the gap; in between = partly; near 0.056 = something else.

| config | solid | ICP | exclusions | cap | rms_uncapped | rms_capped | p95 | p99 | max + / - | pass_v1 % | pass_legacy % | capped | rot deg |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R0 metrology 1.0.1 as run (Fusion body, ICP, tags, cap 0.2) | 20261114-062754 | on | tags backface+bore+tip (10.2 %) | 0.2 | 0.3226 | 0.0562 | 0.1574 | 0.9079 | +7.92 / -8.27 | 66.25 | 68.78 | 3360 | 108.99 |
| R1 = R0 with the cap OFF | 20261114-062754 | on | tags backface+bore+tip (10.2 %) | off | 0.3226 | 0.0562 | 0.1574 | 0.9079 | +7.92 / -8.27 | 66.25 | 68.78 | 3360 | 108.99 |
| R2 = R1 with engine exclusions instead of tags | 20261114-062754 | on | engine mask (20.4 %) | off | 0.2970 | 0.0528 | 0.1659 | 0.8781 | +9.92 / -10.61 | 71.22 | 74.26 | 3314 | 108.99 |
| R3 = R2 without ICP (scan frame) | 20261114-062754 | off | engine mask (20.4 %) | off | 0.2950 | 0.0509 | 0.1534 | 0.8782 | +10.59 / -9.26 | 72.80 | 75.57 | 2975 | 0.00 |
| R4 CONTROLLED: engine STEP, no ICP, engine exclusions, cap off | 20261114-060647 | off | engine mask (20.4 %) | off | 0.3903 | 0.0476 | 0.2882 | 1.8463 | +10.59 / -9.17 | 72.81 | 77.56 | 4961 | 0.00 |
| R5 = R4 with no exclusions (engine all-vertex 0.4269) | 20261114-060647 | off | none (0.0 %) | off | 0.4264 | 0.0507 | 0.2231 | 1.8451 | +10.86 / -9.25 | 69.37 | 73.29 | 5449 | 0.00 |
| R6 = R4 at the engine's tessellation (lin 0.002, ang 0.05) | 20261114-060647 | off | engine mask (20.4 %) | off | 0.3909 | 0.0475 | 0.2885 | 1.8513 | +10.59 / -9.90 | 72.96 | 77.73 | 4976 | 0.00 |
| R7 engine STEP with metrology 1.0.1 settings (ICP, tags, cap 0.2) | 20261114-060647 | on | tags backface+bore+tip (10.1 %) | 0.2 | 0.4193 | 0.0527 | 0.2683 | 1.8525 | +8.48 / -8.39 | 66.91 | 71.14 | 5440 | 130.94 |
| R8 = R4 but with ICP | 20261114-060647 | on | engine mask (20.4 %) | off | 0.3912 | 0.0491 | 0.3044 | 1.8498 | +10.59 / -8.71 | 71.56 | 76.58 | 5319 | 130.94 |

See the report back for the reading. Where the big errors sit (R4, engine STEP, no ICP): 2,142 inspected vertices with |d| > 1 mm (2.6 %) carry 92 % of the sum of d^2; 1,664 of them lie 200-220 mm from the low-z end and 447 at 50-60 mm, radius 1.6-16 mm, mean -1.39 mm (solid has material where the scan has none). Without the |d| > 1 mm points rms is 0.113 mm. The Fusion body (R0-R3) has none of the 200-220 mm cluster.

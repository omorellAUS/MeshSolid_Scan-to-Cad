# Real scan test data (not committed; download from the source)

| File | Source | License | Scanner |
|---|---|---|---|
| `Pipe_band.stl/Pipe_band.stl` (from `pipe-bend-stl.zip`) | "Pipe bend" by Artec 3D, https://www.artec3d.com/3d-models/pipe-bend (zip: https://cdn.artec3d.com/content-hub-3dmodels/pipe-band-stl.zip) | Bundled license.txt: CC BY 3.0 Unported; web page states Creative Commons v4. Attribution: "Pipe bend by Artec 3D", link to artec3d.com | Artec Micro |

## Runs
- 2026-10-24, Pipe bend: decimated 501,142 → 50,000 triangles (pymeshlab quadric edge collapse, topology/normals preserved;
  decimated-vs-original surface distance rms 0.011, p95 0.021, max 0.091 mm), file `out/real/pipe_bend_dec50k.stl`.
  Run: `MESHSOLID_SETBACK=0 timeout 600 python -m meshsolid out/real/pipe_bend_dec50k.stl ...` (390 s).
  Result: FAILED, no STEP. tol 0.145 (noise sigma 0.032 mm). The part is a threaded fitting (helical threads on both ends +
  a short offset bend): 175 regions (80 tori, 51 cylinders, 18 spheres, 18 cones, 8 planes), 7.6% area unassigned;
  local trim: 28 regions without a trimmed face, 1084 free edges, 33 shells; MakerVolume found no volume.
  Report + preview: `results/real/pipe_bend.report.json`, `results/real/pipe_bend.preview.png`.
  The full-resolution run (501k triangles) was stopped after >40 min in the setback fallback (segmentation alone ~15 min,
  245 regions, 40.8% area unassigned).
- 2026-10-25, Pipe bend with thread recognition (commit 23463cf; M4 freeform stage also active):
  same 50k mesh, `MESHSOLID_SETBACK=0`, 270 s. Threads found: 2 external right-hand —
  G1/2 BSPP (match good; pitch 1.823, measured major 20.54 vs nominal 20.955, zone 8.5 mm, confidence 0.03) and
  G3/4 BSPP (match approximate; pitch 1.851 vs 1.814, measured major 25.85 vs 26.441, zone 10.2 mm, confidence 0.30).
  Measured thread depth ~0.75 mm vs 1.16 mm nominal (scan rounds the crests/roots).
  Regions 175 -> 85 after threads -> 83 after freeform (1 of 6 smooth clusters accepted as a B-spline);
  unassigned area 7.6% -> 4.0%. Build still FAILED, no STEP: 16 regions without a trimmed face (was 28),
  378 free edges (was 1084), 20 shells (was 33). Remaining failures are in the offset bend between the threads.
  Report + preview: `results/real/pipe_bend_threads.report.json`, `results/real/pipe_bend_threads.preview.png`.
- 2026-10-25, Pipe bend with M4 freeform (commit 6cda53e): 286 s. Threads as above. Freeform: 2 of 6 smooth clusters
  accepted as B-spline tubes (bore: rms 0.034 / p95 0.069 mm; a small cluster: rms 0.055 / p95 0.114 mm); 4 rejected
  (rms 0.09-0.21 mm against the 0.5 x tol = 0.072 mm acceptance). Regions 175 -> 85 -> 78. Build still FAILED, no STEP:
  14 regions without a trimmed face, 382 free edges, 15 shells. Some tori in the bend have degenerate parameters
  ("gp_Circ radius should be positive" when bounding them).
  Report + preview: `results/real/pipe_bend_m4.report.json`, `results/real/pipe_bend_m4.preview.png`.

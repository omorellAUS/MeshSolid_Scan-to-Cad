## Inventory and pairing (all 3 batches)

- Inputs: 27 STEP (13 drone + 14 VTOL), 52 STL (38 drone + 14 VTOL). All STEPs are AP214 mm, BRepCheck valid; 14 of them are multi-solid
  (L/R halves, or the wing plus its slicer "MODIFIER" box). STL headers carry the part names.
- Duplicates: 0 byte-identical files and 0 geometrically identical meshes. What looked like duplicates are **17 L/R mirror twins**
  (fingerprint within 0.5 %), e.g. FUS 1 L/R, WING 1 L/R V2, WINGTIP L/R 1, BOOM MOUNT L/R. Only one side of each was run; the other is listed as `skipped`.
- STL->STEP pairing (`pairing.py`): pose-invariant fingerprint, then rigid ICP (also tried mirrored), then a point-to-surface check.
  This gave **35 of 52 STLs matched**: to a whole STEP, one STEP solid, two glued STEP solids (each hatch STL = the L+R halves of the STEP), or the mirror of the STEP
  (the drone set has only right-hand STEPs for wingtips and wings).
  Matched p95 is 0.006-0.08 mm, which is the STL facet sag. WING ROOT VTOL L/R STLs are 1.6 % lighter than their STEP (an older or newer revision).
- Unmatched STLs (17, scored against the input only): AIL 1/2 L/R are print segments of the one-piece STEP `AIL R` (their volumes sum to the STEP within 0.04 %).
  No STEP was sent for FUS 2, FUS 3 V2, FUS ROOT V2, WING ROOT V2, LOCK 1/2, BATTERY PAD, SERVO COVER, FRONT REINFORCEMENT.
- STEPs without an STL: AIL R (one-piece), WING 1 L VTOL (its STL matched the MODIFIER2 wing solid), WING 2 L VTOL MODIFIER2. These were run from the CAD-fine tessellation.
- VTOL vs drone: WING 2 L/R VTOL STL equals drone WING 2 L/R (0.7 %), and WING 2 L VTOL MODIFIER2 equals drone WING 2L MODIFIER (0.6 %).
  Everything else is a real variant: VTOL ailerons are about 10 % smaller, VTOL wing 1 is about 17 % lighter, and the VTOL wing root is a different part (82 vs 6.6 cm3).
  BOOM MOUNT and MOTOR MOUNT TOP/BOTTOM are VTOL-only.

## Failure modes (one representative per unique part, user STL first)

The user STLs are **coarse CAD tessellations** (550-51k triangles; one airfoil wing is 1.8k triangles), not scans.
The engine reads them as "exact" meshes (noise 0, tol 0.003-0.018 mm), which drives most of what follows.

1. **Build-stage timeouts (most parts, 600 s budget)**: large freeform skins (wings, hatches, fuselage, nose, wingtips, ailerons) and even the 55 mm motor mounts.
   Every flat facet strip of a faceted airfoil or fuselage becomes its own plane region (57-430 regions).
   The local-trim builder then spends its time in `_setback_fill` (`build_local.py`): one BRepFill_Filling patch per corner,
   plus dense face sampling and nearest-face queries per patch. py-spy showed every timed-out run there.
   Segmentation and coaxial refit also cost 60-390 s on the 40-50k-triangle nose and fuselage parts.
   SIGTERM is not handled while OCC holds the GIL, so the runner's SIGKILL ends these runs (rc -9 / 124).
2. **Faceted freeform fitted as huge analytic primitives**: AIL 2 L/R give "diameters" of 500-4700 mm and torus radii of 100-250 mm on airfoil facets.
   The fallback cell decomposition then hits its 240 s budget (682 cells).
3. **False gear detection**: AIL 2 L/R (airfoil segments) are reported as a 6-tooth helical gear (`gear.detected=true`).
4. **Open shells after trim/sew**: FRONT HATCH 1 (301 free edges, 13 shells) and FRONT REINFORCEMENT (a 1.6 mm thin ring/plate, 14 shells).
   These are thin-wall multi-shell print STLs (FRONT HATCH STL has 20-98 disconnected bodies).
5. **Engine exceptions (bugs)**: `freeform.apply_freeform` raises `ValueError: zero-size array` (np.ptp on an empty region) on FRONT HATCH 2 and NOSE CLEAN.
   `build_local.py:1021` raises an uncaught `Standard_NullObject` on MOTOR MOUNT TOP VTOL.
6. **Invalid solid / wrong volume**: BOOM MOUNT L (local_trim volume 225,944 vs mesh 23,444, 10x, invalid). LOCK 2 is invalid with non-positive volume.
7. **Deviation gate**: SERVO COVER (1.2 mm thin plate) builds a valid solid that misses surfaces, so no STEP is written.
8. Successes: LOCK 1 (prismatic, 23 faces) builds a valid single solid in 12 s. Volume matches the STL to 0.03 %, input->solid rms 0.0004 mm, and its dims snap (6.000, 3.5, 8.0, 13.5 ...).
9. Scan-like inputs: the first attempt at 0.4 mm edge (400k-800k triangles) timed out in **segmentation** at 600 s on every large skin.
   The cause is per-face Python region growing on smooth skins with no sharp edges.
   Scans were regenerated with the triangle count capped at 150k (edge 0.4-1.0 mm), noise sigma 0.025 mm. Those runs follow the representative pass.

## Top fixes (roadmap, estimated impact on this set)

1. **Faceted-CAD STL mode**: when the mesh is exact but coarse (long facets, constant dihedral steps), set tol from the facet sag or chord height, not 5e-5*diag.
   Fit smooth strips as B-spline or cylinder before plane growing.
   This collapses hundreds of facet planes into a few surfaces, and it unblocks every wing, aileron, hatch, nose and fuselage (~20 of 26 unique parts).
2. **Freeform-first routing for skins**: if the planar-region count per area or the dihedral histogram says "faceted smooth", send the whole smooth component to `freeform` (B-spline) and skip primitive growing.
   Same parts as fix 1; it also removes the 400k+ triangle segmentation timeouts on scans.
3. **Bound `_setback_fill`**: cap the patches, skip corners whose hole is smaller than tol, batch the nearest-face queries, and check a time budget so the run falls back instead of dying.
   This turns the timeouts into results (pass or fail with a reason) and makes the motor mounts and boom mount testable.
4. **Fix the three crashes**: the empty-region guard in `freeform.apply_freeform`, and catching OCC `Standard_NullObject` in `build_local` (line ~1021). That is 3 parts.
5. **Gear-detector guard**: require a closed tooth ring (full 360 degree coverage) and a plausible module and face width. This removes the aileron false positives.
6. **Thin-wall / multi-body print STLs**: split disconnected bodies and build each one, or report per body (hatches have 20-98 shells; reinforcement and servo cover are 1.2-1.6 mm plates).
7. **Timeout handling**: install a watchdog thread or run OCC steps in `occ_child`, so a timeout still writes a "timeout" report with the stage (today SIGTERM is not honoured inside OCC).

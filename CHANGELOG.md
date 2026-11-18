# MeshSolid 1.0.4

Traceability and baseline: every report tagged with spec version/hash/git_commit/run_id; one scoreboard CSV written per run; baseline re-measure of STG1, Stage 3 and pinion under spec v2; add-in fixes for decimal-comma parsing and display-mesh fallback warning.

## What's new (1.0.4)

- **Report tagging.** Every `report.json` now carries `spec_version`, `spec_hash` (from the frozen SPEC dict), `git_commit` (via `git rev-parse HEAD`), and `run_id` (ISO timestamp + short git commit). These are written at the start of every engine run so the full provenance trail is baked into each job's report.
- **Scoreboard CSV.** After each run, one row is appended to the active scoreboard (`results/harness.csv` for spec v1, `results/harness_v2.csv` for spec v2). The row contains the fields from the existing harness files plus the new provenance columns. The file is created with a header row on first write.
- **Baseline re-measure.** A new helper (`scripts/baseline_metrology.py`) re-runs Metrology on STG1, Stage 3 and the pinion under spec v2, producing a reference baseline CSV. Future releases must not score worse than these numbers on the scoreboard.
- **Add-in fixes.** Decimal-comma parsing in the add-in parameter UI; a warning when the display-mesh fallback is used (visible in the conversion report and PDF).

# MeshSolid 1.0.3

Reliability release: per-stage time limits with heartbeat, input normalisation for oversized meshes, and job-folder pruning. Every run ends with a readable reason (crashed / cancelled / watchdog-killed / timed out / gate-rejected / ok) recorded in `report.json["end_reason"]`.

## What's new (1.0.3)

- **Per-stage time limits and heartbeat.** Each named stage (segmentation, build, deviation, freeform, ...) has its own time cap, scaled from the run budget. A heartbeat file next to the report names the current stage and its last touch time, so a stuck run is detectable. The checkpoint report written at every stage still says how far the run got if the process dies.
- **Separate end reasons.** `report.json["end_reason"]` is one of: `ok`, `crashed`, `cancelled`, `watchdog_killed`, `timed_out`, `gate_rejected`. A cancelled run (SIGTERM / user cancel) writes a partial report and exits 124 without claiming success.
- **Input normalisation.** Meshes over ~300k faces are reduced with quadric edge-collapse decimation for recognition; the full mesh is kept for deviation. Scale and units are checked: a part ~1,650 mm across is flagged as potentially mm-or-inches and must be confirmed before build.
- **Job-folder pruning.** On success, intermediate files (preview PNG, deviation PLY, heartbeat) are deleted; only STEP, reports and logs are kept.

# MeshSolid 1.0.2a

Bug-fix release on top of 1.0.2, from the user's 1.0.2 test runs (X2-B1941 STG1, KK&K stage 1, LS pinion).

## What's new (fixes)

- **Face classifier no longer excludes blade passages and the hub (STG1 53.8 % excluded).** 1.0.2 read the hub flow surface between the blades, which sits behind a thin back plate, as end-face features. That gave 15 "balance cuts" 24° apart (the 15 blade passages at the rim), an "unclassified ring" over 37.5 % of the scan (the hub) and two "wide depressions" on the nose. Fixes:
  - Only the true end face is classified: points in the thin axial slab at the end plane whose normals point **out** of the part through that face. Surfaces that face into the part (hub, passages, blade roots) are ignored, and the 2 mm exclusion border no longer reaches through the plate.
  - A family of patches that repeats N-fold about the axis (N = the recognised blade count, or any regular N ≥ 3) is periodic and is **never** called a balance cut. The blade count is passed from the impeller build.
  - A large uncertain region (a near-full ring, wider than 90°, or more than 5 % of the scan) is **flagged and kept inspected**, not auto-excluded.
  - Metrology re-checks conversion reports written by 1.0.2 with the same rules, so re-running Metrology on an old STG1 conversion no longer greys out the blades. It also caps end-face classifier exclusions at 5 % per region and 5 % in total; anything over the cap is flagged and kept inspected.
  - STG1 re-check on the user's data: classifier exclusion fell from 53.8 % to 0.2 % (new conversion) or 0.5 % (old 1.0.2 conversion report).
- **Stage 3 regression fixed (13 blades: rms 0.356 mm, 41.8 % excluded, no STEP on 1.0.2).** The impeller build itself was fine; 1.0.2's classifier recognised the 13 blade passages at the back face as a 13-tooth Hirth coupling and **cut it into the solid**, and excluded 13 passage "balance cuts" and a ring. An N-fold signal whose N equals the blade count (or the full-blade count of a splitter wheel) is now blading: never Hirth, never modelled, never cut. `apply_modelled` refuses such a cut from an old report too. A real coupling with another count (KK&K: 24 teeth on 16 blades) is unchanged. Box rebuild of the Stage 3 scan: 1.0.2 gave a Hirth-cut solid, gate rms 0.155 mm, 43.2 % excluded, FAIL, no STEP. 1.0.2a gives status ok, STEP written, gate rms 0.046 / p95 0.081 mm, classifier exclusion 2.6 %.
- **Prominent WARNING banner.** Total exclusion above 5 %, flagged classifier regions and alignment pose warnings are shown as a red WARNING banner at the top of the PDF, in the HTML results view and in the Fusion Metrology result message. The Metrology CLI prints them as `WARNING:` lines.
- **PDF statistics table** now always lists the spec version, rms uncapped, rms capped and the capped count / % (and the final alignment pose).
- **Alignment bounds (spec v2).** The half-pitch rotation clamp is reapplied after every `refine_on_deviation` step, so the refinement can no longer undo it and jump to a neighbouring blade. The blade / tooth count from the conversion report (or estimate) is used. The final pose (rotation about the axis, axis tilt, translation) is written to `report.json` (`stats.pose_summary`, `alignment.final_pose`) and the PDF, with a WARNING when the rotation hits the bound, the tilt is above 0.5° or the translation above 1 mm.
- **PC stays awake during a job.** On Windows, conversion and Metrology jobs call `SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)` and release it when the job ends, also on error. Only system sleep is blocked, so the display can still turn off. Note: the time budget is wall-clock, so before this fix any time the PC spent asleep counted against it.
- **Recogniser fall-through is explained, and slow stages are capped.** Every recogniser that falls through now writes its reason to the engine log (`[recogniser] impeller: none - ...`), not only to the report. The general method's `coaxial_refit` step is capped at 10 % of the time budget (at most 600 s) and `freeform` at 20 % (at most 1200 s); each stops cleanly with a logged reason. The KK&K run spent 37 min in coaxial_refit and timed out at 7560 s.
- **Part type Impeller no longer falls through.** The Impeller part type now sends `--impeller on`. If the impeller recogniser cannot build the part, the run fails at once with the recogniser's reason instead of spending hours in the general method. Auto still falls through as before.

## Known open issues

- **Input-class misdetection** (a rough scan labelled `cad_tessellation`, as on KK&K) is only logged, not fixed.
- **KK&K stage 1** (holdout, untuned) still needs the engine log of job 20261116-232354 to explain why the impeller recogniser rejected it.
- Tiny fragments next to the blade passages (each under 0.1 % of the scan) can still be listed as balance cuts when an old 1.0.2 conversion report is re-measured; rebuild the conversion to clear them.
- All 1.0.2 open issues below still apply.

# MeshSolid 1.0.2

What's new since 1.0.1, and the known open issues. This is the engine and Fusion add-in package built from the `measure-v1` branch.

## What's new

- **Spec v1 / v2 Metrology and the honest verdict.** One-sided signed distance from scan to solid, 0.2 mm cap, `pass_pct_v1` counting capped points as failures (legacy counting is still logged). Spec v2 alignment is bounded point-to-plane ICP on at least 500k points, with a final rigid refinement and a warning above 0.01° or 0.01 mm. The target verdict is rms ≤ 0.03 mm, at least 90 % of points within ±0.05 mm (capped points count as failures), and no more than 5 % excluded. PASS is refused above 5 % excluded. Both pass percentages are labelled with the spec version.

- **Normalise.** The `--normalise report|apply` stage stores a 4×4 scan-to-normalised transform in the conversion report.

- **Recogniser fall-through.** A recogniser that fails no longer crashes the run; control falls through to the next one.

- **Splitter recognition.** Impeller splitters are detected from ring votes plus a per-sector leading-edge family test, and the build cuts them at the splitter leading edge. Cummins-style splitter vanes are the intended case.

- **Input-class detector.** Scan, CAD tessellation and decimated / low-resolution meshes are classified, with gates loosened for rough meshes (Cummins is treated as a low-resolution class). `--accuracy` defaults to unset so a low-res tessellation is not stuck on machined gates.

- **Cummins fixes.** Blade-trim and related impeller-build fixes so a Cummins 165 run produces a valid solid (fit accuracy is still short of the target; see open issues).

- **Turned-part gate.** A turned body with a minority of milled flats and cross-holes still passes the coaxial-area gate (cut features are credited). Cell-decomposition fallback uses a 2 % volume gate, so a tiny sliver is no longer accepted as the part.

- **Face classifier and balance-cut exclusion.** Front (nose) and back (balance-plane) faces of a recognised impeller are classified adaptively: irregular or non-periodic marks (balance cuts, drill marks, a one-off back-face slot) are not modelled and are listed as named Metrology exclusions with a 2 mm sector boundary; N-fold functional couplings (Hirth, drive dogs) are recognised and reported; concentric bores, spigots and registers stay with the hub. Uncertain features are excluded and flagged. `rim_slots=auto` no longer cuts on a recognised impeller.

- **2 mm auto-exclusion (any part).** Connected scan regions more than 2 mm from the solid become named whole-region exclusions in Metrology and the conversion report. Each is listed with its area, % of points and reason; its boundary is drawn on the deviation map; the total excluded % is stored and flagged when it is over 5 %. Those points are kept out of a cheap alignment refine and the conversion gate. `--no-auto-exclude-2mm` turns it off.

- **Solid-to-scan support gate.** Flags unsupported solid area above 0.5 % (material the solid has that the scan does not back). The result is merged into the honest verdict.

## Known open issues

- **Surface ripples** on stage 1 and stage 3 impeller skins.
- **Blade fit accuracy** is still short of the 90 % within ±0.05 mm target. STG1 is about 60 % within ±0.05 mm (0.063 mm rms, FAIL).
- **Turned-part profile collapse.** A small turned part with a milled flat and cross-holes is now recognised as turned, but the revolve profile can still collapse, and the run fails honestly instead of writing a sliver.
- **KK&K build time** is long; runs have hit the local-trim budget or been incomplete on this part.
- **Hirth modelling pending.** A Hirth coupling is recognised (tooth count, included angle, radii, height) and a cutter exists, but modelling on the real KK&K stage 1 part is not confirmed and is left pending when the profile test is inconclusive.
- **Root fillets.** Variable-radius G1 root fillets (and the impeller hub boolean) are not finished.


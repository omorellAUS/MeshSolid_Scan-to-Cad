# quadremesh-fusion / meshsolid — milestone 1 (prismatic path)

Mesh (STL/OBJ) → segmented regions → fitted analytic primitives → snapped design intent →
watertight Open CASCADE B-rep → STEP + JSON metrology report. Core library + CLI so a Fusion 360
add-in can call it later (subprocess or import). License intent: GPL (personal use).

**Honesty rule:** a STEP file is written only if the B-rep is a single solid, `BRepCheck_Analyzer`
is valid, and mesh↔solid deviation passes acceptance (rms ≤ 1.5·tol, p95 ≤ 3·tol both directions).
Otherwise `status: failed` with a reason, exit code 2, and no STEP.

## Run
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m meshsolid part.stl -o part.step --report report.json \
    --preview preview.png --deviation-ply deviation.ply [--tol 0.1] [--no-round] [--measure D3=40.02]
.venv/bin/python -m pytest            # fast suite (13 tests)
.venv/bin/python -m pytest -m slow    # real coupling mesh (expected: honest failure)
.venv/bin/python scripts/run_synthetic.py out   # truth-vs-recovered table + previews
```

## Pipeline (meshsolid/)
| module | does |
|---|---|
| `meshio.py` | load keeping connectivity; merge vertices, drop degenerate/duplicate faces, fill holes ≤32 edges, report non-manifold edges, fix winding/orientation |
| `segment.py` | dihedral sharp edges → components; single-primitive fast path; else greedy fit-driven region growing (plane/cylinder/cone/sphere candidates from seeds, largest coverage wins); boundary relaxation + refit |
| `fit.py` | least-squares plane, cylinder, sphere, cone (scipy LM) with rms/max/normal error |
| `snap.py` | direction clustering → orthogonal frame; parallel/perpendicular snaps; coplanar & coaxial merge; cylinder–plane tangency (fillets); rounding to 1.0/0.5 mm grids within `round_tol` (logged raw→snapped); any snap that worsens fit is reverted |
| `build.py` | cell decomposition: oversized fitted surfaces split an enlarged bbox (`BRepAlgoAPI_Splitter`), each cell classified by generalized winding number of the mesh, inside cells fused + `UnifySameDomain`, validated |
| `deviation.py` | mesh→solid and solid→mesh distances (max/mean/rms/p95, Hausdorff), per-region and per-vertex signed deviation, colour PLY |
| `pipeline.py` | orchestration + report.json (per-region params raw & snapped, fit residuals, per-region deviation, confidence/flags, named dimensions D1..Dn, measurement checks) |
| `synthetic.py` | known-dimension CAD parts → isotropic remesh (pymeshlab) → Gaussian normal noise |

## Report fields for the refinement loop
`regions[]`: id, kind, params (snapped), params_unsnapped, fit{rms,max,normal_err_deg}, deviation{max,mean,rms,p95},
confidence, flags, low_confidence. `dimensions[]`: name (D1..), type (diameter/distance/span/cone angle), regions,
value (snapped), fit_value (raw LS). `measurement_checks[]` from `--measure NAME=VALUE` (check only — not yet fed
back as constraints).

## Known limits (milestone 1)
* No torus / freeform surfaces → parts with toroidal blends (e.g. the six-lobed coupling) fail honestly.
* Cell decomposition uses untrimmed surfaces; cost grows with surface count (`--max-tools`, default 80).
* Synthetic tests use round truth dims; the non-round rotated test runs with rounding off.

---

# MeshSolid build plan

Releases **1.0.3 → 1.3.1**, each with two or three significant changes, a measurable done line, and a test target.
Generic recognition now comes before the impeller polish. Nothing starts until you say go.

## Roadmap

| Release | Focus |
|---------|-------|
| 1.0.3 | Reliability |
| 1.0.4 | Traceability and baseline |
| 1.0.5 | Test generator and noise limit |
| 1.0.6 | Gates and closing the shell |
| 1.0.7 | Quality and compatibility refinement |
| 1.2.0 | Smooth vanes (hard rule) |
| 1.1.0 | Region connection map |
| 1.1.0 | Region connection map |
| 1.1.1 | Slot and chamfer rules |
| 1.1.2 | Feature builder and generic tab |
| 1.1.3 | Bounding box and cylinder |
| 1.2.0 | Smooth vanes (hard rule) |
| 1.2.1 | Metrology v3 and Refine |
| 1.2.2 | Impeller fillets and nose |
| 1.2.3 | Hirth coupling |
| 1.3.0 | Batch runs |
| 1.3.1 | API and AI assist |
| 1.0.6 | Gates and closing the shell |
| 1.0.7 | Quality and compatibility refinement |
| 1.2.0 | Smooth vanes (hard rule) |
| 1.1.0 | Region connection map |
| 1.0.5 | Test generator and noise limit |

**Order:** reliability and measurement first, then the generic path for parts it has never seen, then impeller quality, then batch and API.

## Rules for every release

1. If a release misses its done line, stop and replan before the next one starts.
2. Each release ships as a lossless Windows zip and a source zip, with release notes.
3. Every part is run twice; KK&K stays the untuned holdout.
4. Reports show capped and uncapped rms, the capped count and %, final pose with warnings, spec version, run time, named exclusions, engine version, run ID and commit.
5. Figures are never compared across spec versions, and the scoreboard must not get worse.
6. Standing rules stay: no balance cuts, Hirth teeth modelled, 2 mm auto-exclusion with overrides, smooth vanes, Python 3.10 left as the default.
7. Development uses small generated parts at low resolution on my computer; big real parts run on your PC at release time.

## 1.0.3 Reliability (shipped)

1. Per-stage time limits and a heartbeat, with separate end reasons: crashed, cancelled by user, killed by watchdog, timed out, rejected by gate.
2. Input normalisation: meshes over about 300k faces are reduced for recognition, with the full mesh kept for deviation; scale and units are checked (such as a mesh about 1,650 mm across that should be mm or inches).
3. Job-folder pruning: successful intermediates are deleted and only STEP, reports and logs are kept.

**Done when** one of the 1.3M–5.3M-face meshes builds or fails with a readable reason inside its limit, the 1,650 mm mesh is normalised or flagged, and the failure table comes straight from the logs.

**Tested on** Box: unit tests. PC: one big mesh plus Stage 3 to check nothing regressed.

## 1.0.4 Traceability and baseline

1. Every report and CSV row tagged with engine, spec and solid version, run ID and git commit; one scoreboard CSV; the repo copied to your PC.
2. Baseline re-measure of the existing STG1, Stage 3 and pinion solids under spec v2 before anything else changes.
3. Add-in fixes: decimal-comma parsing and a warning when the display-mesh fallback is used.

**Done when** the scoreboard holds tagged spec v2 baselines for STG1, Stage 3 and the pinion, and the add-in fixes pass their tests.

**Tested on** PC: three Metrology runs, each twice.

## 1.0.5 Test generator and noise limit

1. Procedural generator that randomises the synthetic parts into hundreds of variants, plus nasty cases: tiny fillets, thin walls, near-tangent faces, holes in the scan.
2. Noise limit measured from a repeat scan of one impeller, which sets the noise model and decides whether 0.03 mm is reachable; a held-out set of real scans that is never tuned against.
3. Scoring on feature recall and precision, valid solid, volume error, run time and rms, written to the scoreboard.

**Done when** one command scores 100+ generated parts and names the broken layer for each failure, and the repeat-scan noise figure is recorded.

**Tested on** Box at low resolution, plus one repeat scan from you.

## 1.0.6 Gates and closing the shell

1. Fixed tolerance classes chosen by you: machined, printed, organic, rough scan. No gates derived from the mesh.
2. Shell-closing ladder: looser sewing tolerance, then shape healing, then the prismatic builder, then a freeform fallback that always gives a valid solid with areas marked approximate.
3. Bracket round 3: fillet only edges the scan shows as rounded, keep sharp corners sharp, add the lip and floor fillets, snap the bore.

**Done when** Stage 3 passes the machined gate, the five shell-failure jobs each give one valid solid, and the bracket passes its gate.

**Tested on** Generator suite on the box; Stage 3 and the bracket on the PC.

## 1.1.0 Region connection map

1. One shared map of regions, their surface types and how they meet (angle, tangency, convex or concave), built once per part.
2. Existing hole, counterbore, fillet, thread and keyway recognisers moved onto the map, one at a time, with results checked against the old code.

**Done when** all existing tests pass and generator scores are no worse than 1.0.6.

**Tested on** Box: test suite plus generator.

## 1.1.1 Slot and chamfer rules

1. General slot rule: two parallel planes joined by two cylinders (or a cylinder pair for keyhole slots).
2. Chamfer rule: a narrow plane strip between two faces at an angle, snapped to standard sizes.

**Done when** slot and chamfer recall and precision each at least 90% on generated parts.

**Tested on** Box: generator.

## 1.1.2 Feature builder and generic tab

1. Feature-by-feature build: keep a feature only if its fit gain beats the measured scan noise, and prefer fewer faces.
2. Generic tab in the add-in that routes unknown parts through this path, with the freeform fallback underneath.

**Done when** on generated parts: feature recall at least 90%, precision at least 90%, volume error within 1%, and a valid solid on at least 95%. On the held-out real scans: a valid solid on every part.

**Tested on** Generator plus held-out scans; bracket on the PC.

## 1.1.3 Bounding box and cylinder

1. Auto box or cylinder fitted to the scan plus 2 mm and shown in Fusion; manual drag snapping in 1 mm or 0.1 mm steps, with axis tilt for cylinders.
2. The engine takes axis, planes, radius and face heights from it as starting references, refines within a small window, warns where the scan disagrees, and excludes everything outside as a named region.

**Done when** generated parts given a deliberately bad auto-axis are recovered when a box or cylinder is supplied, and the settings appear in the report.

**Tested on** Generator; one impeller and one bracket on the PC.

## 1.2.0 Smooth vanes (hard rule)

1. Vanes built from 5–7 smooth blade sections with few control points, lofted hub to tip; ruled vanes used automatically when the scan fits.
2. Curvature ripple check: a rippled vane is refit with fewer control points; if smoothing cannot meet tolerance, the build ends as a fail, never a rippled pass.

**Done when** zero ripple flags on Stage 3, Stage 1 and KK&K, filtered rms rises by no more than 0.005 mm against the 1.0.4 baseline, and Stage 3 still passes the machined gate.

**Tested on** PC: Stage 3, Stage 1, KK&K, each twice.

## 1.2.1 Metrology v3 and Refine

1. Two-pass Metrology (spec v3): raw base result, then a noise-filtered rerun over about 3–5 times the point spacing, side by side with the difference, and a raw/filtered map toggle.
2. Refine with Metrology: 2–3 rounds moving control points toward the filtered deviation inside the smoothness limit, fitted on half the scan points and validated on the other half; a round that improves the fit half but not the check half is rolled back.

**Done when** Refine lowers filtered rms on the validation half for Stage 3 and Stage 1 without tripping the ripple check.

**Tested on** PC: Stage 3, Stage 1; KK&K as holdout.

## 1.2.2 Impeller fillets and nose

1. Variable-radius root fillets where the vanes meet the hub.
2. Smooth transition from the vane tops into the nose cone, and a fitted nose cone profile.

**Done when** the orange areas at the vane roots and nose on Stage 3 are gone from the filtered map and the solid stays valid.

**Tested on** PC: Stage 3, Stage 1, KK&K.

## 1.2.3 Hirth coupling

1. Hirth tooth recognition on impeller back faces: tooth count, angle and height, snapped to the standard.
2. Hirth teeth modelled as a patterned feature, with balance cuts still excluded.

**Done when** KK&K stage 1 builds with Hirth teeth that match the scan within the machined gate, and balance cuts are listed as excluded.

**Tested on** PC: KK&K, run twice.

## 1.3.0 Batch runs

1. Batch panel: load each part, set its box or cylinder and build settings, Add to Batch; settings saved beside each part.
2. Run logic: 2 attempts per part, then a retry pass with 2 more, each retry changing settings from a fixed ladder in code; PC kept awake; 2–3 parallel jobs if cores and RAM allow.
3. Output: one batch folder with every job folder, a focused batch summary log, and a failure log that feeds new cases into the generator.

**Done when** an overnight batch of 20 parts runs without the runner crashing, every part ends as pass or logged fail, and at least 80% of parts give a valid solid.

**Tested on** PC overnight run.

## 1.3.1 API and AI assist

1. Local job server and MCP connector (submit batch, status, report, rerun) so Claude or Ollama can drive MeshSolid.
2. Your local Qwen model does the morning triage, grouping failures and writing the summary, choosing only from a fixed menu.
3. Locked down: localhost only, token required, paths limited to the batch folder, every setting validated, no shell commands.

**Done when** Qwen submits and reviews a batch end to end, and tests show bad paths, bad settings and missing tokens are refused.

**Tested on** PC with Ollama.

## 1.0.6 Gates and closing the shell

Fixed tolerance classes: machined, printed, organic, rough scan. No gates derived from the mesh.

Shell-closing ladder: looser sewing tolerance, then shape healing, then the prismatic builder, then a freeform fallback that always gives a valid solid with areas marked approximate.

Bracket round 3: fillet only edges the scan shows as rounded, keep sharp corners sharp, add the lip and floor fillets, snap the bore.

Done when Stage 3 passes the machined gate, the five shell-failure jobs each give one valid solid, and the bracket passes its gate.

## 1.1.0 Region connection map

One shared map of regions, their surface types and how they meet (angle, tangency, convex or concave), built once per part.

Existing hole, counterbore, fillet, thread and keyway recognisers moved onto the map, one at a time, with results checked against the old code.

Done when All existing tests pass and generator scores are no worse than 1.0.6.

Tested on Box: test suite plus generator.

## 1.0.7 Quality and compatibility refinement

Quality and compatibility refinement release with bug fixes, stability improvements, 
enhanced compatibility, and performance optimizations.

## 1.2.0 Smooth vanes (hard rule)

1. Vanes built from 5–7 smooth blade sections with few control points, lofted hub to tip; ruled vanes used automatically when the scan fits.
2. Curvature ripple check: a rippled vane is refit with fewer control points; if smoothing cannot meet tolerance, the build ends as a fail, never a rippled pass.

**Done when** Zero ripple flags on Stage 3, Stage 1 and KK&K, filtered rms rises by no more than 0.005 mm against the 1.0.4 baseline, and Stage 3 still passes the machined gate.

**Tested on** PC: Stage 3, Stage 1, KK&K, each twice.

## After 1.3.1





Parametric output (Fusion timeline features with editable dimensions), speed work on parallel blade and solid building, and the drone curved-skin backlog.

Then an optional **Make Drawing** step using draftwright 0.6 (github.com/pzfreo/draftwright): STEP to a dimensioned SVG, PDF or DXF drawing. It is called as a separate installed program because of its AGPL-3.0 licence, unless a commercial licence is bought.

The numeric bars (90% recall and precision, 1% volume, 0.005 mm vane allowance, 80% overnight valid solids) are starting proposals; change any of them and I'll update the page.

## 1.0.5 Test generator and noise limit

Procedural generation of 100+ synthetic parts with nasty cases (tiny fillets, thin walls, near-tangent faces, holes in the scan); noise limit measured from a repeat scan of one impeller, deciding whether 0.03 mm is reachable; scoring on feature recall and precision, valid solid, volume error, run time and rms, written to the scoreboard.

**Done line.** One command scores 100+ generated parts and names the broken layer for each failure, and the repeat-scan noise figure is recorded.

**Tested on** Box at low resolution, plus one repeat scan from you.

# quadremesh-fusion / meshsolid — milestone 1 (prismatic path)
# MeshSolid
Mesh (STL/OBJ) → segmented regions → fitted analytic primitives → snapped design intent →
watertight Open CASCADE B-rep → STEP + JSON metrology report. Core library + CLI so a Fusion 360
add-in can call it later (subprocess or import). License intent: GPL (personal use).
MeshSolid is an Autodesk Fusion 360 add-in and a standalone Python engine that turns a 3D scan into a clean, watertight solid. It reads a triangle mesh (STL, OBJ, or PLY), builds an Open CASCADE B-rep, writes a STEP file, and reports how far that solid sits from the scan. Optional NVIDIA CUDA acceleration covers the heavy closest-point queries. The same engine also drives a FreeCAD workbench.
**Honesty rule:** a STEP file is written only if the B-rep is a single solid, `BRepCheck_Analyzer`
is valid, and mesh↔solid deviation passes acceptance (rms ≤ 1.5·tol, p95 ≤ 3·tol both directions).
Otherwise `status: failed` with a reason, exit code 2, and no STEP.
Generic quad remeshers leave the scan as another mesh. The quads can still be noisy, non-manifold, or a long way from a solid a CAD system can edit and machine. MeshSolid fits surfaces, builds one validated solid, and measures it under a fixed tolerance class. A run that misses its class stays a failure: the report says why, and Fusion imports nothing.
## Run
The current package version is **2.0.1**, recorded in `meshsolid/__init__.py`, `fusion_addin/MeshSolid/MeshSolid.manifest`, and `package.xml` (GPL-3.0-or-later).
## Current release (2.0.1)
**[MeshSolid 2.0.1](https://github.com/omorellAUS/MeshSolid_Scan-to-Cad/releases/tag/2.0.1)** is the current release. 2.0.0 put a reliability and traceability layer on the 1.0.2a engine. 2.0.1 adds honest measurement (spec v3) and Stage 3 fuse timing. Stage 3 solid-building geometry is unchanged: the impeller fuse and the STEP deviation gate are the 1.0.2a path. Full notes are in `CHANGELOG.md`. The measurement contract is `docs/measurement_spec_v3.md`.
### 2.0.0 — reliability and traceability
2.0.0 restarted from the 1.0.2a engine. A Stage 3 run that passes the gate still writes its STEP.
- Every report ends as one of `ok`, `crashed`, `cancelled`, `timed_out`, or `rejected_by_gate`, including a user cancel (SIGTERM) and a watchdog timeout (`END_REASONS` in `meshsolid/pipeline.py`).
- Each stage has its own time limit (`Params.stage_limits`). With a run budget the limit shrinks to the time still left, so the stage stops with a reason (`meshsolid/budget.py`, `meshsolid/watchdog.py`).
- A child process keeps rewriting the heartbeat during a long Open CASCADE call. A thread inside the engine cannot run while that call holds the interpreter (`meshsolid/heartbeat.py`).
- Above 300,000 faces, recognition uses a reduced copy. Deviation is measured on the full-resolution mesh, taken before decimation (`recognition_meshes` in `meshsolid/normalise.py`).
- The units check flags a mesh whose size looks scaled by 25.4 or by 1,000. The coordinates stay as stored (`check_units`).
- Finished job folders drop the copied mesh, the deviation PLY, and the heartbeat. The STEP, the reports, the logs, and the preview PNG stay (`prune_job_folder` in `meshsolid/cli.py`). The Fusion review opens that PNG as soon as the run returns.
- A display name may contain `&`. Reports keep it. File names use a safe id from `meshsolid/names.py` (`Hub & Cover` becomes `Hub_and_Cover`).
- Seven report bugs from that baseline are fixed: the scoreboard `Path` import, the units-check `asarray` axis, gate text that hid a blade-scatter keep, the stalled heartbeat, an impeller build line that printed `solids=None`, a PDF shift labelled in metres, and a PDF beyond-cap percentage that disagreed with the JSON.
When the blade-scatter rule keeps a solid that missed the class limit, the verdict says so. The accept boolean, and whether the STEP is written, stays the same.
### 2.0.1 — honest checks (spec v3) and fuse profile
New reports carry `spec_version = v3`. `python -m meshsolid.remeasure` re-measures an existing STEP and mesh under v3 and lists every metric that moved versus v2. It does not rebuild the solid.
- **One exclusion list.** `meshsolid.exclusions` is the policy the engine and Metrology both call, so the excluded percentage and the balance-cut count stay in step.
- **Balance-cut cap.** Billet balance cuts use their own limits, 5 mm depth and 2,500 mm² total area (`--balance-depth-cap`, `--balance-area-cap`). They are reported on their own line and left out of the general 5% exclusion cap.
- **Bores.** Volume error is `100 * (V_solid - (V_mesh - V_hand)) / |V_mesh - V_hand|`. A hand-entered design bore is subtracted from the mesh volume. The solid-to-scan support check ignores samples that lie on that feature.
- **`valid_solid`.** Exactly one closed solid that passes `BRepCheck_Analyzer`. The solid count is always logged (`meshsolid/solid_check.py`).
- **Declared mesh source.** `parts.csv` has a `source` column, `scan` or `cad_export`. That declaration is authoritative (`--mesh-source` overrides the file). The classifier warns when it disagrees. On a declared scan, exact-mesh and facet-sag tightening is ignored and the gate stays at the scan-noise tolerance (`meshsolid/mesh_source.py`).
- **Uncapped and capped RMS.** `rms_uncapped` keeps distances past the 0.2 mm cap. `rms_capped` is the legacy clipped figure. Reports also list the capped count and percentage (`meshsolid/metrology_spec.py`, `meshsolid/metrology_pdf.py`).
- **`FuseProfile`.** Each Stage 3 impeller boolean (floor envelope, hub-and-floors fuse, base-and-blades fuse) is timed into the report. `--fuse-bench` is off by default. When it is on, Open CASCADE parallel and glue options are timed and those shapes are discarded. The solid that is kept is the ordinary serial fuse (`meshsolid/fuse_profile.py`).
## Key capabilities and architecture
```
fusion_addin/     Fusion 360 add-in, dialog options, Windows installer
freecad_addin/    FreeCAD workbench (same engine, shared mslib)
meshsolid/        Python / Open CASCADE engine, Metrology, CLI
scripts/          pack_install_zip.py, device bench, part generators
scoreboard.py     SHA-256 job ledger (reads job folders only)
parts.csv         part id, mesh SHA-256, source (scan | cad_export)
tests/            engine, add-in, metrology, scoreboard, installer packer
docs/             measurement specs (v1, v2, v3)
```
### Fusion 360 add-in
`fusion_addin/MeshSolid/` is a Windows add-in. `MeshSolid.py` places a **MESHSOLID 2.0.1** panel on the Design workspace UTILITIES and MESH tabs, and keeps a command in the ADD-INS dropdown. The two commands are **Mesh to Solid** and **Metrology**.
Dialog choices in `mslib/ui_options.py` map onto flags that exist on `python -m meshsolid`:
| Part type | Engine request |
| --- | --- |
| Auto | All recognisers on |
| Mechanical | Prismatic or turned. Gears, impeller, and screws off |
| Gear | Involute gear on a turned body |
| Impeller | Impeller path only. If the recogniser cannot build it, the run stops with that reason |
| Freeform | Printed accuracy class, so the section-loft builder is tried first |
Accuracy is **Machined** (vertex-to-solid rms ≤ 1.5·tol, p95 ≤ 3·tol) or **Printed** (rms ≤ 0.05 mm, p95 ≤ 0.10 mm). Compute is Auto, CPU, or GPU only. The time-limit slider runs from 5 to 180 minutes. Optional measurements use `NAME=VALUE`, for example `bore_d=14.29`.
The engine runs in its own process (`mslib/engine_client.py`). On success the review dialog shows the outcome, deviation against the class limits, flags, the preview PNG, and one tick box per user-parameter group. Import brings the STEP in as one body and creates the ticked `ms_*` parameters. Those parameters record recognised design values (gear, thread, impeller, snapped dimensions) beside the body. The imported STEP stays a plain B-rep. A failure imports nothing and names the case: engine not ready, cancelled, timed out, crashed, missing report, outside the deviation limits, or the engine's own text, plus the job folder under `%LOCALAPPDATA%\meshsolid\jobs`.
The pure-Python add-in library (`mslib/`) is covered by `tests/test_addin.py`. `MeshSolid.py` is the Fusion entry point; the repo records that it still needs its first live session inside Fusion. The checklist is in `fusion_addin/README.md`.
### Engine
`meshsolid/` is the library and the CLI (`python -m meshsolid`). `meshsolid/meshio.py` loads STL, OBJ, and PLY (the loader also names 3MF), merges vertices, drops degenerate and duplicate faces, and fills holes of up to 32 edges. Open CASCADE comes from `cadquery-ocp`. A STEP file is written only when the report status is `ok`: one valid solid, and deviation inside the selected class. Any other ending leaves no STEP. The CLI returns exit code 2. A cancel (SIGTERM) exits 124.
Stages, in order, are load and repair, the units check, recognition (with the 300,000-face reduction), normalise, segmentation, coaxial refit, threads, freeform, snapping, blends, consolidate, build, deviation, and preview. `--builder` selects `auto`, `extrude`, `local`, `revolve`, or `cell`. Recognisers already in the tree cover prismatic cell decomposition, 2.5D extrude, turned bodies, involute gears, threads, screws, splines, keys, catalogue bearings, impellers, and freeform B-splines.
The Windows installer builds a private virtual environment at `%LOCALAPPDATA%\meshsolid\venv` with Python 3.13 (3.12 is also accepted). That environment is the only interpreter the add-in launches. System Python and PATH stay as they were. See [Installation](#installation-and-quick-start).
CUDA is optional. `requirements-gpu.txt` installs NVIDIA Warp. `--device auto` uses CUDA when a driver and a device are present, and the CPU otherwise. `--device cuda` stops with a clear error when no GPU is usable. The deviation stage's closest-point query runs on the GPU; the distance is recomputed in float64 on the chosen face, so it matches the CPU path (`meshsolid/device.py`). `report.json` records the device under `device`. `scripts/bench_device.py` times a scan.
On Windows, a conversion or Metrology job blocks system sleep for the length of the job and releases that request when the job ends, including on error (`meshsolid/power.py`). The display can still turn off.
### Metrology
`python -m meshsolid.metrology SCAN.stl SOLID.step --out DIR` writes a signed deviation map (positive means the scan is outside the solid), zone and histogram views, a PDF (`meshsolid/metrology_pdf.py`), and JSON stats. Spec v3 keeps spec v2 alignment. Points past the 0.2 mm cap stay inside `rms_uncapped` and count as failures in the pass percentage. `rms_capped` is the legacy clipped number.
Connected scan regions more than 2 mm from the solid become named exclusions (`meshsolid/exclude_2mm.py`). `--no-auto-exclude-2mm` puts those regions back into the inspection. The solid-to-scan support check (`meshsolid/support_gate.py`) flags solid area the scan does not back. The default limit is 0.5% of the solid area. Samples on a hand-entered bore are ignored. The Fusion **Metrology** command (`mslib/metrology_cmd.py`) runs this same module.
### Scoreboard and tests
`parts.csv` is the part index: `part_id`, `input_mesh_sha256`, `source` (`scan` or `cad_export`), face count, file size, and notes. `scoreboard.py` reads job folders and never writes back into the engine. Each row stores the report fields, the SHA-256 of the input mesh, and the SHA-256 of the STEP when one was written, plus the engine, spec, solid version, run id, and commit. A job with no STEP and an end reason is a valid row.
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m meshsolid part.stl -o part.step --report report.json \
    --preview preview.png --deviation-ply deviation.ply [--tol 0.1] [--no-round] [--measure D3=40.02]
.venv/bin/python -m pytest            # fast suite (13 tests)
.venv/bin/python -m pytest -m slow    # real coupling mesh (expected: honest failure)
.venv/bin/python scripts/run_synthetic.py out   # truth-vs-recovered table + previews
python scoreboard.py --jobs DIR [DIR ...] --parts parts.csv -o scoreboard.csv \
    --engine 2.0.1 --spec v3 --solid-version 2.0.1 [--commit SHA] [--append] [--recursive]
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
`pytest.ini` skips tests marked `slow` unless you ask for them. Release behaviour is covered by `tests/test_step1_reliability.py` (2.0.0), `tests/test_v201_honest.py` (2.0.1), `tests/test_scoreboard.py`, `tests/test_pack_install_zip.py`, `tests/test_addin.py`, `tests/test_metrology.py`, `tests/test_exclude_2mm.py`, and `tests/test_device.py`, next to the recogniser and builder tests.
## Report fields for the refinement loop
`regions[]`: id, kind, params (snapped), params_unsnapped, fit{rms,max,normal_err_deg}, deviation{max,mean,rms,p95},

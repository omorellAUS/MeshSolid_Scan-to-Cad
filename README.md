# MeshSolid

MeshSolid is an Autodesk Fusion 360 add-in and a standalone Python engine that turns a 3D scan into a clean, watertight solid. It reads a triangle mesh (STL, OBJ, or PLY), builds an Open CASCADE B-rep, writes a STEP file, and reports how far that solid sits from the scan. Optional NVIDIA CUDA acceleration covers the heavy closest-point queries. The same engine also drives a FreeCAD workbench.

Generic quad remeshers leave the scan as another mesh. The quads can still be noisy, non-manifold, or a long way from a solid a CAD system can edit and machine. MeshSolid fits surfaces, builds one validated solid, and measures it under a fixed tolerance class. A run that misses its class stays a failure: the report says why, and Fusion imports nothing.

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
python scoreboard.py --jobs DIR [DIR ...] --parts parts.csv -o scoreboard.csv \
    --engine 2.0.1 --spec v3 --solid-version 2.0.1 [--commit SHA] [--append] [--recursive]
```

`pytest.ini` skips tests marked `slow` unless you ask for them. Release behaviour is covered by `tests/test_step1_reliability.py` (2.0.0), `tests/test_v201_honest.py` (2.0.1), `tests/test_scoreboard.py`, `tests/test_pack_install_zip.py`, `tests/test_addin.py`, `tests/test_metrology.py`, `tests/test_exclude_2mm.py`, and `tests/test_device.py`, next to the recogniser and builder tests.

### FreeCAD workbench

`freecad_addin/MeshSolid/` is a FreeCAD 1.0+ workbench. It calls the same engine process and shares `mslib` with the Fusion add-in. Install with `fusion_addin/install_windows.ps1 -FreeCAD` (add `-NoFusion` to skip the Fusion copy). Field notes are in `freecad_addin/README.md`.

## Installation and quick start

### Windows, from the release zip

You need Windows 10 or 11, Autodesk Fusion, and 64-bit Python 3.12 or 3.13. The first install downloads the pinned wheels (about 180 MB: Open CASCADE through `cadquery-ocp`, pymeshlab, scipy, and the rest of `requirements.txt`). The CPU path always runs. An NVIDIA GPU is optional.

1. Download [MeshSolid-2.0.1-windows.zip](https://github.com/omorellAUS/MeshSolid_Scan-to-Cad/releases/tag/2.0.1) and unzip it to a stable folder, for example `C:\meshsolid\MeshSolid-2.0.1`. The archive root is `MeshSolid-2.0.1/` and contains `INSTALL_MeshSolid.bat`, `fusion_addin\install_windows.ps1`, `meshsolid\`, `requirements.txt`, and `requirements-gpu.txt`. `scripts/pack_install_zip.py` is the script that builds this zip (`dist/MeshSolid-2.0.1-windows.zip`).
2. Double-click `INSTALL_MeshSolid.bat`. It looks for Python 3.13, then 3.12, via the `py` launcher or under `%LOCALAPPDATA%\Programs\Python`. If neither is present it installs Python 3.13 for the current user with winget and leaves PATH unchanged, so the existing system Python stays the default. It then runs `fusion_addin\install_windows.ps1 -Python <that python> -Gpu`.
3. For a CPU-only install, from the unzipped folder:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File fusion_addin\install_windows.ps1
```

Add `-Gpu` to install NVIDIA Warp and check CUDA. Other switches: `-Python`, `-VenvDir`, `-AddinsDir`, `-CheckOnly`, `-QuickCheck`, `-NoAddinCopy`, `-FreeCAD`, `-NoFusion`.

4. The script creates `%LOCALAPPDATA%\meshsolid\venv`, installs into that environment, runs `python -m meshsolid.selfcheck` (imports, then a synthetic box meshed and rebuilt to STEP), copies `fusion_addin\MeshSolid` to `%APPDATA%\Autodesk\Autodesk Fusion 360\API\AddIns\MeshSolid`, and writes `settings.json` (venv `python.exe` and engine directory). It stops at the first step that fails and prints that step's error.
5. In Fusion: **Utilities > Add-Ins** (Shift+S) > **Add-Ins** > **MeshSolid** > **Run**. Tick **Run on Startup** if the panel should load with Fusion. A **MESHSOLID 2.0.1** panel with **Mesh to Solid** appears on the UTILITIES and MESH tabs. Placement is logged to `%LOCALAPPDATA%\meshsolid\addin_ui.log`.

Select a mesh body, or browse to an STL, OBJ, or PLY file. Keep **Review before import** ticked for the first runs. **Convert to Solid** starts the engine. A small prismatic mesh is the right first check: the progress text should update, cancel should stop the run, and a successful review should import one body.

To update, replace the folder (or pull the repo) and run the installer again. The virtual environment is reused. The add-in folder and `settings.json` are rewritten.

### From a source checkout

Use Python 3.12 or 3.13. The wheels in `requirements.txt` are checked for those two versions. A virtual environment in the clone keeps this interpreter off the system PATH.

```bash
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
# optional CUDA closest-point path
.venv/bin/pip install -r requirements-gpu.txt

.venv/bin/python -m meshsolid.selfcheck
.venv/bin/python -m meshsolid scan.stl -o scan.step \
    --report scan.report.json --preview scan.preview.png \
    --deviation-ply scan.deviation.ply
.venv/bin/python -m meshsolid.metrology scan.stl scan.step --out metrology_out
.venv/bin/python -m pytest
.venv/bin/python -m pytest -m slow
```

On Windows the interpreter is `.venv\Scripts\python.exe`. `python -m meshsolid --help` lists the flags, including `--accuracy machined|printed`, `--mesh-source scan|cad_export`, `--parts parts.csv`, `--device auto|cpu|cuda`, `--time-budget`, `--impeller`, `--gears`, `--fuse-bench`, and `--measure NAME=VALUE`.

Rebuild the Windows zip from the checkout:

```bash
.venv/bin/python scripts/pack_install_zip.py
```

That writes `dist/MeshSolid-<version>-windows.zip`. The packer leaves out virtual environments, `settings.json`, caches, and any `user_scans` tree (`tests/test_pack_install_zip.py`).

## Engineering and quality principles

- **Deterministic builds.** A scoreboard row is identified by the SHA-256 of the input mesh and, when a solid was written, the SHA-256 of the STEP, plus the engine version, the spec version, and the commit (`scoreboard.py`).
- **Honest reporting.** The user picks a tolerance class and the report uses that class. The classes in 2.0.1 are machined and printed (`meshsolid/accuracy.py` and the Fusion accuracy dropdown). Organic and rough scan are the further fixed classes to be signed off in 2.0.2. A miss is a failure. The `_APPROX` freeform fallback planned for 2.0.3 will be labelled approximate, so it is never reported as meeting the selected class.
- **Measure before redesigning.** Profile and failure data come before a rewrite. The 2.0.1 `FuseProfile` is the measurement the 2.0.3 fuse speedup has to beat. A new construction path stays behind a flag until the scoreboard shows it ahead of the baseline.
- **Balance cuts stay off the model.** Billet-specific impeller balance cuts are excluded under their own cap (5 mm depth, 2,500 mm²) and are left unmodelled. They appear on their own report line (`meshsolid/exclusions.py`).
- **Vanes stay smooth.** Impeller vanes are held to a smooth, ripple-free skin. The curvature-ripple check that enforces this is part of 2.1.1.
- **Fillets on the inside, sharp tips.** Concave internal junctions take a fillet. Leading-edge tips stay sharp. Fillet radius comes from a cylinder or sphere fitted to the scan, with a largest-working-radius fallback (2.0.5).
- **Outliers above 2 mm.** Connected scan regions more than 2 mm from the solid are named exclusions. `--no-auto-exclude-2mm` returns them to the inspection (`meshsolid/exclude_2mm.py`).

## Roadmap (2.0.0 through 2.3.1)

2.0.0 and 2.0.1 are in this tree. Every later row is the plan for that release.

| Release | Focus | Status | Key deliverables |
| --- | --- | --- | --- |
| 2.0.0 | Reliability and traceability | Done | Clean restart from the 1.0.2a baseline. `parts.csv` and `scoreboard.py`. Per-stage time limits, units check, child-process heartbeat, recognition-only reduction above 300,000 faces, job-folder pruning, and the seven report bug fixes. |
| 2.0.1 | Honest checks (spec v3) and fuse profile | Released | Shared exclusion list. Bore-adjusted volume and support checks. Balance-cut cap (5 mm / 2,500 mm²). Uncapped and capped RMS. Exact-mesh gate tightening ignored on declared scans. Stage 3 `FuseProfile`. |
| 2.0.2 | Test parts and fixed gate classes | Next | Synthetic scan generator for housings, covers, levers, pulleys, gears, brackets, castings, and bolts. Repeat-scan noise-floor measurement. Signed-off fixed tolerance classes. |
| 2.0.3 | Faster fuse, and always a solid | Planned | Stage 3 boolean speedup, with a target under 5 minutes guided by the 2.0.1 profile. `split-and-keep` surfacing behind a flag. `_APPROX` freeform solid fallback so a part still receives a STEP. |
| 2.0.4 | Prismatic auto-construct | Provisional | Build-plan preview. Snapped section sketches, one per step. A feature is kept only when it improves deviation. |
| 2.0.5 | Fillets that finish | Provisional | Cylinder and sphere radii fitted to the scan. Fillet ladder: largest working radius, with a time limit per fillet group. |
| 2.0.6 | Turned parts and scan merging | Provisional | Axial section profiles: grooves, recesses, keyways, flats, and cross holes. Open3D registration and merging of several scans. |
| 2.1.0 | Generic unseen parts | Provisional | Shared region and adjacency map. Per-segment choice of primitive or freeform. A Generic tab in the Fusion add-in. |
| 2.1.1 | Smooth vanes and a surface network | Provisional | Section lofts with few control points. G1/G2 B-spline surface network. Curvature ripple check. |
| 2.1.2 | Refine and finish impellers | Provisional | Two-pass, noise-filtered Metrology. Bounded control-point refinement on points held back from the fit. Fitted nose profile and Hirth teeth. |
| 2.2.0 | Parametric Fusion timeline | Provisional | Native editable sketches and features in Fusion's timeline, with rule-based fillets and chamfers. |
| 2.3.0 | Overnight batch runs | Provisional | Batch panel with per-part settings, a retry ladder, a wake lock, and parallel jobs. |
| 2.3.1 | Local API and AI assist | Provisional | Localhost-only MCP server with token authentication: build plans, batch submission, and local-LLM triage. |

### Front-end geometry router

A short pre-check sits in front of the two builders. A simple prismatic part is routed to BuildAgentOllama, which returns an editable script immediately. A complex, freeform, or worn scan is routed to MeshSolid. The operator can override the route by hand.

### Metrology refinement loop

Parked until the 2.0.x line is in place. The operator tags zones as trust, ignore, or fix. A fix zone takes an add-material or a remove-material pass, and the report shows the before and after numbers.

### Later

- Curvature-aligned quad remeshing, where a mesh is still the right output.
- The curved-skin freeform backlog.
- Optional drawing generation, through tools that are installed separately.

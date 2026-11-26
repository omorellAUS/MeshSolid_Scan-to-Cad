# MeshSolid Fusion add-in (Windows, personal use)

Mesh body or STL/OBJ file -> meshsolid engine (its own Python, separate process) -> STEP imported back
as one body, with an optional review step and user parameters for the recognised design values.

## Install on Windows
Needs: Windows 10/11, Fusion, Python 3.12 or 3.13 from python.org (64-bit; the `py` launcher is
enough), internet for the first `pip install` (about 180 MB of wheels for the pinned packages plus their dependencies: OCP, pymeshlab, scipy, ...).

1. Copy this repository folder to the PC, e.g. `C:\meshsolid\quadremesh-fusion` (git clone from your
   local copy, or a zip; the user meshes are not in the repo and are not needed).
2. In PowerShell, from the repo folder:
   `powershell -ExecutionPolicy Bypass -File fusion_addin\install_windows.ps1`
   It creates the engine venv at `%LOCALAPPDATA%\meshsolid\venv`, installs `requirements.txt`, runs the
   engine self-check (imports, then a synthetic box meshed and rebuilt to STEP), copies
   `fusion_addin\MeshSolid` to `%APPDATA%\Autodesk\Autodesk Fusion 360\API\AddIns\MeshSolid` and writes
   its `settings.json` (engine `python.exe` and `engine_dir`). It stops at the first failing step and
   prints that step's error. Options: `-Python <path\python.exe>`, `-VenvDir`, `-AddinsDir`,
   `-CheckOnly` (self-check an existing venv), `-QuickCheck` (imports only), `-NoAddinCopy`.
3. In Fusion: Utilities > Add-Ins (Shift+S) > Add-Ins tab > MeshSolid > Run (tick "Run on Startup"
   if wanted). From 1.0.2 a MESHSOLID 2.0.0 panel with a large "Mesh to Solid" button appears on the
   Design workspace's UTILITIES and MESH tabs (it also stays in the Utilities > ADD-INS dropdown).
   Where it was placed is written to %LOCALAPPDATA%\meshsolid\addin_ui.log.

   The dialog: Input (select a mesh body, or a text Browse... button for a file), Part type (Auto / Mechanical / Gear /
   Impeller / Freeform dropdown), Accuracy and speed (accuracy class, Compute: Auto GPU / CPU / GPU only,
   time-limit slider), Measurements (optional), Output (review, user parameters, hide input mesh) and
   Advanced (fit tolerance, batch folder, the engine flags that will be used). OK is "Convert to Solid".
   Every option maps to an existing engine flag (mslib/ui_options.py).

Update later: pull/copy the repo again and rerun the script (the venv is reused; the add-in folder and
settings are rewritten).

## Use
Select a mesh body (or type an STL/OBJ path), optionally add measurements (`bore_d=14.29; D3=40.02`)
and notes, keep "Review before import" ticked, OK. A progress dialog shows the engine stage and can
cancel. Each run gets a job folder in `%LOCALAPPDATA%\meshsolid\jobs\<time>` with the exported mesh,
`*.report.json`, `*.engine.log`, `*.preview.png` and (only on success) `*.step`.

- **Review dialog** (when the engine succeeded): outcome, build facts, deviation against its limits,
  everything flagged (sharp LE, balance cuts built to nominal, excluded scan faces, warnings), notes
  (e.g. the unscanned boss), the preview image, and one tick box per user-parameter group. "Import"
  imports the STEP into the root component and creates the ticked parameters.
- **User parameters** (`ms_*`, from report.json): gear (teeth, DP/module, pressure and helix angle,
  lead, pitch/base/tip/root diameters, span and over-pins measurements), threads (major diameter,
  pitch, length; standard values when matched), impeller (blades, bore, tip radius, exit height, LE
  plane, LE round, wrap and inlet/exit blade angles at hub/mid/tip), snapped dimensions (ticked) and
  other fitted dimensions (unticked; blend tori are many). They record the design values next to the
  body; the imported STEP is a plain B-rep, so they do not drive its geometry. Existing parameters
  with the same name are left unchanged and listed.
- **Failures** never import anything. The dialog says which case it is: engine not ready (self-check
  result), cancelled, timed out (stage reached), crashed (stage reached, log tail), no report (exit
  code, log tail), solid outside its deviation limits (numbers vs limits), or the engine's own failure
  text, plus the job folder. Add-in errors are labelled as add-in errors, not engine results.

## Tested so far (outside Fusion)
- `tests/test_addin.py` (10 tests): STL export from Fusion-style arrays (cm -> mm, quads, polygons);
  engine round trip through the client; bad-mesh failure, cancel and timeout paths; every outcome
  message; report.json -> parameters on synthetic gear/thread data and on the real stg1 impeller
  report; review model/HTML; `MeshSolid.py`'s import step against a stub `adsk` (body-count check,
  existing and failing parameters).
- `install_windows.ps1` ran end to end under PowerShell 7 on Linux (fresh venv from
  requirements.txt, self-check round trip, add-in copy, settings.json read back by the add-in).
- All engine wheels exist for win_amd64 on Python 3.12 and 3.13 (checked with `pip download`).
- The engine's fillet child process has a no-fork path for Windows (BRep file + edge midpoints),
  tested on Linux with `MESHSOLID_NO_FORK=1`: stg1 LE round identical to the fork path.

## Not yet verified (first session on Windows)
`MeshSolid.py` has not run inside Fusion. Check, in order:
1. The add-in loads and the MESHSOLID 2.0.0 panel shows on the UTILITIES and MESH tabs (see addin_ui.log if not).
2. "engine not ready" appears if `settings.json` is wrong (rename it to test), and goes away when fixed.
3. Mesh body export: a Fusion mesh body (Insert Mesh of a known STL) comes out the right size in mm.
   Compare `mesh_body.stl` in the job folder with the original.
4. A small part first: make one with
   `%LOCALAPPDATA%\meshsolid\venv\Scripts\python.exe -c "from meshsolid import synthetic as S; S.to_scan(S.chamfered_block()[0]).export('C:/temp/block.stl')"`
   (run from the repo folder), or use any simple prismatic STL: progress text updates, cancel works, the review dialog opens (it opens through a
   custom event after the first command ends), the preview image shows, Import gives 1 body.
5. User parameters appear in Modify > Change Parameters with the `ms_` prefix.
6. A long run (stg1 impeller, about 1 h): Fusion stays responsive enough through the progress loop.
Known limits: the engine runs while the command waits (Fusion is busy but repaints via doEvents); a
worker thread with custom events, an HTML palette review and native parametric features (sketch +
extrude/revolve from the fitted primitives) are later steps.

## Optional GPU acceleration (NVIDIA, M8b)

The engine can run its closest-point queries (the deviation stage, the heaviest step on big scans) on
an NVIDIA GPU through NVIDIA Warp. The add-ins don't change; only the engine's Python gets the extra.

    powershell -ExecutionPolicy Bypass -File fusion_addin\install_windows.ps1 -Gpu

This installs `requirements-gpu.txt` and checks CUDA. The engine's `--device` option is `auto` by
default (CUDA when usable, otherwise the CPU); `cpu` forces the CPU and `cuda` fails clearly if there is
no usable GPU. report.json records the device used under `device`. GPU results match the CPU path:
the BVH picks the face and the distance is computed in float64 (tests: `tests/test_device.py`, run on
Warp's CPU device on machines without a GPU).

The speedup has only been measured on a machine without a GPU, so measure it on yours:

    %LOCALAPPDATA%\meshsolid\venv\Scripts\python.exe scripts\bench_device.py path\to\scan.stl

# MeshSolid FreeCAD workbench (FreeCAD 1.0+, Windows, personal use)

Mesh object (or STL/OBJ file) -> meshsolid engine (its own Python, separate process) -> one Part
solid, with a review task panel and the recognised design values in a spreadsheet. It shares the
engine client, outcome messages, review content and parameter mapping (`mslib`) with the Fusion add-in.

## Install on Windows
Needs FreeCAD 1.0 or newer, and Python 3.12 or 3.13 from python.org for the engine. The engine
does not run in FreeCAD's own Python.

1. Copy the repository folder to the PC, e.g. `C:\meshsolid\quadremesh-fusion`.
2. From the repo folder, FreeCAD only:
   `powershell -ExecutionPolicy Bypass -File fusion_addin\install_windows.ps1 -FreeCAD -NoFusion`
   (drop `-NoFusion` to install the Fusion add-in as well). It creates or reuses the engine venv at
   `%LOCALAPPDATA%\meshsolid\venv`, runs the engine self-check, and copies `freecad_addin\MeshSolid`
   plus `mslib` into FreeCAD's user Mod folder: `%APPDATA%\FreeCAD\Mod\MeshSolid` for FreeCAD 1.0, or
   the newest `%APPDATA%\FreeCAD\v1-x\Mod` if one exists (FreeCAD 1.1+). It writes `settings.json` there
   and in `%LOCALAPPDATA%\meshsolid\`. If FreeCAD uses another user data folder (Edit > Preferences >
   General shows it), pass `-FreeCADModDir "<that folder>\Mod"`.
3. Restart FreeCAD and switch to the MeshSolid workbench (workbench selector). Use the toolbar button
   or the MeshSolid > Mesh to Solid menu.

Addon Manager layout: the repo root has a `package.xml` that points at `freecad_addin/MeshSolid`.
A clone of the whole repo into the Mod folder (or an Addon Manager custom repository pointing at a git
URL of it) loads the workbench. In that case `mslib` is found in the clone's `fusion_addin` folder
and the engine settings come from `%LOCALAPPDATA%\meshsolid\settings.json`, so you still need to run the
installer once for the engine venv. Do not use both install methods at once: two copies register the
same workbench twice.

## Use
1. Import or open a mesh (File > Import, STL/OBJ) and select the Mesh object, or leave nothing
   selected and type a mesh file path in the panel. The object's placement is applied on export.
2. Optional: measurements (`bore_d=14.29; D3=40.02`) and notes. OK starts the engine. The panel shows
   the stage and elapsed time; FreeCAD stays usable (a timer polls the engine process). Cancel stops it.
3. Review panel (when the engine succeeded): outcome, build facts, deviation against its limits,
   flags (sharp LE, balance cuts built to nominal, excluded scan faces, warnings), notes, the preview
   image, and one tick box per parameter group. OK imports the solid. Cancel keeps the job folder
   and imports nothing.
4. Result: one `Part::Feature` named `MeshSolid_<mesh>`, checked to be exactly one valid solid before
   it is added. It has a `MeshSolidReport` property pointing at the report. Also a spreadsheet
   `MeshSolidParams` with columns parameter / value / units / group / comment, where each value cell has
   the `ms_` alias (e.g. `MeshSolidParams.ms_imp_bore_d`). Running it again updates existing aliases
   in place. The import is wrapped in one undo transaction (not yet tested).

**Driving:** the imported solid is a plain B-rep. The spreadsheet values record the design values and
do not drive it (every comment says so). You can use the aliases in features you add yourself, e.g. a
PartDesign Pocket depth `= MeshSolidParams.ms_D3`. Rebuilding the part as driven features from the
fitted primitives (sketch + pad/revolve) is a later step.

Failures never import anything. The message names the case (engine not ready, cancelled, timed out,
crashed, engine error with its error line, outside the deviation limits, engine failure) and gives
the job folder, report and log paths. Jobs go to `%LOCALAPPDATA%\meshsolid\jobs\<time>`.

## Tested (Linux, FreeCAD 1.0.0 from Debian, under freecadcmd and xvfb)
- `tests/test_freecad.py`, 3 tests:
  - core: engine run, STEP import as one valid Part solid (volume matches the engine), `ms_` aliases;
  - stg1 impeller STEP: 1 valid solid with 276 faces, and 15 impeller parameters such as
    `ms_imp_blades` = 15 and `ms_imp_bore_d` = 14.29 mm;
  - GUI flow under a real event loop: workbench loads, the command opens the run panel with the
    selected Mesh object, progress text, review panel, import, spreadsheet, a bad file reported as an
    engine error, cancel reported, no dialog left open.
- Also run by hand: the p01 gear STEP (1 valid solid, 328 parameters including the gear group), the
  installer's `-FreeCAD` copy loaded from a fresh FreeCAD user folder with the installer-made venv, and
  a full-repo clone in Mod loading through the root `package.xml`.
- Not tested: FreeCAD on Windows itself, Qt6 builds of FreeCAD (1.1), and very large meshes in the GUI.

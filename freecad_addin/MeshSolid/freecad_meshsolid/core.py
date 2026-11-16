"""Non-GUI parts of the FreeCAD workbench (usable from freecadcmd): mesh export, engine run, STEP
import as one Part solid, parameters -> Spreadsheet with ms_ aliases.

Honesty rules (same as the Fusion add-in): nothing is imported unless the engine's report says ok and
its STEP exists; the import is checked for exactly one valid solid; the spreadsheet says the values
are recorded, not driving the imported solid."""
import os
import time

import FreeCAD

from .shared import SETTINGS_FILE, engine_client as EC, review as RV

SHEET_NAME = "MeshSolidParams"
NOTE = "recorded from the meshsolid report; does not drive the imported solid"


def settings_path():
    """MESHSOLID_SETTINGS, else settings.json in the workbench folder, else the installer's shared copy
    %LOCALAPPDATA%\\meshsolid\\settings.json (used when the workbench came from the Addon Manager)."""
    if os.environ.get("MESHSOLID_SETTINGS"):
        return os.environ["MESHSOLID_SETTINGS"]
    if os.path.exists(SETTINGS_FILE):
        return SETTINGS_FILE
    base = os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), ".local", "share")
    return os.path.join(base, "meshsolid", "settings.json")


def settings():
    return EC.EngineSettings.load(settings_path())


def jobs_dir():
    base = os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), ".local", "share")
    d = os.path.join(base, "meshsolid", "jobs", time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(d, exist_ok=True)
    return d


def export_mesh(obj, path):
    """Mesh::Feature (or anything with a .Mesh) -> STL in mm, with the object's placement applied.
    Returns the facet count."""
    mesh = obj.Mesh.copy()
    pl = getattr(obj, "Placement", None)
    if pl is not None and not pl.isIdentity() and mesh.Placement.isIdentity():
        mesh.Placement = pl
    mesh.write(path)
    return mesh.CountFacets


def start(mesh_path, out_dir, measurements=None, notes=None, s=None):
    s = s or settings()
    job = EC.EngineJob(mesh_path, out_dir, dict(measurements or {}), list(notes or []))
    return job, EC.EngineRun(job, s)


def finish(job, rc, s=None):
    s = s or settings()
    res = EC.read_result(job, rc, s.timeout_s)
    res["model"] = RV.review_model(res["report"], res["outcome"])
    return res


def run_blocking(mesh_path, out_dir, measurements=None, notes=None, s=None, poll=0.5):
    """Engine run without a GUI (freecadcmd, tests)."""
    s = s or settings()
    job, run = start(mesh_path, out_dir, measurements, notes, s)
    rc = run.poll()
    while rc is None:
        time.sleep(poll); rc = run.poll()
    return finish(job, rc, s)


def run_batch(folder, out_dir, s=None, recursive=False, poll=0.5, tick=None):
    """Batch folder mode: every mesh in folder through the engine (meshsolid.batch); meshes with a
    same-name STEP are scored against it.  Returns (exit code, summary text).  tick(run) is called while
    it runs (GUI progress); returning True from it cancels."""
    s = s or settings()
    bj = EC.BatchJob(folder, out_dir, recursive=recursive)
    if not bj.meshes():
        return None, "No STL/OBJ/PLY files in the folder, so nothing was run."
    run = EC.EngineRun(bj, bj.settings(s))
    rc = run.poll()
    while rc is None:
        if tick is not None and tick(run):
            run.cancel()
        time.sleep(poll); rc = run.poll()
    return rc, EC.batch_summary_text(bj)


def import_step(doc, step_path, name="MeshSolid"):
    """STEP -> one Part::Feature.  Returns (feature or None, message)."""
    import Part
    shape = Part.read(step_path)
    solids = shape.Solids
    if len(solids) != 1:
        return None, f"the STEP holds {len(solids)} solids, expected 1; nothing was added"
    solid = solids[0]
    if not solid.isValid():
        return None, "the STEP solid is not valid in FreeCAD's check; nothing was added"
    f = doc.addObject("Part::Feature", name)
    f.Shape = solid
    doc.recompute()
    return f, f"added 1 solid ({len(solid.Faces)} faces, volume {solid.Volume:.1f} mm^3)"


def write_parameters(doc, specs, groups, sheet_name=SHEET_NAME):
    """Parameters of the ticked groups -> Spreadsheet rows (name | value | units | group | comment) with
    the value cell aliased to the parameter name.  Existing aliases are updated in place.  Returns
    (sheet, created, updated, failed)."""
    sheet = doc.getObject(sheet_name) or doc.addObject("Spreadsheet::Sheet", sheet_name)
    for c, h in zip("ABCDE", ("parameter", "value", "units", "group", "comment")):
        sheet.set(f"{c}1", h)
    row = 2
    while _has(sheet, f"A{row}"):
        row += 1
    created, updated, failed = 0, 0, []
    for sp in [s for s in specs if s["group"] in groups]:
        name = sp["name"]
        try:
            cell = sheet.getCellFromAlias(name)
            if cell:
                sheet.set(cell, sp["expr"]); updated += 1
            else:
                sheet.set(f"A{row}", name)
                sheet.set(f"B{row}", sp["expr"])
                sheet.set(f"C{row}", sp["units"] or "count")
                sheet.set(f"D{row}", sp["group"])
                sheet.set(f"E{row}", f"{sp['comment']}; {NOTE}")
                sheet.setAlias(f"B{row}", name)
                created += 1; row += 1
        except Exception as e:  # noqa: BLE001
            failed.append(f"{name} = {sp['expr']}: {e}")
    doc.recompute()
    return sheet, created, updated, failed


def _has(sheet, cell):
    try:
        return sheet.getContents(cell) != ""
    except Exception:  # noqa: BLE001
        return False


def apply(doc, res, groups, name=None):
    """Import + parameters for a successful result.  Returns (feature, summary text)."""
    if not res["outcome"]["importable"]:
        return None, res["outcome"]["message"]
    base = name or ("MeshSolid_" + os.path.splitext(os.path.basename(res["step"]))[0])
    feat, msg = import_step(doc, res["step"], base)
    lines = [msg]
    if feat is not None:
        feat.addProperty("App::PropertyString", "MeshSolidReport", "MeshSolid", "engine report path")
        feat.MeshSolidReport = res["report_path"]
        specs = res["model"]["params"]
        if any(s["group"] in groups for s in specs):
            _, c, u, f = write_parameters(doc, specs, groups)
            lines.append(f"spreadsheet {SHEET_NAME}: {c} parameters added, {u} updated (they record the "
                         "design values; they do not drive the solid)")
            if f:
                lines.append("could not write:\n" + "\n".join(f[:10]))
    return feat, "\n".join(lines)

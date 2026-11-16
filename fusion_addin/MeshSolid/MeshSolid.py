"""MeshSolid: Fusion add-in (Windows, personal use).

Mesh body or STL/OBJ file -> meshsolid engine (its own Python, run as a separate process) -> STEP
imported back as one body, with an optional review dialog (outcome, deviation against its limits,
everything the engine flagged, preview image, which user parameters to create) before import.

Honesty rules carried over from the engine: nothing is imported unless the engine's report says ok
and its STEP exists; every failure dialog says what is known (stage, limits, log tail) and where the
job folder is.  The pure-Python parts (mslib/) are unit-tested outside Fusion; this file still needs
its first run inside Fusion (see README, "First test on Windows")."""
import os
import sys
import time
import traceback

import adsk.core
import adsk.fusion

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from mslib.engine_client import EngineSettings, EngineJob, EngineRun, BatchJob, batch_summary_text, read_result, selfcheck  # noqa: E402
from mslib.mesh_io import triangles_from_polygon_mesh, write_binary_stl  # noqa: E402
from mslib.review import review_model, review_html  # noqa: E402
from mslib.outcome import missing_measurements  # noqa: E402
from mslib import ui_options as U  # noqa: E402
from mslib import metrology_cmd as Metro  # noqa: E402

VERSION = U.VERSION
CMD_ID = "meshsolid_mesh_to_solid"
REVIEW_ID = "meshsolid_review"
EVENT_ID = "meshsolid_open_review"
RES = os.path.join(_HERE, "resources")              # icon folders (UserInterface_UM, "Command Icons")
WORKSPACE_ID = "FusionSolidEnvironment"             # Design workspace (Autodesk UserInterface_UM)
PANEL_NAME = f"MESHSOLID {VERSION}"
# (our panel id, tab ids to try, tab display names to fall back on).  "ToolsTab" is the Design workspace's
# UTILITIES tab (formerly TOOLS); "MeshTab" its MESH tab.  Tab ids are not published by Autodesk, so if
# an id is missing the tab is looked up by its displayed name, and the result is logged (addin_ui.log).
TAB_TARGETS = [("MeshSolidPanel_Utilities", ("ToolsTab", "UtilitiesTab"), ("UTILITIES", "TOOLS")),
               ("MeshSolidPanel_Mesh", ("MeshTab",), ("MESH",))]
ADDINS_PANEL_ID = "SolidScriptsAddinsPanel"         # UTILITIES > ADD-INS dropdown (kept as a fallback)
PANEL_ID = ADDINS_PANEL_ID                          # backwards-compatible name
_handlers = []
_state = {"checked": False, "pending": None, "event": None, "placed": [], "last": {}, "input_body": None}


def _app():
    return adsk.core.Application.get()


def _ui():
    return _app().userInterface


def _jobs_dir():
    root = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "meshsolid", "jobs",
                        time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(root, exist_ok=True)
    return root


def _export_mesh_body(body, path):
    """MeshBody -> binary STL in mm (Fusion works in cm)."""
    m = body.mesh
    coords = list(m.nodeCoordinatesAsDouble)

    def _arr(name, count_name):
        # Fusion raises InternalValidationError (return_size > 0) when an index array is empty,
        # e.g. quadNodeIndices on an all-triangle mesh, so check the count first and guard anyway.
        try:
            if count_name and not int(getattr(m, count_name, 0) or 0):
                return []
            return list(getattr(m, name, []) or [])
        except Exception:
            return []

    tris = []
    try:
        tris = triangles_from_polygon_mesh(coords, _arr("triangleNodeIndices", "triangleCount"),
                                           _arr("quadNodeIndices", "quadCount"),
                                           _arr("polygonNodeIndices", "polygonCount"),
                                           _arr("polygonNodeCounts", "polygonCount"))
    except Exception:
        tris = []
    if not tris:                                    # fall back to the display mesh (always triangles)
        dm = body.displayMesh
        coords = list(dm.nodeCoordinatesAsDouble); idx = list(dm.nodeIndices)
        tris = [tuple(idx[i:i + 3]) for i in range(0, len(idx), 3)]
    return write_binary_stl(path, coords, tris, scale=10.0)


def _parse_measure(text):
    """Parse the Measurements field: name=value pairs in mm, separated by ';' (or ',').
    Example accepted by the engine: bore_d=14.275"""
    out = {}
    for part in (text or "").replace(",", ";").split(";"):
        if part.strip():
            if "=" not in part:
                raise ValueError(f"measurement '{part.strip()}' is not name=value (e.g. bore_d=14.275)")
            k, v = part.split("=", 1)
            out[k.strip()] = float(v)
    return out


def _format_measure(d):
    return "; ".join(f"{k}={v}" for k, v in (d or {}).items())


def _prompt_measures(ui, specs):
    """Ask for each missing design measurement with Fusion's inputBox. Returns a dict, or None if cancelled."""
    out = {}
    for s in specs:
        prompt = (f"The engine could not find the {s['label'].lower()} on the scan.\n"
                  f"Enter the design value in millimetres (use a dot, e.g. 14.275).\n\n"
                  f"This is the same as typing {s['name']}=14.275 in Measurements.")
        title = "MeshSolid: " + s["label"]
        while True:
            try:
                ret = ui.inputBox(prompt, title, "")
            except Exception as e:  # noqa: BLE001
                ui.messageBox(f"MeshSolid: could not open an input box ({e}). "
                              f"Enter {s['name']}=14.275 in Measurements and run again.")
                return None
            cancelled, value = False, ret
            if isinstance(ret, (tuple, list)) and len(ret) >= 1:
                value = ret[0]
                cancelled = bool(ret[1]) if len(ret) > 1 else False
            if cancelled or value is None:
                return None
            raw = str(value).strip()
            if not raw:
                return None
            if "=" in raw:
                raw = raw.split("=", 1)[1].strip()
            raw = raw.replace(",", ".")
            try:
                out[s["name"]] = float(raw)
                break
            except ValueError:
                prompt = (f"'{value}' is not a number. Enter the {s['label'].lower()} in millimetres, "
                          f"e.g. 14.275.")
    return out


def _retry_with_prompted_measures(ui, res, measure):
    """If the engine failed for a missing required measurement, prompt for it and retry.
    Returns ('retry', updated), ('cancel', measure) or ('fail', measure)."""
    specs = missing_measurements(res.get("outcome"), res.get("report"), measure)
    if not specs:
        return "fail", measure
    added = _prompt_measures(ui, specs)
    if not added:
        return "cancel", measure
    updated = dict(measure)
    updated.update(added)
    return "retry", updated


def _show_failure(res):
    o = res["outcome"]
    msg = (f"{o['message']}\n\n" + (f"Next step: {o['hint']}\n\n" if o.get("hint") else "") +
           f"Job folder: {res['out_dir']}\nReport: {res['report_path']}\nEngine log: {res['log']}")
    _ui().messageBox(msg, "MeshSolid: " + o["title"])


def _count_bodies(design):
    return sum(c.bRepBodies.count for c in design.allComponents)


def _bodies_of(design):
    out = []
    for c in design.allComponents:
        for i in range(c.bRepBodies.count):
            out.append(c.bRepBodies.item(i))
    return out


def _part_name(res):
    raw = (res or {}).get("part_name") or ""
    if not raw:
        raw = os.path.splitext(os.path.basename((res or {}).get("step") or ""))[0]
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in raw).strip("_")
    return cleaned or "solid"


def _rename_bodies(bodies, name):
    label = f"MeshSolid_{name}"
    for i, b in enumerate(bodies):
        try:
            b.name = label if len(bodies) == 1 else f"{label}_{i + 1}"
        except Exception:  # noqa: BLE001
            pass
    return label if len(bodies) <= 1 else f"{label}_1..{len(bodies)}"


def _iter_collection(coll):
    if coll is None:
        return []
    n = getattr(coll, "count", None)
    if n is not None:
        try:
            return [coll.item(i) for i in range(int(n))]
        except Exception:  # noqa: BLE001
            return []
    try:
        return list(coll)
    except TypeError:
        return [coll]


def _log_import(lines):
    """Append import attempts to addin_ui.log. Failed methods belong here, not in the success dialog."""
    try:
        d = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "meshsolid")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "addin_ui.log"), "a", encoding="utf-8") as f:
            f.write(f"\nMeshSolid {VERSION} import {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("\n".join(lines) + "\n")
    except OSError:
        pass


def _png_size(path):
    """Width, height of a PNG from its IHDR chunk (no image library required)."""
    import struct
    with open(path, "rb") as f:
        if f.read(8) != b"\x89PNG\r\n\x1a\n":
            return None
        length = struct.unpack(">I", f.read(4))[0]
        if f.read(4) != b"IHDR" or length < 8:
            return None
        w, h = struct.unpack(">II", f.read(8))
        return int(w), int(h)


def _design_of_document(app, doc):
    if doc is None:
        return None
    try:
        return adsk.fusion.Design.cast(doc.products.itemByProductType("DesignProductType"))
    except Exception:  # noqa: BLE001
        pass
    try:
        if app.activeDocument == doc:
            return adsk.fusion.Design.cast(app.activeProduct)
    except Exception:  # noqa: BLE001
        pass
    return None


def _add_temp_bodies(design, bodies, name):
    """Persist temporary (or copied) BRep bodies in the current design's root.
    Parametric designs need a BaseFeature in edit mode; direct designs take the body alone.
    Official API: TemporaryBRepManager / BRepBodies.add / BaseFeature.startEdit+finishEdit."""
    root = design.rootComponent
    parametric = False
    try:
        parametric = design.designType == adsk.fusion.DesignTypes.ParametricDesignType
    except Exception:  # noqa: BLE001
        parametric = True
    baseFeat = None
    if parametric:
        baseFeat = root.features.baseFeatures.add()
        baseFeat.startEdit()
    added = []
    try:
        for body in bodies:
            if baseFeat is not None:
                new = root.bRepBodies.add(body, baseFeat)
            else:
                new = root.bRepBodies.add(body)
            if new is not None:
                added.append(new)
        _rename_bodies(added, name)
    finally:
        if baseFeat is not None:
            try:
                baseFeat.finishEdit()
            except Exception:  # noqa: BLE001
                pass
    return added


def _create_from_file_candidates(step):
    """createFromFile is documented for SMT, SMB, SAT, SAB; we still try the STEP first, then siblings."""
    out = [step]
    stem, _ = os.path.splitext(step)
    for ext in (".sat", ".sab", ".smt", ".smb"):
        alt = stem + ext
        if os.path.isfile(alt) and alt not in out:
            out.append(alt)
    return out


def _import_via_create_from_file(design, step, name):
    """TemporaryBRepManager.createFromFile -> BRepBodies.add on the current root. No extra component."""
    mgr = adsk.fusion.TemporaryBRepManager.get()
    last = None
    for path in _create_from_file_candidates(step):
        try:
            coll = mgr.createFromFile(path)
            temps = _iter_collection(coll)
            if not temps:
                last = f"createFromFile({os.path.basename(path)}) returned nothing"
                continue
            added = _add_temp_bodies(design, temps, name)
            if added:
                return added, os.path.basename(path)
            last = f"createFromFile({os.path.basename(path)}) produced bodies that could not be added"
        except Exception as e:  # noqa: BLE001
            last = f"createFromFile({os.path.basename(path)}): {e}"
    raise RuntimeError(last or "createFromFile failed")


def _import_via_temp_document(app, design, im, step, name):
    """Import the STEP into a new document, copy its BRep bodies into this design's root, then close it.
    This is the Part Design path: a Part cannot hold a second component, and importToTarget often fails."""
    orig_doc = None
    try:
        orig_doc = design.parentDocument
    except Exception:  # noqa: BLE001
        orig_doc = getattr(app, "activeDocument", None)
    opts = im.createSTEPImportOptions(step)
    try:
        opts.isViewFit = False
    except Exception:  # noqa: BLE001
        pass
    new_doc = im.importToNewDocument(opts)
    if new_doc is None:
        return [], None, "importToNewDocument returned nothing"
    try:
        src = _design_of_document(app, new_doc)
        if src is None:
            return [], new_doc, "new document has no Design"
        mgr = adsk.fusion.TemporaryBRepManager.get()
        temps = []
        for body in _bodies_of(src):
            try:
                temps.append(mgr.copy(body))
            except Exception:  # noqa: BLE001
                temps.append(body)
        if not temps:
            return [], new_doc, "new document had no BRep bodies"
        if orig_doc is not None:
            try:
                orig_doc.activate()
            except Exception:  # noqa: BLE001
                pass
        added = _add_temp_bodies(design, temps, name)
        if not added:
            return [], new_doc, "could not add copied bodies to this design"
        try:
            new_doc.close(False)
        except Exception as e:  # noqa: BLE001
            return added, new_doc, f"copied into this design, but could not close the temporary document: {e}"
        return added, None, None
    except Exception as e:  # noqa: BLE001
        return [], new_doc, str(e)


def _import_step(app, design, im, step, name="solid"):
    """Import a STEP into the current design's root whenever possible.
    Order: TemporaryBRepManager.createFromFile (no extra component), importToTarget2, importToTarget
    into the root, importToTarget into a new component, copy from a temporary new document, and
    finally leave a new document open. Returns (ok, how, errs, dest_design, added, extra_note).
    dest_design is the design that now holds the solid (this one, or the new document)."""
    errs = []
    root = design.rootComponent
    before = set(id(b) for b in _bodies_of(design))

    def _added_here():
        return [b for b in _bodies_of(design) if id(b) not in before]

    try:
        added = _import_via_create_from_file(design, step, name)
        if added[0]:
            return True, "createFromFile", errs, design, len(added[0]), None
    except Exception as e:  # noqa: BLE001
        errs.append(str(e))

    if hasattr(im, "importToTarget2"):
        try:
            r = im.importToTarget2(im.createSTEPImportOptions(step), root)
            if r is not None and getattr(r, "count", 1):
                found = _iter_collection(r) or _added_here()
                _rename_bodies(found, name)
                return True, "importToTarget2", errs, design, len(found) or 1, None
            errs.append("importToTarget2 returned nothing")
        except Exception as e:  # noqa: BLE001
            errs.append(f"importToTarget2: {e}")
    try:
        if im.importToTarget(im.createSTEPImportOptions(step), root):
            found = _added_here()
            _rename_bodies(found, name)
            return True, "importToTarget", errs, design, len(found) or 1, None
        errs.append("importToTarget returned False")
    except Exception as e:  # noqa: BLE001
        errs.append(f"importToTarget: {e}")
    try:
        occ = root.occurrences.addNewComponent(adsk.core.Matrix3D.create())
        occ.component.name = "MeshSolid"
        if im.importToTarget(im.createSTEPImportOptions(step), occ.component):
            found = _added_here()
            _rename_bodies(found, name)
            return True, "new_component", errs, design, len(found) or 1, None
        errs.append("importToTarget (new component) returned False")
    except Exception as e:  # noqa: BLE001
        errs.append(f"new component: {e}")

    leftover_doc, tried_new_doc = None, False
    try:
        added, leftover_doc, note = _import_via_temp_document(app, design, im, step, name)
        tried_new_doc = True
        if added:
            return True, "copy_from_new_document", errs, design, len(added), note
        if note:
            errs.append(f"copy from new document: {note}")
    except Exception as e:  # noqa: BLE001
        errs.append(f"copy from new document: {e}")

    if leftover_doc is None and not tried_new_doc:
        try:
            leftover_doc = im.importToNewDocument(im.createSTEPImportOptions(step))
            if leftover_doc is None:
                errs.append("importToNewDocument returned nothing")
        except Exception as e:  # noqa: BLE001
            errs.append(f"importToNewDocument: {e}")
            leftover_doc = None
    if leftover_doc is not None:
        dest = _design_of_document(app, leftover_doc) or adsk.fusion.Design.cast(app.activeProduct)
        n = _count_bodies(dest) if dest is not None else 1
        if dest is not None:
            _rename_bodies(_bodies_of(dest), name)
        return True, "new_document", errs, dest or design, n or 1, None
    return False, "failed", errs, design, 0, None


def _tag_report(design, body_label, report_path):
    """Remember the conversion report on the imported solid (Attributes), so Metrology can find the
    hand-entered features (e.g. a typed bore diameter) and leave them out of the inspection."""
    if not report_path:
        return
    try:
        for b in _bodies_of(design):
            if (b.name or "").startswith(body_label):
                b.attributes.add(Metro.ATTR_GROUP, Metro.ATTR_REPORT, str(report_path))
    except Exception:  # noqa: BLE001  (bookkeeping only)
        pass


def _import(res, groups, hide_body=None):
    """Import the STEP and create the user parameters of the ticked groups.  Returns a summary text;
    says plainly if the import gave other than one new body or a parameter could not be created.
    hide_body: the input mesh body to hide after a successful import (Output > Hide input mesh).
    Failed import attempts are written to addin_ui.log, not the success dialog."""
    app = _app()
    design = adsk.fusion.Design.cast(app.activeProduct)
    if design is None:
        return "No active Fusion design: open or create a design and run the command again."
    im = app.importManager
    name = _part_name(res)
    ok, how, errs, dest, added, extra = _import_step(app, design, im, res["step"], name)
    _log_import([f"step: {res['step']}", f"result: {how} ok={ok} bodies={added}"] +
                ([f"failed attempt: {e}" for e in errs] if errs else ["no earlier failures"]))
    lines = []
    body_label = f"MeshSolid_{name}"
    if how == "new_document":
        lines.append("Fusion would not import the solid into this design, so it was opened in a new document "
                     f"instead, as {body_label}. The user parameters below were created in that new document, "
                     f"not the original one. STEP: {res['step']}")
        ok, added = True, added or 1
        if dest is None:
            dest = design
    elif not ok:
        lines.append(f"Fusion reported that the STEP import failed ({res['step']}). "
                     "Open that STEP with File > Open to get the solid.")
        if errs:
            lines.append("Attempts: " + " | ".join(errs))
    elif added == 1:
        lines.append(f"Imported 1 solid body into this design as {body_label} ({how}).")
    else:
        lines.append(f"Warning: the import added {added} bodies, expected 1 ({how}). Check the browser.")
    _tag_report(dest or design, body_label, res.get("report_path"))
    if extra:
        lines.append(extra)
    made, skipped, failed = 0, [], []
    specs = [s for s in res["model"]["params"] if s["group"] in groups]
    ups = dest.userParameters if dest is not None else design.userParameters
    for s in specs:
        try:
            if ups.itemByName(s["name"]) is not None:
                skipped.append(s["name"]); continue
            ups.add(s["name"], adsk.core.ValueInput.createByString(s["expr"]), s["units"], s["comment"])
            made += 1
        except Exception as e:  # noqa: BLE001
            failed.append(f"{s['name']} = {s['expr']}: {e}")
    if specs:
        where = ""
        if how == "new_document":
            where = " in the new document"
        lines.append(f"User parameters created{where}: {made} of {len(specs)} (they record the design values; "
                     "they do not drive the imported body).")
    if skipped:
        lines.append(f"Already present, left unchanged: {', '.join(skipped[:10])}" + (" ..." if len(skipped) > 10 else ""))
    if failed:
        lines.append("Could not create:\n" + "\n".join(failed[:10]))
    if hide_body is not None and ok and added >= 1 and how != "new_document":
        try:
            hide_body.isLightBulbOn = False
            lines.append("Input mesh body hidden (turn its eye icon back on in the browser to show it).")
        except Exception as e:  # noqa: BLE001
            lines.append(f"Could not hide the input mesh body: {e}")
    elif hide_body is not None and how == "new_document":
        lines.append("Input mesh was left visible because the solid is in a new document.")
    return "\n".join(lines)


# ---- command dialog ------------------------------------------------------------------------------------
def _tip(inp, text, desc=None):
    try:
        inp.tooltip = text
        if desc:
            inp.tooltipDescription = desc
    except Exception:  # noqa: BLE001  (not every input type takes a tooltip in every Fusion build)
        pass
    return inp


def _group(ins, gid, name, expanded=True, tip=None):
    g = ins.addGroupCommandInput(gid, name)
    g.isExpanded = expanded
    if tip:
        _tip(g, tip)
    return g.children


def _dropdown(c, iid, name, table, selected_id, tip):
    dd = c.addDropDownCommandInput(iid, name, adsk.core.DropDownStyles.TextListDropDownStyle)
    for row in table:
        dd.listItems.add(row[1], row[0] == selected_id)
    return _tip(dd, tip)


def _text_button(c, iid, name, text):
    """Text-only push button: BoolValueCommandInput with no resourceFolder and its `text` set."""
    b = c.addBoolValueInput(iid, name, False, "", False)
    b.text = text
    return b


def _res(name):
    return os.path.join(RES, name)


def _build_dialog(cmd, last=None, timeout_s=5400):
    """Lay out the Mesh to Solid dialog: collapsible groups, tooltips, icon buttons."""
    last = last or {}
    cmd.okButtonText = "Convert to Solid"
    try:
        cmd.setDialogInitialSize(400, 720)
    except Exception:  # noqa: BLE001
        pass
    ins = cmd.commandInputs

    c = _group(ins, "g_input", "Input", tip="What to convert: a mesh body in this design or a mesh file.")
    sel = c.addSelectionInput("mesh", "Mesh body", "Select a mesh body in the canvas or browser")
    sel.addSelectionFilter("MeshBodies")
    sel.setSelectionLimits(0, 1)
    _tip(sel, "Mesh body to convert", "Click a mesh body. Leave empty to use a file instead.")
    _tip(_text_button(c, "browse_file", "Mesh file", "Browse..."),
         "Browse for an STL / OBJ / PLY file", "Used when no mesh body is selected.")
    f = c.addStringValueInput("file", "File", last.get("file", ""))
    f.isReadOnly = True
    _tip(f, "Chosen mesh file (use the Browse button above)")

    c = _group(ins, "g_part", "Part type", tip="Tells the engine which recognisers to use.")
    dd = _dropdown(c, "part", "Part type", U.PART_TYPES, last.get("part", "auto"), "Part type")
    _tip(dd, "Part type", "<br>".join(f"<b>{l}</b>: {t}" for _, l, t, _o in U.PART_TYPES))
    c.addTextBoxCommandInput("part_hint", "", U.by_id(U.PART_TYPES, last.get("part", "auto"))[2], 2, True)

    c = _group(ins, "g_quality", "Accuracy and speed")
    _dropdown(c, "accuracy", "Accuracy", U.ACCURACY, last.get("accuracy", "auto"),
              "Deviation gate the solid must pass before it can be imported")
    _dropdown(c, "device", "Compute", U.DEVICE, last.get("device", "auto"),
              "Where the heavy maths runs. Auto uses an NVIDIA GPU (CUDA) when one works, else the CPU.")
    sl = c.addIntegerSliderCommandInput("time_min", "Time limit (min)", U.TIME_MIN, U.TIME_MAX, False)
    sl.valueOne = int(last.get("time_min", U.default_minutes(timeout_s)))
    _tip(sl, "Stop the engine after this long",
         "Past the limit the engine stops and reports the stage it reached; nothing is imported.")

    c = _group(ins, "g_meas", "Measurements (optional)", expanded=False,
               tip="Caliper values the engine checks named dimensions against, as name=value in mm")
    _tip(c.addStringValueInput("measure", "Measured", last.get("measure", "")),
         "name=value pairs in millimetres, separated by ';'",
         "Format: bore_d=14.275  (also D3=40.02). If the impeller scan has no bore, type the design bore this way.")
    hint = c.addTextBoxCommandInput(
        "measure_hint", "",
        "Format: bore_d=14.275 &nbsp;(name=value in mm; separate several with ;)", 1, True)
    try:
        hint.isFullWidth = True
    except Exception:  # noqa: BLE001
        pass
    _tip(c.addStringValueInput("notes", "Notes", last.get("notes", "")), "Notes copied into the report (';' separated)")

    c = _group(ins, "g_output", "Output")
    _tip(c.addBoolValueInput("review", "Review before import", True, "", last.get("review", True)),
         "Show the result, deviation and parameters before importing")
    _tip(c.addBoolValueInput("params", "Create user parameters", True, "", last.get("params", True)),
         "Record recognised design values as ms_* user parameters",
         "They record values (gear, thread, impeller, snapped dimensions); they do not drive the body.")
    _tip(c.addBoolValueInput("hide_mesh", "Hide input mesh", True, "", last.get("hide_mesh", True)),
         "Hide the selected mesh body after a successful import")

    c = _group(ins, "g_adv", "Advanced", expanded=False)
    tol = c.addFloatSpinnerCommandInput("tol", "Fit tolerance (mm, 0 = auto)", "", 0.0, 1.0, 0.005,
                                        float(last.get("tol", 0.0)))
    _tip(tol, "Engine --tol", "0 = estimate from the scan noise (recommended).")
    _tip(_text_button(c, "browse_batch", "Batch folder", "Browse..."),
         "Convert every STL/OBJ/PLY in a folder instead", "Results go to the job folder; nothing is imported.")
    b = c.addStringValueInput("batch", "Folder", "")
    _tip(b, "Batch folder (clear it to convert a single mesh)")
    c.addTextBoxCommandInput("cmdline", "Engine", "", 2, True)
    _refresh(ins)
    return ins


def _selected_id(inp, table):
    try:
        item = inp.selectedItem
        return U.by_label(table, item.name)[0] if item is not None else table[0][0]
    except (KeyError, AttributeError):
        return table[0][0]


def _read_form(ins):
    """Every dialog value as plain Python (also what is remembered for the next run this session)."""
    g = ins.itemById
    return {"part": _selected_id(g("part"), U.PART_TYPES), "accuracy": _selected_id(g("accuracy"), U.ACCURACY),
            "device": _selected_id(g("device"), U.DEVICE), "time_min": int(g("time_min").valueOne),
            "tol": float(g("tol").value or 0.0), "measure": g("measure").value, "notes": g("notes").value,
            "review": bool(g("review").value), "params": bool(g("params").value),
            "hide_mesh": bool(g("hide_mesh").value), "file": g("file").value.strip().strip('"'),
            "batch": g("batch").value.strip().strip('"')}


def _refresh(ins):
    """Update the part hint and the engine command preview after any change."""
    try:
        f = _read_form(ins)
        ins.itemById("part_hint").formattedText = U.by_id(U.PART_TYPES, f["part"])[2]
        ins.itemById("cmdline").formattedText = U.summary(f["part"], f["accuracy"], f["device"], f["time_min"], f["tol"])
    except Exception:  # noqa: BLE001  (preview only; never block the dialog)
        pass


class _Changed(adsk.core.InputChangedEventHandler):
    def notify(self, args):
        try:
            inp, ins = args.input, args.inputs
            ui = _ui()
            if inp.id == "browse_file":
                fd = ui.createFileDialog()
                fd.title = "MeshSolid: choose a mesh file"
                fd.filter = "Mesh files (*.stl;*.obj;*.ply);;All files (*.*)"
                fd.isMultiSelectEnabled = False
                if fd.showOpen() == adsk.core.DialogResults.DialogOK:
                    ins.itemById("file").value = fd.filename
                    try:
                        ins.itemById("mesh").clearSelection()
                    except Exception:  # noqa: BLE001
                        pass
            elif inp.id == "browse_batch":
                fd = ui.createFolderDialog()
                fd.title = "MeshSolid: folder of meshes to convert"
                if fd.showDialog() == adsk.core.DialogResults.DialogOK:
                    ins.itemById("batch").value = fd.folder
            _refresh(ins)
        except Exception:  # noqa: BLE001
            _ui().messageBox("MeshSolid: " + traceback.format_exc())


class _Created(adsk.core.CommandCreatedEventHandler):
    def notify(self, args):
        try:
            cmd = args.command
            _build_dialog(cmd, _state.get("last"), EngineSettings.load().timeout_s)
            on_change = _Changed(); cmd.inputChanged.add(on_change); _handlers.append(on_change)
            on_exec = _Execute(); cmd.execute.add(on_exec); _handlers.append(on_exec)
        except Exception:  # noqa: BLE001
            _ui().messageBox("MeshSolid: " + traceback.format_exc())


def _progress_loop(ui, run, title, period):
    pd = ui.createProgressDialog(); pd.isCancelButtonShown = True
    pd.show(title, "engine: starting", 0, 1, 0)
    rc = run.poll()
    while rc is None:
        adsk.doEvents(); time.sleep(period)
        pd.message = f"engine ({run.elapsed():.0f} s): {run.last_stage()}"
        if pd.wasCancelled:
            run.cancel()
        rc = run.poll()
    pd.hide()
    return rc


class _Execute(adsk.core.CommandEventHandler):
    def notify(self, args):
        ui = _ui()
        try:
            ins = args.command.commandInputs
            form = _read_form(ins)
            _state["last"] = {k: v for k, v in form.items() if k != "batch"}
            s = EngineSettings.load()
            s.timeout_s = U.timeout_for(form["time_min"])
            if not _state["checked"]:
                ok, lines = selfcheck(s, quick=True)
                if not ok:
                    ui.messageBox("The meshsolid engine is not ready, so nothing was run:\n\n" + "\n".join(lines) +
                                  "\n\nRun INSTALL_MeshSolid.bat again, or fix settings.json next to the add-in.",
                                  "MeshSolid: engine not ready")
                    return
                _state["checked"] = True
            try:
                measure = _parse_measure(form["measure"])
            except ValueError as e:
                ui.messageBox(f"MeshSolid: {e}. Use name=value in millimetres, e.g. bore_d=14.275"); return
            opts = U.engine_options(form["part"], form["accuracy"], form["device"], form["tol"])
            out_dir = _jobs_dir()
            folder = form["batch"]
            if folder:
                if not os.path.isdir(folder):
                    ui.messageBox("MeshSolid: the batch folder does not exist."); return
                bj = BatchJob(folder, os.path.join(out_dir, "batch"), options=opts)
                if not bj.meshes():
                    ui.messageBox("MeshSolid: no STL/OBJ/PLY files in the batch folder, so nothing was run."); return
                run = EngineRun(bj, bj.settings(s))
                _progress_loop(ui, run, "MeshSolid batch", 0.5)
                ui.messageBox(batch_summary_text(bj), "MeshSolid batch")
                return
            sel = ins.itemById("mesh")
            body = None
            if sel.selectionCount == 1:
                body = adsk.fusion.MeshBody.cast(sel.selection(0).entity)
                mesh_path = os.path.join(out_dir, "mesh_body.stl")
                n = _export_mesh_body(body, mesh_path)
                if n == 0:
                    ui.messageBox("MeshSolid: the selected mesh body has no triangles, so nothing was run."); return
            else:
                mesh_path = form["file"]
                if not mesh_path or not os.path.exists(mesh_path):
                    ui.messageBox("MeshSolid: select a mesh body, or press Browse and pick a mesh file."); return
            notes = [t.strip() for t in form["notes"].split(";") if t.strip()]
            notes.append(f"addin {VERSION}: part={form['part']} accuracy={form['accuracy']} device={form['device']}")
            retries = 0
            while True:
                if retries:
                    out_dir = _jobs_dir()
                job = EngineJob(mesh_path, out_dir, measure, notes, options=opts)
                run = EngineRun(job, s)
                rc = _progress_loop(ui, run, "MeshSolid" + (" (retry)" if retries else ""), 0.25)
                res = read_result(job, rc, s.timeout_s)
                res["model"] = review_model(res["report"], res["outcome"])
                if res["outcome"]["importable"]:
                    break
                action, measure = _retry_with_prompted_measures(ui, res, measure)
                if action == "retry" and retries < 3:
                    _state.setdefault("last", {})["measure"] = _format_measure(measure)
                    retries += 1
                    continue
                if action == "cancel":
                    ui.messageBox(
                        "MeshSolid needs a design measurement to continue (the scan has no bore, for example).\n\n"
                        "Enter it in Measurements as name=value in millimetres, e.g. bore_d=14.275, and run again.",
                        "MeshSolid: measurement needed")
                    return
                _show_failure(res); return
            res["create_params"] = form["params"]
            try:
                res["part_name"] = body.name if body is not None else os.path.splitext(os.path.basename(mesh_path))[0]
            except Exception:  # noqa: BLE001
                res["part_name"] = os.path.splitext(os.path.basename(mesh_path))[0]
            _state["input_body"] = body if form["hide_mesh"] else None
            if form["review"]:
                _state["pending"] = res
                _app().fireCustomEvent(EVENT_ID, "")      # opens the review command after this one ends
            else:
                groups = {g["group"] for g in res["model"]["param_groups"] if g["default"]} if form["params"] else set()
                ui.messageBox(_import(res, groups, _state.pop("input_body", None)), "MeshSolid")
        except Exception:  # noqa: BLE001
            ui.messageBox("MeshSolid failed (add-in error, not an engine result):\n" + traceback.format_exc())


class _OpenReview(adsk.core.CustomEventHandler):
    def notify(self, args):
        try:
            _ui().commandDefinitions.itemById(REVIEW_ID).execute()
        except Exception:  # noqa: BLE001
            _ui().messageBox("MeshSolid: could not open the review:\n" + traceback.format_exc())


def _size_review_dialog(cmd):
    """Wide enough for the group labels and the 4-up preview (engine PNGs are ~2420x715)."""
    # Command.setDialogMinimumSize / setDialogInitialSize are the only sizing calls in the Fusion API.
    for fn, args in (("setDialogMinimumSize", (560, 640)),
                     ("setDialogInitialSize", (640, 780))):
        try:
            getattr(cmd, fn)(*args)
        except Exception:  # noqa: BLE001
            pass


def _add_review_preview(ins, path):
    """Show the preview at a size that fits the dialog: isFullWidth, plus scaleFactor from the PNG size."""
    img = ins.addImageCommandInput("preview", "", path)
    try:
        img.isFullWidth = True
    except Exception:  # noqa: BLE001
        pass
    try:
        wh = _png_size(path)
        if wh and wh[0] > 0:
            # Target width ~600 px so the 4-up strip is fully visible, not cropped.
            img.scaleFactor = min(1.0, 600.0 / float(wh[0]))
    except Exception:  # noqa: BLE001
        pass
    return img


class _ReviewCreated(adsk.core.CommandCreatedEventHandler):
    def notify(self, args):
        try:
            res = _state.get("pending")
            cmd = args.command
            cmd.okButtonText = "Import"
            _size_review_dialog(cmd)
            ins = cmd.commandInputs
            if res is None:
                ins.addTextBoxCommandInput("t", "", "Nothing to review: run Mesh to Solid first.", 2, True); return
            m = res["model"]
            summary = ins.addTextBoxCommandInput("summary", "Result", review_html(m), 14, True)
            try:
                summary.isFullWidth = True
            except Exception:  # noqa: BLE001
                pass
            if res.get("preview"):
                try:
                    _add_review_preview(ins, res["preview"])
                except Exception:  # noqa: BLE001  (very large PNGs may be refused)
                    ins.addTextBoxCommandInput("pv", "Preview", f"Preview image: {res['preview']}", 1, True)
            want = res.get("create_params", True)
            for g in m["param_groups"]:
                # Short labels so they are not truncated to "Param..." in the dialog.
                inp = ins.addBoolValueInput("grp_" + g["group"], f"{g['label']} ({g['count']})", True, "",
                                            bool(g["default"]) and want)
                _tip(inp, f"Create {g['label']} user parameters ({g['count']})")
            paths = ins.addTextBoxCommandInput("paths", "Files", f"Job folder: {res['out_dir']}", 2, True)
            try:
                paths.isFullWidth = True
            except Exception:  # noqa: BLE001
                pass
            h = _ReviewExecute(); cmd.execute.add(h); _handlers.append(h)
        except Exception:  # noqa: BLE001
            _ui().messageBox("MeshSolid review failed:\n" + traceback.format_exc())


class _ReviewExecute(adsk.core.CommandEventHandler):
    def notify(self, args):
        try:
            res = _state.pop("pending", None)
            if res is None:
                return
            ins = args.command.commandInputs
            groups = {g["group"] for g in res["model"]["param_groups"]
                      if ins.itemById("grp_" + g["group"]) and ins.itemById("grp_" + g["group"]).value}
            _ui().messageBox(_import(res, groups, _state.pop("input_body", None)), "MeshSolid")
        except Exception:  # noqa: BLE001
            _ui().messageBox("MeshSolid import failed:\n" + traceback.format_exc())


def _log_ui(lines):
    try:
        d = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "meshsolid")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "addin_ui.log"), "w", encoding="utf-8") as f:
            f.write(f"MeshSolid {VERSION} toolbar placement\n" + "\n".join(lines) + "\n")
    except OSError:
        pass


def _find_tab(ws, ids, names):
    """Toolbar tab by id, else by its displayed name (ids are not published; names are what users see)."""
    tabs = ws.toolbarTabs
    for i in ids:
        t = tabs.itemById(i)
        if t:
            return t, f"id {i}"
    for k in range(tabs.count):
        t = tabs.item(k)
        if (t.name or "").strip().upper() in names:
            return t, f"name {t.name!r} (id {t.id})"
    return None, None


def _place_ui(ui, *cmds):
    """Own MESHSOLID panel with promoted (large) buttons on the Design UTILITIES and MESH tabs, plus the
    old entries in the ADD-INS dropdown as a fallback.  Returns log lines."""
    log, placed = [], []
    ws = ui.workspaces.itemById(WORKSPACE_ID)
    if ws is None:
        log.append(f"workspace {WORKSPACE_ID} not found")
    for pid, ids, names in TAB_TARGETS if ws else []:
        try:
            tab, how = _find_tab(ws, ids, names)
            if tab is None:
                log.append(f"{pid}: no tab with id {ids} or name {names}"); continue
            panel = tab.toolbarPanels.itemById(pid) or tab.toolbarPanels.add(pid, PANEL_NAME)
            for cd in cmds:
                ctrl = panel.controls.itemById(cd.id) or panel.controls.addCommand(cd)
                ctrl.isPromotedByDefault = True
                ctrl.isPromoted = True
            placed.append(pid)
            log.append(f"{pid}: panel {PANEL_NAME!r} on tab {how}, buttons promoted: "
                       + ", ".join(cd.id for cd in cmds))
        except Exception as e:  # noqa: BLE001
            log.append(f"{pid}: failed: {e}")
    try:
        ap = ui.allToolbarPanels.itemById(ADDINS_PANEL_ID)
        for cd in cmds:
            if ap and not ap.controls.itemById(cd.id):
                ap.controls.addCommand(cd)
        log.append(f"{ADDINS_PANEL_ID}: dropdown entry " + ("added" if ap else "panel not found"))
    except Exception as e:  # noqa: BLE001
        log.append(f"{ADDINS_PANEL_ID}: failed: {e}")
    _state["placed"] = placed
    return log


def _remove_ui(ui):
    cmd_ids = [CMD_ID, Metro.METRO_ID]
    for pid in [t[0] for t in TAB_TARGETS] + [ADDINS_PANEL_ID]:
        try:
            panel = ui.allToolbarPanels.itemById(pid)
            if not panel:
                continue
            for cid in cmd_ids:
                c = panel.controls.itemById(cid)
                if c:
                    c.deleteMe()
            if pid != ADDINS_PANEL_ID:
                panel.deleteMe()
        except Exception:  # noqa: BLE001
            pass
    _state["placed"] = []


def run(context):
    try:
        ui = _ui()
        cd = ui.commandDefinitions.itemById(CMD_ID) or ui.commandDefinitions.addButtonDefinition(
            CMD_ID, "Mesh to Solid", f"MeshSolid {VERSION}: convert a mesh body or STL/OBJ/PLY file to one solid body",
            _res("MeshSolid"))
        h = _Created(); cd.commandCreated.add(h); _handlers.append(h)
        rd = ui.commandDefinitions.itemById(REVIEW_ID) or ui.commandDefinitions.addButtonDefinition(
            REVIEW_ID, "MeshSolid review", "Review the engine result before import", _res("MeshSolid"))
        h = _ReviewCreated(); rd.commandCreated.add(h); _handlers.append(h)
        ev = _app().registerCustomEvent(EVENT_ID)
        h = _OpenReview(); ev.add(h); _handlers.append(h); _state["event"] = ev
        md = ui.commandDefinitions.itemById(Metro.METRO_ID) or ui.commandDefinitions.addButtonDefinition(
            Metro.METRO_ID, "Metrology",
            f"MeshSolid {VERSION}: signed deviation of a scan against a solid (colour map, zones, probes)",
            _res("Metrology"))
        Metro.attach(md, _export_mesh_body, _jobs_dir, _progress_loop, _handlers)
        mrd = ui.commandDefinitions.itemById(Metro.METRO_RESULT_ID) or ui.commandDefinitions.addButtonDefinition(
            Metro.METRO_RESULT_ID, "MeshSolid metrology result", "Metrology result", _res("Metrology"))
        def _scale_preview(img, path):
            try:
                wh = _png_size(path)
                if wh and wh[0] > 0:
                    img.scaleFactor = min(1.0, Metro.RESULTS_W * 0.9 / float(wh[0]))
            except Exception:  # noqa: BLE001
                pass
        Metro.attach_results(mrd, _handlers, png_sizer=_scale_preview)
        _log_ui(_place_ui(ui, cd, md))
    except Exception:  # noqa: BLE001
        _ui().messageBox("MeshSolid run failed:\n" + traceback.format_exc())


def stop(context):
    try:
        ui = _ui()
        _remove_ui(ui)
        for i in (CMD_ID, REVIEW_ID, Metro.METRO_ID, Metro.METRO_RESULT_ID):
            cd = ui.commandDefinitions.itemById(i)
            if cd:
                cd.deleteMe()
        Metro.detach()
        if _state.get("event") is not None:
            _app().unregisterCustomEvent(EVENT_ID)
            _state["event"] = None
    except Exception:  # noqa: BLE001
        pass

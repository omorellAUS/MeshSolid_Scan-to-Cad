"""MeshSolid Metrology command: scan vs solid deviation after a conversion.

The engine does the work (`python -m meshsolid.metrology`). This module is the Fusion command dialog,
the export of the selected bodies, the import of per-band mesh bodies, and the results dialog.
Interactive point picking on a mesh is not used (unreliable in the API); probes are the N worst
vertices plus any coordinates typed in.

Flow (no silent failures; every error is written to %LOCALAPPDATA%\\meshsolid\\addin_ui.log and shown):
  commandCreated -> execute/inputChanged handlers are attached FIRST, then the dialog is built
  execute        -> reads the form, stores it, fires METRO_EVENT_ID (the command then ends)
  custom event   -> exports the inputs, runs the engine with a progress dialog, imports the coloured
                    band meshes (inside a BaseFeature in parametric designs), opens the results dialog.
Only Fusion API members that exist in Autodesk's API reference are used (tests/fusion_api_members.json).
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback

import adsk.core
import adsk.fusion

from .engine_client import EngineSettings, EngineRun, MetrologyJob
from .mesh_io import write_binary_stl
from .metrology_ui_defaults import (default_bands, parse_xyz_list, parse_section, band_json_dump, find_job_files,
                                    parse_zones)
from . import ui_options as U
from . import metrology_view as V

METRO_ID = "meshsolid_metrology"
METRO_RESULT_ID = "meshsolid_metrology_result"
PALETTE_ID = "meshsolid_metrology_results_palette"
PALETTE_W, PALETTE_H = 1150, 900
METRO_EVENT_ID = "meshsolid_metrology_run"
MAX_BANDS = 11
DEFAULT_BANDS = 7
EXCLUDE_TAGS = (
    ("backface", "Impeller back-face zone"),
    ("bore", "Impeller bore zone"),
    ("tip", "Outer rim + blade / tooth tips (r >= 92 % of outer radius)"),
)
EXCLUDE_TAG_TIPS = {
    "backface": "Flat faces at the back of an impeller (geometric zone rules). Listed with its own count.",
    "bore": "Scan near the impeller's bore radius (geometric zone rules). A hand-entered bore is excluded "
            "automatically by 'Exclude hand-entered features'.",
    "tip": "Outer-radius band (r >= 92 % of the outer radius). On an IMPELLER this removes the outer rim / blade "
           "tips, not the nose cone: to leave out the cone, pick it with 'Disregard faces'.",
}
# Fusion Appearance Library id (stable across versions; its display name changed from
# "Fusion 360 Appearance Library" to "Fusion Appearance Library").
APPEARANCE_LIB_ID = "BA5EE55E-9982-449B-9D66-9F036540E140"
APPEARANCE_LIB_NAMES = ("Fusion Appearance Library", "Fusion 360 Appearance Library")
APPEARANCE_SOURCES = ("Paint - Enamel Glossy (Yellow)", "Paint - Enamel Glossy (Green)",
                      "Plastic - Glossy (Yellow)", "Plastic - Matte (Yellow)")
SOLID_TESS_TOL_CM = 0.001          # 0.01 mm, same as the engine's STEP tessellation (--lin 0.01)
ATTR_GROUP, ATTR_REPORT = "MeshSolid", "report"   # set on imported solids by Mesh to Solid (Attributes.add)
DEFAULT_TITLE = "Metrology inspection report"

# Fusion has no colour-picker command input, so band colours are a dropdown of named swatches.
PALETTE = [
    ("Navy", (15, 20, 105)), ("Blue", (25, 60, 210)), ("Light blue", (60, 140, 255)),
    ("Teal", (0, 120, 120)), ("Cyan", (30, 210, 225)), ("Green", (36, 170, 65)),
    ("Light green", (140, 210, 90)), ("Yellow", (245, 215, 35)), ("Orange", (245, 140, 25)),
    ("Magenta", (215, 40, 190)), ("Purple", (110, 30, 180)), ("Brown", (110, 60, 25)),
    ("Red", (255, 0, 0)), ("Dark red", (140, 15, 25)), ("Pink", (250, 150, 200)),
    ("Grey", (150, 150, 154)), ("White", (245, 245, 245)), ("Black", (25, 25, 25)),
]


def _app():
    return adsk.core.Application.get()


def _ui():
    return _app().userInterface


# ---- logging and error reporting --------------------------------------------------------------------
def log_path():
    return os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "meshsolid", "addin_ui.log")


def log(title, text=""):
    """Append to addin_ui.log. Never raises."""
    try:
        p = log_path()
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] metrology: {title}\n")
            if text:
                f.write(text.rstrip() + "\n")
    except Exception:  # noqa: BLE001
        pass


def report_error(where, tb=None):
    """Full traceback to addin_ui.log AND a message box. Never raises."""
    tb = tb or traceback.format_exc()
    log(f"ERROR in {where}", tb)
    try:
        _ui().messageBox(f"MeshSolid metrology: {where} failed.\n\n{tb}\n\nThis was also written to {log_path()}",
                         "MeshSolid metrology")
    except Exception:  # noqa: BLE001
        pass


def tell(text, title="MeshSolid metrology"):
    log(title, text)
    try:
        _ui().messageBox(text, title)
    except Exception:  # noqa: BLE001
        pass


# ---- dialog -----------------------------------------------------------------------------------------
def _tip(inp, text, desc=None):
    inp.tooltip = text
    if desc:
        inp.tooltipDescription = desc
    return inp


def _group(ins, gid, name, expanded=True, tip=None):
    g = ins.addGroupCommandInput(gid, name)
    g.isExpanded = expanded
    if tip:
        _tip(g, tip)
    return g.children


def _color(r, g, b):
    return adsk.core.Color.create(int(r), int(g), int(b), 255)


def _size_dialog(cmd, w=640, h=860, min_w=560, min_h=640):
    # Command.setDialogMinimumSize / setDialogInitialSize are the only sizing calls in the API.
    try:
        cmd.setDialogMinimumSize(min_w, min_h)
        cmd.setDialogInitialSize(w, h)
    except Exception:  # noqa: BLE001  (cosmetic; logged, never blocks the dialog)
        log("dialog sizing ignored", traceback.format_exc())


def _nearest_palette(rgb):
    r, g, b = [int(v) for v in rgb]
    return min(range(len(PALETTE)),
               key=lambda k: sum((a - c) ** 2 for a, c in zip(PALETTE[k][1], (r, g, b))))


def _set_color(inp, rgb):
    """Select the palette swatch nearest to rgb (a text dropdown selects one item at a time)."""
    k = _nearest_palette(rgb)
    inp.listItems.item(k).isSelected = True
    return PALETTE[k][0]


def _read_color(inp):
    try:
        item = inp.selectedItem
        if item is not None:
            for n, rgb in PALETTE:
                if n == item.name:
                    return list(rgb)
    except Exception:  # noqa: BLE001
        log("could not read a band colour", traceback.format_exc())
    return [128, 128, 128]


def _clamp_bands(n):
    try:
        n = int(n)
    except (TypeError, ValueError):
        n = DEFAULT_BANDS
    return max(3, min(MAX_BANDS, n))


def fill_band_defaults(ins, vmin, vmax, pass_tol, n_bands, bands=None):
    """Show n_bands rows with default (or given) limits and colours; hide the rest."""
    n_bands = _clamp_bands(n_bands)
    if not bands or len(bands) != n_bands:
        bands = default_bands(vmin, vmax, pass_tol, n_bands)
    n = min(len(bands), MAX_BANDS)
    for i in range(MAX_BANDS):
        lo, hi, col = ins.itemById(f"blo_{i}"), ins.itemById(f"bhi_{i}"), ins.itemById(f"bcol_{i}")
        on = i < n
        for inp in (lo, hi, col):
            if inp is not None:
                inp.isVisible = on
        if not on or lo is None:
            continue
        b = bands[i]
        lo.value = float(b["lo"])
        hi.value = float(b["hi"])
        if col is not None:
            _set_color(col, b.get("rgb") or (128, 128, 128))
    return bands


def build_dialog(cmd, last=None):
    last = last or {}
    cmd.okButtonText = "Compute metrology"
    _size_dialog(cmd)
    ins = cmd.commandInputs

    c = _group(ins, "g_in", "Inputs",
               tip="The scan and the converted solid. MeshSolid builds the solid in the scan's coordinates, so they already line up.")
    mesh = c.addSelectionInput("scan", "Scan mesh", "Select the original scan mesh body")
    mesh.addSelectionFilter("MeshBodies")
    mesh.setSelectionLimits(0, 1)
    _tip(mesh, "Scan mesh body", "Leave empty and use a job folder if you prefer.")
    sol = c.addSelectionInput("solid", "Solid body", "Select the converted solid BRep body")
    sol.addSelectionFilter("SolidBodies")
    sol.setSelectionLimits(0, 1)
    _tip(sol, "Converted solid", "The BRep MeshSolid imported. Not moved.")
    b = c.addBoolValueInput("browse_job", "Job folder", False, "", False)
    b.text = "Browse job folder..."
    _tip(b, "Use a previous MeshSolid job", "Auto-fills the scan STL and the STEP from that folder.")
    job = c.addStringValueInput("job", "Folder", last.get("job", ""))
    job.isReadOnly = True
    _tip(c.addBoolValueInput("align", "Best-fit align (spec v2)", True, "", bool(last.get("align", True))),
         "Spec-v2 best fit (on by default)",
         "Bounded point-to-plane best fit (rotation limited to half a blade / tooth pitch) plus a final refinement on the signed deviations, as in measurement spec v2. The original bodies are never moved and the coloured map is written back in the scan's own position.")

    c = _group(ins, "g_rep", "Report",
               tip="Header of the PDF inspection report (report.pdf, always written to the job folder).")
    _tip(c.addStringValueInput("title", "Report title", last.get("title", DEFAULT_TITLE)), "Title on every page")
    _tip(c.addStringValueInput("part_name", "Part name", last.get("part_name", "")),
         "Part name", "Leave empty to use the solid body's name (or the job folder's).")
    _tip(c.addStringValueInput("operator", "Operator", last.get("operator", "")), "Who did the inspection")
    ap = c.addFloatSpinnerCommandInput("accept_pct", "PASS if within band ≥ (%)", "", 0.0, 100.0, 1.0,
                                       float(last.get("accept_pct", 95.0)))
    _tip(ap, "PASS / FAIL rule", "PASS when at least this percentage of the inspected scan vertices is inside the pass band.")

    vmin0, vmax0 = float(last.get("vmin", -0.2)), float(last.get("vmax", 0.2))
    pass0, nb0 = float(last.get("pass_tol", 0.05)), _clamp_bands(last.get("n_bands", DEFAULT_BANDS))
    c = _group(ins, "g_cmap", "Colour map")
    c.addFloatSpinnerCommandInput("vmin", "Scale min (mm)", "", -10.0, 10.0, 0.01, vmin0)
    c.addFloatSpinnerCommandInput("vmax", "Scale max (mm)", "", -10.0, 10.0, 0.01, vmax0)
    _tip(c.addBoolValueInput("symmetric", "Symmetric scale", True, "", bool(last.get("symmetric", True))),
         "Centre the scale on zero")
    nb = c.addIntegerSpinnerCommandInput("n_bands", "Number of bands", 3, MAX_BANDS, 1, nb0)
    _tip(nb, "How many colour bands in the legend and the imported mesh")
    pt = c.addFloatSpinnerCommandInput("pass_tol", "Pass band ± (mm)", "", 0.0, 10.0, 0.005, pass0)
    _tip(pt, "Vertices within ± this of zero are the green in-tolerance band")
    rb = c.addBoolValueInput("reset_bands", "Reset bands", False, "", False)
    rb.text = "Reset bands to default"
    _tip(rb, "Rebuild band limits and colours from the scale, pass band and band count")

    c = _group(ins, "g_bands", "Tolerance bands",
               tip="Each band has its own limits and colour. Default: navy → blue → cyan → green (in tol) → yellow → orange → magenta. "
                   "Red is kept for points beyond the outlier cap; colours that look alike are made distinct automatically.")
    for i in range(MAX_BANDS):
        c.addFloatSpinnerCommandInput(f"blo_{i}", f"Band {i + 1} from (mm)", "", -10.0, 10.0, 0.005, 0.0)
        c.addFloatSpinnerCommandInput(f"bhi_{i}", f"Band {i + 1} to (mm)", "", -10.0, 10.0, 0.005, 0.0)
        dd_c = c.addDropDownCommandInput(f"bcol_{i}", f"Band {i + 1} colour", adsk.core.DropDownStyles.TextListDropDownStyle)
        for cname, _rgb in PALETTE:
            dd_c.listItems.add(cname, cname == "Green", "")
    fill_band_defaults(ins, vmin0, vmax0, pass0, nb0, last.get("bands"))

    c = _group(ins, "g_excl", "Exclusions (not inspected)", expanded=False,
               tip="Excluded scan vertices are grey on the map, left out of the statistics and listed in the PDF.")
    _tip(c.addBoolValueInput("auto_excl", "Exclude hand-entered features", True, "", bool(last.get("auto_excl", True))),
         "Hand-entered features are not inspected",
         "Features built from a typed value (e.g. bore_d=14.275 when the scan has no bore) cannot be checked against the scan. Their region is excluded and listed as 'not inspected: hand-entered'.")
    fsel = c.addSelectionInput("faces", "Disregard faces", "Select faces of the solid to leave out")
    fsel.addSelectionFilter("SolidFaces")
    fsel.setSelectionLimits(0, 0)
    _tip(fsel, "Disregard faces", "Scan vertices whose nearest solid point lies on these faces are not inspected.")
    zt = c.addStringValueInput("zones", "Excluded zones", last.get("zones", ""))
    _tip(zt, "Typed exclusion zones (mm)",
         "box(xmin,ymin,zmin, xmax,ymax,zmax) or cyl(px,py,pz, dx,dy,dz, r[, t0,t1]); several separated by ';'. "
         "For cyl, t0..t1 is the range along the axis from the point (leave out for the full length). "
         "Example: cyl(0,0,0, 0,0,1, 7.2); box(-5,-5,30, 5,5,40)")
    for tag, label in EXCLUDE_TAGS:
        _tip(c.addBoolValueInput("ex_" + tag, label, True, "", bool(tag in (last.get("exclude") or []))),
             f"Exclude the '{tag}' zone from the statistics (grey on the map)", EXCLUDE_TAG_TIPS.get(tag, ""))
    md = c.addFloatSpinnerCommandInput("max_dist", "Outlier cap |d| (mm, 0 = off)", "", 0.0, 100.0, 0.05,
                                       float(last.get("max_dist") or 0.0))
    _tip(md, "Outlier cap: leave far points out of the statistics",
         "Points farther than this from the solid (either sign) are NOT counted in rms / p95 / % within / PASS-FAIL. "
         "They are listed as 'beyond outlier cap' with their count and %, drawn in the cap colour (own mesh body), "
         "and the true max and the worst-points table still show their real deviations. "
         "This does not change the colour scale (that is the Range above). 0 = off: every point is counted.")
    ccap = c.addDropDownCommandInput("cap_color", "Beyond-cap colour", adsk.core.DropDownStyles.TextListDropDownStyle)
    cap_name = last.get("cap_color_name") or "Red"
    for cname, _rgb in PALETTE:
        ccap.listItems.add(cname, cname == cap_name, "")
    _tip(ccap, "Colour of points beyond the outlier cap", "Default red. Grey is reserved for faces / zones that are not inspected.")

    c = _group(ins, "g_an", "Analysis")
    _tip(c.addBoolValueInput("do_map", "Full-surface deviation map", True, "", bool(last.get("do_map", True))),
         "Shaded colour map, imported as one mesh body per colour band")
    _tip(c.addBoolValueInput("do_zones", "Per-zone statistics", True, "", bool(last.get("do_zones", True))),
         "Impeller: blades, hub, front face, back face, bore. Gears: flank, tip, end face. Skipped honestly if the part is not recognised.")
    _tip(c.addBoolValueInput("do_probes", "Point probes / annotations", True, "", bool(last.get("do_probes", True))),
         "Auto-labels the N worst vertices. Type extra coordinates below. The Fusion API has no reliable click-to-probe on a mesh.")
    np_ = c.addIntegerSpinnerCommandInput("n_probes", "Worst points to label", 0, 50, 1,
                                          int(last.get("n_probes", 10) or 10))
    _tip(np_, "Largest |deviation| vertices, written on the report and in the results")
    pr = c.addStringValueInput("probe_xyz", "Extra probes (mm)", last.get("probe_xyz", ""))
    _tip(pr, "Typed coordinates", "x,y,z in millimetres, several separated by ';'. Example: 0,0,14.3; 10,0,0")
    _tip(c.addBoolValueInput("do_section", "Section / cross-section", True, "", bool(last.get("do_section", False))),
         "Deviation along a plane. Type the origin and normal below, or leave blank to skip.")
    so = c.addStringValueInput("section_origin", "Section origin (mm)", last.get("section_origin", "0,0,0"))
    sn = c.addStringValueInput("section_normal", "Section normal", last.get("section_normal", "0,0,1"))
    _tip(so, "A point on the section plane, millimetres")
    _tip(sn, "Plane normal, e.g. 0,0,1 for a Z section")
    _tip(c.addBoolValueInput("do_hist", "Histogram", True, "", bool(last.get("do_hist", True))),
         "Deviation histogram on the report PNG")

    c = _group(ins, "g_dev", "Compute", expanded=False)
    dd = c.addDropDownCommandInput("device", "Compute", adsk.core.DropDownStyles.TextListDropDownStyle)
    for row in U.DEVICE:
        dd.listItems.add(row[1], row[0] == last.get("device", "auto"), "")
    _tip(dd, "Where the distance query runs. Auto uses an NVIDIA GPU (CUDA / Warp) when one works.")
    return ins


def read_form(ins):
    g = ins.itemById

    def _sel_all(iid):
        s = g(iid)
        out = []
        if s is None:
            return out
        for k in range(s.selectionCount):
            out.append(s.selection(k).entity)
        return out

    def _sel(iid):
        s = g(iid)
        try:
            return s.selection(0).entity if s is not None and s.selectionCount == 1 else None
        except Exception:  # noqa: BLE001
            log(f"could not read selection {iid}", traceback.format_exc())
            return None

    def _bool(iid, default=False):
        inp = g(iid)
        return bool(inp.value) if inp is not None else default

    def _num(iid, default=0.0):
        inp = g(iid)
        return float(inp.value) if inp is not None else float(default)

    def _int(iid, default=0):
        inp = g(iid)
        return int(inp.value) if inp is not None else int(default)

    def _str(iid, default=""):
        inp = g(iid)
        return (inp.value or "").strip() if inp is not None else default

    n_bands = _clamp_bands(_int("n_bands", DEFAULT_BANDS))
    bands = []
    for i in range(n_bands):
        rgb = _read_color(g(f"bcol_{i}"))
        bands.append({"lo": _num(f"blo_{i}"), "hi": _num(f"bhi_{i}"), "rgb": rgb,
                      "color": "#%02x%02x%02x" % tuple(rgb)})
    device = "auto"
    dev = g("device")
    if dev is not None and dev.selectedItem is not None:
        try:
            device = U.by_label(U.DEVICE, dev.selectedItem.name)[0]
        except KeyError:
            device = "auto"
    exclude = [tag for tag, _ in EXCLUDE_TAGS if _bool("ex_" + tag)]
    return {
        "scan": _sel("scan"), "solid": _sel("solid"), "job": _str("job"),
        "align": _bool("align"), "vmin": _num("vmin", -0.2), "vmax": _num("vmax", 0.2),
        "symmetric": _bool("symmetric", True), "n_bands": n_bands,
        "pass_tol": _num("pass_tol", 0.05), "bands": bands,
        "exclude": exclude, "max_dist": _num("max_dist", 0.0),
        "cap_color": _read_color(g("cap_color")) if g("cap_color") is not None else [255, 0, 0],
        "cap_color_name": (g("cap_color").selectedItem.name if g("cap_color") is not None and
                           g("cap_color").selectedItem is not None else "Red"),
        "do_map": _bool("do_map", True), "do_zones": _bool("do_zones", True),
        "do_probes": _bool("do_probes", True), "n_probes": _int("n_probes", 10) if _bool("do_probes", True) else 0,
        "probe_xyz": _str("probe_xyz"),
        "do_section": _bool("do_section"), "section_origin": _str("section_origin", "0,0,0"),
        "section_normal": _str("section_normal", "0,0,1"),
        "do_hist": _bool("do_hist", True), "device": device,
        "faces": _sel_all("faces"), "zones": _str("zones"), "auto_excl": _bool("auto_excl", True),
        "title": _str("title", DEFAULT_TITLE) or DEFAULT_TITLE, "part_name": _str("part_name"),
        "operator": _str("operator"), "accept_pct": _num("accept_pct", 95.0),
    }


def form_options(form, band_json_path=None):
    """Map the dialog to MetrologyJob.options (CLI flags)."""
    probes = parse_xyz_list(form.get("probe_xyz") or "")
    section = None
    if form.get("do_section"):
        try:
            section = parse_section(form.get("section_origin") or "0,0,0",
                                    form.get("section_normal") or "0,0,1")
        except ValueError:
            section = None
    return {
        "range": [form["vmin"], form["vmax"]],
        "symmetric": form["symmetric"],
        "pass_tol": form["pass_tol"],
        "bands": form["n_bands"],
        "band_json": band_json_path,
        "exclude": form.get("exclude") or [],
        "max_dist": form["max_dist"] or None,
        "cap_color": "#%02x%02x%02x" % tuple(int(x) for x in (form.get("cap_color") or (255, 0, 0))[:3]),
        "align": form["align"],
        "device": form["device"],
        "n_probes": form["n_probes"],
        "probe": probes,
        "section": section,
        "no_map": not form["do_map"],
        "no_zones": not form["do_zones"],
        "no_histogram": not form["do_hist"],
        "no_section": not form["do_section"] or section is None,
        "zones": form.get("zones") or "",
        "auto_exclude": bool(form.get("auto_excl", True)),
        "accept_pct": form.get("accept_pct", 95.0),
        "title": form.get("title") or DEFAULT_TITLE,
        "part_name": form.get("part_name") or "",
        "operator": form.get("operator") or "",
        "exclude_faces": list(form.get("exclude_faces") or []),
        "convert_report": form.get("convert_report"),
    }


class Changed(adsk.core.InputChangedEventHandler):
    _busy = False                     # setting values below can fire inputChanged again

    def notify(self, args):
        if Changed._busy:
            return
        Changed._busy = True
        try:
            inp, ins = args.input, args.inputs
            if inp.id == "browse_job":
                fd = _ui().createFolderDialog()
                fd.title = "MeshSolid: job folder (scan + STEP)"
                if fd.showDialog() == adsk.core.DialogResults.DialogOK:
                    ins.itemById("job").value = fd.folder
                    ins.itemById("scan").clearSelection()
                    ins.itemById("solid").clearSelection()
            elif inp.id in ("vmin", "vmax", "symmetric", "n_bands", "pass_tol", "reset_bands"):
                vmin, vmax = float(ins.itemById("vmin").value), float(ins.itemById("vmax").value)
                if ins.itemById("symmetric").value:
                    m = max(abs(vmin), abs(vmax), 1e-6)
                    if ins.itemById("vmin").value != -m:
                        ins.itemById("vmin").value = -m
                    if ins.itemById("vmax").value != m:
                        ins.itemById("vmax").value = m
                    vmin, vmax = -m, m
                fill_band_defaults(ins, vmin, vmax, float(ins.itemById("pass_tol").value),
                                   int(ins.itemById("n_bands").value))
        except Exception:  # noqa: BLE001
            report_error("updating the dialog")
        finally:
            Changed._busy = False


# ---- results ----------------------------------------------------------------------------------------
def _num_txt(v, nd=4, signed=False):
    try:
        return f"{float(v):+.{nd}f}" if signed else f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return "—"


def summary_html(rep):
    """The key result, large and bold, for the top of the results window (Qt rich text: font-size works)."""
    st = rep.get("stats") or {}
    from .metrology_ui_defaults import verdict_html, warnings_html
    sm = st.get("spec_metrics") or {}
    return (warnings_html(st) + verdict_html(st) + "<br>"
            f"<span style='font-size:13pt'>{sm.get('pass_pct_v1', '—')} % within ±{st.get('pass_tol_mm', '—')} mm with "
            f"beyond-cap points as failures (target ≥ 90 %) &nbsp;·&nbsp; {st.get('within_pass_pct', '—')} % with them "
            f"dropped (1.0.1 needs ≥ {st.get('accept_pct', 95):g} %)</span><br>"
            + (f"<span style='font-size:16pt; font-weight:bold'>rms uncapped {_num_txt(sm.get('rms_uncapped'))} mm</span>"
               f"<span style='font-size:13pt'> &nbsp; (beyond-cap dropped {_num_txt(st.get('rms'))} mm)</span><br>"
               if sm else f"<span style='font-size:16pt; font-weight:bold'>rms {_num_txt(st.get('rms'))} mm</span><br>")
            +
            f"<span style='font-size:14pt'>p95 {_num_txt(st.get('p95'))} mm</span><br>"
            f"<span style='font-size:14pt'>mean {_num_txt(st.get('mean'), signed=True)} mm &nbsp;&nbsp; "
            f"max {_num_txt(st.get('max_plus'), 3, True)} / {_num_txt(st.get('max_minus'), 3, True)} mm"
            + (f" (inspected) &nbsp; true max {_num_txt(st.get('true_max_plus'), 3, True)} / "
               f"{_num_txt(st.get('true_max_minus'), 3, True)} mm" if st.get("cap_mm") else "") + "</span><br>"
            + (f"<span style='font-size:13pt; color:#b3261e'><b>{st.get('n_beyond_cap', 0):,}</b> points "
               f"({st.get('pct_beyond_cap', 0)} %) beyond the {st['cap_mm']:g} mm outlier cap, left out of the stats"
               "</span><br>" if st.get("cap_mm") else "") +
            f"<span style='font-size:13pt'>{st.get('n', 0):,} vertices inspected, "
            f"{st.get('n_excluded', 0):,} not inspected</span>")


def results_html(rep):
    import html as H
    st = rep.get("stats") or {}
    e = H.escape
    h = "<span style='font-size:14pt; font-weight:bold'>{}</span>"
    body = "<span style='font-size:13pt'>{}</span>"
    parts = []
    nin = rep.get("not_inspected") or []
    if nin:
        parts.append(h.format("Not inspected: where the excluded points come from") + "<br>" + body.format(
            "<br>".join(f"• <b>{e(x.get('label') or x['reason'])}</b> ({e(x['reason'])}): {x.get('n_new', x['n']):,} counted"
                        f" ({x['n']:,} matched) — {e(x['detail'])}" for x in nin)))
    z = st.get("zones") or rep.get("zones") or {}
    if z:
        lines = [f"• {e(k)}: rms {_num_txt(v['rms'])} mm, p95 {_num_txt(v['p95'])} mm, n={v['n']:,}"
                 for k, v in z.items() if v]
        parts.append(h.format("Zones") + "<br>" + body.format("<br>".join(lines)))
    if rep.get("probes"):
        lines = [f"• {p['kind']} ({', '.join(f'{x:.2f}' for x in p['xyz_mm'])}): {p['deviation_mm']:+.4f} mm"
                 for p in rep["probes"][:10]]
        parts.append(h.format("Worst points / probes") + "<br>" + body.format("<br>".join(e(x) for x in lines)))
    if rep.get("notes"):
        parts.append(h.format("Notes") + "<br>" + "<span style='font-size:11pt'>" +
                     "<br>".join("• " + e(str(n)) for n in rep["notes"]) + "</span>")
    return "<br><br>".join(parts)


def results_text(rep):
    import re
    return re.sub(r"<span[^>]*>|</span>", "", summary_html(rep) + "<br><br>" + results_html(rep)
                  ).replace("&nbsp;", " ").replace("<br>", "\n").replace("<b>", "").replace("</b>", "")



# ---- appearance and band-mesh import ------------------------------------------------------------------
def _appearance_library(app):
    libs = app.materialLibraries
    lib = libs.itemById(APPEARANCE_LIB_ID)
    if lib is not None:
        return lib
    for name in APPEARANCE_LIB_NAMES:
        lib = libs.itemByName(name)
        if lib is not None:
            return lib
    return None


def _color_property(appear):
    props = appear.appearanceProperties
    for key in ("Color", "color"):
        p = props.itemByName(key)
        if p is not None:
            return p
    for key in ("opaque_albedo", "generic_diffuse"):
        p = props.itemById(key)
        if p is not None:
            return p
    return None


def band_appearance(design, rgb, name):
    """A design appearance of colour rgb named name (created once by copying a library paint)."""
    appear = design.appearances.itemByName(name)
    if appear is None:
        lib = _appearance_library(_app())
        if lib is None:
            raise RuntimeError("the Fusion Appearance Library is not loaded")
        src = None
        for src_name in APPEARANCE_SOURCES:
            src = lib.appearances.itemByName(src_name)
            if src is not None:
                break
        if src is None:
            raise RuntimeError("no paint appearance found in the Fusion Appearance Library")
        appear = design.appearances.addByCopy(src, name)
    prop = _color_property(appear)
    if prop is None:
        raise RuntimeError(f"appearance {appear.name!r} has no colour property")
    prop.value = _color(*rgb)
    return appear


def _is_parametric(design):
    try:
        return design.designType == adsk.fusion.DesignTypes.ParametricDesignType
    except Exception:  # noqa: BLE001
        return True


def import_band_meshes(design, band_files):
    """Add each colour-band STL (mm) as a mesh body on the root component. Existing bodies are not moved.
    Parametric designs need a BaseFeature in edit mode for MeshBodies.add (API reference)."""
    root = design.rootComponent
    unit = adsk.fusion.MeshUnits.MillimeterMeshUnit
    notes, added = [], []
    files = [b for b in (band_files or []) if b.get("path") and os.path.isfile(b["path"])]
    if not files:
        return 0, ["The engine wrote no band meshes to import."]
    base = None
    if _is_parametric(design):
        base = root.features.baseFeatures.add()
        try:
            base.name = "MeshSolid metrology"
        except Exception:  # noqa: BLE001  (cosmetic)
            pass
        base.startEdit()
    try:
        for b in files:
            try:
                if base is not None:
                    lst = root.meshBodies.add(b["path"], unit, base)
                else:
                    try:
                        lst = root.meshBodies.add(b["path"], unit)
                    except TypeError:          # bindings that want the optional argument spelled out
                        lst = root.meshBodies.add(b["path"], unit, None)
                if lst is None or lst.count == 0:
                    notes.append(f"Fusion imported nothing from {os.path.basename(b['path'])}")
                    continue
                body = lst.item(0)
                if b.get("beyond_cap"):
                    cap = b.get("cap_mm")
                    body.name = "Metrology_beyond_cap" + (f" (|d| > {float(cap):.3f} mm)" if cap else "")
                elif b.get("excluded") or b.get("lo") is None:
                    body.name = "Metrology_excluded (not inspected)"
                else:
                    body.name = f"Metrology_{float(b['lo']):+.3f}_{float(b['hi']):+.3f}"
                added.append((body, b))
            except Exception as e:  # noqa: BLE001
                log(f"mesh import failed: {b['path']}", traceback.format_exc())
                notes.append(f"could not import {os.path.basename(b['path'])}: {e}")
    finally:
        if base is not None:
            base.finishEdit()
    for body, b in added:
        try:
            body.appearance = band_appearance(design, b.get("rgb") or [128, 128, 128],
                                              f"MeshSolid metrology {b.get('color', 'band')}")
        except Exception as e:  # noqa: BLE001
            log(f"appearance not applied to {b['path']}", traceback.format_exc())
            notes.append(f"{body.name}: colour not applied ({e})")
    return len(added), notes


# ---- exporting the selected bodies -------------------------------------------------------------------
def _is_identity(occ):
    try:
        m = occ.transform2 if hasattr(occ, "transform2") else occ.transform
        return m is None or m.isEqualTo(adsk.core.Matrix3D.create())
    except Exception:  # noqa: BLE001
        return False


def export_solid(body, out_dir):
    """Selected BRep body -> a file the engine reads, in mm. Returns (path, how).
    STEP when the body is alone in an untransformed component (ExportManager STEP export takes a
    Component, not a body); otherwise an STL tessellated at 0.01 mm with the body's own mesh
    calculator (unit-safe: Fusion returns cm, written x10)."""
    body = adsk.fusion.BRepBody.cast(body) or body
    errors = []
    try:
        comp = body.parentComponent
        ctx = body.assemblyContext
        if (comp is not None and comp.bRepBodies.count == 1 and comp.occurrences.count == 0
                and (ctx is None or _is_identity(ctx))):
            path = os.path.join(out_dir, "solid.step")
            em = comp.parentDesign.exportManager
            opts = em.createSTEPExportOptions(path, comp)
            if em.execute(opts) and os.path.isfile(path):
                return path, "STEP"
            errors.append("STEP export returned False")
    except Exception as e:  # noqa: BLE001
        errors.append(f"STEP export: {e}")
        log("STEP export of the solid failed, tessellating instead", traceback.format_exc())
    calc = body.meshManager.createMeshCalculator()
    calc.setQuality(adsk.fusion.TriangleMeshQualityOptions.VeryHighQualityTriangleMesh)
    calc.surfaceTolerance = SOLID_TESS_TOL_CM
    tm = calc.calculate()
    if tm is None or not tm.triangleCount:
        raise RuntimeError("Fusion could not tessellate the solid body" + (f" ({'; '.join(errors)})" if errors else ""))
    coords = list(tm.nodeCoordinatesAsDouble)
    idx = list(tm.nodeIndices)
    tris = [tuple(idx[i:i + 3]) for i in range(0, len(idx), 3)]
    path = os.path.join(out_dir, "solid_tess.stl")
    write_binary_stl(path, coords, tris, scale=10.0)
    return path, "STL (0.01 mm tessellation)"


def _face_kind(face):
    """'plane' / 'cylinder' / 'cone' / ... from BRepFace.geometry.surfaceType (for the exclusions breakdown)."""
    try:
        t = face.geometry.surfaceType
        S = adsk.core.SurfaceTypes
        names = ((S.PlaneSurfaceType, "plane"), (S.CylinderSurfaceType, "cylinder"), (S.ConeSurfaceType, "cone"),
                 (S.SphereSurfaceType, "sphere"), (S.TorusSurfaceType, "torus"),
                 (S.EllipticalCylinderSurfaceType, "elliptical cylinder"),
                 (S.EllipticalConeSurfaceType, "elliptical cone"), (S.NurbsSurfaceType, "freeform"))
        for v, n in names:
            if t == v:
                return n
    except Exception:  # noqa: BLE001  (cosmetic label only)
        pass
    return "face"


def export_faces(faces, out_dir):
    """Tessellate each 'Disregard faces' selection into its own STL (mm), named exclude_face_NN_<kind>.stl,
    so the report can say how many points each face removed. Returns the list of paths."""
    paths = []
    for k, f in enumerate(faces):
        face = adsk.fusion.BRepFace.cast(f) or f
        calc = face.meshManager.createMeshCalculator()
        calc.setQuality(adsk.fusion.TriangleMeshQualityOptions.VeryHighQualityTriangleMesh)
        calc.surfaceTolerance = SOLID_TESS_TOL_CM
        tm = calc.calculate()
        if tm is None or not tm.triangleCount:
            log(f"selected face {k + 1} has no tessellation; skipped")
            continue
        coords = list(tm.nodeCoordinatesAsDouble)
        idx = list(tm.nodeIndices)
        tris = [(idx[i], idx[i + 1], idx[i + 2]) for i in range(0, len(idx), 3)]
        kind = _face_kind(face).replace(" ", "_")
        path = os.path.join(out_dir, f"exclude_face_{k + 1:02d}_{kind}.stl")
        write_binary_stl(path, coords, tris, scale=10.0)
        paths.append(path)
    return paths


def _body_report(body):
    """The conversion report path Mesh to Solid stored on the body (Attributes), if the file exists."""
    try:
        a = body.attributes.itemByName(ATTR_GROUP, ATTR_REPORT)
        if a is not None and a.value and os.path.isfile(a.value):
            return a.value
    except Exception:  # noqa: BLE001
        pass
    return None


def resolve_inputs(form, out_dir, export_mesh_body):
    """Write scan STL (mm) and solid file into out_dir, or take them from a job folder.
    Selected bodies win; a job folder fills whatever is not selected. Returns (scan, solid, notes);
    raises ValueError with a user-facing message when the inputs are incomplete."""
    notes = []
    scan = solid = None
    job = form.get("job") or ""
    found = {}
    if job:
        if not os.path.isdir(job):
            raise ValueError(f"The job folder does not exist: {job}")
        found = find_job_files(job)
    if form.get("scan") is not None:
        body = adsk.fusion.MeshBody.cast(form["scan"]) or form["scan"]
        scan = os.path.join(out_dir, "scan.stl")
        n = export_mesh_body(body, scan)
        if not n:
            raise ValueError("The selected mesh body has no triangles.")
        notes.append(f"Scan: selected mesh body ({n} triangles).")
    elif found.get("scan"):
        scan = found["scan"]
        notes.append(f"Scan: {scan}")
    if form.get("solid") is not None:
        solid, how = export_solid(form["solid"], out_dir)
        notes.append(f"Solid: selected body exported as {how}.")
    elif found.get("solid"):
        solid = found["solid"]
        notes.append(f"Solid: {solid}")
    if scan is None or solid is None:
        missing = " and ".join(x for x, v in (("a scan mesh", scan), ("a solid", solid)) if v is None)
        if job:
            raise ValueError(f"The job folder {job} does not contain {missing}. "
                             "Select the missing body in the dialog, or pick another job folder.")
        raise ValueError("Select a scan mesh body and a solid body, or press 'Browse job folder...' "
                         "and pick a previous MeshSolid job.")
    try:
        parse_zones(form.get("zones") or "")
    except ValueError as e:
        raise ValueError(f"Excluded zones: {e}\n\nUse box(xmin,ymin,zmin, xmax,ymax,zmax) or "
                         "cyl(px,py,pz, dx,dy,dz, r[, t0,t1]), separated by ';'.")
    faces = list(form.get("faces") or [])
    form["exclude_faces"] = []
    if faces:
        fps = export_faces(faces, out_dir)
        form["exclude_faces"] = fps
        notes.append(f"Disregarded faces: {len(faces)} selected, {len(fps)} exported "
                     f"({', '.join(os.path.basename(x) for x in fps)}).")
    conv = None
    if form.get("solid") is not None:
        conv = _body_report(form["solid"])
    if conv is None and found.get("report") and os.path.basename(found["report"]) != "report.json":
        conv = found["report"]
    form["convert_report"] = conv
    if not form.get("part_name"):
        name = ""
        if form.get("solid") is not None:
            try:
                name = form["solid"].name or ""
            except Exception:  # noqa: BLE001
                name = ""
        form["part_name"] = name or (os.path.basename(os.path.normpath(job)) if job else "")
    return scan, solid, notes


# ---- the run (after the command has ended) -----------------------------------------------------------
_pending = {}


def _tail(path, n=25):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-n:])
    except OSError:
        return ""


def run_metrology(form, export_mesh_body, jobs_dir, progress_loop):
    """Export, run the engine with a progress dialog, import the band meshes, show the results.
    Returns the report dict, or None after telling the user why."""
    ui = _ui()
    design = adsk.fusion.Design.cast(_app().activeProduct)
    if design is None:
        tell("Open a Fusion design (the Design workspace) and run Metrology again.")
        return None
    out_dir = jobs_dir()
    log("run", f"job folder {out_dir}; form: " +
        json.dumps({k: v for k, v in form.items() if k not in ("scan", "solid", "faces")}, default=str))
    try:
        scan, solid, in_notes = resolve_inputs(form, out_dir, export_mesh_body)
    except ValueError as e:
        tell(str(e))
        return None
    s = EngineSettings.load()
    probs = s.problems()
    if probs:
        tell("The meshsolid engine is not ready, so nothing was run:\n\n" + "\n".join(probs) +
             "\n\nRun INSTALL_MeshSolid.bat again.")
        return None
    band_path = os.path.join(out_dir, "bands_spec.json")
    band_json_dump(band_path, form["bands"])
    job = MetrologyJob(scan, solid, out_dir, form_options(form, band_path))
    log("engine command", " ".join(job.command(s)))
    run = EngineRun(job, s)
    rc = progress_loop(ui, run, "MeshSolid metrology", 0.25)
    report_path = job.paths()["report"]
    if not os.path.isfile(report_path):
        tell(f"The metrology engine did not write a report (exit code {rc}).\n\n"
             f"Job folder: {out_dir}\nEngine log: {job.paths()['log']}\n\n{_tail(job.paths()['log'])}")
        return None
    with open(report_path, encoding="utf-8") as f:
        rep = json.load(f)
    rep["out_dir"] = out_dir
    notes = list(in_notes)
    added = 0
    if form.get("do_map"):
        added, imp_notes = import_band_meshes(design, (rep.get("files") or {}).get("bands") or [])
        notes.extend(imp_notes)
        if added:
            notes.append(f"Imported {added} coloured band mesh body(ies) into this design "
                         "(MeshSolid metrology). The original scan and solid were not moved.")
    rep["notes"] = list(rep.get("notes") or []) + notes
    _pending["result"] = rep
    log("done", f"{added} band meshes imported; report {report_path}")
    if show_results_palette(rep) is None:
        try:
            ok = ui.commandDefinitions.itemById(METRO_RESULT_ID).execute()
            if ok is False:
                raise RuntimeError("execute() returned False")
        except Exception:  # noqa: BLE001
            log("results dialog could not open; showing a message box", traceback.format_exc())
            tell(results_text(rep) + f"\n\nJob folder: {out_dir}")
    return rep


# ---- wiring -----------------------------------------------------------------------------------------
class Execute(adsk.core.CommandEventHandler):
    """Reads the dialog and hands the run to the custom event, so the export, the long engine run,
    the mesh import and the results command all happen after this command has ended."""
    def __init__(self, export_mesh_body, jobs_dir, progress_loop):
        super().__init__()
        self.export_mesh_body = export_mesh_body
        self.jobs_dir = jobs_dir
        self.progress_loop = progress_loop

    def notify(self, args):
        try:
            form = read_form(args.command.commandInputs)
            _pending["last"] = {k: v for k, v in form.items() if k not in ("scan", "solid")}
            _pending["run"] = form
            fired = False
            if _pending.get("event_registered"):
                try:
                    fired = bool(_app().fireCustomEvent(METRO_EVENT_ID, ""))
                except Exception:  # noqa: BLE001
                    log("fireCustomEvent failed; running inside the command", traceback.format_exc())
            if not fired:
                _pending.pop("run", None)
                run_metrology(form, self.export_mesh_body, self.jobs_dir, self.progress_loop)
        except Exception:  # noqa: BLE001
            report_error("Compute metrology")


class _RunEvent(adsk.core.CustomEventHandler):
    def __init__(self, export_mesh_body, jobs_dir, progress_loop):
        super().__init__()
        self.export_mesh_body = export_mesh_body
        self.jobs_dir = jobs_dir
        self.progress_loop = progress_loop

    def notify(self, args):
        try:
            form = _pending.pop("run", None)
            if form is None:
                return
            run_metrology(form, self.export_mesh_body, self.jobs_dir, self.progress_loop)
        except Exception:  # noqa: BLE001
            report_error("Compute metrology")


class _Created(adsk.core.CommandCreatedEventHandler):
    def __init__(self, export_mesh_body, jobs_dir, progress_loop, handlers):
        super().__init__()
        self.args = (export_mesh_body, jobs_dir, progress_loop)
        self.handlers = handlers

    def notify(self, args):
        try:
            cmd = args.command
            # Handlers first: if the dialog layout ever fails, Compute still runs (and errors still show).
            ch = Changed(); cmd.inputChanged.add(ch); self.handlers.append(ch)
            ex = Execute(*self.args); cmd.execute.add(ex); self.handlers.append(ex)
            build_dialog(cmd, _pending.get("last") or {})
        except Exception:  # noqa: BLE001
            report_error("opening the Metrology dialog")


def attach(cmd_def, export_mesh_body, jobs_dir, progress_loop, handlers):
    """Wire the Metrology command definition and its run event. handlers is MeshSolid._handlers (a global
    list) so Fusion's event handlers are never garbage-collected."""
    h = _Created(export_mesh_body, jobs_dir, progress_loop, handlers)
    cmd_def.commandCreated.add(h); handlers.append(h)
    try:
        app = _app()
        try:
            app.unregisterCustomEvent(METRO_EVENT_ID)
        except Exception:  # noqa: BLE001
            pass
        ev = app.registerCustomEvent(METRO_EVENT_ID)
        rh = _RunEvent(export_mesh_body, jobs_dir, progress_loop)
        ev.add(rh); handlers.append(rh)
        _pending["event_registered"] = True
    except Exception:  # noqa: BLE001
        _pending["event_registered"] = False
        log("custom event not registered; Metrology will run inside its command", traceback.format_exc())
    return h


def detach():
    if _pending.pop("event_registered", False):
        try:
            _app().unregisterCustomEvent(METRO_EVENT_ID)
        except Exception:  # noqa: BLE001
            pass
    try:
        pal = _ui().palettes.itemById(PALETTE_ID)
        if pal is not None:
            pal.deleteMe()
    except Exception:  # noqa: BLE001
        pass
    _palette_handlers.clear()


# ---- zoomable results palette (HTML: zoom buttons, Ctrl+wheel, both scrollbars, drag to pan) ---------
_palette_handlers = []
_view = {}


def palette_action(action, rep):
    """What a toolbar button in the results palette does. Returns the status text shown in the page."""
    files = (rep or {}).get("files") or {}
    if action == "ready":
        return "ok"
    if action == "exportPdf":
        dest = export_pdf_report(rep or {})
        return f"Saved {dest}" if dest else "Export cancelled"
    if action == "openFull":
        pdf = files.get("report_pdf")
        target = pdf if pdf and os.path.isfile(pdf) else files.get("report_png")
        return open_file(target) or f"Opened {os.path.basename(target or '')}"
    if action == "openPng":
        return open_file(files.get("report_png")) or "Opened report.png"
    if action == "openFolder":
        return open_file((rep or {}).get("out_dir")) or "Opened the job folder"
    return f"Unknown action {action!r}"


class _PaletteHTML(adsk.core.HTMLEventHandler):
    def notify(self, args):
        try:
            a = adsk.core.HTMLEventArgs.cast(args) or args
            action = V.parse_action(a.action, a.data)
            if not action:
                return
            a.returnData = palette_action(action, _view.get("rep") or _pending.get("result"))
        except Exception:  # noqa: BLE001
            report_error("the metrology results window")


def show_results_palette(rep):
    """Show the result in a resizable HTML palette (zoom in/out/fit, % readout, both scrollbars,
    Ctrl+wheel, drag to pan, Export PDF / Open full-size). Returns the Palette, or None when it could not
    be shown (the caller then opens the command dialog)."""
    try:
        ui = _ui()
        path = V.write_page(rep)
        url = V.palette_url(path)
        _view["rep"] = rep
        pal = ui.palettes.itemById(PALETTE_ID)
        if pal is None:
            pal = ui.palettes.add(PALETTE_ID, "MeshSolid metrology result", url, False, True, True,
                                  PALETTE_W, PALETTE_H, True)
            if pal is None:
                raise RuntimeError("Palettes.add returned nothing")
            h = _PaletteHTML()
            pal.incomingFromHTML.add(h); _palette_handlers.append(h)
            try:
                pal.dockingState = adsk.core.PaletteDockingStates.PaletteDockStateFloating
                pal.setMinimumSize(640, 480)
            except Exception:  # noqa: BLE001  (cosmetic)
                log("palette docking/size ignored", traceback.format_exc())
        else:
            pal.htmlFileURL = url
        pal.isVisible = True
        log("results palette", url)
        return pal
    except Exception:  # noqa: BLE001
        log("results palette could not open; falling back to the dialog", traceback.format_exc())
        return None


RESULTS_W, RESULTS_H = 1000, 1050        # results window (pixels); minimum below
RESULTS_MIN_W, RESULTS_MIN_H = 820, 760


def open_file(path):
    """Open a file in the default viewer (os.startfile on Windows). Returns an error string or None."""
    if not path or not (os.path.isfile(path) or os.path.isdir(path)):
        return f"File not found: {path}"
    try:
        if hasattr(os, "startfile"):
            os.startfile(path)  # noqa: S606  (Windows default viewer)
        else:
            import subprocess
            subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", path])
        return None
    except Exception as e:  # noqa: BLE001
        log(f"could not open {path}", traceback.format_exc())
        return f"Could not open {path}: {e}"


def export_pdf_report(res, ui=None):
    """'Export PDF report...': save-as dialog, copy report.pdf there, offer to open it.
    Returns the destination path or None."""
    import shutil
    ui = ui or _ui()
    src = (res.get("files") or {}).get("report_pdf")
    if not src or not os.path.isfile(src):
        tell("This run has no PDF report (report.pdf is missing from the job folder"
             + (f" {res.get('out_dir')}" if res.get("out_dir") else "") + ").")
        return None
    dlg = ui.createFileDialog()
    dlg.title = "Export metrology report (PDF)"
    dlg.filter = "PDF files (*.pdf)"
    part = ((res.get("meta") or {}).get("part_name") or "metrology").strip() or "metrology"
    safe = "".join(ch if ch.isalnum() or ch in "-_. " else "_" for ch in part)
    dlg.initialFilename = f"{safe}_metrology_report.pdf"
    if dlg.showSave() != adsk.core.DialogResults.DialogOK:
        return None
    dest = dlg.filename
    if not dest.lower().endswith(".pdf"):
        dest += ".pdf"
    try:
        shutil.copyfile(src, dest)
    except OSError as e:
        tell(f"Could not save the PDF to {dest}:\n{e}")
        return None
    ans = ui.messageBox(f"Saved the metrology report to\n{dest}\n\nOpen it now?", "MeshSolid metrology",
                        adsk.core.MessageBoxButtonTypes.YesNoButtonType,
                        adsk.core.MessageBoxIconTypes.QuestionIconType)
    if ans == adsk.core.DialogResults.DialogYes:
        err = open_file(dest)
        if err:
            tell(err)
    return dest


class _ResultsChanged(adsk.core.InputChangedEventHandler):
    def notify(self, args):
        try:
            inp = args.input
            res = _pending.get("result") or {}
            files = res.get("files") or {}
            if inp.id == "export_pdf":
                inp.value = False
                export_pdf_report(res)
            elif inp.id == "open_full":
                inp.value = False
                err = open_file(files.get("report_pdf") if os.path.isfile(files.get("report_pdf") or "")
                                else files.get("report_png"))
                if err:
                    tell(err)
            elif inp.id == "open_view":
                inp.value = False
                if show_results_palette(res) is None:
                    tell("The zoomable results window could not be opened. See addin_ui.log.")
            elif inp.id == "open_png":
                inp.value = False
                err = open_file(files.get("report_png"))
                if err:
                    tell(err)
        except Exception:  # noqa: BLE001
            report_error("the metrology results buttons")


class _ResultsCreated(adsk.core.CommandCreatedEventHandler):
    def __init__(self, png_sizer=None, handlers=None):
        super().__init__()
        self.png_sizer = png_sizer
        self.handlers = handlers if handlers is not None else []

    def notify(self, args):
        try:
            res = _pending.get("result")
            c = args.command
            h = _ResultsChanged()
            c.inputChanged.add(h); self.handlers.append(h)
            c.okButtonText = "Close"
            _size_dialog(c, RESULTS_W, RESULTS_H, RESULTS_MIN_W, RESULTS_MIN_H)
            ins = c.commandInputs
            if not res:
                ins.addTextBoxCommandInput("t", "", "<span style='font-size:14pt'>No metrology result yet.</span>", 2, True)
                return
            top = ins.addTextBoxCommandInput("headline", "", summary_html(res), 6, True)
            top.isFullWidth = True
            files = res.get("files") or {}
            if files.get("report_pdf") and os.path.isfile(files["report_pdf"]):
                b = ins.addBoolValueInput("export_pdf", "Export PDF report...", False, "", False)
                b.text = "Export PDF report..."
                _tip(b, "Save the PDF inspection report", "Choose where to save a copy of report.pdf, then open it.")
            b = ins.addBoolValueInput("open_view", "Zoomable view", False, "", False)
            b.text = "Open zoomable view (zoom / pan)"
            _tip(b, "Zoom and pan", "Opens the result in a resizable window with zoom buttons and both scrollbars.")
            b = ins.addBoolValueInput("open_full", "Open full-size report", False, "", False)
            b.text = "Open full-size report"
            _tip(b, "Open in the default viewer", "Opens report.pdf (or report.png) outside Fusion, full size.")
            png = files.get("report_png")
            if png and os.path.isfile(png):
                b = ins.addBoolValueInput("open_png", "Open report image", False, "", False)
                b.text = "Open report image (PNG)"
                try:
                    img = ins.addImageCommandInput("preview", "", png)
                    img.isFullWidth = True
                    if self.png_sizer:
                        self.png_sizer(img, png)
                except Exception:  # noqa: BLE001  (very large PNGs may be refused)
                    log("report image not shown", traceback.format_exc())
                    ins.addTextBoxCommandInput("pv", "Report", png, 1, True)
            box = ins.addTextBoxCommandInput("summary", "", results_html(res), 14, True)
            box.isFullWidth = True
            paths = ins.addTextBoxCommandInput(
                "paths", "", f"<span style='font-size:12pt'>Job folder: {res.get('out_dir', '')}</span>", 2, True)
            paths.isFullWidth = True
        except Exception:  # noqa: BLE001
            report_error("showing the metrology result")


def attach_results(cmd_def, handlers, png_sizer=None):
    h = _ResultsCreated(png_sizer, handlers)
    cmd_def.commandCreated.add(h); handlers.append(h)
    return h

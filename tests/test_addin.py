"""Add-in parts outside Fusion: STL export from Fusion-style mesh arrays, the engine subprocess client
against the real engine (ok, failure and cancel paths), report.json -> user parameters, the review
model and the import step of MeshSolid.py against a stub `adsk`."""
import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest
import trimesh

ROOT = Path(__file__).resolve().parents[1]
ADDIN = ROOT / "fusion_addin" / "MeshSolid"
sys.path.insert(0, str(ADDIN))
from mslib.mesh_io import triangles_from_polygon_mesh, write_binary_stl  # noqa: E402
from mslib.engine_client import EngineSettings, EngineJob, EngineRun, read_result, selfcheck  # noqa: E402
from mslib.outcome import CANCELLED, TIMEOUT, explain  # noqa: E402
from mslib.params import parameter_specs  # noqa: E402
from mslib.report_params import parameters_from_report, by_group  # noqa: E402
from mslib.review import review_model, review_html  # noqa: E402


def _settings(**kw):
    return EngineSettings(python=sys.executable, engine_dir=str(ROOT), **kw)


def _wait(run, cancel_after=None):
    t0 = time.time()
    while True:
        rc = run.poll()
        if rc is not None:
            return rc
        if cancel_after is not None and time.time() - t0 > cancel_after:
            run.cancel()
        time.sleep(0.2)


def test_stl_export_cm_to_mm_with_quads(tmp_path):
    box = trimesh.creation.box((2.0, 3.0, 4.0))                    # cm
    coords = box.vertices.ravel().tolist()
    tri = box.faces.ravel().tolist()
    tris = triangles_from_polygon_mesh(coords, tri[6:], quad_idx=[*box.faces[0], [v for v in box.faces[1] if v not in box.faces[0]][0]])
    assert len(tris) == len(box.faces)
    p = tmp_path / "b.stl"; write_binary_stl(p, coords, triangles_from_polygon_mesh(coords, tri))
    m = trimesh.load(p)
    assert np.allclose(m.extents, [20, 30, 40]) and m.is_watertight and abs(m.volume - 24000) < 1e-3


def test_parameter_specs_compat():
    dims = [{"type": "diameter", "value": 14.5, "fit_value": 14.507, "name": "D3", "units": "mm"},
            {"type": "cone_half_angle_deg", "value": 5.0, "fit_value": 5.0, "name": "A 1", "units": "deg"}]
    ps = parameter_specs(dims)
    assert ps[0][:3] == ("ms_D3", "14.5000 mm", "mm") and "snapped" in ps[0][3]
    assert ps[1][:3] == ("ms_A_1", "5.0000 deg", "deg") and "fitted" in ps[1][3]


GEAR_REP = {"gear": {"detected": True, "build_from": "snapped", "teeth": 33, "hand": "left", "face_width_mm": 41.6,
                     "snapped": {"normal_DP": 24.0, "normal_module": 1.05833, "normal_pressure_angle_deg": 20.0,
                                 "helix_angle_pitch_deg": 20.0148, "lead_mm": 320.573, "pitch_diameter": 37.1699,
                                 "tip_snapped": True, "tip_diameter_standard": 39.2866, "root_snapped": False,
                                 "root_diameter_measured": 33.7575, "profile_shift_x": 0.0,
                                 "span": {"teeth_spanned": 4, "W_normal_mm": 11.4075},
                                 "over_pins": {"pin_diameter_mm": 1.8288, "M_over_pins_mm": 39.3923}}},
            "threads": [{"kind": "external", "hand": "right", "major_d": 11.976, "pitch": 1.748, "length": 23.6,
                         "standard": {"designation": "M12x1.75", "nominal_major_d": 12.0, "nominal_pitch": 1.75,
                                      "match": "good"}, "modelled_as": "cylinder at major diameter"}],
            "dimensions": [{"type": "diameter", "value": 22.2, "fit_value": 22.2043, "name": "D1", "units": "mm"},
                           {"type": "torus_minor_radius", "value": 1.0, "fit_value": 1.02, "name": "D2", "units": "mm"},
                           {"type": "diameter", "value": float("nan"), "fit_value": 1, "name": "D9", "units": "mm"}]}


def test_parameters_from_report_gear_thread_dims():
    ps = {s["name"]: s for s in parameters_from_report(GEAR_REP)}
    assert ps["ms_gear_teeth"]["expr"] == "33" and ps["ms_gear_teeth"]["units"] == ""
    assert ps["ms_gear_pressure_angle"]["expr"] == "20.0000 deg"
    assert ps["ms_gear_tip_dia"]["expr"] == "39.2866 mm"           # snapped standard tip
    assert ps["ms_gear_root_dia"]["expr"] == "33.7575 mm"          # measured root (not snapped)
    assert "left hand" in ps["ms_gear_teeth"]["comment"]
    assert ps["ms_thread1_major_d"]["expr"] == "12.0000 mm" and "M12x1.75" in ps["ms_thread1_major_d"]["comment"]
    assert ps["ms_D1"]["group"] == "snapped" and ps["ms_D1"]["default"]
    assert ps["ms_D2"]["group"] == "snapped" and not ps["ms_D2"]["default"]   # tori start unticked
    assert "ms_D9" not in ps                                                   # NaN never becomes a parameter
    assert all(n[0].isalpha() and n.replace("_", "").isalnum() for n in ps)


def test_parameters_from_real_impeller_report():
    p = ROOT / "out" / "user" / "stg1" / "pipe4" / "report.json"
    if not p.exists():
        pytest.skip("user part not on this machine")
    rep = json.loads(p.read_text())
    g = by_group(parameters_from_report(rep))
    names = {s["name"]: s for s in g["impeller"]}
    assert names["ms_imp_blades"]["expr"] == "15"
    assert names["ms_imp_bore_d"]["expr"] == "14.2900 mm"
    assert names["ms_imp_le_round_r"]["units"] == "mm"
    m = review_model(rep, explain(0, rep, "/nonexistent", step_path=None) | {"importable": True})
    h = review_html(m)
    assert "balance cut" in h and "unscanned" in h and "15 blades" in h


def test_explain_paths(tmp_path):
    log = tmp_path / "x.log"; log.write_text("[  1.0s] load_repair\n[  2.0s] build\nTraceback: boom\n")
    assert explain(CANCELLED, None, str(log))["kind"] == "cancelled"
    o = explain(TIMEOUT, {"last_stage": "build"}, str(log), timeout_s=60)
    assert o["kind"] == "timeout" and "60 s" in o["message"] and "build" in o["message"] and not o["importable"]
    o = explain(1, None, str(log)); assert o["kind"] == "no_report" and "boom" in o["message"]
    o = explain(-9, {"status": "incomplete", "last_stage": "deviation"}, str(log))
    assert o["kind"] == "crash" and "deviation" in o["message"]
    log2 = tmp_path / "y.log"; log2.write_text("[  0.0s] load_repair\nTraceback (most recent call last):\n  File x\nValueError: zero-size array\n")
    o = explain(1, {"status": "incomplete", "last_stage": "load_repair"}, str(log2))
    assert o["kind"] == "engine_error" and "ValueError: zero-size array" in o["message"] and "could not be read" in o["hint"]
    gate = {"status": "failed", "failure": "deviates", "deviation": {
        "mesh_vertices_to_solid": {"rms": 0.3, "p95": 0.6}, "acceptance": {"rms_limit": 0.1, "p95_limit": 0.2, "passed": False}}}
    o = explain(2, gate, str(log)); assert o["kind"] == "gate" and "OUTSIDE" in o["message"]
    o = explain(0, {"status": "ok", "step": str(tmp_path / "missing.step")}, str(log))
    assert o["kind"] == "missing_step" and not o["importable"]
    o = explain(2, {"status": "failed", "failure": "impeller: the scan has no bore"}, str(log))
    assert "bore_d" in o["hint"]


def test_engine_round_trip(tmp_path):
    from meshsolid import synthetic as S
    shape, _ = S.box_with_hole()
    mesh = tmp_path / "box.stl"; S.to_scan(shape, noise=0.02, seed=1).export(mesh)
    s = _settings(timeout_s=600)
    ok, lines = selfcheck(s, quick=True)
    assert ok, lines
    job = EngineJob(str(mesh), str(tmp_path / "job"), notes=["addin test"])
    rc = _wait(EngineRun(job, s))
    res = read_result(job, rc)
    assert res["status"] == "ok" and res["outcome"]["importable"], res["outcome"]
    assert res["step"] and os.path.exists(res["step"]) and res["preview"]
    m = review_model(res["report"], res["outcome"])
    assert any("deviation" in f for f in m["facts"]) and m["importable"]


def test_engine_failure_and_cancel(tmp_path):
    bad = tmp_path / "bad.stl"; bad.write_bytes(b"not a mesh")
    s = _settings(timeout_s=600)
    job = EngineJob(str(bad), str(tmp_path / "j1"))
    rc = _wait(EngineRun(job, s))
    res = read_result(job, rc)
    assert rc != 0 and not res["outcome"]["importable"] and res["step"] is None
    from meshsolid import synthetic as S
    shape, _ = S.box_with_hole()
    mesh = tmp_path / "box.stl"; S.to_scan(shape, noise=0.02, seed=1).export(mesh)
    job = EngineJob(str(mesh), str(tmp_path / "j2"))
    rc = _wait(EngineRun(job, s), cancel_after=1.0)
    res = read_result(job, rc)
    assert rc == CANCELLED and res["outcome"]["kind"] == "cancelled" and not res["outcome"]["importable"]
    run = EngineRun(job, _settings(timeout_s=0))
    assert _wait(run) == TIMEOUT


def test_settings_problems(tmp_path):
    s = EngineSettings(python=str(tmp_path / "nope.exe"), engine_dir=str(tmp_path))
    assert len(s.problems()) == 2
    ok, lines = selfcheck(s); assert not ok and len(lines) == 2
    p = tmp_path / "settings.json"; s.save(p); assert EngineSettings.load(p).python == s.python


# ---- MeshSolid.py against a stub adsk -------------------------------------------------------------
class _Coll(list):
    @property
    def count(self):
        return len(self)

    def item(self, i):
        return self[i]

    def add(self, body, feat=None):
        new = types.SimpleNamespace(name=getattr(body, "name", "") or "", source=body, feat=feat)
        self.append(new)
        return new


class _Params:
    def __init__(self):
        self.d = {}

    def itemByName(self, n):
        return self.d.get(n)

    def add(self, n, v, u, c):
        if " deg" in v and u != "deg":
            raise RuntimeError("unit mismatch")
        self.d[n] = (v, u, c)


class _BaseFeat:
    def startEdit(self):
        self.editing = True

    def finishEdit(self):
        self.editing = False


def _make_comp(part_design=False, design_ref=None):
    bodies = _Coll()

    def add_occ(_matrix):
        if part_design:
            raise RuntimeError("3 : Failed to create component: Part Design documents can only contain one component, "
                               "please add this Part to an Assembly to add multiple components.")
        child = types.SimpleNamespace(
            bRepBodies=_Coll(), name="MeshSolid",
            features=types.SimpleNamespace(baseFeatures=types.SimpleNamespace(add=lambda: _BaseFeat())))
        design_ref[0].allComponents.append(child)
        return types.SimpleNamespace(component=child)

    comp = types.SimpleNamespace(
        bRepBodies=bodies, name="Root",
        occurrences=types.SimpleNamespace(addNewComponent=add_occ),
        features=types.SimpleNamespace(baseFeatures=types.SimpleNamespace(add=lambda: _BaseFeat())))
    return comp


def _stub_adsk(bodies_added=1, import_ok=True, create_from_file=None, import_to_target2=None,
               import_error=None, part_design=False, parametric=True, import_to_new_document=None,
               copy_ok=True, close_ok=True, add_bodies_ok=True):
    """create_from_file: None (returns nothing), a list of temp bodies, or an Exception to raise.
    import_to_target2: None (method absent), True/False, or 'raise'.
    import_to_new_document: None (method returns None, so a failed in-place import stays failed),
    True (opens a new doc with one body), or 'raise'."""
    holder = [None]
    comp = _make_comp(part_design, holder)
    design = types.SimpleNamespace(allComponents=[comp], rootComponent=comp, userParameters=_Params(),
                                   designType=1 if parametric else 0, parentDocument=None)
    holder[0] = design
    calls = []

    class _Doc:
        def __init__(self, des, app):
            self.design, self._app, self.closed = des, app, False
            des.parentDocument = self
            self.products = types.SimpleNamespace(itemByProductType=lambda t: des)

        def close(self, save):
            if not close_ok:
                raise RuntimeError("close failed")
            self.closed = True
            return True

        def activate(self):
            self._app.activeDocument = self
            self._app.activeProduct = self.design
            return True

    def _new_doc(app):
        ncomp = _make_comp(False, [None])
        ncomp.bRepBodies.append(types.SimpleNamespace(name="Body1"))
        ndesign = types.SimpleNamespace(allComponents=[ncomp], rootComponent=ncomp, userParameters=_Params(),
                                        designType=1, parentDocument=None)
        ncomp.occurrences = types.SimpleNamespace(addNewComponent=lambda m: (_ for _ in ()).throw(AssertionError("no")))
        ncomp.features = types.SimpleNamespace(baseFeatures=types.SimpleNamespace(add=lambda: _BaseFeat()))
        holder_n = [ndesign]
        # leftover add_occ unused
        doc = _Doc(ndesign, app)
        app.activeDocument, app.activeProduct = doc, ndesign
        app.opened_docs.append(doc)
        return doc

    orig_add = comp.bRepBodies.add
    def _add(body, feat=None):
        if not add_bodies_ok:
            raise RuntimeError("bRepBodies.add failed")
        return orig_add(body, feat)
    comp.bRepBodies.add = _add

    class IM:
        def createSTEPImportOptions(self, p):
            opts = types.SimpleNamespace(path=p, isViewFit=True)
            return opts

        def importToTarget(self, o, t):
            calls.append(("importToTarget", getattr(o, "path", o)))
            if import_error:
                raise RuntimeError(import_error)
            if import_ok:
                t.bRepBodies.extend([types.SimpleNamespace(name="") for _ in range(bodies_added)])
            return import_ok

        def importToNewDocument(self, o):
            calls.append(("importToNewDocument", getattr(o, "path", o)))
            if import_to_new_document == "raise":
                raise RuntimeError("importToNewDocument failed")
            if import_to_new_document:
                return _new_doc(app)
            return None

    im = IM()
    if import_to_target2 is not None:
        def importToTarget2(o, t):
            calls.append(("importToTarget2", getattr(o, "path", o)))
            if import_to_target2 == "raise":
                raise RuntimeError("2 : InternalValidationError : pResult")
            if import_to_target2:
                t.bRepBodies.extend([types.SimpleNamespace(name="") for _ in range(bodies_added)])
                return _Coll(list(t.bRepBodies[-bodies_added:]))
            return None
        im.importToTarget2 = importToTarget2

    class TBM:
        @staticmethod
        def get():
            return TBM()

        def createFromFile(self, path):
            calls.append(("createFromFile", path))
            if isinstance(create_from_file, Exception):
                raise create_from_file
            if create_from_file is None:
                return None
            return _Coll(list(create_from_file))

        def copy(self, body):
            if not copy_ok:
                raise RuntimeError("TemporaryBRepManager.copy failed")
            return types.SimpleNamespace(name=getattr(body, "name", ""), copied=True)

    app = types.SimpleNamespace(activeProduct=design, activeDocument=None, importManager=im, calls=calls,
                                opened_docs=[])
    app.activeDocument = _Doc(design, app)
    core = types.ModuleType("adsk.core"); fusion = types.ModuleType("adsk.fusion")
    for n in ("CommandCreatedEventHandler", "CommandEventHandler", "CustomEventHandler", "InputChangedEventHandler",
              "HTMLEventHandler"):
        setattr(core, n, type(n, (), {}))
    core.DropDownStyles = types.SimpleNamespace(TextListDropDownStyle=1, LabeledIconDropDownStyle=0)
    core.DialogResults = types.SimpleNamespace(DialogOK=0, DialogCancel=1)
    core.Application = types.SimpleNamespace(get=lambda: app)
    core.ValueInput = types.SimpleNamespace(createByString=lambda s: s)
    core.Color = types.SimpleNamespace(create=lambda r, g, b, a=255: types.SimpleNamespace(red=r, green=g, blue=b, opacity=a))
    fusion.MeshUnits = types.SimpleNamespace(MillimeterMeshUnit=1)
    core.Matrix3D = types.SimpleNamespace(create=lambda: object())
    fusion.Design = types.SimpleNamespace(cast=lambda x: x)
    fusion.DesignTypes = types.SimpleNamespace(ParametricDesignType=1, DirectDesignType=0)
    fusion.TemporaryBRepManager = TBM
    adsk = types.ModuleType("adsk"); adsk.core, adsk.fusion = core, fusion; adsk.doEvents = lambda: None
    return {"adsk": adsk, "adsk.core": core, "adsk.fusion": fusion}, design


def _load_addin(monkeypatch, **kw):
    mods, design = _stub_adsk(**kw)
    for k, v in mods.items():
        monkeypatch.setitem(sys.modules, k, v)
    import importlib.util
    spec = importlib.util.spec_from_file_location("MeshSolid_under_test", ADDIN / "MeshSolid.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m, design


def test_addin_import_and_parameters(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    m, design = _load_addin(monkeypatch)
    specs = parameters_from_report(GEAR_REP)
    res = {"step": "x.step", "model": {"params": specs}, "part_name": "pinion"}
    design.userParameters.d["ms_gear_teeth"] = ("33", "", "")
    txt = m._import(res, {"gear", "thread"})
    n = len([s for s in specs if s["group"] in ("gear", "thread")])
    assert "Imported 1 solid body" in txt and "MeshSolid_pinion" in txt and "(importToTarget)" in txt
    assert f"created: {n - 1} of {n}" in txt and "ms_gear_teeth" in txt
    assert "Import notes" not in txt and "createFromFile" not in txt   # failures go to the log, not the dialog
    log = (tmp_path / "meshsolid" / "addin_ui.log").read_text()
    assert "failed attempt:" in log and "createFromFile" in log
    assert m._parse_measure("bore_d=14.275") == {"bore_d": 14.275}
    assert m._parse_measure("bore_d=14.29; D3 = 40") == {"bore_d": 14.29, "D3": 40.0}
    with pytest.raises(ValueError):
        m._parse_measure("bore 14")


def test_addin_import_warns_on_body_count(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    m, _ = _load_addin(monkeypatch, bodies_added=3)
    assert "expected 1" in m._import({"step": "x.step", "model": {"params": []}}, set())
    m, _ = _load_addin(monkeypatch, bodies_added=0, import_ok=False)
    txt = m._import({"step": "x.step", "model": {"params": []}}, set())
    assert "import failed" in txt and "Attempts:" in txt


def test_parameters_from_screw_report():
    rep = {"screw_rotor": {"detected": True, "kind": "screw_compressor_rotor", "lobes": 4, "hand": "right",
                           "lead_mm": 300.001, "lead_range_mm": None, "wrap_deg": 144.0, "rotor_length_mm": 120.0,
                           "tip_diameter": 79.9942, "root_diameter": 48.073}}
    ps = {s["name"]: s for s in parameters_from_report(rep)}
    assert ps["ms_screw_lobes"]["expr"] == "4"
    assert ps["ms_screw_lead"]["expr"] == "300.0010 mm"
    assert ps["ms_screw_wrap"]["expr"] == "144.0000 deg"
    assert "ms_screw_lead_start" not in ps and "right hand" in ps["ms_screw_lobes"]["comment"]
    assert "screw" in by_group(list(ps.values()))
    vp = {"screw_rotor": {"detected": True, "kind": "variable_pitch_screw", "lobes": 1, "hand": "right",
                          "lead_mm": None, "lead_range_mm": [30.0, 50.0]}}
    ps = {s["name"]: s for s in parameters_from_report(vp)}
    assert ps["ms_screw_lead_start"]["expr"] == "30.0000 mm" and "ms_screw_lead" not in ps


def test_parameters_from_keyway_report():
    rep = {"keyways": [{"kind": "shaft keyseat", "ends": "round", "width_b": 11.9973, "depth_t1": 5.0, "length": 49.998,
                        "standard": {"standard": "DIN 6885-1", "designation": "12 x 8", "b": 12, "h": 8,
                                     "depth_nominal": 5.0, "depth_misfit_mm": 0.0}}]}
    ps = {s["name"]: s for s in parameters_from_report(rep)}
    assert ps["ms_key1_width"]["expr"] == "12.0000 mm" and "DIN 6885-1 12 x 8" in ps["ms_key1_width"]["comment"]
    assert ps["ms_key1_depth"]["expr"] == "5.0000 mm" and ps["ms_key1_length"]["group"] == "keyway"


def test_parameters_from_spline_and_rolling_element_report():
    rep = {"splines": [{"teeth": 14, "side": "external", "form": "involute 30 deg", "major_diameter": 29.5588,
                        "minor_diameter": 25.6265, "length": 40.0, "module": 2.0, "standard": "DIN 5480",
                        "designation": "W 30 x 2 x 14"},
                       {"teeth": 8, "side": "external", "form": "straight-sided", "major_diameter": 35.98,
                        "minor_diameter": 32.0, "length": 45.0, "width_B": 5.9987, "standard": "ISO 14",
                        "designation": "8 x 32 x 36", "snap": {"B": 6}}],
           "rolling_elements": {"elements": "ball", "count": 9, "diameter": 7.94, "pitch_diameter": 38.5}}
    ps = {s["name"]: s for s in parameters_from_report(rep)}
    assert ps["ms_spl1_module"]["expr"] == "2.0000 mm" and "DIN 5480 W 30 x 2 x 14" in ps["ms_spl1_teeth"]["comment"]
    assert ps["ms_spl2_width"]["expr"] == "6.0000 mm" and ps["ms_spl2_width"]["group"] == "spline"
    assert ps["ms_brg_Z"]["group"] == "bearing" and ps["ms_brg_Dpw"]["expr"] == "38.5000 mm"


def test_batch_job_runs_engine_batch_and_summarises(tmp_path):
    from mslib.engine_client import BatchJob, batch_summary_text
    from meshsolid import occ_util as U
    from meshsolid import synthetic as S
    inp = tmp_path / "in"; inp.mkdir()
    sh, _ = S.box_with_hole()
    S.to_scan(sh, edge=1.0, noise=0.02, seed=0).export(inp / "box.stl")
    U.write_step(sh, str(inp / "box.step"))
    bj = BatchJob(str(inp), str(tmp_path / "out"))
    s = _settings(timeout_s=900)
    assert bj.meshes() == [str(inp / "box.stl")]
    cmd = bj.command(s)
    assert cmd[1:5] == ["-u", "-m", "meshsolid.batch", str(inp)] and "--out" in cmd
    assert bj.settings(s).timeout_s == 900 + 600
    assert "No batch summary" in batch_summary_text(bj)
    run = EngineRun(bj, bj.settings(s))
    rc = _wait(run)
    assert rc == 0, open(bj.paths()["log"]).read()[-2000:]
    txt = batch_summary_text(bj)
    assert "1 of 1 meshes converted" in txt and "1 scored against a STEP" in txt and "box: ok" in txt


# ---- 1.0.0 UI: toolbar panels, grouped dialog, option mapping (stub adsk) ---------------------------------
from mslib import ui_options as UO  # noqa: E402


class _Ctrl:
    def __init__(self, owner, cid):
        self.owner, self.id, self.isPromoted, self.isPromotedByDefault = owner, cid, False, False

    def deleteMe(self):
        self.owner.pop(self.id)


class _Controls(dict):
    def itemById(self, i):
        return self.get(i)

    def addCommand(self, cd):
        c = _Ctrl(self, cd.id); self[cd.id] = c; return c


class _Panel:
    def __init__(self, reg, pid, name):
        self.reg, self.id, self.name, self.controls = reg, pid, name, _Controls()

    def deleteMe(self):
        self.reg.pop(self.id)


class _Panels:
    def __init__(self, reg):
        self.reg, self.mine = reg, []

    def itemById(self, i):
        return self.reg.get(i) if i in self.mine else None

    def add(self, pid, name):
        assert pid not in self.reg, "panel ids must be unique across all panels"
        self.reg[pid] = _Panel(self.reg, pid, name); self.mine.append(pid); return self.reg[pid]


class _Tab:
    def __init__(self, reg, tid, name):
        self.id, self.name, self.toolbarPanels = tid, name, _Panels(reg)


class _Tabs(list):
    def itemById(self, i):
        return next((t for t in self if t.id == i), None)

    def item(self, k):
        return self[k]

    @property
    def count(self):
        return len(self)


class _CmdDef:
    def __init__(self, reg, cid, name, tip, res):
        self.reg, self.id, self.name, self.tooltip, self.resourceFolder = reg, cid, name, tip, res
        self.commandCreated = _Coll(); self.commandCreated.add = self.commandCreated.append

    def deleteMe(self):
        self.reg.pop(self.id)


class _CmdDefs(dict):
    def itemById(self, i):
        return self.get(i)

    def addButtonDefinition(self, cid, name, tip, res):
        self[cid] = _CmdDef(self, cid, name, tip, res); return self[cid]


def _fake_ui(tab_ids=(("SolidTab", "SOLID"), ("MeshTab", "MESH"), ("ToolsTab", "UTILITIES"))):
    reg = {}
    reg["SolidScriptsAddinsPanel"] = _Panel(reg, "SolidScriptsAddinsPanel", "ADD-INS")
    tabs = _Tabs(_Tab(reg, i, n) for i, n in tab_ids)
    ws = types.SimpleNamespace(toolbarTabs=tabs)
    ui = types.SimpleNamespace(commandDefinitions=_CmdDefs(), workspaces=types.SimpleNamespace(
        itemById=lambda i: ws if i == "FusionSolidEnvironment" else None),
        allToolbarPanels=types.SimpleNamespace(itemById=lambda i: reg.get(i)), messages=[])
    ui.messageBox = lambda *a: ui.messages.append(a)
    return ui, reg


def _wire_app(m, ui):
    app = m._app()
    app.userInterface = ui
    app.registerCustomEvent = lambda i: types.SimpleNamespace(add=lambda h: None)
    app.unregisterCustomEvent = lambda i: None


def test_ui_options_map_only_to_existing_engine_flags():
    import subprocess
    help_txt = subprocess.run([sys.executable, "-m", "meshsolid", "--help"], cwd=ROOT, capture_output=True,
                              text=True, timeout=120).stdout
    seen = set()
    for pid, *_ in UO.PART_TYPES:
        for acc, *_ in UO.ACCURACY:
            for dev, *_ in UO.DEVICE:
                for k, v in UO.engine_options(pid, acc, dev, 0.02).items():
                    flag = "--" + k.replace("_", "-")
                    assert flag in help_txt, flag
                    m = __import__("re").search(re.escape(flag) + r" \{([^}]*)\}", help_txt)
                    if m:
                        assert str(v) in m.group(1).split(","), (flag, v)
                    seen.add(flag)
    assert {"--accuracy", "--device", "--gears", "--impeller", "--screws", "--tol"} <= seen
    assert "--time-budget" in help_txt
    assert UO.engine_options("freeform")["accuracy"] == "printed"
    assert UO.engine_options("freeform", "machined")["accuracy"] == "machined"     # explicit choice wins
    assert "accuracy" not in UO.engine_options("gear") and UO.engine_options("auto", device="cpu")["device"] == "cpu"
    assert "tol" not in UO.engine_options(tol_mm=0)
    assert UO.timeout_for(1) == UO.TIME_MIN * 60 and UO.timeout_for(999) == UO.TIME_MAX * 60
    assert UO.default_minutes(5400) == 90
    job = EngineJob("x.stl", "/tmp/o", options=UO.engine_options("impeller", "auto", "cuda"))
    s = _settings(timeout_s=UO.timeout_for(30))
    cmd = job.command(s)
    assert cmd[cmd.index("--device") + 1] == "cuda" and cmd[cmd.index("--gears") + 1] == "off"
    assert cmd[cmd.index("--time-budget") + 1] == str(30 * 60 - 60)


import re  # noqa: E402


def test_icon_resources_exist_with_fusion_names():
    from PIL import Image
    res = ADDIN / "resources"
    for name, px in (("16x16", 16), ("16x16@2x", 32), ("32x32", 32), ("32x32@2x", 64), ("64x64", 64)):
        im = Image.open(res / "MeshSolid" / f"{name}.png")
        assert im.size == (px, px) and im.mode == "RGBA" and im.getbbox() is not None
    for name, px in (("16x16", 16), ("16x16@2x", 32), ("32x32", 32), ("32x32@2x", 64), ("64x64", 64)):
        im = Image.open(res / "Metrology" / f"{name}.png")
        assert im.size == (px, px) and im.mode == "RGBA" and im.getbbox() is not None
    assert sorted(d.name for d in res.iterdir()) == ["MeshSolid", "Metrology"]


def test_toolbar_panels_promoted_on_utilities_and_mesh_and_cleaned_up(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    m, _ = _load_addin(monkeypatch)
    ui, reg = _fake_ui()
    _wire_app(m, ui)
    m.run(None)
    assert not ui.messages, ui.messages
    cd = ui.commandDefinitions.itemById(m.CMD_ID)
    assert cd.name == "Mesh to Solid" and os.path.isdir(cd.resourceFolder)
    assert os.path.isfile(os.path.join(cd.resourceFolder, "32x32.png"))
    for pid in ("MeshSolidPanel_Utilities", "MeshSolidPanel_Mesh"):
        p = reg[pid]
        assert p.name == "MESHSOLID 2.0.0"
        for cid in (m.CMD_ID, "meshsolid_metrology"):
            c = p.controls.itemById(cid)
            assert c is not None and c.isPromoted and c.isPromotedByDefault, cid
    assert reg["SolidScriptsAddinsPanel"].controls.itemById(m.CMD_ID)
    assert reg["SolidScriptsAddinsPanel"].controls.itemById("meshsolid_metrology")
    metro = ui.commandDefinitions.itemById("meshsolid_metrology")
    assert metro.name == "Metrology" and os.path.isdir(metro.resourceFolder)
    assert os.path.isfile(os.path.join(metro.resourceFolder, "32x32.png"))
    ws = ui.workspaces.itemById("FusionSolidEnvironment")
    assert "MeshSolidPanel_Utilities" in ws.toolbarTabs.itemById("ToolsTab").toolbarPanels.mine
    assert "MeshSolidPanel_Mesh" in ws.toolbarTabs.itemById("MeshTab").toolbarPanels.mine
    log = (tmp_path / "meshsolid" / "addin_ui.log").read_text()
    assert "id ToolsTab" in log and "id MeshTab" in log
    m.stop(None)
    assert "MeshSolidPanel_Utilities" not in reg and "MeshSolidPanel_Mesh" not in reg
    assert not reg["SolidScriptsAddinsPanel"].controls and not ui.commandDefinitions


def test_toolbar_tab_found_by_name_when_id_differs(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    m, _ = _load_addin(monkeypatch)
    ui, reg = _fake_ui(tab_ids=(("SolidTab", "SOLID"), ("MeshTabV2", "MESH"), ("UtilsX", "UTILITIES")))
    _wire_app(m, ui)
    m.run(None)
    assert {"MeshSolidPanel_Utilities", "MeshSolidPanel_Mesh"} <= set(reg)
    assert "name 'UTILITIES'" in (tmp_path / "meshsolid" / "addin_ui.log").read_text()
    m.stop(None)
    ui, reg = _fake_ui(tab_ids=(("SolidTab", "SOLID"),))                      # no such tabs: fallback only
    _wire_app(m, ui)
    m.run(None)
    assert set(reg) == {"SolidScriptsAddinsPanel"} and reg["SolidScriptsAddinsPanel"].controls


# minimal command-input stubs: enough for _build_dialog / _read_form / _Changed
class _In:
    def __init__(self, reg, iid, name, value=None, **kw):
        self.id, self.name, self.value, self.isReadOnly, self.tooltip = iid, name, value, False, ""
        self.__dict__.update(kw); reg[iid] = self


class _Items(list):
    @property
    def count(self):
        return len(self)

    def item(self, i):
        return self[i]

    def add(self, name, sel, icon=""):
        if sel:
            for it in self:
                it.isSelected = False
        it = types.SimpleNamespace(name=name, isSelected=sel, icon=icon); self.append(it); return it


class _ListIn(_In):
    @property
    def selectedItem(self):
        return next((i for i in self.listItems if i.isSelected), None)


class _Ins:
    def __init__(self, reg):
        self.reg = reg

    def itemById(self, i):
        return self.reg.get(i)

    def addGroupCommandInput(self, iid, name):
        return _In(self.reg, iid, name, children=_Ins(self.reg), isExpanded=True)

    def addSelectionInput(self, iid, name, prompt):
        return _In(self.reg, iid, name, selectionCount=0, addSelectionFilter=lambda f: None,
                   setSelectionLimits=lambda a, b: None, clearSelection=lambda: None)

    def addBoolValueInput(self, iid, name, is_check, res, value):
        assert res == "", f"{iid}: no icon folders except the main button (got {res!r})"
        return _In(self.reg, iid, name, value, isCheckBox=is_check, resourceFolder=res, text="")

    def addStringValueInput(self, iid, name, value):
        return _In(self.reg, iid, name, value)

    def addButtonRowCommandInput(self, iid, name, multi):
        return _ListIn(self.reg, iid, name, listItems=_Items())

    def addDropDownCommandInput(self, iid, name, style):
        return _ListIn(self.reg, iid, name, listItems=_Items(), style=style)

    def addIntegerSliderCommandInput(self, iid, name, lo, hi, two):
        return _In(self.reg, iid, name, valueOne=lo, minimumValue=lo, maximumValue=hi)

    def addIntegerSpinnerCommandInput(self, iid, name, lo, hi, step, value):
        return _In(self.reg, iid, name, value, minimumValue=lo, maximumValue=hi)

    def addFloatSpinnerCommandInput(self, iid, name, units, lo, hi, step, value):
        return _In(self.reg, iid, name, value)

    def addTextBoxCommandInput(self, iid, name, text, rows, ro):
        return _In(self.reg, iid, name, formattedText=text, isFullWidth=False)

    def addImageCommandInput(self, iid, name, imageFile):
        return _In(self.reg, iid, name, imageFile=imageFile, isFullWidth=False, scaleFactor=1.0)


def test_dialog_layout_read_and_browse(monkeypatch, tmp_path):
    m, _ = _load_addin(monkeypatch)
    m._app().userInterface = types.SimpleNamespace(messageBox=lambda *a: (_ for _ in ()).throw(AssertionError(a)))
    reg = {}
    cmd = types.SimpleNamespace(commandInputs=_Ins(reg), okButtonText="OK", setDialogInitialSize=lambda w, h: None)
    m._build_dialog(cmd, {}, 5400)
    assert cmd.okButtonText == "Convert to Solid"
    for g in ("g_input", "g_part", "g_quality", "g_meas", "g_output", "g_adv"):
        assert g in reg and hasattr(reg[g], "children")
    assert not reg["g_adv"].isExpanded and not reg["g_meas"].isExpanded and reg["g_input"].isExpanded
    assert [i.name for i in reg["part"].listItems] == [r[1] for r in UO.PART_TYPES]
    assert reg["part"].style == 1 and all(not i.icon for i in reg["part"].listItems)      # plain text dropdown
    assert reg["browse_file"].text == "Browse..." and reg["browse_batch"].text == "Browse..."
    assert not reg["browse_file"].isCheckBox and reg["browse_file"].resourceFolder == ""
    assert reg["file"].isReadOnly and reg["time_min"].valueOne == 90 and reg["time_min"].maximumValue == UO.TIME_MAX
    assert all(reg[i].tooltip for i in ("mesh", "browse_file", "accuracy", "device", "time_min", "review", "params",
                                        "hide_mesh", "tol", "browse_batch", "measure"))
    assert "bore_d=14.275" in (reg["measure_hint"].formattedText or "")
    assert "bore_d=14.275" in (reg["measure"].tooltipDescription or "")
    f = m._read_form(cmd.commandInputs)
    assert f["part"] == "auto" and f["accuracy"] == "auto" and f["device"] == "auto" and f["review"] and f["params"]
    assert "--device auto" in reg["cmdline"].formattedText
    # pick Freeform + CPU, then browse for a file through a stub dialog
    for it in reg["part"].listItems:
        it.isSelected = it.name == "Freeform"
    for it in reg["device"].listItems:
        it.isSelected = it.name.startswith("CPU")
    stl = tmp_path / "wing.stl"; stl.write_bytes(b"solid x\nendsolid x\n")
    fd = types.SimpleNamespace(showOpen=lambda: 0, filename=str(stl))
    m._ui().createFileDialog = lambda: fd
    m._Changed().notify(types.SimpleNamespace(input=reg["browse_file"], inputs=cmd.commandInputs))
    f = m._read_form(cmd.commandInputs)
    assert f["file"] == str(stl) and f["part"] == "freeform" and f["device"] == "cpu"
    assert "--accuracy printed" in reg["cmdline"].formattedText and "--device cpu" in reg["cmdline"].formattedText
    assert "section-loft" in reg["part_hint"].formattedText
    folder = types.SimpleNamespace(showDialog=lambda: 0, folder=str(tmp_path))
    m._ui().createFolderDialog = lambda: folder
    m._Changed().notify(types.SimpleNamespace(input=reg["browse_batch"], inputs=cmd.commandInputs))
    assert m._read_form(cmd.commandInputs)["batch"] == str(tmp_path)
    # remembered choices come back next time
    reg2 = {}
    m._build_dialog(types.SimpleNamespace(commandInputs=_Ins(reg2), okButtonText="", setDialogInitialSize=None), f, 5400)
    assert m._read_form(_Ins(reg2))["part"] == "freeform" and reg2["file"].value == str(stl)


def test_import_hides_input_mesh(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    m, _ = _load_addin(monkeypatch)
    body = types.SimpleNamespace(isLightBulbOn=True)
    txt = m._import({"step": "x.step", "model": {"params": []}}, set(), body)
    assert not body.isLightBulbOn and "hidden" in txt
    body = types.SimpleNamespace(isLightBulbOn=True)
    m, _ = _load_addin(monkeypatch, bodies_added=0, import_ok=False)
    m._import({"step": "x.step", "model": {"params": []}}, set(), body)
    assert body.isLightBulbOn                                                  # failed import: mesh left visible


def test_create_from_file_is_first_and_skips_import_manager(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    temp = types.SimpleNamespace(name="tmp")
    m, design = _load_addin(monkeypatch, create_from_file=[temp])
    txt = m._import({"step": str(tmp_path / "pad.step"), "model": {"params": []}, "part_name": "battery_pad"}, set())
    assert "Imported 1 solid body" in txt and "MeshSolid_battery_pad" in txt and "(createFromFile)" in txt
    assert "Import notes" not in txt
    assert design.rootComponent.bRepBodies[-1].name == "MeshSolid_battery_pad"
    assert [c[0] for c in m._app().calls] == ["createFromFile"]
    feat = design.rootComponent.bRepBodies[-1].feat
    assert feat is not None  # parametric: added inside a BaseFeature


def test_part_design_copies_bodies_into_current_root(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    specs = [{"name": "ms_imp_bore_d", "expr": "14.2750 mm", "units": "mm", "comment": "bore", "group": "impeller"}]
    m, design = _load_addin(monkeypatch, import_ok=False,
                            import_error="2 : InternalValidationError : results",
                            part_design=True, import_to_new_document=True)
    orig_params = design.userParameters
    txt = m._import({"step": str(tmp_path / "s07.step"), "model": {"params": specs}, "part_name": "s07"},
                    {"impeller"})
    assert "Imported 1 solid body into this design as MeshSolid_s07 (copy_from_new_document)" in txt
    assert "new document" not in txt.lower() or "copied" in txt.lower()
    assert "Import notes" not in txt
    assert "created: 1 of 1" in txt and "in the new document" not in txt
    assert "ms_imp_bore_d" in orig_params.d
    assert design.rootComponent.bRepBodies[-1].name == "MeshSolid_s07"
    docs = m._app().opened_docs
    assert docs and docs[-1].closed is True
    assert m._app().activeDocument.closed is False  # original design still active
    log = (tmp_path / "meshsolid" / "addin_ui.log").read_text()
    assert "importToTarget" in log and "Part Design" in log


def test_new_document_fallback_creates_parameters_there(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    specs = [{"name": "ms_D1", "expr": "10.0000 mm", "units": "mm", "comment": "d", "group": "snapped"}]
    m, design = _load_addin(monkeypatch, import_ok=False,
                            import_error="2 : InternalValidationError : results",
                            part_design=True, import_to_new_document=True, add_bodies_ok=False)
    txt = m._import({"step": str(tmp_path / "pad.step"), "model": {"params": specs}, "part_name": "pad"},
                    {"snapped"})
    assert "opened in a new document" in txt and "MeshSolid_pad" in txt
    assert "created in that new document" in txt and "in the new document" in txt
    assert "ms_D1" not in design.userParameters.d
    new_doc = m._app().opened_docs[-1]
    assert new_doc.closed is False and "ms_D1" in new_doc.design.userParameters.d
    body = types.SimpleNamespace(isLightBulbOn=True)
    m2, _ = _load_addin(monkeypatch, import_ok=False, import_error="boom", part_design=True,
                        import_to_new_document=True, add_bodies_ok=False)
    txt2 = m2._import({"step": "x.step", "model": {"params": []}}, set(), body)
    assert body.isLightBulbOn and "left visible" in txt2


def test_review_dialog_fits_preview_and_shows_full_param_labels(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    from PIL import Image
    png = tmp_path / "wide.preview.png"
    Image.new("RGB", (2420, 715), (200, 200, 200)).save(png)
    m, _ = _load_addin(monkeypatch)
    m._state["pending"] = {
        "model": {"title": "Solid ready", "importable": True, "facts": ["ok"], "flags": [], "notes": [],
                  "param_groups": [{"group": "snapped", "label": "Snapped dimensions", "count": 12, "default": True},
                                   {"group": "impeller", "label": "Impeller", "count": 8, "default": True}],
                  "params": []},
        "preview": str(png), "out_dir": str(tmp_path), "create_params": True,
    }
    sizes = []
    reg = {}
    cmd = types.SimpleNamespace(commandInputs=_Ins(reg), okButtonText="OK", sizes=sizes,
                                execute=types.SimpleNamespace(add=lambda h: sizes.append(("exec", h))))
    cmd.setDialogMinimumSize = lambda w, h: sizes.append(("min", w, h))
    cmd.setDialogInitialSize = lambda w, h: sizes.append(("init", w, h))
    cmd.setDialogSize = lambda w, h: sizes.append(("size", w, h))
    m._ReviewCreated().notify(types.SimpleNamespace(command=cmd))
    assert cmd.okButtonText == "Import"
    assert "Parameters:" not in reg["grp_snapped"].name
    assert reg["grp_snapped"].name == "Snapped dimensions (12)"
    assert reg["grp_impeller"].name == "Impeller (8)"
    assert reg["preview"].isFullWidth and abs(reg["preview"].scaleFactor - 600 / 2420) < 1e-6
    assert any(k == "init" and w >= 560 for k, w, *rest in (s if len(s) == 3 else (s[0], 0, 0) for s in sizes if s[0] != "exec"))
    assert reg["summary"].isFullWidth


def test_missing_measurement_prompt_and_format(monkeypatch):
    from mslib.outcome import missing_measurements, explain
    failure = "impeller: the scan has no bore; give the design bore with --measure bore_d=<mm>"
    o = explain(2, {"status": "failed", "failure": failure}, "/nonexistent")
    specs = missing_measurements(o, {"failure": failure})
    assert specs == [{"name": "bore_d", "label": "Bore diameter (mm)"}]
    assert missing_measurements(o, {"failure": failure}, already={"bore_d": 14.275}) == []
    m, _ = _load_addin(monkeypatch)
    # user types the raw number
    ui = types.SimpleNamespace(inputBox=lambda *a: ("14.275", False), messages=[])
    ui.messageBox = lambda *a, **k: ui.messages.append(a)
    assert m._prompt_measures(ui, specs) == {"bore_d": 14.275}
    # pasted name=value, then a retry after a typo, then cancel
    answers = [("bore_d=14.275", False)]
    ui.inputBox = lambda *a: answers.pop(0)
    assert m._prompt_measures(ui, specs) == {"bore_d": 14.275}
    answers = [("nope", False), ("14.275", False)]
    ui.inputBox = lambda *a: answers.pop(0)
    assert m._prompt_measures(ui, specs) == {"bore_d": 14.275}
    ui.inputBox = lambda *a: ("", True)
    assert m._prompt_measures(ui, specs) is None
    action, measure = m._retry_with_prompted_measures(
        types.SimpleNamespace(inputBox=lambda *a: ("14.275", False), messageBox=lambda *a, **k: None),
        {"outcome": o, "report": {"failure": failure}}, {})
    assert action == "retry" and measure == {"bore_d": 14.275}
    action, _ = m._retry_with_prompted_measures(
        types.SimpleNamespace(inputBox=lambda *a: ("", True), messageBox=lambda *a, **k: None),
        {"outcome": o, "report": {"failure": failure}}, {})
    assert action == "cancel"
    action, _ = m._retry_with_prompted_measures(
        types.SimpleNamespace(inputBox=lambda *a: (_ for _ in ()).throw(AssertionError("should not prompt"))),
        {"outcome": {"message": "deviation too high", "hint": ""}, "report": {"failure": "deviates"}}, {})
    assert action == "fail"
    assert m._format_measure({"bore_d": 14.275}) == "bore_d=14.275"


def test_metrology_dialog_and_options(monkeypatch, tmp_path):
    m, _ = _load_addin(monkeypatch)
    from mslib import metrology_cmd as Metro
    reg = {}
    cmd = _SCommand(reg)                       # strict Fusion API stub (see below)
    Metro.build_dialog(cmd, {})
    assert cmd.okButtonText == "Compute metrology"
    for g in ("g_in", "g_cmap", "g_bands", "g_excl", "g_an", "g_dev"):
        assert g in reg
    assert reg["align"].value is True          # spec-v2 alignment on by default
    assert reg["do_map"].value and reg["do_zones"].value and reg["do_probes"].value and reg["do_hist"].value
    assert not reg["do_section"].value
    assert "bore_d" not in (reg["probe_xyz"].tooltipDescription or "")  # not the convert dialog
    assert "x,y,z" in (reg["probe_xyz"].tooltipDescription or "")
    assert "never moved" in (reg["align"].tooltipDescription or "").lower() or "not moved" in (reg["align"].tooltipDescription or "").lower()
    form = Metro.read_form(cmd.commandInputs)
    assert form["n_bands"] == 7 and form["pass_tol"] == 0.05 and form["device"] == "auto"
    assert len(form["bands"]) == 7
    assert form["bands"][0]["lo"] < 0 < form["bands"][-1]["hi"]
    greens = [b for b in form["bands"] if b["lo"] >= -0.05 and b["hi"] <= 0.05]
    assert greens
    opts = Metro.form_options(form, band_json_path=str(tmp_path / "bands.json"))
    assert opts["align"] is True and opts["no_map"] is False and opts["section"] is None
    form["do_section"] = True
    form["section_origin"] = "0,0,0"
    form["section_normal"] = "0,0,1"
    form["probe_xyz"] = "1,2,3; 4,5,6"
    form["exclude"] = ["tip"]
    opts = Metro.form_options(form, "bands.json")
    assert opts["probe"] == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    assert opts["section"] == ([0.0, 0.0, 0.0], [0.0, 0.0, 1.0])
    assert opts["exclude"] == ["tip"] and opts["no_section"] is False
    # options produce a valid engine command
    from mslib.engine_client import MetrologyJob, EngineSettings
    job = MetrologyJob("s.stl", "s.step", str(tmp_path), opts)
    cmdl = job.command(EngineSettings(python="python", engine_dir="."))
    assert cmdl[1:5] == ["-u", "-m", "meshsolid.metrology", "s.stl"]
    assert "--align" in cmdl                                  # spec-v2 best fit by default
    assert "--probe" in cmdl and "--section" in cmdl and "--exclude" in cmdl


def test_mslib_default_bands_match_engine():
    from mslib.metrology_ui_defaults import default_bands as a
    from meshsolid.metrology import default_bands as b
    for args in ((-0.2, 0.2, 0.05, 7), (-0.5, 0.5, 0.1, 9), (-1, 1, 0.05, 5)):
        A, B = a(*args), b(*args)
        assert len(A) == len(B)
        for x, y in zip(A, B):
            assert x["lo"] == pytest.approx(y["lo"]) and x["hi"] == pytest.approx(y["hi"])
            assert x["rgb"] == y["rgb"]


# ---- strict Fusion API stub: only members that exist in Autodesk's API reference ----------------------
# tests/fusion_api_members.json is generated from Autodesk's own intellisense stubs (the `adsk` package)
# by scripts/gen_fusion_api_members.py. A strict fake raises AttributeError for any public attribute that
# is not a real member of its Fusion class, exactly like Fusion did for CommandInputs.addColorCommandInput.
API = json.loads((ROOT / "tests" / "fusion_api_members.json").read_text())
ADDIN_SOURCES = [ADDIN / "MeshSolid.py"] + sorted((ADDIN / "mslib").glob("*.py"))


class _Strict:
    _api = None

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

    def _check(self, name):
        api = type(self)._api
        if api and not name.startswith("_") and name not in API[api]:
            raise AttributeError(f"'{api.split('.')[-1]}' object has no attribute '{name}' (not in the Fusion API)")

    def __getattribute__(self, name):
        object.__getattribute__(self, "_check")(name)
        return object.__getattribute__(self, name)

    def __setattr__(self, name, value):
        self._check(name)
        object.__setattr__(self, name, value)


def _strict(api, **kw):
    return type("S_" + api.split(".")[-1], (_Strict,), {"_api": api})(**kw)


class _SEvent(_Strict):
    _api = "core.CommandEvent"

    def __init__(self):
        object.__setattr__(self, "_handlers", [])

    def add(self, h):
        self._handlers.append(h)
        return True


class _SListItem(_Strict):
    _api = "core.ListItem"

    def __init__(self, parent, name, sel, icon):
        object.__setattr__(self, "_parent", parent)
        object.__setattr__(self, "_sel", False)
        self.name, self.icon = name, icon
        if sel:
            self.isSelected = True

    @property
    def isSelected(self):
        return self._sel

    @isSelected.setter
    def isSelected(self, v):
        if v and self._parent._single:
            for it in self._parent._items:
                object.__setattr__(it, "_sel", False)
        object.__setattr__(self, "_sel", bool(v))


class _SListItems(_Strict):
    _api = "core.ListItems"

    def __init__(self, single=True):
        object.__setattr__(self, "_items", [])
        object.__setattr__(self, "_single", single)

    def add(self, name, isSelected, icon="", beforeIndex=-1):
        it = _SListItem(self, name, isSelected, icon)
        self._items.append(it)
        return it

    def item(self, i):
        return self._items[i]

    @property
    def count(self):
        return len(self._items)

    def __iter__(self):                      # test convenience only (dunder, not an API member)
        return iter(self._items)


def _sinput(api, reg, iid, name, **kw):
    o = _strict(api, id=iid, name=name, isVisible=True, isEnabled=True, tooltip="", tooltipDescription="", **kw)
    reg[iid] = o
    return o


class _SSelection(_Strict):
    _api = "core.SelectionCommandInput"

    def __init__(self, reg, iid, name):
        object.__setattr__(self, "_sel", [])
        self.id, self.name, self.isVisible, self.tooltip, self.tooltipDescription = iid, name, True, "", ""
        reg[iid] = self

    def addSelectionFilter(self, f):
        assert f in API["core.SelectionCommandInput"], f"{f!r} is not a Fusion selection filter"
        return True

    def setSelectionLimits(self, lo, hi=0):
        return True

    def clearSelection(self):
        self._sel.clear()
        return True

    @property
    def selectionCount(self):
        return len(self._sel)

    def selection(self, i):
        return _strict("core.Selection", entity=self._sel[i])


class _SDropDown(_Strict):
    _api = "core.DropDownCommandInput"

    def __init__(self, reg, iid, name, style):
        self.id, self.name, self.isVisible, self.tooltip, self.tooltipDescription = iid, name, True, "", ""
        object.__setattr__(self, "listItems", _SListItems(style != 2))
        object.__setattr__(self, "dropDownStyle", style)
        reg[iid] = self

    @property
    def selectedItem(self):
        return next((i for i in self.listItems if i.isSelected), None)


class _SInputs(_Strict):
    _api = "core.CommandInputs"

    def __init__(self, reg):
        object.__setattr__(self, "_reg", reg)

    def itemById(self, i):
        return self._reg.get(i)

    def addGroupCommandInput(self, iid, name):
        return _sinput("core.GroupCommandInput", self._reg, iid, name, children=_SInputs(self._reg), isExpanded=True)

    def addSelectionInput(self, iid, name, commandPrompt):
        return _SSelection(self._reg, iid, name)

    def addBoolValueInput(self, iid, name, isCheckBox, resourceFolder="", initialValue=False):
        assert isinstance(initialValue, bool), (iid, initialValue)
        return _sinput("core.BoolValueCommandInput", self._reg, iid, name, value=initialValue,
                       isCheckBox=isCheckBox, resourceFolder=resourceFolder, text="")

    def addStringValueInput(self, iid, name, initialValue=""):
        return _sinput("core.StringValueCommandInput", self._reg, iid, name, value=initialValue, isReadOnly=False)

    def addDropDownCommandInput(self, iid, name, dropDownStyle):
        return _SDropDown(self._reg, iid, name, dropDownStyle)

    def addIntegerSliderCommandInput(self, iid, name, min, max, hasTwoSliders=False):
        return _sinput("core.IntegerSliderCommandInput", self._reg, iid, name, valueOne=min,
                       minimumValue=min, maximumValue=max)

    def addIntegerSpinnerCommandInput(self, iid, name, min, max, spinStep, initialValue):
        assert isinstance(initialValue, int), (iid, initialValue)
        return _sinput("core.IntegerSpinnerCommandInput", self._reg, iid, name, value=initialValue,
                       minimumValue=min, maximumValue=max, spinStep=spinStep)

    def addFloatSpinnerCommandInput(self, iid, name, unitType, min, max, spinStep, initialValue):
        assert isinstance(unitType, str), iid
        return _sinput("core.FloatSpinnerCommandInput", self._reg, iid, name, value=float(initialValue),
                       unitType=unitType, minimumValue=min, maximumValue=max, spinStep=spinStep)

    def addTextBoxCommandInput(self, iid, name, formattedText, numRows, isReadOnly):
        return _sinput("core.TextBoxCommandInput", self._reg, iid, name, formattedText=formattedText,
                       numRows=numRows, isReadOnly=isReadOnly, isFullWidth=False)

    def addImageCommandInput(self, iid, name, imageFile):
        return _sinput("core.ImageCommandInput", self._reg, iid, name, imageFile=imageFile, isFullWidth=False)


class _SCommand(_Strict):
    _api = "core.Command"

    def __init__(self, reg):
        object.__setattr__(self, "commandInputs", _SInputs(reg))
        object.__setattr__(self, "inputChanged", _SEvent())
        object.__setattr__(self, "execute", _SEvent())
        object.__setattr__(self, "_sizes", [])
        self.okButtonText = "OK"

    def setDialogInitialSize(self, w, h):
        self._sizes.append(("init", w, h)); return True

    def setDialogMinimumSize(self, w, h):
        self._sizes.append(("min", w, h)); return True


def _strict_enums(core, fusion):
    def enum(api, **vals):
        assert set(vals) <= set(API[api]), set(vals) - set(API[api])
        return _strict(api, **vals)
    core.DropDownStyles = enum("core.DropDownStyles", LabeledIconDropDownStyle=0, TextListDropDownStyle=1,
                               CheckBoxDropDownStyle=2)
    core.DialogResults = enum("core.DialogResults", DialogOK=0, DialogCancel=1, DialogError=-1, DialogNo=3, DialogYes=2)
    core.MessageBoxButtonTypes = enum("core.MessageBoxButtonTypes", OKButtonType=0, OKCancelButtonType=1,
                                      RetryCancelButtonType=2, YesNoButtonType=3, YesNoCancelButtonType=4)
    core.MessageBoxIconTypes = enum("core.MessageBoxIconTypes", NoIconIconType=0, InformationIconType=1,
                                    WarningIconType=2, CriticalIconType=3, QuestionIconType=4)
    core.PaletteDockingStates = enum("core.PaletteDockingStates", PaletteDockStateFloating=0, PaletteDockStateTop=1,
                                     PaletteDockStateBottom=2, PaletteDockStateLeft=3, PaletteDockStateRight=4)
    core.HTMLEventArgs = types.SimpleNamespace(cast=lambda x: x)
    fusion.MeshUnits = enum("fusion.MeshUnits", CentimeterMeshUnit=0, MillimeterMeshUnit=1, MeterMeshUnit=2,
                            InchMeshUnit=3, FootMeshUnit=4)
    fusion.DesignTypes = enum("fusion.DesignTypes", DirectDesignType=0, ParametricDesignType=1)
    fusion.TriangleMeshQualityOptions = enum("fusion.TriangleMeshQualityOptions", LowQualityTriangleMesh=8,
                                             NormalQualityTriangleMesh=11, HighQualityTriangleMesh=13,
                                             VeryHighQualityTriangleMesh=15)


def _load_strict(monkeypatch, tmp_path, **kw):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    m, design = _load_addin(monkeypatch, **kw)
    _strict_enums(sys.modules["adsk.core"], sys.modules["adsk.fusion"])
    from mslib import metrology_cmd as Metro
    import importlib
    Metro = importlib.reload(Metro)                 # bind to this test's adsk stub
    m.Metro = Metro
    msgs = []
    ui = _strict("core.UserInterface", messageBox=lambda *a: msgs.append(a))
    m._app().userInterface = ui
    return m, Metro, design, msgs


def test_strict_api_member_list_is_sane():
    assert "addColorCommandInput" not in API["core.CommandInputs"]
    assert "setDialogSize" not in API["core.Command"]
    assert {"addDropDownCommandInput", "addFloatSpinnerCommandInput", "addSelectionInput"} <= set(API["core.CommandInputs"])
    assert {"MeshBodies", "SolidBodies"} <= set(API["core.SelectionCommandInput"])
    assert "isVisible" in API["core.FloatSpinnerCommandInput"] and "isVisible" in API["core.DropDownCommandInput"]
    with pytest.raises(AttributeError, match="addColorCommandInput"):
        _SInputs({}).addColorCommandInput("x", "x", None)
    # the lenient dialog stubs above may only offer real CommandInputs methods too
    for name in dir(_Ins):
        if name.startswith("add"):
            assert name in API["core.CommandInputs"], name


def test_addin_sources_only_use_real_fusion_api_names():
    """Static net for branches no test executes: CommandInputs.add* calls, selection filters, enum values
    and Command sizing calls in every add-in source file must exist in the Fusion API."""
    ins_methods = set(API["core.CommandInputs"])
    for path in ADDIN_SOURCES:
        src = path.read_text(encoding="utf-8")
        for name in re.findall(r"\.(add\w*(?:CommandInput|Input))\(", src):
            assert name in ins_methods, f"{path.name}: CommandInputs.{name} does not exist"
        for f in re.findall(r"addSelectionFilter\(\s*[\"'](\w+)[\"']", src):
            assert f in API["core.SelectionCommandInput"], f"{path.name}: selection filter {f!r}"
        for mod, cls, member in re.findall(r"adsk\.(core|fusion)\.(\w+)\.(\w+)", src):
            key = f"{mod}.{cls}"
            if key in API:
                assert member in API[key], f"{path.name}: adsk.{key}.{member} does not exist"
        for fn in re.findall(r"[\"'](setDialog\w+)[\"']|\.(setDialog\w+)\(", src):
            name = fn[0] or fn[1]
            assert name in API["core.Command"], f"{path.name}: Command.{name} does not exist"
        assert "addColorCommandInput" not in src and "tooltipDescription = (" not in src


def test_main_dialog_builds_against_strict_api(monkeypatch, tmp_path):
    m, _Metro, _d, msgs = _load_strict(monkeypatch, tmp_path)
    reg = {}
    cmd = _SCommand(reg)
    m._build_dialog(cmd, {}, 5400)
    f = m._read_form(cmd.commandInputs)
    assert f["part"] == "auto" and f["device"] == "auto" and reg["time_min"].valueOne == 90
    assert cmd.okButtonText == "Convert to Solid" and not msgs
    # the created handler wires execute + inputChanged and keeps them referenced
    n0 = len(m._handlers)
    m._Created().notify(_strict("core.CommandCreatedEventArgs", command=_SCommand({})))
    assert not msgs, msgs
    assert len(m._handlers) == n0 + 2


def _metro_cmd(Metro, last=None):
    reg = {}
    cmd = _SCommand(reg)
    Metro.build_dialog(cmd, last or {})
    return cmd, reg


def _visible_rows(reg):
    return [i for i in range(11) if reg[f"blo_{i}"].isVisible]


def test_metrology_dialog_strict_bands_defaults_reset_and_count(monkeypatch, tmp_path):
    m, Metro, _d, msgs = _load_strict(monkeypatch, tmp_path)
    cmd, reg = _metro_cmd(Metro)
    assert not msgs and cmd.okButtonText == "Compute metrology"
    assert ("min", 560, 640) in cmd._sizes
    assert _visible_rows(reg) == list(range(7))
    assert all(not reg[f"bcol_{i}"].isVisible and not reg[f"bhi_{i}"].isVisible for i in range(7, 11))
    lo = [reg[f"blo_{i}"].value for i in range(7)]
    hi = [reg[f"bhi_{i}"].value for i in range(7)]
    assert lo[0] == pytest.approx(-0.2) and hi[-1] == pytest.approx(0.2)
    assert all(hi[i] == pytest.approx(lo[i + 1]) for i in range(6))
    cols = [reg[f"bcol_{i}"].selectedItem.name for i in range(7)]
    assert cols[3] == "Green" and cols[0] == "Navy" and cols[-1] == "Magenta" and len(set(cols)) == 7
    assert "Red" not in cols                                   # red is reserved for beyond the outlier cap
    assert len({c for c in cols}) >= 4
    # every colour dropdown has exactly one selection and the whole palette
    for i in range(11):
        items = list(reg[f"bcol_{i}"].listItems)
        assert len(items) == len(Metro.PALETTE) and sum(it.isSelected for it in items) == 1
    form = Metro.read_form(cmd.commandInputs)
    assert form["n_bands"] == 7 and len(form["bands"]) == 7
    assert form["bands"][3]["rgb"] == list(dict(Metro.PALETTE)["Green"])
    assert form["bands"][3]["lo"] == pytest.approx(-0.05) and form["bands"][3]["hi"] == pytest.approx(0.05)
    ch = Metro.Changed()
    # fewer bands -> rows hidden
    reg["n_bands"].value = 5
    ch.notify(_strict("core.InputChangedEventArgs", input=reg["n_bands"], inputs=cmd.commandInputs))
    assert _visible_rows(reg) == list(range(5)) and len(Metro.read_form(cmd.commandInputs)["bands"]) == 5
    # more bands -> rows shown
    reg["n_bands"].value = 9
    ch.notify(_strict("core.InputChangedEventArgs", input=reg["n_bands"], inputs=cmd.commandInputs))
    assert _visible_rows(reg) == list(range(9))
    # edit a band, then Reset restores the defaults
    reg["blo_0"].value = -3.0
    reg["bcol_0"].listItems.item(14).isSelected = True
    ch.notify(_strict("core.InputChangedEventArgs", input=reg["reset_bands"], inputs=cmd.commandInputs))
    assert reg["blo_0"].value == pytest.approx(-0.2) and reg["bcol_0"].selectedItem.name != "White"
    # symmetric scale follows the larger end
    reg["vmax"].value = 0.5
    ch.notify(_strict("core.InputChangedEventArgs", input=reg["vmax"], inputs=cmd.commandInputs))
    assert reg["vmin"].value == pytest.approx(-0.5) and reg["blo_0"].value == pytest.approx(-0.5)
    assert not msgs, msgs
    # remembered bands come back when the band count matches
    last = {k: v for k, v in Metro.read_form(cmd.commandInputs).items() if k not in ("scan", "solid")}
    last["bands"][0]["lo"] = -0.45
    _c2, reg2 = _metro_cmd(Metro, last)
    assert reg2["blo_0"].value == pytest.approx(-0.45) and _visible_rows(reg2) == list(range(9))


def test_metrology_created_attaches_handlers_even_if_dialog_fails(monkeypatch, tmp_path):
    m, Metro, _d, msgs = _load_strict(monkeypatch, tmp_path)
    handlers = []
    cd = types.SimpleNamespace(commandCreated=_SEvent())
    Metro.attach(cd, m._export_mesh_body, m._jobs_dir, m._progress_loop, handlers)
    assert len(handlers) >= 1 and cd.commandCreated._handlers
    monkeypatch.setattr(Metro, "build_dialog", lambda c, last: (_ for _ in ()).throw(AttributeError("boom layout")))
    cmd = _SCommand({})
    cd.commandCreated._handlers[0].notify(_strict("core.CommandCreatedEventArgs", command=cmd))
    assert cmd.execute._handlers and cmd.inputChanged._handlers          # Compute still wired
    assert any(isinstance(h, Metro.Execute) for h in handlers) and any(isinstance(h, Metro.Changed) for h in handlers)
    assert msgs and "boom layout" in msgs[-1][0] and "Traceback" in msgs[-1][0]
    log = (tmp_path / "meshsolid" / "addin_ui.log").read_text()
    assert "boom layout" in log and "opening the Metrology dialog" in log


def test_metrology_execute_never_fails_silently(monkeypatch, tmp_path):
    m, Metro, design, msgs = _load_strict(monkeypatch, tmp_path)
    m._app().fireCustomEvent = lambda *a: False                         # no event: runs in the command
    cmd, reg = _metro_cmd(Metro)
    ex = Metro.Execute(m._export_mesh_body, m._jobs_dir, m._progress_loop)
    ex.notify(_strict("core.CommandEventArgs", command=cmd))
    assert msgs and "Select a scan mesh body and a solid body" in msgs[-1][0]
    # an unexpected exception: traceback in a message box and in the log
    monkeypatch.setattr(Metro, "read_form", lambda ins: 1 / 0)
    ex.notify(_strict("core.CommandEventArgs", command=cmd))
    assert "ZeroDivisionError" in msgs[-1][0] and "Traceback" in msgs[-1][0]
    assert "ZeroDivisionError" in (tmp_path / "meshsolid" / "addin_ui.log").read_text()


# strict design / app fakes for the metrology run
class _SBaseFeature(_Strict):
    _api = "fusion.BaseFeature"

    def __init__(self):
        object.__setattr__(self, "_editing", False)
        object.__setattr__(self, "_finished", False)

    def startEdit(self):
        object.__setattr__(self, "_editing", True); return True

    def finishEdit(self):
        object.__setattr__(self, "_editing", False); object.__setattr__(self, "_finished", True); return True


class _SMeshBodies(_Strict):
    _api = "fusion.MeshBodies"

    def __init__(self, parametric):
        object.__setattr__(self, "_bodies", [])
        object.__setattr__(self, "_parametric", parametric)

    def add(self, fullFilename, units, baseOrFormFeature=None):
        assert os.path.isfile(fullFilename) and units == 1           # MillimeterMeshUnit
        if self._parametric:
            assert baseOrFormFeature is not None and baseOrFormFeature._editing, "parametric needs an editing BaseFeature"
        b = _strict("fusion.MeshBody", name="Body", appearance=None)
        object.__setattr__(b, "_file", fullFilename)
        self._bodies.append(b)
        lst = _strict("fusion.MeshBodyList")
        object.__setattr__(lst, "_l", [b])
        type(lst).count = property(lambda s: len(s._l))
        type(lst).item = lambda s, i: s._l[i]
        return lst

    @property
    def count(self):
        return len(self._bodies)


def _strict_design(parametric=True):
    feats = []

    def add_feat():
        f = _SBaseFeature(); feats.append(f); return f
    appearances = {}

    def add_by_copy(src, name):
        a = _strict("core.Appearance", name=name, appearanceProperties=_color_props())
        appearances[name] = a
        return a
    root = _strict("fusion.Component", name="root", meshBodies=_SMeshBodies(parametric),
                   features=_strict("fusion.Features", baseFeatures=_strict("fusion.BaseFeatures", add=add_feat)))
    design = _strict("fusion.Design", rootComponent=root, designType=1 if parametric else 0,
                     appearances=_strict("core.Appearances", itemByName=lambda n: appearances.get(n),
                                         addByCopy=add_by_copy))
    object.__setattr__(design, "_feats", feats)
    object.__setattr__(design, "_appearances", appearances)
    return design


def _color_props():
    prop = _strict("core.ColorProperty", name="Color", value=None)
    return _strict("core.Properties", itemByName=lambda n: prop if n == "Color" else None, itemById=lambda i: None)


def _strict_libs():
    lib = _strict("core.MaterialLibrary", name="Fusion Appearance Library",
                  appearances=_strict("core.Appearances", itemByName=lambda n: _strict("core.Appearance", name=n)
                                      if n == "Paint - Enamel Glossy (Yellow)" else None))
    return _strict("core.MaterialLibraries", itemById=lambda i: lib if i == "BA5EE55E-9982-449B-9D66-9F036540E140" else None,
                   itemByName=lambda n: None)


class _SProgress(_Strict):
    _api = "core.ProgressDialog"

    def __init__(self):
        object.__setattr__(self, "_msgs", [])
        self.isCancelButtonShown = False

    def show(self, title, message, minimumValue, maximumValue, initialValue=0):
        return True

    def hide(self):
        return True

    @property
    def message(self):
        return self._msgs[-1] if self._msgs else ""

    @message.setter
    def message(self, v):
        self._msgs.append(v)

    @property
    def wasCancelled(self):
        return False


def _metro_job_folder(tmp_path):
    from meshsolid import synthetic as S
    from meshsolid import occ_util as OU
    shape, _ = S.box_with_hole()
    job = tmp_path / "jobs" / "20261111-113301"; job.mkdir(parents=True)
    scan = S.to_scan(shape, edge=1.5, noise=0.01, seed=3)
    scan.export(job / "mesh_body.stl")
    OU.write_step(shape, str(job / "mesh_body.step"))
    return job, shape, scan


def _wire_metro_run(m, Metro, design, monkeypatch):
    app = m._app()
    app.activeProduct = design
    app.materialLibraries = _strict_libs()
    progress = []

    def make_pd():
        pd = _SProgress(); progress.append(pd); return pd
    executed = []
    ui = app.userInterface
    object.__setattr__(ui, "createProgressDialog", make_pd)
    object.__setattr__(ui, "commandDefinitions", _strict("core.CommandDefinitions", itemById=lambda i: _strict(
        "core.CommandDefinition", execute=lambda: executed.append(i) or True)))
    fired = []
    app.fireCustomEvent = lambda eid, info="": fired.append(eid) or True
    monkeypatch.setattr(Metro.EngineSettings, "load", classmethod(lambda cls, *a: _settings(timeout_s=900)))
    return progress, executed, fired


def test_metrology_job_folder_end_to_end(monkeypatch, tmp_path):
    m, Metro, _d, msgs = _load_strict(monkeypatch, tmp_path)
    design = _strict_design(parametric=True)
    progress, executed, fired = _wire_metro_run(m, Metro, design, monkeypatch)
    job, _shape, _scan = _metro_job_folder(tmp_path)
    Metro._pending["event_registered"] = True
    cmd, reg = _metro_cmd(Metro)
    reg["job"].value = str(job)
    reg["n_probes"].value = 3
    reg["zones"].value = "box(-1,-1,19.5, 41,31,21)"          # top face: typed exclusion zone
    reg["operator"].value = "QA"
    Metro.Execute(m._export_mesh_body, m._jobs_dir, m._progress_loop).notify(
        _strict("core.CommandEventArgs", command=cmd))
    assert fired == [Metro.METRO_EVENT_ID] and not msgs                # handed to the custom event
    Metro._RunEvent(m._export_mesh_body, m._jobs_dir, m._progress_loop).notify(
        _strict("core.CustomEventArgs") if "core.CustomEventArgs" in API else types.SimpleNamespace())
    assert not msgs, msgs
    rep = Metro._pending["result"]
    assert executed == [Metro.METRO_RESULT_ID]
    assert rep["stats"]["rms"] < 0.05 and progress and progress[0]._msgs
    bodies = design.rootComponent.meshBodies._bodies
    assert bodies and len(bodies) == len(rep["files"]["bands"])
    assert all(b.name.startswith("Metrology_") and b.appearance is not None for b in bodies)
    assert any(b.name == "Metrology_excluded (not inspected)" for b in bodies)
    assert rep["not_inspected"] and rep["not_inspected"][0]["reason"] == "typed zone"
    assert os.path.isfile(rep["files"]["report_pdf"]) and rep["meta"]["operator"] == "QA"
    assert rep["meta"]["part_name"] == job.name
    assert design._feats and all(f._finished for f in design._feats)    # BaseFeature opened and closed
    a = next(iter(design._appearances.values()))
    assert a.appearanceProperties.itemByName("Color").value is not None
    assert any("Imported" in n for n in rep["notes"])
    # results dialog renders against the strict API
    rcmd = _SCommand({})
    Metro._ResultsCreated().notify(_strict("core.CommandCreatedEventArgs", command=rcmd))
    rreg = rcmd.commandInputs._reg
    assert "rms" in rreg["headline"].formattedText and rreg["summary"].isFullWidth and not msgs
    assert "export_pdf" in rreg and "typed zone" in rreg["summary"].formattedText


def _fusion_mesh_body(mesh_mm):
    v = (np.asarray(mesh_mm.vertices) / 10.0).ravel().tolist()               # Fusion works in cm
    f = np.asarray(mesh_mm.faces).ravel().tolist()
    pm = _strict("fusion.PolygonMesh", nodeCoordinatesAsDouble=v, triangleNodeIndices=f,
                 triangleCount=len(mesh_mm.faces), quadCount=0, polygonCount=0, quadNodeIndices=[],
                 polygonNodeIndices=[], nodeCountPerPolygon=[])
    return _strict("fusion.MeshBody", name="scan", mesh=pm)


def _fusion_brep_body(shape_tm, n_bodies=2, step_src=None):
    v = (np.asarray(shape_tm.vertices) / 10.0).ravel().tolist()
    f = np.asarray(shape_tm.faces).ravel().tolist()
    tri = _strict("fusion.TriangleMesh", nodeCoordinatesAsDouble=v, nodeIndices=f, triangleCount=len(shape_tm.faces))
    calc = _strict("fusion.TriangleMeshCalculator", surfaceTolerance=0.1, calculate=lambda: tri,
                   setQuality=lambda q: True)
    exported = []

    def create_step(filename, geometry=None):
        return _strict("fusion.STEPExportOptions", filename=filename, geometry=geometry)

    def execute(opts):
        import shutil
        shutil.copy(step_src, opts.filename); exported.append(opts.filename); return True
    em = _strict("fusion.ExportManager", createSTEPExportOptions=create_step, execute=execute)
    comp = _strict("fusion.Component", bRepBodies=_strict("fusion.BRepBodies", count=n_bodies),
                   occurrences=_strict("fusion.Occurrences", count=0),
                   parentDesign=_strict("fusion.Design", exportManager=em))
    body = _strict("fusion.BRepBody", name="MeshSolid_box", parentComponent=comp, assemblyContext=None,
                   meshManager=_strict("fusion.MeshManager", createMeshCalculator=lambda: calc))
    return body, exported


def test_metrology_selected_bodies_export_and_run(monkeypatch, tmp_path):
    m, Metro, _d, msgs = _load_strict(monkeypatch, tmp_path)
    sys.modules["adsk.fusion"].BRepBody = types.SimpleNamespace(cast=lambda x: x)
    sys.modules["adsk.fusion"].MeshBody = types.SimpleNamespace(cast=lambda x: x)
    design = _strict_design(parametric=False)
    progress, executed, fired = _wire_metro_run(m, Metro, design, monkeypatch)
    job, shape, scan = _metro_job_folder(tmp_path)
    from meshsolid import occ_util as OU
    shape_tm = OU.shape_to_trimesh(shape, lin=0.01)
    # 1) the solid shares its component with other bodies -> tessellated STL in mm
    body, exported = _fusion_brep_body(shape_tm, n_bodies=2, step_src=job / "mesh_body.step")
    out = tmp_path / "o1"; out.mkdir()
    form = {"scan": _fusion_mesh_body(scan), "solid": body, "job": ""}
    s_path, so_path, notes = Metro.resolve_inputs(form, str(out), m._export_mesh_body)
    assert so_path.endswith("solid_tess.stl") and not exported
    t = trimesh.load(so_path)
    assert np.allclose(t.bounds, shape_tm.bounds, atol=1e-4)                  # mm, not cm
    assert np.allclose(trimesh.load(s_path).bounds, scan.bounds, atol=1e-3)
    # 2) alone in its component -> STEP through ExportManager with the component as geometry
    body2, exported2 = _fusion_brep_body(shape_tm, n_bodies=1, step_src=job / "mesh_body.step")
    out2 = tmp_path / "o2"; out2.mkdir()
    _s, so2, _n = Metro.resolve_inputs({"scan": _fusion_mesh_body(scan), "solid": body2, "job": ""},
                                       str(out2), m._export_mesh_body)
    assert so2.endswith("solid.step") and exported2 == [so2]
    # 3) one selection + job folder: the job fills the missing input
    _s3, so3, _n3 = Metro.resolve_inputs({"scan": _fusion_mesh_body(scan), "solid": None, "job": str(job)},
                                         str(out2), m._export_mesh_body)
    assert so3 == str(job / "mesh_body.step")
    with pytest.raises(ValueError, match="does not contain a scan mesh and a solid"):
        empty = tmp_path / "emptyjob"; empty.mkdir()
        Metro.resolve_inputs({"scan": None, "solid": None, "job": str(empty)}, str(out2), m._export_mesh_body)
    # full run with the tessellated solid (direct design: no BaseFeature)
    rep = Metro.run_metrology(dict(Metro.read_form(_metro_cmd(Metro)[0].commandInputs), scan=form["scan"], solid=body),
                              m._export_mesh_body, m._jobs_dir, m._progress_loop)
    assert not msgs, msgs
    assert rep is not None and rep["stats"]["rms"] < 0.05 and not design._feats
    assert design.rootComponent.meshBodies._bodies and executed == [Metro.METRO_RESULT_ID]


# ---- round 2: exclusions, report header, readable results window, PDF export --------------------------
def test_metrology_dialog_exclusion_and_report_inputs_map_to_engine(monkeypatch, tmp_path):
    m, Metro, _d, msgs = _load_strict(monkeypatch, tmp_path)
    cmd, reg = _metro_cmd(Metro)
    for iid in ("faces", "zones", "auto_excl", "title", "part_name", "operator", "accept_pct"):
        assert iid in reg, iid
    assert reg["auto_excl"].value is True and reg["title"].value == Metro.DEFAULT_TITLE
    f1, f2 = object(), object()
    reg["faces"]._sel.extend([f1, f2])
    reg["zones"].value = "cyl(0,0,0, 0,0,1, 7.2)"
    reg["auto_excl"].value = False
    reg["part_name"].value = "s07"; reg["operator"].value = "Olly Ross"; reg["accept_pct"].value = 90.0
    form = Metro.read_form(cmd.commandInputs)
    assert form["faces"] == [f1, f2] and form["zones"].startswith("cyl(")
    form["exclude_faces"] = [str(tmp_path / "exclude_faces.stl")]
    form["convert_report"] = str(tmp_path / "x.report.json")
    opts = Metro.form_options(form, str(tmp_path / "bands.json"))
    c = Metro.MetrologyJob("scan.stl", "solid.step", str(tmp_path), opts).command(_settings())
    j = " ".join(c)
    for flag in ("--zones", "--exclude-faces", "--convert-report", "--no-auto-exclude", "--accept-pct",
                 "--title", "--part-name", "--operator"):
        assert flag in c, flag
    assert c[c.index("--operator") + 1] == "Olly Ross" and "90.0" in j
    engine_src = (ROOT / "meshsolid" / "metrology.py").read_text(encoding="utf-8")
    for flag in ("--zones", "--exclude-faces", "--convert-report", "--no-auto-exclude", "--accept-pct",
                 "--title", "--part-name", "--operator"):
        assert f'"{flag}"' in engine_src, f"engine CLI has no {flag}"


def _brep_face(tri_mm):
    v = (np.asarray(tri_mm.vertices) / 10.0).ravel().tolist()
    f = np.asarray(tri_mm.faces).ravel().tolist()
    tm = _strict("fusion.TriangleMesh", nodeCoordinatesAsDouble=v, nodeIndices=f, triangleCount=len(tri_mm.faces))
    calc = _strict("fusion.TriangleMeshCalculator", surfaceTolerance=0.1, calculate=lambda: tm,
                   setQuality=lambda q: True)
    return _strict("fusion.BRepFace", meshManager=_strict("fusion.MeshManager", createMeshCalculator=lambda: calc))


def test_metrology_faces_zones_and_convert_report_resolved(monkeypatch, tmp_path):
    m, Metro, _d, msgs = _load_strict(monkeypatch, tmp_path)
    sys.modules["adsk.fusion"].BRepFace = types.SimpleNamespace(cast=lambda x: x)
    sys.modules["adsk.fusion"].BRepBody = types.SimpleNamespace(cast=lambda x: x)
    sys.modules["adsk.fusion"].MeshBody = types.SimpleNamespace(cast=lambda x: x)
    job, shape, scan = _metro_job_folder(tmp_path)
    from meshsolid import occ_util as OU
    shape_tm = OU.shape_to_trimesh(shape, lin=0.01)
    body, _ex = _fusion_brep_body(shape_tm, n_bodies=2)
    conv = tmp_path / "part.report.json"; conv.write_text("{}")
    object.__setattr__(body, "attributes", _strict("core.Attributes", itemByName=lambda g, n: _strict(
        "core.Attribute", value=str(conv)) if (g, n) == (Metro.ATTR_GROUP, Metro.ATTR_REPORT) else None))
    t1 = trimesh.Trimesh([[0, 0, 20], [40, 0, 20], [40, 30, 20]], [[0, 1, 2]])
    t2 = trimesh.Trimesh([[0, 0, 0], [40, 30, 0], [0, 30, 0]], [[0, 1, 2]])
    out = tmp_path / "o"; out.mkdir()
    form = {"scan": _fusion_mesh_body(scan), "solid": body, "job": "", "faces": [_brep_face(t1), _brep_face(t2)],
            "zones": "box(0,0,0, 1,1,1)", "part_name": ""}
    _s, _so, notes = Metro.resolve_inputs(form, str(out), m._export_mesh_body)
    assert [os.path.basename(x) for x in form["exclude_faces"]] == ["exclude_face_01_face.stl",
                                                                    "exclude_face_02_face.stl"]
    f1, f2 = (trimesh.load(x) for x in form["exclude_faces"])                  # one STL per face, in mm
    assert len(f1.faces) == len(f2.faces) == 1
    assert np.allclose(f1.bounds, [[0, 0, 20], [40, 30, 20]], atol=1e-4) and np.allclose(f2.bounds[:, 2], 0)
    assert form["convert_report"] == str(conv) and form["part_name"] == "MeshSolid_box"
    assert any("Disregarded faces" in n for n in notes)
    form["zones"] = "box(1,2,3)"
    with pytest.raises(ValueError, match="Excluded zones"):
        Metro.resolve_inputs(form, str(out), m._export_mesh_body)


def test_band_import_names_grey_excluded_body(monkeypatch, tmp_path):
    m, Metro, _d, msgs = _load_strict(monkeypatch, tmp_path)
    app = m._app(); app.materialLibraries = _strict_libs()
    design = _strict_design(parametric=True)
    p1 = tmp_path / "b0.stl"; p2 = tmp_path / "bx.stl"
    trimesh.creation.box().export(p1); trimesh.creation.box().export(p2)
    n, notes = Metro.import_band_meshes(design, [
        {"path": str(p1), "lo": -0.05, "hi": 0.05, "rgb": [40, 170, 70], "color": "#28aa46"},
        {"path": str(p2), "lo": None, "hi": None, "excluded": True, "rgb": [150, 150, 154], "color": "#96969a"}])
    names = [b.name for b in design.rootComponent.meshBodies._bodies]
    assert n == 2 and names == ["Metrology_-0.050_+0.050", "Metrology_excluded (not inspected)"], (names, notes)


def test_results_window_readable_with_pdf_export_and_open(monkeypatch, tmp_path):
    m, Metro, _d, msgs = _load_strict(monkeypatch, tmp_path)
    core = sys.modules["adsk.core"]
    pdf = tmp_path / "report.pdf"; pdf.write_bytes(b"%PDF-1.4 test")
    png = tmp_path / "report.png"
    from PIL import Image
    Image.new("RGB", (2400, 4300), "white").save(png)
    Metro._pending["result"] = {
        "stats": {"rms": 0.21098, "mean": 0.00626, "p95": 0.13124, "max_plus": 6.84592, "max_minus": -0.69774,
                  "within_pass_pct": 78.556, "pass_tol_mm": 0.05, "accept_pct": 95.0, "result": "FAIL",
                  "n": 64885, "n_excluded": 1337},
        "not_inspected": [{"reason": "hand-entered feature", "detail": "bore Ø14.275 mm", "n": 1337, "n_new": 1337}],
        "files": {"report_pdf": str(pdf), "report_png": str(png)}, "meta": {"part_name": "s07"},
        "out_dir": str(tmp_path), "notes": []}
    rcmd = _SCommand({})
    handlers = []
    sized = []
    Metro._ResultsCreated(lambda img, p: sized.append(p), handlers).notify(
        _strict("core.CommandCreatedEventArgs", command=rcmd))
    reg = rcmd.commandInputs._reg
    assert not msgs, msgs
    head = reg["headline"].formattedText
    assert "font-size:20pt" in head and "FAIL" in head and "rms 0.2110" in head and "78.556" in head
    assert "font-size:13pt" in reg["summary"].formattedText and "hand-entered feature" in reg["summary"].formattedText
    assert ("min", Metro.RESULTS_MIN_W, Metro.RESULTS_MIN_H) in rcmd._sizes
    assert ("init", Metro.RESULTS_W, Metro.RESULTS_H) in rcmd._sizes and Metro.RESULTS_W >= 900
    assert reg["preview"].isFullWidth and sized == [str(png)]
    assert reg["export_pdf"].text == "Export PDF report..." and reg["open_full"].text == "Open full-size report"
    assert rcmd.inputChanged._handlers and rcmd.inputChanged._handlers[0] in handlers
    # Open full-size report -> the PDF in the default viewer
    opened = []
    monkeypatch.setattr(Metro, "open_file", lambda p: opened.append(p))
    ch = rcmd.inputChanged._handlers[0]
    ch.notify(_strict("core.InputChangedEventArgs", input=reg["open_full"]))
    assert opened == [str(pdf)] and reg["open_full"].value is False
    # Export PDF report... -> save dialog, copy, offer to open (Yes)
    dest = tmp_path / "saved" / "s07_report"
    dest.parent.mkdir()
    dlg = _strict("core.FileDialog", title="", filter="", initialFilename="", filename=str(dest),
                  showSave=lambda: core.DialogResults.DialogOK)
    ui = m._app().userInterface
    asked = []
    object.__setattr__(ui, "createFileDialog", lambda: dlg)
    object.__setattr__(ui, "messageBox", lambda *a: asked.append(a) or core.DialogResults.DialogYes)
    ch.notify(_strict("core.InputChangedEventArgs", input=reg["export_pdf"]))
    saved = Path(str(dest) + ".pdf")
    assert saved.read_bytes() == pdf.read_bytes() and dlg.initialFilename == "s07_metrology_report.pdf"
    assert asked and asked[0][2] == core.MessageBoxButtonTypes.YesNoButtonType and opened[-1] == str(saved)
    # Cancel in the save dialog: nothing written, nothing asked
    object.__setattr__(dlg, "showSave", lambda: core.DialogResults.DialogCancel)
    n_asked = len(asked)
    assert Metro.export_pdf_report(Metro._pending["result"], ui) is None and len(asked) == n_asked


def test_imported_solid_remembers_its_convert_report(monkeypatch, tmp_path):
    m, Metro, _d, msgs = _load_strict(monkeypatch, tmp_path)
    added = []
    b = _strict("fusion.BRepBody", name="MeshSolid_s07", attributes=_strict(
        "core.Attributes", add=lambda g, n, v: added.append((g, n, v))))
    other = _strict("fusion.BRepBody", name="Body1", attributes=_strict("core.Attributes", add=lambda *a: 1 / 0))
    monkeypatch.setattr(m, "_bodies_of", lambda d: [b, other])
    m._tag_report(object(), "MeshSolid_s07", "C:/jobs/s07.report.json")
    assert added == [(Metro.ATTR_GROUP, Metro.ATTR_REPORT, "C:/jobs/s07.report.json")]


# ---- zoomable results palette ------------------------------------------------------------------------
class _SPalette(_Strict):
    _api = "core.Palette"

    def __init__(self, pid, name, url):
        ev = _SEvent()
        object.__setattr__(ev, "_api", None)
        object.__setattr__(self, "incomingFromHTML", ev)
        object.__setattr__(self, "_calls", [])
        self.id, self.name, self.htmlFileURL, self.isVisible = pid, name, url, False

    def setMinimumSize(self, w, h):
        self._calls.append(("min", w, h)); return True

    def deleteMe(self):
        self._calls.append(("deleted",)); return True


class _SPalettes(_Strict):
    _api = "core.Palettes"

    def __init__(self):
        object.__setattr__(self, "_items", {})
        object.__setattr__(self, "_adds", [])

    def add(self, id, name, htmlFileURL, isVisible, showCloseButton, isResizable, width, height, useNewWebBrowser):
        self._adds.append((id, name, htmlFileURL, isVisible, showCloseButton, isResizable, width, height,
                           useNewWebBrowser))
        p = _SPalette(id, name, htmlFileURL); self._items[id] = p
        return p

    def itemById(self, id):
        return self._items.get(id)


def _view_result(tmp_path, pdf=True):
    from PIL import Image
    out = tmp_path / "job"; out.mkdir(exist_ok=True)
    png = out / "report.png"; Image.new("RGB", (2400, 4300), "white").save(png)
    files = {"report_png": str(png)}
    if pdf:
        (out / "report.pdf").write_bytes(b"%PDF-1.4 x"); files["report_pdf"] = str(out / "report.pdf")
    return {"stats": {"rms": 0.21098, "mean": 0.00626, "p95": 0.13124, "max_plus": 6.84592, "max_minus": -0.69774,
                      "within_pass_pct": 78.556, "pass_tol_mm": 0.05, "accept_pct": 95.0, "result": "FAIL",
                      "n": 64885, "n_excluded": 1337,
                      "zones": {"hub": {"n": 10, "rms": 0.1, "mean": 0.0, "p95": 0.2, "max_plus": 0.3, "max_minus": -0.1}}},
            "bands": [{"lo": -0.05, "hi": 0.05, "rgb": [40, 170, 70], "in_tol": True, "pct": 78.556}],
            "probes": [{"kind": "worst", "xyz_mm": [1.0, 2.0, 3.0], "deviation_mm": 6.84592}],
            "not_inspected": [{"reason": "hand-entered feature", "detail": "bore Ø14.275 mm", "n": 1337}],
            "notes": ["<script>alert(1)</script> & co"], "meta": {"part_name": "s07", "title": "Metrology"},
            "files": files, "out_dir": str(out)}


def test_results_view_page_has_zoom_toolbar_scrollbars_and_escapes(tmp_path):
    sys.path.insert(0, str(ADDIN))
    from mslib import metrology_view as V
    rep = _view_result(tmp_path)
    page = Path(V.write_page(rep)).read_text(encoding="utf-8")
    for needle in ("id=\"zin\"", "id=\"zout\"", "id=\"fitw\"", "id=\"fitp\"", "id=\"zoomPct\"",
                   "overflow:auto", "e.ctrlKey", "passive: false", "mousedown", "adsk.fusionSendData",
                   "Export PDF report...", "Open full-size report", "src='report.png?t="):
        assert needle in page, needle
    assert "<script>alert(1)" not in page and "&lt;script&gt;alert(1)" in page
    assert "FAIL" in page and "rms 0.2110 mm" in page and "hand-entered feature" in page and "1,337" in page
    assert "/*__" not in page                                          # all placeholders filled
    assert "if (!true)" in page
    nopdf = V.page_html(_view_result(tmp_path / "x" if (tmp_path / "x").mkdir() is None else tmp_path, pdf=False))
    assert "if (!false)" in nopdf
    assert V.palette_url("C:\\jobs\\a\\results_view.html").count("\\") == 0
    assert V.parse_action("exportPdf", "") == "exportPdf"
    assert V.parse_action("send", '{"action": "openFull"}') == "openFull"
    assert V.parse_action("navigate", "x") == ""


def test_results_palette_strict_api_buttons_and_reuse(monkeypatch, tmp_path):
    m, Metro, _d, msgs = _load_strict(monkeypatch, tmp_path)
    ui = m._app().userInterface
    pals = _SPalettes()
    object.__setattr__(ui, "palettes", pals)
    rep = _view_result(tmp_path)
    pal = Metro.show_results_palette(rep)
    assert pal is not None and not msgs
    pid, name, url, vis, close, resizable, w, h, new_browser = pals._adds[0]
    assert pid == Metro.PALETTE_ID and url.endswith("/results_view.html") and "\\" not in url
    assert (close, resizable, new_browser) == (True, True, True) and w >= 1000 and h >= 800
    assert pal.isVisible is True and pal.dockingState == 0 and ("min", 640, 480) in pal._calls
    assert Path(url).is_file() and len(pal.incomingFromHTML._handlers) == 1
    # buttons in the page -> Python
    opened = []
    monkeypatch.setattr(Metro, "open_file", lambda p: opened.append(p))
    hnd = pal.incomingFromHTML._handlers[0]

    def fire(action):
        a = _strict("core.HTMLEventArgs", action=action, data='{"action": "%s"}' % action, returnData="")
        hnd.notify(a)
        return a.returnData
    assert fire("openFull").startswith("Opened") and opened[-1] == rep["files"]["report_pdf"]
    fire("openFolder"); assert opened[-1] == rep["out_dir"]
    dlg = _strict("core.FileDialog", title="", filter="", initialFilename="", filename="",
                  showSave=lambda: sys.modules["adsk.core"].DialogResults.DialogCancel)
    object.__setattr__(ui, "createFileDialog", lambda: dlg)
    assert fire("exportPdf") == "Export cancelled"
    assert fire("ready") == "ok" and fire("bogus") == "" and not msgs
    # a second result reuses the palette and points it at the new page
    rep2 = _view_result(tmp_path / "second" if (tmp_path / "second").mkdir() is None else tmp_path)
    pal2 = Metro.show_results_palette(rep2)
    assert pal2 is pal and len(pals._adds) == 1 and pal.htmlFileURL.startswith(rep2["out_dir"].replace("\\", "/"))
    Metro.detach()
    assert ("deleted",) in pal._calls


def test_results_palette_failure_falls_back_to_dialog(monkeypatch, tmp_path):
    m, Metro, _d, msgs = _load_strict(monkeypatch, tmp_path)
    ui = m._app().userInterface
    object.__setattr__(ui, "palettes", _strict("core.Palettes", itemById=lambda i: None, add=lambda *a: None))
    assert Metro.show_results_palette(_view_result(tmp_path)) is None and not msgs


def test_metrology_run_opens_zoomable_palette(monkeypatch, tmp_path):
    m, Metro, _d, msgs = _load_strict(monkeypatch, tmp_path)
    design = _strict_design(parametric=True)
    progress, executed, fired = _wire_metro_run(m, Metro, design, monkeypatch)
    pals = _SPalettes()
    object.__setattr__(m._app().userInterface, "palettes", pals)
    job, _shape, _scan = _metro_job_folder(tmp_path)
    cmd, reg = _metro_cmd(Metro)
    reg["job"].value = str(job)
    reg["n_probes"].value = 3
    rep = Metro.run_metrology(Metro.read_form(cmd.commandInputs), m._export_mesh_body, m._jobs_dir, m._progress_loop)
    assert rep is not None and not msgs, msgs
    assert executed == [] and len(pals._adds) == 1               # palette instead of the old dialog
    page = Path(pals._adds[0][2]).read_text(encoding="utf-8")
    assert "Fit width" in page and "report.png" in page and os.path.dirname(pals._adds[0][2]) == \
        rep["out_dir"].replace("\\", "/")


def test_results_view_zoom_and_horizontal_scroll_in_headless_chrome(tmp_path):
    """Runs the page's own JavaScript in Chromium (the engine behind Fusion's palettes)."""
    import shutil
    import subprocess
    chrome = shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chromium-browser")
    if not chrome:
        pytest.skip("no Chromium on this machine")
    sys.path.insert(0, str(ADDIN))
    from mslib import metrology_view as V
    rep = _view_result(tmp_path)
    page = Path(V.write_page(rep)).read_text(encoding="utf-8")
    probe = """<script>setTimeout(function(){
      var v=document.getElementById('view'), r={};
      r.fit=msZoom.get(); r.fitPct=document.getElementById('zoomPct').textContent; r.fitFont=parseFloat(document.getElementById('text').style.fontSize);
      document.getElementById('zin').click(); document.getElementById('zin').click();
      r.inZ=msZoom.get(); r.inPct=document.getElementById('zoomPct').textContent; r.sw=v.scrollWidth; r.cw=v.clientWidth;
      r.inFont=parseFloat(document.getElementById('text').style.fontSize);
      v.scrollLeft=200; r.left=v.scrollLeft;
      var ev=new WheelEvent('wheel',{deltaY:-100,ctrlKey:true,clientX:400,clientY:300,bubbles:true,cancelable:true});
      r.wheelPrevented=!v.dispatchEvent(ev); r.wheelZ=msZoom.get();
      document.getElementById('zout').click(); r.outZ=msZoom.get();
      document.getElementById('fitw').click(); r.backZ=msZoom.get(); r.backSw=v.scrollWidth;
      var d=document.createElement('pre'); d.id='RESULT'; d.textContent=JSON.stringify(r); document.body.appendChild(d);
    },700);</script></body>"""
    tp = Path(rep["out_dir"]) / "probe.html"
    tp.write_text(page.replace("</body>", probe), encoding="utf-8")
    try:
        out = subprocess.run([chrome, "--headless=new", "--no-sandbox", "--disable-gpu", "--window-size=1150,900",
                              "--virtual-time-budget=4000", "--dump-dom", tp.as_uri()],
                             capture_output=True, text=True, timeout=90).stdout
    except (OSError, subprocess.TimeoutExpired) as e:
        pytest.skip(f"Chromium did not run: {e}")
    mt = re.search(r'<pre id="RESULT">(.*?)</pre>', out, re.S)
    assert mt, out[-1500:]
    import html as H
    r = json.loads(H.unescape(mt.group(1)))
    assert r["fit"] == pytest.approx((1135 - 30) / 2400, rel=0.05)           # fit width of a 2400 px report
    assert r["inZ"] == pytest.approx(r["fit"] * 1.25 ** 2, rel=1e-3) and r["inPct"] != r["fitPct"]
    assert r["sw"] > r["cw"] and r["left"] == 200                           # horizontal scrolling when zoomed in
    assert r["inFont"] > r["fitFont"]                                       # text scales with the zoom
    assert r["wheelPrevented"] and r["wheelZ"] > r["inZ"]                  # Ctrl + wheel zooms
    assert r["outZ"] < r["wheelZ"] and r["backZ"] == pytest.approx(r["fit"], rel=1e-3) and r["backSw"] <= r["cw"] + 1


# ---- outlier cap in the dialog, the engine command, the imported bodies and the results ---------------
def test_outlier_cap_option_colour_and_beyond_cap_body(monkeypatch, tmp_path):
    m, Metro, _d, msgs = _load_strict(monkeypatch, tmp_path)
    cmd, reg = _metro_cmd(Metro)
    assert reg["max_dist"].name.startswith("Outlier cap") and "NOT counted" in reg["max_dist"].tooltipDescription
    assert reg["cap_color"].selectedItem.name == "Red"
    reg["max_dist"].value = 0.1
    next(it for it in reg["cap_color"].listItems if it.name == "Magenta").isSelected = True
    form = Metro.read_form(cmd.commandInputs)
    assert form["cap_color"] == list(dict(Metro.PALETTE)["Magenta"]) and form["cap_color_name"] == "Magenta"
    opts = Metro.form_options(form, str(tmp_path / "b.json"))
    c = Metro.MetrologyJob("s.stl", "p.step", str(tmp_path), opts).command(_settings())
    assert c[c.index("--max-dist") + 1] == "0.1" and c[c.index("--cap-color") + 1] == "#d728be"
    # imported bodies: a separate 'beyond cap' body in its own colour, grey stays 'not inspected'
    app = m._app(); app.materialLibraries = _strict_libs()
    design = _strict_design(parametric=True)
    paths = []
    for k in range(3):
        pth = tmp_path / f"b{k}.stl"; trimesh.creation.box().export(pth); paths.append(str(pth))
    n, notes = Metro.import_band_meshes(design, [
        {"path": paths[0], "lo": -0.05, "hi": 0.05, "rgb": [36, 170, 65], "color": "#24aa41"},
        {"path": paths[1], "lo": None, "hi": None, "beyond_cap": True, "cap_mm": 0.1, "rgb": [220, 30, 30],
         "color": "#dc1e1e"},
        {"path": paths[2], "lo": None, "hi": None, "excluded": True, "rgb": [150, 150, 154], "color": "#96969a"}])
    names = [b.name for b in design.rootComponent.meshBodies._bodies]
    assert names == ["Metrology_-0.050_+0.050", "Metrology_beyond_cap (|d| > 0.100 mm)",
                     "Metrology_excluded (not inspected)"], (names, notes)
    # results: true max and the beyond-cap count are shown
    rep = {"stats": {"rms": 0.0351, "mean": -0.0012, "p95": 0.0718, "max_plus": 0.1, "max_minus": -0.1,
                     "within_pass_pct": 84.056, "pass_tol_mm": 0.05, "accept_pct": 90.0, "result": "FAIL",
                     "n": 49296, "n_excluded": 15589, "cap_mm": 0.1, "n_beyond_cap": 4577, "pct_beyond_cap": 8.497,
                     "true_max_plus": 0.912, "true_max_minus": -0.431},
           "settings": {"cap_color": "#dc1e1e"}, "bands": [], "files": {}, "notes": []}
    head = Metro.summary_html(rep)
    assert "true max +0.912 / -0.431" in head and "4,577" in head and "outlier cap" in head
    sys.path.insert(0, str(ADDIN))
    from mslib import metrology_view as V
    page = V.summary_html(rep)
    assert "true max +0.912 / -0.431" in page and "4,577" in page and "beyond cap" in page


def test_face_exports_are_named_by_surface_type(monkeypatch, tmp_path):
    m, Metro, _d, msgs = _load_strict(monkeypatch, tmp_path)
    core = sys.modules["adsk.core"]
    core.SurfaceTypes = _strict("core.SurfaceTypes", PlaneSurfaceType=0, CylinderSurfaceType=1, ConeSurfaceType=2,
                                SphereSurfaceType=3, TorusSurfaceType=4, EllipticalCylinderSurfaceType=5,
                                EllipticalConeSurfaceType=6, NurbsSurfaceType=7)
    sys.modules["adsk.fusion"].BRepFace = types.SimpleNamespace(cast=lambda x: x)
    t = trimesh.Trimesh([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]])
    cone, plane = _brep_face(t), _brep_face(t)
    object.__setattr__(cone, "geometry", _strict("core.Surface", surfaceType=2))
    object.__setattr__(plane, "geometry", _strict("core.Surface", surfaceType=0))
    out = Metro.export_faces([cone, plane], str(tmp_path))
    assert [os.path.basename(x) for x in out] == ["exclude_face_01_cone.stl", "exclude_face_02_plane.stl"]
    from meshsolid.metrology_exclude import face_label
    assert face_label(out[0]) == "face 1 (cone)" and face_label(out[1], 1) == "face 2 (plane)"
    cmd, reg = _metro_cmd(Metro)
    assert "Outer rim" in reg["ex_tip"].name and "nose cone" in reg["ex_tip"].tooltipDescription
